// SPDX-License-Identifier: MIT
/*
 * WPE WebKit web view for flutter-pi (ByteHomeDash's in-app browser).
 *
 * A WebKitWebView renders off-screen through wpebackend-fdo, initialised on
 * flutter-pi's own EGLDisplay, so each frame arrives as an EGLImage the GPU
 * already holds: it's bound to a GL texture in a context shared with
 * Flutter's and pushed to the texture registry — no copy through the CPU.
 *
 * WebKit runs on its own thread with its own GMainContext; the platform
 * channel handlers (flutter-pi's platform thread) hand work over with
 * g_main_context_invoke, and state goes back through platform tasks.
 *
 * Channel "bytehomedash/webview" (JSON method calls):
 *   create {url, width, height, scale} -> {id}   (id = the Flutter texture id)
 *   load {id, url} · back {id} · forward {id} · reload {id} · stop {id}
 *   touch {id, type: 0 down 1 move 2 up 3 cancel, x, y}   (logical px)
 *   click {id, x, y}            a left click (move, press, release)
 *   scroll {id, x, y, dx, dy}   smooth scroll; dx/dy is how far the finger moved
 *   Positions and distances for touch/click/scroll are in device pixels (the
 *   texture's): WebKit divides them by the scale factor itself.
 *   type {id, back, text, enter}   key presses into the focused field: `back`
 *                                  backspaces, then `text`, then Return if `enter`
 *   eval {id, js}
 *   resize {id, width, height}   (logical px; the page reflows)
 *   dispose {id}
 * Calls into Dart: state {id, url, title, progress, loading, canGoBack, canGoForward}
 *                  field {id, data}   data: JSON from the page's script when a
 *                  text field is tapped or loses focus (see FIELD_SCRIPT)
 *
 * Written against WPE WebKit 2.48 (libwpe + wpebackend-fdo); the frame source
 * is the only part that changes for the WPEPlatform API of 2.54+.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <wpe/fdo-egl.h>
#include <wpe/fdo.h>
#include <wpe/webkit.h>
#include <wpe/wpe.h>

#include "flutter-pi.h"
#include "gl_renderer.h"
#include "platformchannel.h"
#include "pluginregistry.h"
#include "texture_registry.h"
#include "util/logging.h"

#define CHANNEL "bytehomedash/webview"
#define MAX_VIEWS 4

struct view {
    int64_t id;
    struct texture *texture;
    struct wpe_view_backend_exportable_fdo *exportable;
    struct wpe_view_backend *backend;
    WebKitWebView *web;
    int width, height;
    double scale;
    bool closing;
};

struct frame_ref {
    struct view *view;
    struct wpe_fdo_egl_exported_image *image;
    GLuint texture;
};

static struct {
    struct flutterpi *flutterpi;
    EGLDisplay display;
    EGLContext context;
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target;
    GMainContext *main;
    GMainLoop *loop;
    pthread_t thread;
    bool started;
    struct view *views[MAX_VIEWS];
    WebKitNetworkSession *session;
} wv;

static uint32_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t) (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

static struct view *find_view(int64_t id) {
    for (int i = 0; i < MAX_VIEWS; i++) {
        if (wv.views[i] && wv.views[i]->id == id) return wv.views[i];
    }
    return NULL;
}

// ─── State back to Dart ─────────────────────────────────────────────────

struct state_msg {
    int64_t id;
    char *url, *title;
    double progress;
    bool loading, back, forward;
};

static int send_state_on_platform(void *userdata) {
    struct state_msg *m = userdata;
    platch_call_json(
        CHANNEL,
        "state",
        &JSONOBJECT7(
            "id", JSONNUM((double) m->id),
            "url", JSONSTRING(m->url ? m->url : ""),
            "title", JSONSTRING(m->title ? m->title : ""),
            "progress", JSONNUM(m->progress),
            "loading", JSONBOOL(m->loading),
            "canGoBack", JSONBOOL(m->back),
            "canGoForward", JSONBOOL(m->forward)
        ),
        NULL,
        NULL
    );
    free(m->url);
    free(m->title);
    free(m);
    return 0;
}

static void send_state(struct view *v) {
    struct state_msg *m = calloc(1, sizeof *m);
    m->id = v->id;
    const char *uri = webkit_web_view_get_uri(v->web);
    const char *title = webkit_web_view_get_title(v->web);
    m->url = uri ? strdup(uri) : NULL;
    m->title = title ? strdup(title) : NULL;
    m->progress = webkit_web_view_get_estimated_load_progress(v->web);
    m->loading = webkit_web_view_is_loading(v->web);
    m->back = webkit_web_view_can_go_back(v->web);
    m->forward = webkit_web_view_can_go_forward(v->web);
    flutterpi_post_platform_task(send_state_on_platform, m);
}

static void on_notify(GObject *obj, GParamSpec *spec, gpointer data) {
    (void) obj;
    (void) spec;
    send_state(data);
}

// ─── Text fields ────────────────────────────────────────────────────────

/// Runs in every frame of every page: after each click, says whether the
/// focused element takes text (and what's before its caret, so the app's
/// keyboard can delete what's there), and says so again when it loses
/// focus. Only clicks count, so a page that focuses its own search box on
/// load doesn't pop the keyboard. A frame whose focus moved into an iframe
/// stays quiet: the iframe's own copy reports. `search` is a best guess from
/// the field's role, name and labels (Google's and DuckDuckGo's boxes are
/// one-row textareas with nothing else to go by).
static const char FIELD_SCRIPT[] =
    "(() => {"
    "  const post = (m) => { try { window.webkit.messageHandlers.bhdField.postMessage(JSON.stringify(m)); } catch (e) {} };"
    "  const deep = () => { let el = document.activeElement;"
    "    while (el && el.shadowRoot && el.shadowRoot.activeElement) el = el.shadowRoot.activeElement; return el; };"
    "  const takesText = (el) => {"
    "    if (!el || el.disabled || el.readOnly) return false;"
    "    if (el.isContentEditable || el.tagName === 'TEXTAREA') return true;"
    "    if (el.tagName !== 'INPUT') return false;"
    "    return !['button','checkbox','color','file','hidden','image','radio','range','reset','submit']"
    "      .includes((el.type || 'text').toLowerCase()); };"
    "  const report = () => { const el = deep();"
    "    if (el && el.tagName === 'IFRAME') return;"
    "    if (!takesText(el)) { post({editable: false}); return; }"
    "    let before = '';"
    "    if (typeof el.value === 'string') {"
    "      try { before = el.value.slice(0, el.selectionStart ?? el.value.length); } catch (e) { before = el.value; } }"
    "    const label = [el.name, el.id, el.placeholder, el.getAttribute('aria-label'), el.title].join(' ');"
    "    const search = el.type === 'search' || el.getAttribute('role') === 'combobox' || !!el.closest('[role=search]')"
    "      || /search|pesquis|procur/i.test(label) || el.name === 'q';"
    "    post({editable: true, before, search,"
    "      multiline: (el.tagName === 'TEXTAREA' && !(el.rows <= 1)) || !!el.isContentEditable,"
    "      type: (el.type || 'text').toLowerCase(), hint: (el.enterKeyHint || '').toLowerCase()}); };"
    "  addEventListener('mouseup', () => setTimeout(report, 0), true);"
    "  addEventListener('focusout', () => setTimeout(() => { const el = deep();"
    "    if (!(el && el.tagName === 'IFRAME') && !takesText(el)) post({editable: false}); }, 0), true);"
    "})();";

struct field_msg {
    int64_t id;
    char *data;
};

static int send_field_on_platform(void *userdata) {
    struct field_msg *m = userdata;
    platch_call_json(CHANNEL, "field", &JSONOBJECT2("id", JSONNUM((double) m->id), "data", JSONSTRING(m->data)), NULL, NULL);
    free(m->data);
    free(m);
    return 0;
}

static void on_field_message(WebKitUserContentManager *manager, JSCValue *value, gpointer data) {
    (void) manager;
    struct view *v = data;
    if (!jsc_value_is_string(value)) return;
    struct field_msg *m = calloc(1, sizeof *m);
    m->id = v->id;
    m->data = jsc_value_to_string(value);
    // jsc_value_to_string's buffer is g_malloc'd; keep one allocator.
    char *copy = strdup(m->data);
    g_free(m->data);
    m->data = copy;
    flutterpi_post_platform_task(send_field_on_platform, m);
}

// ─── Frames ─────────────────────────────────────────────────────────────

static gboolean release_frame_on_wpe(gpointer data) {
    struct frame_ref *f = data;
    eglMakeCurrent(wv.display, EGL_NO_SURFACE, EGL_NO_SURFACE, wv.context);
    glDeleteTextures(1, &f->texture);
    eglMakeCurrent(wv.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (!f->view->closing) wpe_view_backend_exportable_fdo_egl_dispatch_release_exported_image(f->view->exportable, f->image);
    free(f);
    return G_SOURCE_REMOVE;
}

static void on_frame_destroy(const struct texture_frame *frame, void *userdata) {
    (void) frame;
    // Called on a Flutter thread once the frame is no longer shown.
    g_main_context_invoke(wv.main, release_frame_on_wpe, userdata);
}

static void on_export_image(void *data, struct wpe_fdo_egl_exported_image *image) {
    struct view *v = data;
    // Closed, but WebKit hasn't released the view yet: hand the frame
    // straight back; the texture is being (or has been) freed.
    if (v->closing) {
        wpe_view_backend_exportable_fdo_egl_dispatch_release_exported_image(v->exportable, image);
        return;
    }
    EGLImageKHR egl_image = wpe_fdo_egl_exported_image_get_egl_image(image);

    eglMakeCurrent(wv.display, EGL_NO_SURFACE, EGL_NO_SURFACE, wv.context);
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    wv.image_target(GL_TEXTURE_2D, egl_image);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
    eglMakeCurrent(wv.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

    struct frame_ref *f = calloc(1, sizeof *f);
    f->view = v;
    f->image = image;
    f->texture = tex;
    texture_push_frame(
        v->texture,
        &(struct texture_frame){
            .gl = {
                .target = GL_TEXTURE_2D,
                .name = tex,
                .format = GL_RGBA8_OES,
                .width = wpe_fdo_egl_exported_image_get_width(image),
                .height = wpe_fdo_egl_exported_image_get_height(image),
            },
            .destroy = on_frame_destroy,
            .userdata = f,
        }
    );
    // Ready for the next one; WebKit only draws when something changed.
    wpe_view_backend_exportable_fdo_dispatch_frame_complete(v->exportable);
}

static const struct wpe_view_backend_exportable_fdo_egl_client exportable_client = {
    .export_fdo_egl_image = on_export_image,
};

// ─── Work on the WebKit thread ──────────────────────────────────────────

/// One session for every page, in fixed folders, with cookies saved to
/// disk: sign-ins survive the app restarting. WebKit's default session
/// kept cookies only in memory, and named its folders after whatever
/// GLib's program name happened to be when it started (`wpe`, or
/// `flutter-pi` once GStreamer had set it), so site data was split too.
static WebKitNetworkSession *session(void) {
    if (wv.session) return wv.session;
    char *data = g_build_filename(g_get_user_data_dir(), "bytehomedash", "browser", NULL);
    char *cache = g_build_filename(g_get_user_cache_dir(), "bytehomedash", "browser", NULL);
    wv.session = webkit_network_session_new(data, cache);
    char *cookies = g_build_filename(data, "cookies.sqlite", NULL);
    WebKitCookieManager *jar = webkit_network_session_get_cookie_manager(wv.session);
    webkit_cookie_manager_set_persistent_storage(jar, cookies, WEBKIT_COOKIE_PERSISTENT_STORAGE_SQLITE);
    webkit_network_session_set_persistent_credential_storage_enabled(wv.session, TRUE);
    g_free(cookies);
    g_free(cache);
    g_free(data);
    return wv.session;
}

struct create_args {
    struct view *view;
    char *url;
};

static gboolean create_on_wpe(gpointer data) {
    struct create_args *a = data;
    struct view *v = a->view;
    v->exportable = wpe_view_backend_exportable_fdo_egl_create(&exportable_client, v, v->width, v->height);
    v->backend = wpe_view_backend_exportable_fdo_get_view_backend(v->exportable);
    // WebKit owns the view backend: it calls this when it releases the view,
    // and nothing else may destroy the exportable. (Passing NULL made WebKit
    // destroy the view backend itself, so destroying the exportable as well
    // freed it twice: a segfault a few seconds after closing a page.)
    WebKitWebViewBackend *backend =
        webkit_web_view_backend_new(v->backend, (GDestroyNotify) wpe_view_backend_exportable_fdo_destroy, v->exportable);
    WebKitUserContentManager *content = webkit_user_content_manager_new();
    g_signal_connect(content, "script-message-received::bhdField", G_CALLBACK(on_field_message), v);
    webkit_user_content_manager_register_script_message_handler(content, "bhdField", NULL);
    WebKitUserScript *script =
        webkit_user_script_new(FIELD_SCRIPT, WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES, WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START, NULL, NULL);
    webkit_user_content_manager_add_script(content, script);
    webkit_user_script_unref(script);
    v->web = g_object_new(WEBKIT_TYPE_WEB_VIEW, "backend", backend, "user-content-manager", content, "network-session", session(), NULL);
    g_object_unref(content);
    if (v->scale > 0) wpe_view_backend_dispatch_set_device_scale_factor(v->backend, (float) v->scale);
    wpe_view_backend_add_activity_state(v->backend, wpe_view_activity_state_visible | wpe_view_activity_state_focused | wpe_view_activity_state_in_window);

    g_signal_connect(v->web, "notify::uri", G_CALLBACK(on_notify), v);
    g_signal_connect(v->web, "notify::title", G_CALLBACK(on_notify), v);
    g_signal_connect(v->web, "notify::estimated-load-progress", G_CALLBACK(on_notify), v);
    g_signal_connect(v->web, "notify::is-loading", G_CALLBACK(on_notify), v);

    webkit_web_view_load_uri(v->web, a->url);
    free(a->url);
    free(a);
    return G_SOURCE_REMOVE;
}

struct touch_args {
    struct view *view;
    int type;
    double x, y;
};

static gboolean touch_on_wpe(gpointer data) {
    struct touch_args *a = data;
    struct view *v = a->view;
    if (v->web) {
        enum wpe_input_touch_event_type type = a->type == 0 ? wpe_input_touch_event_type_down
                                             : a->type == 1 ? wpe_input_touch_event_type_motion
                                                            : wpe_input_touch_event_type_up;
        struct wpe_input_touch_event_raw point = {
            .type = type,
            .time = now_ms(),
            .id = 0,
            .x = (int32_t) a->x,
            .y = (int32_t) a->y,
        };
        struct wpe_input_touch_event event = {
            .touchpoints = &point,
            .touchpoints_length = 1,
            .type = type,
            .id = 0,
            .time = point.time,
            .modifiers = 0,
        };
        wpe_view_backend_dispatch_touch_event(v->backend, &event);
    }
    free(a);
    return G_SOURCE_REMOVE;
}

struct pointer_args {
    struct view *view;
    bool click;
    double x, y, dx, dy;
};

static gboolean pointer_on_wpe(gpointer data) {
    struct pointer_args *a = data;
    struct view *v = a->view;
    if (v->web) {
        uint32_t t = now_ms();
        int x = (int) a->x, y = (int) a->y;
        if (a->click) {
            // Pages that ignore touch (consent dialogs, buttons with their own
            // scripts) all take a mouse click.
            struct wpe_input_pointer_event move = { .type = wpe_input_pointer_event_type_motion, .time = t, .x = x, .y = y };
            wpe_view_backend_dispatch_pointer_event(v->backend, &move);
            struct wpe_input_pointer_event press = {
                .type = wpe_input_pointer_event_type_button, .time = t, .x = x, .y = y, .button = 1, .state = 1,
                .modifiers = wpe_input_pointer_modifier_button1,
            };
            wpe_view_backend_dispatch_pointer_event(v->backend, &press);
            struct wpe_input_pointer_event release = { .type = wpe_input_pointer_event_type_button, .time = t + 1, .x = x, .y = y, .button = 1, .state = 0 };
            wpe_view_backend_dispatch_pointer_event(v->backend, &release);
        } else {
            struct wpe_input_axis_2d_event scroll = {
                .base = {
                    .type = wpe_input_axis_event_type_mask_2d | wpe_input_axis_event_type_motion_smooth,
                    .time = t,
                    .x = x,
                    .y = y,
                },
                .x_axis = a->dx,
                .y_axis = a->dy,
            };
            wpe_view_backend_dispatch_axis_event(v->backend, &scroll.base);
        }
    }
    free(a);
    return G_SOURCE_REMOVE;
}

struct type_args {
    struct view *view;
    int back;
    char *text;
    bool enter;
};

static void press_key(struct view *v, uint32_t keysym) {
    uint32_t t = now_ms();
    struct wpe_input_keyboard_event down = { .time = t, .key_code = keysym, .pressed = true };
    wpe_view_backend_dispatch_keyboard_event(v->backend, &down);
    struct wpe_input_keyboard_event up = { .time = t + 1, .key_code = keysym, .pressed = false };
    wpe_view_backend_dispatch_keyboard_event(v->backend, &up);
}

/// The X keysym for a character: Latin-1 maps to itself, anything else
/// to the Unicode range xkb (and so WebKit) turns back into text.
static uint32_t keysym_for(uint32_t cp) {
    if (cp == '\n') return 0xff0d;  // Return
    if ((cp >= 0x20 && cp < 0x7f) || (cp >= 0xa0 && cp <= 0xff)) return cp;
    return 0x01000000 | cp;
}

static gboolean type_on_wpe(gpointer data) {
    struct type_args *a = data;
    struct view *v = a->view;
    if (v->web) {
        for (int i = 0; i < a->back; i++) press_key(v, 0xff08);  // BackSpace
        const unsigned char *s = (const unsigned char *) (a->text ? a->text : "");
        while (*s) {
            uint32_t cp;
            int n = *s < 0x80 ? 1 : (*s >> 5) == 0x6 ? 2 : (*s >> 4) == 0xe ? 3 : 4;
            cp = n == 1 ? *s : n == 2 ? *s & 0x1f : n == 3 ? *s & 0x0f : *s & 0x07;
            for (int i = 1; i < n && s[i]; i++) cp = (cp << 6) | (s[i] & 0x3f);
            for (int i = 0; i < n && *s; i++) s++;
            press_key(v, keysym_for(cp));
        }
        if (a->enter) press_key(v, 0xff0d);
    }
    free(a->text);
    free(a);
    return G_SOURCE_REMOVE;
}

static void on_eval_done(GObject *obj, GAsyncResult *res, gpointer data) {
    (void) data;
    GError *error = NULL;
    JSCValue *value = webkit_web_view_evaluate_javascript_finish(WEBKIT_WEB_VIEW(obj), res, &error);
    if (error) {
        fprintf(stderr, "[webview] eval failed: %s\n", error->message);
        g_error_free(error);
    } else {
        g_object_unref(value);
    }
}

struct simple_args {
    struct view *view;
    int op;
    char *text;
    int width, height;
};

enum { OP_LOAD, OP_BACK, OP_FORWARD, OP_RELOAD, OP_STOP, OP_EVAL, OP_RESIZE, OP_DISPOSE };

static int destroy_texture_on_platform(void *userdata) {
    texture_destroy(userdata);
    return 0;
}

static gboolean simple_on_wpe(gpointer data) {
    struct simple_args *a = data;
    struct view *v = a->view;
    if (v->web) {
        switch (a->op) {
            case OP_LOAD: webkit_web_view_load_uri(v->web, a->text); break;
            case OP_BACK: webkit_web_view_go_back(v->web); break;
            case OP_FORWARD: webkit_web_view_go_forward(v->web); break;
            case OP_RELOAD: webkit_web_view_reload(v->web); break;
            case OP_STOP: webkit_web_view_stop_loading(v->web); break;
            case OP_EVAL: webkit_web_view_evaluate_javascript(v->web, a->text, -1, NULL, NULL, NULL, on_eval_done, NULL); break;
            case OP_RESIZE:
                v->width = a->width;
                v->height = a->height;
                wpe_view_backend_dispatch_set_size(v->backend, (uint32_t) a->width, (uint32_t) a->height);
                break;
            case OP_DISPOSE:
                v->closing = true;
                g_signal_handlers_disconnect_by_data(v->web, v);
                g_object_unref(v->web);
                v->web = NULL;
                break;
        }
    }
    if (a->op == OP_DISPOSE) {
        // closing is set on this (the WebKit) thread, where frames arrive,
        // so no on_export_image() pushes to the texture from here on.
        // Freeing it earlier (from the platform thread, while a playing
        // video was still delivering frames here) crashed the app in
        // texture_push_frame.
        flutterpi_post_platform_task(destroy_texture_on_platform, v->texture);
    }
    free(a->text);
    free(a);
    return G_SOURCE_REMOVE;
}

static void *wpe_thread(void *arg) {
    (void) arg;
    g_main_context_push_thread_default(wv.main);
    wpe_loader_init("libWPEBackend-fdo-1.0.so.1");
    if (!wpe_fdo_initialize_for_egl_display(wv.display)) {
        LOG_ERROR("[webview] wpe_fdo_initialize_for_egl_display failed\n");
    }
    g_main_loop_run(wv.loop);
    g_main_context_pop_thread_default(wv.main);
    return NULL;
}

static bool ensure_started(void) {
    if (wv.started) return true;
    struct gl_renderer *renderer = flutterpi_get_gl_renderer(wv.flutterpi);
    if (renderer == NULL) {
        LOG_ERROR("[webview] no GL renderer\n");
        return false;
    }
    wv.display = gl_renderer_get_egl_display(renderer);
    wv.context = gl_renderer_create_context(renderer);
    wv.image_target = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC) gl_renderer_get_proc_address(renderer, "glEGLImageTargetTexture2DOES");
    if (wv.context == EGL_NO_CONTEXT || wv.image_target == NULL) {
        LOG_ERROR("[webview] could not set up EGL\n");
        return false;
    }
    // Pages play sound through GStreamer in WebKit's own processes, which
    // find PipeWire through XDG_RUNTIME_DIR. The kiosk runs as a system
    // service without a login session, so it isn't set and pages were
    // silent. Set before WebKit starts, so its processes inherit it.
    if (getenv("XDG_RUNTIME_DIR") == NULL) {
        char dir[64];
        snprintf(dir, sizeof dir, "/run/user/%u", (unsigned) getuid());
        if (access(dir, R_OK | X_OK) == 0) setenv("XDG_RUNTIME_DIR", dir, 0);
    }
    wv.main = g_main_context_new();
    wv.loop = g_main_loop_new(wv.main, FALSE);
    if (pthread_create(&wv.thread, NULL, wpe_thread, NULL) != 0) return false;
    pthread_setname_np(wv.thread, "wpe-webview");
    wv.started = true;
    return true;
}

// ─── Channel ────────────────────────────────────────────────────────────

static double num_arg(struct json_value *args, char *key, double fallback) {
    struct json_value *v = jsobject_get(args, key);
    return v && v->type == kJsonNumber ? v->number_value : fallback;
}

static char *str_arg(struct json_value *args, char *key) {
    struct json_value *v = jsobject_get(args, key);
    return v && v->type == kJsonString ? v->string_value : NULL;
}

static void put_utf8(char **out, uint32_t cp) {
    char *o = *out;
    if (cp < 0x80) {
        *o++ = (char) cp;
    } else if (cp < 0x800) {
        *o++ = (char) (0xC0 | (cp >> 6));
        *o++ = (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        *o++ = (char) (0xE0 | (cp >> 12));
        *o++ = (char) (0x80 | ((cp >> 6) & 0x3F));
        *o++ = (char) (0x80 | (cp & 0x3F));
    } else {
        *o++ = (char) (0xF0 | (cp >> 18));
        *o++ = (char) (0x80 | ((cp >> 12) & 0x3F));
        *o++ = (char) (0x80 | ((cp >> 6) & 0x3F));
        *o++ = (char) (0x80 | (cp & 0x3F));
    }
    *out = o;
}

/// flutter-pi's JSON decoder hands strings over still escaped (`\"`, `\\`,
/// `\u00e1`…); a script or URL needs them undone. Returns a new string.
static char *json_unescape(const char *in) {
    if (in == NULL) return NULL;
    char *out = malloc(strlen(in) * 2 + 1), *o = out;
    for (const char *c = in; *c; c++) {
        if (*c != '\\' || c[1] == 0) {
            *o++ = *c;
            continue;
        }
        c++;
        switch (*c) {
            case 'n': *o++ = '\n'; break;
            case 't': *o++ = '\t'; break;
            case 'r': *o++ = '\r'; break;
            case 'b': *o++ = '\b'; break;
            case 'f': *o++ = '\f'; break;
            case 'u': {
                uint32_t cp = 0;
                if (sscanf(c + 1, "%4x", &cp) != 1) {
                    *o++ = 'u';
                    break;
                }
                c += 4;
                if (cp >= 0xD800 && cp < 0xDC00 && c[1] == '\\' && c[2] == 'u') {
                    uint32_t lo = 0;
                    if (sscanf(c + 3, "%4x", &lo) == 1 && lo >= 0xDC00 && lo < 0xE000) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        c += 6;
                    }
                }
                put_utf8(&o, cp);
                break;
            }
            default: *o++ = *c; break;  // \" \\ \/
        }
    }
    *o = 0;
    return out;
}

static int on_message(char *channel, struct platch_obj *object, FlutterPlatformMessageResponseHandle *handle) {
    (void) channel;
    struct json_value *args = &object->json_arg;
    const char *method = object->method;

    if (strcmp(method, "create") == 0) {
        char *url = str_arg(args, "url");
        if (url == NULL) return platch_respond_illegal_arg_json(handle, "url is required");
        if (!ensure_started()) return platch_respond_error_json(handle, "unavailable", "WPE could not start", NULL);
        int slot = -1;
        for (int i = 0; i < MAX_VIEWS; i++) {
            if (wv.views[i] == NULL) {
                slot = i;
                break;
            }
        }
        if (slot < 0) return platch_respond_error_json(handle, "busy", "too many web views", NULL);

        struct view *v = calloc(1, sizeof *v);
        v->texture = texture_new(flutterpi_get_texture_registry(wv.flutterpi));
        v->id = texture_get_id(v->texture);
        v->width = (int) num_arg(args, "width", 1280);
        v->height = (int) num_arg(args, "height", 1600);
        v->scale = num_arg(args, "scale", 1);
        wv.views[slot] = v;

        struct create_args *a = calloc(1, sizeof *a);
        a->view = v;
        a->url = json_unescape(url);
        g_main_context_invoke(wv.main, create_on_wpe, a);
        return platch_respond_success_json(handle, &JSONOBJECT1("id", JSONNUM((double) v->id)));
    }

    struct view *v = find_view((int64_t) num_arg(args, "id", -1));
    if (v == NULL) return platch_respond_illegal_arg_json(handle, "unknown id");

    if (strcmp(method, "touch") == 0) {
        struct touch_args *a = calloc(1, sizeof *a);
        a->view = v;
        a->type = (int) num_arg(args, "type", 0);
        a->x = num_arg(args, "x", 0);
        a->y = num_arg(args, "y", 0);
        g_main_context_invoke(wv.main, touch_on_wpe, a);
        return platch_respond_success_json(handle, &JSONNULL);
    }

    if (strcmp(method, "click") == 0 || strcmp(method, "scroll") == 0) {
        struct pointer_args *a = calloc(1, sizeof *a);
        a->view = v;
        a->click = method[0] == 'c';
        a->x = num_arg(args, "x", 0);
        a->y = num_arg(args, "y", 0);
        a->dx = num_arg(args, "dx", 0);
        a->dy = num_arg(args, "dy", 0);
        g_main_context_invoke(wv.main, pointer_on_wpe, a);
        return platch_respond_success_json(handle, &JSONNULL);
    }

    if (strcmp(method, "type") == 0) {
        struct type_args *a = calloc(1, sizeof *a);
        a->view = v;
        a->back = (int) num_arg(args, "back", 0);
        a->text = json_unescape(str_arg(args, "text"));
        struct json_value *enter = jsobject_get(args, "enter");
        a->enter = enter && enter->type == kJsonTrue;
        g_main_context_invoke(wv.main, type_on_wpe, a);
        return platch_respond_success_json(handle, &JSONNULL);
    }

    int op = strcmp(method, "load") == 0      ? OP_LOAD
           : strcmp(method, "back") == 0      ? OP_BACK
           : strcmp(method, "forward") == 0   ? OP_FORWARD
           : strcmp(method, "reload") == 0    ? OP_RELOAD
           : strcmp(method, "stop") == 0      ? OP_STOP
           : strcmp(method, "eval") == 0      ? OP_EVAL
           : strcmp(method, "resize") == 0    ? OP_RESIZE
           : strcmp(method, "dispose") == 0   ? OP_DISPOSE
                                              : -1;
    if (op < 0) return platch_respond_not_implemented(handle);

    struct simple_args *a = calloc(1, sizeof *a);
    a->view = v;
    a->op = op;
    char *text = str_arg(args, op == OP_EVAL ? "js" : "url");
    a->text = json_unescape(text);
    a->width = (int) num_arg(args, "width", v->width);
    a->height = (int) num_arg(args, "height", v->height);
    if ((op == OP_LOAD || op == OP_EVAL) && a->text == NULL) {
        free(a);
        return platch_respond_illegal_arg_json(handle, "missing argument");
    }
    g_main_context_invoke(wv.main, simple_on_wpe, a);

    if (op == OP_DISPOSE) {
        for (int i = 0; i < MAX_VIEWS; i++) {
            if (wv.views[i] == v) wv.views[i] = NULL;
        }
        // The view struct stays alive: frames still in flight refer to it.
        // Its texture is freed by simple_on_wpe once WebKit has stopped.
    }
    return platch_respond_success_json(handle, &JSONNULL);
}

enum plugin_init_result wpe_webview_init(struct flutterpi *flutterpi, void **userdata_out) {
    wv.flutterpi = flutterpi;
    if (plugin_registry_set_receiver_locked(CHANNEL, kJSONMethodCall, on_message) != 0) return PLUGIN_INIT_RESULT_ERROR;
    *userdata_out = NULL;
    return PLUGIN_INIT_RESULT_INITIALIZED;
}

void wpe_webview_deinit(struct flutterpi *flutterpi, void *userdata) {
    (void) flutterpi;
    (void) userdata;
    plugin_registry_remove_receiver_locked(CHANNEL);
    if (wv.started) {
        g_main_loop_quit(wv.loop);
        pthread_join(wv.thread, NULL);
    }
}

FLUTTERPI_PLUGIN("wpe webview", wpe_webview, wpe_webview_init, wpe_webview_deinit)
