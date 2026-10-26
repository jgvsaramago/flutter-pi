# flutter-pi for ByteHomeDash

This fork's `master` is upstream [flutter-pi](https://github.com/ardera/flutter-pi)
`master` (based on `release/1.1.1`, the version `flutterpi_tool` 0.12.0
bundles for Flutter 3.44.6) plus the changes the
[ByteHomeDash](https://github.com/jgvsaramago/bytehomedash) kiosk needs,
on top. One commit per change, each explaining why:

| Commit | What | Why |
|---|---|---|
| gstreamer video player: make live RTSP streams play | live sources report `initialized`, start in PLAYING, 1080p system-memory frames upload, `setVolume` works | the kiosk shows the door camera's live RTSP stream |
| Add a WPE WebKit web view plugin | `src/plugins/wpe_webview.c`, `-DBUILD_WPE_WEBVIEW_PLUGIN=ON` | the kiosk's in-app browser, zero-copy GPU frames |
| platformchannel: fix quotes in strings sent to Dart | the JSON encoder wrote `"` as `\t` | page titles and JSON-in-strings arrived garbled |

## Build

```sh
git clone https://github.com/jgvsaramago/flutter-pi.git
cd flutter-pi && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WPE_WEBVIEW_PLUGIN=ON ..
make -j"$(nproc)"
```

ByteHomeDash's `device/setup.sh` does this and installs the binary to
`/usr/local/bin/flutter-pi`.

## Updating from upstream

Rebase these commits onto upstream's `master` (`git pull --rebase
https://github.com/ardera/flutter-pi.git master`), check the kiosk still
builds and runs, then push.
