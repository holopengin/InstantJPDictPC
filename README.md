# InstantJPDict

Japanese OCR overlay and dictionary lookup for KDE Plasma. Point it at a
screenshot (or press a global shortcut), select a word, and read the
dictionary entry in the overlay.

The OCR/dictionary pipeline is a UI-free Rust crate (`jpdict-core`); the
desktop binary (`instantjpdict`) adds the Rune UI, KWin capture, and settings
window.

## Build

The OCR backend uses a pinned ncnn build. Build it once, then build the app:

```sh
./tools/build_ncnn_pc.sh
cargo build --release
```

The binary is `target/release/instantjpdict`. `assets/` and `fonts/` are
staged next to it by the build script.

For development:

```sh
cargo run                      # open the settings window
cargo run -- screenshot.png    # OCR an image and open the viewer
cargo test --workspace
```

You can also drag a JPEG, PNG, JPEG XL, or WebP file onto either the launcher
window or an open OCR window. Dropping on the launcher opens a new OCR window;
dropping on an existing viewer reloads that window with the new image.

## Set up window capture

Window capture uses KWin's `ScreenShot2` D-Bus API, so it needs two things:
a desktop file that authorizes the binary, and a global shortcut that runs it
with `--capture`. The app does both in one step:

```sh
./target/release/instantjpdict --setup-capture
```

This registers `Meta+Print` (falling back to `Ctrl+Alt+S` if that key is
already taken) and applies the shortcut to the running Plasma session
immediately. KWin scans its screenshot allow-list at startup, so a fresh
install may need one logout/login before the first capture works; upgrades
keep a compatibility entry that authorizes the current session. The same
setup is available in the app under **Settings → Window Capture → Set Up
Capture Shortcut**.

The command the shortcut runs is:

```sh
/path/to/instantjpdict --capture
```

To use a different key, add that command in **System Settings → Keyboard →
Shortcuts → Add → Command or Script**. The settings window can copy the
command and open that page for you.

Pressing the shortcut screenshots the focused window and opens the OCR viewer
on it directly; nothing is written to `~/Pictures/Screenshots`.

## Layout

```
src/                desktop binary: UI, capture, settings
jpdict-core/        UI-free OCR & dictionary pipeline
assets/             models, dictionary data, licenses
fonts/              bundled Noto Sans/Serif JP
native/             shared PP-OCRv6 / kana ncnn C++ cores
tests/              integration tests + shared conformance corpus
test_images/        OCR fixtures
tools/              ncnn build and mobile-sync scripts
packaging/          .desktop file, icon, AppImage build
docs/pipeline/      pipeline invariant specifications
```

## License

AGPL-3.0-or-later. Third-party assets and their licenses are documented under
`assets/licenses/` and next to each asset.
