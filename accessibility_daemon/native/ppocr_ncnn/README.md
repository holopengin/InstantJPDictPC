# Shared PP-OCRv6 ncnn backend

Detection and recognition for this app run the same ncnn models and the same
native kernel as the Android app (InstantJPDict). There is deliberately one
backend, not two:

| Piece | Mobile (InstantJPDict) | PC (this repo) |
|---|---|---|
| Kernel / net config / blob extraction / top-K | `app/src/main/cpp/ppocr_ncnn_core.{h,cpp}` | `native/ppocr_ncnn/ppocr_ncnn_core.{h,cpp}` (identical copy) |
| Platform shell | `app/src/main/cpp/ncnn_jni.cpp` (JNI) | `native/ppocr_ncnn/ppocr_ncnn_capi.{h,cpp}` (C ABI) |
| Rust side | — | `src/ppocr_ncnn.rs` (FFI), `src/ppocr.rs` (pre/decode), `src/ocr_engine.rs` (det pre/post) |
| Models | `app/src/main/assets/PP-OCRv6_small_ncnn/` | `assets/PP-OCRv6_small_ncnn/` (copied) |

The core files must stay byte-identical. The mobile repo is the source of
truth: land kernel changes there first, then mirror them here with

```
tools/sync_ppocr_core.sh --check  /path/to/InstantJPDict   # verify
tools/sync_ppocr_core.sh --update /path/to/InstantJPDict   # mirror mobile -> PC
```

Model assets (`det.param/.bin`, `rec_dyn.param/.bin`, `vocab.json`,
`rec_remap.txt`) are copied from the mobile repo the same way. The CTC head
width is always derived from `rec_remap.txt` — never hardcode it.

## Building the pinned ncnn fork

```
tools/build_ncnn_pc.sh
```

Clones `vgf89/ncnn` at the same pinned commit the Android build ships
(`FORK_PIN` in the script must stay equal to InstantJPDict's
`tools/build_ncnn.sh`), verifies the fork's required patches, builds a
CPU-only static `libncnn.a`, and installs it to `third_party/ncnn-pc/install`.
`build.rs` links that tree; `NCNN_PC_DIR` overrides the location.

## What the PC wrappers add

- `src/ppocr_ncnn.rs`: handle lifetimes, float buffer copies, `Send`/`Sync`
  (one `Net` per model, one `Extractor` per inference).
- `src/ppocr.rs`: mobile's preprocessing (rotate 270° portrait, resize to
  48×W, mult-of-8 zero pad, `(gray/127.5)-1`, lengthwise squish) and CTC
  decode (class remap, top-15, sub-column peak interpolation), plus the
  long-line chunk-and-stitch path above the 2000 targetW gate.
- `src/ocr_engine.rs`: mobile's det letterbox/ImageNet normalisation plus its
  furigana line rejection (#28) and ruby-gutter trim (#48), then the PC app's
  own rotated-box post-processing on the ncnn probability map.

Parity smoke tests: `cargo test synth_` runs the mobile androidTest synth set
(`test_images/synth/`) through the same models and gates on per-line CER and
detection IoU.
