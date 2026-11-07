# Shared PP-OCRv6 ncnn backend

Detection and recognition for this app run the same ncnn models and the same
native kernel as the Android app (InstantJPDict). There is deliberately one
backend, not two:

| Piece | Mobile (InstantJPDict) | PC (this repo) |
|---|---|---|
| PP-OCR kernel / net config / blob extraction / top-K | `app/src/main/cpp/ppocr_ncnn_core.{h,cpp}` | `native/ppocr_ncnn/ppocr_ncnn_core.{h,cpp}` (identical copy) |
| PP-OCR platform shell | `app/src/main/cpp/ncnn_jni.cpp` (JNI) | `native/ppocr_ncnn/ppocr_ncnn_capi.{h,cpp}` (C ABI) |
| Kana small/large kernel | `app/src/main/cpp/kana_size_core.{h,cpp}` | `native/kana_size/kana_size_core.{h,cpp}` (identical copy) |
| Kana platform shell | `app/src/main/cpp/kana_size_ncnn.cpp` (JNI) | `native/kana_size/kana_size_capi.{h,cpp}` (C ABI) |
| Rust side | — | `jpdict-core/src/ppocr_ncnn.rs` (FFI), `jpdict-core/src/ppocr.rs` (pre/decode), `jpdict-core/src/ocr_engine.rs` (det pre/post) |
| Models | `app/src/main/assets/PP-OCRv6_small_ncnn/`, `kana_size/` | `assets/PP-OCRv6_small_ncnn/`, `assets/kana_size/` (copied) |
| Shared pipeline data | `app/src/main/assets/{deinflect.json,catalog,components,lm,variants}` | `assets/{...}` (copied) |

The core files must stay byte-identical. The mobile repo is the source of
truth: land kernel changes there first, then mirror them here with

```
tools/sync_ppocr_core.sh --check  /path/to/InstantJPDict   # verify
tools/sync_ppocr_core.sh --update /path/to/InstantJPDict   # mirror mobile -> PC
```

The same script mirrors the model assets, the kana model, and the shared
pipeline data listed above. The CTC head width is always derived from
`rec_remap.txt` — never hardcode it. `.githooks/pre-commit` and the
`shared-ocr-sync` GitHub workflow run `--check`, so drift fails a commit or a
PR instead of waiting to be noticed.

## Building the pinned ncnn fork

```
tools/build_ncnn_pc.sh
```

Clones `holopengin/ncnn` at the same pinned commit the Android build ships
(`FORK_PIN` in the script must stay equal to InstantJPDict's
`tools/build_ncnn.sh`), verifies the fork's required patches, builds a
CPU-only static `libncnn.a`, and installs it to `third_party/ncnn-pc/install`.
`build.rs` links that tree; `NCNN_PC_DIR` overrides the location.

## What the PC wrappers add

- `jpdict-core/src/ppocr_ncnn.rs`: handle lifetimes, float buffer copies, `Send`/`Sync`
  (one `Net` per model, one `Extractor` per inference).
- `jpdict-core/src/ppocr.rs`: mobile's preprocessing (rotate 270° portrait, resize to
  48×W, mult-of-8 zero pad, `(gray/127.5)-1`, lengthwise squish) and CTC
  decode (class remap, top-15, sub-column peak interpolation), plus the
  long-line chunk-and-stitch path above the 2000 targetW gate.
- `jpdict-core/src/ocr_engine.rs`: mobile's det letterbox/ImageNet normalisation plus its
  furigana line rejection (#28) and ruby-gutter trim (#48), then the PC app's
  own rotated-box post-processing on the ncnn probability map.

## Boundary decision: what stays platform-side

The shared backend is deliberately three layers:

1. **Shared native kernels and models.** `ppocr_ncnn_core` / `kana_size_core`
   and the model/data assets above are mirrored byte-for-byte and checked by
   CI and the pre-commit hook.
2. **Shared pure Rust pipeline.** `jpdict-core` owns decode, placement,
   reading order, lookup and formatting; Android consumes it through
   `nav_graph_core`'s UniFFI boundary.
3. **Platform I/O and hot pixel loops.** Capture/decoding, `Bitmap`/`Canvas`
   scaling, det/rec pixel packing, and the long-line inference orchestration
   stay per-platform. They are bound to each platform's image type, and
   crossing a full pixel/logit buffer per line over UniFFI costs more than the
   duplicated code.

The duplicated stages (det letterbox, rec input packing, long-line stitch)
are pinned by `tests/conformance` and the synth parity tests. If a parity bug
ever originates in one of them, move **that stage** into `jpdict-core` rather
than porting the whole pipeline at once.

No shared repo or submodule is planned: the two-file C++ mirror is now
enforced by `.githooks/pre-commit` and the `shared-ocr-sync` workflow, which
removes the friction that would have justified one. The full decision record
is ADR 0002 in the Android repo
(`docs/adr/0002-shared-ocr-backend-boundary.md`).

Parity smoke tests: `cargo test synth_` runs the mobile androidTest synth set
(`test_images/synth/`) through the same models and gates on per-line CER and
detection IoU.
