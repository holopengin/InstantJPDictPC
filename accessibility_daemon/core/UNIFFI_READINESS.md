# UniFFI-readiness note (ticket pipeline-sharing/03, doc only — no binding implemented)

The `jpdict_core` crate (`accessibility_daemon/core/`) is the unit a
Kotlin/Android host would consume through generated bindings (UniFFI or a
thin JNI shim). Boundary vocabulary follows what crosses the JNI line in
the Android app today (`OcrEngine.kt` + `DetNcnn`/`RecNcnn`/`KanaSizeNcnn`
external functions, pure-module stages like `FuriganaRule`,
`RotatedGeometry`, `CharLm`, `KanaSize` windows/policy, `Definitions`/
format): image bytes in; boxes, text, char geometry, formatted entries out.

## What a binding needs

1. **Asset location.** The core never finds its own files: every loader
   takes an explicit directory (`OcrEngine::new(asset_dir, …)`,
   `CharLm::load(path)`, `Deinflector::from_json_file(path)`,
   `DictionaryDatabase::open(path)`). The host passes its asset directory
   (Android `ModelAssets`), mirroring how `OcrEngine` prefs resolve today.
   The `include_str!` catalog (`data/catalog.rs`) is the one exception — it
   is embedded at build time and needs no host path.
2. **Error type.** Fallible entry points return `anyhow::Result`; a binding
   needs one flat error enum (e.g. `CoreError { AssetMissing, Inference,
   Database, Import, InvalidInput }` with a message) instead. The
   `anyhow` chains are for desktop logs, not for crossing FFI.
3. **Result serialisation.** Public outputs are plain data and already
   `serde`-friendly or trivially made so: `BoundingBox`/`RotatedBox`/`LineResult`
   (ints, floats, strings), `FormattedEntry` reading/headword/sense groups,
   `ImportProgress` (phase + counts). A binding serialises these (JSON or
   UniFFI records) — noHandles, no borrowed references escape.
4. **Progress callback.** Import and recognition already take
   caller-provided callbacks (`DictionaryImporter::import_zip(path, Option<cb>)`,
   `recognize_boxes_streaming(…, emitter)`); UniFFI would wrap these as
   callback interfaces. They are call-and-return (never stored), so no
   lifetime crosses the boundary.
5. **Threading contract.** `OcrEngine` is `Send` but not `Sync` (the
   detector is `Send`-only); detection runs on the calling thread while
   recognition fans out internally and joins before returning. The host
   must call the engine from one thread (or one engine per thread).
   `DictionaryDatabase` is `Send + Sync` and shareable.

## Open before any binding: `OcrOverlayState` is `!Send + !Sync`

The state machine moved into core deliberately (it is the lookup/format
pipeline, with no UI types — only `Rc` caches and `Cell` viewport fields),
but its `Rc`/`Cell` interior makes it thread-confined. A binding needs one
of: (a) pin the state to a defining thread and expose it as an opaque
handle with a single-threaded executor on the host; (b) split the pure
functions (`lookup_*`, `format_dictionary_results`, definition parsing)
from the viewport state (`current_scale/trans`, window `Cell`s, nav graph)
so the pure half becomes `Send`. Option (b) is recommended — the pure half
is already separable (it takes `&DictionaryDatabase` + `&Deinflector` and
returns owned data), while the viewport half stays desktop-only. Not done
here: the desktop binary uses the state directly, and the split is only
required once a real binding exists.
