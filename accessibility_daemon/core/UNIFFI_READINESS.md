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

## Resolved: `OcrOverlayState` is `!Send + !Sync` (ticket 04b, 2026-09-21)

Decision: **split**, not pin. The state machine stays in core for the
desktop binary and remains thread-confined (`Rc` cache, `Cell` viewport);
it is desktop-only and is never exposed to a binding.

The pure lookup/format half now lives in `core/src/lookup.rs` as free
functions over plain data:

- `lookup::prepare_search_candidates(following_text, &Deinflector)`
- `lookup::process_results(db_rows, candidates_by_length, following_text)`
- `lookup::format_dictionary_results(matches, dict_names)`
- `lookup::following_text(active_all_chars, global_idx)`
- `lookup::lookup_term(active_all_chars, global_idx, following_text, db, deinflector)`
  → `Option<LookupOutcome>`

`LookupOutcome` carries an owned `Vec<FormattedEntry>`, not an `Rc`. The
module has no `Rc`/`Cell`/interior mutability and is `Send`; the desktop
`OcrOverlayState::lookup` / `format_dictionary_results` delegate to it with
unchanged signatures, so the PC call sites and the viewport state (zoom,
pan, nav graph, cursor, the `Rc` cache) are untouched. Nav-graph data
(`nav_graph::NavGraph`) was already pure and sits on the same binding
surface.

A `uniffi::Object` was rejected: UniFFI 0.28 requires exported objects to
be `Send + Sync` and calls them through `Arc<Self>` (so `&mut self` is not
usable), which would force either an actor shell around the `!Send` state
or an `Arc`/atomic/lock rewrite of the desktop viewport for no PC benefit.
The binding surface is free functions, matching how mobile already consumes
`nav_graph_core` (`build_nav_graph` / `navigate` + records, called
synchronously from the UI thread). Full option scoring is in the ticket-04
comment.
