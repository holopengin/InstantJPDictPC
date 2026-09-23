# `jpdict_core` feature contract (ticket pipeline-sharing/05)

Two configurations, no duplicated modules. The desktop binary depends on the
crate with default features, so default behaviour is unchanged.

## Features

| Feature | Default | What it pulls in |
|---------|---------|------------------|
| `native` | on | C++ inference stack: pinned ncnn fork + `ppocr_ncnn` core + `kana_size` C ABI (compiled in `build.rs`), plus the `image` / `imageproc` / `ttf-parser` crates. |
| `db` | on | Dictionary stack: `rusqlite` (bundled SQLite), `zip` (Yomitan import), `ureq` + `sha2` (verified catalog downloads). |

`default = ["native", "db"]`. Either can be enabled alone
(`--no-default-features --features native`, etc.).

## Module availability

Always available (pure Rust, no toolchain / prefix / network beyond crates.io):

- `models` — shared geometry + result types (the `nav_graph` surface).
- `nav_graph` — D-pad navigation graph over detection boxes.
- `furigana`, `ruby_style` — ruby filtering rules and overlay styling.
- `char_placement` — CTC-anchored per-character box placement (CAP; the
  `BOX_PLACEMENT_CAP` switch in `ocr_engine` picks it over the legacy chain).
- `kana_size` (pure half) — window encoder (`window`), ε policy
  (`correct_lines`, `prob_big`, `BASE_ORDER`, `Flip`/`Declined`/`Correction`).
  The `KanaSizeNet` native handle needs `native`.
- `app_settings`, `overlay_font` — settings + font-face resolution.
- `util` — `char_lm`, `component_table`, `deinflector`, `gap_candidates`,
  `japanese`, `kanji_variants`, `oov_candidates`, `oov_suggestions`, `pitch`.
- `data::{catalog, models}` — pinned catalog (embedded JSON) + DB row types.

`native` only:

- `ppocr_ncnn` — FFI bindings to the shared PP-OCRv6 ncnn core.
- `ppocr` — preprocess / decode / batch inference over `ppocr_ncnn`.
- `ocr_engine` — `OcrEngine` detection + recognition pipeline.
- `kana_size::KanaSizeNet` — the `nb_all` ncnn model handle.

`db` only:

- `data::{db, importer, download, bundled}` — SQLite database, Yomitan ZIP
  import, verified catalog download, first-run bundled install.
- `overlay_state` — lookup/format pipeline over `DictionaryDatabase`
  (thread-confined; see `UNIFFI_READINESS.md`).
- `lookup` — the pure free-function half of the pipeline
  (`prepare_search_candidates`, `process_results`,
  `format_dictionary_results`, `lookup_term`); calls into `overlay_state`
  parsers, hence `db`-gated with it.

Test-only `conformance` harness needs both features
(`#[cfg(all(test, feature = "native", feature = "db"))]`).

## Mobile-swap notes (ticket pipeline-sharing/04)

- An Android consumer that only needs the algorithm modules depends on
  `jpdict_core` with `default-features = false` (optionally `features = ["db"]`
  if it also wants the dictionary stack without inference) and never pays for
  — or requires the toolchain for — the native inference stack.
- `build.rs` returns immediately when `native` is off: no C++ compiler
  invoked, no ncnn sources read, no `NCNN_PC_DIR` consulted, no panic.
- With `native` on, `build.rs` behaviour is identical to before (pinned-fork
  check + panic message, both C ABI compilations, assets/fonts copy).
