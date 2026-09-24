//! UI-free OCR + dictionary pipeline.
//!
//! This crate owns the behaviour the PC port reverse-engineered out of the
//! Android app: detection post-processing, rotated geometry, reading order,
//! recognition decode, kana correction, ruby rules, blank handling, and
//! dictionary import/lookup/formatting. The desktop UI (iced), capture
//! (KWin/zbus), and binary glue stay in the `accessibility_daemon` binary,
//! which consumes this crate. No module here may depend on iced, winit,
//! KWin/zbus, or rfd.
//!
//! # Data in / data out
//!
//! * In: image bytes (`image::DynamicImage` / raw RGB), asset directories
//!   (model files, `deinflect.json`, LM tables), dictionary files (Yomitan
//!   ZIPs, catalog entries), settings data ([`app_settings::AppSettings`]).
//! * Out: detection boxes/quads ([`models::BoundingBox`],
//!   [`models::RotatedBox`]), ordered line texts with char geometry
//!   ([`models::LineResult`]), formatted dictionary entries
//!   ([`models::FormattedEntry`]), install progress
//!   (`data::importer::ImportProgress`, `db` feature).
//! * No UI types cross the boundary: colours are `[f32; 3]` RGB
//!   ([`ruby_style::RubyStyle`]), sizes are `f32` display points, geometry is
//!   plain pixel structs. The binary converts these to iced types.
//!
//! # Thread-safety and ownership
//!
//! * [`ocr_engine::OcrEngine`] (`native` feature) is `Send` but **not**
//!   `Sync`: the detector (`ppocr_ncnn::DetNet`) is `Send`-only, so
//!   detection runs on one thread (today the caller's); recognition fans out
//!   to worker threads over the `Sync` recognizer behind `Arc`
//!   (`recognize_boxes_streaming` borrows the engine and joins the workers
//!   before returning — no handle escapes).
//! * [`data::db::DictionaryDatabase`] (`db` feature) is `Send + Sync`
//!   (`Arc<Mutex<Connection>>`).
//! * [`overlay_state::OcrOverlayState`] (`db` feature) is `!Send + !Sync`
//!   (`Rc` caches, `Cell` viewport) and is **desktop-only**: confine it to
//!   the UI thread. The pure lookup/format pipeline lives in [`lookup`]
//!   (`db` feature) as free functions over plain data (no `Rc`, no `Cell`),
//!   so it is `Send` and is the surface a binding exposes; the desktop state
//!   delegates to it. See `core/UNIFFI_READINESS.md`.
//! * Callbacks are ownership-free: `recognize_boxes_collect` returns owned
//!   `Vec`s; the streaming variant drives a caller-provided `Fn` emitter and
//!   blocks until drained. Import progress is a pull/poll callback
//!   (`Option<impl Fn(ImportProgress)>`), never a stored handle.
//!
//! Test-only: the shared conformance corpus lives at
//! `accessibility_daemon/tests/conformance/` (kept there so the documented
//! location in `tests/conformance/FORMAT.md` and the Android re-sync
//! procedure keep working); this crate's harness (`conformance`, `#[cfg(test)]`)
//! reaches it via `../tests`, as do the `assets/`, `fonts/`, and
//! `test_images/` fixtures.
//!
//! # Features
//!
//! The crate builds in two configurations (see `FEATURES.md` for the full
//! module table). Default (`native`, `db`) is the whole pipeline. With
//! `--no-default-features` only the pure-Rust algorithm modules remain —
//! starting with `nav_graph` (+ its `models` surface) — and the crate builds
//! with no C++ toolchain, no ncnn sources, and no install prefix.

pub mod app_settings;
pub mod blank_gaps;
pub mod char_boxes;
pub mod char_placement;
pub mod data;
pub mod furigana;
pub mod kana_size;
#[cfg(feature = "db")]
pub mod lookup;
pub mod models;
pub mod nav_graph;
#[cfg(feature = "native")]
pub mod ocr_engine;
pub mod overlay_font;
#[cfg(feature = "db")]
pub mod overlay_state;
#[cfg(feature = "native")]
pub mod ppocr;
#[cfg(feature = "native")]
pub mod ppocr_ncnn;
pub mod ruby_style;
pub mod util;

#[cfg(all(test, feature = "native", feature = "db"))]
mod conformance;
