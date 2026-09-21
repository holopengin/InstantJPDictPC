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
//!   ([`data::importer::ImportProgress`]).
//! * No UI types cross the boundary: colours are `[f32; 3]` RGB
//!   ([`ruby_style::RubyStyle`]), sizes are `f32` display points, geometry is
//!   plain pixel structs. The binary converts these to iced types.
//!
//! # Thread-safety and ownership
//!
//! * [`ocr_engine::OcrEngine`] is `Send` but **not** `Sync`: the detector
//!   (`ppocr_ncnn::DetNet`) is `Send`-only, so detection runs on one thread
//!   (today the caller's); recognition fans out to worker threads over the
//!   `Sync` recognizer behind `Arc` (`recognize_boxes_streaming` borrows the
//!   engine and joins the workers before returning — no handle escapes).
//! * [`data::db::DictionaryDatabase`] is `Send + Sync` (`Arc<Mutex<Connection>>`).
//! * [`overlay_state::OcrOverlayState`] is `!Send + !Sync` (`Rc` caches,
//!   `Cell` viewport): confine it to one thread. A future UniFFI binding
//!   must either pin it to a defining thread or split the pure
//!   lookup/format functions from the viewport state (see
//!   `core/UNIFFI_READINESS.md`).
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

pub mod app_settings;
pub mod data;
pub mod furigana;
pub mod kana_size;
pub mod models;
pub mod nav_graph;
pub mod ocr_engine;
pub mod overlay_font;
pub mod overlay_state;
pub mod ppocr;
pub mod ppocr_ncnn;
pub mod ruby_style;
pub mod util;

#[cfg(test)]
mod conformance;
