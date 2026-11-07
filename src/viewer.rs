//! OCR viewer, rendered with Rune.
//!
//! The screenshot is an `El::pixels` image inside a camera (a world element
//! sized to the window and panned by `.offset`). Zoom is applied to layout
//! metrics, not through `El::scale`, so glyphs rasterize at their final size
//! instead of being stretched from an offscreen layer. Every OCR box and glyph
//! is a normal element in that world, so hit testing, hover and accessibility
//! come from Rune's layout instead of a hand-written canvas program.

use std::cell::{Cell as StdCell, RefCell};
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::anyhow;
use rune::tok::TextStyle;
use rune::{
    col, el, row, text, AppSpec, Color, Cx, Drag, El, Key, KeyEvent, Phase, Pixels, ScrollSpec,
    Spring, Transition, Tx, WheelEvent,
};

use jpdict_core::data::db::DictionaryDatabase;
use jpdict_core::data::models::DictionaryEntry;
use jpdict_core::models::*;
use jpdict_core::overlay_font::FontFace;
use jpdict_core::overlay_state::OcrOverlayState;
use jpdict_core::util::deinflector::Deinflector;
use jpdict_core::util::japanese;
use jpdict_core::util::japanese::{is_half_width, to_vertical_glyph};

// ── palette ─────────────────────────────────────────────────────────────
const WHITE: Color = Color {
    r: 1.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};
const BLACK: Color = Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
const PINK: Color = Color {
    r: 1.0,
    g: 0.184475,
    b: 0.184475,
    a: 1.0,
};
const YELLOW: Color = Color {
    r: 1.0,
    g: 1.0,
    b: 0.0,
    a: 1.0,
};
const CYAN: Color = Color {
    r: 0.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};
const LT_GRAY: Color = Color {
    r: 0.522522,
    g: 0.522522,
    b: 0.522522,
    a: 1.0,
};
const PANEL_BG: Color = Color {
    r: 0.009721,
    g: 0.009721,
    b: 0.009721,
    a: 245.0 / 255.0,
};
const CHIP_BG: Color = Color {
    r: 0.052861,
    g: 0.052861,
    b: 0.052861,
    a: 1.0,
};

/// Messages the background bootstrap thread sends to the viewer.
pub enum BootstrapMsg {
    /// RGBA pixels (width × height × 4) and dimensions.
    ImageReady(Vec<u8>, u32, u32),
    DictReady(Arc<DictionaryDatabase>),
    DeinflectReady(Arc<Deinflector>),
    /// Complete initial result set: `annotations[i]` is detection box `i`,
    /// detection-only (`line: None`) where recognition produced no text.
    Results(Vec<DetectedAnnotation>),
    /// Complete replacement detection+recognition result after a live retune.
    Retuned(Vec<DetectedAnnotation>),
}

/// A live retune request sent from the viewer to the OCR worker.
#[derive(Clone, Copy)]
pub struct TuneCmd {
    pub thresh: f32,
    pub unclip: f32,
}

/// Commands sent from the viewer to the worker that owns the OCR engine and
/// the current image.
pub enum WorkerCmd {
    Tune(TuneCmd),
    Reload(PathBuf),
}

/// Text width available inside the fixed 300px dictionary column: panel and
/// scroll padding removed, with slack so estimated line breaks cannot
/// overflow the column.
const DICT_TEXT_WIDTH: f32 = 278.0;

/// Body text size in definitions and examples (mobile: 15sp) and the ruby
/// size that sits above it (mobile: 9sp).
const DEF_TEXT_SIZE: f32 = 15.0;
const DEF_RUBY_SIZE: f32 = 9.0;

/// Body text size in definitions and examples (mobile 15sp).
const DICT_WIDTH: f32 = 300.0;
const NEIGHBOR_WIDTH: f32 = 42.0;

/// Mobile line-box / quad-border fill: `Color.argb(100, 0, 0, 0)`.
const BOX_FILL_ALPHA: f32 = 100.0 / 255.0;
/// Mobile corner radius on the box fill (raw source px).
const BOX_CORNER_RADIUS: f32 = 4.0;
/// Mobile `updateCursor`: a 2dp white outline, rounded 4px.
const CURSOR_RADIUS: f32 = 4.0;
const CURSOR_STROKE: f32 = 2.0;
/// Font fill ratio for character buttons in the neighbor/alternatives panels.
const BUTTON_CHAR_RATIO: f32 = 0.6;
/// Fraction of the em that the reference glyph `あ` inks vertically; the
/// em is divided by it so the ink, not the em square, fills the OCR box.
const DEFAULT_INK_RATIO: f32 = 0.85;
/// Mobile LineOverlayView `ASCII_GLYPH_SCALE` (#49): halfwidth glyphs render
/// ~10% smaller than CJK at the same em.
const ASCII_GLYPH_SCALE: f32 = 0.9;
/// Fraction of the char box the ink fills along the cross axis, once the em
/// is corrected for the face's ink ratio.
const GLYPH_FILL_RATIO: f32 = 0.98;
/// Mobile LineOverlayView per-glyph box fit: `maxW = boxW * 0.92f`.
const GLYPH_FIT_RATIO: f32 = 0.92;

/// Atomic inline cell for the definition flow layout.
enum InlineCell {
    Char(char),
    Ruby {
        term: String,
        reading: String,
        width: f32,
    },
    Tag {
        text: String,
        width: f32,
    },
}

/// Fontdue metrics for the panel face, given once by the app at startup.
static PANEL_METRICS: std::sync::OnceLock<fontdue::Font> = std::sync::OnceLock::new();

/// Hand the flow layout the panel font's real advances. Call once with the
/// same bytes Rune registers through `AppSpec::fonts`.
pub fn init_panel_metrics(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    if let Ok(font) = fontdue::Font::from_bytes(bytes.to_vec(), fontdue::FontSettings::default()) {
        let _ = PANEL_METRICS.set(font);
    }
}

fn measured_advance(font: &fontdue::Font, text: &str, size: f32) -> f32 {
    let mut width = 0.0;
    let mut prev: Option<u16> = None;
    for ch in text.chars() {
        let gid = font.lookup_glyph_index(ch);
        if let Some(p) = prev {
            if let Some(kern) = font.horizontal_kern_indexed(p, gid, size) {
                width += kern;
            }
        }
        width += font.metrics(ch, size).advance_width;
        prev = Some(gid);
    }
    width
}

/// Advance estimate for one character at `size`: the app's halfwidth rule
/// (ASCII and halfwidth kana) is half an em; everything else is one em.
fn inline_char_width(ch: char, size: f32) -> f32 {
    if let Some(font) = PANEL_METRICS.get() {
        return font.metrics(ch, size).advance_width;
    }
    if ch == '\u{3000}' {
        size
    } else if is_half_width(ch) {
        size * 0.5
    } else {
        size
    }
}

fn inline_text_width(text: &str, size: f32) -> f32 {
    if let Some(font) = PANEL_METRICS.get() {
        return measured_advance(font, text, size);
    }
    text.chars().map(|c| inline_char_width(c, size)).sum()
}

/// Japanese line-start prohibition (kinsoku): these may not begin a line.
fn must_not_start_line(ch: char) -> bool {
    matches!(
        ch,
        '、' | '。'
            | '，'
            | '．'
            | '！'
            | '？'
            | '：'
            | '；'
            | '）'
            | ')'
            | '］'
            | ']'
            | '｝'
            | '}'
            | '」'
            | '』'
            | '〕'
            | '】'
            | '〉'
            | '》'
            | 'ー'
            | '々'
            | 'ぁ'
            | 'ぃ'
            | 'ぅ'
            | 'ぇ'
            | 'ぉ'
            | 'っ'
            | 'ゃ'
            | 'ゅ'
            | 'ょ'
            | 'ゎ'
            | 'ァ'
            | 'ィ'
            | 'ゥ'
            | 'ェ'
            | 'ォ'
            | 'ッ'
            | 'ャ'
            | 'ュ'
            | 'ョ'
            | 'ヮ'
            | 'ヽ'
            | 'ヾ'
            | 'ゝ'
            | 'ゞ'
    )
}

fn inline_cell_width(cell: &InlineCell) -> f32 {
    match cell {
        InlineCell::Char(c) => inline_char_width(*c, DEF_TEXT_SIZE),
        InlineCell::Ruby { width, .. } | InlineCell::Tag { width, .. } => *width,
    }
}

/// Mobile `LineResult.glyphSizePx()`: the source-pixel size the overlay
/// measures a line's glyphs against.
fn line_glyph_px(line: &LineResult, quad: Option<&RotatedBox>) -> f32 {
    if let Some(q) = quad.filter(|q| q.is_rotated()) {
        return if q.is_vertical() { q.w } else { q.h }.max(1.0);
    }
    line.char_boxes
        .iter()
        .map(|b| b.h as f32)
        .fold(0.0, f32::max)
}

/// Content scale for fitting the image into the window at `zoom`.
fn content_scale(window_w: f32, window_h: f32, img_w: u32, img_h: u32, zoom: f32) -> Option<f32> {
    if img_w <= 1 || img_h <= 1 {
        return None;
    }
    Some(f32::min(window_w / img_w as f32, window_h / img_h as f32) * zoom)
}

/// The font pixel size a line's glyphs are drawn at in image pixels.
fn line_text_px(line: &LineResult, quad: Option<&RotatedBox>) -> f32 {
    line_glyph_px(line, quad) * GLYPH_FILL_RATIO / DEFAULT_INK_RATIO
}

/// Mobile `LineOverlayView` per-glyph shrink-to-box.
fn glyph_fit_scale(is_vertical: bool, is_half: bool, cell: (f32, f32), ink: (f32, f32)) -> f32 {
    let limit = if is_vertical { cell.1 } else { cell.0 } * GLYPH_FIT_RATIO;
    let limit = if is_half { limit * 2.0 } else { limit };
    let measured = if is_vertical { ink.1 } else { ink.0 };
    if measured > limit {
        limit / measured
    } else {
        1.0
    }
}

/// Mobile chip rule: vertical presentation forms only in landscape.
fn chip_text(text: &str, is_landscape: bool) -> String {
    if is_landscape {
        text.chars().map(to_vertical_glyph).collect()
    } else {
        text.to_string()
    }
}

/// Mobile `BlankGaps.apply`: materialise every measured gap as a placeholder.
fn apply_blank_gaps(line: &LineResult) -> LineResult {
    jpdict_core::blank_gaps::apply_blank_gaps(line)
}

fn font(size: f32) -> TextStyle {
    TextStyle::new(size, 400.0)
}

fn bold(size: f32) -> TextStyle {
    TextStyle::new(size, 700.0)
}

/// Rune's `Color` is linear-light; the viewer's palette is authored in
/// sRGB (the mobile values), so decode each channel.
fn srgb_decode(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
    Color {
        r: srgb_decode(r),
        g: srgb_decode(g),
        b: srgb_decode(b),
        a,
    }
}

// ---------------------------------------------------------------------------
// OcrViewer
// ---------------------------------------------------------------------------

pub struct OcrViewer {
    /// Live screenshot pixels; `.pixels()` draws them into the world.
    image_pixels: Option<Pixels>,
    /// Decoded image, cached for character preview crops.
    decoded_image: RefCell<Option<image::DynamicImage>>,
    pub img_w: u32,
    pub img_h: u32,
    /// Window dimensions in points.
    pub window_width: f32,
    pub window_height: f32,
    pub annotations: Rc<Vec<DetectedAnnotation>>,
    pub annotations_sync_dirty: StdCell<bool>,
    /// Lines whose text has been user-edited (via alternative selection).
    pub edited_lines: HashSet<usize>,
    pub state: OcrOverlayState,
    pub selected_word: Option<SelectedWord>,
    pub alternatives_visible: bool,
    pub db: Option<Arc<DictionaryDatabase>>,
    pub deinflector: Option<Arc<Deinflector>>,
    pub defer_lookup: bool,
    pub is_zooming: bool,
    pub zoom_idle_frames: u32,
    /// Cached character preview image for the alternatives panel.
    cached_preview: RefCell<Option<(usize, usize, Pixels)>>,
    pub det_thresh: f32,
    pub det_unclip: f32,
    pub det_thresh_default: f32,
    pub det_unclip_default: f32,
    pub det_box_count: usize,
    pub det_busy: bool,
    pub det_hud_visible: bool,
    /// #43/#86: whether the dictionary panel draws the pitch-accent line.
    pub show_pitch: bool,
}

impl OcrViewer {
    pub fn new_empty(window_w: f32, window_h: f32, _face: FontFace) -> Self {
        Self {
            image_pixels: None,
            decoded_image: RefCell::new(None),
            img_w: 1,
            img_h: 1,
            window_width: window_w,
            window_height: window_h,
            annotations: Rc::new(Vec::new()),
            annotations_sync_dirty: StdCell::new(false),
            edited_lines: HashSet::new(),
            state: OcrOverlayState::new(window_w, window_h),
            selected_word: None,
            alternatives_visible: false,
            db: None,
            deinflector: None,
            defer_lookup: false,
            is_zooming: false,
            zoom_idle_frames: 0,
            cached_preview: RefCell::new(None),
            det_thresh: jpdict_core::ocr_engine::default_det_thresh(),
            det_unclip: jpdict_core::ocr_engine::default_det_unclip(),
            det_thresh_default: jpdict_core::ocr_engine::default_det_thresh(),
            det_unclip_default: jpdict_core::ocr_engine::default_det_unclip(),
            det_box_count: 0,
            det_busy: false,
            det_hud_visible: false,
            show_pitch: false,
        }
    }

    /// Set the screenshot after the bootstrap thread decoded it.
    pub fn set_image(&mut self, rgba_pixels: Vec<u8>, w: u32, h: u32) {
        let pixels = Pixels::new();
        pixels.set(w, h, rgba_pixels.clone());
        self.image_pixels = Some(pixels);
        self.img_w = w;
        self.img_h = h;
        self.state.img_w = w;
        self.state.img_h = h;
        *self.decoded_image.borrow_mut() =
            image::RgbaImage::from_raw(w, h, rgba_pixels).map(image::DynamicImage::ImageRgba8);
    }

    /// Clear the current image state before a dropped image is processed.
    /// The window keeps its size; the new image is fitted into it.
    pub fn prepare_for_reload(&mut self) {
        self.image_pixels = None;
        *self.decoded_image.borrow_mut() = None;
        self.cached_preview.borrow_mut().take();
        self.annotations = Rc::new(Vec::new());
        self.annotations_sync_dirty.set(true);
        self.edited_lines.clear();
        self.selected_word = None;
        self.alternatives_visible = false;
        self.state.active_line_results.clear();
        self.state.cached_entries = Rc::new(Vec::new());
        self.state.cached_lookup_term.clear();
        self.state.current_word_length = 0;
        self.state.is_dictionary_visible = false;
        self.state.current_tapped_idx = -1;
        self.state.current_tapped_line_idx = -1;
        self.state.current_tapped_char_idx_in_line = -1;
        self.state.last_highlighted_coords.clear();
        self.state.current_scale = 1.0;
        self.state.current_trans_x = 0.0;
        self.state.current_trans_y = 0.0;
    }

    /// Crop the screenshot to show the given character with padding.
    pub fn crop_character_image(&self, line_idx: usize, char_idx: usize) -> Option<Pixels> {
        let line = self
            .state
            .active_line_results
            .get(line_idx)
            .and_then(|l| l.as_ref())?;
        let box_item = line.char_boxes.get(char_idx)?;
        let img = self.decoded_image.borrow();
        let img = img.as_ref()?;
        let (img_w, img_h) = (img.width() as i32, img.height() as i32);

        let pad = (box_item.h as f32 * 0.2) as i32;
        let crop_left = (box_item.left() - pad).max(0);
        let crop_top = (box_item.top() - pad).max(0);
        let crop_right = (box_item.right() + pad).min(img_w);
        let crop_bottom = (box_item.bottom() + pad).min(img_h);
        let crop_w = (crop_right - crop_left).max(1) as u32;
        let crop_h = (crop_bottom - crop_top).max(1) as u32;

        let cropped = img
            .crop_imm(crop_left as u32, crop_top as u32, crop_w, crop_h)
            .to_rgba8();
        let pixels = Pixels::new();
        pixels.set(crop_w, crop_h, cropped.into_raw());
        Some(pixels)
    }

    pub fn select_character(&mut self, line_idx: usize, char_idx: usize) {
        let was_visible = self.state.is_dictionary_visible;
        let is_same = self.state.current_tapped_line_idx == line_idx as isize
            && self.state.current_tapped_char_idx_in_line == char_idx as isize;
        self.set_cursor_pos(line_idx, char_idx);
        self.state.is_dictionary_visible = true;
        if is_same && was_visible {
            self.alternatives_visible = !self.alternatives_visible;
        } else {
            self.alternatives_visible = false;
        }

        let box_item = self
            .state
            .active_line_results
            .get(line_idx)
            .and_then(|l| l.as_ref())
            .and_then(|line| line.char_boxes.get(char_idx))
            .cloned();
        if let Some(b) = box_item {
            self.state.update_gravity(&b, self.panel_width());
        }
        self.do_lookup(line_idx, char_idx);
    }

    /// Move the cursor without opening the dictionary (keyboard navigation).
    pub fn move_cursor_to(&mut self, line_idx: usize, char_idx: usize) {
        self.set_cursor_pos(line_idx, char_idx);
        self.state.update_highlight_coords(line_idx, char_idx, 1);

        if self.state.current_scale <= 1.01 {
            if let Some(box_item) = self
                .state
                .active_line_results
                .get(line_idx)
                .and_then(|l| l.as_ref())
                .and_then(|line| line.char_boxes.get(char_idx))
                .cloned()
            {
                self.state.update_gravity(&box_item, self.panel_width());
            }
        }

        if self.state.current_scale > 1.01 {
            if let Some(ann) = self
                .state
                .active_line_results
                .get(line_idx)
                .and_then(|l| l.as_ref())
            {
                if let Some(char_box) = ann.char_boxes.get(char_idx) {
                    let img_w_f = self.state.img_w as f32;
                    let img_h_f = self.state.img_h as f32;
                    let base_scale =
                        f32::min(self.window_width / img_w_f, self.window_height / img_h_f);
                    let total_scale = base_scale * self.state.current_scale;
                    let base_offset_x = (self.window_width - img_w_f * base_scale) / 2.0;
                    let base_offset_y = (self.window_height - img_h_f * base_scale) / 2.0;

                    let sx = char_box.x as f32 * total_scale
                        + base_offset_x
                        + self.state.current_trans_x;
                    let sy = char_box.y as f32 * total_scale
                        + base_offset_y
                        + self.state.current_trans_y;
                    let sw = (char_box.w as f32).max(1.0) * total_scale;
                    let sh = (char_box.h as f32).max(1.0) * total_scale;

                    let margin = 30.0;
                    let panel_margin = 10.0;
                    let (vis_left, vis_right) = if self.state.is_dictionary_visible {
                        let pw = self.panel_width();
                        if self.state.last_landscape_gravity == Gravity::End {
                            (margin, self.window_width - pw - panel_margin)
                        } else {
                            (pw + panel_margin, self.window_width - margin)
                        }
                    } else {
                        (margin, self.window_width - margin)
                    };

                    if sx + sw > vis_right {
                        let overshoot = (sx + sw) - vis_right;
                        self.state.current_trans_x -= overshoot;
                    }
                    if sx < vis_left {
                        self.state.current_trans_x += vis_left - sx;
                    }
                    if sy + sh > self.window_height - margin {
                        let overshoot = (sy + sh) - (self.window_height - margin);
                        self.state.current_trans_y -= overshoot;
                    }
                    if sy < margin {
                        self.state.current_trans_y += margin - sy;
                    }
                }
            }
        }
    }

    fn set_cursor_pos(&mut self, line_idx: usize, char_idx: usize) {
        self.state.current_tapped_line_idx = line_idx as isize;
        self.state.current_tapped_char_idx_in_line = char_idx as isize;
        self.state.current_tapped_idx = self.state.get_global_idx(line_idx, char_idx) as isize;
    }

    pub fn select_neighbor(&mut self, line_idx: usize, char_idx: usize) {
        let is_same = self.state.current_tapped_line_idx == line_idx as isize
            && self.state.current_tapped_char_idx_in_line == char_idx as isize;
        self.state.current_tapped_line_idx = line_idx as isize;
        self.state.current_tapped_char_idx_in_line = char_idx as isize;
        self.state.current_tapped_idx = self.state.get_global_idx(line_idx, char_idx) as isize;
        self.state.update_highlight_coords(line_idx, char_idx, 1);
        self.state.is_dictionary_visible = true;
        if is_same {
            self.alternatives_visible = !self.alternatives_visible;
        }
        self.do_lookup(line_idx, char_idx);
    }

    pub(crate) fn do_lookup(&mut self, line_idx: usize, char_idx: usize) {
        let full_line_text = {
            let Some(line) = self
                .state
                .active_line_results
                .get(line_idx)
                .and_then(|l| l.as_ref())
            else {
                return;
            };
            self.selected_word = Some(SelectedWord { line_idx, char_idx });
            line.text.clone()
        };
        if full_line_text.chars().nth(char_idx) == Some(GAP_CHAR) {
            self.state.cached_entries = Rc::new(Vec::new());
            self.state.current_word_length = 1;
            self.state.update_highlight_coords(line_idx, char_idx, 1);
            return;
        }
        if let (Some(db), Some(deinf)) = (self.db.as_ref(), self.deinflector.as_ref()) {
            if let Some(result) = self.state.lookup(line_idx, char_idx, db, deinf) {
                self.state.cached_entries = result.matches;
                self.state.current_word_length = result.max_len;
                self.state
                    .update_highlight_coords(line_idx, char_idx, result.max_len);

                // Second pass: each individual kanji, with a per-session cache.
                let term_len = result.max_len;
                let matched_term = self
                    .state
                    .matched_term_at(self.state.get_global_idx(line_idx, char_idx), term_len);
                let mut append_kanji: Vec<FormattedEntry> = Vec::new();
                for ch in matched_term.chars() {
                    if !('\u{4E00}'..='\u{9FFF}').contains(&ch)
                        && !('\u{3400}'..='\u{4DBF}').contains(&ch)
                    {
                        continue;
                    }
                    let kanji_str = ch.to_string();
                    if self
                        .state
                        .cached_entries
                        .iter()
                        .any(|e| e.term == kanji_str)
                        || append_kanji.iter().any(|e| e.term == kanji_str)
                    {
                        continue;
                    }
                    let kanji_only: Vec<DictionaryEntry> =
                        if let Some(cached) = self.state.kanji_cache.get(&kanji_str) {
                            cached.clone()
                        } else if let Ok(kanji_results) = db.find_by_texts(&[kanji_str.clone()]) {
                            let filtered: Vec<DictionaryEntry> = kanji_results
                                .into_iter()
                                .filter(|e| e.onyomi.is_some() || e.kunyomi.is_some())
                                .collect();
                            self.state
                                .kanji_cache
                                .insert(kanji_str.clone(), filtered.clone());
                            filtered
                        } else {
                            Vec::new()
                        };
                    if !kanji_only.is_empty() {
                        let dict_names = db.dictionary_names().unwrap_or_default();
                        let formatted = self.state.format_dictionary_results(
                            &[TermMatch {
                                term: kanji_str.clone(),
                                entries: kanji_only,
                                chain: None,
                            }],
                            &dict_names,
                        );
                        append_kanji.extend(formatted);
                    }
                }
                if !append_kanji.is_empty() {
                    let mut entries = (*self.state.cached_entries).clone();
                    entries.extend(append_kanji);
                    self.state.cached_entries = Rc::new(entries);
                }
            } else {
                self.state.cached_entries = Rc::new(Vec::new());
                self.state.current_word_length = 1;
                self.state.update_highlight_coords(line_idx, char_idx, 1);
            }
        }
    }

    /// Select one of the alternatives for the current character.
    pub fn select_alternative(&mut self, c: char) {
        let Some(sel) = self.selected_word.as_ref() else {
            return;
        };
        let (li, ci) = (sel.line_idx, sel.char_idx);
        let cur = self
            .state
            .active_line_results
            .get(li)
            .and_then(|l| l.as_ref())
            .and_then(|l| l.text.chars().nth(ci));
        if cur == Some(c) {
            self.alternatives_visible = false;
            return;
        }
        self.state.update_character(li, ci, c);
        self.annotations_sync_dirty.set(true);
        self.edited_lines.insert(li);
        // Dataset collection: rewrite the crop's .txt sidecar with the
        // corrected text so /tmp yields curated (crop, label) pairs.
        if let Some(line) = self
            .state
            .active_line_results
            .get(li)
            .and_then(|l| l.as_ref())
        {
            if let Some(txt) = line.sample_txt.as_ref() {
                if let Err(e) = std::fs::write(txt, &line.text) {
                    eprintln!("[dataset] failed to update {}: {e}", txt.display());
                } else {
                    eprintln!("[dataset] updated {}", txt.display());
                }
            }
        }
        let _ = self.db.as_ref().and_then(|db| {
            self.deinflector
                .as_ref()
                .and_then(|deinf| self.state.lookup(li, ci, db, deinf))
        });
    }

    /// Navigate the alternatives selection when the alt panel is visible.
    pub fn navigate_alternatives(&mut self, dir: GamepadAction) {
        let is_landscape = self.window_width > self.window_height;
        let (line_idx, char_idx) = match self.state.current_cursor() {
            Some(c) => c,
            None => return,
        };
        let line = match self
            .state
            .active_line_results
            .get(line_idx)
            .and_then(|l| l.as_ref())
        {
            Some(l) => l,
            None => return,
        };
        let alts = match line.alternatives.get(char_idx) {
            Some(a) => a,
            None => return,
        };
        let current_char = match line.text.chars().nth(char_idx) {
            Some(c) => c,
            None => return,
        };
        let diff = match (dir, is_landscape) {
            (GamepadAction::NavigateDown, true) | (GamepadAction::NavigateRight, false) => 1,
            (GamepadAction::NavigateUp, true) | (GamepadAction::NavigateLeft, false) => -1,
            _ => 0,
        };
        if diff == 0 {
            return;
        }
        let candidates: Vec<char> = alts.iter().take(15).map(|(c, _)| *c).collect();
        let current_idx = candidates
            .iter()
            .position(|c| *c == current_char)
            .unwrap_or(0);
        let new_idx =
            (current_idx as isize + diff).clamp(0, candidates.len() as isize - 1) as usize;
        if new_idx != current_idx {
            let new_char = candidates[new_idx];
            self.state.update_character(line_idx, char_idx, new_char);
            let _ = self.db.as_ref().and_then(|db| {
                self.deinflector
                    .as_ref()
                    .and_then(|deinf| self.state.lookup(line_idx, char_idx, db, deinf))
            });
            self.state
                .update_highlight_coords(line_idx, char_idx, self.state.current_word_length);
        }
    }

    fn is_landscape(&self) -> bool {
        self.window_width > self.window_height
    }

    /// The iced build made the logical viewport 1280 pt wide and scaled the
    /// whole UI by `image_width / 1280`. Keep that convention: at 1280 the
    /// panel is 300 pt, at 1920 it is 450 pt, so it keeps the same share of
    /// the screen instead of shrinking on larger displays.
    fn ui_scale(&self) -> f32 {
        (self.window_width / 1280.0).clamp(0.5, 4.0)
    }

    fn logical_width(&self) -> f32 {
        self.window_width / self.ui_scale()
    }

    fn logical_height(&self) -> f32 {
        self.window_height / self.ui_scale()
    }

    fn panel_logical_width(&self) -> f32 {
        DICT_WIDTH + NEIGHBOR_WIDTH + 6.0
    }

    fn panel_width(&self) -> f32 {
        (DICT_WIDTH + NEIGHBOR_WIDTH * 2.0 + 6.0) * self.ui_scale()
    }

    pub fn apply_ocr_batch(&mut self, annotations: Vec<DetectedAnnotation>) {
        let mut lines: Vec<(usize, LineResult)> = Vec::new();
        let mut anns: Vec<DetectedAnnotation> = Vec::with_capacity(annotations.len());
        for (index, mut annotation) in annotations.into_iter().enumerate() {
            if let Some(line) = annotation.line.take() {
                annotation.line = Some(apply_blank_gaps(&line));
            }
            let line = annotation.line.clone();
            if let Some(ref line) = line {
                println!(
                    "[VIEWER] recv ann idx={}: is_vertical={}, has_text={}, char_boxes={}",
                    index,
                    line.is_vertical,
                    !line.text.is_empty(),
                    line.char_boxes.len()
                );
                if !self.edited_lines.contains(&index) {
                    lines.push((index, line.clone()));
                }
            }
            anns.push(annotation);
        }
        self.annotations = Rc::new(anns);
        self.det_box_count = self.annotations.len();
        if !lines.is_empty() {
            self.state.set_line_results_batch(lines);
        }
        self.state.mark_nav_dirty();
        if self.state.current_tapped_line_idx < 0 || self.state.current_tapped_char_idx_in_line < 0
        {
            self.state.ensure_cursor_position();
        }
    }

    pub fn adjust_det_tuning(&mut self, param: DetParam, delta: f32) {
        match param {
            DetParam::Threshold => self.det_thresh = (self.det_thresh + delta).clamp(0.01, 0.99),
            DetParam::Unclip => self.det_unclip = (self.det_unclip + delta).clamp(0.0, 5.0),
        }
    }

    pub fn reset_det_tuning(&mut self) {
        self.det_thresh = self.det_thresh_default;
        self.det_unclip = self.det_unclip_default;
    }

    pub fn toggle_det_hud(&mut self) {
        self.det_hud_visible = !self.det_hud_visible;
    }

    pub fn apply_retune(&mut self, annotations: Vec<DetectedAnnotation>) {
        self.edited_lines.clear();
        self.selected_word = None;
        self.alternatives_visible = false;
        self.state.active_line_results.clear();
        self.state.cached_entries = Rc::new(Vec::new());
        self.state.cached_lookup_term.clear();
        self.state.current_word_length = 0;
        self.state.is_dictionary_visible = false;
        self.state.current_tapped_idx = -1;
        self.state.current_tapped_line_idx = -1;
        self.state.current_tapped_char_idx_in_line = -1;
        self.state.last_highlighted_coords.clear();
        self.apply_ocr_batch(annotations);
        self.annotations_sync_dirty.set(true);
    }

    fn det_hud_text(&self) -> String {
        format!(
            "DET {:.2}  UNCLIP {:.2}  boxes {}{}   [/] unclip  -/= thresh  R reset  F1 hide",
            self.det_thresh,
            self.det_unclip,
            self.det_box_count,
            if self.det_busy {
                "  (re-running…)"
            } else {
                ""
            },
        )
    }

    // ── view ────────────────────────────────────────────────────────────

    pub fn view(&self, s: &ViewerApp) -> El {
        let has_panel = self.selected_word.is_some();
        let mut root = el()
            .key("root")
            .w_full()
            .h_full()
            .fill(BLACK)
            .clip_rounded(0.0)
            .transition(Transition::none());

        root = root.child(self.world(s));

        if has_panel {
            // The alternatives strip is a sibling *before* the panel, so its
            // exit fade draws under the neighbour/dictionary columns instead
            // of over them. Keeping it out of the panel row also means the
            // columns do not shift when it opens or closes.
            if self.alternatives_visible {
                root = root.child(self.alternatives_overlay(s));
            }
            root = root.child(self.panel(s));
        }

        if self.det_hud_visible {
            let k = self.ui_scale();
            root = root.child(
                el().key("det_hud")
                    .absolute(8.0 * k, 8.0 * k)
                    .padding(6.0 * k)
                    .radius(4.0 * k)
                    .fill(rgba(0.0, 0.0, 0.0, 0.65))
                    .child(
                        text(self.det_hud_text())
                            .font(font(13.0 * k))
                            .color(rgba(0.95, 0.95, 0.95, 1.0)),
                    ),
            );
        }
        root
    }

    /// The camera: a world sized to the window, with the image, boxes and
    /// glyphs laid out at the current zoom. Scaling is applied to the layout
    /// metrics rather than to the whole subtree, so text is rasterized at its
    /// final size instead of being stretched from a cached layer mid-zoom.
    ///
    /// The pan offset lives on an inner, unclipped `camera` node: Rune's clip
    /// is the node's drawn rect, so offsetting the clipped node itself moved
    /// the scissor with the content, and zooming in left the right and bottom
    /// of the window unpainted (the image looked cut off mid-pan).
    fn world(&self, _s: &ViewerApp) -> El {
        let (iw, ih) = (self.img_w.max(1) as f32, self.img_h.max(1) as f32);
        let base_scale = content_scale(
            self.window_width,
            self.window_height,
            self.img_w,
            self.img_h,
            1.0,
        )
        .unwrap_or(1.0);
        let total_scale = base_scale * self.state.current_scale;
        let origin_x = (self.window_width - iw * base_scale) / 2.0;
        let origin_y = (self.window_height - ih * base_scale) / 2.0;

        let mut cam = el()
            .key("camera")
            .absolute(0.0, 0.0)
            .size(self.window_width, self.window_height)
            .offset(self.state.current_trans_x, self.state.current_trans_y)
            .motion(Spring::NONE)
            .transition(Transition::none());

        if let Some(p) = self.image_pixels.as_ref() {
            cam = cam.child(
                el().key("image")
                    .absolute(origin_x, origin_y)
                    .size(iw * total_scale, ih * total_scale)
                    .pixels(p),
            );
        }

        let mut glyphs = Vec::new();
        let mut cursor_glyph = None;
        for (line_idx, annotation) in self.annotations.iter().enumerate() {
            let bbox = &annotation.bbox;
            let bx1 = origin_x + bbox.x as f32 * total_scale;
            let by1 = origin_y + bbox.y as f32 * total_scale;

            if let Some(q) = annotation.quad {
                // Angled line: draw the rotated quad instead of its AABB.
                cam = cam.child(
                    el().key(format!("qb{line_idx}"))
                        .absolute(
                            origin_x + (q.cx - q.w / 2.0) * total_scale,
                            origin_y + (q.cy - q.h / 2.0) * total_scale,
                        )
                        .size(q.w * total_scale, q.h * total_scale)
                        .radius(BOX_CORNER_RADIUS * total_scale)
                        .rotate(q.angle)
                        .fill(rgba(0.0, 0.0, 0.0, BOX_FILL_ALPHA))
                        .inert(),
                );
            } else {
                cam = cam.child(
                    el().key(format!("bx{line_idx}"))
                        .absolute(bx1, by1)
                        .size(
                            bbox.w.max(1) as f32 * total_scale,
                            bbox.h.max(1) as f32 * total_scale,
                        )
                        .radius(BOX_CORNER_RADIUS * total_scale)
                        .fill(rgba(0.0, 0.0, 0.0, BOX_FILL_ALPHA))
                        .inert(),
                );
            }

            let Some(line) = &annotation.line else {
                continue;
            };
            if line.text.is_empty() || line.char_boxes.is_empty() {
                continue;
            }
            let px = line_text_px(line, annotation.quad.as_ref()).max(1.0);
            let tilt = annotation
                .quad
                .filter(|q| q.is_rotated())
                .map(|q| q.angle)
                .unwrap_or(0.0);
            let cursor = self.state.current_cursor();
            for (i, char_box) in line.char_boxes.iter().enumerate() {
                let Some(ch) = line.text.chars().nth(i) else {
                    continue;
                };
                let is_half = is_half_width(ch);
                let fit = glyph_fit_scale(
                    line.is_vertical,
                    is_half,
                    (char_box.w.max(1) as f32, char_box.h.max(1) as f32),
                    (char_box.w as f32, char_box.h as f32),
                );
                let em = (px
                    * fit
                    * if is_half { ASCII_GLYPH_SCALE } else { 1.0 }
                    * total_scale)
                    .clamp(1.0, 4096.0);
                let box_w = char_box.w.max(1) as f32 * total_scale;
                let box_h = char_box.h.max(1) as f32 * total_scale;
                let highlighted = self.state.last_highlighted_coords.contains(&(line_idx, i));
                let color = if highlighted { YELLOW } else { PINK };
                let is_cursor = cursor == Some((line_idx, i));
                let key = format!("g{line_idx}_{i}");
                let li = line_idx;
                let ci = i;
                let mut g = el()
                    .key(key)
                    .absolute(
                        origin_x + char_box.x as f32 * total_scale,
                        origin_y + char_box.y as f32 * total_scale,
                    )
                    .size(box_w, box_h)
                    .center()
                    .rotate(tilt)
                    .on_drag(pan_drag)
                    .on_press_tx(move |s: &mut ViewerApp, tx: &mut Tx| {
                        tx.instant();
                        s.viewer.select_character(li, ci);
                        tx.scroll_to_instant("dict_scroll", 0.0, 0.0);
                        tx.scroll_into_view(format!("nb:{li}:{ci}"));
                    });
                if is_cursor {
                    g = g
                        .border(CURSOR_STROKE * total_scale, WHITE)
                        .radius(CURSOR_RADIUS * total_scale);
                }
                g = g.child(
                    text(ch.to_string())
                        .font(TextStyle::new(em, 400.0).lh(box_h))
                        .color(color)
                        .align(rune::Align::Center)
                        .a11y_hidden(),
                );
                if is_cursor {
                    // Draw the cursor above the other glyphs, but stay inside the
                    // world layer so the dictionary panel still paints on top.
                    cursor_glyph = Some(g);
                } else {
                    glyphs.push(g);
                }
            }
        }
        for g in glyphs {
            cam = cam.child(g);
        }
        if let Some(g) = cursor_glyph {
            cam = cam.child(g);
        }
        el().key("world")
            .size(self.window_width, self.window_height)
            .clip_rounded(0.0)
            .transition(Transition::none())
            .on_drag(pan_drag)
            .child(cam)
    }
}

// ---------------------------------------------------------------------------
// ViewerApp — the Rune application state around OcrViewer
// ---------------------------------------------------------------------------

pub struct ViewerApp {
    pub viewer: OcrViewer,
    rx: Receiver<BootstrapMsg>,
    worker_tx: Mutex<Sender<WorkerCmd>>,
    pan_origin: [f32; 2],
    dict_scroll: Rc<StdCell<[f32; 2]>>,
    held: Option<(GamepadAction, Instant)>,
    repeat_count: u64,
}

impl ViewerApp {
    pub fn new(viewer: OcrViewer, rx: Receiver<BootstrapMsg>, worker_tx: Sender<WorkerCmd>) -> Self {
        Self {
            viewer,
            rx,
            worker_tx: Mutex::new(worker_tx),
            pan_origin: [0.0, 0.0],
            dict_scroll: Rc::new(StdCell::new([0.0, 0.0])),
            held: None,
            repeat_count: 0,
        }
    }

    /// Drop a new image into this window: clear the current result and ask
    /// the OCR worker to process the image at the existing window size.
    pub fn reload_image(&mut self, path: PathBuf) {
        self.viewer.prepare_for_reload();
        if let Ok(tx) = self.worker_tx.lock() {
            let _ = tx.send(WorkerCmd::Reload(path));
        }
    }

    fn resize(&mut self, w: f32, h: f32) {
        if (self.viewer.window_width - w).abs() > 0.5 || (self.viewer.window_height - h).abs() > 0.5
        {
            self.viewer.window_width = w;
            self.viewer.window_height = h;
            self.viewer.state.window_width.set(w);
            self.viewer.state.window_height.set(h);
        }
    }

    fn handle_bootstrap(&mut self) -> bool {
        let mut changed = false;
        while let Ok(event) = self.rx.try_recv() {
            match event {
                BootstrapMsg::ImageReady(rgba, w, h) => self.viewer.set_image(rgba, w, h),
                BootstrapMsg::DictReady(db) => self.viewer.db = Some(db),
                BootstrapMsg::DeinflectReady(d) => self.viewer.deinflector = Some(d),
                BootstrapMsg::Results(annotations) => self.viewer.apply_ocr_batch(annotations),
                BootstrapMsg::Retuned(annotations) => {
                    self.viewer.apply_retune(annotations);
                    self.viewer.det_busy = false;
                }
            }
            changed = true;
        }
        changed
    }

    fn back(&mut self, tx: &mut Tx) {
        tx.instant();
        if self.viewer.alternatives_visible {
            self.viewer.alternatives_visible = false;
        } else if self.viewer.selected_word.is_some() {
            self.viewer.selected_word = None;
            self.viewer.state.is_dictionary_visible = false;
        } else {
            tx.quit();
        }
        self.held = None;
        self.repeat_count = 0;
    }

    fn confirm(&mut self, tx: &mut Tx) {
        if self.viewer.alternatives_visible {
            self.viewer.alternatives_visible = false;
        } else if self.viewer.selected_word.is_some() {
            self.viewer.alternatives_visible = true;
            if let Some((li, ci)) = self.viewer.state.current_cursor() {
                tx.scroll_into_view(format!("alt:{li}:{ci}"));
            }
        } else if let Some((li, ci)) = self.viewer.state.current_cursor() {
            self.viewer.select_character(li, ci);
            tx.scroll_to_instant("dict_scroll", 0.0, 0.0);
            tx.scroll_into_view(format!("nb:{li}:{ci}"));
        }
    }

    fn navigate(&mut self, action: GamepadAction, tx: &mut Tx) {
        if self.viewer.alternatives_visible {
            self.viewer.navigate_alternatives(action);
            if let Some((li, ci)) = self.viewer.state.current_cursor() {
                tx.scroll_into_view(format!("alt:{li}:{ci}"));
            }
            return;
        }
        self.viewer.defer_lookup = false;
        self.viewer.state.navigate(action);
        if let Some((li, ci)) = self.viewer.state.current_cursor() {
            self.viewer.move_cursor_to(li, ci);
            if self.viewer.state.is_dictionary_visible {
                self.viewer.do_lookup(li, ci);
                tx.scroll_to_instant("dict_scroll", 0.0, 0.0);
            }
            tx.scroll_into_view(format!("nb:{li}:{ci}"));
        }
        self.held = Some((action, Instant::now()));
        self.repeat_count = 0;
    }

    /// A navigation key came back up: stop hold-to-repeat. Before this, the
    /// timer kept repeating after a single tap.
    fn key_released(&mut self, e: &KeyEvent) -> bool {
        let nav = match &e.key {
            Key::Up | Key::Down | Key::Left | Key::Right => true,
            Key::Text(t) => matches!(
                t.chars().next(),
                Some('w' | 'a' | 's' | 'd' | 'h' | 'j' | 'k' | 'l' | 'u' | 'e' | 'f' | 'm' | 'c')
            ),
            _ => false,
        };
        if nav {
            self.held = None;
            self.repeat_count = 0;
            true
        } else {
            false
        }
    }

    fn scroll_dict(&mut self, tx: &mut Tx, delta: f32) {
        let at = self.dict_scroll.get();
        tx.scroll_to_y("dict_scroll", (at[1] + delta).max(0.0));
    }

    fn zoom_at(&mut self, delta: f32, x: f32, y: f32) {
        let factor = if delta > 0.0 { 1.1 } else { 0.9 };
        let old = self.viewer.state.current_scale;
        let new = (old * factor).clamp(0.5, 5.0);
        let actual = if old > 0.0 { new / old } else { 1.0 };
        // The image is laid out from its fit-to-window origin and then scaled,
        // so express the cursor relative to that origin, not the window centre.
        let base_scale = content_scale(
            self.viewer.window_width,
            self.viewer.window_height,
            self.viewer.img_w,
            self.viewer.img_h,
            1.0,
        )
        .unwrap_or(1.0);
        let origin_x =
            (self.viewer.window_width - self.viewer.img_w.max(1) as f32 * base_scale) / 2.0;
        let origin_y =
            (self.viewer.window_height - self.viewer.img_h.max(1) as f32 * base_scale) / 2.0;
        let dx = x - origin_x;
        let dy = y - origin_y;
        self.viewer.state.current_scale = new;
        self.viewer.state.current_trans_x = dx - (dx - self.viewer.state.current_trans_x) * actual;
        self.viewer.state.current_trans_y = dy - (dy - self.viewer.state.current_trans_y) * actual;
        self.viewer.is_zooming = true;
        self.viewer.zoom_idle_frames = 0;
    }

    fn tune(&mut self, param: DetParam, delta: f32) {
        self.viewer.adjust_det_tuning(param, delta);
        self.viewer.det_busy = true;
        if let Ok(tx) = self.worker_tx.lock() {
            let _ = tx.send(WorkerCmd::Tune(TuneCmd {
                thresh: self.viewer.det_thresh,
                unclip: self.viewer.det_unclip,
            }));
        }
    }

    fn tune_reset(&mut self) {
        self.viewer.reset_det_tuning();
        self.viewer.det_busy = true;
        if let Ok(tx) = self.worker_tx.lock() {
            let _ = tx.send(WorkerCmd::Tune(TuneCmd {
                thresh: self.viewer.det_thresh,
                unclip: self.viewer.det_unclip,
            }));
        }
    }

    fn key(&mut self, e: &KeyEvent, tx: &mut Tx) -> bool {
        // Fades are disabled throughout the viewer; land every key-driven
        // change (panel open/close, lookup rows, HUD) in the same frame.
        tx.instant();
        let key = &e.key;
        match key {
            Key::Escape | Key::Backspace => {
                self.back(tx);
                return true;
            }
            Key::Enter => {
                self.confirm(tx);
                return true;
            }
            Key::Text(t) => {
                let c = t.chars().next().unwrap_or('\0');
                match c {
                    'q' => {
                        self.back(tx);
                        return true;
                    }
                    ' ' => {
                        self.confirm(tx);
                        return true;
                    }
                    'l' | 'd' | 'h' | 'a' | 'j' | 's' | 'k' | 'w' => {
                        let action = match c {
                            'l' | 'd' => GamepadAction::NavigateRight,
                            'h' | 'a' => GamepadAction::NavigateLeft,
                            'j' | 's' => GamepadAction::NavigateDown,
                            _ => GamepadAction::NavigateUp,
                        };
                        // The OS sends its own key repeats. Once one press armed
                        // our timer, ignore those so the two don't double-step.
                        if matches!(self.held, Some((held, _)) if held == action) {
                            return true;
                        }
                        self.navigate(action, tx);
                        return true;
                    }
                    'u' | 'e' | 'f' => {
                        self.scroll_dict(tx, -120.0);
                        return true;
                    }
                    'm' | 'c' => {
                        self.scroll_dict(tx, 120.0);
                        return true;
                    }
                    '[' => {
                        self.tune(DetParam::Unclip, -0.05);
                        return true;
                    }
                    ']' => {
                        self.tune(DetParam::Unclip, 0.05);
                        return true;
                    }
                    '-' | '_' => {
                        self.tune(DetParam::Threshold, -0.01);
                        return true;
                    }
                    '=' | '+' => {
                        self.tune(DetParam::Threshold, 0.01);
                        return true;
                    }
                    'r' | 'R' => {
                        self.tune_reset();
                        return true;
                    }
                    _ => {}
                }
            }
            Key::Function(1) => {
                self.viewer.toggle_det_hud();
                return true;
            }
            Key::Up => {
                self.navigate(GamepadAction::NavigateUp, tx);
                return true;
            }
            Key::Down => {
                self.navigate(GamepadAction::NavigateDown, tx);
                return true;
            }
            Key::Left => {
                self.navigate(GamepadAction::NavigateLeft, tx);
                return true;
            }
            Key::Right => {
                self.navigate(GamepadAction::NavigateRight, tx);
                return true;
            }
            _ => {}
        }
        false
    }
}

/// A drag over the OCR panel belongs to the panel (scrolling the dictionary,
/// pressing a row), never to the image underneath: the viewer's root pans the
/// image on drag, and the panel sits above it in paint order but has no drag
/// handler of its own, so the root was winning and moving the overlay.
fn panel_drag(_s: &mut ViewerApp, _d: &mut Drag, _tx: &mut Tx) {}

fn pan_drag(s: &mut ViewerApp, d: &mut Drag, _tx: &mut Tx) {
    match d.phase {
        Phase::Down => {
            s.pan_origin = [
                s.viewer.state.current_trans_x,
                s.viewer.state.current_trans_y,
            ];
        }
        Phase::Move => {
            s.viewer.state.current_trans_x = s.pan_origin[0] + d.dx;
            s.viewer.state.current_trans_y = s.pan_origin[1] + d.dy;
            s.viewer.is_zooming = true;
            s.viewer.zoom_idle_frames = 0;
        }
        Phase::Up => {
            s.viewer.zoom_idle_frames = 0;
        }
    }
}

fn zoom_wheel(s: &mut ViewerApp, e: WheelEvent, tx: &mut Tx) -> bool {
    let dy = if e.precise { e.dy / 40.0 } else { e.dy };
    if dy == 0.0 {
        return false;
    }

    // The dictionary column is a scroll view, but the root wheel handler was
    // winning over it. Route wheel input over that column to the dictionary
    // instead of the camera.
    let scale = s.viewer.ui_scale();
    let panel_left = if s.viewer.state.last_landscape_gravity == Gravity::End {
        s.viewer.window_width - s.viewer.panel_logical_width() * scale
    } else {
        0.0
    };
    let dict_left = if s.viewer.state.last_landscape_gravity == Gravity::End {
        panel_left + NEIGHBOR_WIDTH * scale
    } else {
        panel_left
    };
    let dict_right = dict_left + DICT_WIDTH * scale;
    if (dict_left..=dict_right).contains(&e.x)
        && (0.0..=s.viewer.window_height).contains(&e.y)
    {
        // WheelEvent::dy is already in layout points (Rune turned a notched
        // wheel's line steps into 40 points each), which is exactly the unit
        // scroll_dict's offset uses — the same 1:1 delta Rune's own scroll
        // views get. Scaling it again made one notch fly past a whole page.
        // The sign is inverted because Rune's wheel moves the content by
        // `offset - dy`, while scroll_dict adds to the offset.
        if e.dy != 0.0 {
            s.scroll_dict(tx, -e.dy);
        }
        return true;
    }

    // An instant commit makes Rune rasterize the glyphs at the new scale in
    // this frame instead of stretching the previous texture while it settles.
    tx.instant();
    s.zoom_at(dy, e.x, e.y);
    true
}

fn app_view(s: &ViewerApp, _cx: &mut Cx<ViewerApp>) -> El {
    let root = s
        .viewer
        .view(s)
        .on_layout(|s: &mut ViewerApp, [_, _, w, h]| {
            s.resize(w, h);
        });
    root.accepts_extensions(&["jpg", "jpeg", "png", "jxl", "webp"])
        .on_drop_files(|s: &mut ViewerApp, files, _tx| {
            if let Some(path) = files.first() {
                s.reload_image(path.clone());
            }
        })
        .on_secondary_press(|s: &mut ViewerApp, _p, tx| s.back(tx))
        .on_drag(pan_drag)
        .on_wheel(zoom_wheel)
        .on_magnify(|s: &mut ViewerApp, e, tx| {
            tx.instant();
            s.zoom_at(e.delta, e.x, e.y);
            true
        })
        .on_key_up(|s: &mut ViewerApp, e, _tx| s.key_released(e))
}

fn app_poll(s: &mut ViewerApp, _t: f64, tx: &mut Tx) -> bool {
    let mut changed = s.handle_bootstrap();
    if s.viewer.zoom_idle_frames > 3 {
        s.viewer.is_zooming = false;
    }
    s.viewer.zoom_idle_frames = s.viewer.zoom_idle_frames.saturating_add(1);

    // Hold-to-repeat for the viewer's navigation keys: 250 ms delay, then
    // 20 repeats/s, matching the previous viewer subscription.
    if let Some((action, started)) = s.held {
        let elapsed = started.elapsed();
        if elapsed >= Duration::from_millis(250) {
            let due = ((elapsed.as_millis() - 250) / 40) as u64;
            if due > s.repeat_count {
                s.repeat_count = due;
                s.navigate(action, tx);
                // A repeat must not restart the hold timer.
                s.held = Some((action, started));
                changed = true;
            }
        } else {
            s.repeat_count = 0;
        }
    }

    // A late image/result commit mounts the whole OCR overlay after the
    // window's first frame. Its default fade is the blurry load-in; land
    // bootstrap changes in the same frame instead.
    if changed {
        tx.instant();
    }

    // Keep polling bootstrap channels and the repeat clock. The previous
    // viewer subscription ticked every 16 ms; Rune's wake schedule replaces it.
    tx.wake_after(Duration::from_millis(16));
    changed
}

/// Run the viewer window. `size` is the image's pixel size, which becomes the
/// window's point size so the screenshot maps 1:1 on a scale-1 display.
pub fn run(
    init: impl FnOnce() -> ViewerApp + 'static,
    size: (f32, f32),
    family: &'static str,
) -> anyhow::Result<()> {
    static FONTS: &[&[u8]] = &[crate::frontend::SANS_FONT, crate::frontend::SERIF_FONT];
    let spec = AppSpec::<ViewerApp> {
        title: "Instant JP Dict",
        size,
        fonts: FONTS,
        families: (family, family),
        poll: Some(app_poll),
        on_key: Some(|s: &mut ViewerApp, e, tx| s.key(e, tx)),
        ..AppSpec::new(
            "Instant JP Dict",
            size,
            || unreachable!("ViewerApp is built by App::init"),
            app_view,
        )
    };
    rune::App::new(spec)
        .init(init)
        .try_run()
        .map_err(|e| anyhow!(e))
}

// ---------------------------------------------------------------------------
// Dictionary panel
// ---------------------------------------------------------------------------

impl OcrViewer {
    /// The dictionary/neighbour columns in logical points, before UI scaling.
    fn panel_content(&self, s: &ViewerApp) -> El {
        let panel_on_right = self.state.last_landscape_gravity == Gravity::End;
        let dict = self.dictionary_panel(s);
        let neigh = self.neighbor_panel(s);

        let mut content = row()
            .padding(2.0)
            .gap(2.0)
            .w(self.panel_logical_width())
            .h_full()
            .min_h(0.0);
        if panel_on_right {
            content = content.child(neigh).child(dict);
        } else {
            content = content.child(dict).child(neigh);
        }
        content
            .key("panel")
            .background_panel()
    }

    /// The panel drawn at `ui_scale`. A wrapper owns the on-screen box; the
    /// logical panel is scaled about the screen edge so text stays crisp and
    /// every metric (type, padding, radii) follows the 1280-pt logical grid.
    fn panel(&self, s: &ViewerApp) -> El {
        let scale = self.ui_scale();
        let panel_on_right = self.state.last_landscape_gravity == Gravity::End;
        let logical_w = self.panel_logical_width();
        let logical_h = self.logical_height();
        let on_screen_w = logical_w * scale;
        let x = if panel_on_right {
            (self.window_width - on_screen_w).max(0.0)
        } else {
            0.0
        };
        let mut inner = self
            .panel_content(s)
            .w(logical_w)
            .h(logical_h)
            .motion(Spring::NONE);
        inner = if panel_on_right {
            inner.pin(None, Some(0.0), Some(0.0), None)
        } else {
            inner.pin(Some(0.0), Some(0.0), None, None)
        };
        el().key("panel_slot")
            .absolute(x, 0.0)
            .w(on_screen_w)
            .h(self.window_height)
            .transition(Transition::none())
            .on_drag(panel_drag)
            .child(
                inner
                    .scale(scale)
                    .origin(if panel_on_right { 1.0 } else { 0.0 }, 0.0),
            )
    }

    /// The alternatives strip, laid out beside the panel but *before* it in
    /// paint order (see `view`). It fades in place instead of sliding.
    fn alternatives_overlay(&self, s: &ViewerApp) -> El {
        let scale = self.ui_scale();
        let panel_on_right = self.state.last_landscape_gravity == Gravity::End;
        let panel_w = self.panel_logical_width() * scale;
        let strip_w = NEIGHBOR_WIDTH * scale;
        let logical_h = self.logical_height();
        let x = if panel_on_right {
            (self.window_width - panel_w - strip_w).max(0.0)
        } else {
            panel_w
        };
        let mut inner = self
            .alternatives_panel(s)
            .w(NEIGHBOR_WIDTH)
            .h(logical_h)
            .motion(Spring::NONE);
        inner = if panel_on_right {
            inner.pin(None, Some(0.0), Some(0.0), None)
        } else {
            inner.pin(Some(0.0), Some(0.0), None, None)
        };
        el().key("alt_overlay")
            .absolute(x, 0.0)
            .w(strip_w)
            .h(self.window_height)
            .motion(Spring::NONE)
            .transition(Transition::none())
            .on_drag(panel_drag)
            .child(
                inner
                    .scale(scale)
                    .origin(if panel_on_right { 1.0 } else { 0.0 }, 0.0),
            )
    }

    fn dictionary_panel(&self, s: &ViewerApp) -> El {
        let entries = if !self.state.cached_entries.is_empty() {
            Rc::clone(&self.state.cached_entries)
        } else {
            Rc::new(Vec::new())
        };

        let mut content = col().padding(4.0).gap(4.0).w_full().motion(Spring::NONE);
        if entries.is_empty() {
            content = content.child(
                text("No dictionary entries found.")
                    .font(font(14.0))
                    .color(WHITE),
            );
        }
        for (i, entry) in entries.iter().enumerate() {
            let key = format!(
                "entry:{i}:{}:{}",
                entry.term,
                entry.dictionary_name.as_deref().unwrap_or("")
            );
            content = content.child(
                entry_view(entry, self.show_pitch)
                    .key(key)
                    .transition(Transition::none()),
            );
        }

        let scroll = s.dict_scroll.clone();
        content
            .key("dict_scroll")
            .w(DICT_WIDTH)
            .h_full()
            .min_h(0.0)
            .scroll(ScrollSpec {
                stick_bottom: false,
            })
            .on_scroll_offset(move |at| scroll.set(at.offset))
            .fill(PANEL_BG)
    }

    fn neighbor_panel(&self, _s: &ViewerApp) -> El {
        let state = self.state.get_neighbor_ui_state();
        let item_size = self.logical_width() / 11.0;
        let box_size = item_size.clamp(24.0, 40.0);
        let mut content = col().padding(2.0).gap(2.0).motion(Spring::NONE);
        for line in state {
            for cs in line.chars {
                let key = format!("nb:{}:{}", line.line_idx, cs.char_idx);
                let is_selected = cs.is_selected;
                let text_value = chip_text(&cs.text, self.is_landscape());
                let li = line.line_idx;
                let ci = cs.char_idx;
                let btn = el()
                    .key(key)
                    .absolute(0.0, 0.0)
                    .size(box_size, box_size)
                    .radius(0.0)
                    .fill(if is_selected { YELLOW } else { CHIP_BG })
                    .center()
                    .transition(Transition::none())
                    .on_press_tx(move |s: &mut ViewerApp, tx: &mut Tx| {
                        // Fill and text colour must change together; without this the
                        // text keeps Rune's short colour fade while the fill snaps.
                        tx.instant();
                        s.viewer.select_neighbor(li, ci);
                        tx.scroll_to_instant("dict_scroll", 0.0, 0.0);
                    })
                    .child(
                        text(text_value)
                            .font(font(box_size * BUTTON_CHAR_RATIO))
                            .color(if is_selected { BLACK } else { WHITE })
                            .a11y_hidden(),
                    );
                // Absolute inside a zero-height row encourages overlap; wrap
                // in a sized slot instead so the column stacks chips.
                content = content.child(
                    el().key(format!("nbslot:{li}:{ci}"))
                        .w(box_size)
                        .h(box_size)
                        .child(btn.absolute(0.0, 0.0)),
                );
            }
        }
        content
            .key("neighbor_scroll")
            .w(NEIGHBOR_WIDTH)
            .h_full()
            .min_h(0.0)
            .scroll(ScrollSpec {
                stick_bottom: false,
            })
            .fill(PANEL_BG)
    }

    fn alternatives_panel(&self, s: &ViewerApp) -> El {
        let item_size = self.logical_width() / 11.0;
        let box_size = item_size.clamp(24.0, 40.0);
        let mut content = col().padding(2.0).gap(2.0).motion(Spring::NONE);
        let selection = self
            .selected_word
            .as_ref()
            .map(|sw| (sw.line_idx, sw.char_idx))
            .unwrap_or((usize::MAX, usize::MAX));

        // Character preview image at the top, sized to match the chips.
        let preview = self.selected_word.as_ref().and_then(|sw| {
            let mut cache = self.cached_preview.borrow_mut();
            if let Some((cl, cc, px)) = cache.as_ref() {
                if *cl == sw.line_idx && *cc == sw.char_idx {
                    return Some(px.clone());
                }
            }
            let px = self.crop_character_image(sw.line_idx, sw.char_idx)?;
            *cache = Some((sw.line_idx, sw.char_idx, px.clone()));
            Some(px)
        });
        if let Some(px) = preview {
            content = content.child(
                el().key(format!("alt_preview:{}:{}", selection.0, selection.1))
                    .size(box_size, box_size)
                    .transition(Transition::none())
                    .pixels(&px),
            );
        }

        if let Some(alt_state) = self.state.get_alternatives_ui_state() {
            for (index, c) in alt_state.candidates.into_iter().enumerate() {
                let ch = c.char;
                let is_selected = c.is_selected;
                let tint = match c.source {
                    jpdict_core::util::oov_suggestions::Source::Head => WHITE,
                    jpdict_core::util::oov_suggestions::Source::Components
                    | jpdict_core::util::oov_suggestions::Source::Variant => {
                        rgba(1.0, 0.8, 0.4, 1.0)
                    }
                    jpdict_core::util::oov_suggestions::Source::Lm => rgba(0.55, 0.85, 1.0, 1.0),
                };
                let vertical_ch = chip_text(&ch.to_string(), self.is_landscape());
                // Include the slot and the selected character: when a neighbour
                // changes, a shared candidate may move slots. A key that follows
                // the char would animate it across the list; keying the slot
                // makes the old one fade out and the new one fade in.
                let key = format!(
                    "alt:{}:{}:{index}:{}",
                    selection.0, selection.1, c.char as u32
                );
                content = content.child(
                    el().key(key)
                        .size(box_size, box_size)
                        .fill(if is_selected { YELLOW } else { CHIP_BG })
                        .center()
                        .transition(Transition::none())
                        .on_press_tx(move |s: &mut ViewerApp, tx: &mut Tx| {
                            // Match the neighbour chips: selection colour and
                            // background change in the same frame.
                            tx.instant();
                            s.viewer.select_alternative(ch);
                        })
                        .child(
                            text(vertical_ch)
                                .font(font(box_size * BUTTON_CHAR_RATIO))
                                .color(if is_selected { BLACK } else { tint })
                                .a11y_hidden(),
                        ),
                );
            }
        }

        let _ = s;
        content
            .key("alt_scroll")
            .w(NEIGHBOR_WIDTH)
            .h_full()
            .min_h(0.0)
            .scroll(ScrollSpec {
                stick_bottom: false,
            })
            .fill(PANEL_BG)
    }
}

/// Add the shared dark panel background without clobbering an existing fill.
trait PanelBackground {
    fn background_panel(self) -> El;
}

impl PanelBackground for El {
    fn background_panel(self) -> El {
        self.fill(PANEL_BG).radius(4.0)
    }
}

// ---------------------------------------------------------------------------
// Entry rendering
// ---------------------------------------------------------------------------

fn entry_view(entry: &FormattedEntry, show_pitch: bool) -> El {
    let mut entry_col = col().gap(4.0).w_full();

    if let Some(chain) = &entry.deinflection {
        entry_col = entry_col.child(deinflection_row(chain, &entry.term));
    }
    entry_col = entry_col.child(headword_block(entry, show_pitch));

    let last_rendered = entry
        .reading_groups
        .iter()
        .rposition(|g| g.render_senses)
        .filter(|_| entry.dictionary_name.is_some());
    for (i, group) in entry.reading_groups.iter().enumerate() {
        if !group.render_senses {
            continue;
        }
        for sg in &group.sense_groups {
            entry_col = entry_col.child(sense_group(sg));
        }
        if Some(i) == last_rendered {
            continue;
        }
        entry_col = entry_col.child(
            col()
                .py(10.0)
                .w_full()
                .child(el().w_full().h(1.0).fill(rgba(
                    169.0 / 255.0,
                    169.0 / 255.0,
                    169.0 / 255.0,
                    0.3,
                ))),
        );
    }

    if let Some(name) = &entry.dictionary_name {
        entry_col = entry_col.child(
            row()
                .w_full()
                .pt(4.0)
                .justify_end()
                .child(text(name.clone()).font(font(12.0)).color(LT_GRAY)),
        );
    }
    entry_col
}

fn deinflection_row(chain: &DeinflectionChain, term: &str) -> El {
    let mut r = row().w_full().gap(4.0).center_y().flex_wrap().child(
        text(format!("{} → {}", chain.surface, term))
            .font(font(12.0))
            .color(LT_GRAY),
    );
    for step in &chain.steps {
        r = r.child(tag_chip(step));
    }
    r
}

fn headword_block(entry: &FormattedEntry, show_pitch: bool) -> El {
    let mut content = col().w_full().gap(2.0);

    let kanji_groups: Vec<&FormattedReadingGroup> = entry
        .reading_groups
        .iter()
        .filter(|g| g.is_kanji_entry)
        .collect();
    for group in kanji_groups {
        for hw in &group.headwords {
            let mut r = row().gap(10.0).center_y();
            r = r.child(text(hw.kanji.clone()).font(bold(48.0)).color(CYAN));
            let mut readings = col().gap(0.0);
            if let Some(k) = &hw.kunyomi {
                if !k.is_empty() {
                    readings = readings.child(kun_on_row("訓", &OcrViewer::kana_list_text(k)));
                }
            }
            if let Some(o) = &hw.onyomi {
                if !o.is_empty() {
                    readings = readings.child(kun_on_row("音", &OcrViewer::kana_list_text(o)));
                }
            }
            r = r.child(readings);
            content = content.child(r);
        }
    }

    let term_groups: Vec<&FormattedReadingGroup> = entry
        .reading_groups
        .iter()
        .filter(|g| !g.is_kanji_entry)
        .collect();
    if !term_groups.is_empty() {
        let pairs: Vec<(String, String)> = term_groups
            .iter()
            .flat_map(|g| {
                g.headwords
                    .iter()
                    .map(|h| (h.kanji.clone(), g.reading.clone()))
            })
            .collect();
        let reserve_ruby = pairs.iter().any(|(kanji, reading)| kanji != reading);
        let mut r = row().w_full().gap(2.0).items_end().flex_wrap();
        for (i, (kanji, reading)) in pairs.iter().enumerate() {
            r = r.child(ruby_view(kanji, reading, false, reserve_ruby));
            if i + 1 < pairs.len() {
                r = r.child(text("、").font(font(24.0)).color(LT_GRAY));
            }
        }
        content = content.child(r);

        let items: Vec<(String, Vec<i32>)> = pitch_items(entry, show_pitch);
        if !items.is_empty() {
            content = content.child(pitch_line(&items));
        }
    }
    content
}

impl OcrViewer {
    fn kana_list_text(raw: &str) -> String {
        japanese::split_kana_list(raw).join("、")
    }
}

fn kun_on_row(label: &str, readings: &str) -> El {
    row()
        .gap(6.0)
        .center_y()
        .child(text(label.to_string()).font(font(12.0)).color(LT_GRAY))
        .child(text(readings.to_string()).font(font(13.0)).color(LT_GRAY))
}

fn pitch_items(entry: &FormattedEntry, show_pitch: bool) -> Vec<(String, Vec<i32>)> {
    if !show_pitch {
        return Vec::new();
    }
    entry
        .reading_groups
        .iter()
        .filter(|g| !g.is_kanji_entry && !g.pitch_positions.is_empty())
        .map(|g| (g.reading.clone(), g.pitch_positions.clone()))
        .collect()
}

fn pitch_line(items: &[(String, Vec<i32>)]) -> El {
    let accent = WHITE;
    let plain = Color {
        r: 0.406448,
        g: 0.406448,
        b: 0.406448,
        a: 1.0,
    };
    let mut r = row().w_full().gap(1.0).items_start().flex_wrap();
    for (i, (reading, positions)) in items.iter().enumerate() {
        let morae = japanese::morae_of(reading);
        if morae.is_empty() {
            continue;
        }
        if i > 0 {
            r = r.child(text("、").font(font(13.0)).color(plain));
        }
        let position = positions.first().copied().unwrap_or(0);
        let contour = japanese::pitch_pattern(morae.len(), position);
        for (mora_index, mora) in morae.iter().enumerate() {
            let color = if contour.get(mora_index).copied().unwrap_or(false) {
                accent
            } else {
                plain
            };
            r = r.child(text(mora.clone()).font(font(13.0)).color(color));
        }
        if japanese::falls_beyond_word(morae.len(), position) {
            r = r.child(text("↓").font(font(9.0)).color(plain));
        }
    }
    r
}

fn tag_color(tag: &str) -> Color {
    if tag == "pos" || tag.starts_with('v') || tag == "adj-i" || tag == "adj-na" {
        rgba(0.23, 0.35, 0.48, 1.0)
    } else if tag == "n" || tag == "adv" || tag == "pn" {
        rgba(0.23, 0.48, 0.35, 1.0)
    } else if tag.starts_with("jlpt") || tag.starts_with("grade") || tag == "★" || tag == "meta" {
        rgba(0.48, 0.23, 0.23, 1.0)
    } else {
        rgba(0.27, 0.27, 0.27, 1.0)
    }
}

fn tag_chip(tag: &str) -> El {
    let bg = tag_color(tag);
    el().px(4.0)
        .py(1.0)
        .radius(5.0)
        .fill(bg)
        .child(text(tag.to_string()).font(bold(10.0)).color(WHITE))
}

fn ruby_base(style: &jpdict_core::ruby_style::RubyStyle) -> Color {
    rgba(style.base[0], style.base[1], style.base[2], 1.0)
}

fn ruby_tint(style: &jpdict_core::ruby_style::RubyStyle) -> Color {
    rgba(style.ruby[0], style.ruby[1], style.ruby[2], 1.0)
}

fn ruby_view(term: &str, reading: &str, is_mini: bool, reserve_ruby_space: bool) -> El {
    let style = jpdict_core::ruby_style::ruby_style(is_mini);
    let base = |text_value: String| -> El {
        let label = text(text_value)
            .font(if style.bold {
                bold(style.base_size)
            } else {
                font(style.base_size)
            })
            .color(ruby_base(&style));
        label
    };
    if term == reading {
        if !reserve_ruby_space {
            return base(term.to_string());
        }
        return col()
            .center()
            .gap(0.0)
            .child(
                text(" ")
                    .font(font(style.ruby_size))
                    .color(ruby_tint(&style)),
            )
            .child(base(term.to_string()));
    }
    let segments = japanese::align_furigana(term, reading);
    let Some(segments) = segments else {
        return full_ruby_view(term, reading, is_mini);
    };
    if !segments.iter().any(|s| s.ruby.is_some()) {
        return full_ruby_view(term, reading, is_mini);
    }
    let mut r = row().items_end();
    for seg in segments {
        match seg.ruby {
            None => r = r.child(base(seg.base)),
            Some(ruby) => {
                r = r.child(
                    col()
                        .center()
                        .gap(0.0)
                        .child(
                            text(ruby)
                                .font(font(style.ruby_size))
                                .color(ruby_tint(&style)),
                        )
                        .child(base(seg.base)),
                );
            }
        }
    }
    r
}

fn full_ruby_view(term: &str, reading: &str, is_mini: bool) -> El {
    let style = jpdict_core::ruby_style::ruby_style(is_mini);
    col()
        .center()
        .gap(0.0)
        .child(
            text(reading.to_string())
                .font(font(style.ruby_size))
                .color(ruby_tint(&style)),
        )
        .child(
            text(term.to_string())
                .font(if style.bold {
                    bold(style.base_size)
                } else {
                    font(style.base_size)
                })
                .color(ruby_base(&style)),
        )
}

fn sense_group(sg: &FormattedSenseGroup) -> El {
    let mut content = col().gap(3.0).w_full();

    if !sg.tags.is_empty() {
        let mut tag_row = row().w_full().gap(3.0).center_y().pt(4.0).flex_wrap();
        for tag in &sg.tags {
            tag_row = tag_row.child(tag_chip(tag));
        }
        content = content.child(tag_row);
    }

    if !sg.header.is_empty() {
        content = content.child(render_definition(&sg.header, DICT_TEXT_WIDTH).pt(2.0));
    }

    if sg.is_forms {
        for sense in &sg.senses {
            content = content.child(render_definition(&sense.nodes, DICT_TEXT_WIDTH));
        }
    } else {
        for sense in &sg.senses {
            // The body must take the space left after the number, not the
            // whole row: `w_full` on a flex item after the number would
            // overflow the panel and clip example-box borders.
            let body = render_definition(&sense.nodes, DICT_TEXT_WIDTH - 20.0);
            content = content.child(
                row()
                    .w_full()
                    .gap(3.0)
                    .items_start()
                    .child(
                        text(format!("{}. ", sense.index))
                            .font(font(15.0))
                            .color(WHITE),
                    )
                    .child(el().grow(1.0).min_w(0.0).child(body)),
            );
        }
    }

    if !sg.trailing.is_empty() {
        content = content.child(render_definition(&sg.trailing, DICT_TEXT_WIDTH).pb(2.0));
    }
    content
}

fn render_definition(nodes: &[DefinitionNode], width: f32) -> El {
    let mut column = col().gap(2.0).w_full();
    let mut inline: Vec<DefinitionNode> = Vec::new();
    for node in nodes {
        match node {
            DefinitionNode::Text(_) | DefinitionNode::Ruby { .. } | DefinitionNode::Tag { .. } => {
                inline.push(node.clone());
            }
            DefinitionNode::Citation(text_value) => {
                column = flush_inline(column, &mut inline, width);
                column = column.child(text(text_value.clone()).font(font(11.0)).color(rgba(
                    1.0,
                    1.0,
                    1.0,
                    120.0 / 255.0,
                )));
            }
            DefinitionNode::Example(example) => {
                column = flush_inline(column, &mut inline, width);
                column = column.child(example_box(example, width));
            }
            DefinitionNode::ListBlock { items, .. } => {
                column = flush_inline(column, &mut inline, width);
                let mut block = col().gap(3.0).padding(4.0).pl(2.0);
                for item in items {
                    block = block.child(render_definition(item, width - 8.0));
                }
                column = column.child(block);
            }
            DefinitionNode::Table { rows } => {
                if !rows.is_empty() {
                    column = flush_inline(column, &mut inline, width);
                    column = column.child(definition_table(rows, width));
                }
            }
            DefinitionNode::Group { nodes, is_inline } => {
                if *is_inline {
                    inline.extend(nodes.iter().cloned());
                } else {
                    column = flush_inline(column, &mut inline, width);
                    column = column.child(render_definition(nodes, width));
                }
            }
        }
    }
    flush_inline(column, &mut inline, width)
}

fn flush_inline(mut column: El, inline: &mut Vec<DefinitionNode>, width: f32) -> El {
    if inline.is_empty() {
        return column;
    }
    let cells = inline_cells(inline);
    inline.clear();
    let mut flow = col().gap(2.0).w_full();
    for line in wrap_inline_cells(cells, width) {
        flow = flow.child(inline_line(line));
    }
    column = column.child(flow);
    column
}

fn inline_cells(nodes: &[DefinitionNode]) -> Vec<InlineCell> {
    let mut cells = Vec::new();
    for node in nodes {
        match node {
            DefinitionNode::Text(text_value) => {
                cells.extend(text_value.chars().map(InlineCell::Char))
            }
            DefinitionNode::Ruby { term, reading } => cells.push(InlineCell::Ruby {
                term: term.clone(),
                reading: reading.clone(),
                width: inline_text_width(term, DEF_TEXT_SIZE)
                    .max(inline_text_width(reading, DEF_RUBY_SIZE)),
            }),
            DefinitionNode::Tag { text: tag } => cells.push(InlineCell::Tag {
                text: tag.clone(),
                width: inline_text_width(tag, 10.0) + 12.0,
            }),
            _ => {}
        }
    }
    cells
}

fn wrap_inline_cells(cells: Vec<InlineCell>, width: f32) -> Vec<Vec<InlineCell>> {
    let avail = width.max(1.0);
    let mut lines: Vec<Vec<InlineCell>> = Vec::new();
    let mut line: Vec<InlineCell> = Vec::new();
    let mut used = 0.0f32;
    let line_width = |l: &[InlineCell]| l.iter().map(inline_cell_width).sum::<f32>();
    for cell in cells {
        let w = inline_cell_width(&cell);
        if line.is_empty() || used + w <= avail {
            line.push(cell);
            used += w;
            continue;
        }
        let mut split = line.len();
        while split > 0 {
            match &line[split - 1] {
                InlineCell::Char(c) if c.is_ascii_alphanumeric() => split -= 1,
                _ => break,
            }
        }
        let tail = if split > 0 && split < line.len() {
            line.split_off(split)
        } else {
            Vec::new()
        };
        let mut carry = None;
        if tail.is_empty() && line.len() >= 2 {
            if let InlineCell::Char(c) = &cell {
                if must_not_start_line(*c) {
                    carry = line.pop();
                }
            }
        }
        lines.push(std::mem::take(&mut line));
        line = tail;
        if let Some(c) = carry {
            line.push(c);
        }
        used = line_width(&line);
        line.push(cell);
        used += w;
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

fn inline_line(line: Vec<InlineCell>) -> El {
    let mut r = row().items_end();
    let mut run = String::new();
    let flush_run = |r: El, run: &mut String| {
        if run.is_empty() {
            r
        } else {
            r.child(
                text(std::mem::take(run))
                    .font(font(DEF_TEXT_SIZE))
                    .color(WHITE),
            )
        }
    };
    for cell in line {
        match cell {
            InlineCell::Char(c) => run.push(c),
            InlineCell::Ruby { term, reading, .. } => {
                r = flush_run(r, &mut run);
                r = r.child(ruby_view(&term, &reading, true, false));
            }
            InlineCell::Tag { text: tag, .. } => {
                r = flush_run(r, &mut run);
                r = r.child(tag_chip(&tag));
            }
        }
    }
    flush_run(r, &mut run)
}

fn example_box(example: &ExampleNode, width: f32) -> El {
    let mut inner = col().gap(3.0).w_full();
    if let Some(jp) = &example.japanese {
        inner = inner.child(
            text(jp.clone())
                .font(font(16.0))
                .color(WHITE)
                .wrap()
                .w_full(),
        );
        if let Some(en) = &example.english {
            inner = inner.child(
                text(en.clone())
                    .font(font(14.0))
                    .color(LT_GRAY)
                    .wrap()
                    .w_full(),
            );
        }
    } else if !example.parts.is_empty() {
        for part in &example.parts {
            inner = inner.child(render_definition(part, width - 16.0));
        }
    } else if !example.content.is_empty() {
        inner = inner.child(render_definition(&example.content, width - 16.0));
    }
    inner
        .w_full()
        .padding(8.0)
        .radius(8.0)
        .fill(rgba(1.0, 1.0, 1.0, 10.0 / 255.0))
        .border(1.0, rgba(1.0, 1.0, 1.0, 80.0 / 255.0))
}

fn definition_table(rows: &[Vec<Vec<DefinitionNode>>], width: f32) -> El {
    let border = rgba(1.0, 1.0, 1.0, 80.0 / 255.0);
    let columns = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let cell_width = if columns > 0 {
        (width - columns.saturating_sub(1) as f32) / columns as f32
    } else {
        width
    };
    let mut table = col().w_full().gap(0.0);
    for (row_index, cells) in rows.iter().enumerate() {
        let mut r = row().w_full().items_stretch();
        for column in 0..columns {
            if column > 0 {
                r = r.child(el().w(1.0).h_full().fill(border));
            }
            let mut cell = col().padding(4.0).px(6.0).w(cell_width);
            if let Some(nodes) = cells.get(column) {
                cell = cell.child(render_definition(nodes, cell_width - 12.0));
            }
            r = r.child(cell);
        }
        table = table.child(r);
        if row_index + 1 < rows.len() {
            table = table.child(el().w_full().h(1.0).fill(border));
        }
    }
    table
        .w_full()
        .radius(8.0)
        .fill(rgba(1.0, 1.0, 1.0, 10.0 / 255.0))
        .border(1.0, border)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_metrics_follow_the_1280_logical_grid() {
        let mut viewer = OcrViewer::new_empty(1280.0, 720.0, FontFace::Sans);
        assert!((viewer.ui_scale() - 1.0).abs() < 1e-6);
        assert!((viewer.panel_logical_width() - 348.0).abs() < 1e-6);
        assert!((viewer.panel_width() - 390.0).abs() < 1e-6);

        viewer.window_width = 1920.0;
        viewer.window_height = 1080.0;
        assert!((viewer.ui_scale() - 1.5).abs() < 1e-6);
        assert!((viewer.panel_logical_width() - 348.0).abs() < 1e-6);
        assert!((viewer.panel_width() - 585.0).abs() < 1e-6);

        viewer.window_width = 2560.0;
        assert!((viewer.ui_scale() - 2.0).abs() < 1e-6);
        assert!((viewer.panel_width() - 780.0).abs() < 1e-6);
    }

    #[test]
    fn blank_gap_placeholder_is_the_reversible_one() {
        // The detector itself lives in core; this guards that the viewer's
        // edits and the core's gap materialiser use the same placeholder.
        assert_eq!(GAP_CHAR, '\u{25CC}');
    }

    struct PanelState;

    fn panel_view(_: &PanelState, _: &mut Cx<PanelState>) -> El {
        col()
            .w(300.0)
            .h_full()
            .padding(4.0)
            .child(entry_view(&sample_entry(), true))
    }

    fn sample_entry() -> FormattedEntry {
        FormattedEntry {
            term: "食べる".into(),
            reading_groups: vec![FormattedReadingGroup {
                reading: "たべる".into(),
                headwords: vec![FormattedHeadword {
                    kanji: "食べる".into(),
                    onyomi: None,
                    kunyomi: None,
                }],
                sense_groups: vec![FormattedSenseGroup {
                    tags: vec!["v1".into()],
                    senses: vec![FormattedSense {
                        index: 1,
                        nodes: vec![
                            DefinitionNode::Text("to eat; ".into()),
                            DefinitionNode::Ruby {
                                term: "食べる".into(),
                                reading: "たべる".into(),
                            },
                            DefinitionNode::Tag { text: "v1".into() },
                            DefinitionNode::Example(ExampleNode {
                                japanese: Some("ご飯を食べる。".into()),
                                english: Some("to eat a meal".into()),
                                ..Default::default()
                            }),
                        ],
                    }],
                    is_forms: false,
                    header: Vec::new(),
                    trailing: Vec::new(),
                }],
                is_kanji_entry: false,
                pitch_positions: vec![1],
                render_senses: true,
            }],
            deinflection: Some(DeinflectionChain {
                surface: "食べた".into(),
                steps: vec!["past".into()],
            }),
            dictionary_name: Some("Jitendex".into()),
        }
    }

    /// The panel's full entry path (ruby, tags, examples, pitch line) must
    /// lay out without a panic. It used to be a set of widgets; this
    /// pins the Rune translation's structure.
    #[test]
    fn dictionary_entry_lays_out() {
        let mut rt = rune::Runtime::new(
            PanelState,
            panel_view,
            (300.0, 900.0),
            rune::Env::new(true, 1.0),
        );
        rt.frame(0.0);
        assert!(rt.find("panel").is_none());
    }
}
