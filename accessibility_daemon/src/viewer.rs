use std::cell::RefCell;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use iced::widget::canvas::{
    self, Canvas, Frame, Geometry, Path as CanvasPath, Stroke as CanvasStroke,
};
use iced::widget::{
    Column, Container, Image as IcedImage, Row, Scrollable, Stack, Text,
};
use iced::widget::container;
use iced::advanced::widget::operation::scrollable::{
    scroll_to, scroll_by, AbsoluteOffset,
};
use iced::widget::Id;
use iced::advanced::widget::operate;
use iced::{
    alignment, Color, Element, Font as IcedFont, Length, Pixels, Point, Rectangle, Renderer,
    Size, Theme, mouse, touch,
};

use jpdict_core::data::db::DictionaryDatabase;
use jpdict_core::data::models::DictionaryEntry;
use jpdict_core::models::*;
use jpdict_core::overlay_font::FontFace;
use jpdict_core::overlay_state::OcrOverlayState;
use jpdict_core::util::deinflector::Deinflector;
use jpdict_core::util::japanese;
use jpdict_core::util::japanese::{is_half_width, to_vertical_glyph};

use fontdue::Font;
use iced::widget::image::Handle as ImageHandle;

/// Text width available inside the fixed 300px dictionary column: panel and
/// scroll padding removed, with slack so estimated line breaks cannot
/// overflow the column.
const DICT_TEXT_WIDTH: f32 = 278.0;

/// Body text size in definitions and examples (mobile: 15sp) and the ruby
/// size that sits above it (mobile: 9sp).
const DEF_TEXT_SIZE: f32 = 15.0;
const DEF_RUBY_SIZE: f32 = 9.0;

/// Atomic inline cell for the definition flow layout.
enum InlineCell {
    Char(char),
    Ruby { term: String, reading: String, width: f32 },
    Tag { text: String, width: f32 },
}

/// Ruby typography for the two display modes, shared by the aligned path
/// (`ruby_view`) and the `full_ruby_view` fallback so both paint identically
/// within a mode.
///
/// INTENTIONAL DEPARTURE FROM MOBILE (2026-09-21, maintainer decision,
/// ticket 06): Android `createRubyView` paints the base CYAN + bold in *both*
/// modes — full-size (32sp base / 13sp ruby) for the headword/term display,
/// mini (15sp / 9sp) for definition-flow ruby (`OcrOverlayView.kt:2115-2124`
/// `createBaseTextView`; definition `Ruby` nodes are built with `isMini =
/// true` and rendered at `OcrOverlayView.kt:2316`). The PC instead renders
/// body (mini) ruby in the surrounding body typeface — WHITE regular base at
/// body size, gray ruby row — so ruby-annotated spans match plain
/// `InlineCell::Char` runs. The term display keeps bold cyan. Do NOT "fix"
/// mini back to cyan+bold for parity: that would reintroduce the ticket-06
/// symptom on both codebases by design.
///
/// The resolved values live in [`jpdict_core::ruby_style`] (the single
/// implementation; the conformance corpus pins it). This file only converts
/// them to iced colours at the paint sites via [`ruby_base`] / [`ruby_tint`].

/// Fontdue metrics for the same face iced renders the panel with, given once
/// by the app at startup (`init_panel_metrics`). Without it — unit tests —
/// the flow falls back to em-category estimates.
static PANEL_METRICS: std::sync::OnceLock<fontdue::Font> = std::sync::OnceLock::new();

/// Hand the flow layout the panel font's real advances. Call once with the
/// same bytes that are registered with iced.
pub fn init_panel_metrics(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    if let Ok(font) = fontdue::Font::from_bytes(bytes.to_vec(), fontdue::FontSettings::default()) {
        let _ = PANEL_METRICS.set(font);
    }
}

/// Real advance of `text` at `size` in the panel face, kerning included.
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

/// Mobile `LineOverlayView.textSize = fixedSize * 0.90f`: the one glyph size
/// per line, taken from the line's own box (never a measured pitch).
const TEXT_SIZE_RATIO: f32 = 0.9;
/// Mobile LineOverlayView `ASCII_GLYPH_SCALE` (#49): the shared line-height
/// text size renders halfwidth glyphs ~10% oversized next to CJK.
const ASCII_GLYPH_SCALE: f32 = 0.9;
/// Mobile LineOverlayView per-glyph box fit: `maxW = boxW * 0.92f`.
const GLYPH_FIT_RATIO: f32 = 0.92;
/// Mobile `updateCursor`: a 2dp white outline (no fill) with 4px rounded
/// corners, on a box inflated 2dp per side (a 4dp oversize in each axis).
const CURSOR_PAD: f32 = 2.0;
const CURSOR_RADIUS: f32 = 4.0;
const CURSOR_STROKE: f32 = 2.0;
/// Mobile line-box / quad-border fill: `Color.argb(100, 0, 0, 0)`.
const BOX_FILL_ALPHA: f32 = 100.0 / 255.0;
/// Mobile corner radius on the box fill (raw source px, scaled by the
/// content transform): `cornerRadius = 4f` / `CornerPathEffect(4f)`.
const BOX_CORNER_RADIUS: f32 = 4.0;
/// Mobile `OverlayBackdrop.SCREENSHOT_ALPHA` (#64): the screenshot is shown
/// dimmed behind the overlay.
const SCREENSHOT_ALPHA: f32 = 0.7;
/// Mobile `OverlayBackdrop.SCRIM_COLOR = 0x8C000000`, alpha channel only.
const SCRIM_ALPHA: f32 = 140.0 / 255.0;
/// iced's `draw_image` has no alpha, so the two mobile layers fold into one
/// scrim over the opaque screenshot: `img · α_screenshot · (1 − α_scrim)`.
/// Desktop has no status strip, so the scrim is flat everywhere (SOLID).
const BACKDROP_SCRIM: f32 = 1.0 - SCREENSHOT_ALPHA * (1.0 - SCRIM_ALPHA);
/// Font fill ratio for character buttons in the neighbor/alternatives panels.
const BUTTON_CHAR_RATIO: f32 = 0.6;

/// Dead zone for drag start in the TapOrDrag widget.
/// The first few pixels of movement don't count as drag, preventing
/// accidental drags from taps.
const DRAG_DEAD_ZONE: f32 = 3.0;

// ---------------------------------------------------------------------------
// GlyphCache — rasterizes characters with fontdue, caches as Iced image
// handles. Completely bypasses cosmic-text to avoid font atlas corruption.
// ---------------------------------------------------------------------------

/// Standard pink color for overlay text. (#FF7777)
const OVERLAY_FG: (u8, u8, u8) = (255, 119, 119);
/// Yellow for highlighted/matched characters.
const OVERLAY_HL: (u8, u8, u8) = (255, 255, 0);

/// Which face/variant a cached raster belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum GlyphKind {
    /// Regular face, pink.
    Pink,
    /// Bold companion face, highlight yellow.
    Bold,
    /// Synthetic bold derived from the regular face (no bold companion).
    Synthetic,
}

/// Lazily rasterizes characters at requested pixel sizes, caching RGBA
/// image handles. Thread‑safe via RefCell for interior mutability.
pub struct GlyphCache {
    font: Font,
    /// Real bold companion for highlighted glyphs; `None` falls back to
    /// synthetic emboldening (mobile ships Regular only).
    bold_font: Option<Font>,
    /// Rasterized glyphs keyed by `(kind, glyph id, px)`: the id may be a
    /// GSUB substitution (vertical forms), not just the cmap glyph.
    cache: HashMap<(GlyphKind, u16, u32), RasterGlyph>,
    /// GSUB `vert`/`vrt2` resolution results (source glyph -> vertical glyph).
    vert_cache: HashMap<u16, u16>,
    /// Same for the bold companion face.
    bold_vert_cache: HashMap<u16, u16>,
    /// Font-parser view of the regular bytes, for cmap and GSUB `vert`/`vrt2`
    /// resolution (fontdue exposes neither).
    vface: Option<ttf_parser::Face<'static>>,
    /// Same for the bold companion.
    bold_vface: Option<ttf_parser::Face<'static>>,
}

/// Resolve the Japanese UI font file used by BOTH the OCR overlay glyph
/// cache and the iced dictionary panel (they must render identically).
/// Face selection and file resolution live in [`jpdict_core::overlay_font`]:
/// sans (the default) or serif, with a serif selection whose bundled file
/// is missing falling back to sans.
///
/// [`GlyphCache::new_for`] resolves the face actually painted.

impl GlyphCache {
    /// Cache for the selected face, with the bold companion that matches the
    /// file actually resolved: a serif selection that falls back to the sans
    /// file (its own missing) pairs with the sans bold, never a mismatched
    /// one. `None` means the overlay draws boxes without glyphs — mobile
    /// degrades the same way (#84) rather than crashing.
    pub fn new_for(face: FontFace) -> Option<Rc<RefCell<Self>>> {
        let regular = jpdict_core::overlay_font::find_font_path(face);
        let bold = regular
            .as_deref()
            .map(jpdict_core::overlay_font::face_of)
            .and_then(jpdict_core::overlay_font::find_bold_font_path);
        Self::from_paths(regular, bold)
    }

    /// Test/helper entry: regular face only (synthetic bold fallback).
    #[cfg(test)]
    fn from_path(path: Option<std::path::PathBuf>) -> Option<Rc<RefCell<Self>>> {
        Self::from_paths(path, None)
    }

    /// Load the cache from specific font files: the regular face and an
    /// optional bold companion. A missing regular path or an unreadable face
    /// yields `None` and the overlay draws boxes without glyphs — mobile
    /// falls back to the platform face and never crashes the overlay for a
    /// missing bundled font (OverlayFont #84).
    fn from_paths(
        path: Option<std::path::PathBuf>,
        bold_path: Option<std::path::PathBuf>,
    ) -> Option<Rc<RefCell<Self>>> {
        let Some(path) = path else {
            eprintln!("[GlyphCache] no CJK font found, overlay text will not render");
            return None;
        };
        let Ok(data) = std::fs::read(&path) else {
            eprintln!("[GlyphCache] failed to read font {}", path.display());
            return None;
        };
        // ttf-parser borrows the font bytes; they live for the process (one
        // font, one cache), so leak that copy.
        let face_data: &'static [u8] = Box::leak(data.clone().into_boxed_slice());
        let vface = ttf_parser::Face::parse(face_data, 0).ok();
        let Ok(font) = Font::from_bytes(data, fontdue::FontSettings::default()) else {
            eprintln!("[GlyphCache] failed to parse font {}", path.display());
            return None;
        };
        let (bold_font, bold_vface) = match bold_path {
            Some(bp) => match std::fs::read(&bp) {
                Ok(bdata) => {
                    let bface_data: &'static [u8] =
                        Box::leak(bdata.clone().into_boxed_slice());
                    let bface = ttf_parser::Face::parse(bface_data, 0).ok();
                    match Font::from_bytes(bdata, fontdue::FontSettings::default()) {
                        Ok(bf) => {
                            println!("[GlyphCache] bold face: {}", bp.display());
                            (Some(bf), bface)
                        }
                        Err(_) => {
                            eprintln!(
                                "[GlyphCache] failed to parse bold font {}; highlights use synthetic bold",
                                bp.display()
                            );
                            (None, None)
                        }
                    }
                }
                Err(_) => (None, None),
            },
            None => {
                eprintln!("[GlyphCache] no bold face found; highlights use synthetic bold");
                (None, None)
            }
        };
        Some(Rc::new(RefCell::new(GlyphCache {
            font,
            bold_font,
            cache: HashMap::new(),
            vert_cache: HashMap::new(),
            bold_vert_cache: HashMap::new(),
            vface,
            bold_vface,
        })))
    }

    /// Look up or rasterize `ch` and return its drawable variant. Highlighted
    /// glyphs come from the bundled bold companion face when one is present;
    /// otherwise they fall back to synthetic emboldening (mobile's
    /// `paint.isFakeBoldText`, which grows the stroke without clipping it).
    fn glyph_for(
        &mut self,
        ch: char,
        vertical: bool,
        px: u32,
        highlighted: bool,
    ) -> Option<RasterGlyph> {
        if highlighted {
            if self.bold_font.is_some() {
                if let Some(gid) = self.bold_glyph_id(ch, vertical) {
                    return Some(self.variant(GlyphKind::Bold, gid, px));
                }
            }
            let gid = self.glyph_id(ch, vertical)?;
            return Some(self.variant(GlyphKind::Synthetic, gid, px));
        }
        let gid = self.glyph_id(ch, vertical)?;
        Some(self.variant(GlyphKind::Pink, gid, px))
    }

    /// Rasterize (or fetch) one glyph variant. `gid` must belong to the face
    /// selected by `kind`.
    fn variant(&mut self, kind: GlyphKind, gid: u16, px: u32) -> RasterGlyph {
        if let Some(entry) = self.cache.get(&(kind, gid, px)) {
            return entry.clone();
        }
        let raster = match kind {
            GlyphKind::Synthetic => {
                let (m, coverage) = Self::rasterize(&self.font, gid, px);
                let (w, h) = (m.width as u32, m.height as u32);
                let radius = fake_bold_radius(px);
                let (bold, bw, bh) = synthetic_bold(&coverage, w, h, radius);
                RasterGlyph {
                    handle: Self::make_handle(bw, bh, &bold, OVERLAY_HL.0, OVERLAY_HL.1, OVERLAY_HL.2),
                    w: bw,
                    h: bh,
                    xmin: m.xmin - radius,
                    ymin: m.ymin - radius,
                }
            }
            GlyphKind::Bold => {
                let Some(face) = self.bold_font.as_ref() else {
                    return self.variant(GlyphKind::Synthetic, gid, px);
                };
                let (m, coverage) = Self::rasterize(face, gid, px);
                let (w, h) = (m.width as u32, m.height as u32);
                RasterGlyph {
                    handle: Self::make_handle(w, h, &coverage, OVERLAY_HL.0, OVERLAY_HL.1, OVERLAY_HL.2),
                    w,
                    h,
                    xmin: m.xmin,
                    ymin: m.ymin,
                }
            }
            GlyphKind::Pink => {
                let (m, coverage) = Self::rasterize(&self.font, gid, px);
                let (w, h) = (m.width as u32, m.height as u32);
                RasterGlyph {
                    handle: Self::make_handle(w, h, &coverage, OVERLAY_FG.0, OVERLAY_FG.1, OVERLAY_FG.2),
                    w,
                    h,
                    xmin: m.xmin,
                    ymin: m.ymin,
                }
            }
        };
        self.cache.insert((kind, gid, px), raster.clone());
        raster
    }

    fn rasterize(font: &Font, gid: u16, px: u32) -> (fontdue::Metrics, Vec<u8>) {
        font.rasterize_config(fontdue::layout::GlyphRasterConfig {
            glyph_index: gid,
            px: px as f32,
            font_hash: 0,
        })
    }

    fn make_handle(w: u32, h: u32, cov: &[u8], r: u8, g: u8, b: u8) -> ImageHandle {
        if w == 0 || h == 0 {
            // No ink: a 1×1 transparent pixel keeps the handle valid; the
            // draw path never places it.
            return ImageHandle::from_rgba(1, 1, vec![0, 0, 0, 0]);
        }
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for &a in cov {
            rgba.push(r);
            rgba.push(g);
            rgba.push(b);
            rgba.push(a);
        }
        ImageHandle::from_rgba(w, h, rgba)
    }

    /// Pre-warm both tinted variants for a character at the given pixel size.
    /// Called from the update handler so glyph rasterization happens off the
    /// view/draw path, preventing first-frame stutter.
    pub fn ensure_glyph(&mut self, ch: char, vertical: bool, px_size: u32) {
        let _ = self.glyph_for(ch, vertical, px_size, false);
        let _ = self.glyph_for(ch, vertical, px_size, true);
    }

    /// The glyph to rasterize for `ch` in this line orientation. Vertical
    /// text uses the font's own GSUB `vert`/`vrt2` substitution; where the
    /// font has no entry for a character (e.g. ！ ？ … ；), the Unicode
    /// vertical presentation form is used as the fallback.
    fn glyph_id(&mut self, ch: char, vertical: bool) -> Option<u16> {
        let gid = match self.vface.as_ref() {
            Some(face) => match face.glyph_index(ch) {
                Some(g) => g.0,
                // Not in the font: fontdue's lookup returns .notdef (0),
                // which is what rasterize-by-char used to draw.
                None => return Some(self.font.lookup_glyph_index(ch)),
            },
            // No parser view (font tables unreadable): keep drawing the
            // plain horizontal glyph rather than nothing.
            None => return Some(self.font.lookup_glyph_index(ch)),
        };
        if !vertical {
            return Some(gid);
        }
        if let Some(v) = self.vert_cache.get(&gid) {
            return Some(*v);
        }
        let fallback = self.vface.as_ref().and_then(|face| {
            let vch = jpdict_core::util::japanese::to_vertical_glyph(ch);
            if vch != ch {
                face.glyph_index(vch).map(|g| g.0)
            } else {
                None
            }
        });
        let vert = self
            .vface
            .as_ref()
            .and_then(|face| gsub_vert_glyph(face, gid))
            .or(fallback)
            .unwrap_or(gid);
        self.vert_cache.insert(gid, vert);
        Some(vert)
    }

    /// Same resolution as [`Self::glyph_id`], on the bold companion face.
    /// `None` when no bold face is loaded or it has no glyph for `ch`.
    fn bold_glyph_id(&mut self, ch: char, vertical: bool) -> Option<u16> {
        if self.bold_font.is_none() {
            return None;
        }
        let gid = match self.bold_vface.as_ref() {
            Some(face) => match face.glyph_index(ch) {
                Some(g) => g.0,
                None => self.bold_font.as_ref()?.lookup_glyph_index(ch),
            },
            None => self.bold_font.as_ref()?.lookup_glyph_index(ch),
        };
        if !vertical {
            return Some(gid);
        }
        if let Some(v) = self.bold_vert_cache.get(&gid) {
            return Some(*v);
        }
        let fallback = self.bold_vface.as_ref().and_then(|face| {
            let vch = jpdict_core::util::japanese::to_vertical_glyph(ch);
            if vch != ch {
                face.glyph_index(vch).map(|g| g.0)
            } else {
                None
            }
        });
        let vert = self
            .bold_vface
            .as_ref()
            .and_then(|face| gsub_vert_glyph(face, gid))
            .or(fallback)
            .unwrap_or(gid);
        self.bold_vert_cache.insert(gid, vert);
        Some(vert)
    }

}

/// GSUB `vert`/`vrt2` single substitution for a glyph, if the font has one.
/// Both features carry the same lookups in this font; the union of their
/// single-substitution subtables is applied.
fn gsub_vert_glyph(face: &ttf_parser::Face, gid: u16) -> Option<u16> {
    use ttf_parser::gsub::{SingleSubstitution, SubstitutionSubtable};

    let gsub = face.tables().gsub?;
    let vert = ttf_parser::Tag::from_bytes(b"vert");
    let vrt2 = ttf_parser::Tag::from_bytes(b"vrt2");
    for fi in 0..gsub.features.len() {
        let Some(feature) = gsub.features.get(fi) else { continue };
        if feature.tag != vert && feature.tag != vrt2 {
            continue;
        }
        for li in feature.lookup_indices {
            let Some(lookup) = gsub.lookups.get(li) else { continue };
            for sub in lookup
                .subtables
                .into_iter::<SubstitutionSubtable>()
            {
                if let SubstitutionSubtable::Single(single) = sub {
                    match single {
                        SingleSubstitution::Format1 { coverage, delta } => {
                            if coverage.get(ttf_parser::GlyphId(gid)).is_some() {
                                return Some(gid.wrapping_add(delta as u16));
                            }
                        }
                        SingleSubstitution::Format2 {
                            coverage,
                            substitutes,
                        } => {
                            if let Some(idx) = coverage.get(ttf_parser::GlyphId(gid)) {
                                if let Some(sub) = substitutes.get(idx) {
                                    return Some(sub.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Skia's fake-bold stroke is ~1/24 of the text size end to end, so the
/// coverage grows by about 1/48 of the size per side.
fn fake_bold_radius(px: u32) -> i32 {
    ((px as f32 / 48.0).round() as i32).max(1)
}

/// Grow a glyph's coverage by `radius` pixels in every direction — the
/// bitmap equivalent of `paint.isFakeBoldText`, used for highlighted glyphs
/// when no bold companion face is bundled.
fn embolden(coverage: &[u8], w: u32, h: u32, radius: i32) -> Vec<u8> {
    let (w, h) = (w as i32, h as i32);
    if w <= 0 || h <= 0 || radius <= 0 {
        return coverage.to_vec();
    }
    let src = |x: i32, y: i32| -> u8 {
        if x < 0 || y < 0 || x >= w || y >= h {
            0
        } else {
            coverage[(y * w + x) as usize]
        }
    };
    let mut out = vec![0u8; coverage.len()];
    for y in 0..h {
        for x in 0..w {
            let mut best = 0u8;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    best = best.max(src(x + dx, y + dy));
                }
            }
            out[(y * w + x) as usize] = best;
        }
    }
    out
}

/// Grow a glyph's coverage without clipping it: pad the ink box by `radius`
/// first so the dilation has room, then dilate into the padding. The caller
/// must offset the placement by `-radius` to keep the ink aligned.
fn synthetic_bold(coverage: &[u8], w: u32, h: u32, radius: i32) -> (Vec<u8>, u32, u32) {
    let r = radius.max(0) as u32;
    if w == 0 || h == 0 || r == 0 {
        return (coverage.to_vec(), w, h);
    }
    let (bw, bh) = (w + 2 * r, h + 2 * r);
    let mut padded = vec![0u8; (bw * bh) as usize];
    for y in 0..h {
        for x in 0..w {
            padded[((y + r) * bw + (x + r)) as usize] = coverage[(y * w + x) as usize];
        }
    }
    (embolden(&padded, bw, bh, radius), bw, bh)
}

/// A rasterized glyph ready to draw: the ink bitmap plus the font metrics
/// needed to place it on its natural baseline.
#[derive(Clone)]
struct RasterGlyph {
    handle: ImageHandle,
    w: u32,
    h: u32,
    xmin: i32,
    ymin: i32,
}

/// Look up or rasterize a glyph, returning its ink bitmap and metrics.
/// Returns None if no glyph cache is available.
fn draw_glyph(
    cache: &RefCell<GlyphCache>,
    ch: char,
    vertical: bool,
    px_size: u32,
    highlighted: bool,
) -> Option<RasterGlyph> {
    cache.borrow_mut().glyph_for(ch, vertical, px_size, highlighted)
}

/// Screen-space top-left of a glyph's ink bitmap inside its char box,
/// mirroring mobile LineOverlayView: along the reading axis the glyph's own
/// ink centre sits on the box centre; across it, the reference glyph (`あ`)
/// ink centre does, so punctuation rides its natural baseline instead of
/// being centred in the em box. Metrics are `(w, h, xmin, ymin)` with
/// fontdue's y-up `ymin` (the ink bottom above the baseline), in pixels.
fn glyph_ink_origin(
    is_vertical: bool,
    cx: f32,
    cy: f32,
    glyph: (f32, f32, f32, f32),
    reference: (f32, f32, f32, f32),
) -> (f32, f32) {
    let (gw, gh, gxmin, gymin) = glyph;
    let (rw_ref, rh_ref, rxmin_ref, rymin_ref) = reference;
    if is_vertical {
        // Reading top→bottom: centre the char's own ink; x follows the
        // reference so the column stays optically centred.
        (
            cx - (rxmin_ref + rw_ref / 2.0) + gxmin,
            cy - gh / 2.0,
        )
    } else {
        // Reading left→right: centre the char's own ink; y sits on the
        // reference's ink centre (i.e. natural baseline relative to `あ`).
        (
            cx - gw / 2.0,
            cy + (rymin_ref + rh_ref / 2.0) - (gymin + gh),
        )
    }
}

/// Clockwise radians to rotate a line's glyphs by, about their char-box
/// centres (mobile `LineOverlayView.tiltDeg`). `RotatedBox.angle` is the
/// frame's x-axis angle from +x, and that already equals mobile's `tiltDeg`
/// in both orientations: for a horizontal frame the x axis is the reading
/// axis, and for a vertical frame `atan2(-y.x, y.y)` on the rotated y axis
/// reduces to the same x-axis angle. Adding 90° for vertical frames (as if
/// the angle were the long axis) turned a −1.1° column into ~89° sideways
/// glyphs. Unrotated lines draw exactly as before.
fn glyph_tilt(quad: Option<&RotatedBox>) -> f32 {
    let Some(q) = quad.filter(|q| q.is_rotated()) else {
        return 0.0;
    };
    q.angle
}

/// Mobile `LineResult.glyphSizePx()`: the source-pixel size the overlay
/// measures a line's glyphs against. The default path is the tallest char
/// box — the detector box height for a horizontal line, the 1em cell for a
/// vertical one. A rotated line measures its upright frame's **cross axis**
/// instead (`isVertical ? w : h`, matching mobile's `cropW`/`cropH`), because
/// its char boxes are AABBs of rotated cells and would oversize the glyphs.
/// Zero means the line has no measurable box: mobile skips it.
fn line_glyph_px(line: &LineResult, quad: Option<&RotatedBox>) -> f32 {
    if let Some(q) = quad.filter(|q| q.is_rotated()) {
        return if q.is_vertical() { q.w } else { q.h }.max(1.0);
    }
    line.char_boxes.iter().map(|b| b.h as f32).fold(0.0, f32::max)
}

/// Content scale for sizing overlay work: `None` until the image size is
/// known (`img_w`/`img_h` start at 1). A scale computed from the default 1
/// is `min(window_w, window_h)` — ~960 instead of ~1 — which sized pre-warm
/// glyphs at thousands of px and made `embolden` take seconds per line.
fn content_scale(window_w: f32, window_h: f32, img_w: u32, img_h: u32, zoom: f32) -> Option<f32> {
    if img_w <= 1 || img_h <= 1 {
        return None;
    }
    Some(f32::min(window_w / img_w as f32, window_h / img_h as f32) * zoom)
}

/// The font pixel size a line's glyphs are rasterized at: mobile's
/// `fixedSize * 0.90`, in source pixels, scaled to screen pixels by the
/// content transform. Zero (no measurable box) means the line draws nothing.
fn line_text_px(line: &LineResult, quad: Option<&RotatedBox>, total_scale: f32) -> f32 {
    line_glyph_px(line, quad) * TEXT_SIZE_RATIO * total_scale
}

/// Fillet radius for one polygon corner: the requested radius, clamped so
/// it cannot eat more than 45% of either adjacent edge (Android's
/// `CornerPathEffect` degrades the same way on tiny boxes).
fn fillet_radius(prev: (f32, f32), corner: (f32, f32), next: (f32, f32), radius: f32) -> f32 {
    let edge = |a: (f32, f32), b: (f32, f32)| ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
    radius
        .min(edge(prev, corner) * 0.45)
        .min(edge(corner, next) * 0.45)
        .max(0.0)
}

/// A closed polygon with every corner rounded by `radius` — the iced
/// equivalent of Android's `CornerPathEffect` on a rotated quad's path.
fn rounded_polygon(points: &[(f32, f32)], radius: f32) -> CanvasPath {
    CanvasPath::new(|p| {
        let n = points.len();
        if n < 3 {
            let Some(&(x, y)) = points.first() else { return };
            p.move_to(Point::new(x, y));
            for &(x, y) in &points[1..] {
                p.line_to(Point::new(x, y));
            }
            return;
        }
        // Start mid-edge so the first corner gets its fillet too.
        let start = (
            (points[n - 1].0 + points[0].0) / 2.0,
            (points[n - 1].1 + points[0].1) / 2.0,
        );
        p.move_to(Point::new(start.0, start.1));
        for i in 0..n {
            let corner = points[i];
            let next = points[(i + 1) % n];
            let prev = points[(i + n - 1) % n];
            p.arc_to(
                Point::new(corner.0, corner.1),
                Point::new(next.0, next.1),
                fillet_radius(prev, corner, next, radius),
            );
        }
        p.close();
    })
}

/// Mobile chip rule (`OcrOverlayView` neighbour and alternatives panels):
/// the chips take the vertical presentation forms only in landscape, where
/// they run down the side of the screen; portrait chips stay horizontal.
fn chip_text(text: &str, is_landscape: bool) -> String {
    if is_landscape {
        text.chars().map(to_vertical_glyph).collect()
    } else {
        text.to_string()
    }
}

/// Mobile `updateCursor` geometry: the cursor box is the char box inflated
/// by 2dp per side, in the same (content-transform) space as the boxes.
fn cursor_rect(box_pt: Point, box_size: Size, total_scale: f32) -> (Point, Size) {
    let pad = CURSOR_PAD * total_scale;
    (
        Point::new(box_pt.x - pad, box_pt.y - pad),
        Size::new(box_size.width + 2.0 * pad, box_size.height + 2.0 * pad),
    )
}

/// Mobile `LineOverlayView` per-glyph shrink-to-box: the glyph is measured
/// along the reading axis (height for a vertical line, width for a
/// horizontal one) against its own char box ×0.92 — doubled for halfwidth
/// ink, whose true advance is 0.5em but whose face may overflow it — and
/// scaled about the box centre when it overflows. One factor per glyph;
/// there is no line-wide cross-axis cap.
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

// ---------------------------------------------------------------------------
// Blank gaps — mobile `BlankGaps` / `GapDetector` (#44 Feature 2)
// ---------------------------------------------------------------------------

/// Mobile `GapDetector.DEFAULT_VERTICAL_RATIO`: measured recall 1.00 and no
/// false positives on the vertical bench; horizontal lines are never
/// eligible (`BlankGaps.apply` returns early on them).
const BLANK_GAP_RATIO: f32 = 1.6;

/// Mobile `medianOf`: even counts average the two middles.
fn median_of(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

/// Mobile `GapDetector.detect` on the char-box geometry: character indices
/// where the spacing to the next character is at least [BLANK_GAP_RATIO]×
/// the median spacing of the line. Vertical lines only.
fn blank_gap_positions(line: &LineResult) -> Vec<usize> {
    if !line.is_vertical {
        return Vec::new();
    }
    let n = line.text.chars().count();
    if n < 2 || line.char_boxes.len() < n {
        return Vec::new();
    }
    let centres: Vec<f32> = line
        .char_boxes
        .iter()
        .take(n)
        .map(|b| b.y as f32 + b.h as f32 / 2.0)
        .collect();
    let spacings: Vec<f32> = centres
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .collect();
    let pitch = median_of(&mut spacings.clone());
    if pitch <= 0.0 {
        return Vec::new();
    }
    spacings
        .iter()
        .enumerate()
        .filter(|(_, s)| **s / pitch >= BLANK_GAP_RATIO)
        .map(|(k, _)| k + 1)
        .collect()
}

/// Mobile `interpolateGapBox`: a placeholder box centred between the two
/// neighbours it was dropped from, sized as the mean of their extents along
/// the reading axis.
fn interpolate_gap_box(boxes: &[BoundingBox], index: usize, is_vertical: bool) -> BoundingBox {
    let before = index.checked_sub(1).and_then(|i| boxes.get(i));
    let after = boxes.get(index);
    let (Some(before), Some(after)) = (before, after) else {
        return before
            .or(after)
            .cloned()
            .unwrap_or_else(|| BoundingBox::new(0, 0, 0, 0, 1.0));
    };
    // Integer maths, matching the mobile `JpDictRect` arithmetic exactly.
    let (bl, bt, br, bb) = (before.left(), before.top(), before.right(), before.bottom());
    let (al, at, ar, ab) = (after.left(), after.top(), after.right(), after.bottom());
    if is_vertical {
        let centre_y = ((bt + bb) / 2 + (at + ab) / 2) / 2;
        let height = ((bb - bt + (ab - at)) / 2).max(1);
        BoundingBox::new(
            bl.min(al),
            centre_y - height / 2,
            br.max(ar) - bl.min(al),
            height,
            before.confidence,
        )
    } else {
        let centre_x = ((bl + br) / 2 + (al + ar) / 2) / 2;
        let width = ((br - bl + (ar - al)) / 2).max(1);
        BoundingBox::new(
            centre_x - width / 2,
            bt.min(at),
            width,
            bb.max(ab) - bt.min(at),
            before.confidence,
        )
    }
}

/// Mobile `LineResult.withGapCharAt`: insert the placeholder at `index`,
/// growing text, char boxes and alternatives together so every parallel list
/// still describes the same characters at the same indices. PC lines carry
/// no CTC columns or overrides, so those mobile-list steps have no analogue.
fn with_gap_char(line: &LineResult, index: usize) -> LineResult {
    let chars: Vec<char> = line.text.chars().collect();
    if index > chars.len() {
        return line.clone();
    }
    let mut text: String = chars[..index].iter().collect();
    text.push(GAP_CHAR);
    text.extend(chars[index..].iter());

    let char_boxes = if line.char_boxes.len() == chars.len() && !line.char_boxes.is_empty() {
        let mut out = line.char_boxes.clone();
        out.insert(
            index,
            interpolate_gap_box(&line.char_boxes, index, line.is_vertical),
        );
        out
    } else {
        line.char_boxes.clone()
    };
    let alternatives = if line.alternatives.len() == chars.len() && !line.alternatives.is_empty() {
        let mut out = line.alternatives.clone();
        out.insert(index, vec![(GAP_CHAR, 0.0)]);
        out
    } else {
        line.alternatives.clone()
    };
    LineResult {
        text,
        char_boxes,
        alternatives,
        is_vertical: line.is_vertical,
        ..line.clone()
    }
}

/// Mobile `BlankGaps.apply`: materialise every measured gap as a placeholder.
/// Vertical lines only, idempotent, insertions right-to-left so the
/// detector's indices (computed against the original text) stay valid.
fn apply_blank_gaps(line: &LineResult) -> LineResult {
    if !line.is_vertical || line.text.contains(GAP_CHAR) {
        return line.clone();
    }
    let gaps = blank_gap_positions(line);
    if gaps.is_empty() {
        return line.clone();
    }
    let mut out = line.clone();
    for &index in gaps.iter().rev() {
        out = with_gap_char(&out, index);
    }
    out
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PanState {
    pub is_panning: bool,
    pub start_x: f32,
    pub start_y: f32,
    pub trans_x_at_start: f32,
    pub trans_y_at_start: f32,
    /// Whether the current interaction is a tap (short press without movement).
    /// Set to true on press, moved to false once movement exceeds TAP_THRESHOLD.
    pub is_tap: bool,
    /// Last cursor position during pan, used to compute incremental deltas.
    pub last_x: f32,
    pub last_y: f32,
    /// Accumulated movement since press. Used for dead zone:
    /// drag only starts after this exceeds DRAG_DEAD_ZONE.
    pub accumulated_dx: f32,
    pub accumulated_dy: f32,
    /// Whether the drag dead zone has been exceeded.
    pub drag_started: bool,
    /// Pinch zoom: the ID and position of the first finger.
    pub pinch_finger1: Option<(touch::Finger, Point)>,
    /// Pinch zoom: the ID and position of the second finger.
    pub pinch_finger2: Option<(touch::Finger, Point)>,
    /// Pinch zoom: previous frame's distance between fingers.
    pub pinch_prev_dist: f32,
    /// Pinch zoom: previous frame's midpoint X.
    pub pinch_prev_focus_x: f32,
    /// Pinch zoom: previous frame's midpoint Y.
    pub pinch_prev_focus_y: f32,
}

/// OverlayProgram
// OverlayProgram
// ---------------------------------------------------------------------------


#[derive(Clone)]
pub struct OverlayProgram {
    pub annotations: Rc<Vec<DetectedAnnotation>>,
    /// None when no CJK font was found: boxes still draw, glyphs are skipped
    /// (mobile falls back to the platform face rather than crashing).
    pub glyph_cache: Option<Rc<RefCell<GlyphCache>>>,
    pub img_w: u32,
    pub img_h: u32,
    /// The screenshot image to draw on the canvas.
    pub image: Option<ImageHandle>,
    /// Whether the panel is visible.
    pub panel_visible: bool,
    /// Which side the panel is on.
    pub panel_on_right: bool,
    /// Fixed width of the dictionary panel in pixels.
    pub dict_width: f32,
    /// Current cursor position (line_idx, char_idx) for keyboard navigation.
    pub cursor_pos: Option<(usize, usize)>,
    /// Current zoom scale.
    pub current_scale: f32,
    /// Current translation X.
    pub current_trans_x: f32,
    /// Current translation Y.
    pub current_trans_y: f32,
    /// Whether this canvas draws annotations (true) or just the image (false).
    pub draw_annotations: bool,
    /// Whether this canvas instance should handle pan/zoom events.
    /// The image canvas (behind the annotation canvas) must NOT handle them,
    /// or pan/zoom would be applied twice.
    pub handle_pan_zoom: bool,
    /// Coordinates of characters to highlight in yellow (matched word).
    pub highlighted_coords: Vec<(usize, usize)>,
    // Debug-overlay-only data (navigation graph arrows), read under
    // `#[cfg(debug_assertions)]`.
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub nav_edges_initial: Option<Vec<[usize; 4]>>,
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub nav_edges_final: Option<Vec<[usize; 4]>>,
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub nav_centers: Vec<(f32, f32)>,
}

impl OverlayProgram {
    /// Compute the base scale (fit image to bounds) and offset (center in bounds).
    fn base_transform(&self, bounds: Rectangle) -> (f32, f32, f32) {
        let img_w_f = self.img_w as f32;
        let img_h_f = self.img_h as f32;
        let base_scale = f32::min(bounds.width / img_w_f, bounds.height / img_h_f);
        let offset_x = (bounds.width - img_w_f * base_scale) / 2.0;
        let offset_y = (bounds.height - img_h_f * base_scale) / 2.0;
        (base_scale, offset_x, offset_y)
    }

    /// Transform a screen-space point back to image-space.
    fn screen_to_image(&self, bounds: Rectangle, screen_x: f32, screen_y: f32) -> (f32, f32) {
        let (base_scale, offset_x, offset_y) = self.base_transform(bounds);
        let total_scale = base_scale * self.current_scale;
        (
            (screen_x - offset_x - self.current_trans_x) / total_scale,
            (screen_y - offset_y - self.current_trans_y) / total_scale,
        )
    }

    /// Check whether a screen-space point is over the panel area.
    /// The panel is positioned at the correct edge of the screen.
    fn is_over_panel(&self, bounds: Rectangle, screen_x: f32, _screen_y: f32) -> bool {
        if !self.panel_visible {
            return false;
        }
        let panel_w = self.dict_width + 42.0 + 42.0 + 6.0; // dict + neighbors + alt + spacing
        if self.panel_on_right {
            screen_x >= bounds.width - panel_w
        } else {
            screen_x <= panel_w
        }
    }

    /// Check whether a screen-space point hits any character bounding box.
    /// Returns `Some((line_idx, char_idx))` if a hit is found.
    fn hit_test(&self, bounds: Rectangle, screen_x: f32, screen_y: f32) -> Option<(usize, usize)> {
        let (img_x, img_y) = self.screen_to_image(bounds, screen_x, screen_y);
        for (line_idx, annotation) in self.annotations.iter().enumerate() {
            if let Some(line) = &annotation.line {
                for (char_idx, char_box) in line.char_boxes.iter().enumerate() {
                    if img_x >= char_box.left() as f32
                        && img_x <= char_box.right() as f32
                        && img_y >= char_box.top() as f32
                        && img_y <= char_box.bottom() as f32
                    {
                        return Some((line_idx, char_idx));
                    }
                }
            }
        }
        None
    }
}

impl canvas::Program<Message, Theme, Renderer> for OverlayProgram {
    type State = PanState;

    fn mouse_interaction(
        &self,
        state: &Self::State,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if state.drag_started {
            mouse::Interaction::Grabbing
        } else {
            mouse::Interaction::Idle
        }
    }

    fn update(
        &self,
        state: &mut Self::State,
        event: &iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<iced::widget::Action<Message>> {
        // The image canvas (behind the annotation canvas) must NOT handle pan/zoom,
        // or it would be applied twice (both canvases receive the same events).
        if !self.handle_pan_zoom {
            return None;
        }

        // Check if cursor is over the panel area — if so, don't handle pan/zoom
        let over_panel = self.panel_visible
            && cursor.position().map_or(false, |p| self.is_over_panel(bounds, p.x, p.y));

        match event {
            // --- Scroll wheel zoom (only when NOT over panel) ---
            iced::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                if over_panel { return None; }
                if let Some(cursor_position) = cursor.position_in(bounds) {
                    let (dy, _dx) = match delta {
                        mouse::ScrollDelta::Lines { y, x } => (*y, *x),
                        mouse::ScrollDelta::Pixels { y, x } => (*y / 100.0, *x / 100.0),
                    };
                    if dy != 0.0 {
                        return Some(iced::widget::Action::publish(
                            Message::ZoomOnCursor {
                                delta: dy,
                                cursor_x: cursor_position.x,
                                cursor_y: cursor_position.y,
                            },
                        ));
                    }
                }
            }

            // --- Mouse: left button pressed (pan start or tap) ---
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if over_panel { return None; } // Let panel handle it
                if let Some(cursor_position) = cursor.position_in(bounds) {
                    state.is_panning = true;
                    state.start_x = cursor_position.x;
                    state.start_y = cursor_position.y;
                    state.last_x = cursor_position.x;
                    state.last_y = cursor_position.y;
                    state.trans_x_at_start = self.current_trans_x;
                    state.trans_y_at_start = self.current_trans_y;
                    state.is_tap = true;
                    state.accumulated_dx = 0.0;
                    state.accumulated_dy = 0.0;
                    state.drag_started = false;
                }
                return None;
            }

            // --- Mouse: cursor moved (pan with dead zone) ---
            iced::Event::Mouse(mouse::Event::CursorMoved { position }) => {
                // If cursor moved over panel during pan, cancel the pan
                if over_panel && state.is_panning {
                    state.is_panning = false;
                    state.is_tap = false;
                    state.drag_started = false;
                    state.accumulated_dx = 0.0;
                    state.accumulated_dy = 0.0;
                    return None;
                }
                if state.is_panning {
                    let dx = position.x - state.last_x;
                    let dy = position.y - state.last_y;
                    state.last_x = position.x;
                    state.last_y = position.y;

                    state.accumulated_dx += dx;
                    state.accumulated_dy += dy;
                    let total = (state.accumulated_dx.powi(2) + state.accumulated_dy.powi(2)).sqrt();

                    if !state.drag_started && total > DRAG_DEAD_ZONE {
                        state.drag_started = true;
                        state.is_tap = false;
                    }

                    if state.drag_started && (dx != 0.0 || dy != 0.0) {
                        return Some(iced::widget::Action::publish(Message::PanDelta { dx, dy }));
                    }
                }
            }

            // --- Mouse: left button released (pan end or tap) ---
            iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                // If over panel, don't process — let the panel handle it
                if over_panel {
                    state.is_panning = false;
                    state.is_tap = false;
                    state.drag_started = false;
                    return None;
                }
                let was_tap = state.is_tap;
                state.is_panning = false;
                state.is_tap = false;
                state.drag_started = false;
                state.accumulated_dx = 0.0;
                state.accumulated_dy = 0.0;

                if was_tap {
                    if let Some(cursor_position) = cursor.position_in(bounds) {
                        if let Some((li, ci)) = self.hit_test(bounds, cursor_position.x, cursor_position.y) {
                            return Some(iced::widget::Action::publish(Message::SelectCharacter(li, ci)));
                        }
                        // Left-click on blank area: do nothing (don't close the app)
                    }
                }
            }

            // --- Touch: finger pressed ---
            iced::Event::Touch(touch::Event::FingerPressed { id, position }) => {
                let already_tracking = state.pinch_finger1.is_some() || state.pinch_finger2.is_some();

                if !already_tracking && !over_panel {
                    // First finger — start pan
                    state.is_panning = true;
                    state.pinch_finger1 = Some((*id, *position));
                    state.start_x = position.x;
                    state.start_y = position.y;
                    state.last_x = position.x;
                    state.last_y = position.y;
                    state.trans_x_at_start = self.current_trans_x;
                    state.trans_y_at_start = self.current_trans_y;
                    state.is_tap = true;
                    state.accumulated_dx = 0.0;
                    state.accumulated_dy = 0.0;
                    state.drag_started = false;
                    state.pinch_finger2 = None;
                    state.pinch_prev_dist = 0.0;
                } else if already_tracking && state.pinch_finger1.is_some() && state.pinch_finger2.is_none() {
                    // Second finger — switch to pinch zoom mode
                    state.pinch_finger2 = Some((*id, *position));
                    // Compute initial distance and midpoint
                    if let Some((_, f1_pos)) = state.pinch_finger1 {
                        let dx = position.x - f1_pos.x;
                        let dy = position.y - f1_pos.y;
                        state.pinch_prev_dist = (dx * dx + dy * dy).sqrt().max(10.0);
                        state.pinch_prev_focus_x = (f1_pos.x + position.x) / 2.0;
                        state.pinch_prev_focus_y = (f1_pos.y + position.y) / 2.0;
                    }
                    state.is_tap = false;
                    state.drag_started = true;
                }
                // Third and subsequent fingers — ignore entirely
                return None;
            }

            // --- Touch: finger moved (pan or pinch) ---
            iced::Event::Touch(touch::Event::FingerMoved { id, position }) => {
                if state.is_panning {
                    // Only update positions for our two tracked fingers — ignore any others
                    if let Some((fid, _)) = state.pinch_finger1 {
                        if fid == *id { state.pinch_finger1 = Some((*id, *position)); }
                    }
                    if let Some((fid, _)) = state.pinch_finger2 {
                        if fid == *id { state.pinch_finger2 = Some((*id, *position)); }
                    }

                    if state.pinch_finger2.is_some() {
                        // Pinch zoom: both fingers active
                        if let (Some((_, f1)), Some((_, f2))) = (state.pinch_finger1, state.pinch_finger2) {
                            let dx = f2.x - f1.x;
                            let dy = f2.y - f1.y;
                            let current_dist = (dx * dx + dy * dy).sqrt();
                            let scale_factor = if state.pinch_prev_dist > 0.0 { current_dist / state.pinch_prev_dist } else { 1.0 };
                            let focus_x = (f1.x + f2.x) / 2.0;
                            let focus_y = (f1.y + f2.y) / 2.0;
                            let prev_focus_x = state.pinch_prev_focus_x;
                            let prev_focus_y = state.pinch_prev_focus_y;

                            // Update prev for next frame
                            state.pinch_prev_dist = current_dist;
                            state.pinch_prev_focus_x = focus_x;
                            state.pinch_prev_focus_y = focus_y;

                            // Compute base_offset for the Y axis so the handler can
                            // correctly account for the image centering.
                            let img_h_f = self.img_h as f32;
                            let base_scale = f32::min(bounds.width / self.img_w as f32, bounds.height / img_h_f);
                            let base_offset_y = (bounds.height - img_h_f * base_scale) / 2.0;

                            return Some(iced::widget::Action::publish(
                                Message::PinchZoom {
                                    scale_factor,
                                    focus_x,
                                    focus_y,
                                    prev_focus_x,
                                    prev_focus_y,
                                    base_offset_y,
                                },
                            ));
                        }
                    } else {
                        // Single finger pan with dead zone
                        let dx = position.x - state.last_x;
                        let dy = position.y - state.last_y;
                        state.last_x = position.x;
                        state.last_y = position.y;

                        state.accumulated_dx += dx;
                        state.accumulated_dy += dy;
                        let total = (state.accumulated_dx.powi(2) + state.accumulated_dy.powi(2)).sqrt();

                        if !state.drag_started && total > DRAG_DEAD_ZONE {
                            state.drag_started = true;
                            state.is_tap = false;
                        }

                        if state.drag_started && (dx != 0.0 || dy != 0.0) {
                            return Some(iced::widget::Action::publish(Message::PanDelta { dx, dy }));
                        }
                    }
                }
            }

            // --- Touch: finger lifted (pan end) — detect taps ---
            iced::Event::Touch(touch::Event::FingerLifted { id, position }) => {
                // Only process lifts for our two tracked fingers
                let tracked_f1 = state.pinch_finger1.map(|(fid, _)| fid) == Some(*id);
                let tracked_f2 = state.pinch_finger2.map(|(fid, _)| fid) == Some(*id);

                if !tracked_f1 && !tracked_f2 {
                    // Unknown finger (3rd+) — ignore entirely
                    return None;
                }

                let was_pinch = state.pinch_finger2.is_some();

                // Remove the lifted finger from tracking
                if tracked_f1 { state.pinch_finger1 = None; }
                if tracked_f2 { state.pinch_finger2 = None; }

                // If either tracked finger lifted, end the gesture entirely.
                // Don't try to continue pinch/pan with remaining fingers — this
                // prevents flickering when 3+ fingers are involved.
                let was_tap = state.is_tap;
                state.is_panning = false;
                state.is_tap = false;
                state.drag_started = false;
                state.accumulated_dx = 0.0;
                state.accumulated_dy = 0.0;
                state.pinch_prev_dist = 0.0;
                state.pinch_prev_focus_x = 0.0;
                state.pinch_prev_focus_y = 0.0;

                if was_pinch {
                    return Some(iced::widget::Action::publish(Message::PinchEnd));
                }

                if was_tap {
                    return Some(iced::widget::Action::publish(
                        match self.hit_test(bounds, position.x, position.y) {
                            Some((li, ci)) => Message::SelectCharacter(li, ci),
                            None => Message::Back,
                        },
                    ));
                }
            }

            _ => {}
        }
        None
    }

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let _t_draw = std::time::Instant::now();
        let mut frame = Frame::new(renderer, bounds.size());
        let (base_scale, base_offset_x, base_offset_y) = self.base_transform(bounds);
        let total_scale = base_scale * self.current_scale;
        let total_offset_x = (base_offset_x + self.current_trans_x).round();
        let total_offset_y = (base_offset_y + self.current_trans_y).round();

        // Fill the image canvas with black so any gap around the scaled image
        // (due to aspect-ratio mismatch) is black, not the window's white background.
        // Only do this on the actual image canvas (the one that has an image handle),
        // not on the annotation canvas during zoom-out blanking.
        if self.image.is_some() {
            frame.fill_rectangle(Point::ORIGIN, bounds.size(), Color::BLACK);
        }

        // Warm up the font atlas on the very first frame by rendering an off-screen
        // character. This populates Iced's internal cosmic-text glyph cache *before*
        // OCR results arrive and trigger a burst of fill_text calls. Without this,
        // the first frame of annotation rendering can corrupt the atlas (sporadic
        // wrong-size glyphs, artifacts). Zooming in/out fixes it because it triggers
                // When this is the image-only canvas (draw_annotations == false),
        // draw the screenshot as the bottom layer.
        if !self.draw_annotations {
            if let Some(image) = self.image.as_ref() {
                let img_w = self.img_w as f32;
                let img_h = self.img_h as f32;
                let dest_size = Size::new(img_w * total_scale, img_h * total_scale);
                let dest_pos = Point::new(total_offset_x, total_offset_y);
                frame.draw_image(
                    Rectangle::new(dest_pos, dest_size),
                    image,
                );
                // Mobile #64 backdrop: a dark scrim over the dimmed
                // screenshot. Desktop has no status strip, so the scrim is
                // flat over the whole overlay (SOLID).
                frame.fill_rectangle(
                    Point::ORIGIN,
                    bounds.size(),
                    Color::from_rgba(0.0, 0.0, 0.0, BACKDROP_SCRIM),
                );
            }
            return vec![frame.into_geometry()];
        }

        let transform = |bbox: &BoundingBox| -> (Point, Size) {
            (Point::new(
                bbox.x as f32 * total_scale + total_offset_x,
                bbox.y as f32 * total_scale + total_offset_y,
            ), Size::new(
                bbox.w as f32 * total_scale,
                bbox.h as f32 * total_scale,
            ))
        };

        if self.draw_annotations {
            // Pre-compute visible screen bounds in image coordinates to skip off-screen annotations
            let inv_scale = if total_scale > 0.0 { 1.0 / total_scale } else { 1.0 };
            let vis_left = (-total_offset_x) * inv_scale - 100.0;
            let vis_top = (-total_offset_y) * inv_scale - 100.0;
            let vis_right = (bounds.width - total_offset_x) * inv_scale + 100.0;
            let vis_bottom = (bounds.height - total_offset_y) * inv_scale + 100.0;

            for (line_idx, annotation) in self.annotations.iter().enumerate() {
                let bbox = &annotation.bbox;
                // Skip annotations entirely off-screen
                let bx2 = (bbox.x + bbox.w) as f32;
                let by2 = (bbox.y + bbox.h) as f32;
                let bx1 = bbox.x as f32;
                let by1 = bbox.y as f32;
                if bx2 < vis_left || bx1 > vis_right || by2 < vis_top || by1 > vis_bottom {
                    continue;
                }

                // Mobile `borderDrawable` / `QuadBorderView`: argb(100,0,0,0)
                // with the corners rounded at 4 source px.
                let fill_color = Color::from_rgba(0.0, 0.0, 0.0, BOX_FILL_ALPHA);
                if let Some(q) = annotation.quad {
                    // Angled line: draw the rotated quad (min-area rect)
                    // instead of its axis-aligned AABB. The rounded corners
                    // are the iced equivalent of Android's CornerPathEffect;
                    // no view padding is needed because the canvas does not
                    // clip the path.
                    let (ux, uy) = (q.angle.cos(), q.angle.sin());
                    let (vx, vy) = (-uy, ux);
                    let hw = q.w / 2.0;
                    let hh = q.h / 2.0;
                    let corners = [
                        (q.cx + ux * hw + vx * hh, q.cy + uy * hw + vy * hh),
                        (q.cx - ux * hw + vx * hh, q.cy - uy * hw + vy * hh),
                        (q.cx - ux * hw - vx * hh, q.cy - uy * hw - vy * hh),
                        (q.cx + ux * hw - vx * hh, q.cy + uy * hw - vy * hh),
                    ];
                    let screen: Vec<(f32, f32)> = corners
                        .iter()
                        .map(|&(cx, cy)| {
                            (cx * total_scale + total_offset_x, cy * total_scale + total_offset_y)
                        })
                        .collect();
                    frame.fill(
                        &rounded_polygon(&screen, BOX_CORNER_RADIUS * total_scale),
                        fill_color,
                    );
                } else {
                    let (pt, sz) = transform(bbox);
                    frame.fill(
                        &CanvasPath::rounded_rectangle(
                            pt,
                            sz,
                            (BOX_CORNER_RADIUS * total_scale).into(),
                        ),
                        fill_color,
                    );
                }

                if let Some(line) = &annotation.line {
                    // No font: the box above is all this line gets.
                    let Some(cache) = self.glyph_cache.as_ref() else {
                        continue;
                    };
                    // Mobile `LineOverlayView`: one text size per line, the
                    // line's own box height ×0.90 (the upright frame's cross
                    // axis for rotated lines) — never a measured pitch, and
                    // never a line-wide cross-axis cap. Horizontal glyphs use
                    // `あ` as the ink reference; vertical ones centre their own
                    // ink (see below). A line with no measurable box draws no
                    // glyphs, exactly as mobile returns early on `fixedSize == 0`.
                    let text_px = line_text_px(line, annotation.quad.as_ref(), total_scale);
                    if text_px <= 0.0 || line.text.is_empty() || line.char_boxes.is_empty() {
                        continue;
                    }
                    let em_px = text_px.round().clamp(1.0, 1024.0) as u32;
                    let ref_ink = draw_glyph(cache, 'あ', line.is_vertical, em_px, false)
                        .map(|g| (g.w as f32, g.h as f32, g.xmin as f32, g.ymin as f32));

                    for (i, char_box) in line.char_boxes.iter().enumerate() {
                        let (pt_c, sz_c) = transform(char_box);
                        let (cx, cy) = (pt_c.x + sz_c.width / 2.0, pt_c.y + sz_c.height / 2.0);

                        // Mobile cursor chrome: a white rounded outline, no
                        // fill, inflated 2dp per side and scaling with the
                        // content transform like the boxes themselves.
                        if self.cursor_pos == Some((line_idx, i)) {
                            let (c_pt, c_sz) = cursor_rect(pt_c, sz_c, total_scale);
                            let cursor_path = CanvasPath::rounded_rectangle(
                                c_pt,
                                c_sz,
                                (CURSOR_RADIUS * total_scale).into(),
                            );
                            frame.stroke(
                                &cursor_path,
                                CanvasStroke::default()
                                    .with_color(Color::WHITE)
                                    .with_width(CURSOR_STROKE * total_scale),
                            );
                        }

                        let Some(ch) = line.text.chars().nth(i) else { continue };
                        // Mobile highlights the matched word only; the cursor
                        // is drawn as chrome and keeps the glyph pink.
                        let highlighted = self.highlighted_coords.contains(&(line_idx, i));
                        // GSUB `vert`/`vrt2` picks the vertical presentation
                        // glyph for vertical lines (Unicode form fallback for
                        // what the font does not cover).
                        let Some(g) = draw_glyph(cache, ch, line.is_vertical, em_px, highlighted) else { continue };
                        // Mobile skips a glyph with no ink (space, .notdef).
                        if g.w == 0 || g.h == 0 {
                            continue;
                        }
                        let (gw, gh) = (g.w as f32, g.h as f32);
                        let reference = ref_ink.unwrap_or((gw, gh, g.xmin as f32, g.ymin as f32));
                        let glyph_metrics = (gw, gh, g.xmin as f32, g.ymin as f32);
                        // Mobile centres the glyph's own ink on the char box
                        // along the reading axis; across it the reference `あ`
                        // anchors the baseline so punctuation keeps its
                        // position. Vertical text is no different: the font's
                        // GSUB `vert` form is centred, not laid out on vmtx.
                        let (dx, dy) = glyph_ink_origin(
                            line.is_vertical,
                            cx,
                            cy,
                            glyph_metrics,
                            reference,
                        );

                        // Mobile per-glyph shrink-to-box along the reading
                        // axis, then the halfwidth trim — both about the box
                        // centre.
                        let is_half = is_half_width(ch);
                        let fit = glyph_fit_scale(
                            line.is_vertical,
                            is_half,
                            (sz_c.width.max(1.0), sz_c.height.max(1.0)),
                            (gw, gh),
                        );
                        let draw_scale = fit * if is_half { ASCII_GLYPH_SCALE } else { 1.0f32 };
                        let off_x = (dx - cx) * draw_scale;
                        let off_y = (dy - cy) * draw_scale;
                        let draw_size = Size::new(gw * draw_scale, gh * draw_scale);

                        let tilt = glyph_tilt(annotation.quad.as_ref());
                        if tilt != 0.0 {
                            // Rotated line: draw the glyph at the line's tilt
                            // around the char box centre so it matches the
                            // source orientation (mobile `canvas.rotate`).
                            frame.with_save(|frame| {
                                frame.translate(iced::Vector::new(cx, cy));
                                frame.rotate(tilt);
                                frame.draw_image(
                                    Rectangle::new(Point::new(off_x, off_y), draw_size),
                                    &g.handle,
                                );
                            });
                        } else {
                            frame.draw_image(
                                Rectangle::new(Point::new(cx + off_x, cy + off_y), draw_size),
                                &g.handle,
                            );
                        }
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        // --- Debug: draw navigation graph arrows ---
        if !self.nav_centers.is_empty() {
            // Compute cursor global index for filtering
            let cursor_global = self.cursor_pos.and_then(|(li, ci)| {
                let mut g = 0;
                for (a_li, ann) in self.annotations.iter().enumerate() {
                    if a_li < li {
                        if let Some(l) = &ann.line {
                            g += l.text.chars().count();
                        }
                    } else {
                        return Some(g + ci);
                    }
                }
                None
            });
            let arrow_to_screen = |cx: f32, cy: f32| -> Point {
                let pt = transform(&BoundingBox::new(cx as i32, cy as i32, 1, 1, 0.0));
                Point::new(pt.0.x + pt.1.width / 2.0, pt.0.y + pt.1.height / 2.0)
            };
            let dir_color = |dir: usize| -> Color {
                match dir {
                    0 => Color::from_rgb(0.0, 1.0, 1.0),   // N = cyan
                    1 => Color::from_rgb(1.0, 0.0, 0.0),   // S = red
                    2 => Color::from_rgb(0.0, 1.0, 0.0),   // E = green
                    _ => Color::from_rgb(1.0, 1.0, 0.0),   // W = yellow
                }
            };
            let dir_controls = |from: Point, to: Point, dir: usize, bias_x: f32, bias_y: f32| -> (Point, Point) {
                let dx = to.x - from.x;
                let dy = to.y - from.y;
                let dist = (dx * dx + dy * dy).sqrt().max(20.0);
                let card_pull = dist * 0.3;
                match dir {
                    0 => (Point::new(from.x + bias_x, from.y - card_pull + bias_y), Point::new(to.x + bias_x * 0.3, to.y + dy * 0.2)),  // N
                    1 => (Point::new(from.x + bias_x, from.y + card_pull + bias_y), Point::new(to.x + bias_x * 0.3, to.y - dy * 0.2)),  // S
                    2 => (Point::new(from.x + card_pull + bias_x, from.y + bias_y), Point::new(to.x - dx * 0.2, to.y + bias_y * 0.3)),  // E
                    _ => (Point::new(from.x - card_pull + bias_x, from.y + bias_y), Point::new(to.x - dx * 0.2, to.y + bias_y * 0.3)),  // W
                }
            };
            // Draw a cubic bezier with arrowhead at target
            let draw_curve = |frame: &mut Frame, from: Point, to: Point, color: Color, dir: usize, bias_x: f32, bias_y: f32| {
                let (ctrl_a, ctrl_b) = dir_controls(from, to, dir, bias_x, bias_y);
                let mut pb = iced::widget::canvas::path::Builder::new();
                pb.move_to(from);
                pb.bezier_curve_to(ctrl_a, ctrl_b, to);
                let path = pb.build();
                frame.stroke(&path, CanvasStroke::default().with_color(color).with_width(1.5));

                // Arrowhead: tangent at endpoint = (to - ctrl_b) direction
                let dx = to.x - ctrl_b.x;
                let dy = to.y - ctrl_b.y;
                let len = (dx * dx + dy * dy).sqrt().max(1.0);
                let ux = dx / len;
                let uy = dy / len;
                let tip = 12.0;
                let left = Point::new(to.x - ux * tip * 0.866 + uy * tip * 0.5, to.y - uy * tip * 0.866 - ux * tip * 0.5);
                let right = Point::new(to.x - ux * tip * 0.866 - uy * tip * 0.5, to.y - uy * tip * 0.866 + ux * tip * 0.5);
                frame.stroke(&iced::widget::canvas::Path::line(to, left), CanvasStroke::default().with_color(color).with_width(1.5));
                frame.stroke(&iced::widget::canvas::Path::line(to, right), CanvasStroke::default().with_color(color).with_width(1.5));
            };

            // Draw initial edges (pre-connectivity, dimmer)
            if let Some(ref initial) = self.nav_edges_initial {
                for (i, edges) in initial.iter().enumerate() {
                    if i >= self.nav_centers.len() { break; }
                    for (d, &target) in edges.iter().enumerate() {
                        if target >= self.nav_centers.len() { continue; }
                        let related = cursor_global.map_or(false, |cg| i == cg || target == cg);
                        if !related { continue; }
                        let from = arrow_to_screen(self.nav_centers[i].0, self.nav_centers[i].1);
                        let to = arrow_to_screen(self.nav_centers[target].0, self.nav_centers[target].1);
                        let color = dir_color(d);
                        let alpha = if cursor_global == Some(i) { 0.5 } else { 0.25 };
                        let is_out = cursor_global == Some(i);
                        let bx = if d < 2 { if is_out { 10.0 } else { -10.0 } } else { 0.0 };
                        let by = if d >= 2 { if is_out { -10.0 } else { 10.0 } } else { 0.0 };
                        draw_curve(&mut frame, from, to, Color::from_rgba(color.r, color.g, color.b, alpha), d, bx, by);
                    }
                }
            }
            // Draw final edges (finalized) — drawn on top
            if let Some(ref final_edges) = self.nav_edges_final {
                for (i, edges) in final_edges.iter().enumerate() {
                    if i >= self.nav_centers.len() { break; }
                    for (d, &target) in edges.iter().enumerate() {
                        if target >= self.nav_centers.len() { continue; }
                        let related = cursor_global.map_or(false, |cg| i == cg || target == cg);
                        if !related { continue; }
                        let from = arrow_to_screen(self.nav_centers[i].0, self.nav_centers[i].1);
                        let to = arrow_to_screen(self.nav_centers[target].0, self.nav_centers[target].1);
                        let color = dir_color(d);
                        let alpha = if cursor_global == Some(i) { 1.0 } else { 0.4 };
                        let is_out = cursor_global == Some(i);
                        let bx = if d < 2 { if is_out { 10.0 } else { -10.0 } } else { 0.0 };
                        let by = if d >= 2 { if is_out { -10.0 } else { 10.0 } } else { 0.0 };
                        draw_curve(&mut frame, from, to, Color::from_rgba(color.r, color.g, color.b, alpha), d, bx, by);
                    }
                }
            }
        }

        // Print draw timing (only when >1ms to avoid idle-frame spam)
        let _t_draw_end = std::time::Instant::now();
        let elapsed = _t_draw_end.duration_since(_t_draw);
        if elapsed.as_micros() > 1000 {
            // eprintln!("[TIMING] draw: {}us", elapsed.as_micros());
        }

        vec![frame.into_geometry()]
    }
}

// ---------------------------------------------------------------------------
// OcrViewer
// ---------------------------------------------------------------------------

pub struct OcrViewer {
    pub image_handle: Option<iced::widget::image::Handle>,
    /// Raw PNG bytes of the original screenshot, used for cropping character previews.
    image_bytes: Option<Vec<u8>>,
    /// Decoded image, cached to avoid re-decoding PNG on every crop_character_image call.
    /// Decoded image, cached to avoid re-decoding PNG on every crop_character_image call.
    decoded_image: RefCell<Option<image::DynamicImage>>,
    pub img_w: u32,
    pub img_h: u32,
    /// Physical window dimensions. Used for computing font sizes in neighbor/alt panels.
    pub window_width: f32,
    pub window_height: f32,
    pub annotations: Rc<Vec<DetectedAnnotation>>,
    /// Cached synced annotations for overlay rendering. Same data as `annotations`
    /// but with text sync'd from active_line_results. Cheap Rc clone; only
    /// re-synced on text edits (set annotations_sync_dirty). Avoids cloning
    /// all 49 annotations every frame in view().
    synced_annotations: RefCell<Rc<Vec<DetectedAnnotation>>>,
    /// Set to true when a text edit requires re-syncing annotations with
    /// active_line_results. view() re-syncs and clears this flag.
    pub annotations_sync_dirty: Cell<bool>,
    /// Lines whose text has been user-edited (via alternative selection).
    /// `apply_ocr_batch` skips the overlay line-result update for these
    /// lines, preserving the user's edit against incoming OCR results.
    pub edited_lines: HashSet<usize>,
    /// Cached total_scale from last view() frame, used for glyph pre-warm
    last_total_scale: Cell<f32>,
    pub state: OcrOverlayState,
    pub selected_word: Option<SelectedWord>,
    pub alternatives_visible: bool,
    pub db: Option<Arc<DictionaryDatabase>>,
    pub deinflector: Option<Arc<Deinflector>>,
    /// The index of the character that should be scrolled into view in the neighbor panel.
    pub scroll_neighbor_to: Option<usize>,
    /// The index of the character that should be scrolled into view in the alt panel.
    pub scroll_alt_to: Option<usize>,
    /// Requested dictionary scroll delta (px), set by L1/R1 gamepad, D/F keyboard.
    pub dict_scroll_request: Option<f32>,
    /// When true, lookup is deferred until the held gamepad button is released.
    pub defer_lookup: bool,
    /// When true, the user is actively panning/zooming — annotation drawing is disabled.
    pub is_zooming: bool,
    /// Frames since last zoom/pan event — used to re-enable annotations after zoom ends.
    pub zoom_idle_frames: u32,
    /// Cached character preview image for the alternatives panel.
    /// Stores (line_idx, char_idx, handle) so we only regenerate when the selection changes.
    cached_preview: RefCell<Option<(usize, usize, iced::widget::image::Handle)>>,
    /// Fontdue-based glyph rasterization cache. Bypasses cosmic-text to avoid
    /// font atlas corruption triggered by rendering OCR text via Iced's pipeline.
    pub glyph_cache: Option<Rc<RefCell<GlyphCache>>>,
    /// Native screen width (physical pixels, detected at startup). Used to
    /// deduce the window system's native scale factor from the first resize.
    pub screen_physical_width: f32,
    /// Window system's native scale factor (e.g. 1.0, 2.0 on HiDPI).
    /// Deduced from the first WindowResized event: native_scale = physical / (logical * app_scale).
    /// 0.0 means "not yet determined".
    pub native_scale: f32,
    /// Dynamic UI scale factor for iced's built-in scale_factor.
    /// Recalculated from the physical window width after each resize:
    /// debounced_ui_scale = physical_width / 1280.0
    pub debounced_ui_scale: f32,
    /// Live detection tuning values, adjusted with the viewer's tune keys and
    /// sent to the OCR worker (`TuneCmd`). Shown in the HUD.
    pub det_thresh: f32,
    pub det_unclip: f32,
    /// Startup values, restored by the `R` reset key.
    pub det_thresh_default: f32,
    pub det_unclip_default: f32,
    /// Boxes in the last detection, shown in the HUD.
    pub det_box_count: usize,
    /// True while a live retune round-trip is in flight.
    pub det_busy: bool,
    /// HUD visibility, toggled at runtime with F1. Hidden by default so the
    /// screenshot viewer opens clean; the tuning keys still work while hidden.
    pub det_hud_visible: bool,
    /// #43/#86: whether the dictionary panel draws the pitch-accent line.
    /// Loaded once from [`jpdict_core::app_settings::AppSettings`] at startup, so a
    /// settings change applies to the next launch (same as the other
    /// Behaviour switches). Off by default, matching mobile.
    pub show_pitch: bool,
}

impl OcrViewer {
    /// Create a minimal viewer with no image, no db, no annotations.
    /// Everything is populated asynchronously via bootstrap events.
    /// `window_w` / `window_h` should be the native screen resolution (in physical pixels).
    /// `face` selects the overlay's bundled font (sans default, serif opt-in).
    pub fn new_empty(window_w: f32, window_h: f32, face: FontFace) -> Self {
        Self {
            image_handle: None,
            image_bytes: None,
            decoded_image: RefCell::new(None),
            img_w: 1,
            img_h: 1,
            window_width: window_w,
            window_height: window_h,
            annotations: Rc::new(Vec::new()),
            synced_annotations: RefCell::new(Rc::new(Vec::new())),
            annotations_sync_dirty: Cell::new(false),
            edited_lines: HashSet::new(),
            last_total_scale: Cell::new(1.0),
            state: OcrOverlayState::new(window_w, window_h),
            selected_word: None,
            alternatives_visible: false,
            db: None,
            deinflector: None,
            scroll_neighbor_to: None,
            scroll_alt_to: None,
            dict_scroll_request: None,
            defer_lookup: false,
            is_zooming: false,
            zoom_idle_frames: 0,
            cached_preview: RefCell::new(None),
            glyph_cache: GlyphCache::new_for(face),
            screen_physical_width: window_w,
            native_scale: 0.0,
            debounced_ui_scale: window_w / 1280.0,
            det_thresh: jpdict_core::ocr_engine::default_det_thresh(),
            det_unclip: jpdict_core::ocr_engine::default_det_unclip(),
            det_thresh_default: jpdict_core::ocr_engine::default_det_thresh(),
            det_unclip_default: jpdict_core::ocr_engine::default_det_unclip(),
            det_box_count: 0,
            det_busy: false,
            // The tuning HUD ships hidden (F1 reveals it); see `toggle_det_hud`.
            det_hud_visible: false,
            show_pitch: false,
        }
    }

    /// Set the screenshot image after it's loaded by the bootstrap thread.
    pub fn set_image(&mut self, handle: iced::widget::image::Handle, bytes: Vec<u8>, w: u32, h: u32) {
        self.image_handle = Some(handle);
        self.image_bytes = Some(bytes);
        self.img_w = w;
        self.img_h = h;
        self.state.img_w = w;
        self.state.img_h = h;
    }

    /// Crop the screenshot to show the given character with padding.
    /// Returns an image Handle for the cropped region.
    pub fn crop_character_image(&self, line_idx: usize, char_idx: usize) -> Option<iced::widget::image::Handle> {
        let line = self.state.active_line_results.get(line_idx).and_then(|l| l.as_ref())?;
        let box_item = line.char_boxes.get(char_idx)?;

        // Lazily decode and cache the original image
        if self.decoded_image.borrow().is_none() {
            if let Some(ref bytes) = self.image_bytes {
                if let Ok(img) = image::load_from_memory(bytes) {
                    *self.decoded_image.borrow_mut() = Some(img);
                }
            }
        }
        let img = self.decoded_image.borrow();
        let img = img.as_ref()?;
        let (img_w, img_h) = (img.width() as i32, img.height() as i32);

        // Calculate crop rect with 20% padding
        let pad = (box_item.h as f32 * 0.2) as i32;
        let crop_left = (box_item.left() - pad).max(0);
        let crop_top = (box_item.top() - pad).max(0);
        let crop_right = (box_item.right() + pad).min(img_w);
        let crop_bottom = (box_item.bottom() + pad).min(img_h);
        let crop_w = (crop_right - crop_left).max(1) as u32;
        let crop_h = (crop_bottom - crop_top).max(1) as u32;

        // Crop and encode to PNG
        let cropped = img.crop_imm(crop_left as u32, crop_top as u32, crop_w, crop_h);
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        cropped.write_to(&mut cursor, image::ImageFormat::Png).ok()?;

        Some(iced::widget::image::Handle::from_bytes(buf))
    }

    pub fn select_character(&mut self, line_idx: usize, char_idx: usize) {
        let was_visible = self.state.is_dictionary_visible;
        let is_same = self.state.current_tapped_line_idx == line_idx as isize
            && self.state.current_tapped_char_idx_in_line == char_idx as isize;
        self.set_cursor_pos(line_idx, char_idx);
        self.state.is_dictionary_visible = true;
        if is_same && was_visible {
            // Clicking the already-selected character on an already-open
            // dictionary toggles the alternatives panel.
            self.alternatives_visible = !self.alternatives_visible;
        } else {
            self.alternatives_visible = false;
        }

        // Update gravity so panel opens on the opposite side of the character.
        let box_item = self.state.active_line_results.get(line_idx)
            .and_then(|l| l.as_ref())
            .and_then(|line| line.char_boxes.get(char_idx))
            .cloned();
        if let Some(b) = box_item {
            self.state.update_gravity(&b, self.panel_width());
        }
        self.do_lookup(line_idx, char_idx);
        self.compute_scroll_targets(line_idx, char_idx);
    }

    /// Move the cursor without opening the dictionary (for gamepad navigation).
    pub fn move_cursor_to(&mut self, line_idx: usize, char_idx: usize) {
        self.set_cursor_pos(line_idx, char_idx);
        // Highlight just the one character at the cursor position.
        self.state.update_highlight_coords(line_idx, char_idx, 1);
        self.compute_scroll_targets(line_idx, char_idx);

        // If the dictionary panel is open and we're at default zoom (no panning),
        // flip the panel side when the character would be underneath it.
        // When zoomed in, the view pans instead — the panel stays put.
        if self.state.current_scale <= 1.01 {
            if let Some(box_item) = self.state.active_line_results.get(line_idx)
                .and_then(|l| l.as_ref())
                .and_then(|line| line.char_boxes.get(char_idx))
                .cloned()
            {
                self.state.update_gravity(&box_item, self.panel_width());
            }
        }

        // Keep the selected character visible on screen by panning if needed.
        // At default zoom (1.0) the full image is already visible — skip panning.
        if self.state.current_scale > 1.01 {
            if let Some(ann) = self.state.active_line_results.get(line_idx)
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

                    let sx = char_box.x as f32 * total_scale + base_offset_x
                        + self.state.current_trans_x;
                    let sy = char_box.y as f32 * total_scale + base_offset_y
                        + self.state.current_trans_y;
                    let sw = (char_box.w as f32).max(1.0) * total_scale;
                    let sh = (char_box.h as f32).max(1.0) * total_scale;

                    let margin = 30.0; // px margin from screen edge
                    let panel_margin = 10.0; // px margin from panel edge

                    // Compute the allowed visible region accounting for the panel.
                    // If the panel is open, keep the character from being pushed under it.
                    let (vis_left, vis_right) = if self.state.is_dictionary_visible {
                        let pw = self.panel_width();
                        let panel_on_right = self.state.last_landscape_gravity == Gravity::End;
                        if panel_on_right {
                            (margin, self.window_width - pw - panel_margin)
                        } else {
                            (pw + panel_margin, self.window_width - margin)
                        }
                    } else {
                        (margin, self.window_width - margin)
                    };

                    // Pan horizontally — keep the full character within the visible region
                    if sx + sw > vis_right {
                        let overshoot = (sx + sw) - vis_right;
                        self.state.current_trans_x -= overshoot;
                    }
                    if sx < vis_left {
                        self.state.current_trans_x += vis_left - sx;
                    }

                    // Pan vertically
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

        // Don't change gravity when selecting from neighbor/alternative views —
        // the panel is already open and positioned from the initial character click.

        if is_same {
            // Clicking the already-selected character toggles the
            // alternatives panel: open if closed, close if open.
            self.alternatives_visible = !self.alternatives_visible;
        }
        // Always do the lookup for the newly selected character.
        // If alternatives were already open (or just toggled open), they stay
        // open and update with the new character's alternatives.
        self.do_lookup(line_idx, char_idx);

        // Don't scroll the neighbor panel when selecting from it — the user
        // already clicked the item they wanted to see. Only image clicks
        // (select_character) should trigger auto-scroll.
        self.scroll_neighbor_to = None;
        self.scroll_alt_to = None;
    }

    pub(crate) fn do_lookup(&mut self, line_idx: usize, char_idx: usize) {
        let full_line_text = {
            let Some(line) = self.state.active_line_results.get(line_idx)
                .and_then(|l| l.as_ref()) else { return; };
            self.selected_word = Some(SelectedWord { line_idx, char_idx });
            line.text.clone()
        };
        // Mobile `OcrOverlayStateController.lookup` returns null for a blank
        // placeholder: the position is selectable but has no definition.
        if full_line_text.chars().nth(char_idx) == Some(GAP_CHAR) {
            self.state.cached_entries = Rc::new(Vec::new());
            self.state.current_word_length = 1;
            self.state.update_highlight_coords(line_idx, char_idx, 1);
            return;
        }
        // Only look up if db/deinflector are loaded (bootstrap may not be done yet)
        if let (Some(db), Some(deinf)) = (self.db.as_ref(), self.deinflector.as_ref()) {
            if let Some(result) = self.state.lookup(line_idx, char_idx, db, deinf) {
                self.state.cached_entries = result.matches;
                self.state.current_word_length = result.max_len;
                // Highlight the full matched word in yellow (like Kotlin)
                self.state.update_highlight_coords(line_idx, char_idx, result.max_len);

                // ── Second pass: look up each individual kanji with per-session cache ──
                let term_len = result.max_len;
                // Take the matched surface from the whole OCR stream, not just
                // the tapped line, so a word crossing a line boundary keeps its
                // trailing kanji.
                let matched_term = self
                    .state
                    .matched_term_at(self.state.get_global_idx(line_idx, char_idx), term_len);
                let mut append_kanji: Vec<FormattedEntry> = Vec::new();
                for ch in matched_term.chars() {
                    // Only CJK Unified Ideographs (kanji)
                    if !('\u{4E00}'..='\u{9FFF}').contains(&ch)
                        && !('\u{3400}'..='\u{4DBF}').contains(&ch) {
                        continue;
                    }
                    let kanji_str = ch.to_string();
                    // Deduplicate: skip if this kanji already has an entry from the
                    // first-pass term lookup or we already appended it above
                    if self.state.cached_entries.iter().any(|e| e.term == kanji_str)
                        || append_kanji.iter().any(|e| e.term == kanji_str) {
                        continue;
                    }
                    // Check per-session kanji cache before querying DB
                    let kanji_only: Vec<DictionaryEntry> =
                        if let Some(cached) = self.state.kanji_cache.get(&kanji_str) {
                            cached.clone()
                        } else if let Ok(kanji_results) = db.find_by_texts(&[kanji_str.clone()]) {
                            let filtered: Vec<DictionaryEntry> = kanji_results
                                .into_iter()
                                .filter(|e| e.onyomi.is_some() || e.kunyomi.is_some())
                                .collect();
                            self.state.kanji_cache.insert(kanji_str.clone(), filtered.clone());
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

    /// Landscape orientation, as mobile's `isLandscape = root.width > root.height`
    /// — the gate for the chips' vertical forms (D28).
    fn is_landscape(&self) -> bool {
        self.window_width > self.window_height
    }

    /// Total width of the panel (dict + neighbors + alt + spacing + padding).
    /// Must match the values used in view().
    fn panel_width(&self) -> f32 {
        let dict_width: f32 = 300.0;
        let neigh_width: f32 = 42.0;
        let alt_width: f32 = 42.0;
        let spacing: f32 = 2.0;
        let padding: f32 = 4.0;
        dict_width + neigh_width + alt_width + spacing + padding
    }

    /// Compute the scroll targets for the neighbor and alternatives panels
    /// so the selected character is centered (or as close as possible).
    pub(crate) fn compute_scroll_targets(&mut self, line_idx: usize, char_idx: usize) {
        // Compute the global character index across all lines
        let mut global_idx = 0;
        for (li, line_opt) in self.state.active_line_results.iter().enumerate() {
            if li == line_idx {
                global_idx += char_idx;
                break;
            }
            if let Some(line) = line_opt.as_ref() {
                global_idx += line.text.chars().count();
            }
        }
        self.scroll_neighbor_to = Some(global_idx);
        self.scroll_alt_to = Some(char_idx);
    }

    /// Scroll the dictionary panel to the top.
    pub fn scroll_dict_to_top_task(&self) -> iced::Task<Message> {
        operate(scroll_to(
            Id::new("dict_scroll"),
            AbsoluteOffset { x: Some(0.0), y: Some(0.0) },
        ))
    }

    /// Scroll the dictionary panel by a delta (px) relative to the current scroll position.
    /// This acts like mouse-wheel scrolling — clamped by Iced, no phantom accumulation.
    pub fn scroll_dict_by_delta(&self, delta: f32) -> iced::Task<Message> {
        operate(scroll_by(Id::new("dict_scroll"), AbsoluteOffset { x: 0.0, y: delta }))
    }
    /// Returns None if no scroll is needed.
    #[allow(dead_code)]
    pub fn scroll_neighbor_task(&self) -> Option<iced::Task<Message>> {
        let target = self.scroll_neighbor_to?;
        // Compute box_size matching neighbor_panel (window_width / 11, clamped 24..40)
        let box_size = (self.window_width / 11.0).clamp(24.0, 40.0);
        let item_height = box_size + 2.0; // button height + spacing
        let target_y = target as f32 * item_height;
        // Center in viewport using actual window height (neighbor panel is ~70% of window)
        let viewport_h = self.window_height * 0.7;
        let scroll_y = (target_y - viewport_h / 2.0).max(0.0);
        Some(operate(scroll_to(
            Id::new("neighbor_scroll"),
            AbsoluteOffset { x: Some(0.0), y: Some(scroll_y) },
        )))
    }

    /// Create a scroll task for the alternatives panel to center the selected character.
    /// Returns None if no scroll is needed.
    #[allow(dead_code)]
    pub fn scroll_alt_task(&self) -> Option<iced::Task<Message>> {
        let target = self.scroll_alt_to?;
        let box_size = (self.window_width / 11.0).clamp(24.0, 40.0);
        let item_height = box_size + 2.0;
        let target_y = target as f32 * item_height;
        let viewport_h = self.window_height * 0.7;
        let scroll_y = (target_y - viewport_h / 2.0).max(0.0);
        Some(operate(scroll_to(
            Id::new("alt_scroll"),
            AbsoluteOffset { x: Some(0.0), y: Some(scroll_y) },
        )))
    }

    pub fn build_nav_centers(&self) -> Vec<(f32, f32)> {
        let mut centers = Vec::new();
        for line_opt in &self.state.active_line_results {
            if let Some(line) = line_opt {
                for b in &line.char_boxes {
                    let cx = b.left() as f32 + (b.w as f32) / 2.0;
                    let cy = b.top() as f32 + (b.h as f32) / 2.0;
                    centers.push((cx, cy));
                }
            }
        }
        centers
    }

    /// Called once per complete recognition batch from the background OCR
    /// thread. `annotations[i]` is detection box `i`; boxes whose
    /// recognition produced no text stay as detection-only placeholders.
    /// Per line this does exactly what the old per-result handler did
    /// (BlankGaps, overlay line result, glyph pre-warm); the nav-dirty
    /// marker, cursor seed and synced-annotation refresh run once for the
    /// whole batch instead of once per line.
    pub fn apply_ocr_batch(&mut self, annotations: Vec<DetectedAnnotation>) {
        let mut lines: Vec<(usize, LineResult)> = Vec::new();
        let mut anns: Vec<DetectedAnnotation> = Vec::with_capacity(annotations.len());
        for (index, mut annotation) in annotations.into_iter().enumerate() {
            // Mobile `addLineToResults` runs BlankGaps before layout (#44
            // Feature 2), so the placeholder is part of the line from here on:
            // text, char boxes and alternatives all grow together.
            if let Some(line) = annotation.line.take() {
                annotation.line = Some(apply_blank_gaps(&line));
            }
            let line = annotation.line.clone();
            let quad = annotation.quad;
            if let Some(ref line) = line {
                println!(
                    "[VIEWER] recv ann idx={}: is_vertical={}, has_text={}, char_boxes={}",
                    index,
                    line.is_vertical,
                    !line.text.is_empty(),
                    line.char_boxes.len()
                );
            }
            // If the annotation has a line result, update the overlay state
            // UNLESS the user has edited this line's text — skip overwrite in
            // that case to preserve the user's edit.
            if let Some(ref line) = line {
                if !self.edited_lines.contains(&index) {
                    lines.push((index, line.clone()));
                }
                self.prewarm_line_glyphs(line, quad.as_ref());
            }
            anns.push(annotation);
        }
        self.annotations = Rc::new(anns);
        self.det_box_count = self.annotations.len();
        if !lines.is_empty() {
            self.state.set_line_results_batch(lines);
        }
        // Mark nav graph dirty — rebuilt lazily on next navigation or render.
        self.state.mark_nav_dirty();
        // Update cursor if not yet set
        if self.state.current_tapped_line_idx < 0 || self.state.current_tapped_char_idx_in_line < 0 {
            self.state.ensure_cursor_position();
        }
        // Keep synced_annotations cache in sync. One fresh Rc for the whole
        // batch so self.annotations keeps refcount=1 (in-place updates).
        *self.synced_annotations.borrow_mut() = Rc::new((*self.annotations).clone());
    }

    /// Pre-warm the glyph cache for one line's characters so view() doesn't
    /// rasterize them on the draw path (which causes visible stutter on the
    /// first frame new characters appear).
    fn prewarm_line_glyphs(&self, line: &LineResult, quad: Option<&RotatedBox>) {
        let Some(gc) = self.glyph_cache.as_ref() else { return };
        // The scale needs the real image size: before ImageReady arrives the
        // content scale is unknown, and pre-warming against a bogus one costs
        // seconds per line in `embolden`.
        let Some(total_scale) = content_scale(
            self.window_width,
            self.window_height,
            self.img_w,
            self.img_h,
            self.state.current_scale,
        ) else {
            return;
        };
        // Exactly the draw pass's per-line text size.
        let px = line_text_px(line, quad, total_scale)
            .round()
            .clamp(1.0, 1024.0) as u32;
        let mut cache = gc.borrow_mut();
        let vertical = line.is_vertical;
        cache.ensure_glyph('あ', false, px);
        for ch in line.text.chars() {
            cache.ensure_glyph(ch, vertical, px);
        }
    }

    /// Nudge a detection tunable (viewer tune keys). Clamped to plausible
    /// search ranges: threshold [0.01, 0.99], unclip [0.0, 5.0].
    pub fn adjust_det_tuning(&mut self, param: DetParam, delta: f32) {
        match param {
            DetParam::Threshold => self.det_thresh = (self.det_thresh + delta).clamp(0.01, 0.99),
            DetParam::Unclip => self.det_unclip = (self.det_unclip + delta).clamp(0.0, 5.0),
        }
    }

    /// Restore the startup tuning values (`R` key).
    pub fn reset_det_tuning(&mut self) {
        self.det_thresh = self.det_thresh_default;
        self.det_unclip = self.det_unclip_default;
    }

    /// Show/hide the tuning HUD (`F1` key). Runtime-only state: each process
    /// starts hidden, so there is nothing to persist to settings.json.
    pub fn toggle_det_hud(&mut self) {
        self.det_hud_visible = !self.det_hud_visible;
    }

    /// Replace the whole detection result after a live retune. The box set
    /// and its indices change, so selection/edit state is dropped first.
    pub fn apply_retune(&mut self, annotations: Vec<DetectedAnnotation>) {
        self.edited_lines.clear();
        self.selected_word = None;
        self.alternatives_visible = false;
        self.scroll_neighbor_to = None;
        self.scroll_alt_to = None;
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

    /// One-line HUD: current tunables, box count, and the tuning keys.
    fn det_hud_text(&self) -> String {
        format!(
            "DET {:.2}  UNCLIP {:.2}  boxes {}{}   [/] unclip  -/= thresh  R reset  F1 hide",
            self.det_thresh,
            self.det_unclip,
            self.det_box_count,
            if self.det_busy { "  (re-running…)" } else { "" },
        )
    }

    /// Overlay the tuning HUD on the top-left of the viewer.
    fn with_det_hud<'a>(&'a self, base: Element<'a, Message>) -> Element<'a, Message> {
        if !self.det_hud_visible {
            return base;
        }
        let hud = Container::new(
            Text::new(self.det_hud_text())
                .size(13)
                .color(Color::from_rgb(0.95, 0.95, 0.95)),
        )
        .padding(6)
        .style(|_t: &Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(0.0, 0.0, 0.0, 0.65))),
            border: iced::Border {
                radius: 4.0.into(),
                ..Default::default()
            },
            ..Default::default()
        });
        Stack::new()
            .push(base)
            .push(
                Container::new(hud)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .align_x(alignment::Horizontal::Left)
                    .align_y(alignment::Vertical::Top),
            )
            .into()
    }

    pub fn view<'a>(&'a self) -> Element<'a, Message> {
        // Cache total_scale for glyph pre-warming in handle_ocr_recognition_result
        // Only once the image size is known: before ImageReady, img_w/img_h
        // are 1 and min(window_w, window_h) would be mistaken for the content
        // scale (~960 instead of ~1).
        if let Some(scale) = content_scale(
            self.window_width,
            self.window_height,
            self.img_w,
            self.img_h,
            self.state.current_scale,
        ) {
            self.last_total_scale.set(scale);
        }
        let has_panel = self.selected_word.is_some();

        // Gravity is computed in select_character/update_gravity with the full
        // transform (base + pan/zoom), so the panel opens on the opposite side
        // of the character's actual screen position.
        // Sync annotation text from active_line_results so alt character changes
        // are reflected in the canvas overlay. Only re-syncs when text has been
        // edited (annotations_sync_dirty), avoiding a full Vec clone every frame.
        let synced_annotations = if self.annotations_sync_dirty.get() {
            let mut ann = (*self.annotations).clone();
            for (i, line_opt) in self.state.active_line_results.iter().enumerate() {
                if let (Some(ann_line), Some(active_line)) = (
                    ann.get_mut(i).and_then(|a| a.line.as_mut()),
                    line_opt.as_ref(),
                ) {
                    ann_line.text.clone_from(&active_line.text);
                }
            }
            self.annotations_sync_dirty.set(false);
            let result = Rc::new(ann);
            // Cache the synced result so subsequent frames use it until
            // the next edit triggers another sync. view() takes &self so
            // we use RefCell for interior mutability.
            *self.synced_annotations.borrow_mut() = result.clone();
            result
        } else {
            self.synced_annotations.borrow().clone()
        };
        let panel_on_right = self.state.last_landscape_gravity == Gravity::End;
        let overlay = OverlayProgram {
            annotations: synced_annotations.clone(),
            img_w: self.img_w,
            glyph_cache: self.glyph_cache.clone(),
            img_h: self.img_h,
            image: None,
            panel_visible: has_panel,
            panel_on_right,
            dict_width: 300.0,
            cursor_pos: self.state.current_cursor(),
            current_scale: self.state.current_scale,
            current_trans_x: self.state.current_trans_x,
            current_trans_y: self.state.current_trans_y,
            draw_annotations: !self.is_zooming,
            handle_pan_zoom: true,
            highlighted_coords: self.state.last_highlighted_coords.clone(),
            nav_edges_initial: self.state.nav_graph.as_ref().map(|g| g.initial_edges.clone()),
            nav_edges_final: self.state.nav_graph.as_ref().map(|g| g.edges.clone()),
            nav_centers: self.build_nav_centers(),
        };

        let annotation_canvas = Canvas::new(overlay)
            .width(Length::Fill)
            .height(Length::Fill);

        // The image is drawn in a SEPARATE canvas underneath, because tiny_skia
        // always composites images after primitives. Two separate Canvas widgets
        // in a Stack gives us correct z-ordering: image (bottom) → annotations (top).
        let image_canvas = Canvas::new(OverlayProgram {
            annotations: Rc::clone(&self.annotations),
            img_w: self.img_w,
            glyph_cache: self.glyph_cache.clone(),
            img_h: self.img_h,
            image: self.image_handle.as_ref().cloned(),
            panel_visible: false,
            panel_on_right: false,
            dict_width: 300.0,
            cursor_pos: None,
            current_scale: self.state.current_scale,
            current_trans_x: self.state.current_trans_x,
            current_trans_y: self.state.current_trans_y,
            draw_annotations: false,
            handle_pan_zoom: false,
            highlighted_coords: Vec::new(),
            nav_edges_initial: None,
            nav_edges_final: None,
            nav_centers: Vec::new(),
        })
        .width(Length::Fill)
        .height(Length::Fill);

        if !has_panel {
            return self.with_det_hud(
                Container::new(Stack::new().push(image_canvas).push(annotation_canvas))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .into(),
            );
        }

        // Build panel components
        let dict_panel = self.dictionary_panel(if !self.state.cached_entries.is_empty() {
            Rc::clone(&self.state.cached_entries)
        } else {
            Rc::new(Vec::new())
        });
        let neigh_panel = self.neighbor_panel();

        // Crop the character preview image from the screenshot (cached per selection)
        let preview_image = self.selected_word.as_ref().and_then(|sw| {
            let mut cache = self.cached_preview.borrow_mut();
            if let Some((cached_li, cached_ci, handle)) = cache.as_ref() {
                if *cached_li == sw.line_idx && *cached_ci == sw.char_idx {
                    return Some(handle.clone());
                }
            }
            let handle = self.crop_character_image(sw.line_idx, sw.char_idx)?;
            *cache = Some((sw.line_idx, sw.char_idx, handle.clone()));
            Some(handle)
        });
        let alt_panel = self.alternatives_panel(preview_image.as_ref());

        let dict_width = Pixels(300.0);

        // Stable inner row: neighbors + dictionary. This never changes
        // structure, so the Scrollables inside never reset.
        let mut inner_row = Row::new().spacing(2);
        if panel_on_right {
            inner_row = inner_row.push(neigh_panel);
            inner_row = inner_row.push(dict_panel.width(dict_width));
        } else {
            inner_row = inner_row.push(dict_panel.width(dict_width));
            inner_row = inner_row.push(neigh_panel);
        }

        // The panel container wraps the stable row.
        let panel = Container::new(inner_row).padding(2).style(move |_t: &Theme| {
            container::Style {
                background: Some(iced::Background::Color(Color::from_rgba(
                    25.0 / 255.0,
                    25.0 / 255.0,
                    25.0 / 255.0,
                    245.0 / 255.0,
                ))), // argb(245, 25, 25, 25)
                border: iced::Border {
                    radius: 4.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        });

        // The stable panel content: neighbors + dictionary in the correct
        // order. This NEVER changes — same children, same structure.
        let panel_content = Container::new(panel)
            .width(Length::Shrink)
            .height(Length::Fill);

        // Position the panel_content at the correct screen edge using a
        // Row with a Spacer. This Row structure also never changes.
        let positioned_panel: Element<'a, Message> = if panel_on_right {
            Row::new()
                .width(Length::Fill)
                .push(iced::widget::Space::new().width(Length::Fill).height(Length::Fill))
                .push(panel_content)
                .into()
        } else {
            Row::new()
                .width(Length::Fill)
                .push(panel_content)
                .push(iced::widget::Space::new().width(Length::Fill).height(Length::Fill))
                .into()
        };

        // The alt panel is in a Stack ON TOP of the positioned panel.
        // When hidden, it's simply not in the Stack — zero layout impact.
        // When visible, it floats on top at the correct edge.
        let mut content_stack = Stack::new().push(positioned_panel);

        if self.alternatives_visible {
            // Estimate neighbor panel width for spacer (box_size = window_width/11 clamped)
            let estimated_neigh = (self.window_width / 11.0).clamp(24.0, 40.0);
            // Add extra margin so the alt panel doesn't overlap with the neighbor panel
            let alt_margin = 4.0;
            let panel_content_w = dict_width + Pixels(2.0) + Pixels(estimated_neigh) + Pixels(alt_margin);

            // The alt panel Container.
            let alt_container = Container::new(alt_panel)
                .height(Length::Fill)
                .style(move |_t: &Theme| container::Style {
                    background: Some(iced::Background::Color(Color::from_rgba(
                        25.0 / 255.0,
                        25.0 / 255.0,
                        25.0 / 255.0,
                        245.0 / 255.0,
                    ))), // argb(245, 25, 25, 25)
                    ..Default::default()
                });

            // Position the alt panel using spacers on both sides.
            // The fixed-width spacer ensures the alt sits adjacent to the
            // panel content (not overlapping it), while the fill spacer
            // takes the remaining space.
            let alt_positioned: Element<'a, Message> = if panel_on_right {
                // Panel content is on the right. Alt should be to its left.
                // [spacer(fill) | alt | spacer(panel_content_w)]
                // The fill spacer pushes alt right, the fixed spacer ensures
                // the alt doesn't overlap the panel content.
                Row::new()
                    .width(Length::Fill)
                    .push(iced::widget::Space::new().width(Length::Fill).height(Length::Fill))
                    .push(alt_container)
                    .push(iced::widget::Space::new().width(panel_content_w).height(Length::Fill))
                    .into()
            } else {
                // Panel content is on the left. Alt should be to its right.
                // [spacer(panel_content_w) | alt | spacer(fill)]
                // The fixed spacer pushes alt left, the fill spacer takes
                // the remaining space on the right.
                Row::new()
                    .width(Length::Fill)
                    .push(iced::widget::Space::new().width(panel_content_w).height(Length::Fill))
                    .push(alt_container)
                    .push(iced::widget::Space::new().width(Length::Fill).height(Length::Fill))
                    .into()
            };

            content_stack = content_stack.push(alt_positioned);
        }

        // Root Stack: image (bottom) → annotations (middle) → panel content (top).
        // Two separate canvases because tiny_skia composites images after primitives.
        self.with_det_hud(
            Container::new(
                Stack::new()
                    .push(image_canvas)
                    .push(annotation_canvas)
                    .push(content_stack),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .into(),
        )
    }

    fn dictionary_panel<'a>(&'a self, entries: Rc<Vec<FormattedEntry>>) -> Container<'a, Message> {
        let gray = Color::from_rgb(0.75, 0.75, 0.75); // Android LTGRAY (#BEBEBE)
        let mut content = Column::new().padding(4).spacing(4).width(Length::Fill);
        if entries.is_empty() {
            content = content.push(Text::new("No dictionary entries found.").size(14));
        }
        for entry in entries.iter() {
            let mut entry_col = Column::new().spacing(4).width(Length::Fill);
            // #62: deinflection chain row directly below the headwords.
            if let Some(chain) = &entry.deinflection {
                entry_col = entry_col.push(Self::deinflection_row(chain, &entry.term));
            }
            entry_col = entry_col.push(Self::headword_block(entry, self.show_pitch));
            // The rule after the final reading group would only separate the
            // entry from its own source caption, so drop it when the caption
            // follows. Reading-group rules inside the entry stay.
            let last_rendered = entry
                .reading_groups
                .iter()
                .rposition(|g| g.render_senses)
                .filter(|_| entry.dictionary_name.is_some());
            for (i, group) in entry.reading_groups.iter().enumerate() {
                // A reading that repeats an already-rendered glossary shows its
                // headword but not a second copy of the senses (and examples).
                if !group.render_senses {
                    continue;
                }
                for sg in &group.sense_groups {
                    entry_col = entry_col.push(Self::sense_group(sg.clone(), DICT_TEXT_WIDTH));
                }
                if Some(i) == last_rendered {
                    continue;
                }
                // 1px divider per reading group (Android: DKGRAY, alpha 0.3).
                // The background must wrap only the line: padding on the
                // styled container would paint a full-width translucent box.
                let divider = Container::new(
                    iced::widget::Space::new().height(Pixels(1.0)).width(Length::Fill),
                )
                .width(Length::Fill)
                .style(|_t: &Theme| container::Style {
                    background: Some(iced::Background::Color(Color::from_rgba(
                        169.0 / 255.0,
                        169.0 / 255.0,
                        169.0 / 255.0,
                        0.3,
                    ))),
                    ..Default::default()
                });
                entry_col = entry_col.push(
                    Column::new()
                        .push(divider)
                        .padding(iced::Padding { top: 10.0, bottom: 10.0, ..Default::default() }),
                );
            }
            // Dictionary source, bottom of the entry (one caption per section).
            if let Some(name) = &entry.dictionary_name {
                entry_col = entry_col.push(
                    Container::new(Text::new(name.clone()).size(12).color(gray))
                        .width(Length::Fill)
                        .align_x(alignment::Horizontal::Right)
                        .padding(iced::Padding { top: 4.0, ..Default::default() }),
                );
            }
            content = content.push(entry_col);
        }
        Container::new(
            Scrollable::new(content)
                .id(Id::new("dict_scroll"))
                .direction(iced::widget::scrollable::Direction::Vertical(
                    iced::widget::scrollable::Scrollbar::hidden(),
                )),
        )
        .width(Length::Fill)
        .padding(4)
        .style(move |_t: &Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(
                25.0 / 255.0,
                25.0 / 255.0,
                25.0 / 255.0,
                245.0 / 255.0,
            ))),
            ..Default::default()
        })
    }

    fn bold_font() -> IcedFont {
        IcedFont {
            weight: iced::font::Weight::Bold,
            ..IcedFont::default()
        }
    }

    /// `JapaneseUtil.splitKanaList`: space-split, strip leading "-", join with 、.
    fn kana_list_text(raw: &str) -> String {
        japanese::split_kana_list(raw).join("、")
    }

    /// #62: compact deinflection chain row, e.g. "食べた → 食べる" + past chip.
    fn deinflection_row(chain: &DeinflectionChain, term: &str) -> Row<'static, Message> {
        let gray = Color::from_rgb(0.75, 0.75, 0.75);
        let mut row = Row::new().spacing(4).align_y(alignment::Vertical::Center);
        row = row.push(
            Text::new(format!("{} → {}", chain.surface, term))
                .size(12)
                .color(gray),
        );
        for step in &chain.steps {
            row = row.push(Self::tag_chip(step));
        }
        row
    }

    /// Headword block for one entry. Kanji (KANJIDIC) entries render their big
    /// glyph with 訓/音 rows; ordinary term entries render every headword with
    /// its reading in one comma-separated row.
    ///
    /// #43/#86: the pitch-accent line below the headwords only renders when
    /// `show_pitch` is on (mobile's default-off toggle); the Kanjium data is
    /// still attached either way.
    fn headword_block(entry: &FormattedEntry, show_pitch: bool) -> Container<'static, Message> {
        let cyan = Color::from_rgb(0.0, 1.0, 1.0); // Android CYAN
        let gray = Color::from_rgb(0.75, 0.75, 0.75);
        let mut content = Column::new().spacing(2);

        let kanji_groups: Vec<&FormattedReadingGroup> = entry
            .reading_groups
            .iter()
            .filter(|g| g.is_kanji_entry)
            .collect();
        for group in kanji_groups {
            for hw in &group.headwords {
                let mut row = Row::new().spacing(10).align_y(alignment::Vertical::Center);
                row = row.push(
                    Text::new(hw.kanji.clone())
                        .size(48)
                        .color(cyan)
                        .font(Self::bold_font()),
                );
                let mut reading_stack = Column::new().spacing(0);
                // 訓 (kun) above 音 (on).
                if let Some(k) = &hw.kunyomi {
                    if !k.is_empty() {
                        reading_stack =
                            reading_stack.push(Self::kun_on_row("訓", &Self::kana_list_text(k)));
                    }
                }
                if let Some(o) = &hw.onyomi {
                    if !o.is_empty() {
                        reading_stack =
                            reading_stack.push(Self::kun_on_row("音", &Self::kana_list_text(o)));
                    }
                }
                row = row.push(reading_stack);
                content = content.push(row);
            }
        }

        // Every (kanji, reading) pair across the remaining reading groups, one
        // row, comma separated.
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
            let mut row = Row::new()
                .spacing(2)
                .align_y(alignment::Vertical::Bottom);
            for (i, (kanji, reading)) in pairs.iter().enumerate() {
                row = row.push(Self::ruby_view(kanji, reading, false, reserve_ruby));
                if i + 1 < pairs.len() {
                    row = row.push(Text::new("、").size(24).color(gray));
                }
            }
            content = content.push(row);

            // #43: pitch accents for every reading of this entry, one line —
            // gated by the pitch switch (#43/#86), off by default.
            let items: Vec<(String, Vec<i32>)> = Self::pitch_items(entry, show_pitch);
            if !items.is_empty() {
                content = content.push(Self::pitch_line(&items));
            }
        }
        Container::new(content)
    }

    /// One 訓 / 音 row — label (12, GRAY) beside its readings (13, LTGRAY).
    fn kun_on_row(label: &str, readings: &str) -> Row<'static, Message> {
        let gray = Color::from_rgb(0.75, 0.75, 0.75);
        let light_gray = Color::from_rgb(0.75, 0.75, 0.75);
        Row::new()
            .spacing(6)
            .align_y(alignment::Vertical::Center)
            .push(Text::new(label.to_string()).size(12).color(gray))
            .push(Text::new(readings.to_string()).size(13).color(light_gray))
    }

    /// #43/#86: the pitch rows the headword block draws — every non-kanji
    /// reading group that carries Kanjium downstep positions, or nothing when
    /// the pitch switch is off (or no group has pitch data). Split out so the
    /// gate is unit-testable without rendering iced widgets.
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

    /// #43: one comma-separated pitch line; high morae white, low gray, with a
    /// fall arrow where the downstep leaves the word.
    fn pitch_line(items: &[(String, Vec<i32>)]) -> Row<'static, Message> {
        let accent = Color::WHITE;
        let plain = Color::from_rgb(0.67, 0.67, 0.67);
        let mut row = Row::new().spacing(1).align_y(alignment::Vertical::Top);
        for (i, (reading, positions)) in items.iter().enumerate() {
            let morae = japanese::morae_of(reading);
            if morae.is_empty() {
                continue;
            }
            if i > 0 {
                row = row.push(Text::new("、").size(13).color(plain));
            }
            let position = positions.first().copied().unwrap_or(0);
            let contour = japanese::pitch_pattern(morae.len(), position);
            for (mora_index, mora) in morae.iter().enumerate() {
                let color = if contour.get(mora_index).copied().unwrap_or(false) {
                    accent
                } else {
                    plain
                };
                row = row.push(Text::new(mora.clone()).size(13).color(color));
            }
            if japanese::falls_beyond_word(morae.len(), position) {
                row = row.push(Text::new("↓").size(9).color(plain));
            }
        }
        row
    }

    /// Android `createTagView` palette: POS/verb/adjective blue, common noun
    /// classes green, jlpt/grade/favourite red, everything else gray.
    fn tag_color(tag: &str) -> Color {
        if tag == "pos"
            || tag.starts_with('v')
            || tag == "adj-i"
            || tag == "adj-na"
        {
            Color::from_rgb(0.23, 0.35, 0.48) // #3a5a7a
        } else if tag == "n" || tag == "adv" || tag == "pn" {
            Color::from_rgb(0.23, 0.48, 0.35) // #3a7a5a
        } else if tag.starts_with("jlpt")
            || tag.starts_with("grade")
            || tag == "★"
            || tag == "meta"
        {
            Color::from_rgb(0.48, 0.23, 0.23) // #7a3a3a
        } else {
            Color::from_rgb(0.27, 0.27, 0.27) // #444444
        }
    }

    fn tag_chip(tag: &str) -> Container<'static, Message> {
        let bg = Self::tag_color(tag);
        Container::new(
            Text::new(tag.to_string())
                .size(10)
                .color(Color::WHITE)
                .font(Self::bold_font()),
        )
        .padding([1.0, 4.0])
        .style(move |_t: &Theme| container::Style {
            background: Some(iced::Background::Color(bg)),
            border: iced::Border {
                radius: 5.0.into(),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    /// Base-text colour of the resolved [`jpdict_core::ruby_style::RubyStyle`].
    fn ruby_base(style: &jpdict_core::ruby_style::RubyStyle) -> Color {
        Color::from_rgb(style.base[0], style.base[1], style.base[2])
    }

    /// Ruby-row colour of the resolved [`jpdict_core::ruby_style::RubyStyle`].
    fn ruby_tint(style: &jpdict_core::ruby_style::RubyStyle) -> Color {
        Color::from_rgb(style.ruby[0], style.ruby[1], style.ruby[2])
    }

    /// Minimal furigana (#55): ruby only over kanji spans, okurigana as plain
    /// base text. Falls back to full-reading ruby when unalignable.
    fn ruby_view(
        term: &str,
        reading: &str,
        is_mini: bool,
        reserve_ruby_space: bool,
    ) -> Element<'static, Message> {
        let style = jpdict_core::ruby_style::ruby_style(is_mini);
        let base = |text: String| -> Element<'static, Message> {
            let label = Text::new(text).size(style.base_size).color(Self::ruby_base(&style));
            if style.bold { label.font(Self::bold_font()).into() } else { label.into() }
        };
        if term == reading {
            if !reserve_ruby_space {
                return base(term.to_string());
            }
            // Mixed group: reserve the same ruby row a furigana-bearing
            // sibling has, so baselines align.
            let mut stack = Column::new()
                .align_x(alignment::Horizontal::Center)
                .spacing(0);
            stack = stack.push(Text::new(" ").size(style.ruby_size).color(Self::ruby_tint(&style)));
            stack = stack.push(base(term.to_string()));
            return stack.into();
        }
        let segments = japanese::align_furigana(term, reading);
        let Some(segments) = segments else {
            return Self::full_ruby_view(term, reading, is_mini);
        };
        if !segments.iter().any(|s| s.ruby.is_some()) {
            return Self::full_ruby_view(term, reading, is_mini);
        }
        let mut row = Row::new().align_y(alignment::Vertical::Bottom);
        for seg in segments {
            match seg.ruby {
                None => row = row.push(base(seg.base)),
                Some(ruby) => {
                    let mut stack = Column::new()
                        .align_x(alignment::Horizontal::Center)
                        .spacing(0);
                    stack = stack.push(Text::new(ruby).size(style.ruby_size).color(Self::ruby_tint(&style)));
                    stack = stack.push(base(seg.base));
                    row = row.push(stack);
                }
            }
        }
        row.into()
    }

    /// Fallback when the reading cannot align over kanji spans: the whole
    /// reading sits above the whole term. Paints from the same
    /// [`jpdict_core::ruby_style::RubyStyle`]
    /// as the aligned path, so unalignable (usually longest-compound) terms
    /// match the mode's style exactly.
    fn full_ruby_view(term: &str, reading: &str, is_mini: bool) -> Element<'static, Message> {
        let style = jpdict_core::ruby_style::ruby_style(is_mini);
        let mut stack = Column::new()
            .align_x(alignment::Horizontal::Center)
            .spacing(0);
        stack = stack.push(Text::new(reading.to_string()).size(style.ruby_size).color(Self::ruby_tint(&style)));
        if style.bold {
            stack = stack.push(
                Text::new(term.to_string())
                    .size(style.base_size)
                    .color(Self::ruby_base(&style))
                    .font(Self::bold_font()),
            );
        } else {
            stack = stack.push(Text::new(term.to_string()).size(style.base_size).color(Self::ruby_base(&style)));
        }
        stack.into()
    }

    /// Render a sense group. `width` is the text width available inside the
    /// dictionary column (see [`DICT_TEXT_WIDTH`]).
    fn sense_group(sg: FormattedSenseGroup, width: f32) -> Column<'static, Message> {
        let white = Color::WHITE;
        let mut content = Column::new().spacing(3).width(Length::Fill);

        if !sg.tags.is_empty() {
            let mut tag_row = Row::new()
                .spacing(3)
                .align_y(alignment::Vertical::Center)
                .padding(iced::Padding { top: 4.0, ..Default::default() });
            for tag in &sg.tags {
                tag_row = tag_row.push(Self::tag_chip(tag));
            }
            content = content.push(tag_row);
        }

        // #88: Jitendex group metadata (POS/field info) is structured content
        // rendered once as the header.
        if !sg.header.is_empty() {
            content = content.push(
                Self::render_definition(&sg.header, width)
                    .padding(iced::Padding { top: 2.0, ..Default::default() }),
            );
        }

        if sg.is_forms {
            // JMdict "Forms" groups render as unnumbered rows.
            for sense in &sg.senses {
                content = content.push(Self::render_definition(&sense.nodes, width));
            }
        } else {
            for sense in &sg.senses {
                let mut sense_row = Row::new().spacing(3).align_y(alignment::Vertical::Top);
                sense_row = sense_row.push(Text::new(format!("{}. ", sense.index)).size(15).color(white));
                sense_row = sense_row
                    .push(Self::render_definition(&sense.nodes, width - 20.0).width(Length::Fill));
                content = content.push(sense_row);
            }
        }

        // #88: the forms table and attribution trail the senses, unnumbered.
        if !sg.trailing.is_empty() {
            content = content.push(
                Self::render_definition(&sg.trailing, width)
                    .padding(iced::Padding { bottom: 2.0, ..Default::default() }),
            );
        }
        content
    }

    /// Render a definition node list. Inline runs (text, ruby, tags) flow in
    /// a wrapping layout like mobile's `FlowLayout`; block nodes (examples,
    /// tables, lists, non-inline groups, citations) start their own row.
    fn render_definition(nodes: &[DefinitionNode], width: f32) -> Column<'static, Message> {
        let mut col = Column::new().spacing(2).width(Length::Fill);
        let mut inline: Vec<DefinitionNode> = Vec::new();
        for node in nodes {
            match node {
                DefinitionNode::Text(_) | DefinitionNode::Ruby { .. } | DefinitionNode::Tag { .. } => {
                    inline.push(node.clone());
                }
                DefinitionNode::Citation(text) => {
                    col = Self::flush_inline(col, &mut inline, width);
                    col = col.push(
                        Text::new(text.clone())
                            .size(11)
                            .color(Color::from_rgba(1.0, 1.0, 1.0, 120.0 / 255.0)),
                    );
                }
                DefinitionNode::Example(example) => {
                    col = Self::flush_inline(col, &mut inline, width);
                    col = col.push(Self::example_box(example, width));
                }
                DefinitionNode::ListBlock { items, .. } => {
                    col = Self::flush_inline(col, &mut inline, width);
                    let mut block = Column::new().spacing(3).padding([4.0, 2.0]);
                    for item in items {
                        block = block.push(Self::render_definition(item, width - 8.0));
                    }
                    col = col.push(block);
                }
                DefinitionNode::Table { rows } => {
                    if !rows.is_empty() {
                        col = Self::flush_inline(col, &mut inline, width);
                        col = col.push(Self::definition_table(rows, width));
                    }
                }
                DefinitionNode::Group { nodes, is_inline } => {
                    if *is_inline {
                        inline.extend(nodes.iter().cloned());
                    } else {
                        col = Self::flush_inline(col, &mut inline, width);
                        col = col.push(Self::render_definition(nodes, width));
                    }
                }
            }
        }
        Self::flush_inline(col, &mut inline, width)
    }

    /// Push the pending inline nodes as one wrapping flow block.
    fn flush_inline<'a>(
        col: Column<'a, Message>,
        inline: &mut Vec<DefinitionNode>,
        width: f32,
    ) -> Column<'a, Message> {
        if inline.is_empty() {
            return col;
        }
        let cells = Self::inline_cells(inline);
        inline.clear();
        let mut flow = Column::new().spacing(2).width(Length::Fill);
        for line in Self::wrap_inline_cells(cells, width) {
            flow = flow.push(Self::inline_line(line));
        }
        col.push(flow)
    }

    /// Atomic inline cells: characters, ruby stacks and tag chips.
    fn inline_cells(nodes: &[DefinitionNode]) -> Vec<InlineCell> {
        let mut cells = Vec::new();
        for node in nodes {
            match node {
                DefinitionNode::Text(text) => cells.extend(text.chars().map(InlineCell::Char)),
                DefinitionNode::Ruby { term, reading } => cells.push(InlineCell::Ruby {
                    term: term.clone(),
                    reading: reading.clone(),
                    width: inline_text_width(term, DEF_TEXT_SIZE)
                        .max(inline_text_width(reading, DEF_RUBY_SIZE)),
                }),
                DefinitionNode::Tag { text } => cells.push(InlineCell::Tag {
                    text: text.clone(),
                    width: inline_text_width(text, 10.0) + 12.0,
                }),
                _ => {}
            }
        }
        cells
    }

    /// Greedy line packing at `width`, mobile `FlowLayout` style: cells fill a
    /// row until the next one would overflow. A trailing ASCII word moves to
    /// the next line whole when it fits there; an oversized cell still gets
    /// its own line rather than being dropped.
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
            // Wrap. Keep a trailing ASCII word together when it can move down.
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
            // Kinsoku: a character that may not start a line (、。 etc.) takes
            // the character before it down too, unless that would orphan the
            // previous line's only cell.
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

    /// One flow line: merged text runs with ruby stacks and chips between them,
    /// bottom-aligned so plain text sits on the ruby bases' baseline.
    fn inline_line(line: Vec<InlineCell>) -> Row<'static, Message> {
        let white = Color::WHITE;
        let mut row = Row::new().align_y(alignment::Vertical::Bottom);
        let mut run = String::new();
        let flush_run = |row: Row<'static, Message>, run: &mut String| {
            if run.is_empty() {
                row
            } else {
                row.push(Text::new(std::mem::take(run)).size(DEF_TEXT_SIZE).color(white))
            }
        };
        for cell in line {
            match cell {
                InlineCell::Char(c) => run.push(c),
                InlineCell::Ruby { term, reading, .. } => {
                    row = flush_run(row, &mut run);
                    // Body flow: mini ruby in the body typeface (see RubyStyle —
                    // deliberate departure from mobile's cyan+bold mini).
                    row = row.push(Self::ruby_view(&term, &reading, true, false));
                }
                InlineCell::Tag { text, .. } => {
                    row = flush_run(row, &mut run);
                    row = row.push(Self::tag_chip(&text));
                }
            }
        }
        flush_run(row, &mut run)
    }

    /// #88: an example box — Japanese sentence (ruby intact) over its
    /// translation, on the example palette. `width` is the outer text width;
    /// the box's own 8px padding is subtracted before laying out parts.
    fn example_box(example: &ExampleNode, width: f32) -> Container<'static, Message> {
        let white = Color::WHITE;
        let light_gray = Color::from_rgb(0.75, 0.75, 0.75);
        let mut inner = Column::new().spacing(3).width(Length::Fill);
        if let Some(jp) = &example.japanese {
            inner = inner.push(
                Text::new(jp.clone())
                    .size(16)
                    .color(white)
                    .width(Length::Fill)
                    .wrapping(iced::widget::text::Wrapping::Word),
            );
            if let Some(en) = &example.english {
                inner = inner.push(
                    Text::new(en.clone())
                        .size(14)
                        .color(light_gray)
                        .width(Length::Fill)
                        .wrapping(iced::widget::text::Wrapping::Word),
                );
            }
        } else if !example.parts.is_empty() {
            for part in &example.parts {
                inner = inner.push(Self::render_definition(part, width - 16.0));
            }
        } else if !example.content.is_empty() {
            inner = inner.push(Self::render_definition(&example.content, width - 16.0));
        }
        Container::new(inner)
            .width(Length::Fill)
            .padding([8.0, 8.0])
            .style(|_t: &Theme| container::Style {
                background: Some(iced::Background::Color(Color::from_rgba(
                    1.0,
                    1.0,
                    1.0,
                    10.0 / 255.0,
                ))),
                border: iced::Border {
                    color: Color::from_rgba(1.0, 1.0, 1.0, 80.0 / 255.0),
                    width: 1.0,
                    radius: 8.0.into(),
                },
                ..Default::default()
            })
    }

    /// #88: render a structured-content table as a real grid: equal-weight
    /// columns, a shared 1px rule between neighbours, and the example box's
    /// card palette around the outside.
    fn definition_table(rows: &[Vec<Vec<DefinitionNode>>], width: f32) -> Container<'static, Message> {
        let border = Color::from_rgba(1.0, 1.0, 1.0, 80.0 / 255.0);
        let columns = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        // 1px dividers between the equal-width cells; each cell pays its own
        // 6px horizontal padding.
        let cell_width = if columns > 0 {
            (width - columns.saturating_sub(1) as f32) / columns as f32
        } else {
            width
        };
        let mut table = Column::new().width(Length::Fill).spacing(0);
        for (row_index, cells) in rows.iter().enumerate() {
            let mut row = Row::new().width(Length::Fill);
            for column in 0..columns {
                if column > 0 {
                    row = row.push(
                        Container::new(
                            iced::widget::Space::new()
                                .width(Pixels(1.0))
                                .height(Length::Fill),
                        )
                        .height(Length::Fill)
                        .style(move |_t: &Theme| container::Style {
                            background: Some(iced::Background::Color(border)),
                            ..Default::default()
                        }),
                    );
                }
                let mut cell = Column::new().padding([4.0, 6.0]).width(Length::FillPortion(1));
                if let Some(nodes) = cells.get(column) {
                    cell = cell.push(Self::render_definition(nodes, cell_width - 12.0));
                }
                row = row.push(cell);
            }
            table = table.push(row);
            if row_index + 1 < rows.len() {
                table = table.push(
                    Container::new(
                        iced::widget::Space::new()
                            .width(Length::Fill)
                            .height(Pixels(1.0)),
                    )
                    .width(Length::Fill)
                    .style(move |_t: &Theme| container::Style {
                        background: Some(iced::Background::Color(border)),
                        ..Default::default()
                    }),
                );
            }
        }
        Container::new(table)
            .width(Length::Fill)
            .style(move |_t: &Theme| container::Style {
                background: Some(iced::Background::Color(Color::from_rgba(
                    1.0,
                    1.0,
                    1.0,
                    10.0 / 255.0,
                ))),
                border: iced::Border {
                    color: border,
                    width: 1.0,
                    radius: 8.0.into(),
                },
                ..Default::default()
            })
    }

    fn neighbor_panel<'a>(&'a self) -> Container<'a, Message> {
        let state = self.state.get_neighbor_ui_state();
        // Compute button size dynamically based on window width
        let item_size = self.window_width / 11.0;
        let box_size = item_size.clamp(24.0, 40.0);
        let mut content = Column::new().spacing(2);
        for line in state {
            for cs in line.chars {
                let msg = Message::SelectNeighbor(line.line_idx, cs.char_idx);
                let is_selected = cs.is_selected;
                let text = chip_text(&cs.text, self.is_landscape());

                let btn: Element<'a, Message> = Container::new(
                    Text::new(text)
                        .size(Pixels(box_size * BUTTON_CHAR_RATIO))
                        .color(if is_selected { Color::BLACK } else { Color::WHITE })
                        .align_x(alignment::Horizontal::Center)
                        .align_y(alignment::Vertical::Center),
                )
                .width(Pixels(box_size))
                .height(Pixels(box_size))
                .align_x(alignment::Horizontal::Center)
                .align_y(alignment::Vertical::Center)
                .style(move |_t: &Theme| {
                    if is_selected {
                        container::Style {
                            background: Some(iced::Background::Color(
                                Color::from_rgb(1.0, 1.0, 0.0), // YELLOW
                            )),
                            ..Default::default()
                        }
                    } else {
                        container::Style {
                            background: Some(iced::Background::Color(
                                Color::from_rgb(0.255, 0.255, 0.255), // argb(255, 65, 65, 65)
                            )),
                            ..Default::default()
                        }
                    }
                })
                .into();

                content = content.push(
                    TapOrDrag::new(btn, msg),
                );
            }
        }
        // Scrollable with hidden scrollbar — sizes to content so buttons stay square
        Container::new(
            Scrollable::new(content)
                .id(Id::new("neighbor_scroll"))
                .direction(iced::widget::scrollable::Direction::Vertical(
                    iced::widget::scrollable::Scrollbar::hidden(),
                )),
        )
        .height(Length::Fill)
        .style(move |_t: &Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(
                25.0 / 255.0,
                25.0 / 255.0,
                25.0 / 255.0,
                245.0 / 255.0,
            ))), // argb(245, 25, 25, 25)
            ..Default::default()
        })
    }

    fn alternatives_panel<'a>(
        &'a self,
        preview_image: Option<&iced::widget::image::Handle>,
    ) -> Container<'a, Message> {
        let mut content = Column::new().spacing(2);
        // Compute button size dynamically based on window width (same as neighbor_panel)
        let item_size = self.window_width / 11.0;
        let box_size = item_size.clamp(24.0, 40.0);

        // Character preview image at the top — sized to match buttons
        if let Some(img_handle) = preview_image {
            content = content.push(
                IcedImage::new(img_handle.clone())
                    .width(Pixels(box_size))
                    .height(Pixels(box_size)),
            );
        }

        if let Some(alt_state) = self.state.get_alternatives_ui_state() {
            for c in alt_state.candidates {
                let msg = Message::SelectAlternative(c.char);
                let is_selected = c.is_selected;
                let ch = c.char;
                // Source tint: head evidence stays white, component-table
                // entries (neighbours + variants) read amber, LM-ranked
                // blank entries read blue. Selection still wins outright.
                let tint = match c.source {
                    jpdict_core::util::oov_suggestions::Source::Head => Color::WHITE,
                    jpdict_core::util::oov_suggestions::Source::Components
                    | jpdict_core::util::oov_suggestions::Source::Variant => {
                        Color::from_rgb(1.0, 0.8, 0.4)
                    }
                    jpdict_core::util::oov_suggestions::Source::Lm => {
                        Color::from_rgb(0.55, 0.85, 1.0)
                    }
                };

                let vertical_ch = chip_text(&ch.to_string(), self.is_landscape());

                let btn: Element<'a, Message> = Container::new(
                    Text::new(vertical_ch)
                        .size(Pixels(box_size * BUTTON_CHAR_RATIO))
                        .color(if is_selected { Color::BLACK } else { tint })
                        .align_x(alignment::Horizontal::Center)
                        .align_y(alignment::Vertical::Center),
                )
                .width(Pixels(box_size))
                .height(Pixels(box_size))
                .align_x(alignment::Horizontal::Center)
                .align_y(alignment::Vertical::Center)
                .style(move |_t: &Theme| {
                    if is_selected {
                        container::Style {
                            background: Some(iced::Background::Color(
                                Color::from_rgb(1.0, 1.0, 0.0), // YELLOW
                            )),
                            ..Default::default()
                        }
                    } else {
                        container::Style {
                            background: Some(iced::Background::Color(
                                Color::from_rgb(0.255, 0.255, 0.255), // argb(255, 65, 65, 65)
                            )),
                            ..Default::default()
                        }
                    }
                })
                .into();

                content = content.push(
                    TapOrDrag::new(btn, msg),
                );
            }
        }
        Container::new(
            Scrollable::new(content)
                .id(Id::new("alt_scroll"))
                .direction(iced::widget::scrollable::Direction::Vertical(
                    iced::widget::scrollable::Scrollbar::hidden(),
                )),
        )
        .height(Length::Fill)
        .padding(2)
        .style(move |_t: &Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(
                25.0 / 255.0,
                25.0 / 255.0,
                25.0 / 255.0,
                245.0 / 255.0,
            ))), // argb(245, 25, 25, 25)
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// TapOrDrag — fires tap only if pointer didn't move between press and release
// ---------------------------------------------------------------------------

use iced::advanced::widget::tree::{self, Tree};
use iced::advanced::{self, Layout, Widget};
use iced::advanced::widget::Operation;
use iced::advanced::Clipboard;

/// A widget that distinguishes taps from drags.
/// Only fires `on_tap` if the pointer moved less than the threshold
/// between press and release. This allows the parent Scrollable to
/// handle drag-to-scroll while still supporting tap-to-activate.
pub struct TapOrDrag<'a, Message> {
    content: Element<'a, Message>,
    on_tap: Message,
    threshold: f32,
}

impl<'a, Message: Clone> TapOrDrag<'a, Message> {
    pub fn new(content: impl Into<Element<'a, Message>>, on_tap: Message) -> Self {
        Self {
            content: content.into(),
            on_tap,
            threshold: 5.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TapState {
    pressed: bool,
    start_x: f32,
    start_y: f32,
    is_tap: bool,
    /// Accumulated movement distance since press.
    accumulated: f32,
}

impl<'a, Message: Clone + 'static> Widget<Message, Theme, Renderer> for TapOrDrag<'a, Message> {
    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
    }

    fn size_hint(&self) -> Size<Length> {
        self.content.as_widget().size_hint()
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &iced::advanced::layout::Limits,
    ) -> iced::advanced::layout::Node {
        self.content.as_widget_mut().layout(&mut tree.children[0], renderer, limits)
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &iced::advanced::renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.content.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            style,
            layout,
            cursor,
            viewport,
        );
    }

    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<TapState>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(TapState::default())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_ref(&self.content));
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &iced::Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut advanced::Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_mut::<TapState>();

        match event {
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if cursor.position_over(layout.bounds()).is_some() {
                    state.pressed = true;
                    // Use the actual cursor position, not clamped to bounds
                    if let Some(pos) = cursor.position() {
                        state.start_x = pos.x;
                        state.start_y = pos.y;
                    }
                    state.accumulated = 0.0;
                    state.is_tap = true;
                }
            }
            iced::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                if state.pressed {
                    if let Some(pos) = cursor.position() {
                        let dx = pos.x - state.start_x;
                        let dy = pos.y - state.start_y;
                        state.accumulated += (dx * dx + dy * dy).sqrt();
                        if state.accumulated > self.threshold {
                            state.is_tap = false;
                        }
                    }
                }
            }
            iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                let was_tap = state.is_tap;
                state.pressed = false;
                state.is_tap = false;
                state.accumulated = 0.0;
                if was_tap {
                    shell.publish(self.on_tap.clone());
                }
            }
            iced::Event::Touch(touch::Event::FingerPressed { .. }) => {
                if cursor.position_over(layout.bounds()).is_some() {
                    state.pressed = true;
                    if let Some(pos) = cursor.position() {
                        state.start_x = pos.x;
                        state.start_y = pos.y;
                    }
                    state.accumulated = 0.0;
                    state.is_tap = true;
                }
            }
            iced::Event::Touch(touch::Event::FingerMoved { .. }) => {
                if state.pressed {
                    if let Some(pos) = cursor.position() {
                        let dx = pos.x - state.start_x;
                        let dy = pos.y - state.start_y;
                        state.accumulated += (dx * dx + dy * dy).sqrt();
                        if state.accumulated > self.threshold {
                            state.is_tap = false;
                        }
                    }
                }
            }
            iced::Event::Touch(touch::Event::FingerLifted { .. }) => {
                let was_tap = state.is_tap;
                state.pressed = false;
                state.is_tap = false;
                state.accumulated = 0.0;
                if was_tap {
                    shell.publish(self.on_tap.clone());
                }
            }
            _ => {}
        }

        self.content.as_widget_mut().update(
            &mut tree.children[0],
            event,
            layout,
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.content.as_widget().mouse_interaction(
            &tree.children[0],
            layout,
            cursor,
            viewport,
            renderer,
        )
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        self.content
            .as_widget_mut()
            .operate(&mut tree.children[0], layout, renderer, operation);
    }
}

impl<'a, Message: Clone + 'static> From<TapOrDrag<'a, Message>> for Element<'a, Message> {
    fn from(widget: TapOrDrag<'a, Message>) -> Self {
        Element::new(widget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// If the glyph is its own reference, both axes land exactly on the box
    /// centre (this is what the old ink-fit path did for every glyph).
    #[test]
    fn glyph_ink_origin_is_identity_for_the_reference_glyph() {
        let g = (40.0f32, 40.0, 2.0, -4.0);
        let (dx, dy) = glyph_ink_origin(false, 100.0, 50.0, g, g);
        assert!((dx - 80.0).abs() < 1e-3, "horizontal dx={dx}");
        assert!((dy - 30.0).abs() < 1e-3, "horizontal dy={dy}");
        let (dx, dy) = glyph_ink_origin(true, 100.0, 50.0, g, g);
        assert!((dx - 80.0).abs() < 1e-3, "vertical dx={dx}");
        assert!((dy - 30.0).abs() < 1e-3, "vertical dy={dy}");
    }

    /// Punctuation rides its natural baseline: a small low comma must sit at
    /// the bottom of the em box, not be centred in it (the "gigantic 、"
    /// regression was the ink-fit path centring/scaling it).
    #[test]
    fn punctuation_rides_the_baseline() {
        // あ: ink spans 40px, bottom 8px below the baseline (fontdue y-up
        // ymin = -8), so its ink centre is 12px above the baseline.
        let reference = (40.0f32, 40.0, 0.0, -8.0);
        // A comma: small ink patch near the baseline.
        let comma = (10.0f32, 10.0, 0.0, -6.0);
        let (dx, dy) = glyph_ink_origin(false, 100.0, 50.0, comma, reference);
        // top = cy + ref_centre - (ymin + h) = 50 + 12 - 4 = 58.
        assert!((dy - 58.0).abs() < 1e-3, "comma dy={dy}");
        assert!((dx - 95.0).abs() < 1e-3, "comma dx={dx}");
    }

    /// Mobile `GapDetector`: a spacing ≥1.6× the median marks a dropped
    /// character; vertical lines only, and never twice.
    #[test]
    fn blank_gaps_detect_wide_vertical_spacing() {
        let boxes = vec![
            BoundingBox::new(0, 0, 40, 20, 1.0),
            BoundingBox::new(0, 20, 40, 20, 1.0),
            BoundingBox::new(0, 120, 40, 20, 1.0),
            BoundingBox::new(0, 140, 40, 20, 1.0),
        ];
        let mut line = line_with_boxes("あいうえ", boxes.clone(), true);
        // Spacings 20/100/20 → pitch 20, the 100 gap triggers at index 2.
        assert_eq!(blank_gap_positions(&line), vec![2]);
        line.is_vertical = false;
        assert!(blank_gap_positions(&line).is_empty(), "horizontal is not eligible");

        // Idempotent, and the placeholder lands between its neighbours.
        let gapped = apply_blank_gaps(&line_with_boxes("あいうえ", boxes, true));
        assert_eq!(gapped.text.chars().collect::<Vec<_>>(), vec!['あ', 'い', GAP_CHAR, 'う', 'え']);
        assert_eq!(gapped.char_boxes.len(), 5);
        // Between box 1 (20..40) and box 2 (120..140): centre 80, height 20.
        let gap_box = &gapped.char_boxes[2];
        assert_eq!((gap_box.top(), gap_box.bottom()), (70, 90), "interpolated box");
        assert_eq!(apply_blank_gaps(&gapped).text, gapped.text, "idempotent");

        // Alternatives stay index-aligned.
        let with_alts = LineResult {
            alternatives: vec![vec![('あ', 1.0)], vec![('い', 1.0)], vec![('う', 1.0)], vec![('え', 1.0)]],
            ..line_with_boxes(
                "あいうえ",
                vec![
                    BoundingBox::new(0, 0, 40, 20, 1.0),
                    BoundingBox::new(0, 20, 40, 20, 1.0),
                    BoundingBox::new(0, 120, 40, 20, 1.0),
                    BoundingBox::new(0, 140, 40, 20, 1.0),
                ],
                true,
            )
        };
        let gapped = apply_blank_gaps(&with_alts);
        assert_eq!(gapped.alternatives.len(), 5);
        assert_eq!(gapped.alternatives[2][0].0, GAP_CHAR);
        assert_eq!(gapped.alternatives[3][0].0, 'う');
    }

    /// Mobile degrades to a platform face and keeps rendering when the
    /// bundled font is missing; the PC must not panic for the same case.
    #[test]
    fn missing_font_degrades_instead_of_panicking() {
        assert!(GlyphCache::from_path(None).is_none());
        assert!(GlyphCache::from_path(Some("/nonexistent/font.ttf".into())).is_none());
        // The bundled asset still loads when present.
        assert!(GlyphCache::new_for(FontFace::Sans).is_some());
    }

    /// Selecting serif loads the bundled serif face, resolves horizontal and
    /// GSUB-vertical glyphs, and rasterizes real ink — i.e. the switch changes
    /// what the overlay draws rather than reusing the sans cache.
    #[test]
    fn serif_face_loads_and_resolves_vertical_forms() {
        let cache = GlyphCache::new_for(FontFace::Serif).expect("bundled serif face");
        {
            let mut c = cache.borrow_mut();
            let gid = c.glyph_id('一', false).expect("serif glyph for 一");
            assert_ne!(gid, 0);
            let stop = c.glyph_id('。', true).expect("serif vertical 。");
            assert_ne!(stop, 0);
        }
        let g = draw_glyph(&cache, 'あ', true, 54, false).expect("serif kana");
        assert!(g.w > 0 && g.h > 0, "serif kana has ink: {}x{}", g.w, g.h);
        // Serif ships Regular only, so highlights fake-bold through the same
        // face (the cache loaded no bold companion).
        let hl = draw_glyph(&cache, 'あ', true, 54, true).expect("serif highlight");
        assert!(hl.w >= g.w && hl.h >= g.h, "synthetic bold grows the ink box");
    }

    /// Mobile skips glyphs whose measured ink is degenerate (`glyphW <= 0 ||
    /// glyphH <= 0`); a blank cell must not draw a placeholder pixel.
    #[test]
    fn blank_glyphs_report_no_ink() {
        let cache = GlyphCache::new_for(FontFace::Sans).expect("bundled JP font");
        let g = draw_glyph(&cache, ' ', false, 54, false).expect("glyph");
        assert_eq!((g.w, g.h), (0, 0), "space must report no ink");
        let g = draw_glyph(&cache, 'あ', false, 54, false).expect("glyph");
        assert!(g.w > 0 && g.h > 0, "kana has ink");
    }

    /// Mobile box appearance constants: argb(100,0,0,0) and 4px corners.
    #[test]
    fn box_fill_and_corners_match_mobile() {
        assert!((BOX_FILL_ALPHA - 100.0 / 255.0).abs() < 1e-6);
        assert!((BOX_FILL_ALPHA - 0.40).abs() < 0.01, "was argb(102)");
        assert_eq!(BOX_CORNER_RADIUS, 4.0);
        // A 4px fillet fits a 10px edge, clamps on a 2px one.
        assert_eq!(
            fillet_radius((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), 4.0),
            4.0
        );
        assert!((fillet_radius((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), 4.0) - 0.9).abs() < 1e-4);
        // Degenerate input must not panic.
        let _ = rounded_polygon(&[(0.0, 0.0), (1.0, 1.0)], 4.0);
        let _ = rounded_polygon(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], 4.0);
    }

    /// The folded scrim equals mobile's two layers over black.
    #[test]
    fn backdrop_scrim_matches_mobile_layers() {
        let expected = 1.0 - 0.7 * (1.0 - 140.0 / 255.0);
        assert!((BACKDROP_SCRIM - expected).abs() < 1e-6, "{BACKDROP_SCRIM}");
        assert!((BACKDROP_SCRIM - 0.68431).abs() < 1e-4, "{BACKDROP_SCRIM}");
    }

    /// Chips follow mobile's orientation gate for vertical forms.
    #[test]
    fn chip_vertical_forms_are_landscape_only() {
        assert_eq!(chip_text("「", true), "\u{FE41}");
        assert_eq!(chip_text("「", false), "「");
        assert_eq!(chip_text("あ", true), "あ");
    }

    /// Mobile cursor: 2dp inflation per side, scaling with the transform.
    #[test]
    fn cursor_box_is_inflated_like_mobile() {
        let (pt, sz) = cursor_rect(Point::new(10.0, 20.0), Size::new(30.0, 40.0), 1.0);
        assert_eq!(pt, Point::new(8.0, 18.0));
        assert_eq!(sz, Size::new(34.0, 44.0));
        // At 2× zoom the whole cursor scales with the content.
        let (pt, sz) = cursor_rect(Point::new(10.0, 20.0), Size::new(30.0, 40.0), 2.0);
        assert_eq!(pt, Point::new(6.0, 16.0));
        assert_eq!(sz, Size::new(38.0, 48.0));
    }

    /// Fake bold: a single lit pixel grows into a (2r+1)² block, and the
    /// radius tracks the raster size (~1/24 em stroke).
    #[test]
    fn embolden_grows_coverage_symmetrically() {
        let mut cov = vec![0u8; 25];
        cov[12] = 255; // centre of 5×5
        let bold = embolden(&cov, 5, 5, 1);
        for y in 1..=3 {
            for x in 1..=3 {
                assert_eq!(bold[y * 5 + x], 255, "({x},{y}) should be bold");
            }
        }
        assert_eq!(bold[0], 0, "corner stays clear");
        // A zero radius returns the glyph untouched.
        assert_eq!(embolden(&cov, 5, 5, 0), cov);
        assert_eq!(fake_bold_radius(54), 1);
        assert_eq!(fake_bold_radius(200), 4);
    }

    /// Synthetic bold must grow the ink box instead of clipping the stroke:
    /// a fully inked 1×1 glyph becomes a fully inked 3×3 at radius 1.
    #[test]
    fn synthetic_bold_does_not_clip_grown_ink() {
        let (bold, w, h) = synthetic_bold(&[255u8], 1, 1, 1);
        assert_eq!((w, h), (3, 3));
        assert!(bold.iter().all(|&a| a == 255), "grown ink was clipped: {bold:?}");
    }

    /// The bundled bold companion face loads and resolves glyphs, including
    /// the GSUB vertical forms, so highlights rasterize from it rather than
    /// the synthetic fallback.
    #[test]
    fn bundled_bold_face_loads_and_resolves() {
        let dir = format!("{}/fonts", env!("CARGO_MANIFEST_DIR"));
        let cache = GlyphCache::from_paths(
            Some(std::path::PathBuf::from(format!("{dir}/NotoSansJP-Regular.ttf"))),
            Some(std::path::PathBuf::from(format!("{dir}/NotoSansJP-Bold.ttf"))),
        )
        .expect("glyph cache with bundled faces");
        let mut c = cache.borrow_mut();
        let gid = c.bold_glyph_id('一', false).expect("bold glyph for 一");
        assert_ne!(gid, 0, "bold cmap must resolve 一");
        // Vertical forms resolve on the bold face too (GSUB vert).
        let stop = c.bold_glyph_id('。', true).expect("bold vertical 。");
        assert_ne!(stop, 0);
        // No bold face is loaded when the bold path is absent.
        let no_bold = GlyphCache::from_paths(
            Some(std::path::PathBuf::from(format!("{dir}/NotoSansJP-Regular.ttf"))),
            None,
        )
        .expect("regular-only cache");
        assert!(no_bold.borrow_mut().bold_glyph_id('一', false).is_none());
    }

    #[test]
    fn halfwidth_classification_matches_mobile() {
        assert!(is_half_width('A'));
        assert!(is_half_width('~'));
        assert!(is_half_width('\u{FF76}')); // ｶ halfwidth katakana
        assert!(!is_half_width('あ'));
        assert!(!is_half_width('漢'));
        assert!(!is_half_width('。'));
    }

    fn line_with_boxes(text: &str, boxes: Vec<BoundingBox>, is_vertical: bool) -> LineResult {
        LineResult {
            text: text.into(),
            char_boxes: boxes,
            alternatives: vec![],
            raw_alternatives: vec![],
            sample_txt: None,
            is_vertical,
            chunk_boxes: vec![],
        }
    }

    /// Mobile `LineResult.glyphSizePx()`: the tallest char box (the detector
    /// box height, unclip padding included), drawn at ×0.90.
    #[test]
    fn line_text_size_uses_box_height_like_mobile() {
        // Horizontal: 60px-tall boxes, however the centres are spaced.
        let line = line_with_boxes(
            "日本語",
            vec![
                BoundingBox::new(10, 0, 40, 60, 1.0),
                BoundingBox::new(30, 0, 40, 60, 1.0),
                BoundingBox::new(50, 0, 40, 60, 1.0),
            ],
            false,
        );
        assert!((line_glyph_px(&line, None) - 60.0).abs() < 1e-3);
        assert!((line_text_px(&line, None, 1.0) - 54.0).abs() < 1e-3);
        // Single char: same box-height rule, no pitch fallback.
        let single = line_with_boxes("あ", vec![BoundingBox::new(0, 0, 40, 60, 1.0)], false);
        assert!((line_text_px(&single, None, 1.0) - 54.0).abs() < 1e-3);
        // Tracked ASCII: the box height still wins (49 × 0.9), not the pitch.
        let tracked = line_with_boxes(
            "Memo1Memo",
            (0..9).map(|i| BoundingBox::new(i * 35, 0, 35, 49, 1.0)).collect(),
            false,
        );
        assert!((line_text_px(&tracked, None, 1.0) - 44.1).abs() < 1e-3);
        // Vertical: the cell length (height) is the basis.
        let vertical = line_with_boxes(
            "翻訳",
            vec![
                BoundingBox::new(0, 0, 40, 80, 1.0),
                BoundingBox::new(0, 80, 40, 80, 1.0),
            ],
            true,
        );
        assert!((line_text_px(&vertical, None, 1.0) - 72.0).abs() < 1e-3);
        // Rotated line: the upright frame's cross axis (the fitted rect's
        // short side), not the AABB-inflated char box height. Horizontal
        // frames measure their local height…
        let quad = RotatedBox::new(50.0, 50.0, 120.0, 30.0, 10.0f32.to_radians(), 1.0);
        assert!((line_glyph_px(&vertical, Some(&quad)) - 30.0).abs() < 1e-3);
        // …and vertical ones their local width (mobile `cropW`); using the
        // height would size tategaki glyphs to the whole column length.
        let column = RotatedBox::new(50.0, 50.0, 30.0, 120.0, 10.0f32.to_radians(), 1.0);
        assert!(column.is_vertical());
        assert!((line_glyph_px(&vertical, Some(&column)) - 30.0).abs() < 1e-3);
        // No measurable box: mobile returns before drawing.
        let empty = line_with_boxes("あ", vec![], false);
        assert_eq!(line_text_px(&empty, None, 1.0), 0.0);
        // The zoom transform scales the source size.
        assert!((line_text_px(&line, None, 2.0) - 108.0).abs() < 1e-3);
    }

    /// The pre-warm freeze: before ImageReady, img_w/img_h are 1, so a scale
    /// computed then is min(window_w, window_h) (~960) instead of ~1. That
    /// scaled glyphs to thousands of px and `embolden` took seconds per line.
    #[test]
    fn content_scale_needs_a_real_image_size() {
        assert_eq!(content_scale(960.0, 1018.0, 1, 1, 1.0), None);
        assert_eq!(content_scale(960.0, 1018.0, 0, 0, 1.0), None);

        let normal = content_scale(960.0, 1018.0, 960, 1018, 1.0).expect("image known");
        assert!((normal - 1.0).abs() < 1e-3, "expected ~1.0, got {normal}");

        let zoomed = content_scale(1280.0, 1357.0, 960, 1018, 2.0).expect("image known");
        assert!((zoomed - 2.6667).abs() < 1e-3, "expected ~2.667, got {zoomed}");
    }

    /// Mobile per-glyph fit: only the overflowing glyph shrinks, along the
    /// reading axis, and halfwidth ink gets 2× the box before the fit.
    #[test]
    fn glyph_fit_scales_only_the_overflowing_axis() {
        // Horizontal: a 100px-wide glyph in a 40px-wide box scales to 0.368.
        let s = glyph_fit_scale(false, false, (40.0, 60.0), (100.0, 30.0));
        assert!((s - 40.0 * 0.92 / 100.0).abs() < 1e-4, "s={s}");
        // Its height is irrelevant (cross-axis ink is not capped).
        assert_eq!(glyph_fit_scale(false, false, (40.0, 60.0), (30.0, 500.0)), 1.0);
        // Vertical: fit runs on the height.
        let s = glyph_fit_scale(true, false, (40.0, 60.0), (100.0, 120.0));
        assert!((s - 60.0 * 0.92 / 120.0).abs() < 1e-4, "s={s}");
        // Halfwidth: the limit doubles, so the same ink fits.
        assert_eq!(glyph_fit_scale(false, true, (40.0, 60.0), (60.0, 30.0)), 1.0);
        assert!((glyph_fit_scale(false, true, (40.0, 60.0), (100.0, 30.0))
            - 40.0 * 0.92 * 2.0 / 100.0)
            .abs() < 1e-4);
    }

    /// Mobile `RotatedGeometry.tiltDeg` on our axes reduces to the stored
    /// x-axis angle for both orientations — horizontal: `atan2(x.y, x.x)`
    /// with x the reading axis; vertical: `atan2(-y.x, y.y)` with y the
    /// reading axis, which is the same value, not 90° away from it.
    #[test]
    fn glyph_tilt_follows_the_reading_axis() {
        let q = |deg: f32| RotatedBox::new(0.0, 0.0, 40.0, 20.0, deg.to_radians(), 1.0);
        // Axis-aligned: no rotation on either orientation. A straight
        // vertical frame's x axis is horizontal, so its angle is 0.
        assert_eq!(glyph_tilt(Some(&q(0.0))), 0.0);
        assert_eq!(glyph_tilt(None), 0.0);
        // Sub-tolerance tilt is not "rotated" (mobile AXIS_ALIGNED_TOL
        // widened by the fit quantization: a 40px side allows ~2.1°).
        assert_eq!(glyph_tilt(Some(&q(1.0))), 0.0);
        // Above the band the glyphs turn by the frame angle exactly.
        assert!((glyph_tilt(Some(&q(5.0))) - 5.0f32.to_radians()).abs() < 1e-5);
        // Regression: the reported tategaki column (fitted 32×188, −1.1°
        // past vertical) must draw essentially upright at −1.1°, not roll
        // ~89° sideways as when the angle was read as a long axis.
        let column = RotatedBox::new(100.0, 100.0, 32.0, 188.0, -1.1f32.to_radians(), 1.0);
        assert!(column.is_vertical() && column.is_rotated());
        let tilt = glyph_tilt(Some(&column));
        assert!(
            (tilt + 1.1f32.to_radians()).abs() < 1e-4,
            "tategaki column drew at {tilt} rad"
        );
    }

    /// Vertical glyphs centre their own ink on the cell (mobile
    /// `LineOverlayView`), rather than keeping their vmtx place in the em
    /// cell: the vertical comma sits mid-cell, not top-right.
    #[test]
    fn vertical_glyphs_centre_their_own_ink() {
        let cache = GlyphCache::new_for(FontFace::Sans).expect("bundled JP font");
        let px = 54u32;
        let (cx, cy) = (100.0f32, 100.0f32);
        let ref_ink = {
            let g = draw_glyph(&cache, 'あ', true, px, false).expect("glyph");
            (g.w as f32, g.h as f32, g.xmin as f32, g.ymin as f32)
        };
        let place = |ch: char| -> (f32, f32, f32, f32) {
            let g = draw_glyph(&cache, ch, true, px, false).expect("glyph");
            let glyph = (g.w as f32, g.h as f32, g.xmin as f32, g.ymin as f32);
            let (dx, dy) = glyph_ink_origin(true, cx, cy, glyph, ref_ink);
            (dx, dy, glyph.0, glyph.1)
        };
        // U+FE11 VERTICAL IDEOGRAPHIC COMMA: its own ink is centred on the
        // cell along the reading axis (vmtx would have put it above centre).
        let (_, dy, _, gh) = place('\u{FE11}');
        assert!(
            ((dy + gh / 2.0) - cy).abs() < 0.5,
            "comma ink centre {} should be the cell centre {cy}",
            dy + gh / 2.0
        );
        // Across the axis the reference `あ` anchors the ink, so a bracket's
        // x is its own bearing relative to the reference ink centre.
        let (dx, _, gw, _) = place('\u{FE41}');
        let g = draw_glyph(&cache, '\u{FE41}', true, px, false).expect("glyph");
        let ref_centre = ref_ink.2 + ref_ink.0 / 2.0;
        let own_centre = g.xmin as f32 + gw / 2.0;
        assert!(
            (dx + gw / 2.0 - (cx + own_centre - ref_centre)).abs() < 0.5,
            "vertical x must ride the reference ink centre"
        );
        // Kanji: ink centre on the cell centre in y.
        let (_, dy, _, gh) = place('漢');
        assert!(((dy + gh / 2.0) - cy).abs() < px as f32 * 0.15);
    }
    /// Vertical glyph selection comes from the font's GSUB `vert`/`vrt2`
    /// feature, with the Unicode vertical form only filling the entries the
    /// font does not cover.
    #[test]
    fn vertical_glyphs_come_from_gsub() {
        let cache = GlyphCache::new_for(FontFace::Sans).expect("bundled JP font");
        let gid_of = |ch: char| -> u16 {
            let borrowed = cache.borrow();
            let face = borrowed.vface.as_ref().expect("vface");
            face.glyph_index(ch).expect("cmap").0
        };
        // GSUB-covered: 、 resolves to the font's vertical ideographic comma
        // glyph (the same glyph U+FE11 maps to), and horizontally it stays.
        let comma_v = cache.borrow_mut().glyph_id('、', true);
        let comma_h = cache.borrow_mut().glyph_id('、', false);
        let stop_v = cache.borrow_mut().glyph_id('。', true);
        let bracket_v = cache.borrow_mut().glyph_id('「', true);
        let dash_v = cache.borrow_mut().glyph_id('ー', true).unwrap();
        let bang_v = cache.borrow_mut().glyph_id('！', true);
        assert_eq!(comma_v, Some(gid_of('\u{FE11}')));
        assert_eq!(comma_h, Some(gid_of('、')));
        assert_eq!(stop_v, Some(gid_of('\u{FE12}')));
        assert_eq!(bracket_v, Some(gid_of('\u{FE41}')));
        // ー has its own GSUB vertical glyph, distinct from the horizontal and
        // from the manual table's FE31 pick.
        assert_ne!(dash_v, gid_of('ー'));
        assert_ne!(dash_v, gid_of('\u{FE31}'));
        // ！ has no vert entry; the Unicode vertical form is the fallback.
        assert_eq!(bang_v, Some(gid_of('\u{FE15}')));
    }

    /// The batch apply must fill every detection slot in one pass: placeholders
    /// survive, recognized lines reach the overlay state, and the cursor seeds
    /// on the first non-empty line.
    #[test]
    fn apply_ocr_batch_fills_all_slots_in_one_pass() {
        let mut v = OcrViewer::new_empty(1280.0, 720.0, jpdict_core::overlay_font::FontFace::Sans);
        v.set_image(
            iced::widget::image::Handle::from_bytes(Vec::new()),
            Vec::new(),
            2400,
            1080,
        );
        let batch = vec![
            DetectedAnnotation {
                bbox: BoundingBox::new(0, 0, 10, 10, 0.5),
                quad: None,
                line: None,
            },
            DetectedAnnotation {
                bbox: BoundingBox::new(0, 20, 60, 60, 0.9),
                quad: None,
                line: Some(line_with_boxes(
                    "日本",
                    vec![
                        BoundingBox::new(0, 20, 60, 60, 0.9),
                        BoundingBox::new(60, 20, 60, 60, 0.9),
                    ],
                    false,
                )),
            },
            DetectedAnnotation {
                bbox: BoundingBox::new(0, 80, 60, 60, 0.9),
                quad: None,
                line: Some(line_with_boxes(
                    "語",
                    vec![BoundingBox::new(0, 80, 60, 60, 0.9)],
                    false,
                )),
            },
        ];
        v.apply_ocr_batch(batch);

        assert_eq!(v.annotations.len(), 3, "every detection slot survives");
        assert!(v.annotations[0].line.is_none(), "placeholder stays a box");
        assert_eq!(v.annotations[1].line.as_ref().unwrap().text, "日本");
        assert_eq!(v.annotations[2].line.as_ref().unwrap().text, "語");
        assert_eq!(v.det_box_count, 3, "HUD box count comes from the batch");

        assert_eq!(v.state.active_line_results.len(), 3);
        assert!(v.state.active_line_results[0].is_none());
        assert_eq!(
            v.state.active_line_results[1].as_ref().unwrap().text,
            "日本"
        );
        assert_eq!(
            v.state.active_line_results[2].as_ref().unwrap().text,
            "語"
        );
        // Cursor seeds on the first non-empty line, not on the placeholder.
        assert_eq!(v.state.current_tapped_line_idx, 1);
        assert_eq!(v.state.current_tapped_char_idx_in_line, 0);
        // Nav graph is left dirty for the next navigation/render.
        assert!(v.state.nav_graph.is_none());
    }

    /// A batch of detection-only placeholders must not seed the cursor.
    #[test]
    fn apply_ocr_batch_without_text_leaves_cursor_unset() {
        let mut v = OcrViewer::new_empty(1280.0, 720.0, jpdict_core::overlay_font::FontFace::Sans);
        v.apply_ocr_batch(vec![DetectedAnnotation {
            bbox: BoundingBox::new(0, 0, 10, 10, 0.5),
            quad: None,
            line: None,
        }]);
        assert_eq!(v.annotations.len(), 1);
        assert!(
            v.state.active_line_results.is_empty(),
            "a placeholder is not an overlay line result"
        );
        assert_eq!(v.state.current_tapped_line_idx, -1);
        assert_eq!(v.state.current_tapped_char_idx_in_line, -1);
    }

    /// The definition flow packs inline cells at the available width: two
    /// 15px characters per 40px line, ruby cells move down whole, ASCII words
    /// stay together when they fit on the next line, and an oversized cell
    /// still renders on its own line.
    #[test]
    fn inline_flow_wraps_at_the_available_width() {
        assert_eq!(inline_char_width('あ', 15.0), 15.0);
        assert_eq!(inline_char_width('a', 15.0), 7.5);

        let chars: Vec<InlineCell> = "あああ".chars().map(InlineCell::Char).collect();
        let lines = OcrViewer::wrap_inline_cells(chars, 40.0);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].len(), 2);
        assert_eq!(lines[1].len(), 1);

        let cells = vec![
            InlineCell::Char('あ'),
            InlineCell::Ruby { term: "漢".into(), reading: "かん".into(), width: 30.0 },
        ];
        assert_eq!(OcrViewer::wrap_inline_cells(cells, 40.0).len(), 2, "ruby moves down whole");

        let cells = vec![InlineCell::Ruby {
            term: "漢字".into(),
            reading: "かんじ".into(),
            width: 99.0,
        }];
        assert_eq!(
            OcrViewer::wrap_inline_cells(cells, 40.0).len(),
            1,
            "an oversized cell still renders on its own line"
        );

        let text: Vec<InlineCell> = "abc de".chars().map(InlineCell::Char).collect();
        let lines = OcrViewer::wrap_inline_cells(text, 33.0);
        assert_eq!(lines.len(), 2);
        let second: String = lines[1]
            .iter()
            .map(|c| match c {
                InlineCell::Char(ch) => *ch,
                _ => '?',
            })
            .collect();
        assert_eq!(second, "de", "the ASCII word moves down together");
    }

    /// With real font metrics the flow measures true advances (CJK ≈ 1 em,
    /// ASCII narrower) instead of the em-category estimates.
    #[test]
    fn measured_advances_follow_the_font() {
        let bytes = std::fs::read(format!(
            "{}/fonts/NotoSansJP-Regular.ttf",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let font = fontdue::Font::from_bytes(bytes, fontdue::FontSettings::default()).unwrap();
        let em = measured_advance(&font, "あ", DEF_TEXT_SIZE);
        let ascii = measured_advance(&font, "a", DEF_TEXT_SIZE);
        assert!((em - DEF_TEXT_SIZE).abs() < 1.0, "CJK advance ~1em, got {em}");
        assert!(ascii < em * 0.8, "ASCII narrower than CJK, got {ascii}");
    }

    /// Kinsoku: a character that may not start a line (。、) takes the
    /// character before it down rather than sitting alone.
    #[test]
    fn inline_flow_keeps_punctuation_off_the_line_start() {
        let cells: Vec<InlineCell> = "ああ。".chars().map(InlineCell::Char).collect();
        let lines = OcrViewer::wrap_inline_cells(cells, 30.0);
        assert_eq!(lines.len(), 2);
        let second: String = lines[1]
            .iter()
            .map(|c| match c {
                InlineCell::Char(ch) => *ch,
                _ => '?',
            })
            .collect();
        assert_eq!(second, "あ。", "。 rides with the previous character");
    }

    /// Ticket 06 design departure (2026-09-21): definition/example body ruby
    /// uses the surrounding body typeface — a WHITE regular base at body
    /// size with a gray ruby row — instead of mobile's bold-cyan mini ruby
    /// (`OcrOverlayView.createRubyView(..., isMini = true)` via
    /// `createBaseTextView`). Clear names and pinned values so these can
    /// graduate to the conformance spec later.
    #[test]
    fn body_ruby_uses_body_typeface_not_term_display() {
        let body = jpdict_core::ruby_style::ruby_style(true);
        assert_eq!(body.base_size, DEF_TEXT_SIZE, "mini base sits at body size");
        assert_eq!(body.ruby_size, DEF_RUBY_SIZE);
        assert_eq!(OcrViewer::ruby_base(&body), Color::WHITE, "body ruby base matches plain body runs");
        assert_eq!(OcrViewer::ruby_tint(&body), Color::from_rgb(0.75, 0.75, 0.75), "body ruby row stays gray");
        assert!(!body.bold, "body ruby is regular weight, like the surrounding text");
    }

    /// The headword block and term rows keep the full-size bold-cyan term
    /// display (mobile non-mini `createRubyView`: 32sp base / 13sp ruby).
    #[test]
    fn term_ruby_keeps_full_size_term_display() {
        let term = jpdict_core::ruby_style::ruby_style(false);
        assert_eq!(term.base_size, 32.0);
        assert_eq!(term.ruby_size, 13.0);
        assert_eq!(OcrViewer::ruby_base(&term), Color::from_rgb(0.0, 1.0, 1.0), "term base stays cyan");
        assert_eq!(OcrViewer::ruby_tint(&term), Color::from_rgb(0.75, 0.75, 0.75), "term ruby stays gray");
        assert!(term.bold, "term display stays bold");
    }

    /// One term entry with Kanjium pitch data, for the pitch-gate tests.
    fn entry_with_pitch() -> FormattedEntry {
        FormattedEntry {
            term: "分".to_string(),
            reading_groups: vec![FormattedReadingGroup {
                reading: "ぶん".to_string(),
                headwords: vec![FormattedHeadword {
                    kanji: "分".to_string(),
                    onyomi: None,
                    kunyomi: None,
                }],
                sense_groups: Vec::new(),
                is_kanji_entry: false,
                pitch_positions: vec![1],
                render_senses: true,
            }],
            deinflection: None,
            dictionary_name: None,
        }
    }

    /// #43/#86: the pitch switch gates the panel's pitch line — on shows the
    /// reading's downstep row, off hides it even though the data is attached.
    #[test]
    fn pitch_line_renders_only_when_the_switch_is_on() {
        let entry = entry_with_pitch();
        let on = OcrViewer::pitch_items(&entry, true);
        assert_eq!(on, vec![("ぶん".to_string(), vec![1])]);

        assert!(
            OcrViewer::pitch_items(&entry, false).is_empty(),
            "the switch off hides the pitch line"
        );
    }

    /// No pitch data, no line — the switch on must not invent one, and a
    /// kanji (KANJIDIC) group's positions never reach the term pitch row.
    #[test]
    fn pitch_line_needs_real_term_pitch_data() {
        let mut bare = entry_with_pitch();
        bare.reading_groups[0].pitch_positions.clear();
        assert!(OcrViewer::pitch_items(&bare, true).is_empty());

        let mut kanji = entry_with_pitch();
        kanji.reading_groups[0].is_kanji_entry = true;
        assert!(
            OcrViewer::pitch_items(&kanji, true).is_empty(),
            "kanji groups keep their 訓/音 rows, not a pitch line"
        );
    }

    /// The viewer ships with the line off, matching mobile's default-off
    /// toggle; `main.rs` flips it on from the saved setting at startup.
    #[test]
    fn pitch_line_ships_off() {
        let viewer =
            OcrViewer::new_empty(1280.0, 720.0, jpdict_core::overlay_font::FontFace::Sans);
        assert!(!viewer.show_pitch, "the pitch line ships off");
    }

    /// One plain term row for the chain tests below.
    fn plain_term(term: &str, reading: &str) -> DictionaryEntry {
        DictionaryEntry {
            id: 1,
            kanji: term.to_string(),
            reading: reading.to_string(),
            definitions: r#"["to eat"]"#.to_string(),
            rules: "v1".to_string(),
            popularity: 0,
            dictionary_id: 1,
            onyomi: None,
            kunyomi: None,
            jlpt: None,
        }
    }

    /// #07 (mobile parity `OcrOverlayView.createDeinflectionRow`): a
    /// populated chain renders "surface → term" plus one reason chip per
    /// step. Pin the chain the row consumes, built through the same
    /// `format_dictionary_results` path `lookup` uses.
    #[test]
    fn deinflected_entry_carries_reasons_for_the_chain_row() {
        let mut state = OcrOverlayState::new(1024.0, 768.0);
        let matches = vec![TermMatch {
            term: "食べる".to_string(),
            entries: vec![plain_term("食べる", "たべる")],
            chain: Some(DeinflectionChain {
                surface: "食べた".to_string(),
                steps: vec!["past".to_string()],
            }),
        }];
        let out = state.format_dictionary_results(&matches, &HashMap::new());
        assert_eq!(out.len(), 1);
        let chain = out[0]
            .deinflection
            .as_ref()
            .expect("a populated chain for the row");
        assert_eq!(chain.surface, "食べた");
        assert_eq!(chain.steps, vec!["past"]);
    }

    /// #07: a direct (non-deinflected) match carries no chain, so
    /// `dictionary_panel`'s `if let Some` renders no row and no stray chips.
    #[test]
    fn direct_entry_has_no_chain_so_no_row() {
        let mut state = OcrOverlayState::new(1024.0, 768.0);
        let matches = vec![TermMatch {
            term: "食べる".to_string(),
            entries: vec![plain_term("食べる", "たべる")],
            chain: None,
        }];
        let out = state.format_dictionary_results(&matches, &HashMap::new());
        assert_eq!(out.len(), 1);
        assert!(
            out[0].deinflection.is_none(),
            "a direct match renders exactly as before — no chain row"
        );
    }

    /// The tuning HUD ships hidden and F1 toggles it both ways.
    #[test]
    fn det_hud_ships_hidden_and_f1_toggles_it() {
        let mut viewer =
            OcrViewer::new_empty(1280.0, 720.0, jpdict_core::overlay_font::FontFace::Sans);
        assert!(!viewer.det_hud_visible, "the tuning HUD ships hidden");
        viewer.toggle_det_hud();
        assert!(viewer.det_hud_visible, "first F1 shows the HUD");
        viewer.toggle_det_hud();
        assert!(!viewer.det_hud_visible, "second F1 hides it again");
    }
}
