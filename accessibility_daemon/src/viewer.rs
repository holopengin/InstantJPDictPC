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

use crate::data::db::DictionaryDatabase;
use crate::data::models::DictionaryEntry;
use crate::models::*;
use crate::overlay_state::OcrOverlayState;
use crate::util::deinflector::Deinflector;
use crate::util::japanese::{is_half_width, to_vertical_glyph};

use fontdue::Font;
use iced::widget::image::Handle as ImageHandle;

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

struct CachedGlyph {
    /// Ink bitmap size in pixels. Zero means the glyph has no ink (a space,
    /// or `.notdef` in a font without one) — mobile skips those outright.
    w: u32,
    h: u32,
    xmin: i32,
    ymin: i32,
    handles: [ImageHandle; 2], // [pink, yellow]
}

/// Resolve the Japanese UI font file used by BOTH the OCR overlay glyph
/// cache and the iced dictionary panel (they must render identically).
/// Binary-relative paths first (AppImage deployment), then a bundled
/// `fonts/` dir, then system Noto CJK installs.
pub fn find_jp_font_path() -> Option<std::path::PathBuf> {
    let exe_path = std::env::current_exe().ok();
    let exe_dir = exe_path.as_ref().and_then(|p| p.parent());
    let exe_font = exe_dir.map(|d| d.join("fonts").join("NotoSansJP-Regular.ttf"));
    let appdir = std::env::var("APPDIR").ok();
    let appdir_font = appdir
        .as_ref()
        .map(|d| std::path::Path::new(d).join("usr").join("bin").join("fonts").join("NotoSansJP-Regular.ttf"));
    let candidates = [
        exe_font.as_ref().map(|p| p.as_path()),
        appdir_font.as_ref().map(|p| p.as_path()),
        Some(std::path::Path::new("fonts/NotoSansJP-Regular.ttf")),
        Some(std::path::Path::new("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")),
        Some(std::path::Path::new("/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc")),
    ];
    candidates.iter().flatten().find(|p| p.exists()).map(|p| p.to_path_buf())
}

/// Lazily rasterizes characters at requested pixel sizes, caching RGBA
/// image handles. Thread‑safe via RefCell for interior mutability.
pub struct GlyphCache {
    font: Font,
    /// Rasterized glyphs keyed by `(glyph id, px)`: the id may be a GSUB
    /// substitution (vertical forms), not just the cmap glyph of the char.
    cache: HashMap<(u16, u32), CachedGlyph>,
    /// GSUB `vert`/`vrt2` resolution results (source glyph -> vertical glyph).
    vert_cache: HashMap<u16, u16>,
    /// Font-parser view of the same bytes, for cmap and GSUB `vert`/`vrt2`
    /// resolution (fontdue exposes neither).
    vface: Option<ttf_parser::Face<'static>>,
}

impl GlyphCache {
    pub fn new() -> Option<Rc<RefCell<Self>>> {
        Self::from_path(find_jp_font_path())
    }

    /// Load the cache from a specific font file. A missing path or an
    /// unreadable face yields `None` and the overlay draws boxes without
    /// glyphs — mobile falls back to the platform face and never crashes the
    /// overlay for a missing bundled font (OverlayFont #84).
    fn from_path(path: Option<std::path::PathBuf>) -> Option<Rc<RefCell<Self>>> {
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
        match Font::from_bytes(data, fontdue::FontSettings::default()) {
            Ok(font) => Some(Rc::new(RefCell::new(GlyphCache {
                font,
                cache: HashMap::new(),
                vert_cache: HashMap::new(),
                vface,
            }))),
            Err(_) => {
                eprintln!("[GlyphCache] failed to parse font {}", path.display());
                None
            }
        }
    }

    /// Ensure both tinted handles exist for (char, px_size) and return one
    /// with the ink metrics. The highlighted handle is fake-bolded, like
    /// mobile (`paint.isFakeBoldText`): the bundled faces ship Regular only.
    fn get_handle(&mut self, gid: u16, px: u32, highlighted: bool) -> Option<(&ImageHandle, u32, u32, i32, i32)> {
        let entry = self.cache.entry((gid, px)).or_insert_with(|| {
            let (metrics, coverage) = self.font.rasterize_config(fontdue::layout::GlyphRasterConfig {
                glyph_index: gid,
                px: px as f32,
                font_hash: 0,
            });
            // Keep the true ink size: zero is how a blank glyph reports itself.
            let w = metrics.width as u32;
            let h = metrics.height as u32;
            let pink = Self::make_handle(w, h, &coverage, OVERLAY_FG.0, OVERLAY_FG.1, OVERLAY_FG.2);
            let bold = embolden(&coverage, w, h, fake_bold_radius(px));
            let yellow = Self::make_handle(w, h, &bold, OVERLAY_HL.0, OVERLAY_HL.1, OVERLAY_HL.2);
            CachedGlyph { w, h, xmin: metrics.xmin, ymin: metrics.ymin, handles: [pink, yellow] }
        });
        let idx = if highlighted { 1 } else { 0 };
        Some((&entry.handles[idx], entry.w, entry.h, entry.xmin, entry.ymin))
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

    /// Pre-warm the cache for a glyph at the given pixel size.
    /// Called from the update handler so glyph rasterization happens off
    /// the view/draw path, preventing first-frame stutter.
    pub fn ensure_glyph(&mut self, gid: u16, px_size: u32) {
        self.cache.entry((gid, px_size)).or_insert_with(|| {
            let (metrics, coverage) = self.font.rasterize_config(fontdue::layout::GlyphRasterConfig {
                glyph_index: gid,
                px: px_size as f32,
                font_hash: 0,
            });
            let w = metrics.width as u32;
            let h = metrics.height as u32;
            let pink = Self::make_handle(w, h, &coverage, OVERLAY_FG.0, OVERLAY_FG.1, OVERLAY_FG.2);
            let bold = embolden(&coverage, w, h, fake_bold_radius(px_size));
            let yellow = Self::make_handle(w, h, &bold, OVERLAY_HL.0, OVERLAY_HL.1, OVERLAY_HL.2);
            CachedGlyph { w, h, xmin: metrics.xmin, ymin: metrics.ymin, handles: [pink, yellow] }
        });
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
            let vch = crate::util::japanese::to_vertical_glyph(ch);
            if vch != ch {
                face.glyph_index(vch).map(|g| g.0)
            } else {
                None
            }
        });
        let vert = self.gsub_vert_glyph(gid).or(fallback).unwrap_or(gid);
        self.vert_cache.insert(gid, vert);
        Some(vert)
    }

    /// GSUB `vert`/`vrt2` single substitution for a glyph, if the font has
    /// one. Both features carry the same lookups in this font; the union of
    /// their single-substitution subtables is applied.
    fn gsub_vert_glyph(&self, gid: u16) -> Option<u16> {
        use ttf_parser::gsub::{SingleSubstitution, SubstitutionSubtable};

        let face = self.vface.as_ref()?;
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

}

/// Skia's fake-bold stroke is ~1/24 of the text size end to end, so the
/// coverage grows by about 1/48 of the size per side.
fn fake_bold_radius(px: u32) -> i32 {
    ((px as f32 / 48.0).round() as i32).max(1)
}

/// Grow a glyph's coverage by `radius` pixels in every direction — the
/// bitmap equivalent of `paint.isFakeBoldText`, used for highlighted glyphs
/// because the bundled faces have Regular only.
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

/// A rasterized glyph ready to draw: the ink bitmap plus the font metrics
/// needed to place it on its natural baseline.
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
    gid: u16,
    px_size: u32,
    highlighted: bool,
) -> Option<RasterGlyph> {
    let mut c = cache.borrow_mut();
    let (handle, w, h, xmin, ymin) = c.get_handle(gid, px_size, highlighted)?;
    Some(RasterGlyph { handle: handle.clone(), w, h, xmin, ymin })
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
/// *long axis* angle normalised to [-90°, 90°), so for a vertical line it
/// reads as `φ - 90°` where `φ` is the column's clockwise tilt from
/// straight down — the glyph tilt itself is `φ`, i.e. `angle + 90°`
/// re-normalised. Horizontal lines already measure `φ`. Unrotated lines
/// (either orientation) draw exactly as before.
fn glyph_tilt(quad: Option<&RotatedBox>, is_vertical: bool) -> f32 {
    let Some(q) = quad.filter(|q| q.is_rotated()) else {
        return 0.0;
    };
    let mut tilt = if is_vertical { q.angle + std::f32::consts::FRAC_PI_2 } else { q.angle };
    while tilt >= std::f32::consts::FRAC_PI_2 {
        tilt -= std::f32::consts::PI;
    }
    while tilt < -std::f32::consts::FRAC_PI_2 {
        tilt += std::f32::consts::PI;
    }
    tilt
}

/// Mobile `LineResult.glyphSizePx()`: the source-pixel size the overlay
/// measures a line's glyphs against. The default path is the tallest char
/// box — the detector box height for a horizontal line, the 1em cell for a
/// vertical one. A rotated line measures its upright frame's cross axis
/// instead (`RotatedBox.h`, the short side of the fitted rect), because its
/// char boxes are AABBs of rotated cells and would oversize the glyphs.
/// Zero means the line has no measurable box: mobile skips it.
fn line_glyph_px(line: &LineResult, quad: Option<&RotatedBox>) -> f32 {
    if let Some(q) = quad.filter(|q| q.is_rotated()) {
        return q.h.max(1.0);
    }
    line.char_boxes.iter().map(|b| b.h as f32).fold(0.0, f32::max)
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

/// Mobile `OcrEngine.GAP_CHAR`: the placeholder a dropped character becomes.
pub const GAP_CHAR: char = '\u{25CC}';
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
    pub nav_edges_initial: Option<Vec<[usize; 4]>>,
    pub nav_edges_final: Option<Vec<[usize; 4]>>,
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
                    let ref_gid = cache.borrow_mut().glyph_id('あ', line.is_vertical);
                    let ref_ink = ref_gid
                        .and_then(|gid| draw_glyph(cache, gid, em_px, false))
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
                        let Some(gid) = cache.borrow_mut().glyph_id(ch, line.is_vertical) else { continue };
                        let Some(g) = draw_glyph(cache, gid, em_px, highlighted) else { continue };
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

                        let tilt = glyph_tilt(annotation.quad.as_ref(), line.is_vertical);
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
    /// `handle_ocr_recognition_result` skips `set_single_line_result` for
    /// these lines, preserving the user's edit against incoming OCR results.
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
}

impl OcrViewer {
    /// Create a minimal viewer with no image, no db, no annotations.
    /// Everything is populated asynchronously via bootstrap events.
    /// `window_w` / `window_h` should be the native screen resolution (in physical pixels).
    pub fn new_empty(window_w: f32, window_h: f32) -> Self {
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
            glyph_cache: GlyphCache::new(),
            screen_physical_width: window_w,
            native_scale: 0.0,
            debounced_ui_scale: window_w / 1280.0,
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
                let matched_term: String =
                    full_line_text.chars().skip(char_idx).take(term_len).collect();
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
                        let formatted = self.state.format_dictionary_results(
                            &[(kanji_str.clone(), kanji_only)],
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

    /// Called when a recognition result streams in from the background OCR thread.
    /// Replaces the detection-only annotation (bbox + line: None) at the given
    /// index with the full annotation (bbox + line with text and char_boxes).
    pub fn handle_ocr_recognition_result(&mut self, index: usize, mut annotation: DetectedAnnotation) {
        // Mobile `addLineToResults` runs BlankGaps before layout (#44
        // Feature 2), so the placeholder is part of the line from here on:
        // text, char boxes and alternatives all grow together.
        if let Some(line) = annotation.line.take() {
            annotation.line = Some(apply_blank_gaps(&line));
        }
        let line = annotation.line.clone();
        let quad = annotation.quad;
        if line.is_some() {
            let has_text = line.as_ref().map(|l| !l.text.is_empty()).unwrap_or(false);
            println!(
                "[VIEWER] recv ann idx={}: is_vertical={}, has_text={}, char_boxes={}",
                index,
                line.as_ref().map(|l| l.is_vertical).unwrap_or(false),
                has_text,
                line.as_ref().map(|l| l.char_boxes.len()).unwrap_or(0)
            );
        }
        // Extend annotations vec if this is a new box beyond current length
        if self.annotations.len() <= index {
            // Use make_mut to grow in-place without cloning the whole vec
            let anns = std::rc::Rc::make_mut(&mut self.annotations);
            anns.resize(index + 1, DetectedAnnotation {
                bbox: BoundingBox::new(0, 0, 0, 0, 0.0),
                quad: None,
                line: None,
            });
        }

        // Replace the annotation at the given index — in-place via make_mut
        let anns = std::rc::Rc::make_mut(&mut self.annotations);
        anns[index] = annotation;

        // If the annotation has a line result, update the overlay state
        // UNLESS the user has edited this line's text — skip overwrite in
        // that case to preserve the user's edit.
        if let Some(ref line) = line {
            if !self.edited_lines.contains(&index) {
                self.state.set_single_line_result(index, line.clone());
            }
        }

        // Pre-warm glyph cache for this annotation's characters so view()
        // doesn't rasterize them on the draw path (which causes visible
        // stutter on the first frame new characters appear).
        if let Some(ref line) = line {
            if let Some(ref gc) = self.glyph_cache {
                // Use cached total_scale from last view() frame — avoids
                // computing from potentially-not-yet-loaded img_w/img_h (1 vs real).
                let total_scale = self.last_total_scale.get();
                // Exactly the draw pass's per-line text size.
                let px = line_text_px(line, quad.as_ref(), total_scale).round().clamp(1.0, 1024.0) as u32;
                let mut cache = gc.borrow_mut();
                let vertical = line.is_vertical;
                if let Some(gid) = cache.glyph_id('あ', false) {
                    cache.ensure_glyph(gid, px);
                }
                for ch in line.text.chars() {
                    if let Some(gid) = cache.glyph_id(ch, vertical) {
                        cache.ensure_glyph(gid, px);
                    }
                }
            }
        }

        // Mark nav graph dirty — will be rebuilt lazily on next navigation or
        // render, instead of rebuilding on every streaming result.
        self.state.mark_nav_dirty();

        // Update cursor if not yet set
        if self.state.current_tapped_line_idx < 0 || self.state.current_tapped_char_idx_in_line < 0 {
            self.state.ensure_cursor_position();
        }

        // Keep synced_annotations cache in sync.
        // Uses a NEW Rc with a fresh clone so self.annotations keeps refcount=1.
        // This way Rc::make_mut above mutates in-place without deep-copying the
        // whole Vec (which would happen if refcount > 1).
        *self.synced_annotations.borrow_mut() = Rc::new((*self.annotations).clone());
    }

    pub fn view<'a>(&'a self) -> Element<'a, Message> {
        // Cache total_scale for glyph pre-warming in handle_ocr_recognition_result
        let base_scale = f32::min(
            self.window_width / self.img_w.max(1) as f32,
            self.window_height / self.img_h.max(1) as f32,
        );
        self.last_total_scale.set(base_scale * self.state.current_scale);
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
            return Container::new(Stack::new().push(image_canvas).push(annotation_canvas))
                .width(Length::Fill)
                .height(Length::Fill)
                .into();
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
        Container::new(
            Stack::new()
                .push(image_canvas)
                .push(annotation_canvas)
                .push(content_stack),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    fn dictionary_panel<'a>(&'a self, entries: Rc<Vec<FormattedEntry>>) -> Container<'a, Message> {
        let mut content = Column::new().padding(4).spacing(4).width(Length::Fill);
        if entries.is_empty() {
            content = content.push(Text::new("No dictionary entries found.").size(14));
        }
        for entry in entries.iter() {
            let mut entry_col = Column::new().spacing(4).width(Length::Fill);
            for group in &entry.reading_groups {
                entry_col = entry_col.push(Self::headword_section(group.clone()));
                for sg in &group.sense_groups {
                    entry_col = entry_col.push(Self::sense_group(sg.clone()));
                }
                entry_col = entry_col.push(iced::widget::Space::new().height(Pixels(4.0)));
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

    fn headword_section(group: FormattedReadingGroup) -> Container<'static, Message> {
        let cyan = Color::from_rgb(0.0, 1.0, 1.0);      // Android CYAN
        let gray = Color::from_rgb(0.75, 0.75, 0.75);   // Android LTGRAY (#BEBEBE)
        let mut content = Column::new().spacing(2);

        if group.is_kanji_entry {
            for hw in &group.headwords {
                let mut row = Row::new().spacing(6).align_y(alignment::Vertical::Center);
                row = row.push(Text::new(hw.kanji.clone()).size(36).color(cyan)
                    .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }));
                if let Some(o) = &hw.onyomi { row = row.push(Text::new(format!("音: {o}")).size(14).color(gray)); }
                if let Some(k) = &hw.kunyomi { row = row.push(Text::new(format!("訓: {k}")).size(14).color(gray)); }
                content = content.push(row);
            }
        } else {
            let mut row = Row::new().spacing(4).align_y(alignment::Vertical::Center);
            for (i, hw) in group.headwords.iter().enumerate() {
                if hw.kanji == group.reading {
                    row = row.push(Text::new(hw.kanji.clone()).size(24).color(cyan)
                        .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }));
                } else {
                    let mut rc = Column::new().align_x(alignment::Horizontal::Center).spacing(1);
                    rc = rc.push(Text::new(group.reading.clone()).size(14).color(gray));
                    rc = rc.push(Text::new(hw.kanji.clone()).size(24).color(cyan)
                        .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }));
                    row = row.push(rc);
                }
                if i + 1 < group.headwords.len() {
                    row = row.push(Text::new("、").size(20).color(gray));
                }
            }
            content = content.push(row);
        }
        Container::new(content)
    }

    fn tag_color(tag: &str) -> Color {
        match tag {
            t if t == "pos" || t == "v" || t == "adj" || t == "adj-i" || t == "adj-na" => Color::from_rgb(0.23, 0.35, 0.48),
            t if t == "n" || t == "adv" || t == "pn" => Color::from_rgb(0.23, 0.48, 0.35),
            t if t.starts_with("jlpt") || t.starts_with("grade") || t == "★" => Color::from_rgb(0.48, 0.23, 0.23),
            _ => Color::from_rgb(0.27, 0.27, 0.27),
        }
    }

    fn sense_group(sg: FormattedSenseGroup) -> Column<'static, Message> {
        let cyan = Color::from_rgb(0.0, 1.0, 1.0); // Android CYAN
        let gray = Color::from_rgb(0.75, 0.75, 0.75); // Android LTGRAY (#BEBEBE)
        let white = Color::WHITE;
        let mut content = Column::new().spacing(3);

        if !sg.tags.is_empty() {
            let mut tag_row = Row::new().spacing(3).align_y(alignment::Vertical::Center);
            for tag in &sg.tags {
                let bg = Self::tag_color(tag);
                tag_row = tag_row.push(
                    Container::new(Text::new(tag.clone()).size(13).color(white)
                        .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }))
                    .padding([1.0, 3.0])
                    .style(move |_t: &Theme| container::Style {
                        background: Some(iced::Background::Color(bg)),
                        border: iced::Border { radius: 3.0.into(), ..Default::default() },
                        ..Default::default()
                    }),
                );
            }
            content = content.push(tag_row);
        }

        for sense in &sg.senses {
            let mut sense_row = Row::new().spacing(3).align_y(alignment::Vertical::Top);
            sense_row = sense_row.push(Text::new(format!("{}. ", sense.index)).size(16).color(white));
            let mut nodes_col = Column::new().spacing(1).width(Length::Fill);
            for node in &sense.nodes {
                match node {
                    DefinitionNode::Text(t) => {
                        nodes_col = nodes_col.push(
                            Text::new(t.clone()).size(14).color(white).width(Length::Fill)
                                .wrapping(iced::widget::text::Wrapping::Word),
                        );
                    }
                    DefinitionNode::Ruby { term, reading } => {
                        if term == reading {
                            nodes_col = nodes_col.push(Text::new(term.clone()).size(16).color(cyan)
                                .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }));
                        } else {
                            let mut rc = Column::new().align_x(alignment::Horizontal::Center).spacing(0);
                            rc = rc.push(Text::new(reading.clone()).size(12).color(gray));
                            rc = rc.push(Text::new(term.clone()).size(16).color(cyan)
                                .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }));
                            nodes_col = nodes_col.push(rc);
                        }
                    }
                    DefinitionNode::Tag { text } => {
                        nodes_col = nodes_col.push(Text::new(format!("[{text}]")).size(14).color(gray));
                    }
                    _ => {}
                }
            }
            sense_row = sense_row.push(nodes_col);
            content = content.push(sense_row);
        }
        content
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

                let vertical_ch = chip_text(&ch.to_string(), self.is_landscape());

                let btn: Element<'a, Message> = Container::new(
                    Text::new(vertical_ch)
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
        assert!(GlyphCache::new().is_some());
    }

    /// Mobile skips glyphs whose measured ink is degenerate (`glyphW <= 0 ||
    /// glyphH <= 0`); a blank cell must not draw a placeholder pixel.
    #[test]
    fn blank_glyphs_report_no_ink() {
        let cache = GlyphCache::new().expect("bundled JP font");
        let gid = cache.borrow_mut().glyph_id(' ', false).expect("space glyph");
        let g = draw_glyph(&cache, gid, 54, false).expect("glyph");
        assert_eq!((g.w, g.h), (0, 0), "space must report no ink");
        let gid = cache.borrow_mut().glyph_id('あ', false).expect("glyph id");
        let g = draw_glyph(&cache, gid, 54, false).expect("glyph");
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
        // short side), not the AABB-inflated char box height.
        let quad = RotatedBox::new(50.0, 50.0, 120.0, 30.0, 10.0f32.to_radians(), 1.0);
        assert!((line_glyph_px(&vertical, Some(&quad)) - 30.0).abs() < 1e-3);
        // No measurable box: mobile returns before drawing.
        let empty = line_with_boxes("あ", vec![], false);
        assert_eq!(line_text_px(&empty, None, 1.0), 0.0);
        // The zoom transform scales the source size.
        assert!((line_text_px(&line, None, 2.0) - 108.0).abs() < 1e-3);
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

    /// Mobile `tiltDeg`: a vertical column's long-axis angle reads 90° off
    /// the glyph tilt; axis-aligned lines and tiny tolerances draw upright.
    #[test]
    fn glyph_tilt_follows_the_reading_axis() {
        let q = |deg: f32| RotatedBox::new(0.0, 0.0, 40.0, 20.0, deg.to_radians(), 1.0);
        // Axis-aligned: no rotation on either orientation.
        assert_eq!(glyph_tilt(Some(&q(0.0)), false), 0.0);
        assert_eq!(glyph_tilt(Some(&q(-90.0)), true), 0.0);
        // Sub-tolerance tilt is not "rotated" (mobile AXIS_ALIGNED_TOL).
        assert_eq!(glyph_tilt(Some(&q(1.0)), false), 0.0);
        // Horizontal line tilted clockwise by 5°.
        assert!((glyph_tilt(Some(&q(5.0)), false) - 5.0f32.to_radians()).abs() < 1e-5);
        // Vertical column tilted clockwise by 10°: long-axis angle -80°.
        assert!(
            (glyph_tilt(Some(&q(-80.0)), true) - 10.0f32.to_radians()).abs() < 1e-5,
            "clockwise tategaki"
        );
        // Vertical column tilted counter-clockwise by 10°: long-axis 80°.
        assert!(
            (glyph_tilt(Some(&q(80.0)), true) + 10.0f32.to_radians()).abs() < 1e-5,
            "counter-clockwise tategaki"
        );
    }

    /// Vertical glyphs centre their own ink on the cell (mobile
    /// `LineOverlayView`), rather than keeping their vmtx place in the em
    /// cell: the vertical comma sits mid-cell, not top-right.
    #[test]
    fn vertical_glyphs_centre_their_own_ink() {
        let cache = GlyphCache::new().expect("bundled JP font");
        let px = 54u32;
        let (cx, cy) = (100.0f32, 100.0f32);
        let ref_ink = {
            let gid = cache.borrow_mut().glyph_id('あ', true).expect("glyph id");
            let g = draw_glyph(&cache, gid, px, false).expect("glyph");
            (g.w as f32, g.h as f32, g.xmin as f32, g.ymin as f32)
        };
        let place = |ch: char| -> (f32, f32, f32, f32) {
            let gid = cache.borrow_mut().glyph_id(ch, true).expect("glyph id");
            let g = draw_glyph(&cache, gid, px, false).expect("glyph");
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
        let gid = cache.borrow_mut().glyph_id('\u{FE41}', true).expect("glyph id");
        let g = draw_glyph(&cache, gid, px, false).expect("glyph");
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
        let cache = GlyphCache::new().expect("bundled JP font");
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
}
