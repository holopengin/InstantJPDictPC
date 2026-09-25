//! Pure per-character box stages shared by the desktop engine and the
//! mobile UniFFI shim (`pipeline-sharing/07-ocr-engine-split`).
//!
//! Everything here is data in / data out over plain types (`&str`, `&[f32]`,
//! RGB8 byte slices, [`crate::models`] geometry) — no `image` types, no ncnn
//! handles, no C++ toolchain — so this module is **not** gated on the
//! `native` feature, unlike [`crate::ocr_engine`] and [`crate::ppocr`].
//! Moved here verbatim from those modules (behaviour unchanged; their callers
//! now delegate); the only additions are the `_with` variants, which take the
//! environment knobs and pre-measured ink widths explicitly so the mobile
//! shim can pass its own `Paint`-measured widths and preference flags instead
//! of the desktop's font/env behaviour, and [`sort_order`], the rect-only
//! reading-order permutation behind mobile `OcrEngine.sortDetectedBoxes`.
//!
//! Ink measurement split: the desktop measures glyph ink half-widths from
//! the bundled font ([`ink_half_widths`], needs only the pure-Rust
//! `ttf-parser` crate); Android measures with its own typeface and passes
//! the widths in. The collision resolution itself ([`resolve_ink_collisions`])
//! is pure float math over those widths either way.

use crate::models::{BoundingBox, DetectedAnnotation, LineResult};

/// Mobile `BOX_LAYOUT_MODE` (0 = legacy uniform columns, 1 = legacy + ink
/// snapping; default 1). `BOX_LAYOUT_MODE=0` disables snapping.
pub(crate) fn box_layout_snap() -> bool {
    std::env::var("BOX_LAYOUT_MODE")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1)
        == 1
}

/// Mobile `BOX_UNIFORM_SIZE` (default true): uniform em widths around the
/// resolved centres.
fn box_uniform_size() -> bool {
    std::env::var("BOX_UNIFORM_SIZE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Mobile `legacyCells` (#49): centre ± cross/2 at `(t + 0.5) · L/seqLen`,
/// clamped to `[0, L]`, sorted by start.
pub(crate) fn legacy_cells(cols: &[f32], seq_len: usize, l: f32, cross: f32) -> Vec<(f32, f32)> {
    let avg_col_w = l / seq_len as f32;
    let half = cross / 2.0;
    let mut cells: Vec<(f32, f32)> = cols
        .iter()
        .map(|&t| {
            let c = (t + 0.5) * avg_col_w;
            ((c - half).max(0.0), (c + half).min(l))
        })
        .collect();
    cells.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    cells
}

/// Mobile `isSnapSkipped`: punctuation whose ink centroid is not the em
/// centre, small kana and midline marks keep their legacy boxes.
fn is_snap_skipped(ch: char) -> bool {
    if "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ".contains(ch) {
        return true;
    }
    "、。．，,．「」『』（）〔〕［］｛｝〈〉《》【】〘〙〚〛'\"\"‘’“”()[]{}-+*/<>＜＞＝…‥︙︰：；"
        .contains(ch)
}

// ─────────────────────────────────────────────────────────────────────────────
// Snap evidence: what the ink actually needs, in one place
// ─────────────────────────────────────────────────────────────────────────────

/// Ink is `lum < INK_BELOW` on a light background.
const INK_BELOW: f32 = 110.0;
/// Ink is `lum > INK_ABOVE` on a dark background.
const INK_ABOVE: f32 = 145.0;
/// The border median above `BG_MEDIAN_ABOVE` means "light background".
const BG_MEDIAN_ABOVE: f32 = 128.0;
/// The border sampling stride, in pixels.
const BORDER_STRIDE: usize = 7;
/// The central cross band that carries the body text (ruby sits outside it).
const BAND_LO: f32 = 0.2;
const BAND_HI: f32 = 0.8;
/// The profile box-blur radius.
const BLUR_RADIUS: i32 = 2;
/// The largest ink count a [`SnapEvidence`] count can carry. A cross axis
/// thicker than this cannot be reduced to counts and must pass pixels instead.
pub const SNAP_MAX_COUNT: usize = 255;

/// The exact integer form of the ink test, for a caller that counts ink
/// without touching a float per pixel.
///
/// Luminance is the channel mean `(r + g + b) / 3` of 8-bit channels, so the
/// numerator is an integer in `0..=765`, and both thresholds are exact
/// multiples of three: `(s as f32 / 3.0) < 110.0` holds exactly when
/// `s <= 329`, and `> 145.0` exactly when `s >= 436`. So `ink_below_sum` is the
/// exclusive upper bound of the ink sums and `ink_above_sum` the inclusive
/// lower bound of the *non*-ink… of the ink sums on a dark background.
/// `snap_spec_integer_agrees_with_float` pins the equivalence over every sum.
pub const INK_BELOW_SUM: i32 = (INK_BELOW * 3.0) as i32;
pub const INK_ABOVE_SUM: i32 = (INK_ABOVE * 3.0) as i32 + 1;

/// The measurement constants [`snap_evidence_from_rgb`] derives a
/// [`SnapEvidence`] with, published so a caller that measures the evidence
/// itself (the mobile FFI boundary, which cannot afford to ship the crop)
/// reads them from here instead of hardcoding them. Changing a snap constant
/// then cannot silently desynchronise the boundary: the record moves with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnapSpec {
    pub ink_below: f32,
    pub ink_above: f32,
    pub bg_median_above: f32,
    pub border_stride: usize,
    pub band_lo: f32,
    pub band_hi: f32,
    pub blur_radius: i32,
    /// The largest count a [`SnapEvidence`] count can hold.
    pub max_count: usize,
    /// [`INK_BELOW_SUM`]: ink on a light background is `channel sum <
    /// ink_below_sum`, so a caller can count with an integer compare.
    pub ink_below_sum: i32,
    /// [`INK_ABOVE_SUM`]: ink on a dark background is `channel sum >=
    /// ink_above_sum`.
    pub ink_above_sum: i32,
}

/// The shipped [`SnapSpec`].
pub fn snap_spec() -> SnapSpec {
    SnapSpec {
        ink_below: INK_BELOW,
        ink_above: INK_ABOVE,
        bg_median_above: BG_MEDIAN_ABOVE,
        border_stride: BORDER_STRIDE,
        band_lo: BAND_LO,
        band_hi: BAND_HI,
        blur_radius: BLUR_RADIUS,
        max_count: SNAP_MAX_COUNT,
        ink_below_sum: INK_BELOW_SUM,
        ink_above_sum: INK_ABOVE_SUM,
    }
}

/// The integer ink test this crate's own float test is equivalent to, as one
/// predicate, so a caller can share it verbatim.
pub fn is_ink_sum(sum: i32, bg_light: bool) -> bool {
    if bg_light {
        sum < INK_BELOW_SUM
    } else {
        sum >= INK_ABOVE_SUM
    }
}

/// The compact ink evidence the snap stage consumes, and the *only* image
/// input it has: the background polarity plus the ink count per position over
/// the fixed central cross band — the profile before its box blur.
///
/// Measured from the crop by [`snap_evidence_from_rgb`], or by the caller (the
/// mobile FFI boundary reduces a 30k-pixel crop to this ~1.7 kB before the
/// crossing). `counts.len()` is the profile length: `pix_w` horizontal,
/// `pix_h` vertical. Everything downstream — the blur, the peak walk, the
/// snapping — is unchanged, so identical counts give identical boxes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SnapEvidence {
    /// `true` when the border median says the background is light (so ink is
    /// dark). The caller sends both count series; this picks one.
    pub bg_light: bool,
    /// Ink count per reading-axis position, inside the central cross band.
    pub counts: Vec<f32>,
}

impl SnapEvidence {
    /// An evidence record from a polarity and the per-position counts.
    pub fn new(bg_light: bool, counts: Vec<f32>) -> Self {
        SnapEvidence { bg_light, counts }
    }

    /// `Some` when the record can drive the snap stage: a polarity is always
    /// usable, but a count that saturated at [`SNAP_MAX_COUNT`] means the
    /// cross axis was too thick to reduce, so the evidence is refused rather
    /// than snapped on wrong numbers.
    pub fn validate(&self) -> Option<&SnapEvidence> {
        if self.counts.iter().any(|&c| c >= SNAP_MAX_COUNT as f32) {
            return None;
        }
        Some(self)
    }
}

/// Background polarity from the border sample: the sorted sample's *upper*
/// median against [`BG_MEDIAN_ABOVE`]. (`char_placement` uses the true median
/// of the same sample — the two stages are independent, so they keep their own
/// rule.) `None` on an empty sample, the old `border.is_empty()` bail.
pub fn snap_polarity(border_lum: &[f32]) -> Option<bool> {
    if border_lum.is_empty() {
        return None;
    }
    let mut sorted = border_lum.to_vec();
    sorted.sort_by(f32::total_cmp);
    Some(sorted[sorted.len() / 2] > BG_MEDIAN_ABOVE)
}

/// The border sample, at [`BORDER_STRIDE`]: the top/bottom row pairs first,
/// then the left/right column pairs. Callers that measure the evidence
/// themselves sample the same pixels in the same order (the order is the
/// median's only input, so it cannot change the result).
pub fn snap_border_lum(lum: &[f32], pw: usize, ph: usize) -> Vec<f32> {
    let mut border: Vec<f32> = Vec::new();
    if pw == 0 || ph == 0 {
        return border;
    }
    let mut bi = 0usize;
    while bi < pw {
        border.push(lum[bi]);
        border.push(lum[(ph - 1) * pw + bi]);
        bi += BORDER_STRIDE;
    }
    bi = 0;
    while bi < ph {
        border.push(lum[bi * pw]);
        border.push(lum[bi * pw + pw - 1]);
        bi += BORDER_STRIDE;
    }
    border
}

/// The central cross band as row (vertical) or column (horizontal) indices:
/// `[lo, hi)` at [`BAND_LO`]..[`BAND_HI`] of the cross axis, truncated to
/// integers exactly as the profile loop does (for every cross axis the
/// sub-8px guard lets through, `hi > lo`). Published because a caller that
/// measures the evidence has to measure it in this band.
pub fn snap_band_range(cross_len: usize) -> (usize, usize) {
    (
        (cross_len as f32 * BAND_LO) as usize,
        (cross_len as f32 * BAND_HI) as usize,
    )
}

/// The reference's channel-mean luminance (`(r + g + b) / 3`, f32) over a run
/// of RGB8 samples; a trailing partial sample reads as zero. This is the
/// per-pixel arithmetic [`snap_evidence_from_rgb`] applies to a whole crop,
/// published so a boundary measuring a *border sample* uses the crate's own
/// formula rather than a Kotlin restatement of it.
pub fn luminance_from_rgb_samples(rgb: &[u8]) -> Vec<f32> {
    rgb.chunks_exact(3)
        .map(|c| (c[0] as f32 + c[1] as f32 + c[2] as f32) / 3.0)
        .collect()
}

/// RGB8 crop → the snap stage's ink evidence. The measurement half of
/// [`snap_cells_from_profile`], split out so the pixel path and a boundary that
/// measured the same evidence share one implementation (and one set of
/// constants).
pub fn snap_evidence_from_rgb(
    pixels: &[u8],
    pix_w: u32,
    pix_h: u32,
    vertical: bool,
) -> Option<SnapEvidence> {
    let (pw, ph) = (pix_w as usize, pix_h as usize);
    if pw < 8 || ph < 8 || pixels.len() < pw * ph * 3 {
        return None;
    }
    let lum: Vec<f32> = luminance_from_rgb_samples(pixels);
    let bg_light = snap_polarity(&snap_border_lum(&lum, pw, ph))?;
    let prof_len = if vertical { ph } else { pw };
    let mut counts = vec![0.0f32; prof_len];
    if vertical {
        let (x0, x1) = snap_band_range(pw);
        for y in 0..ph {
            let mut m = 0.0f32;
            for x in x0..x1.min(pw) {
                let v = lum[y * pw + x];
                if if bg_light { v < INK_BELOW } else { v > INK_ABOVE } {
                    m += 1.0;
                }
            }
            counts[y] = m;
        }
    } else {
        let (y0, y1) = snap_band_range(ph);
        for x in 0..pw {
            let mut m = 0.0f32;
            for y in y0..y1.min(ph) {
                let v = lum[y * pw + x];
                if if bg_light { v < INK_BELOW } else { v > INK_ABOVE } {
                    m += 1.0;
                }
            }
            counts[x] = m;
        }
    }
    Some(SnapEvidence { bg_light, counts })
}

/// Box blur radius [`BLUR_RADIUS`], clamped at the edges (numpy-free: this is
/// the profile blur `snapCells` has always used).
fn snap_smooth(prof: &[f32]) -> Vec<f32> {
    let prof_len = prof.len();
    (0..prof_len)
        .map(|i| {
            let mut a = 0.0f32;
            let mut c = 0usize;
            for k in -BLUR_RADIUS..=BLUR_RADIUS {
                let j = (i as i32 + k).clamp(0, prof_len as i32 - 1) as usize;
                a += prof[j];
                c += 1;
            }
            a / c as f32
        })
        .collect()
}

/// Mobile `snapCells` (idea 4): snap legacy box centres to image-ink evidence
/// along the line axis. Profiled over the central 60% cross-band (dodges ruby
/// at the edges); each centre moves to its window ink centroid, clamped to its
/// Voronoi cell with a 0.4-pitch leash. Polarity auto-detects from border
/// pixels. `l` is the full frame length the cells live in, so the profile
/// scales defensively at image edges; `pix_w`/`pix_h` are the *pixel* frame,
/// which may be clamped independently of it.
fn snap_cells_from_profile(
    cells: &[(f32, f32)],
    text: &[char],
    ev: &SnapEvidence,
    pix_w: u32,
    pix_h: u32,
    vertical: bool,
    l: f32,
) -> Vec<(f32, f32)> {
    let prof_len = if vertical { pix_h } else { pix_w } as usize;
    if cells.is_empty() || pix_w < 8 || pix_h < 8 {
        return cells.to_vec();
    }
    if ev.counts.len() != prof_len {
        return cells.to_vec(); // wrong-shaped evidence: no snap, no panic
    }
    let Some(ev) = ev.validate() else {
        return cells.to_vec();
    };
    let sm: Vec<f32> = snap_smooth(&ev.counts);
    let sm = &sm;
    let prof_len = sm.len();
    let centers: Vec<f32> = cells.iter().map(|(a, b)| (a + b) / 2.0).collect();
    // Ink peaks along the axis (local maxima with prominence): each box snaps
    // to the NEAREST peak, and each peak carries ARGMAX + CENTROID.
    let mut peaks: Vec<(f32, f32, f32)> = Vec::new(); // (argmax, centroid, mass)
    let mut p = 1usize;
    while p + 1 < prof_len {
        if sm[p] > sm[p - 1] && sm[p] >= sm[p + 1] {
            let mut lft = p;
            while lft > 0 && sm[lft - 1] >= sm[lft] * 0.5 {
                lft -= 1;
            }
            let mut rgt = p;
            while rgt < prof_len - 1 && sm[rgt + 1] >= sm[rgt] * 0.5 {
                rgt += 1;
            }
            let (mut m2, mut mo, mut am, mut ap) = (0.0f32, 0.0f32, -1.0f32, p);
            for q in lft..=rgt {
                m2 += sm[q];
                mo += sm[q] * q as f32;
                if sm[q] > am {
                    am = sm[q];
                    ap = q;
                }
            }
            if m2 > 0.0 {
                peaks.push((ap as f32, mo / m2, m2));
            }
            p = rgt + 1;
        } else {
            p += 1;
        }
    }
    let mut masses: Vec<f32> = peaks.iter().map(|pk| pk.2).collect();
    masses.sort_by(f32::total_cmp);
    let med_mass = if masses.is_empty() {
        0.0
    } else {
        masses[masses.len() / 2]
    };
    cells
        .iter()
        .enumerate()
        .map(|(i, &(a, b))| {
            if i >= text.len() || is_snap_skipped(text[i]) {
                return (a, b);
            }
            let c = centers[i];
            let pitch = (b - a).max(4.0);
            let scale = prof_len as f32 / l.max(1.0);
            let cp = (c * scale).clamp(0.0, prof_len as f32 - 1.0);
            let mut best: Option<(f32, f32, f32)> = None;
            let mut best_d = 0.5 * pitch * scale + 1.0;
            for &pk in &peaks {
                if pk.2 < 6.0f32.max(0.35 * med_mass) {
                    continue;
                }
                let d = (pk.1 - cp).abs();
                if d < best_d {
                    best_d = d;
                    best = Some(pk);
                }
            }
            let Some(best) = best else { return (a, b) };
            // Agreement veto: centroid far from argmax = merged neighbours.
            if (best.1 - best.0).abs() > 0.3 * pitch * scale {
                return (a, b);
            }
            // Min-move gate: sub-visible moves (< 3px) carry measurement risk.
            if (best.1 / scale - c).abs() < 3.0 {
                return (a, b);
            }
            let mut nc = best.1 / scale;
            // Voronoi clamp between neighbour centres (ends: line bounds).
            let lo_b = if i > 0 {
                (centers[i - 1] + c) / 2.0
            } else {
                0.0
            };
            let hi_b = if i < centers.len() - 1 {
                (c + centers[i + 1]) / 2.0
            } else {
                l
            };
            nc = nc.clamp(lo_b, hi_b);
            nc = nc.clamp(c - 0.4 * pitch, c + 0.4 * pitch);
            let len = b - a;
            let s0 = (nc - len / 2.0).clamp(0.0, (l - len).max(0.0));
            (s0, (s0 + len).min(l))
        })
        .collect()
}

/// Load the PC overlay's JP font for ink measurement (mobile measures with the
/// device DEFAULT typeface; the PC equivalent is the bundled NotoSansJP the
/// viewer draws with). Cached after the first successful read.
fn ink_font_data() -> Option<&'static Vec<u8>> {
    static FONT: std::sync::OnceLock<Option<Vec<u8>>> = std::sync::OnceLock::new();
    FONT.get_or_init(|| {
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(p) = std::env::var("JPDICT_FONT") {
            candidates.push(std::path::PathBuf::from(p));
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("fonts/NotoSansJP-Regular.ttf"));
                candidates.push(dir.join("usr/bin/fonts/NotoSansJP-Regular.ttf"));
            }
        }
        candidates.push(std::path::PathBuf::from("fonts/NotoSansJP-Regular.ttf"));
        candidates.push(std::path::PathBuf::from(concat!(
            concat!(env!("CARGO_MANIFEST_DIR"), "/.."),
            "/fonts/NotoSansJP-Regular.ttf"
        )));
        candidates.push(std::path::PathBuf::from(
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ));
        candidates.push(std::path::PathBuf::from(
            "/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc",
        ));
        candidates.iter().find_map(|p| std::fs::read(p).ok())
    })
    .as_ref()
}

/// Ink half-widths (px) per char at `render_size`, from the font's glyph
/// bounding boxes — mobile's `Paint.getTextBounds(...).width() / 2`.
fn ink_half_widths(chars: &[char], render_size: f32) -> Vec<f32> {
    let Some(data) = ink_font_data() else {
        return vec![0.0; chars.len()];
    };
    let Ok(face) = ttf_parser::Face::parse(data, 0) else {
        return vec![0.0; chars.len()];
    };
    let upem = face.units_per_em().max(1) as f32;
    chars
        .iter()
        .map(|&c| {
            let gid = face.glyph_index(c).unwrap_or(ttf_parser::GlyphId(0));
            match face.glyph_bounding_box(gid) {
                Some(bb) => {
                    ((bb.x_max - bb.x_min) as f32 / upem * render_size).max(0.0) / 2.0
                }
                None => 0.0,
            }
        })
        .collect()
}

/// Mobile `resolveInkCollisions` (#49): move centres only where glyph ink
/// actually collides at render size, splitting just the collision. Widths are
/// preserved; only centres move. `ink_half` holds the caller-measured ink
/// half-width per character (index-aligned with `cells`; short entries read
/// as 0) — the desktop measures them from the bundled font
/// ([`ink_half_widths`]), Android with its own typeface.
fn resolve_ink_collisions(cells: &mut [(f32, f32)], ink_half: &[f32]) {
    let n = cells.len();
    if n < 2 {
        return;
    }
    let mut centers: Vec<f32> = cells.iter().map(|(a, b)| (a + b) / 2.0).collect();
    for ci in 0..n - 1 {
        let ink_r = centers[ci] + ink_half.get(ci).copied().unwrap_or(0.0);
        let ink_l = centers[ci + 1] - ink_half.get(ci + 1).copied().unwrap_or(0.0);
        if ink_r <= ink_l {
            continue;
        }
        let shift = (ink_r - ink_l) / 2.0;
        centers[ci] -= shift;
        centers[ci + 1] += shift;
    }
    for i in 0..n {
        let half = (cells[i].1 - cells[i].0) / 2.0;
        let c = centers[i];
        cells[i] = (c - half, c + half);
    }
}

/// Mobile `uniformCells` (#49): uniform WIDTHS around existing centres
/// (centres bit-identical to the resolve path). em = median width-normalized
/// pitch (`estimate_em`); fullwidth = em, halfwidth = 0.5em; edges clamped,
/// centres never moved.
pub(crate) fn uniform_cells(cells: &[(f32, f32)], chars: &[char], l: f32) -> Vec<(f32, f32)> {
    if cells.len() < 2 || chars.len() != cells.len() {
        return cells.to_vec();
    }
    let centers: Vec<f32> = cells.iter().map(|(a, b)| (a + b) / 2.0).collect();
    let text: String = chars.iter().collect();
    let em = crate::util::japanese::estimate_em(&text, &centers);
    if em <= 0.0 {
        return cells.to_vec();
    }
    cells
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let w = (if crate::util::japanese::is_half_width(chars[i]) {
                0.5
            } else {
                1.0
            }) * em;
            let half = w / 2.0;
            let c = centers[i];
            ((c - half).max(0.0), (c + half).min(l))
        })
        .collect()
}

/// Mobile vertical punctuation sets (`OcrEngine.computeCharBoxes`): closing
/// marks shrink onto the next box's start, opening marks onto the previous
/// box's end. Kept byte-for-byte with Android (D7.3/D46).
pub fn is_vertical_closing_punct(ch: char) -> bool {
    "。.．、,，)）〕》」』】〙〗〟’”］".contains(ch)
}

pub fn is_vertical_opening_punct(ch: char) -> bool {
    "(（〔《「『【〘〖〝‘“［".contains(ch)
}

/// Mobile `computeCharBoxes` (#49): per-character boxes from CTC timestep
/// columns. Horizontal: centre each char at `(t+0.5)·cropW/seqLenTotal` with
/// height `cropH`, then snap / ink-resolve / uniform-size. Vertical: same on
/// the y axis with width `cropW`, with the punctuation rules. `pixels` (RGB8
/// crop + dims) enables the snap stage; pass None to force legacy columns.
///
/// The desktop entry point: snap/uniform come from the `BOX_LAYOUT_MODE` /
/// `BOX_UNIFORM_SIZE` environment and the ink widths are measured from the
/// bundled font, exactly as the old `ocr_engine` version did.
pub fn compute_char_boxes(
    text: &str,
    char_cols: &[f32],
    seq_len_total: usize,
    crop_x: i32,
    crop_y: i32,
    crop_w: u32,
    crop_h: u32,
    is_vertical: bool,
    pixels: Option<(&[u8], u32, u32)>,
) -> Vec<BoundingBox> {
    let chars: Vec<char> = text.chars().collect();
    let ink = ink_half_widths(&chars, crop_h as f32 * 0.90);
    compute_char_boxes_with(
        text,
        char_cols,
        seq_len_total,
        crop_x,
        crop_y,
        crop_w,
        crop_h,
        is_vertical,
        pixels,
        box_layout_snap(),
        box_uniform_size(),
        &ink,
    )
}

/// [`compute_char_boxes`] with the environment behaviour as explicit
/// parameters: `snap` / `uniform` replace the `BOX_LAYOUT_MODE` /
/// `BOX_UNIFORM_SIZE` reads and `ink_half` (per-character ink half-widths,
/// index-aligned with the text; short entries read as 0) replaces the font
/// measurement, so the mobile shim can pass its own typeface widths and
/// preference flags. An empty `ink_half` skips the resolve stage's movement.
///
/// The snap stage's ink evidence is measured from `pixels` here
/// ([`snap_evidence_from_rgb`]); [`compute_char_boxes_with_evidence`] takes the
/// same evidence pre-measured. Identical counts give identical boxes — the two
/// entries differ only in where the 30k-pixel reduction happens.
#[allow(clippy::too_many_arguments)] // mirrors compute_char_boxes' frame
pub fn compute_char_boxes_with(
    text: &str,
    char_cols: &[f32],
    seq_len_total: usize,
    crop_x: i32,
    crop_y: i32,
    crop_w: u32,
    crop_h: u32,
    is_vertical: bool,
    pixels: Option<(&[u8], u32, u32)>,
    snap: bool,
    uniform: bool,
    ink_half: &[f32],
) -> Vec<BoundingBox> {
    let evidence =
        pixels.and_then(|(px, pw, ph)| snap_evidence_from_rgb(px, pw, ph, is_vertical));
    compute_char_boxes_with_evidence(
        text,
        char_cols,
        seq_len_total,
        crop_x,
        crop_y,
        crop_w,
        crop_h,
        is_vertical,
        evidence.as_ref(),
        pw_ph(pixels),
        snap,
        uniform,
        ink_half,
    )
}

/// The pixel frame of an optional crop, as the two scalars the evidence entry
/// validates against (`0, 0` when there is no crop, which fails the length
/// check and skips the snap stage exactly as the old `None` pixels did).
fn pw_ph(pixels: Option<(&[u8], u32, u32)>) -> (u32, u32) {
    match pixels {
        Some((_, w, h)) => (w, h),
        None => (0, 0),
    }
}

/// [`compute_char_boxes_with`] with the snap stage's ink evidence measured by
/// the caller instead of from a crop: the mobile FFI boundary reduces the crop
/// to a [`SnapEvidence`] (a polarity plus one count per reading-axis position)
/// *before* the crossing, because a whole ARGB crop per line was the single
/// most expensive marshalling on the page. `pix_w`/`pix_h` still cross (the
/// evidence is validated against the profile length and scales against the box
/// frame), so the stage behaves exactly as it does on pixels.
#[allow(clippy::too_many_arguments)] // mirrors compute_char_boxes_with's frame
pub fn compute_char_boxes_with_evidence(
    text: &str,
    char_cols: &[f32],
    seq_len_total: usize,
    crop_x: i32,
    crop_y: i32,
    crop_w: u32,
    crop_h: u32,
    is_vertical: bool,
    evidence: Option<&SnapEvidence>,
    pix_dims: (u32, u32),
    snap: bool,
    uniform: bool,
    ink_half: &[f32],
) -> Vec<BoundingBox> {
    let (pix_w, pix_h) = pix_dims;
    let n = char_cols.len();
    if n == 0 || seq_len_total == 0 {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    // Android keeps charCols and text index-aligned; clamp defensively so a
    // mismatched caller cannot panic a recognition worker.
    let n = char_cols.len().min(chars.len());
    let char_cols = &char_cols[..n];
    if !is_vertical {
        // ── HORIZONTAL: x-axis char boxes ──
        let char_w = (crop_h as f32).max(3.0);
        let l = crop_w as f32;
        let base = legacy_cells(char_cols, seq_len_total, l, char_w);
        let cells = match (snap, evidence) {
            (true, Some(ev)) => snap_cells_from_profile(&base, &chars, ev, pix_w, pix_h, false, l),
            _ => base,
        };
        let mut resolved = cells;
        resolve_ink_collisions(&mut resolved, ink_half);
        let sized = if uniform {
            uniform_cells(&resolved, &chars, l)
        } else {
            resolved
        };
        sized
            .iter()
            .map(|&(xl, xr)| {
                let left = (crop_x as f32 + xl).round() as i32;
                let right = (crop_x as f32 + xr).round() as i32;
                BoundingBox::new(left, crop_y, (right - left).max(1), crop_h as i32, 1.0)
            })
            .collect()
    } else {
        // ── VERTICAL: y-axis char boxes with punctuation handling ──
        let avg_ch_h = (crop_w as f32).max(3.0);
        let l = crop_h as f32;
        let base = legacy_cells(char_cols, seq_len_total, l, avg_ch_h);
        let cells = match (snap, evidence) {
            (true, Some(ev)) => snap_cells_from_profile(&base, &chars, ev, pix_w, pix_h, true, l),
            _ => base,
        };
        let is_cp: Vec<bool> = chars.iter().map(|&c| is_vertical_closing_punct(c)).collect();
        let is_op: Vec<bool> = chars.iter().map(|&c| is_vertical_opening_punct(c)).collect();

        // Resolve overlaps with punctuation rules (always; positions).
        let mut resolved = cells;
        for ci in 0..n.saturating_sub(1) {
            if resolved[ci].1 <= resolved[ci + 1].0 {
                continue;
            }
            if is_cp[ci] {
                resolved[ci].1 = resolved[ci + 1].0;
            } else if is_op[ci + 1] {
                resolved[ci + 1].0 = resolved[ci].1;
            } else if is_cp[ci + 1] {
                resolved[ci + 1].0 = resolved[ci].1;
            } else if is_op[ci] {
                resolved[ci].1 = resolved[ci + 1].0;
            } else {
                let h = (resolved[ci].1 - resolved[ci + 1].0) / 2.0;
                resolved[ci].1 -= h;
                resolved[ci + 1].0 += h;
            }
        }
        // Expand punctuation cells to the average non-punctuation height.
        let mut sized = if uniform {
            uniform_cells(&resolved, &chars, l)
        } else {
            resolved
        };
        let hs: Vec<f32> = sized
            .iter()
            .enumerate()
            .filter(|(ci, _)| !is_cp[*ci] && !is_op[*ci])
            .map(|(_, c)| c.1 - c.0)
            .collect();
        let avg_np_h = if hs.is_empty() {
            crop_w as f32
        } else {
            hs.iter().sum::<f32>() / hs.len() as f32
        };
        for ci in 0..n {
            if is_cp[ci] {
                let nx = ((ci + 1)..n)
                    .find(|&j| !is_cp[j] && !is_op[j])
                    .map(|j| sized[j].0)
                    .unwrap_or(f32::INFINITY);
                sized[ci].1 = (sized[ci].0 + avg_np_h).min(nx).max(sized[ci].1);
            } else if is_op[ci] {
                let pb = (0..ci).rev().find(|&j| !is_cp[j] && !is_op[j]);
                let pl = pb.map(|j| sized[j].1).unwrap_or(f32::NEG_INFINITY);
                sized[ci].0 = (sized[ci].1 - avg_np_h).max(pl).min(sized[ci].0);
            }
        }
        sized
            .iter()
            .map(|&(yt, yb)| {
                let ch = (yb - yt).max(1.0);
                let top = (crop_y as f32 + yt).round() as i32;
                let bottom = (crop_y as f32 + yt + ch).round() as i32;
                BoundingBox::new(crop_x, top, crop_w as i32, (bottom - top).max(1), 1.0)
            })
            .collect()
    }
}

/// CAP kill switch (mobile `PREF_BOX_PLACEMENT_CAP`): the improved
/// CTC-anchored placement ([`crate::char_placement`]) is the default; set
/// `BOX_PLACEMENT_CAP=0` to restore the shipped `legacyCells → snap →
/// resolveInkCollisions → uniformCells` chain bit-for-bit (and the
/// `BOX_LAYOUT_MODE` / `BOX_UNIFORM_SIZE` knobs to apply to it again).
pub(crate) fn box_placement_cap() -> bool {
    std::env::var("BOX_PLACEMENT_CAP").map(|v| v != "0").unwrap_or(true)
}

/// Char boxes for one line: CAP when enabled (default), the shipped chain
/// otherwise. `pixels` is the RGB8 crop (CAP needs it for the ink pass;
/// the shipped chain still gates its own snap stage on `BOX_LAYOUT_MODE`),
/// `steps` the decode's per-timestep top-K (`LineResult::raw_alternatives`)
/// — absent steps make CAP fall back to `char_cols` columns, exactly like
/// the reference's safety net.
///
/// Output is rounded to integer [`BoundingBox`]es in the caller's frame with
/// the same conventions as [`compute_char_boxes`] (horizontal: `(crop +
/// edge).round()`, full `crop_h`; vertical: full `crop_w`).
///
/// The desktop entry point: the CAP kill switch and the snap/uniform knobs
/// come from the environment and the ink widths from the bundled font.
#[allow(clippy::too_many_arguments)] // mirrors compute_char_boxes' frame
pub fn compute_char_boxes_line(
    text: &str,
    char_cols: &[f32],
    seq_len_total: usize,
    crop_x: i32,
    crop_y: i32,
    crop_w: u32,
    crop_h: u32,
    is_vertical: bool,
    pixels: Option<(&[u8], u32, u32)>,
    steps: Option<&[Vec<(char, f32)>]>,
) -> Vec<BoundingBox> {
    let chars: Vec<char> = text.chars().collect();
    let ink = ink_half_widths(&chars, crop_h as f32 * 0.90);
    compute_char_boxes_line_with(
        text,
        char_cols,
        seq_len_total,
        crop_x,
        crop_y,
        crop_w,
        crop_h,
        is_vertical,
        pixels,
        steps,
        box_placement_cap(),
        box_layout_snap(),
        box_uniform_size(),
        &ink,
    )
}

/// [`compute_char_boxes_line`] with the environment behaviour as explicit
/// parameters (`cap` replaces `BOX_PLACEMENT_CAP`, the rest as in
/// [`compute_char_boxes_with`]).
#[allow(clippy::too_many_arguments)] // mirrors compute_char_boxes' frame
pub fn compute_char_boxes_line_with(
    text: &str,
    char_cols: &[f32],
    seq_len_total: usize,
    crop_x: i32,
    crop_y: i32,
    crop_w: u32,
    crop_h: u32,
    is_vertical: bool,
    pixels: Option<(&[u8], u32, u32)>,
    steps: Option<&[Vec<(char, f32)>]>,
    cap: bool,
    snap: bool,
    uniform: bool,
    ink_half: &[f32],
) -> Vec<BoundingBox> {
    // Same degenerate-input shape as the shipped chain: nothing to place.
    if char_cols.is_empty() || seq_len_total == 0 {
        return Vec::new();
    }
    if !cap {
        return compute_char_boxes_with(
            text,
            char_cols,
            seq_len_total,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            is_vertical,
            pixels,
            snap,
            uniform,
            ink_half,
        );
    }
    let lum: Option<(Vec<f32>, u32, u32)> = pixels
        .map(|(p, w, h)| (crate::char_placement::luminance_from_rgb(p, w, h), w, h));
    let steps64: Option<Vec<Vec<(char, f64)>>> = steps
        .map(|s| {
            s.iter()
                .map(|alts| alts.iter().map(|&(c, sc)| (c, sc as f64)).collect())
                .collect()
        });
    let boxes = crate::char_placement::place(
        text,
        char_cols,
        seq_len_total,
        crop_w,
        crop_h,
        is_vertical,
        lum.as_ref().map(|(l, w, h)| (l.as_slice(), *w, *h)),
        steps64.as_deref(),
    );
    boxes
        .iter()
        .map(|b| {
            if !is_vertical {
                let left = (crop_x as f32 + b[0] as f32).round() as i32;
                let right = (crop_x as f32 + b[2] as f32).round() as i32;
                BoundingBox::new(left, crop_y, (right - left).max(1), crop_h as i32, 1.0)
            } else {
                let yt = b[1] as f32;
                let ch = (b[3] as f32 - yt).max(1.0);
                let top = (crop_y as f32 + yt).round() as i32;
                let bottom = (crop_y as f32 + yt + ch).round() as i32;
                BoundingBox::new(crop_x, top, crop_w as i32, (bottom - top).max(1), 1.0)
            }
        })
        .collect()
}

#[allow(dead_code)] // no live caller yet: mobile's consumer is the gap-detector fallback
impl DetectedAnnotation {
    /// Mobile `OcrEngine.reDecodeLineResult`: rebuild a line's text, char
    /// boxes and per-character alternatives from its cached
    /// [`LineResult::raw_alternatives`], without re-running the model. The
    /// walk (blank handling, CTC collapse, fractional columns, vertical
    /// punctuation) lives in [`re_decode_raw_alternatives`].
    ///
    /// Mobile reads the crop geometry off its `LineResult`; the PC annotation
    /// owns it instead, so the boxes are recomputed in the same frame the emit
    /// path used — the unclamped `bbox` rect for axis lines, the upright local
    /// crop mapped back through `quad` for rotated ones — whenever
    /// `seq_len_total` is known (the one crop fact the PC line does not
    /// cache). `seq_len_total == 0` keeps the existing boxes, mobile's
    /// `cropW == 0` fallback. Lines with nothing cached come back unchanged,
    /// as do annotation slots that never got a line.
    pub fn re_decode_line(&self, seq_len_total: usize) -> DetectedAnnotation {
        let Some(line) = self.line.as_ref() else {
            return self.clone();
        };
        let Some(re) =
            re_decode_raw_alternatives(&line.raw_alternatives, line.is_vertical)
        else {
            return self.clone();
        };

        let char_boxes = if seq_len_total > 0 {
            // Re-decode has the same cached per-timestep top-K the live path
            // passes; no pixels, so CAP runs on recorded evidence alone.
            let steps = Some(line.raw_alternatives.as_slice());
            match self.quad.filter(|r| r.is_rotated()) {
                Some(r) => {
                    let lw = r.w.round().max(4.0) as u32;
                    let lh = r.h.round().max(4.0) as u32;
                    compute_char_boxes_line(
                        &re.text, &re.char_cols, seq_len_total,
                        0, 0, lw, lh, line.is_vertical, None, steps,
                    )
                    .iter()
                    .map(|b| r.map_local_rect(b.x as f32, b.y as f32, b.w as f32, b.h as f32))
                    .collect()
                }
                None => compute_char_boxes_line(
                    &re.text, &re.char_cols, seq_len_total,
                    self.bbox.x, self.bbox.y,
                    self.bbox.w.max(0) as u32, self.bbox.h.max(0) as u32,
                    line.is_vertical, None, steps,
                ),
            }
        } else {
            line.char_boxes.clone()
        };

        DetectedAnnotation {
            bbox: self.bbox.clone(),
            quad: self.quad,
            line: Some(LineResult {
                text: re.text,
                char_boxes,
                alternatives: re.alternatives,
                raw_alternatives: line.raw_alternatives.clone(),
                sample_txt: line.sample_txt.clone(),
                is_vertical: line.is_vertical,
                chunk_boxes: line.chunk_boxes.clone(),
                // Mobile `reDecodeLineResult` recomputes the columns and
                // carries the crop facts and overrides forward.
                char_cols: re.char_cols,
                overrides: line.overrides.clone(),
                crop_w: line.crop_w,
                crop_h: line.crop_h,
                crop_x: line.crop_x,
                crop_y: line.crop_y,
                seq_len_total: line.seq_len_total,
            }),
        }
    }
}

/// Sub-column peak offset (#49): parabolic interpolation of the winning
/// class value across neighbouring timesteps. Returns 0 when the peak is
/// flat, at a boundary, or prominence is below the mobile gate.
pub fn peak_offset(v0: f32, v1: f32, v2: f32) -> f32 {
    let denom = v0 - 2.0f32 * v1 + v2;
    if denom >= -1e-6f32 {
        return 0.0f32;
    }
    if (v1 - v0).min(v1 - v2) <= 1.0f32 {
        return 0.0f32;
    }
    (0.5f32 * (v0 - v2) / denom).clamp(-0.5, 0.5)
}

// ——— Re-decode from cached raw alternatives (no model) ———

/// Model-free re-decode output: text, per-emitted-character alternatives and
/// fractional CTC columns, ready to rebuild a `LineResult`.
#[allow(dead_code)] // only reached through `DetectedAnnotation::re_decode_line` (unwired yet)
#[derive(Debug, Clone)]
pub struct ReDecodedLine {
    pub text: String,
    pub alternatives: Vec<Vec<(char, f32)>>,
    pub char_cols: Vec<f32>,
}

/// Mobile `OcrEngine.reDecodeLineResult`'s walk, without the `LineResult`
/// plumbing: greedy CTC over the cached raw alternatives — entry 0 is the
/// argmax, `'\u{3000}'` is the blank marker (so blank resets the collapse
/// state), a space never collapses, any other character collapses only
/// against the immediately preceding emitted character. Emits at most one
/// character per timestep; `alternatives` stays index-aligned with `text`.
/// Columns carry the winner's own score at the neighbouring timesteps
/// (matched by character, absent → integer column), like the live decode —
/// except that the live full-logits decode reads the neighbour's whole row,
/// so it can interpolate from a class outside the neighbour's top-K while
/// this walk cannot (mobile's `ctcDecode.wval` vs `reDecodeLineResult.wval`).
/// Vertical lines get the emit path's punctuation normalisation applied to
/// the text and to every alternative entry, so cached pre-fix raw data cannot
/// reintroduce ASCII `?`/horizontal `…`.
///
/// `None` when nothing is cached (mobile returns the line unchanged).
#[allow(dead_code)] // only reached through `DetectedAnnotation::re_decode_line` (unwired yet)
pub fn re_decode_raw_alternatives(
    raw: &[Vec<(char, f32)>],
    is_vertical: bool,
) -> Option<ReDecodedLine> {
    if raw.is_empty() {
        return None;
    }
    let mut text = String::new();
    let mut new_alts: Vec<Vec<(char, f32)>> = Vec::new();
    let mut char_cols: Vec<f32> = Vec::new();
    let mut prev_char: Option<char> = None;

    // Winner score at a neighbouring timestep, matched by character
    // (mobile `wval`; first match wins).
    let wval = |t: usize, ch: char| -> Option<f32> {
        raw.get(t)?
            .iter()
            .find(|(c, _)| *c == ch)
            .map(|(_, s)| *s)
    };
    let frac_for = |t: usize, ch: char, v1: f32| -> f32 {
        let w0 = if t > 0 { wval(t - 1, ch) } else { None };
        match (w0, wval(t + 1, ch)) {
            (Some(w0), Some(w2)) => peak_offset(w0, v1, w2),
            _ => 0.0,
        }
    };

    for (t, alts) in raw.iter().enumerate() {
        let Some(&(top_char, top_score)) = alts.first() else {
            continue;
        };
        if top_char == '\u{3000}' {
            // Blank: resets the repeat state but is never emitted.
            prev_char = None;
        } else if top_char == ' ' {
            text.push(' ');
            prev_char = Some(' ');
            char_cols.push(t as f32 + frac_for(t, ' ', top_score));
            new_alts.push(alts.clone());
        } else if prev_char == Some(top_char) {
            // collapse repeat
        } else {
            text.push(top_char);
            char_cols.push(t as f32 + frac_for(t, top_char, top_score));
            new_alts.push(alts.clone());
            prev_char = Some(top_char);
        }
    }

    if is_vertical {
        text = crate::util::japanese::vertical_punctuation(&text);
        for alts in new_alts.iter_mut() {
            for (c, _) in alts.iter_mut() {
                *c = crate::util::japanese::vertical_punctuation_char(*c);
            }
        }
    }
    Some(ReDecodedLine {
        text,
        alternatives: new_alts,
        char_cols,
    })
}

/// Reading-order permutation for axis-aligned boxes (mobile
/// `OcrEngine.sortDetectedBoxes`): the indices of `boxes` in reading order —
/// horizontals top-to-bottom/left-to-right, then verticals
/// right-edge-to-left/top-to-bottom. Orientation comes from the box's own
/// sizes ([`crate::furigana::is_vertical_box`], near-square counts as
/// horizontal), like the axis-aligned path, which has no separate frame. The
/// desktop engine's pair sort (`OcrEngine::sort_detected_boxes`) instead
/// orients by the rotated frame and stays where it is.
///
/// A permutation (not sorted boxes) so the mobile caller keeps its own box
/// identities: the shared conformance corpus pins reading order through
/// referential equality.
pub fn sort_order(boxes: &[BoundingBox]) -> Vec<usize> {
    let mut horizontal: Vec<usize> = Vec::new();
    let mut vertical: Vec<usize> = Vec::new();
    for (i, b) in boxes.iter().enumerate() {
        if crate::furigana::is_vertical_box(b) {
            vertical.push(i);
        } else {
            horizontal.push(i);
        }
    }
    horizontal.sort_by(|&a, &b| {
        boxes[a]
            .y
            .cmp(&boxes[b].y)
            .then(boxes[a].x.cmp(&boxes[b].x))
    });
    vertical.sort_by(|&a, &b| {
        (boxes[b].x + boxes[b].w)
            .cmp(&(boxes[a].x + boxes[a].w))
            .then(boxes[a].y.cmp(&boxes[b].y))
    });
    horizontal.extend(vertical);
    horizontal
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mobile reading order on plain rects: two horizontal rows sort
    /// top-to-bottom/left-to-right, two vertical columns right-edge-first —
    /// the same expectations the shared corpus pins through
    /// `OcrEngine.sortDetectedBoxes`.
    #[test]
    fn sort_order_matches_the_mobile_rect_sort() {
        let boxes = vec![
            BoundingBox::new(132, 100, 24, 24, 1.0),
            BoundingBox::new(100, 100, 24, 24, 1.0),
            BoundingBox::new(100, 180, 24, 24, 1.0),
            BoundingBox::new(380, 100, 24, 40, 1.0),
            BoundingBox::new(380, 150, 24, 40, 1.0),
            BoundingBox::new(300, 110, 24, 40, 1.0),
            BoundingBox::new(300, 160, 24, 40, 1.0),
        ];
        // Horizontals top-to-bottom/left-to-right, then verticals by right
        // edge descending, top ascending.
        assert_eq!(sort_order(&boxes), vec![1, 0, 2, 3, 4, 5, 6]);
        // Near-square counts as horizontal (the 1.25x rule).
        let square = vec![
            BoundingBox::new(200, 100, 40, 40, 1.0),
            BoundingBox::new(100, 100, 24, 24, 1.0),
        ];
        assert_eq!(sort_order(&square), vec![1, 0]);
        // Empty and singleton inputs are the identity.
        assert!(sort_order(&[]).is_empty());
        assert_eq!(sort_order(&[BoundingBox::new(1, 2, 3, 4, 1.0)]), vec![0]);
    }

    // ─── snap evidence: pixels vs. the pre-measured record ──────────────────

    /// A synthetic line crop: paper background, `glyphs` ink blobs of `weight`
    /// columns each, `invert` for the light-on-dark polarity. Deterministic, so
    /// the two entries can be compared bit-for-bit.
    fn line_crop(w: usize, h: usize, glyphs: &[(usize, usize)], weight: usize, invert: bool) -> Vec<u8> {
        let inkc = 0x28u8;
        let mut px = vec![0xFFu8; w * h * 3];
        let paint = |px: &mut Vec<u8>, x: usize, y: usize, c: u8| {
            let p = (y * w + x) * 3;
            px[p] = c;
            px[p + 1] = c;
            px[p + 2] = c;
        };
        for &(c0, c1) in glyphs {
            for x in c0..(c1 + weight).min(w) {
                for y in 2..h.saturating_sub(2) {
                    paint(&mut px, x, y, inkc);
                }
            }
        }
        // A few speckles outside the band and at the edges, so the band
        // restriction and the border polarity are both load-bearing.
        for (i, x) in [0usize, 1, w / 2, w - 2].iter().enumerate() {
            let y = if i % 2 == 0 { 0 } else { h - 1 };
            paint(&mut px, *x, y, inkc);
        }
        if invert {
            // Light glyphs on a dark page: the same shapes, inverted.
            for c in px.iter_mut() {
                *c = 255 - *c;
            }
        }
        px
    }

    /// A small deterministic LCG, so the sweep below needs no dependency.
    fn lcg(state: &mut u64) -> u32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 33) as u32
    }

    /// The evidence entry is a pure re-plumbing of the pixel entry: for the
    /// same crop, measured evidence and pixels must give *identical* boxes, on
    /// both orientations and both polarities, over a sweep of crop shapes,
    /// glyph layouts and column counts. This is the pin the mobile FFI
    /// boundary's fixture-parity test mirrors on device.
    #[test]
    fn evidence_entry_matches_the_pixel_entry_bit_for_bit() {
        let mut state = 0x5eed_1234u64;
        for case in 0..48usize {
            let w = 40 + (lcg(&mut state) as usize % 220);
            let h = 16 + (lcg(&mut state) as usize % 60);
            let n = 2 + (lcg(&mut state) as usize % 9);
            let weight = 1 + (lcg(&mut state) as usize % 4);
            let invert = case % 2 == 1;
            let vertical = case % 3 == 0;
            let pitch = (w / n).max(6);
            let glyphs: Vec<(usize, usize)> = (0..n)
                .map(|i| (i * pitch + 2 + (lcg(&mut state) as usize % 3), i * pitch + pitch / 2))
                .collect();
            let px = line_crop(w, h, &glyphs, weight, invert);
            let cols: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let text: String = "あいうえおかきくけこさしすせそ".chars().take(n).collect();
            let ink: Vec<f32> = (0..n).map(|i| 8.0 + i as f32).collect();
            let ev = snap_evidence_from_rgb(&px, w as u32, h as u32, vertical)
                .expect("a full-frame crop always yields evidence");
            let dims = (w as u32, h as u32);
            for &(snap, uniform) in &[(true, true), (true, false), (false, true)] {
                let via_pixels = compute_char_boxes_with(
                    &text, &cols, (n + 3) as usize, 0, 0, w as u32, h as u32, vertical,
                    Some((&px, dims.0, dims.1)), snap, uniform, &ink,
                );
                let via_evidence = compute_char_boxes_with_evidence(
                    &text, &cols, (n + 3) as usize, 0, 0, w as u32, h as u32, vertical,
                    Some(&ev), dims, snap, uniform, &ink,
                );
                assert_eq!(
                    via_pixels, via_evidence,
                    "case {case} (w={w} h={h} n={n} vert={vertical} inv={invert} snap={snap} unif={uniform})"
                );
            }
        }
    }

    /// The published integer ink test is the float one, exactly, for every
    /// possible channel sum (0..=765). This is what lets a facade count ink
    /// without a float division per pixel and still place the same boxes.
    #[test]
    fn snap_spec_integer_agrees_with_float() {
        let spec = snap_spec();
        assert_eq!(spec.ink_below_sum, 330);
        assert_eq!(spec.ink_above_sum, 436);
        for sum in 0..=765i32 {
            let lum = sum as f32 / 3.0;
            assert_eq!(
                is_ink_sum(sum, true),
                lum < spec.ink_below,
                "light background, sum {sum}"
            );
            assert_eq!(
                is_ink_sum(sum, false),
                lum > spec.ink_above,
                "dark background, sum {sum}"
            );
        }
    }

    /// The measurement half a boundary has to reproduce: the border sample, the
    /// polarity, the band range and the count series. Pinned here so the shim's
    /// mirror of it cannot drift from the crate's own.
    #[test]
    fn evidence_measurement_is_the_documented_reduction() {
        let spec = snap_spec();
        assert_eq!(spec.ink_below, 110.0);
        assert_eq!(spec.ink_above, 145.0);
        assert_eq!(spec.bg_median_above, 128.0);
        assert_eq!(spec.border_stride, 7);
        assert_eq!((spec.band_lo, spec.band_hi), (0.2, 0.8));
        assert_eq!(spec.blur_radius, 2);
        assert_eq!(spec.max_count, SNAP_MAX_COUNT);
        // Band indices truncate, exactly as the profile loop's slice does.
        assert_eq!(snap_band_range(60), (12, 48));
        assert_eq!(snap_band_range(8), (1, 6));
        assert_eq!(snap_band_range(4), (0, 3));
        // The border sample walks the same pixels, in the same order, at stride 7.
        let (w, h) = (30usize, 20usize);
        let lum: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
        let border = snap_border_lum(&lum, w, h);
        let mut expect = Vec::new();
        let mut i = 0;
        while i < w {
            expect.push(lum[i] as f32);
            expect.push(lum[(h - 1) * w + i] as f32);
            i += 7;
        }
        i = 0;
        while i < h {
            expect.push(lum[i * w] as f32);
            expect.push(lum[i * w + w - 1] as f32);
            i += 7;
        }
        assert_eq!(border, expect);
        assert!(snap_border_lum(&lum, 0, 0).is_empty());
        // Polarity: the *upper* median against 128, and None on no sample.
        assert_eq!(snap_polarity(&[200.0, 210.0, 220.0, 240.0]), Some(true));
        assert_eq!(snap_polarity(&[10.0, 20.0, 30.0, 240.0]), Some(false));
        assert_eq!(snap_polarity(&[]), None);
        // A saturated count is refused rather than snapped on wrong numbers.
        let mut ev = SnapEvidence::new(true, vec![1.0, 2.0, 3.0]);
        assert!(ev.validate().is_some());
        ev.counts[1] = SNAP_MAX_COUNT as f32;
        assert!(ev.validate().is_none(), "a saturated count cannot drive a snap");
    }

    /// The guards the pixel entry already had, on the evidence entry: a
    /// wrong-shaped profile, a missing record and an absent crop frame all skip
    /// the snap stage (the legacy columns) instead of snapping on garbage.
    #[test]
    fn evidence_entry_keeps_the_pixel_guards() {
        let (w, h) = (200usize, 40usize);
        let glyphs = vec![(10usize, 14usize), (60, 64), (110, 114), (160, 164)];
        let px = line_crop(w, h, &glyphs, 8, false);
        let cols = [0.0f32, 1.0, 2.0, 3.0];
        let ink = [8.0f32; 4];
        let snapped = compute_char_boxes_with(
            "あいうえ", &cols, 5, 0, 0, w as u32, h as u32, false,
            Some((&px, w as u32, h as u32)), true, true, &ink,
        );
        let plain = compute_char_boxes_with(
            "あいうえ", &cols, 5, 0, 0, w as u32, h as u32, false, None, true, true, &ink,
        );
        assert_ne!(snapped, plain, "the fixture must actually move boxes");
        let ev = snap_evidence_from_rgb(&px, w as u32, h as u32, false).unwrap();
        // Wrong length: no snap.
        let mut short = ev.clone();
        short.counts.pop();
        assert_eq!(
            compute_char_boxes_with_evidence(
                "あいうえ", &cols, 5, 0, 0, w as u32, h as u32, false,
                Some(&short), (w as u32, h as u32), true, true, &ink,
            ),
            plain
        );
        // No record: no snap.
        assert_eq!(
            compute_char_boxes_with_evidence(
                "あいうえ", &cols, 5, 0, 0, w as u32, h as u32, false,
                None, (w as u32, h as u32), true, true, &ink,
            ),
            plain
        );
        // A sub-8px frame: no snap (the old `pix_w < 8` guard).
        assert_eq!(
            compute_char_boxes_with_evidence(
                "あいうえ", &cols, 5, 0, 0, w as u32, h as u32, false,
                Some(&ev), (0, 0), true, true, &ink,
            ),
            plain
        );
    }

    /// The explicit-flags variant honours its parameters: no snap without
    /// pixels either way, no uniform sizing when `uniform` is off, and an
    /// empty ink table leaves the resolve stage's centres on the columns.
    #[test]
    fn compute_with_honours_explicit_snap_uniform_and_ink() {
        let cols = [0.0f32, 1.0, 2.0, 3.0];
        let frame = (10, 20, 200u32, 40u32);
        let legacy = compute_char_boxes_with(
            "あいうえ",
            &cols,
            4,
            frame.0,
            frame.1,
            frame.2,
            frame.3,
            false,
            None,
            true,
            true,
            &[],
        );
        assert_eq!(legacy.len(), 4);
        let centres: Vec<f32> = legacy
            .iter()
            .map(|b| b.x as f32 + b.w as f32 / 2.0)
            .collect();
        assert_eq!(centres, vec![35.0, 85.0, 135.0, 185.0]);
        // Uniform off: legacy widths (2 * cross/2 clamped to the frame).
        let raw = compute_char_boxes_with(
            "あいうえ",
            &cols,
            4,
            frame.0,
            frame.1,
            frame.2,
            frame.3,
            false,
            None,
            true,
            false,
            &[],
        );
        assert_eq!(raw.len(), 4);
        for b in &raw {
            assert_eq!(b.w, 40, "legacy width without uniform sizing: {b:?}");
        }
        // Huge ink half-widths force every neighbour pair apart symmetrically.
        let pushed = compute_char_boxes_with(
            "あいうえ",
            &cols,
            4,
            frame.0,
            frame.1,
            frame.2,
            frame.3,
            false,
            None,
            false,
            false,
            &[30.0, 30.0, 30.0, 30.0],
        );
        let pushed_centres: Vec<f32> = pushed
            .iter()
            .map(|b| b.x as f32 + b.w as f32 / 2.0)
            .collect();
        assert_eq!(pushed_centres, vec![30.0, 83.0, 134.0, 194.0]);
    }

    /// The env-reading entry points agree with the explicit variant at
    /// default knob settings (knobs default on; no font needed — an empty
    /// ink table is what a missing font measures).
    #[test]
    fn compute_entry_points_match_the_explicit_variant_at_defaults() {
        let cols = [0.0f32, 1.0, 2.0, 3.0];
        let a = compute_char_boxes("あいうえ", &cols, 4, 0, 0, 200, 40, false, None);
        let ink = ink_half_widths(&['あ', 'い', 'う', 'え'], 40.0 * 0.90);
        let b = compute_char_boxes_with(
            "あいうえ",
            &cols,
            4,
            0,
            0,
            200,
            40,
            false,
            None,
            true,
            true,
            &ink,
        );
        assert_eq!(a, b);
        let la = compute_char_boxes_line("あいうえ", &cols, 4, 0, 0, 200, 40, false, None, None);
        let lb = compute_char_boxes_line_with(
            "あいうえ",
            &cols,
            4,
            0,
            0,
            200,
            40,
            false,
            None,
            None,
            true,
            true,
            true,
            &ink,
        );
        assert_eq!(la, lb);
    }
}
