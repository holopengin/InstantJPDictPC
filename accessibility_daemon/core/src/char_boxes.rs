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

/// Mobile `snapCells` (idea 4): snap legacy box centres to image-ink evidence
/// along the line axis. Profiled over the central 60% cross-band (dodges ruby
/// at the edges); each centre moves to its window ink centroid, clamped to its
/// Voronoi cell with a 0.4-pitch leash. Polarity auto-detects from border
/// pixels. `pixels` are RGB8 in the (possibly clamped) crop frame; `l` is the
/// full frame length the cells live in, so the profile scales defensively at
/// image edges.
fn snap_cells(
    cells: &[(f32, f32)],
    text: &[char],
    pixels: &[u8],
    pix_w: u32,
    pix_h: u32,
    vertical: bool,
    l: f32,
) -> Vec<(f32, f32)> {
    if cells.is_empty() || pix_w < 8 || pix_h < 8 {
        return cells.to_vec();
    }
    let (pw, ph) = (pix_w as usize, pix_h as usize);
    if pixels.len() < pw * ph * 3 {
        return cells.to_vec();
    }
    let lum: Vec<f32> = (0..pw * ph)
        .map(|i| {
            let p = i * 3;
            (pixels[p] as f32 + pixels[p + 1] as f32 + pixels[p + 2] as f32) / 3.0
        })
        .collect();
    // Background polarity from border samples.
    let mut border: Vec<f32> = Vec::new();
    let mut bi = 0usize;
    while bi < pw {
        border.push(lum[bi]);
        border.push(lum[(ph - 1) * pw + bi]);
        bi += 7;
    }
    bi = 0;
    while bi < ph {
        border.push(lum[bi * pw]);
        border.push(lum[bi * pw + pw - 1]);
        bi += 7;
    }
    if border.is_empty() {
        return cells.to_vec();
    }
    border.sort_by(f32::total_cmp);
    let bg_light = border[border.len() / 2] > 128.0;
    let is_ink = |v: f32| if bg_light { v < 110.0 } else { v > 145.0 };
    // Axis profile over the central cross-band.
    let prof_len = if vertical { ph } else { pw };
    let mut prof = vec![0.0f32; prof_len];
    if vertical {
        let x0 = (pw as f32 * 0.2) as usize;
        let x1 = (pw as f32 * 0.8) as usize;
        for y in 0..ph {
            let mut m = 0.0f32;
            for x in x0..x1 {
                if is_ink(lum[y * pw + x]) {
                    m += 1.0;
                }
            }
            prof[y] = m;
        }
    } else {
        let y0 = (ph as f32 * 0.2) as usize;
        let y1 = (ph as f32 * 0.8) as usize;
        for x in 0..pw {
            let mut m = 0.0f32;
            for y in y0..y1 {
                if is_ink(lum[y * pw + x]) {
                    m += 1.0;
                }
            }
            prof[x] = m;
        }
    }
    // Box blur radius 2.
    let sm: Vec<f32> = (0..prof_len)
        .map(|i| {
            let mut a = 0.0f32;
            let mut c = 0usize;
            for k in -2i32..=2 {
                let j = (i as i32 + k).clamp(0, prof_len as i32 - 1) as usize;
                a += prof[j];
                c += 1;
            }
            a / c as f32
        })
        .collect();
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
        let cells = match (snap, pixels) {
            (true, Some((px, pw, ph))) => snap_cells(&base, &chars, px, pw, ph, false, l),
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
        let cells = match (snap, pixels) {
            (true, Some((px, pw, ph))) => snap_cells(&base, &chars, px, pw, ph, true, l),
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
