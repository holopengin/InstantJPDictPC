use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use imageproc::contours;
use fontdue::Font;
use std::path::Path;

use crate::models::*;
use crate::ppocr_ncnn::{DetNet, RecNet};

// PP-OCRv6 detection constants. The ncnn det model runs on a square
// letterboxed input (mobile #51 default 896); its output map is thresholded
// and turned into rotated boxes by the PC pipeline below.
const PPOCR_DET_MODEL_SIZE: u32 = 896;
const PPOCR_DET_LONG_SIDE: u32 = 960;
const PPOCR_DET_THRESH: f32 = 0.3;
const PPOCR_DET_BOX_THRESH: f32 = 0.8;
const PPOCR_DET_UNCLIP_RATIO: f32 = 1.5;
const X_OVERLAP_THRESHOLD: f32 = 0.3;

// Furigana (ruby) filter (#28) — conservative: better to recognize ruby than
// to drop real small text. Matching runs on RAW contour geometry (pre-unclip:
// unclip padding fabricates overlap for stacked fragments); only the gap test
// uses UNCLIPPED boxes (raw gutters are real pixels, unclip closes them).
const FURIGANA_SIZE_RATIO: f32 = 0.3; // small long-side < 30% of large long-side
const FURIGANA_THIN_RATIO: f32 = 0.75; // horizontal ruby runs long but thin
const FURIGANA_HSHORT_RATIO: f32 = 0.85; // horizontal short-side ceiling
const FURIGANA_WIDTH_RATIO: f32 = 0.65; // vertical small short-side ceiling
const FURIGANA_GAP_RATIO: f32 = 0.5;
const FURIGANA_OVERLAP_RATIO: f32 = 0.5;
const VERTICAL_MIN_ASPECT: f32 = 1.25;
// Absolute ceiling: real short columns dwarf ruby runs even when the ratio
// matches — ruby longer than 12% of the image side is not ruby.
const FURIGANA_MAX_FRAC: f32 = 0.12;
// Absolute floor on the annotated box: ruby hugs full-size body text, not
// compact blocks (logo boxes, badges).
const FURIGANA_BIG_MIN_FRAC: f32 = 0.2;

// Box fill ratio when rendering glyphs inside detected boxes. 1.0 means match box height, <1.0 leave padding.
const BOX_FILL_RATIO: f32 = 0.9;

/// Rec workers sharing the one loaded ncnn net (mobile fans out to 4;
/// `PPOCR_REC_WORKERS` overrides for tuning).
const DEFAULT_REC_WORKERS: usize = 4;

pub struct OcrEngine {
    /// PP-OCRv6 detection (ncnn DB) — shared core with the Android app.
    pub det_net: DetNet,
    /// PP-OCRv6 dynamic-width recognition (ncnn rec_dyn). `None` when the
    /// model files are missing; recognition is then disabled.
    pub ppocr_rec: Option<std::sync::Arc<RecNet>>,
    pub ppocr_vocab: Vec<String>,
    /// Pruned CTC head remap: remap[pruned_id] = original class id (#39).
    pub rec_remap: Vec<i32>,
    pub recognition_mode: RecognitionMode,
    pub batch_size: usize,
}

/// Load the PP-OCRv6 ncnn models + vocab/remap from `model_dir`.
/// Returns `(rec, vocab, remap)`; `rec` is None when rec_dyn is absent.
fn load_ppocr_models(
    model_dir: &Path,
) -> Result<(Option<std::sync::Arc<RecNet>>, Vec<String>, Vec<i32>)> {
    let ncnn_dir = model_dir.join("PP-OCRv6_small_ncnn");

    let rec_param = ncnn_dir.join("rec_dyn.param");
    let rec_bin = ncnn_dir.join("rec_dyn.bin");
    let ppocr_rec = if rec_param.exists() && rec_bin.exists() {
        // target_w only seeds the handle; inference width comes from the
        // actual input (dynamic width, mobile #23). 1 thread per net like
        // mobile REC_THREADS; workers run in parallel instead.
        match RecNet::create(&rec_param, &rec_bin, 64, 1) {
            Ok(r) => Some(std::sync::Arc::new(r)),
            Err(e) => {
                eprintln!("[PP-OCR] rec_dyn load failed ({e}); recognition disabled");
                None
            }
        }
    } else {
        eprintln!("[PP-OCR] Model not found at {rec_param:?}, recognition disabled");
        None
    };

    let vocab_path = ncnn_dir.join("vocab.json");
    let ppocr_vocab: Vec<String> = match std::fs::read_to_string(&vocab_path) {
        Ok(content) => serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse PP-OCR vocab {vocab_path:?}"))?,
        Err(_) => {
            eprintln!("[PP-OCR] Vocab not found at {vocab_path:?}");
            Vec::new()
        }
    };

    // CTC-head remap: one original class id per pruned output (#39/#44).
    // The head width is derived from this file, never hardcoded, so a
    // re-pruned model and this loader cannot silently disagree.
    let remap_path = ncnn_dir.join("rec_remap.txt");
    let rec_remap: Vec<i32> = match std::fs::read_to_string(&remap_path) {
        Ok(s) => s
            .lines()
            .filter_map(|l| l.trim().parse::<i32>().ok())
            .collect(),
        Err(_) => {
            eprintln!("[PP-OCR] remap not found at {remap_path:?}");
            Vec::new()
        }
    };
    if ppocr_rec.is_some() {
        println!(
            "[PP-OCR] ncnn rec loaded, vocab {} chars, head width {}",
            ppocr_vocab.len(),
            rec_remap.len()
        );
    }

    Ok((ppocr_rec, ppocr_vocab, rec_remap))
}

/// Minimum-area rotated rectangle around a contour (cv2.minAreaRect
/// equivalent): convex hull + exhaustive hull-edge orientations. Returns
/// (cx, cy, w, h, angle_rad) with w = LONGER side and angle = angle of that
/// side from +x (radians, y-down image coords).
fn min_area_rect(points: &[imageproc::point::Point<i32>]) -> Option<(f32, f32, f32, f32, f32)> {
    let hull = imageproc::geometry::convex_hull(points.to_vec());
    if hull.len() < 3 {
        return None;
    }
    let n = hull.len();
    let mut best: Option<(f32, f32, f32, f32)> = None; // (cx, cy, area, angle)
    let mut best_w = 0.0f32;
    let mut best_h = 0.0f32;
    for i in 0..n {
        let p1 = hull[i];
        let p2 = hull[(i + 1) % n];
        let dx = (p2.x - p1.x) as f32;
        let dy = (p2.y - p1.y) as f32;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            continue;
        }
        let (ux, uy) = (dx / len, dy / len);
        let (vx, vy) = (-uy, ux);
        let mut min_u = f32::MAX;
        let mut max_u = f32::MIN;
        let mut min_v = f32::MAX;
        let mut max_v = f32::MIN;
        for p in &hull {
            let u = p.x as f32 * ux + p.y as f32 * uy;
            let v = p.x as f32 * vx + p.y as f32 * vy;
            min_u = min_u.min(u);
            max_u = max_u.max(u);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let w = max_u - min_u;
        let h = max_v - min_v;
        let area = w * h;
        if best.map_or(true, |b| area < b.2) {
            let uc = (min_u + max_u) / 2.0;
            let vc = (min_v + max_v) / 2.0;
            best = Some((
                uc * ux + vc * vx,
                uc * uy + vc * vy,
                area,
                uy.atan2(ux),
            ));
            best_w = w;
            best_h = h;
        }
    }
    let (cx, cy, _area, angle) = best?;
    // Normalize: w = longer side, angle follows the long axis.
    if best_w >= best_h {
        Some((cx, cy, best_w, best_h, angle))
    } else {
        Some((cx, cy, best_h, best_w, angle + std::f32::consts::FRAC_PI_2))
    }
}

/// Rotate an image about its center by `angle` radians. Positive angle
/// rotates the content toward +y (clockwise visually in y-down coords).
/// Output keeps the input dimensions; out-of-bounds pixels are white.
/// Nearest-neighbor sampling — fine for OCR crops, avoids interpolation blur.
fn rotate_crop(img: &DynamicImage, angle: f32) -> DynamicImage {
    let (w, h) = img.dimensions();
    let (wc, hc) = (w as f32 / 2.0, h as f32 / 2.0);
    let (s, c) = angle.sin_cos();
    let mut out = RgbaImage::from_pixel(w, h, Rgba([255, 255, 255, 255]));
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - wc;
            let dy = y as f32 - hc;
            let sx = c * dx + s * dy + wc; // R(-angle) = [c, s; -s, c]
            let sy = -s * dx + c * dy + hc;
            if sx >= 0.0 && sy >= 0.0 && sx < w as f32 && sy < h as f32 {
                out.put_pixel(x, y, img.get_pixel(sx as u32, sy as u32));
            }
        }
    }
    DynamicImage::ImageRgba8(out)
}

/// Mobile shared orientation rule (#28): near-square boxes count as vertical
/// for the ruby checks, so lone upright characters are tested against both
/// rules.
fn is_vertical_box(b: &BoundingBox) -> bool {
    b.h as f32 >= b.w as f32 * VERTICAL_MIN_ASPECT
}

fn is_square_box(b: &BoundingBox) -> bool {
    let (w, h) = (b.w as f32, b.h as f32);
    w.min(h) >= w.max(h) / VERTICAL_MIN_ASPECT
}

fn overlap_len(a1: i32, a2: i32, b1: i32, b2: i32) -> i32 {
    (a2.min(b2) - a1.max(b1)).max(0)
}

fn gap_len(a1: i32, a2: i32, b1: i32, b2: i32) -> i32 {
    (a1.max(b1) - a2.min(b2)).max(0)
}

/// Tiny vertical box hugging a much larger vertical box (either side) (#28).
/// The center must lie OUTSIDE the big box: stacked column fragments (tail of
/// the column above/below, overlapping only via unclip padding) share its
/// x-range. Size/center/overlap use RAW contour geometry; gap uses UNCLIPPED
/// (raw gutters are real pixels, unclip closes them to ruby distance).
fn is_ruby_vertical(
    s_raw: &BoundingBox,
    b_raw: &BoundingBox,
    s_un: &BoundingBox,
    b_un: &BoundingBox,
    img_h: i32,
) -> bool {
    let img_h = img_h as f32;
    let (sh, bh) = (s_raw.h as f32, b_raw.h as f32);
    if bh < img_h * FURIGANA_BIG_MIN_FRAC {
        return false;
    }
    if sh >= bh * FURIGANA_SIZE_RATIO {
        return false;
    }
    if sh >= img_h * FURIGANA_MAX_FRAC {
        return false;
    }
    if s_un.w as f32 >= b_un.w as f32 * FURIGANA_WIDTH_RATIO {
        return false;
    }
    let cx = (s_raw.x + s_raw.x + s_raw.w) / 2;
    if cx >= b_raw.x && cx <= b_raw.x + b_raw.w {
        return false;
    }
    if (gap_len(s_un.x, s_un.x + s_un.w, b_un.x, b_un.x + b_un.w) as f32)
        > b_un.w as f32 * FURIGANA_GAP_RATIO
    {
        return false;
    }
    if (overlap_len(s_raw.y, s_raw.y + s_raw.h, b_raw.y, b_raw.y + b_raw.h) as f32)
        < sh * FURIGANA_OVERLAP_RATIO
    {
        return false;
    }
    true
}

/// Tiny horizontal box right above a much larger horizontal box (#28).
/// Judged by THINNESS alone, not length: horizontal ruby runs long or short,
/// but its glyphs are always smaller.
fn is_ruby_horizontal(
    s_raw: &BoundingBox,
    b_raw: &BoundingBox,
    s_un: &BoundingBox,
    b_un: &BoundingBox,
    img_w: i32,
    img_h: i32,
) -> bool {
    let (img_w, img_h) = (img_w as f32, img_h as f32);
    if (b_raw.w as f32) < img_w * FURIGANA_BIG_MIN_FRAC {
        return false;
    }
    if (s_raw.h as f32) >= b_raw.h as f32 * FURIGANA_THIN_RATIO {
        return false;
    }
    if s_raw.h as f32 >= img_h * FURIGANA_MAX_FRAC {
        return false;
    }
    if (s_un.h as f32) >= b_un.h as f32 * FURIGANA_HSHORT_RATIO {
        return false;
    }
    // Above-ness on RAW geometry: unclip grows both boxes toward each other,
    // flipping genuinely-above ruby to overlapping.
    if s_raw.y + s_raw.h > b_raw.y + 2 {
        return false;
    }
    if (b_un.y - (s_un.y + s_un.h)) as f32 > (b_un.h as f32) * FURIGANA_GAP_RATIO {
        return false;
    }
    if (overlap_len(s_raw.x, s_raw.x + s_raw.w, b_raw.x, b_raw.x + b_raw.w) as f32)
        < s_raw.w as f32 * FURIGANA_OVERLAP_RATIO
    {
        return false;
    }
    true
}

/// Mobile `filterFurigana` (#28): keep-flags for likely-furigana boxes.
/// `raw`/`uncl` are index-aligned (raw contour AABBs vs unclipped boxes).
fn filter_furigana(raw: &[BoundingBox], uncl: &[BoundingBox], img_w: i32, img_h: i32) -> Vec<bool> {
    if raw.len() < 2 {
        return vec![true; raw.len()];
    }
    (0..raw.len())
        .map(|i| {
            let small = &raw[i];
            let check_vert = is_vertical_box(small) || is_square_box(small);
            let check_horiz = !is_vertical_box(small) || is_square_box(small);
            !raw.iter().enumerate().any(|(j, big)| {
                j != i
                    && ((check_vert
                        && is_vertical_box(big)
                        && is_ruby_vertical(&raw[i], big, &uncl[i], &uncl[j], img_h))
                        || (check_horiz
                            && !is_vertical_box(big)
                            && is_ruby_horizontal(&raw[i], big, &uncl[i], &uncl[j], img_w, img_h)))
            })
        })
        .collect()
}

/// Mobile `findRubyGutterCut` (#48): cut x (image coords) for a ruby-widened
/// vertical box, or None to keep. Per-column ink profile over the full box
/// height; the leftmost clean gutter (>= 3 near-empty columns) with ink
/// following it inside [L+0.40W, L+0.80W] is the main/ruby gutter — cut at
/// its start. Touching ruby with no clean gutter but a thin spot falls back
/// to half width. Polarity/thresholds shared with the recognizer.
fn find_ruby_gutter_cut(b: &BoundingBox, image: &DynamicImage) -> Option<i32> {
    let (iw, ih) = (image.width() as i32, image.height() as i32);
    let x0 = b.x.clamp(0, iw - 1);
    let x1 = (b.x + b.w).clamp(1, iw);
    let y0 = b.y.clamp(0, ih - 1);
    let y1 = (b.y + b.h).clamp(1, ih);
    let bw = (x1 - x0) as u32;
    let bh = (y1 - y0) as u32;
    if bw < 24 || bh < 64 {
        return None;
    }
    let (bw, bh) = (bw as usize, bh as usize);
    let crop = image.crop_imm(x0 as u32, y0 as u32, bw as u32, bh as u32).to_rgb8();
    let lum: Vec<f32> = crop
        .pixels()
        .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)
        .collect();
    // Background polarity from border samples (shared with snapping).
    let mut border: Vec<f32> = Vec::new();
    let mut bi = 0usize;
    while bi < bw {
        border.push(lum[bi]);
        border.push(lum[(bh - 1) * bw + bi]);
        bi += 7;
    }
    bi = 0;
    while bi < bh {
        border.push(lum[bi * bw]);
        border.push(lum[bi * bw + bw - 1]);
        bi += 7;
    }
    if border.is_empty() {
        return None;
    }
    border.sort_by(f32::total_cmp);
    let bg_light = border[border.len() / 2] > 128.0;
    let is_ink = |v: f32| if bg_light { v < 110.0 } else { v > 145.0 };
    // Per-column ink fraction over the full box height.
    let frac: Vec<f32> = (0..bw)
        .map(|x| {
            let mut m = 0usize;
            for y in 0..bh {
                if is_ink(lum[y * bw + x]) {
                    m += 1;
                }
            }
            m as f32 / bh as f32
        })
        .collect();
    let lo = ((bw as f32 * 0.40) as usize).min(bw - 1);
    let hi = (((bw as f32 * 0.80) as usize).max(lo + 1)).min(bw);
    // Leftmost clean gutter with ink following it (not trailing padding).
    let mut x = lo;
    while x + 2 < hi {
        if frac[x] < 0.04 && frac[x + 1] < 0.04 && frac[x + 2] < 0.04 {
            let mut follows = false;
            for k in x + 3..(x + 11).min(bw) {
                if frac[k] >= 0.04 {
                    follows = true;
                    break;
                }
            }
            if follows {
                return Some(x0 + x as i32);
            }
            x += 3;
        } else {
            x += 1;
        }
    }
    // Touching-ruby fallback: thin spot → half width ("remove the right half").
    let mut min_f = f32::MAX;
    for k in lo..hi {
        min_f = min_f.min(frac[k]);
    }
    if min_f < 0.06 {
        return Some(x0 + (bw / 2) as i32);
    }
    None
}

/// Mobile `trimRubyGutterVertical` (#48): furigana-widened vertical boxes are
/// cut back to the main column here. A vertical box is a candidate when wider
/// than 1.35x the median vertical width with a removable strip >= 12px; the
/// cut must keep the left 40% and leave 8px on the right. Rotated quads are
/// left alone (mobile has no rotated boxes; AABB-space cuts would desync).
/// Returns the number of boxes trimmed.
fn trim_ruby_gutter_vertical(
    pairs: &mut Vec<(BoundingBox, RotatedBox)>,
    image: &DynamicImage,
) -> usize {
    let mut vert_w: Vec<i32> = pairs
        .iter()
        .filter(|(b, _)| is_vertical_box(b))
        .map(|(b, _)| b.w)
        .collect();
    if vert_w.len() < 2 {
        return 0;
    }
    vert_w.sort_unstable();
    let med_w = vert_w[vert_w.len() / 2];
    if med_w <= 0 {
        return 0;
    }
    let mut trimmed = 0usize;
    for (b, r) in pairs.iter_mut() {
        if r.is_rotated() || !is_vertical_box(b) {
            continue;
        }
        let w = b.w;
        if (w as f32) <= med_w as f32 * 1.35 || w - med_w < 12 {
            continue;
        }
        let Some(cut) = find_ruby_gutter_cut(b, image) else {
            continue;
        };
        if cut <= b.x + 20 || cut >= b.x + b.w - 8 {
            continue;
        }
        if ((cut - b.x) as f32) < (w as f32 * 0.4).round() {
            continue;
        }
        eprintln!(
            "[PP-OCR DET] rubyTrim {}x{}@({},{}) -> w={} (medW={})",
            w,
            b.h,
            b.x,
            b.y,
            cut - b.x,
            med_w
        );
        let delta = (b.x + b.w - cut) as f32;
        b.w = cut - b.x;
        // Non-rotated vertical: the short side is the width; keep the quad
        // in sync so crops/char boxes follow the trim.
        r.h -= delta;
        r.cx -= delta / 2.0;
        trimmed += 1;
    }
    trimmed
}

/// Cross-axis size of a character cell: fullwidth JP occupies the line's
/// cross size, halfwidth latin/katakana half of it.
fn char_cell_width(ch: Option<char>, cross: f32) -> f32 {
    match ch {
        Some(c) if crate::util::japanese::is_half_width(c) => cross * 0.5,
        _ => cross,
    }
}

/// Max length of a character cell along the reading axis: 1.1x its own width
/// (so 1.1x the line cross size for JP, 0.55x for halfwidth latin).
fn char_cell_max_len(ch: Option<char>, cross: f32) -> f32 {
    char_cell_width(ch, cross) * 1.1
}

/// Resolve one overlapping pair of character cells: move the shared boundary
/// to where both boxes reach the same aspect ratio (length / width), clamped
/// into the overlap so at worst the edges align and the boxes just touch.
/// Cells are `(start, end)` along the reading axis.
fn split_overlap_same_aspect(a: &mut (f32, f32), b: &mut (f32, f32), wa: f32, wb: f32) {
    if a.1 <= b.0 {
        return;
    }
    let m = (wa * b.1 + wb * a.0) / (wa + wb);
    let m = m.clamp(b.0, a.1);
    a.1 = m;
    b.0 = m;
}

impl OcrEngine {
    pub fn new(model_dir: &str, recognition_mode: RecognitionMode, batch_size: usize) -> Result<Self> {
        let model_path = Path::new(model_dir);

        // PP-OCRv6 ncnn detection (shared core with the mobile app).
        let det_param = model_path.join("PP-OCRv6_small_ncnn").join("det.param");
        let det_bin = model_path.join("PP-OCRv6_small_ncnn").join("det.bin");
        let det_net = DetNet::create(&det_param, &det_bin)
            .with_context(|| format!("Failed to load PP-OCRv6 detection model at {det_param:?}"))?;
        println!("Detection model loaded (ncnn).");

        let (ppocr_rec, ppocr_vocab, rec_remap) = load_ppocr_models(model_path)?;
        if ppocr_rec.is_none() {
            eprintln!("[PP-OCR] ncnn rec unavailable — recognition disabled");
        }

        println!("Recognition mode: {:?}, batch size: {}", recognition_mode, batch_size);

        Ok(OcrEngine {
            det_net,
            ppocr_rec,
            ppocr_vocab,
            rec_remap,
            recognition_mode,
            batch_size,
        })
    }

    pub fn is_ready(&self) -> bool {
        self.ppocr_rec.is_some() && !self.ppocr_vocab.is_empty() && !self.rec_remap.is_empty()
    }

    /// Detects bounding boxes using PP-OCRv6 segmentation-based detection model.
    /// The model outputs a probability map [1,1,H,W]. Post-processing:
    /// threshold → connected components → min-area rotated rects (PP-OCR's
    /// minAreaRect) → sort. Returns axis-aligned boxes for display plus the
    /// rotated rects for crop un-rotation.
    pub fn detect(&mut self, image: &DynamicImage) -> Result<DetectionResult> {
        let orig_w = image.width() as f32;
        let orig_h = image.height() as f32;

        // 1. Letterbox: keep aspect ratio, longest side = the net input side
        // (mobile #51: 896 by default, DET_MODEL_SIZE env for A/B), center the
        // content in a square and fill the border with gray. This matches the
        // mobile detect() preprocessing (and therefore the model's training
        // geometry) instead of the old top-left 32-multiple padding.
        let model_size = std::env::var("DET_MODEL_SIZE")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(PPOCR_DET_MODEL_SIZE)
            .clamp(320, 960);
        let target_long = PPOCR_DET_LONG_SIDE.min(model_size);
        let scale = target_long as f32 / orig_w.max(orig_h);
        let resize_w = ((orig_w * scale).round() as u32).max(32);
        let resize_h = ((orig_h * scale).round() as u32).max(32);
        let img_left = (model_size - resize_w) / 2;
        let img_top = (model_size - resize_h) / 2;

        let resized = image.resize_exact(resize_w, resize_h, image::imageops::FilterType::Triangle);
        let mut letterbox =
            RgbaImage::from_pixel(model_size, model_size, Rgba([128u8, 128u8, 128u8, 255u8]));
        image::imageops::replace(&mut letterbox, &resized.to_rgba8(), img_left as i64, img_top as i64);

        // 2. Convert to NCHW float with ImageNet normalization, same per-pixel
        // math as the mobile DET_NORM_LUT ((v/255 - mean) / std per channel).
        let w = model_size as usize;
        let h = w;
        let mut data = vec![0.0f32; 3 * h * w];
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        for y in 0..h {
            for x in 0..w {
                let px = letterbox.get_pixel(x as u32, y as u32);
                let r = (px.0[0] as f32 / 255.0 - mean[0]) / std[0];
                let g = (px.0[1] as f32 / 255.0 - mean[1]) / std[1];
                let b = (px.0[2] as f32 / 255.0 - mean[2]) / std[2];
                let idx = y * w + x;
                data[idx] = r;
                data[h * w + idx] = g;
                data[2 * h * w + idx] = b;
            }
        }

        // 3. Run the shared ncnn DB segmentation net.
        let raw_map = self.det_net.infer(&data, w, h).with_context(|| {
            format!("PP-OCRv6 det ncnn inference failed ({model_size}x{model_size})")
        })?;

        // The net should emit model_size²; a smaller square (stride) gets
        // nearest-upsampled like mobile, so downstream coords are always in
        // letterbox space.
        let prob_map: Vec<f32> = if raw_map.len() == h * w {
            raw_map
        } else {
            let dim = (raw_map.len() as f64).sqrt() as u32;
            if dim == 0 || (dim * dim) as usize != raw_map.len() || dim > model_size {
                anyhow::bail!(
                    "unexpected det output size {} for {}x{} input",
                    raw_map.len(),
                    model_size,
                    model_size
                );
            }
            let mut up = vec![0.0f32; h * w];
            let step = dim as f32 / model_size as f32;
            for y in 0..h {
                for x in 0..w {
                    let sx = ((x as f32 * step) as u32).min(dim - 1) as usize;
                    let sy = ((y as f32 * step) as u32).min(dim - 1) as usize;
                    up[y * w + x] = raw_map[sy * dim as usize + sx];
                }
            }
            eprintln!("[PP-OCR DET] upsampled det output {dim}x{dim} -> {model_size}x{model_size}");
            up
        };
        // Debug: print prob_map statistics to understand value range
        if !prob_map.is_empty() {
            let max_val = prob_map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_val = prob_map.iter().cloned().fold(f32::INFINITY, f32::min);
            let sum: f32 = prob_map.iter().sum();
            let mean = sum / prob_map.len() as f32;
            eprintln!("[PP-OCR DET] prob_map: min={min_val:.4} max={max_val:.4} mean={mean:.4}");
        }
        let out_w = model_size;
        let out_h = model_size;

        // Scale factors from letterbox space to the original image; content
        // sits at (img_left, img_top) after centering.
        let scale_w = orig_w / resize_w as f32;
        let scale_h = orig_h / resize_h as f32;

        // 5. Threshold → find contours → bounding boxes
        // DET_THRESH / DET_UNCLIP env overrides let us sweep parameters
        // without rebuilding (defaults = the consts above).
        let det_thresh: f32 = std::env::var("DET_THRESH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(PPOCR_DET_THRESH);
        let det_unclip: f32 = std::env::var("DET_UNCLIP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(PPOCR_DET_UNCLIP_RATIO);
        let mut binary = vec![0u8; (out_w * out_h) as usize];
        for y in 0..out_h {
            for x in 0..out_w {
                let idx = (y * out_w + x) as usize;
                if idx < prob_map.len() && prob_map[idx] > det_thresh {
                    binary[idx] = 255;
                }
            }
        }

        let binary_img = image::GrayImage::from_raw(out_w, out_h, binary.clone())
            .context("Failed to create binary image")?;
        let contours = contours::find_contours_with_threshold::<i32>(&binary_img, 128);
        let mut raw_pairs: Vec<(BoundingBox, RotatedBox)> = Vec::new();
        // Index-aligned with raw_pairs: pre-unclip raw contour AABBs (mobile
        // rawPreBoxes) and unclipped AABBs (rawBoxes) for the furigana filter.
        let mut pre_boxes: Vec<BoundingBox> = Vec::new();
        let mut uncl_boxes: Vec<BoundingBox> = Vec::new();

        for contour in &contours {
            if contour.points.len() < 3 { continue; } // noise filter

            // Minimum-area ROTATED rectangle around the contour (PP-OCR's
            // cv2.minAreaRect equivalent). Angled text lines get a tight
            // quad instead of one big orthogonal box covering the span.
            let Some((cx, cy, rw, rh, angle)) = min_area_rect(&contour.points) else {
                continue;
            };

            // Unclip: expand box using proper PP-OCR formula:
            // distance = area * ratio / perimeter (applied in rect space)
            let area = rw * rh;
            let perimeter = 2.0 * (rw + rh);
            let expand = if perimeter > 0.0 { area * det_unclip / perimeter } else { 0.0 };
            // Cap the expansion in ORIGINAL pixels. The det model's prob
            // blob covers only ~85% of large text (model property), so the
            // proportional unclip alone leaves large boxes tight (≈1.0×ink);
            // a fixed-px cap adds real margin there without inflating the
            // already-bloated small-text boxes.
            let cap: f32 = std::env::var("DET_EXPAND_CAP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(48.0);
            let contour_w_det = rw;
            let contour_h_det = rh;
            // Raw contour box mapped to original image coords, BEFORE the
            // unclip expansion — the furigana filter's geometry (#28).
            let raw_cx = (cx - img_left as f32) * scale_w;
            let raw_cy = (cy - img_top as f32) * scale_h;
            let raw_w = rw * scale_w;
            let raw_h = rh * scale_h;
            // Expansion in ORIGINAL pixels, capped in ORIGINAL pixels.
            let expand_orig = (expand * scale_w).min(cap);
            let rw = rw * scale_w + 2.0 * expand_orig;
            let mut rh = rh * scale_h + 2.0 * expand_orig;

            // Orientation-specific post-adjustments (env-tunable):
            // - horizontal lines: nudge the box DOWN a bit — glyphs sit
            //   slightly above the contour center on most fonts.
            // - vertical lines: optional width trim (DET_V_SHRINK; 1.0 =
            //   off). Disabled by default — trims risk cutting glyphs.
            let h_angle = angle.to_degrees().abs();
            let mut down_shift: f32 = 0.0;
            if h_angle <= 45.0 {
                let h_down: f32 = std::env::var("DET_H_DOWN")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(4.0);
                if rh >= 24.0 {
                    rh += h_down;
                    down_shift = h_down / 2.0;
                }
            } else {
                let v_trim: f32 = std::env::var("DET_V_SHRINK")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0);
                rh *= v_trim;
            }

            // Scale centers back to original image coords: undo the centered
            // letterbox offset first, then the uniform resize scale.
            let cx = (cx - img_left as f32) * scale_w;
            let cy = (cy - img_top as f32) * scale_h + down_shift;

            if rw < 4.0 || rh < 4.0 { continue; }

            eprintln!(
                "[contour] det {:.1}x{:.1} expand {:.1} -> orig {:.0}x{:.0}",
                contour_w_det, contour_h_det, expand, rw, rh
            );

            let rot = RotatedBox::new(cx, cy, rw, rh, angle, PPOCR_DET_BOX_THRESH);
            let (bx, by, bw, bh) = rot.aabb();
            let bbox = BoundingBox::new(
                bx.round() as i32,
                by.round() as i32,
                bw.round() as i32,
                bh.round() as i32,
                PPOCR_DET_BOX_THRESH,
            );
            let raw_rot = RotatedBox::new(raw_cx, raw_cy, raw_w, raw_h, angle, PPOCR_DET_BOX_THRESH);
            let (rx, ry, rww, rhh) = raw_rot.aabb();
            pre_boxes.push(BoundingBox::new(
                rx.round() as i32,
                ry.round() as i32,
                rww.round() as i32,
                rhh.round() as i32,
                PPOCR_DET_BOX_THRESH,
            ));
            uncl_boxes.push(bbox.clone());
            raw_pairs.push((bbox, rot));
        }

        // Furigana line rejection (#28): drop boxes that are ruby to a nearby
        // larger box. Mobile matches on raw contour geometry (pre-unclip, so
        // stacked column fragments are not welded together by unclip padding)
        // and uses the unclipped boxes only for the gap test.
        let keep = filter_furigana(&pre_boxes, &uncl_boxes, orig_w as i32, orig_h as i32);
        let dropped: usize = keep.iter().filter(|k| !**k).count();
        if dropped > 0 {
            for (pre, k) in pre_boxes.iter().zip(keep.iter()) {
                if !k {
                    eprintln!(
                        "[PP-OCR DET] furigana dropped {}x{}@({}, {})",
                        pre.w, pre.h, pre.x, pre.y
                    );
                }
            }
            let mut idx = 0usize;
            raw_pairs.retain(|_| {
                let k = keep[idx];
                idx += 1;
                k
            });
            eprintln!("[PP-OCR DET] furigana {} -> {} boxes", pre_boxes.len(), raw_pairs.len());
        }

        // Debug: print the contour boxes (after furigana rejection)
        eprintln!("[PP-OCR DET] {} boxes:", raw_pairs.len());
        for (i, (b, r)) in raw_pairs.iter().enumerate() {
            eprintln!(
                "  [{i}] x={} y={} w={} h={} angle={:.1}° c={:.3}",
                b.x, b.y, b.w, b.h, r.angle.to_degrees(), b.confidence
            );
        }

        // MERGE DISABLED for diagnosis
        let sorted = self.sort_detected_boxes(raw_pairs);
        // Post-processing
        let mut pp_pairs = sorted;
        // Filter out degenerate tiny boxes (noise specks)
        pp_pairs.retain(|(b, _)| b.w >= 10 && b.h >= 10);
        // 1. Shrink vertical box widths by 10% (centered; mobile shrink).
        // The old one-sided x offset never shrank the width, which both
        // skewed the box and inflated the ruby-trim reference width.
        for (b, r) in pp_pairs.iter_mut().filter(|(b, _)| b.h > b.w) {
            let shrink = (b.w as f32 * 0.05).round();
            if r.h > 2.0 * shrink {
                r.h -= 2.0 * shrink;
                let (nx, ny, nw, nh) = r.aabb();
                *b = BoundingBox::new(
                    nx.round() as i32,
                    ny.round() as i32,
                    nw.round() as i32,
                    nh.round() as i32,
                    b.confidence,
                );
            }
        }
        // 1b. Ruby-gutter trim (#48): detector boxes that swallowed the
        // furigana strip (~2x normal column width) are cut back to the main
        // column here, before cropping, so the ruby width never enters
        // recognition.
        let trimmed = trim_ruby_gutter_vertical(&mut pp_pairs, image);
        if trimmed > 0 {
            eprintln!("[PP-OCR DET] rubyTrim applied to {trimmed} box(es)");
        }
        // 2. Stacked-overlap split — DISABLED: with merging off, detection
        // already produces per-line boxes; splitting stacked lines at the
        // overlap midpoint shaves real text (boxes end up smaller than the
        // glyphs, most visibly on large close-spaced text). Kept behind an
        // env flag in case it's ever needed again.
        if std::env::var("DET_SPLIT_OVERLAP").is_ok() {
            let h_indices: Vec<usize> = pp_pairs
                .iter()
                .enumerate()
                .filter(|(_, (b, _))| b.w >= b.h)
                .map(|(i, _)| i)
                .collect();
            for i in 0..h_indices.len() {
                for j in (i + 1)..h_indices.len() {
                    let ai = h_indices[i];
                    let bi = h_indices[j];
                    // Only split if boxes are in the same column (horizontal overlap too)
                    let a_right = pp_pairs[ai].0.x + pp_pairs[ai].0.w;
                    let b_right = pp_pairs[bi].0.x + pp_pairs[bi].0.w;
                    let h_overlap = a_right.min(b_right) - pp_pairs[ai].0.x.max(pp_pairs[bi].0.x);
                    if h_overlap <= 0 {
                        continue;
                    }
                    let (upper, lower) = if pp_pairs[ai].0.y <= pp_pairs[bi].0.y {
                        (ai, bi)
                    } else {
                        (bi, ai)
                    };
                    let upper_bottom = pp_pairs[upper].0.y + pp_pairs[upper].0.h;
                    let lower_bottom = pp_pairs[lower].0.y + pp_pairs[lower].0.h;
                    // Check vertical overlap: upper box bottom > lower box top
                    if upper_bottom > pp_pairs[lower].0.y {
                        let overlap_mid = (pp_pairs[lower].0.y + upper_bottom.min(lower_bottom)) / 2;
                        pp_pairs[upper].0.h = (overlap_mid - pp_pairs[upper].0.y).max(1);
                        pp_pairs[lower].0.y = pp_pairs[upper].0.y + pp_pairs[upper].0.h;
                        pp_pairs[lower].0.h = (lower_bottom - pp_pairs[lower].0.y).max(1);
                    }
                }
            }
        }
        eprintln!("[PP-OCR DET] final {} boxes:", pp_pairs.len());
        for (i, (b, r)) in pp_pairs.iter().enumerate() {
            eprintln!(
                "  [{i}] x={} y={} w={} h={} angle={:.1}° c={:.3}",
                b.x, b.y, b.w, b.h, r.angle.to_degrees(), b.confidence
            );
        }

        let boxes: Vec<BoundingBox> = pp_pairs.iter().map(|(b, _)| b.clone()).collect();
        let rotated: Vec<RotatedBox> = pp_pairs.into_iter().map(|(_, r)| r).collect();

        // Debug: draw the detected boxes over the original image when
        // DET_DEBUG_DIR is set. Green = rotated quad, red = AABB.
        if let Ok(debug_dir) = std::env::var("DET_DEBUG_DIR") {
            let dir = std::path::Path::new(&debug_dir);
            let _ = std::fs::create_dir_all(dir);
            // Also dump the prob map + thresholded binary for inspection.
            if std::env::var("DET_DUMP_MAPS").is_ok() {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let pm: Vec<u8> = prob_map
                    .iter()
                    .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                    .collect();
                if let Some(img) = image::GrayImage::from_raw(out_w, out_h, pm) {
                    let _ = img.save(dir.join(format!("prob_{ts}.png")));
                }
                if let Some(img) = image::GrayImage::from_raw(out_w, out_h, binary.clone()) {
                    let _ = img.save(dir.join(format!("binary_{ts}.png")));
                }
            }
            let mut rgba = image.to_rgba8();
            for (b, r) in boxes.iter().zip(rotated.iter()) {
                // AABB in red
                let rect = imageproc::rect::Rect::at(b.x, b.y).of_size(b.w.max(1) as u32, b.h.max(1) as u32);
                imageproc::drawing::draw_hollow_rect_mut(
                    &mut rgba, rect, Rgba([255u8, 60u8, 60u8, 255u8]));
                // Rotated quad in green
                let (ux, uy) = (r.angle.cos(), r.angle.sin());
                let (vx, vy) = (-uy, ux);
                let hw = r.w / 2.0;
                let hh = r.h / 2.0;
                let pts = [
                    (r.cx + ux * hw + vx * hh, r.cy + uy * hw + vy * hh),
                    (r.cx - ux * hw + vx * hh, r.cy - uy * hw + vy * hh),
                    (r.cx - ux * hw - vx * hh, r.cy - uy * hw - vy * hh),
                    (r.cx + ux * hw - vx * hh, r.cy + uy * hw - vy * hh),
                ];
                for k in 0..4 {
                    let (x0, y0) = pts[k];
                    let (x1, y1) = pts[(k + 1) % 4];
                    imageproc::drawing::draw_line_segment_mut(
                        &mut rgba, (x0, y0), (x1, y1), Rgba([60u8, 255u8, 60u8, 255u8]));
                }
            }
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            let out = dir.join(format!("det_debug_{ts}.png"));
            if let Err(e) = rgba.save(&out) {
                eprintln!("[PP-OCR DET] debug save failed: {e}");
            } else {
                println!("[PP-OCR DET] debug overlay saved to {}", out.display());
            }
        }

        Ok(DetectionResult { boxes, rotated })
    }

    pub fn merge_overlapping_boxes(&self, boxes: Vec<BoundingBox>) -> Vec<BoundingBox> {
        if boxes.is_empty() {
            return boxes;
        }

        let mut result: Vec<BoundingBox> = Vec::new();
        let mut handled = vec![false; boxes.len()];

        for (idx, box_a) in boxes.iter().enumerate() {
            if handled[idx] {
                continue;
            }

            let mut current_merged = box_a.clone();

            for (idx_b, box_b) in boxes.iter().enumerate() {
                if idx == idx_b || handled[idx_b] {
                    continue;
                }

                if self.should_merge_boxes(&current_merged, box_b) {
                    // Merge box_b into current_merged
                    let new_x = current_merged.left().min(box_b.left());
                    let new_y = current_merged.top().min(box_b.top());
                    let new_right = current_merged.right().max(box_b.right());
                    let new_bottom = current_merged.bottom().max(box_b.bottom());

                    current_merged = BoundingBox::new(
                        new_x,
                        new_y,
                        new_right - new_x,
                        new_bottom - new_y,
                        current_merged.confidence.max(box_b.confidence),
                    );
                    handled[idx_b] = true;
                }
            }

            result.push(current_merged);
        }

        result
    }

    pub fn should_merge_boxes(&self, a: &BoundingBox, b: &BoundingBox) -> bool {
        let inter_left = a.left().max(b.left());
        let inter_top = a.top().max(b.top());
        let inter_right = a.right().min(b.right());
        let inter_bottom = a.bottom().min(b.bottom());

        if inter_left >= inter_right || inter_top >= inter_bottom {
            return false;
        }

        let inter_area = (inter_right - inter_left) * (inter_bottom - inter_top);
        let area_a = a.area();
        let area_b = b.area();
        let min_area = area_a.min(area_b);

        if min_area == 0 {
            return false;
        }

        let iom = inter_area as f32 / min_area as f32;
        if iom < X_OVERLAP_THRESHOLD as f32 {
            return false;
        }

        // Check if boxes are on the same row (vertical overlap)
        let y_diff = (a.y + a.h / 2) - (b.y + b.h / 2);
        let avg_height = (a.h + b.h) / 2;

        if y_diff.abs() > avg_height {
            return false;
        }

        true
    }

    pub fn sort_detected_boxes(&self, mut boxes: Vec<(BoundingBox, RotatedBox)>) -> Vec<(BoundingBox, RotatedBox)> {
        // Separate by orientation
        let mut horizontal: Vec<(BoundingBox, RotatedBox)> = Vec::new();
        let mut vertical: Vec<(BoundingBox, RotatedBox)> = Vec::new();
        for pair in boxes.drain(..) {
            if pair.0.w >= pair.0.h {
                horizontal.push(pair);
            } else {
                vertical.push(pair);
            }
        }

        // Horizontal: top-to-bottom, left-to-right
        horizontal.sort_by(|a, b| a.0.y.cmp(&b.0.y).then(a.0.x.cmp(&b.0.x)));

        // Vertical: right-to-left, top-to-bottom
        // Japanese vertical text is read right-to-left across columns.
        vertical.sort_by(|a, b| b.0.x.cmp(&a.0.x).then(a.0.y.cmp(&b.0.y)));

        // Concatenate: horizontal lines first, then vertical
        boxes = horizontal;
        boxes.extend(vertical);
        boxes
    }

    pub fn image_to_nchw(&self, img: &RgbaImage, width: u32, height: u32) -> Vec<f32> {
        let mut img_data = vec![0.0f32; 3 * width as usize * height as usize];
        let w_usize = width as usize;
        let h_usize = height as usize;
        for y in 0..height {
            for x in 0..width {
                let p = img.get_pixel(x, y);
                let r = p[0] as f32 / 255.0;
                let g = p[1] as f32 / 255.0;
                let b = p[2] as f32 / 255.0;
                let x_usize = x as usize;
                let y_usize = y as usize;
                img_data[0 * h_usize * w_usize + y_usize * w_usize + x_usize] = r;
                img_data[1 * h_usize * w_usize + y_usize * w_usize + x_usize] = g;
                img_data[2 * h_usize * w_usize + y_usize * w_usize + x_usize] = b;
            }
        }
        img_data
    }

    pub fn calculate_x_overlap(&self, box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
        let x1_min = box1[0];
        let x1_max = box1[2];
        let x2_min = box2[0];
        let x2_max = box2[2];
        let intersection = (x1_max.min(x2_max) - x1_min.max(x2_min)).max(0.0);
        let w1 = x1_max - x1_min;
        let w2 = x2_max - x2_min;
        if w1 <= 0.0 || w2 <= 0.0 {
            return 0.0;
        }
        intersection / w1.min(w2)
    }

    pub fn calculate_y_overlap(&self, box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
        let y1_min = box1[1];
        let y1_max = box1[3];
        let y2_min = box2[1];
        let y2_max = box2[3];
        let intersection = (y1_max.min(y2_max) - y1_min.max(y2_min)).max(0.0);
        let h1 = y1_max - y1_min;
        let h2 = y2_max - y2_min;
        if h1 <= 0.0 || h2 <= 0.0 {
            return 0.0;
        }
        intersection / h1.min(h2)
    }

    // Modified: return the annotated image in-memory when `render` is true
    /// Just line detection — returns raw bounding boxes (fast, no character recognition).
    /// Used by the streaming OCR pipeline to show the image immediately.
    pub fn detect_lines(&mut self, image: &DynamicImage) -> Result<DetectionResult> {
        let det = self.detect(image)?;
        let pairs: Vec<(BoundingBox, RotatedBox)> = det.boxes.into_iter().zip(det.rotated).collect();
        // Merge disabled — render ALL boxes
        let sorted = self.sort_detected_boxes(pairs);
        let (boxes, rotated): (Vec<_>, Vec<_>) = sorted.into_iter().unzip();
        Ok(DetectionResult { boxes, rotated })
    }
}


// ---------------------------------------------------------------------------
// Streaming recognition (standalone, Send-friendly)
// ---------------------------------------------------------------------------

/// Sequential ID for dataset line samples — unique across worker threads and runs.
static NEXT_LINE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Save a recognized line crop as `ocr_line_<ts>_<id>.png` in the system
/// temp dir, with a sidecar `<same-stem>.txt` containing the detected
/// text. Returns the `.txt` path (the viewer rewrites it when the user
/// corrects a character), or None if the crop could not be saved.
fn save_line_sample(crop: &image::DynamicImage, text: &str) -> Option<std::path::PathBuf> {
    let id = NEXT_LINE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    // Always write to the system temp dir. The caller may pass the source
    // image's directory (batch mode), which would clutter it and re-trigger
    // the file watcher; /tmp gets cleared on reboot.
    let stem = std::env::temp_dir().join(format!("ocr_line_{ts}_{id}"));
    if let Err(e) = crop.save(stem.with_extension("png")) {
        eprintln!("[dataset] failed to save line crop: {e}");
        return None;
    }
    let txt = stem.with_extension("txt");
    if let Err(e) = std::fs::write(&txt, text) {
        eprintln!("[dataset] failed to save line text: {e}");
    }
    Some(txt)
}

/// Run character recognition on pre-detected boxes and stream each result
/// over the channel as soon as it's ready. Designed to be called from a
/// background thread: all sessions are `Arc`-wrapped and `Send`.
/// Line crops + detected text are saved into `out_dir` for dataset collection.
pub fn recognize_boxes_streaming(
    image: &DynamicImage,
    sorted: &[BoundingBox],
    rotated: &[RotatedBox],
    rec: Option<std::sync::Arc<RecNet>>,
    ppocr_vocab: &[String],
    rec_remap: &[i32],
    _batch_size: usize,
    _recognition_mode: RecognitionMode,
    sender: std::sync::mpsc::Sender<(usize, DetectedAnnotation)>,
    _out_dir: &std::path::Path,
) -> Result<()> {
    use std::time::Instant;
    let t_recognize = Instant::now();

    // Build a single job queue from ALL boxes (horizontal + vertical).
    // Each job carries its orientation so workers can choose the right
    // character-box computation and post-processing.
    struct Job {
        idx: usize,
        bbox: BoundingBox,
        crop: DynamicImage,
        crop_x: u32,
        crop_y: u32,
        crop_w: u32,
        crop_h: u32,
        is_vertical: bool,
        /// Rotated rect for this line; `Some` only when the text axis is
        /// meaningfully off horizontal/vertical (crop was un-rotated).
        rot: Option<RotatedBox>,
    }
    let mut jobs: Vec<Job> = Vec::with_capacity(sorted.len());
    for (i, bbox) in sorted.iter().enumerate() {
        let rot = rotated.get(i).copied();
        let (crop, crop_x, crop_y, crop_w, crop_h, is_vertical, job_rot) = match rot {
            Some(r) if r.is_rotated() => {
                // Angled line: crop the quad's axis-aligned bounds, then
                // un-rotate so the text axis is axis-aligned (the recogniser sees a
                // clean horizontal or vertical line instead of one huge
                // orthogonal crop covering the whole span).
                let (rx, ry, rw, rh) = r.aabb();
                let crop = image.crop_imm(
                    rx.max(0.0) as u32,
                    ry.max(0.0) as u32,
                    rw.max(1.0) as u32,
                    rh.max(1.0) as u32,
                );
                let crop = rotate_crop(&crop, r.unrotate_angle());
                (
                    crop,
                    rx.max(0.0) as u32,
                    ry.max(0.0) as u32,
                    rw.max(1.0) as u32,
                    rh.max(1.0) as u32,
                    r.is_vertical(),
                    Some(r),
                )
            }
            _ => {
                let crop = image.crop_imm(
                    bbox.x.max(0) as u32,
                    bbox.y.max(0) as u32,
                    bbox.w.max(0) as u32,
                    bbox.h.max(0) as u32,
                );
                (
                    crop,
                    bbox.x.max(0) as u32,
                    bbox.y.max(0) as u32,
                    bbox.w.max(0) as u32,
                    bbox.h.max(0) as u32,
                    bbox.h > bbox.w,
                    None,
                )
            }
        };
        if crop.width() < 4 || crop.height() < 4 {
            continue;
        }
        jobs.push(Job {
            idx: i,
            bbox: bbox.clone(),
            crop,
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            is_vertical,
            rot: job_rot,
        });
    }

    if !jobs.is_empty() {
        use std::collections::VecDeque;
        let n_jobs = jobs.len();
        let job_queue = std::sync::Arc::new(std::sync::Mutex::new(VecDeque::from(jobs)));
        let mut handles = Vec::new();

        if !ppocr_vocab.is_empty() && !rec_remap.is_empty() && rec.is_some() {
            let rec = rec.clone().expect("rec is Some");
            // Workers share the one loaded ncnn net; each inference creates
            // its own extractor (the mobile app fans out the same way).
            let workers = std::env::var("PPOCR_REC_WORKERS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(DEFAULT_REC_WORKERS)
                .clamp(1, n_jobs);
            println!(
                "[PP-OCR] Processing {} boxes ({} workers, ncnn rec)",
                n_jobs, workers
            );
            for _worker_id in 0..workers {
            let q = std::sync::Arc::clone(&job_queue);
            let rec = std::sync::Arc::clone(&rec);
            let voc = ppocr_vocab.to_vec();
            let remap = rec_remap.to_vec();
            let snd = sender.clone();

            handles.push(std::thread::spawn(move || loop {
                let job = { let mut ql = q.lock().unwrap(); ql.pop_front() };
                let job = match job { Some(j) => j, None => return };

                let result = crate::ppocr::recognize_ppocr_batch(
                    &rec, &[&job.crop], &voc, &remap,
                );

                if let Ok(mut results) = result {
                    if let Some(res) = results.pop() {
                        let text = res.text;
                        let alternatives = res.alternatives;
                        let char_cols = res.char_cols;
                        let seq_len_total = res.seq_len_total;
                        let step_px = res.step_px;
                        if text.is_empty() { continue; }

                        // Dataset collection: save the line crop + detected text
                        let sample_txt = save_line_sample(&job.crop, &text);

                        let n = char_cols.len();
                        let mut char_boxes = Vec::with_capacity(n);
                        // Char cells are built in the (possibly un-rotated)
                        // crop frame; a rotated line maps each box back to
                        // image coordinates through the inverse rotation.
                        let map_box = |x: f32, y: f32, w: f32, h: f32| -> BoundingBox {
                            let w = w.max(1.0);
                            let h = h.max(1.0);
                            match job.rot.filter(|r| r.is_rotated()) {
                                Some(r) => r.map_char_box(
                                    job.crop_w, job.crop_h, job.crop_x, job.crop_y,
                                    x.round() as i32, y.round() as i32,
                                    w.round() as i32, h.round() as i32,
                                ),
                                None => BoundingBox::new(
                                    (job.crop_x as f32 + x).round() as i32,
                                    (job.crop_y as f32 + y).round() as i32,
                                    w.round() as i32, h.round() as i32, 1.0,
                                ),
                            }
                        };
                        if n > 0 && seq_len_total > 0 && !job.is_vertical {
                            // ---- HORIZONTAL: x-axis char boxes ----
                            // The decoded timestep is the character's centre;
                            // the cell is centred on it with length capped at
                            // 1.1x its width (0.55x for halfwidth latin), then
                            // overlapping neighbours split to a common aspect
                            // ratio. Horizontal punctuation has no special
                            // rules (as before).
                            let step = if step_px > 0.0 { step_px } else { job.crop_w as f32 / seq_len_total as f32 };
                            let cross = (job.crop_h as f32).max(3.0);
                            let text_chars: Vec<char> = text.chars().collect();
                            let mut cells: Vec<(f32, f32)> = char_cols.iter().enumerate().map(|(idx, &t)| {
                                let anchor = (t as f32 + 0.5) * step;
                                let half = char_cell_max_len(text_chars.get(idx).copied(), cross) / 2.0;
                                ((anchor - half).max(0.0), (anchor + half).min(job.crop_w as f32))
                            }).collect();
                            for ci in 0..n.saturating_sub(1) {
                                let wa = char_cell_width(text_chars.get(ci).copied(), cross);
                                let wb = char_cell_width(text_chars.get(ci + 1).copied(), cross);
                                let (left, right) = cells.split_at_mut(ci + 1);
                                split_overlap_same_aspect(&mut left[ci], &mut right[0], wa, wb);
                            }
                            for &(xl, xr) in &cells {
                                char_boxes.push(map_box(xl, 0.0, xr - xl, job.crop_h as f32));
                            }
                        } else if n > 0 && seq_len_total > 0 {
                            // ---- VERTICAL: y-axis char boxes with punct handling ----
                            // Same centred geometry with the column width as
                            // 1: max cell height 1.1x (0.55x halfwidth).
                            // Punctuation keeps its own rules.
                            let step = if step_px > 0.0 { step_px } else { job.crop_h as f32 / seq_len_total as f32 };
                            let cross = (job.crop_w as f32).max(3.0);
                            let text_chars: Vec<char> = text.chars().collect();
                            let mut cells: Vec<(f32, f32)> = char_cols.iter().enumerate().map(|(idx, &t)| {
                                let anchor = (t as f32 + 0.5) * step;
                                let half = char_cell_max_len(text_chars.get(idx).copied(), cross) / 2.0;
                                ((anchor - half).max(0.0), (anchor + half).min(job.crop_h as f32))
                            }).collect();
                            let is_cp: Vec<bool> = text.chars().map(|ch| matches!(ch, '\u{3002}'|'\u{002E}'|'\u{FF0E}'|'\u{3001}'|'\u{002C}'|'\u{FF0C}'|')'|'\u{FF09}'|'\u{3017}'|'\u{300D}'|'\u{300F}'|'\u{3015}'|'\u{3011}'|'\u{3009}'|']'|'\u{FF3D}')).collect();
                            let is_op: Vec<bool> = text.chars().map(|ch| matches!(ch, '('|'\u{FF08}'|'\u{300C}'|'\u{300E}'|'\u{3014}'|'\u{3010}'|'\u{300A}'|'\u{3008}'|'\u{3016}'|'['|'\u{FF3B}')).collect();
                            for ci in 0..n.saturating_sub(1) {
                                if cells[ci].1 <= cells[ci + 1].0 { continue; }
                                if is_cp[ci] { cells[ci].1 = cells[ci + 1].0; }
                                else if is_op[ci + 1] { cells[ci + 1].0 = cells[ci].1; }
                                else if is_cp[ci + 1] { cells[ci + 1].0 = cells[ci].1; }
                                else if is_op[ci] { cells[ci].1 = cells[ci + 1].0; }
                                else {
                                    let wa = char_cell_width(text_chars.get(ci).copied(), cross);
                                    let wb = char_cell_width(text_chars.get(ci + 1).copied(), cross);
                                    let (left, right) = cells.split_at_mut(ci + 1);
                                    split_overlap_same_aspect(&mut left[ci], &mut right[0], wa, wb);
                                }
                            }
                            let avg_np_h: f32 = {
                                let hs: Vec<f32> = cells.iter().enumerate().filter(|(ci,_)| !is_cp[*ci] && !is_op[*ci]).map(|(_,c)| c.1 - c.0).collect();
                                if hs.is_empty() { job.crop_w as f32 } else { hs.iter().sum::<f32>() / hs.len() as f32 }
                            };
                            for ci in 0..n {
                                if is_cp[ci] {
                                    let nx = ((ci + 1)..n).filter(|&j| !is_cp[j] && !is_op[j]).next().map(|j| cells[j].0).unwrap_or(f32::INFINITY);
                                    cells[ci].1 = (cells[ci].0 + avg_np_h).min(nx).max(cells[ci].1);
                                } else if is_op[ci] {
                                    let pb: Option<usize> = (0..ci).rev().filter(|&j| !is_cp[j] && !is_op[j]).next();
                                    let pl = pb.map(|j| cells[j].1).unwrap_or(f32::NEG_INFINITY);
                                    cells[ci].0 = (cells[ci].1 - avg_np_h).max(pl).min(cells[ci].0);
                                }
                            }
                            for &(yt, yb) in &cells {
                                let ch = (yb - yt).max(1.0);
                                char_boxes.push(map_box(0.0, yt, job.crop_w as f32, ch));
                            }
                        }

                        // Text stays as recognised: the overlay picks the
                        // vertical presentation glyphs from the font's GSUB
                        // `vert`/`vrt2` at draw time.
                        let (final_text, final_alts) = (text, alternatives);

                        let annotation = DetectedAnnotation {
                            bbox: job.bbox.clone(),
                            quad: job.rot,
                            line: Some(LineResult {
                                text: final_text,
                                char_boxes,
                                alternatives: final_alts,
                                sample_txt,
                                is_vertical: job.is_vertical,
                                chunk_boxes: vec![BoundingBox::new(
                                    job.crop_x as i32, job.crop_y as i32,
                                    job.crop_w as i32, job.crop_h as i32, 1.0,
                                )],
                            }),
                        };
                        if snd.send((job.idx, annotation)).is_err() { return; }
                        std::thread::yield_now();
                    }
                }
            }));
        }
        }
        for h in handles { h.join().unwrap(); }
    }
    let recognize_ms = t_recognize.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[OCR timing] Character recognition: {:>8.2} ms  ({} boxes, thread)",
        recognize_ms, sorted.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Private helper functions
// ---------------------------------------------------------------------------

fn draw_rectangle(img: &mut RgbaImage, bbox: &BoundingBox, color: Rgba<u8>, thickness: u32) {
    let img_w = img.width() as i32;
    let img_h = img.height() as i32;
    if img_w <= 0 || img_h <= 0 {
        return;
    }

    let left = bbox.left().max(0).min(img_w - 1);
    let top = bbox.top().max(0).min(img_h - 1);
    let right = bbox.right().max(0).min(img_w);
    let bottom = bbox.bottom().max(0).min(img_h);

    for t in 0..(thickness as i32) {
        let lt = left + t;
        let tp = top + t;
        let rt = right - t;
        let bm = bottom - t;

        if lt >= rt || tp >= bm {
            break;
        }

        // top horizontal line
        if tp >= 0 && tp < img_h {
            for x in lt..rt {
                if x >= 0 && x < img_w {
                    img.put_pixel(x as u32, tp as u32, color);
                }
            }
        }

        // bottom horizontal line (at y = bm - 1)
        let yb = bm - 1;
        if yb >= 0 && yb < img_h {
            for x in lt..rt {
                if x >= 0 && x < img_w {
                    img.put_pixel(x as u32, yb as u32, color);
                }
            }
        }

        // left vertical line
        if lt >= 0 && lt < img_w {
            for y in tp..bm {
                if y >= 0 && y < img_h {
                    img.put_pixel(lt as u32, y as u32, color);
                }
            }
        }

        // right vertical line (at x = rt - 1)
        let xr = rt - 1;
        if xr >= 0 && xr < img_w {
            for y in tp..bm {
                if y >= 0 && y < img_h {
                    img.put_pixel(xr as u32, y as u32, color);
                }
            }
        }
    }
}

fn draw_filled_rect(img: &mut RgbaImage, x: i32, y: i32, w: i32, h: i32, color: Rgba<u8>) {
    let img_w = img.width() as i32;
    let img_h = img.height() as i32;
    if img_w <= 0 || img_h <= 0 {
        return;
    }

    let left = x.max(0).min(img_w - 1);
    let top = y.max(0).min(img_h - 1);
    let right = (x + w).max(0).min(img_w);
    let bottom = (y + h).max(0).min(img_h);

    if left >= right || top >= bottom {
        return;
    }

    for yy in top..bottom {
        for xx in left..right {
            img.put_pixel(xx as u32, yy as u32, color);
        }
    }
}

fn draw_text_small(
    img: &mut RgbaImage,
    x: i32,
    y: i32,
    text: &str,
    fg: Rgba<u8>,
    bg: Rgba<u8>,
    scale: u32,
) {
    // Tiny 3x5 font for digits and '.' only. Each entry is 5 rows, bits (2..0) left->right.
    fn glyph(c: char) -> Option<[u8; 5]> {
        match c {
            '0' => Some([0b111, 0b101, 0b101, 0b101, 0b111]),
            '1' => Some([0b010, 0b110, 0b010, 0b010, 0b111]),
            '2' => Some([0b111, 0b001, 0b111, 0b100, 0b111]),
            '3' => Some([0b111, 0b001, 0b111, 0b001, 0b111]),
            '4' => Some([0b101, 0b101, 0b111, 0b001, 0b001]),
            '5' => Some([0b111, 0b100, 0b111, 0b001, 0b111]),
            '6' => Some([0b111, 0b100, 0b111, 0b101, 0b111]),
            '7' => Some([0b111, 0b001, 0b001, 0b001, 0b001]),
            '8' => Some([0b111, 0b101, 0b111, 0b101, 0b111]),
            '9' => Some([0b111, 0b101, 0b111, 0b001, 0b111]),
            '.' => Some([0b000, 0b000, 0b000, 0b000, 0b010]),
            _ => None,
        }
    }

    let chars: Vec<char> = text.chars().collect();
    let n = chars.len() as i32;
    if n == 0 {
        return;
    }

    let s = scale as i32;
    let char_w = 3 * s;
    let char_h = 5 * s;
    let spacing = s; // spacing between chars
    let padding = s; // padding inside bg
    let total_w = n * char_w + (n - 1) * spacing;
    let total_h = char_h;

    // Try to position above the provided y; if not enough space, put below
    let mut label_x = x;
    let mut label_y = y - (total_h + 2 * padding);

    let img_w = img.width() as i32;
    let img_h = img.height() as i32;

    if label_x + total_w + 2 * padding > img_w {
        label_x = img_w - (total_w + 2 * padding);
    }
    if label_x < 0 {
        label_x = 0;
    }

    if label_y < 0 {
        label_y = y + padding;
        if label_y + total_h + 2 * padding > img_h {
            label_y = img_h - (total_h + 2 * padding);
        }
    }

    // Draw background
    draw_filled_rect(
        img,
        label_x,
        label_y,
        total_w + 2 * padding,
        total_h + 2 * padding,
        bg,
    );

    // Draw each glyph
    let mut cx = label_x + padding;
    for ch in chars.iter() {
        if let Some(g) = glyph(*ch) {
            for row in 0..5 {
                for col in 0..3 {
                    let mask = 1 << (2 - col);
                    if (g[row] & mask) != 0 {
                        let px = cx + (col as i32) * s;
                        let py = label_y + padding + (row as i32) * s;
                        // draw scaled pixel block
                        draw_filled_rect(img, px, py, s, s, fg);
                    }
                }
            }
        }
        cx += char_w + spacing;
    }
}

fn draw_text_ttf(
    img: &mut RgbaImage,
    font: &Font,
    text: &str,
    x: i32,
    y: i32,
    size_px: f32,
    color: Rgba<u8>,
) {
    let img_w = img.width() as i32;
    let img_h = img.height() as i32;
    let line_metrics = font.horizontal_line_metrics(size_px).unwrap_or_else(|| {
        fontdue::LineMetrics {
            ascent: size_px,
            descent: 0.0,
            line_gap: 0.0,
            new_line_size: size_px,
        }
    });
    // Baseline is at y + ascent
    let mut cursor_x = x as f32;
    let baseline_y = y as f32 + line_metrics.ascent;

    for ch in text.chars() {
        let (metrics, bitmap) = font.rasterize(ch, size_px);
        if metrics.width == 0 || metrics.height == 0 {
            cursor_x += metrics.advance_width;
            continue;
        }
        // metrics.xmin/ymin are offsets from the cursor position
        let origin_x = cursor_x + metrics.xmin as f32;
        let origin_y = baseline_y + metrics.ymin as f32;

        for row in 0..metrics.height {
            for col in 0..metrics.width {
                let cov = bitmap[row * metrics.width + col];
                if cov == 0 {
                    continue;
                }
                let px = origin_x.round() as i32 + col as i32;
                let py = origin_y.round() as i32 + row as i32;
                if px < 0 || px >= img_w || py < 0 || py >= img_h {
                    continue;
                }
                let alpha = cov as i32;
                let existing = img.get_pixel(px as u32, py as u32);
                let mut out = [0u8; 4];
                for c in 0..3 {
                    let fg = color[c] as i32;
                    let bgc = existing[c] as i32;
                    out[c] = ((fg * alpha + bgc * (255 - alpha)) / 255) as u8;
                }
                out[3] = 255;
                img.put_pixel(px as u32, py as u32, Rgba(out));
            }
        }
        cursor_x += metrics.advance_width;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine() -> OcrEngine {
        let dir = format!("{}/assets", env!("CARGO_MANIFEST_DIR"));
        OcrEngine::new(&dir, RecognitionMode::Both, 4).expect("engine loads")
    }

    /// Detection parity smoke on the mobile synth set: the ncnn det model +
    /// PC post-processing must find each line's box (IoU against the mobile
    /// truth boxes), and every returned box must stay in image bounds.
    #[test]
    fn synth_det_finds_truth_line() {
        let mut eng = test_engine();
        let truth: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!(
                "{}/test_images/synth/truth.json",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
        )
        .unwrap();
        let mut failures: Vec<String> = Vec::new();
        let mut worst = (1.0f32, String::new());
        let mut lines = 0usize;
        for line in truth["lines"].as_array().unwrap() {
            let file = line["file"].as_str().unwrap();
            let boxes = line["boxes"].as_array().unwrap();
            let (mut left, mut top, mut right, mut bottom) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
            for b in boxes {
                let b = b.as_array().unwrap();
                let (x, y, w, h) = (
                    b[0].as_i64().unwrap(),
                    b[1].as_i64().unwrap(),
                    b[2].as_i64().unwrap(),
                    b[3].as_i64().unwrap(),
                );
                left = left.min(x);
                top = top.min(y);
                right = right.max(x + w);
                bottom = bottom.max(y + h);
            }
            let img =
                image::open(format!("{}/test_images/synth/{file}", env!("CARGO_MANIFEST_DIR")))
                    .unwrap();
            let (iw, ih) = img.dimensions();
            let det = eng.detect(&img).unwrap();
            lines += 1;
            for b in &det.boxes {
                if b.x < 0 || b.y < 0 || b.x + b.w > iw as i32 || b.y + b.h > ih as i32 {
                    failures.push(format!("{file}: box out of bounds {:?}", (b.x, b.y, b.w, b.h)));
                }
            }
            let best = det.boxes.iter().max_by_key(|b| (b.w as i64) * (b.h as i64));
            let Some(b) = best else {
                failures.push(format!("{file}: no boxes detected"));
                continue;
            };
            let ix1 = left.max(b.x as i64);
            let iy1 = top.max(b.y as i64);
            let ix2 = right.min(b.x as i64 + b.w as i64);
            let iy2 = bottom.min(b.y as i64 + b.h as i64);
            let inter = ((ix2 - ix1).max(0) * (iy2 - iy1).max(0)) as f32;
            let truth_area = ((right - left) * (bottom - top)) as f32;
            let box_area = (b.w * b.h) as f32;
            let iou = inter / (truth_area + box_area - inter);
            if iou < 0.5 {
                failures.push(format!(
                    "{file}: best box IoU {iou:.2} det=({},{},{},{}) truth=({},{},{},{})",
                    b.x, b.y, b.w, b.h, left, top, right, bottom
                ));
            }
            if iou < worst.0 {
                worst = (iou, file.to_string());
            }
        }
        assert_eq!(lines, 16, "expected the full synth set");
        assert!(
            failures.is_empty(),
            "det failures:\n{}\nworst: {} {:.2}",
            failures.join("\n"),
            worst.1,
            worst.0
        );
    }
    /// Mobile #28's filter: small ruby beside/above a large line is dropped,
    /// but a stacked column fragment (center inside the big box's x-range)
    /// and a merely-short real line survive.
    #[test]
    fn furigana_filter_drops_ruby_keeps_stacked_fragments() {
        // Vertical: ruby to the right of a big column.
        let big_raw = BoundingBox::new(100, 100, 50, 300, 1.0);
        let big_un = BoundingBox::new(90, 90, 70, 320, 1.0);
        let ruby_raw = BoundingBox::new(170, 150, 30, 80, 1.0);
        let ruby_un = BoundingBox::new(165, 145, 40, 90, 1.0);
        // Stacked fragment: same x-range as the big column -> never ruby.
        let stack_raw = BoundingBox::new(110, 150, 30, 80, 1.0);
        let stack_un = BoundingBox::new(100, 145, 50, 90, 1.0);
        let raw = vec![big_raw, ruby_raw, stack_raw];
        let un = vec![big_un, ruby_un, stack_un];
        assert_eq!(
            filter_furigana(&raw, &un, 1000, 1000),
            vec![true, false, true],
            "vertical ruby must be dropped, stacked fragment kept"
        );

        // Horizontal: thin ruby above a big line is dropped; a taller short
        // line above it is not.
        let hbig_raw = BoundingBox::new(100, 100, 400, 60, 1.0);
        let hbig_un = BoundingBox::new(90, 90, 420, 80, 1.0);
        let hruby_raw = BoundingBox::new(150, 60, 200, 20, 1.0);
        let hruby_un = BoundingBox::new(140, 55, 220, 25, 1.0);
        let tall_raw = BoundingBox::new(600, 60, 200, 50, 1.0);
        let tall_un = BoundingBox::new(590, 55, 220, 55, 1.0);
        let raw = vec![hbig_raw, hruby_raw, tall_raw];
        let un = vec![hbig_un, hruby_un, tall_un];
        assert_eq!(
            filter_furigana(&raw, &un, 1000, 1000),
            vec![true, false, true],
            "horizontal ruby must be dropped, tall short line kept"
        );
    }

    /// Mobile #48's trim: a column that swallowed a ruby strip (clean gutter
    /// between main and ruby ink) is cut at the gutter start, and the rotated
    /// rect stays in sync with the AABB.
    #[test]
    fn ruby_gutter_trim_cuts_the_ruby_side() {
        let mut img = image::RgbaImage::from_pixel(400, 300, Rgba([255, 255, 255, 255]));
        for y in 10..230 {
            for x in 30..56 {
                img.put_pixel(x, y, Rgba([0, 0, 0, 255])); // main column ink
            }
            for x in 66..90 {
                img.put_pixel(x, y, Rgba([0, 0, 0, 255])); // ruby ink
            }
            for x in 200..260 {
                img.put_pixel(x, y, Rgba([0, 0, 0, 255])); // normal box ink
            }
            for x in 300..360 {
                img.put_pixel(x, y, Rgba([0, 0, 0, 255])); // normal box ink
            }
        }
        let image = DynamicImage::ImageRgba8(img);
        // Three vertical boxes so the median width is the normal 60, and one
        // wide (120) swallowed the ruby.
        let mut pairs = vec![
            (
                BoundingBox::new(0, 0, 120, 240, 1.0),
                RotatedBox::new(60.0, 120.0, 240.0, 120.0, std::f32::consts::FRAC_PI_2, 1.0),
            ),
            (
                BoundingBox::new(200, 0, 60, 240, 1.0),
                RotatedBox::new(230.0, 120.0, 240.0, 60.0, std::f32::consts::FRAC_PI_2, 1.0),
            ),
            (
                BoundingBox::new(300, 0, 60, 240, 1.0),
                RotatedBox::new(330.0, 120.0, 240.0, 60.0, std::f32::consts::FRAC_PI_2, 1.0),
            ),
        ];
        assert_eq!(trim_ruby_gutter_vertical(&mut pairs, &image), 1);
        // Gutter starts at x=56; the quad is a +90° vertical (h = width).
        assert_eq!(pairs[0].0.w, 56);
        assert!((pairs[0].1.h - 56.0).abs() < 0.01, "quad h {}", pairs[0].1.h);
        assert!((pairs[0].1.cx - 28.0).abs() < 0.01, "quad cx {}", pairs[0].1.cx);
        assert_eq!(pairs[1].0.w, 60);
        assert_eq!(pairs[2].0.w, 60);
    }
    /// The new char-cell geometry: 1.1x max length for JP, 0.55x for
    /// halfwidth, and overlaps resolve to a common aspect ratio (or just
    /// touching edges when the equal-aspect point lies outside the overlap).
    #[test]
    fn char_cell_geometry_and_aspect_split() {
        assert_eq!(char_cell_width(Some('あ'), 50.0), 50.0);
        assert_eq!(char_cell_width(Some('漢'), 50.0), 50.0);
        assert_eq!(char_cell_width(Some('A'), 50.0), 25.0);
        assert_eq!(char_cell_width(Some('ｶ'), 50.0), 25.0);
        assert_eq!(char_cell_width(None, 50.0), 50.0);
        assert!((char_cell_max_len(Some('あ'), 50.0) - 55.0).abs() < 1e-3);
        assert!((char_cell_max_len(Some('A'), 50.0) - 27.5).abs() < 1e-3);

        // Equal widths: the shared boundary lands midway.
        let (mut a, mut b) = ((0.0f32, 110.0f32), (30.0f32, 140.0f32));
        split_overlap_same_aspect(&mut a, &mut b, 100.0, 100.0);
        assert!((a.1 - 70.0).abs() < 1e-3 && (b.0 - 70.0).abs() < 1e-3);

        // Different widths: both boxes end at the same aspect ratio.
        let (mut a, mut b) = ((0.0f32, 110.0f32), (50.0f32, 160.0f32));
        split_overlap_same_aspect(&mut a, &mut b, 50.0, 100.0);
        let (la, lb) = ((a.1 - a.0) / 50.0, (b.1 - b.0) / 100.0);
        assert!((la - lb).abs() < 1e-3, "aspects {la} {lb}");

        // Equal-aspect point beyond the overlap: clamp, edges just touch.
        let (mut a, mut b) = ((0.0f32, 110.0f32), (60.0f32, 170.0f32));
        split_overlap_same_aspect(&mut a, &mut b, 100.0, 50.0);
        assert!((a.1 - 110.0).abs() < 1e-3 && (b.0 - 110.0).abs() < 1e-3);

        // Gapped pair is untouched.
        let (mut a, mut b) = ((0.0f32, 10.0f32), (30.0f32, 40.0f32));
        split_overlap_same_aspect(&mut a, &mut b, 100.0, 100.0);
        assert_eq!((a, b), ((0.0, 10.0), (30.0, 40.0)));
    }

    /// End-to-end: every char box the streaming path emits must respect the
    /// per-class caps (1.1x line cross for JP, 0.55x for halfwidth), the
    /// neighbours must not overlap beyond rounding, and plain-glyph lines
    /// must land on the synth truth centres along the reading axis.
    #[test]
    fn synth_char_boxes_aspect_and_position() {
        let mut eng = test_engine();
        let truth: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!(
                "{}/test_images/synth/truth.json",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
        )
        .unwrap();
        let mut checked = 0usize;
        let mut position_checked = 0usize;
        for line in truth["lines"].as_array().unwrap() {
            let file = line["file"].as_str().unwrap();
            let img =
                image::open(format!("{}/test_images/synth/{file}", env!("CARGO_MANIFEST_DIR")))
                    .unwrap();
            let det = eng.detect_lines(&img).unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            crate::ocr_engine::recognize_boxes_streaming(
                &img, &det.boxes, &det.rotated,
                eng.ppocr_rec.clone(), &eng.ppocr_vocab, &eng.rec_remap,
                4, RecognitionMode::Both, tx, std::path::Path::new("/tmp"),
            )
            .unwrap();
            let truth_boxes: Vec<(f64, f64)> = line["boxes"].as_array().unwrap().iter().map(|b| {
                let b = b.as_array().unwrap();
                let (x, y, w, h) = (b[0].as_f64().unwrap(), b[1].as_f64().unwrap(), b[2].as_f64().unwrap(), b[3].as_f64().unwrap());
                (x + w / 2.0, y + h / 2.0)
            }).collect();
            let truth_text = line["text"].as_str().unwrap();
            for (_i, ann) in rx.into_iter() {
                let Some(line) = ann.line else { continue };
                if ann.quad.is_some() {
                    continue; // rotated lines carry AABBs by design
                }
                // Position regression: on synth the truth boxes are the drawn
                // em boxes, so a correctly recognised line's char centres must
                // land on them (this is what caught the trailing-edge shift and
                // the padded-width timestep scale).
                // Position regression: on synth the truth boxes are the drawn
                // em boxes, so a plain-glyph line's char centres must land on
                // them along the reading axis (this is what caught the
                // trailing-edge half-box shift and the padded timestep scale).
                // Punctuation and bracket-led lines are excluded: their ink
                // sits off the em centre, and a leading whitespace glyph
                // shifts the whole detection crop.
                let is_punct = |ch: char| {
                    matches!(ch, '\u{3002}'|'\u{002E}'|'\u{FF0E}'|'\u{3001}'|'\u{002C}'|'\u{FF0C}'|')'|'\u{FF09}'|'\u{3017}'|'\u{300D}'|'\u{300F}'|'\u{3015}'|'\u{3011}'|'\u{3009}'|']'|'\u{FF3D}')
                        || matches!(ch, '('|'\u{FF08}'|'\u{300C}'|'\u{300E}'|'\u{3014}'|'\u{3010}'|'\u{300A}'|'\u{3008}'|'\u{3016}'|'['|'\u{FF3B}')
                };
                if line.text == truth_text
                    && line.char_boxes.len() == truth_boxes.len()
                    && !line.text.chars().any(is_punct)
                {
                    for (ci, b) in line.char_boxes.iter().enumerate() {
                        let (tcx, tcy) = truth_boxes[ci];
                        let cx = b.x as f64 + b.w as f64 / 2.0;
                        let cy = b.y as f64 + b.h as f64 / 2.0;
                        let err = if line.is_vertical { cy - tcy } else { cx - tcx };
                        assert!(
                            err.abs() <= 13.0,
                            "{file} char {ci} ({:?}): along error {err:.1}px",
                            line.text.chars().nth(ci)
                        );
                    }
                    position_checked += 1;
                }
                let chars: Vec<char> = line.text.chars().collect();
                let vertical = line.is_vertical;
                let mut prev_end: Option<i32> = None;
                for (ci, b) in line.char_boxes.iter().enumerate() {
                    let cross = if vertical { b.w } else { b.h } as f32;
                    let along = if vertical { b.h } else { b.w } as f32;
                    let max_len = char_cell_max_len(chars.get(ci).copied(), cross);
                    assert!(
                        along <= max_len * 1.02 + 2.0,
                        "{file} char {ci} ({:?}): along {along} > max {max_len}",
                        chars.get(ci)
                    );
                    if let Some(prev) = prev_end {
                        let start = if vertical { b.y } else { b.x };
                        assert!(
                            start >= prev - 2,
                            "{file} char {ci}: boxes overlap beyond rounding"
                        );
                    }
                    prev_end = Some(if vertical { b.y + b.h } else { b.x + b.w });
                    checked += 1;
                }
            }
        }
        assert!(checked >= 80, "expected the full synth set, checked {checked}");
        assert!(
            position_checked >= 6,
            "expected at least 6 plain-glyph lines for the position check, got {position_checked}"
        );
    }
}
