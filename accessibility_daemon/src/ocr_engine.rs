use anyhow::{Context, Result};
use image::{DynamicImage, Rgba, RgbaImage};
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
/// Mobile `xOverlapThresh` pref default: union two straight boxes when their
/// intersection covers at least this fraction of the smaller box.
const X_OVERLAP_THRESHOLD: f32 = 0.40;

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
// Absolute ceiling: real short columns dwarf ruby runs even when the ratio
// matches — ruby longer than 12% of the image side is not ruby.
const FURIGANA_MAX_FRAC: f32 = 0.12;
// Absolute floor on the annotated box: ruby hugs full-size body text, not
// compact blocks (logo boxes, badges).
const FURIGANA_BIG_MIN_FRAC: f32 = 0.2;


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
    /// Live detection tuning from the viewer (`Message::TuneDet`). `None`
    /// falls back to the `DET_THRESH` / `DET_UNCLIP` environment, then to the
    /// mobile defaults.
    pub det_thresh_override: Option<f32>,
    pub det_unclip_override: Option<f32>,
}

/// Effective `DET_THRESH`: environment override, else the mobile 0.3.
pub fn default_det_thresh() -> f32 {
    std::env::var("DET_THRESH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(PPOCR_DET_THRESH)
}

/// Effective `DET_UNCLIP`: environment override, else the mobile 1.5.
pub fn default_det_unclip() -> f32 {
    std::env::var("DET_UNCLIP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(PPOCR_DET_UNCLIP_RATIO)
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

/// Bilinear sample of `src` at `(sx, sy)`; None when the point lies more than
/// half a pixel outside (the caller keeps the background there), otherwise
/// clamped to the source edges like Skia's CLAMP tile mode.
fn sample_bilinear(src: &RgbaImage, sx: f32, sy: f32) -> Option<Rgba<u8>> {
    let (sw, sh) = src.dimensions();
    if sw == 0 || sh == 0 {
        return None;
    }
    let (sw_f, sh_f) = (sw as f32, sh as f32);
    if sx < -0.5 || sy < -0.5 || sx > sw_f - 0.5 || sy > sh_f - 0.5 {
        return None;
    }
    let x0 = sx.floor().clamp(0.0, sw_f - 1.0);
    let y0 = sy.floor().clamp(0.0, sh_f - 1.0);
    let x1 = (x0 + 1.0).min(sw_f - 1.0);
    let y1 = (y0 + 1.0).min(sh_f - 1.0);
    let fx = (sx - x0).clamp(0.0, 1.0);
    let fy = (sy - y0).clamp(0.0, 1.0);
    let p00 = src.get_pixel(x0 as u32, y0 as u32).0;
    let p10 = src.get_pixel(x1 as u32, y0 as u32).0;
    let p01 = src.get_pixel(x0 as u32, y1 as u32).0;
    let p11 = src.get_pixel(x1 as u32, y1 as u32).0;
    let mut out = [0u8; 4];
    for c in 0..4 {
        let v0 = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
        let v1 = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
        out[c] = (v0 * (1.0 - fy) + v1 * fy).round().clamp(0.0, 255.0) as u8;
    }
    Some(Rgba(out))
}

/// Bilinear-blit `src` into `dst` with its top-left at the exact float offset
/// `(left, top)` — mobile draws the resized det bitmap at the float centering
/// offset with filtering, so a `.5` offset blends the content edge with the
/// gray letterbox pad.
fn blit_filtered(dst: &mut RgbaImage, src: &RgbaImage, left: f32, top: f32) {
    let (dw, dh) = dst.dimensions();
    let (sw, sh) = src.dimensions();
    if sw == 0 || sh == 0 {
        return;
    }
    let x_start = left.floor().max(0.0) as u32;
    let y_start = top.floor().max(0.0) as u32;
    let x_end = ((left + sw as f32).ceil() as u32).min(dw);
    let y_end = ((top + sh as f32).ceil() as u32).min(dh);
    for y in y_start..y_end {
        for x in x_start..x_end {
            // Source coordinate at this destination pixel's centre.
            let sx = (x as f32 + 0.5) - left - 0.5;
            let sy = (y as f32 + 0.5) - top - 0.5;
            if let Some(px) = sample_bilinear(src, sx, sy) {
                dst.put_pixel(x, y, px);
            }
        }
    }
}

/// Mobile `warpRotatedCrop` (#53): draw the source through the frame's
/// inverse into an upright `localW × localH` bitmap. `setPolyToPoly` maps the
/// frame's four corners to the upright rect, so the inverse samples
/// `c0 + (dx/w)·localWidth·xAxis + (dy/h)·localHeight·yAxis`; bilinear, like
/// the mobile FILTER_BITMAP_FLAG paint. Pixels outside the source stay
/// transparent.
fn warp_quad_crop(image: &DynamicImage, quad: &RotatedBox) -> Option<DynamicImage> {
    let w = quad.w.round().max(4.0) as u32;
    let h = quad.h.round().max(4.0) as u32;
    let (xa, ya) = (quad.x_axis(), quad.y_axis());
    let (ox, oy) = quad.origin();
    let src = image.to_rgba8();
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 + 0.5;
            let dy = y as f32 + 0.5;
            let lx = dx / w as f32 * quad.w;
            let ly = dy / h as f32 * quad.h;
            let sx = ox + lx * xa.0 + ly * ya.0;
            let sy = oy + lx * xa.1 + ly * ya.1;
            if let Some(px) = sample_bilinear(&src, sx, sy) {
                out.put_pixel(x, y, px);
            }
        }
    }
    Some(DynamicImage::ImageRgba8(out))
}

/// A frame's enclosing AABB as the shared `BoundingBox` (rounded like every
/// other box in the pipeline).
fn rect_of(q: &RotatedBox) -> BoundingBox {
    let (x, y, w, h) = q.aabb();
    BoundingBox::new(
        x.round() as i32,
        y.round() as i32,
        w.round() as i32,
        h.round() as i32,
        q.confidence,
    )
}

/// Mobile `detectRotated`'s component walk: 8-connected flood fill over
/// `prob > thresh`, the component's boundary pixel corners in source space,
/// `fitQuad`, then DB unclip on the frame's own axes. Returns index-aligned
/// `(pre-unclip, unclipped)` frames. `expand_cap` is a PC-only cap in source
/// pixels (default infinity = Android's uncapped unclip).
#[allow(clippy::too_many_arguments)]
fn fit_components(
    prob_map: &[f32],
    out_w: u32,
    out_h: u32,
    det_thresh: f32,
    det_unclip: f32,
    expand_cap: f32,
    img_left: f32,
    img_top: f32,
    scale_w: f32,
    scale_h: f32,
) -> (Vec<RotatedBox>, Vec<RotatedBox>) {
    let ow = out_w as usize;
    let oh = out_h as usize;
    let mut visited = vec![0u8; ow * oh];
    let mut queue: Vec<u32> = Vec::with_capacity(ow * oh);
    let mut points: Vec<(f32, f32)> = Vec::new();
    let mut pre_quads: Vec<RotatedBox> = Vec::new();
    let mut quads: Vec<RotatedBox> = Vec::new();
    for y in 0..oh {
        for x in 0..ow {
            let idx = y * ow + x;
            if visited[idx] != 0 || prob_map.get(idx).copied().unwrap_or(0.0) <= det_thresh {
                continue;
            }
            // One 8-connected component (mobile floodFillComponent).
            queue.clear();
            queue.push(idx as u32);
            visited[idx] = 1;
            let mut head = 0usize;
            while head < queue.len() {
                let cur = queue[head] as usize;
                head += 1;
                let cx = cur % ow;
                let cy = cur / ow;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (nx, ny) = (cx as i32 + dx, cy as i32 + dy);
                        if nx < 0 || ny < 0 || nx >= out_w as i32 || ny >= out_h as i32 {
                            continue;
                        }
                        let n_idx = ny as usize * ow + nx as usize;
                        if visited[n_idx] == 0
                            && prob_map.get(n_idx).copied().unwrap_or(0.0) > det_thresh
                        {
                            visited[n_idx] = 1;
                            queue.push(n_idx as u32);
                        }
                    }
                }
            }
            if queue.len() < 3 {
                continue; // mobile noise filter counts ALL component pixels
            }

            // Boundary pixels: any 8-neighbour outside the mask. Their
            // outer corners in source space enclose exactly the pixels the
            // axis-aligned min/max box covers.
            points.clear();
            for &cur in &queue {
                let cur = cur as usize;
                let cx = cur % ow;
                let cy = cur / ow;
                let mut boundary = false;
                'neighbours: for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let (nx, ny) = (cx as i32 + dx, cy as i32 + dy);
                        if nx < 0
                            || ny < 0
                            || nx >= out_w as i32
                            || ny >= out_h as i32
                            || prob_map
                                .get(ny as usize * ow + nx as usize)
                                .copied()
                                .unwrap_or(0.0)
                                <= det_thresh
                        {
                            boundary = true;
                            break 'neighbours;
                        }
                    }
                }
                if !boundary {
                    continue;
                }
                let mx = (cx as f32 - img_left) * scale_w;
                let my = (cy as f32 - img_top) * scale_h;
                points.push((mx - 0.5 * scale_w, my - 0.5 * scale_h));
                points.push((mx + 0.5 * scale_w, my - 0.5 * scale_h));
                points.push((mx + 0.5 * scale_w, my + 0.5 * scale_h));
                points.push((mx - 0.5 * scale_w, my + 0.5 * scale_h));
            }

            // Mobile RotatedGeometry.fitQuad on the source-space corners.
            let Some(pre_quad) = fit_quad(&points) else {
                continue;
            };
            // DB unclip on the frame's own axes (mobile unclip); the cap is
            // a PC-only knob, off by default.
            let quad = if expand_cap.is_finite() {
                let (pw, ph) = (pre_quad.w, pre_quad.h);
                let expand = if pw > 1e-6 && ph > 1e-6 {
                    pw * ph * det_unclip / (2.0 * (pw + ph))
                } else {
                    0.0
                };
                let expand = expand.min(expand_cap);
                pre_quad.inset(-expand, -expand)
            } else {
                pre_quad.unclip(det_unclip)
            };
            if quad.w < 4.0 || quad.h < 4.0 {
                continue;
            }
            pre_quads.push(pre_quad);
            quads.push(quad);
        }
    }
    (pre_quads, quads)
}

/// Mobile `detectRotated`'s post-fit stages: furigana rejection on AABB
/// geometry (same rule and raw-vs-unclipped pair as the axis path), the
/// local-size filter, the enclosing-blob filter (#53) and the vertical 5%
/// cross-axis inset. Returns the surviving frames.
fn filter_fitted_quads(
    pre_quads: &[RotatedBox],
    quads: &[RotatedBox],
    orig_w: i32,
    orig_h: i32,
) -> Vec<RotatedBox> {
    let pre_rects: Vec<BoundingBox> = pre_quads.iter().map(rect_of).collect();
    let uncl_rects: Vec<BoundingBox> = quads.iter().map(rect_of).collect();
    let keep = filter_furigana(&pre_rects, &uncl_rects, orig_w, orig_h);

    // Min-size filter uses the frame's own local sizes (mobile).
    let mut min_sized: Vec<RotatedBox> = Vec::new();
    for (i, q) in quads.iter().enumerate() {
        if !keep.get(i).copied().unwrap_or(true) {
            eprintln!(
                "[PP-OCR DET] furigana dropped {}x{}@({}, {})",
                pre_rects[i].w, pre_rects[i].h, pre_rects[i].x, pre_rects[i].y
            );
            continue;
        }
        if q.w < 10.0 || q.h < 10.0 {
            continue;
        }
        min_sized.push(*q);
    }
    // A DB blob that merged several Lines fits one frame covering them all
    // while the Lines inside arrive as their own smaller fits; the blob is
    // the false positive, so drop it before it can become a Line.
    let kept = filter_enclosing_blobs(&min_sized);
    let blobs = min_sized.len() - kept.len();
    if blobs > 0 {
        eprintln!("[PP-OCR DET] {blobs} enclosing blob(s) filtered");
    }
    kept.iter()
        .map(|q| {
            if q.is_vertical() {
                q.inset(q.w * 0.05, 0.0)
            } else {
                *q
            }
        })
        .collect()
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
        // Non-rotated vertical: the cross axis is the frame's local x; keep
        // the quad in sync so crops/char boxes follow the trim.
        r.w -= delta;
        r.cx -= delta / 2.0;
        trimmed += 1;
    }
    trimmed
}

/// Mobile `BOX_LAYOUT_MODE` (0 = legacy uniform columns, 1 = legacy + ink
/// snapping; default 1). `BOX_LAYOUT_MODE=0` disables snapping.
fn box_layout_snap() -> bool {
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
fn legacy_cells(cols: &[f32], seq_len: usize, l: f32, cross: f32) -> Vec<(f32, f32)> {
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
            env!("CARGO_MANIFEST_DIR"),
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
/// preserved; only centres move.
fn resolve_ink_collisions(cells: &mut [(f32, f32)], chars: &[char], render_text_size: f32) {
    let n = cells.len();
    if n < 2 {
        return;
    }
    let ink_half = ink_half_widths(chars, render_text_size.max(1.0));
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
fn uniform_cells(cells: &[(f32, f32)], chars: &[char], l: f32) -> Vec<(f32, f32)> {
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
        let cells = match (box_layout_snap(), pixels) {
            (true, Some((px, pw, ph))) => snap_cells(&base, &chars, px, pw, ph, false, l),
            _ => base,
        };
        let mut resolved = cells;
        resolve_ink_collisions(&mut resolved, &chars, crop_h as f32 * 0.90);
        let sized = if box_uniform_size() {
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
        let cells = match (box_layout_snap(), pixels) {
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
        let mut sized = if box_uniform_size() {
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

#[allow(dead_code)] // no live caller yet: mobile's consumer is the gap-detector fallback
impl DetectedAnnotation {
    /// Mobile `OcrEngine.reDecodeLineResult`: rebuild a line's text, char
    /// boxes and per-character alternatives from its cached
    /// [`LineResult::raw_alternatives`], without re-running the model. The
    /// walk (blank handling, CTC collapse, fractional columns, vertical
    /// punctuation) lives in [`crate::ppocr::re_decode_raw_alternatives`].
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
            crate::ppocr::re_decode_raw_alternatives(&line.raw_alternatives, line.is_vertical)
        else {
            return self.clone();
        };

        let char_boxes = if seq_len_total > 0 {
            match self.quad.filter(|r| r.is_rotated()) {
                Some(r) => {
                    let lw = r.w.round().max(4.0) as u32;
                    let lh = r.h.round().max(4.0) as u32;
                    compute_char_boxes(
                        &re.text, &re.char_cols, seq_len_total,
                        0, 0, lw, lh, line.is_vertical, None,
                    )
                    .iter()
                    .map(|b| r.map_local_rect(b.x as f32, b.y as f32, b.w as f32, b.h as f32))
                    .collect()
                }
                None => compute_char_boxes(
                    &re.text, &re.char_cols, seq_len_total,
                    self.bbox.x, self.bbox.y,
                    self.bbox.w.max(0) as u32, self.bbox.h.max(0) as u32,
                    line.is_vertical, None,
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
            }),
        }
    }
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
            det_thresh_override: None,
            det_unclip_override: None,
        })
    }

    /// Effective detection threshold: viewer override → `DET_THRESH` env →
    /// mobile default. The viewer's tuning keys set the override live.
    pub fn det_thresh(&self) -> f32 {
        self.det_thresh_override.unwrap_or_else(default_det_thresh)
    }

    /// Effective DB unclip ratio: viewer override → `DET_UNCLIP` env →
    /// mobile default.
    pub fn det_unclip(&self) -> f32 {
        self.det_unclip_override.unwrap_or_else(default_det_unclip)
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
        // Mobile draws the resized bitmap at the exact float centering offset
        // (`(modelSize - resizeW) / 2f`, e.g. 261.5) with a filtered blit, so
        // a half-pixel offset shifts/blends the content edge with the gray
        // pad. Keep the same float offset and bilinear-blit instead of the
        // old integer `replace`.
        let img_left = (model_size - resize_w) as f32 / 2.0;
        let img_top = (model_size - resize_h) as f32 / 2.0;

        let resized = image.resize_exact(resize_w, resize_h, image::imageops::FilterType::Triangle);
        let mut letterbox =
            RgbaImage::from_pixel(model_size, model_size, Rgba([128u8, 128u8, 128u8, 255u8]));
        blit_filtered(&mut letterbox, &resized.to_rgba8(), img_left, img_top);

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
        // letterbox space. Any other shape is treated as flat (mobile's
        // fallback: derive side = sqrt(total) and carry on with a warning)
        // rather than hard-failing the whole detection.
        let (prob_map, out_w, out_h): (Vec<f32>, u32, u32) = if raw_map.len() == h * w {
            (raw_map, model_size, model_size)
        } else {
            let dim = (raw_map.len() as f64).sqrt() as u32;
            if dim > 0 && (dim * dim) as usize == raw_map.len() && dim <= model_size {
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
                (up, model_size, model_size)
            } else {
                let side = (raw_map.len() as f64).sqrt() as u32;
                anyhow::ensure!(side > 0, "unexpected det output size {}", raw_map.len());
                eprintln!(
                    "[PP-OCR DET] unexpected prob size {}, using {side}x{side}",
                    raw_map.len()
                );
                (raw_map, side, side)
            }
        };
        // Debug: print prob_map statistics to understand value range
        if !prob_map.is_empty() {
            let max_val = prob_map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_val = prob_map.iter().cloned().fold(f32::INFINITY, f32::min);
            let sum: f32 = prob_map.iter().sum();
            let mean = sum / prob_map.len() as f32;
            eprintln!("[PP-OCR DET] prob_map: min={min_val:.4} max={max_val:.4} mean={mean:.4}");
        }

        // Scale factors from letterbox space to the original image; content
        // sits at (img_left, img_top) after centering.
        let scale_w = orig_w / resize_w as f32;
        let scale_h = orig_h / resize_h as f32;

        // 5. Threshold → flood-fill components → min-area fits (mobile
        // detectRotated, the default path): an 8-connected fill over
        // `prob > thresh`, the component's boundary pixel corners in source
        // space, then RotatedGeometry.fitQuad. The viewer's tuning keys write
        // the overrides; DET_THRESH / DET_UNCLIP env vars still work when no
        // override is set.
        let det_thresh = self.det_thresh();
        let det_unclip = self.det_unclip();
        // PC-only expansion cap in source pixels; the default is Android's
        // uncapped DB unclip.
        let expand_cap: f32 = std::env::var("DET_EXPAND_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(f32::INFINITY);

        let (pre_quads, quads) = fit_components(
            &prob_map,
            out_w,
            out_h,
            det_thresh,
            det_unclip,
            expand_cap,
            img_left,
            img_top,
            scale_w,
            scale_h,
        );

        // 6-9. Mobile detectRotated's post-fit stages: furigana rejection on
        // AABB geometry, the local-size filter, the enclosing-blob filter
        // (#53) and the vertical 5% cross-axis inset.
        let kept = filter_fitted_quads(&pre_quads, &quads, orig_w as i32, orig_h as i32);

        // The PC-only orientation knobs default to Android behaviour (off).
        let h_down: f32 = std::env::var("DET_H_DOWN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let v_trim: f32 = std::env::var("DET_V_SHRINK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let mut pp_pairs: Vec<(BoundingBox, RotatedBox)> = Vec::with_capacity(kept.len());
        for quad_in in &kept {
            let mut quad = *quad_in;
            if h_down != 0.0 && !quad.is_vertical() && quad.h >= 24.0 {
                quad.h += h_down;
                quad.cy += h_down / 2.0;
            }
            if v_trim != 1.0 && quad.is_vertical() {
                quad.h *= v_trim;
            }
            let (bx, by, bw, bh) = quad.aabb();
            pp_pairs.push((
                BoundingBox::new(
                    bx.round() as i32,
                    by.round() as i32,
                    bw.round() as i32,
                    bh.round() as i32,
                    PPOCR_DET_BOX_THRESH,
                ),
                quad,
            ));
        }

        // Debug: print the fitted boxes (after the filters)
        eprintln!("[PP-OCR DET] {} boxes:", pp_pairs.len());
        for (i, (b, r)) in pp_pairs.iter().enumerate() {
            eprintln!(
                "  [{i}] x={} y={} w={} h={} angle={:.1}° c={:.3}",
                b.x, b.y, b.w, b.h, r.angle.to_degrees(), b.confidence
            );
        }

        // PC-only ruby-gutter trim (#48) behind RUBY_TRIM_VERTICAL: Android's
        // default rotated path does not trim, so the default is off (D7.2).
        if std::env::var("RUBY_TRIM_VERTICAL").map(|v| v != "0").unwrap_or(false) {
            let trimmed = trim_ruby_gutter_vertical(&mut pp_pairs, image);
            if trimmed > 0 {
                eprintln!("[PP-OCR DET] rubyTrim applied to {trimmed} box(es)");
            }
        }
        // PC-only stacked-overlap split (DET_SPLIT_OVERLAP, off by default):
        // Android's default rotated path does not split.
        if std::env::var("DET_SPLIT_OVERLAP").is_ok() {
            let h_indices: Vec<usize> = pp_pairs
                .iter()
                .enumerate()
                .filter(|(_, (_, r))| !r.is_vertical())
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

        // 10. Sort: horizontal top-bottom/left-right, vertical right-left/
        // top-bottom (the frame's own sizes decide the group).
        let pp_pairs = self.sort_detected_boxes(pp_pairs);
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
                let binary: Vec<u8> = prob_map
                    .iter()
                    .map(|&v| if v > det_thresh { 255 } else { 0 })
                    .collect();
                if let Some(img) = image::GrayImage::from_raw(out_w, out_h, binary) {
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



    pub fn sort_detected_boxes(&self, mut boxes: Vec<(BoundingBox, RotatedBox)>) -> Vec<(BoundingBox, RotatedBox)> {
        // Separate by orientation — the frame's own sizes decide, like mobile
        // isVerticalLineBox (near-square counts as horizontal).
        let mut horizontal: Vec<(BoundingBox, RotatedBox)> = Vec::new();
        let mut vertical: Vec<(BoundingBox, RotatedBox)> = Vec::new();
        for pair in boxes.drain(..) {
            if pair.1.is_vertical() {
                vertical.push(pair);
            } else {
                horizontal.push(pair);
            }
        }

        // Horizontal: top-to-bottom, left-to-right
        horizontal.sort_by(|a, b| a.0.y.cmp(&b.0.y).then(a.0.x.cmp(&b.0.x)));

        // Vertical: right-to-left, top-to-bottom (mobile sorts on the AABB's
        // RIGHT edge, not the left).
        vertical.sort_by(|a, b| {
            (b.0.x + b.0.w)
                .cmp(&(a.0.x + a.0.w))
                .then(a.0.y.cmp(&b.0.y))
        });

        // Concatenate: horizontal lines first, then vertical
        boxes = horizontal;
        boxes.extend(vertical);
        boxes
    }




    // Modified: return the annotated image in-memory when `render` is true
    /// Just line detection — returns raw bounding boxes (fast, no character recognition).
    /// Used by the streaming OCR pipeline to show the image immediately.
    pub fn detect_lines(&mut self, image: &DynamicImage) -> Result<DetectionResult> {
        let det = self.detect(image)?;
        let pairs: Vec<(BoundingBox, RotatedBox)> = det.boxes.into_iter().zip(det.rotated).collect();
        // Mobile's axis-aligned path merges overlapping boxes
        // (`mergeOverlappingBoxes`); the rotated path deliberately does not.
        // Apply the same rule to the straight boxes so a long line whose DB
        // blob breaks into overlapping components comes back as one Line.
        let merged = merge_straight_boxes(pairs);
        let sorted = self.sort_detected_boxes(merged);
        let (boxes, rotated): (Vec<_>, Vec<_>) = sorted.into_iter().unzip();
        Ok(DetectionResult { boxes, rotated })
    }
}

/// Mobile `OcrEngine.shouldMerge`: the intersection must cover at least
/// `X_OVERLAP_THRESHOLD` of the smaller box and the centres must sit within
/// one average height of each other.
fn should_merge_straight(a: &BoundingBox, b: &BoundingBox) -> bool {
    let (ax1, ay1, ax2, ay2) = (a.x, a.y, a.x + a.w, a.y + a.h);
    let (bx1, by1, bx2, by2) = (b.x, b.y, b.x + b.w, b.y + b.h);
    let (ix1, iy1) = (ax1.max(bx1), ay1.max(by1));
    let (ix2, iy2) = (ax2.min(bx2), ay2.min(by2));
    if ix1 >= ix2 || iy1 >= iy2 {
        return false;
    }
    let inter = (ix2 - ix1) as f32 * (iy2 - iy1) as f32;
    let min_area = (a.w * a.h).min(b.w * b.h) as f32;
    if min_area <= 0.0 {
        return false;
    }
    if inter / min_area < X_OVERLAP_THRESHOLD {
        return false;
    }
    let y_diff = ((ay1 + ay2) as f32 / 2.0 - (by1 + by2) as f32 / 2.0).abs();
    let avg_h = (a.h + b.h) as f32 / 2.0;
    y_diff <= avg_h
}

/// Union of two AABBs, keeping the stronger confidence.
fn union_boxes(a: &BoundingBox, b: &BoundingBox) -> BoundingBox {
    let x1 = a.x.min(b.x);
    let y1 = a.y.min(b.y);
    let x2 = (a.x + a.w).max(b.x + b.w);
    let y2 = (a.y + a.h).max(b.y + b.h);
    BoundingBox::new(x1, y1, x2 - x1, y2 - y1, a.confidence.max(b.confidence))
}

/// Mobile `OcrEngine.mergeOverlappingBoxes`, restricted to straight frames
/// (no quad): the rotated path has no merge on mobile, so only boxes that
/// would have been axis-aligned there take part. A merged box becomes a
/// plain axis-aligned frame (angle 0).
fn merge_straight_boxes(pairs: Vec<(BoundingBox, RotatedBox)>) -> Vec<(BoundingBox, RotatedBox)> {
    if pairs.len() < 2 {
        return pairs;
    }
    let straight: Vec<bool> = pairs.iter().map(|(_, r)| !r.is_rotated()).collect();
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    // Largest box first, like mobile (`sortedByDescending { area }`).
    order.sort_by_key(|&i| std::cmp::Reverse(pairs[i].0.w as i64 * pairs[i].0.h as i64));
    let mut handled = vec![false; pairs.len()];
    let mut out = Vec::with_capacity(pairs.len());
    for &i in &order {
        if handled[i] {
            continue;
        }
        handled[i] = true;
        if !straight[i] {
            out.push(pairs[i].clone());
            continue;
        }
        let mut cur = pairs[i].0.clone();
        for &j in &order {
            if handled[j] || !straight[j] {
                continue;
            }
            if should_merge_straight(&cur, &pairs[j].0) {
                cur = union_boxes(&cur, &pairs[j].0);
                handled[j] = true;
            }
        }
        let frame = RotatedBox::new(
            cur.x as f32 + cur.w as f32 / 2.0,
            cur.y as f32 + cur.h as f32 / 2.0,
            cur.w as f32,
            cur.h as f32,
            0.0,
            cur.confidence,
        );
        out.push((cur, frame));
    }
    out
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
                // Angled line: mobile warpRotatedCrop — the frame's corners
                // map to an upright localW × localH crop, so the recogniser
                // sees the line's own tight frame instead of an AABB canvas.
                let Some(crop) = warp_quad_crop(image, &r) else {
                    continue;
                };
                (
                    crop,
                    0,
                    0,
                    r.w.round().max(4.0) as u32,
                    r.h.round().max(4.0) as u32,
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
                    is_vertical_box(bbox),
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
                        let mut text = res.text;
                        let mut alternatives = res.alternatives;
                        let char_cols = res.char_cols;
                        let seq_len_total = res.seq_len_total;
                        let mut raw_alternatives = res.raw_alternatives;
                        if text.is_empty() { continue; }

                        // Vertical punctuation normalisation at emit (#56/#63):
                        // ASCII `?` and horizontal `…`/`‥` become their vertical
                        // presentation forms in the text and the alternatives
                        // BEFORE char boxes, so the substitutes get full-em
                        // metrics (dictionary lookup folds them back).
                        if job.is_vertical {
                            text = crate::util::japanese::vertical_punctuation(&text);
                            for alts in alternatives.iter_mut() {
                                for (c, _) in alts.iter_mut() {
                                    *c = crate::util::japanese::vertical_punctuation_char(*c);
                                }
                            }
                            // Cached raw lists get the same treatment (mobile
                            // `recRaw`), so a later re-decode cannot reintroduce
                            // the horizontal forms.
                            for alts in raw_alternatives.iter_mut() {
                                for (c, _) in alts.iter_mut() {
                                    *c = crate::util::japanese::vertical_punctuation_char(*c);
                                }
                            }
                        }

                        // Dataset collection: save the line crop + detected text
                        let sample_txt = save_line_sample(&job.crop, &text);

                        // Char boxes: mobile computeCharBoxes (#49). Rotated
                        // lines compute in the upright local crop
                        // (0,0,localW,localH) and map back through the frame;
                        // axis lines use the UNCLAMPED rect frame (the crop
                        // itself stays clamped, D6).
                        let snap_pixels = if box_layout_snap()
                            && job.crop.width() >= 8
                            && job.crop.height() >= 8
                        {
                            let rgb = job.crop.to_rgb8();
                            Some((rgb.into_raw(), job.crop.width(), job.crop.height()))
                        } else {
                            None
                        };
                        let pixels = snap_pixels
                            .as_ref()
                            .map(|(p, w, h)| (p.as_slice(), *w, *h));
                        let char_boxes = if let Some(r) = job.rot.filter(|r| r.is_rotated()) {
                            let lw = r.w.round().max(4.0) as u32;
                            let lh = r.h.round().max(4.0) as u32;
                            let local = compute_char_boxes(
                                &text, &char_cols, seq_len_total,
                                0, 0, lw, lh, job.is_vertical, pixels,
                            );
                            local
                                .iter()
                                .map(|b| {
                                    r.map_local_rect(
                                        b.x as f32, b.y as f32, b.w as f32, b.h as f32,
                                    )
                                })
                                .collect()
                        } else {
                            compute_char_boxes(
                                &text, &char_cols, seq_len_total,
                                job.bbox.x, job.bbox.y,
                                job.bbox.w.max(0) as u32, job.bbox.h.max(0) as u32,
                                job.is_vertical, pixels,
                            )
                        };

                        let (final_text, final_alts) = (text, alternatives);

                        let annotation = DetectedAnnotation {
                            bbox: job.bbox.clone(),
                            quad: job.rot,
                            line: Some(LineResult {
                                text: final_text,
                                char_boxes,
                                alternatives: final_alts,
                                raw_alternatives,
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    fn test_engine() -> OcrEngine {
        let dir = format!("{}/assets", env!("CARGO_MANIFEST_DIR"));
        OcrEngine::new(&dir, RecognitionMode::Both, 4).expect("engine loads")
    }

    /// Live tuning overrides win over the environment defaults, and clearing
    /// them falls back to the default resolution path.
    #[test]
    fn det_tuning_overrides_beat_the_defaults() {
        let mut engine = test_engine();
        engine.det_thresh_override = Some(0.42);
        engine.det_unclip_override = Some(1.1);
        assert_eq!(engine.det_thresh(), 0.42);
        assert_eq!(engine.det_unclip(), 1.1);

        engine.det_thresh_override = None;
        engine.det_unclip_override = None;
        // No override: resolves via env-or-mobile-default (env unset in tests).
        assert_eq!(engine.det_thresh(), default_det_thresh());
        assert_eq!(engine.det_unclip(), default_det_unclip());
    }

    /// Prob map with `on` pixels set to 0.9 (above the 0.3 threshold).
    fn prob_map(w: usize, h: usize, on: impl Fn(usize, usize) -> bool) -> Vec<f32> {
        let mut map = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                if on(x, y) {
                    map[y * w + x] = 0.9;
                }
            }
        }
        map
    }

    fn components(map: &[f32], w: usize, h: usize, cap: f32) -> (Vec<RotatedBox>, Vec<RotatedBox>) {
        fit_components(map, w as u32, h as u32, 0.3, 1.5, cap, 0.0, 0.0, 1.0, 1.0)
    }

    /// D3.1: mobile flood-fills the component, so an interior hole is not a
    /// second box (imageproc's Suzuki-Abe contours returned hole borders too).
    #[test]
    fn fit_components_ignores_interior_holes() {
        // A 20x20 ring: border pixels on, interior off.
        let map = prob_map(40, 40, |x, y| {
            (5..25).contains(&x) && (5..25).contains(&y) && (x == 5 || x == 24 || y == 5 || y == 24)
        });
        let (pre, uncl) = components(&map, 40, 40, f32::INFINITY);
        assert_eq!(pre.len(), 1, "the hole must not become a box");
        let (x, y, w, h) = pre[0].aabb();
        assert!(
            (x - 4.5).abs() < 0.01 && (y - 4.5).abs() < 0.01,
            "frame origin {x},{y}"
        );
        assert!((w - 20.0).abs() < 0.01 && (h - 20.0).abs() < 0.01, "{w}x{h}");
        // Unclip grows both axes on the frame (no cap).
        assert!(uncl[0].w > pre[0].w && uncl[0].h > pre[0].h);
    }

    /// D3.1/D3.7: specks are counted over all component pixels (mobile), and
    /// anything whose frame is under 4px in either local axis is dropped, so
    /// a 2-pixel speck and a 3-pixel L both go; a real block is kept at its
    /// pixel extents.
    #[test]
    fn fit_components_drops_specks_and_sub_4px_frames() {
        let two = prob_map(10, 10, |x, y| (x, y) == (5, 5) || (x, y) == (6, 5));
        assert!(components(&two, 10, 10, f32::INFINITY).0.is_empty());
        let three = prob_map(10, 10, |x, y| {
            (x, y) == (5, 5) || (x, y) == (6, 5) || (x, y) == (5, 6)
        });
        assert!(
            components(&three, 10, 10, f32::INFINITY).0.is_empty(),
            "3-pixel L is under the 4px frame gate"
        );
        let block = prob_map(20, 20, |x, y| (5..11).contains(&x) && (5..11).contains(&y));
        let (pre, _) = components(&block, 20, 20, f32::INFINITY);
        assert_eq!(pre.len(), 1);
        let (x, y, w, h) = pre[0].aabb();
        assert!((x - 4.5).abs() < 0.01 && (y - 4.5).abs() < 0.01);
        assert!((w - 6.0).abs() < 0.01 && (h - 6.0).abs() < 0.01);
    }

    /// D3.3: the expansion is Android's uncapped `area*ratio/perimeter` per
    /// side by default; DET_EXPAND_CAP is an opt-in clamp in source pixels.
    #[test]
    fn fit_components_expansion_is_uncapped_by_default() {
        // A 40x10 block: area 400, perimeter 100, expand 6 per side.
        let map = prob_map(60, 30, |x, y| (10..50).contains(&x) && (10..20).contains(&y));
        let (pre, uncl) = components(&map, 60, 30, f32::INFINITY);
        assert_eq!(pre.len(), 1);
        assert!((uncl[0].w - pre[0].w - 12.0).abs() < 0.01, "uncapped");
        let (_, capped) = components(&map, 60, 30, 2.0);
        assert!((capped[0].w - pre[0].w - 4.0).abs() < 0.01, "capped at 2/side");
    }

    /// Mobile's axis-path merge applied to straight frames: overlapping
    /// halves of one long line union (IoM ≥ 0.40), rotated frames never.
    #[test]
    fn straight_boxes_merge_like_mobile() {
        let frame = |x: i32, y: i32, w: i32, h: i32| {
            RotatedBox::new(
                x as f32 + w as f32 / 2.0,
                y as f32 + h as f32 / 2.0,
                w as f32,
                h as f32,
                0.0,
                0.8,
            )
        };
        // Overlap 30x20 of a 100x20 box: IoM 0.30 < 0.40, stays split.
        let apart = merge_straight_boxes(vec![
            (BoundingBox::new(0, 0, 100, 20, 0.8), frame(0, 0, 100, 20)),
            (BoundingBox::new(70, 0, 100, 20, 0.8), frame(70, 0, 100, 20)),
        ]);
        assert_eq!(apart.len(), 2);
        // Overlap 50x20: IoM 0.50, union to (0,0,150,20).
        let merged = merge_straight_boxes(vec![
            (BoundingBox::new(0, 0, 100, 20, 0.8), frame(0, 0, 100, 20)),
            (BoundingBox::new(50, 0, 100, 20, 0.8), frame(50, 0, 100, 20)),
        ]);
        assert_eq!(merged.len(), 1, "overlapping halves must merge");
        assert_eq!((merged[0].0.x, merged[0].0.y, merged[0].0.w, merged[0].0.h), (0, 0, 150, 20));
        assert!(!merged[0].1.is_rotated());
        // A genuinely rotated frame takes no part.
        let mixed = merge_straight_boxes(vec![
            (BoundingBox::new(0, 0, 100, 20, 0.8), frame(0, 0, 100, 20)),
            (
                BoundingBox::new(50, 0, 100, 20, 0.8),
                RotatedBox::new(100.0, 10.0, 100.0, 20.0, 10.0f32.to_radians(), 0.8),
            ),
        ]);
        assert_eq!(mixed.len(), 2, "rotated frames never merge");
    }

    /// D3.8: a DB blob that merged several lines — its frame encloses two
    /// line-shaped frames — is dropped while the lines survive.
    #[test]
    fn detect_filters_enclosing_blobs() {
        // A 200x200 ring (one component) around two horizontal bars.
        let map = prob_map(240, 240, |x, y| {
            let ring = (10..210).contains(&x)
                && (10..210).contains(&y)
                && (x == 10 || x == 209 || y == 10 || y == 209);
            let line_a = (30..190).contains(&x) && (40..52).contains(&y);
            let line_b = (30..190).contains(&x) && (120..132).contains(&y);
            ring || line_a || line_b
        });
        let (pre, quads) = components(&map, 240, 240, f32::INFINITY);
        assert_eq!(quads.len(), 3, "ring + two lines");
        let kept = filter_fitted_quads(&pre, &quads, 240, 240);
        assert_eq!(kept.len(), 2, "the enclosing ring must be dropped");
        for q in &kept {
            assert!(q.w > q.h, "line frames survive: {:?}", q);
        }
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
    /// Live tuning reaches detection: a near-1.0 threshold drops the line,
    /// and a zero unclip ratio tightens the fitted boxes.
    #[test]
    fn det_tuning_overrides_change_detection_output() {
        let mut eng = test_engine();
        let truth: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!(
                "{}/test_images/synth/truth.json",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap(),
        )
        .unwrap();
        let file = truth["lines"][0]["file"].as_str().unwrap();
        let img =
            image::open(format!("{}/test_images/synth/{file}", env!("CARGO_MANIFEST_DIR")))
                .unwrap();
        let area = |b: &BoundingBox| (b.w as i64) * (b.h as i64);

        let base = eng.detect(&img).unwrap();
        let base_max = base
            .boxes
            .iter()
            .max_by_key(|b| area(b))
            .expect("baseline detects a line");

        eng.det_thresh_override = Some(1.0);
        let high = eng.detect(&img).unwrap();
        assert!(high.boxes.is_empty(), "a 1.0 threshold should drop every box");

        eng.det_thresh_override = None;
        eng.det_unclip_override = Some(0.0);
        let tight = eng.detect(&img).unwrap();
        match tight.boxes.iter().max_by_key(|b| area(b)) {
            Some(t) => assert!(
                area(t) < area(base_max),
                "unclip 0.0 should be tighter than 1.5: {t:?} vs {base_max:?}"
            ),
            // Everything fell under the min-size filter: also tighter.
            None => {}
        }
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
        // wide (120) swallowed the ruby. Frames are vertical (angle 0, local
        // x = cross width, local y = reading extent).
        let mut pairs = vec![
            (
                BoundingBox::new(0, 0, 120, 240, 1.0),
                RotatedBox::new(60.0, 120.0, 120.0, 240.0, 0.0, 1.0),
            ),
            (
                BoundingBox::new(200, 0, 60, 240, 1.0),
                RotatedBox::new(230.0, 120.0, 60.0, 240.0, 0.0, 1.0),
            ),
            (
                BoundingBox::new(300, 0, 60, 240, 1.0),
                RotatedBox::new(330.0, 120.0, 60.0, 240.0, 0.0, 1.0),
            ),
        ];
        assert_eq!(trim_ruby_gutter_vertical(&mut pairs, &image), 1);
        // Gutter starts at x=56; the frame's local x is the cross width.
        assert_eq!(pairs[0].0.w, 56);
        assert!((pairs[0].1.w - 56.0).abs() < 0.01, "quad w {}", pairs[0].1.w);
        assert!((pairs[0].1.cx - 28.0).abs() < 0.01, "quad cx {}", pairs[0].1.cx);
        assert_eq!(pairs[1].0.w, 60);
        assert_eq!(pairs[2].0.w, 60);
    }

    /// Mobile #49's legacy columns: `(t+0.5)·L/seqLenTotal`, centred with
    /// half the cross size, clamped to the frame.
    #[test]
    fn legacy_cells_use_frame_length_over_seq_len() {
        // L=100, seqLen=4 -> avgColW 25; cross 10 -> half 5.
        let cells = legacy_cells(&[0.0, 1.0, 2.0, 3.0], 4, 100.0, 10.0);
        assert_eq!(cells, vec![(7.5, 17.5), (32.5, 42.5), (57.5, 67.5), (82.5, 92.5)]);
        // A fractional column keeps its fraction (peak interpolation).
        let cells = legacy_cells(&[0.5], 4, 100.0, 10.0);
        assert_eq!(cells, vec![(20.0, 30.0)]);
        // Clamped to [0, L].
        let cells = legacy_cells(&[0.0, 3.0], 4, 100.0, 60.0);
        assert_eq!(cells, vec![(0.0, 42.5), (57.5, 100.0)]);
    }

    /// Mobile #49 uniform sizing: fullwidth cells become one em wide,
    /// halfwidth 0.5 em, around unchanged centres.
    #[test]
    fn uniform_cells_size_to_the_estimated_em() {
        // Centres 20 apart with fullwidth chars -> em 20; widths 20.
        let chars: Vec<char> = "日本語".chars().collect();
        let cells = vec![(0.0f32, 40.0f32), (20.0, 60.0), (40.0, 80.0)];
        let sized = uniform_cells(&cells, &chars, 100.0);
        assert!((sized[0].1 - sized[0].0 - 20.0).abs() < 1e-3, "{sized:?}");
        assert!((sized[1].1 - sized[1].0 - 20.0).abs() < 1e-3, "{sized:?}");
        // Halfwidth ASCII: 10px apart = 0.5 em each -> em 20, width 10.
        let chars: Vec<char> = "ABC".chars().collect();
        let cells = vec![(0.0f32, 20.0f32), (10.0, 30.0), (20.0, 40.0)];
        let sized = uniform_cells(&cells, &chars, 100.0);
        assert!((sized[0].1 - sized[0].0 - 10.0).abs() < 1e-3, "{sized:?}");
        // Single cell: unchanged (needs 2+).
        let chars: Vec<char> = "日".chars().collect();
        assert_eq!(uniform_cells(&[(0.0, 40.0)], &chars, 100.0), vec![(0.0, 40.0)]);
    }

    /// Mobile #49 vertical punctuation sets (D7.3/D46): the full Android
    /// closing/opening sets, including the marks the PC list used to miss.
    #[test]
    fn vertical_punctuation_sets_match_android() {
        for ch in "。.．、,，)）〕》」』】〙〗〟’”］".chars() {
            assert!(is_vertical_closing_punct(ch), "closing {ch:?}");
        }
        for ch in "(（〔《「『【〘〖〝‘“［".chars() {
            assert!(is_vertical_opening_punct(ch), "opening {ch:?}");
        }
        // PC-only strays from the old list must no longer be treated as marks.
        assert!(!is_vertical_closing_punct('〉'));
        assert!(!is_vertical_opening_punct('〈'));
        assert!(!is_vertical_closing_punct('a'));
    }

    /// Mobile `computeCharBoxes` horizontal path end to end: legacy columns,
    /// no pixels (legacy), uniform widths — the cell centres stay on the CTC
    /// columns and the widths follow the estimated em.
    #[test]
    fn compute_char_boxes_horizontal_uniform_em() {
        // crop 200 wide, seqLen 4 -> avgColW 50, cross (cropH) 40.
        let text = "あいうえ";
        let cols = [0.0f32, 1.0, 2.0, 3.0];
        let boxes = compute_char_boxes(text, &cols, 4, 10, 20, 200, 40, false, None);
        assert_eq!(boxes.len(), 4);
        for b in &boxes {
            assert_eq!(b.y, 20);
            assert_eq!(b.h, 40);
        }
        // Centres at (t+0.5)*50 + 10 -> 35, 85, 135, 185; em 50, width 50.
        let centers: Vec<f32> = boxes
            .iter()
            .map(|b| b.x as f32 + b.w as f32 / 2.0)
            .collect();
        assert_eq!(centers, vec![35.0, 85.0, 135.0, 185.0]);
        for b in &boxes {
            assert!((b.w as f32 - 50.0).abs() <= 1.0, "w={}", b.w);
        }
        // Halfwidth chars: 50px apart = 0.5 em -> em 100, width 50.
        let boxes = compute_char_boxes("ABCD", &cols, 4, 0, 0, 200, 40, false, None);
        for b in &boxes {
            assert!((b.w as f32 - 50.0).abs() <= 1.0, "w={}", b.w);
        }
    }

    /// Mobile `computeCharBoxes` vertical path: punctuation shrinks onto the
    /// neighbour and expands back to the average non-punct cell height.
    #[test]
    fn compute_char_boxes_vertical_punctuation_rules() {
        // crop 100 tall, seqLen 4 -> avgColW 25, cross (cropW) 40.
        let cols = [0.0f32, 1.0, 2.0, 3.0];
        let boxes = compute_char_boxes("あ。いあ", &cols, 4, 0, 0, 40, 100, true, None);
        assert_eq!(boxes.len(), 4);
        // Closing 。 shrinks onto the next box's start then expands back to
        // the mean non-punct height (its height matches the plain cells).
        let cp = &boxes[1];
        assert!(cp.y + cp.h <= boxes[2].y + 1, "closing mark {:?}", cp);
        assert!(
            (cp.h - boxes[2].h).abs() <= 2,
            "closing h={} vs plain h={}",
            cp.h,
            boxes[2].h
        );
        // Opening 「 expands downward from its end to the mean height.
        let cols3 = [0.0f32, 1.0, 2.0];
        let boxes = compute_char_boxes("「あい", &cols3, 4, 0, 0, 40, 100, true, None);
        assert!(
            (boxes[0].h - boxes[1].h).abs() <= 2,
            "opening h={} vs plain h={}",
            boxes[0].h,
            boxes[1].h
        );
    }

    /// End-to-end: every char box the streaming path emits must follow the
    /// mobile #49 uniform-em sizing, and plain-glyph lines must land on the
    /// synth truth centres along the reading axis.
    #[test]
    fn synth_char_boxes_uniform_em_and_position() {
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
                let quad_present = ann.quad.is_some();
                let Some(line) = ann.line else { continue };
                if quad_present {
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
                // Uniform-em regression (mobile #49): every emitted cell's
                // along length is the line's estimated em (halfwidth: 0.5em),
                // clamped at the frame edges.
                let centers: Vec<f32> = line
                    .char_boxes
                    .iter()
                    .map(|b| {
                        if vertical {
                            b.y as f32 + b.h as f32 / 2.0
                        } else {
                            b.x as f32 + b.w as f32 / 2.0
                        }
                    })
                    .collect();
                let em = crate::util::japanese::estimate_em(&line.text, &centers);
                for (ci, b) in line.char_boxes.iter().enumerate() {
                    let along = if vertical { b.h } else { b.w } as f32;
                    assert!(along >= 1.0, "{file} char {ci}: empty box");
                    if em > 0.0 {
                        let expected = if chars
                            .get(ci)
                            .copied()
                            .map(crate::util::japanese::is_half_width)
                            .unwrap_or(false)
                        {
                            0.5 * em
                        } else {
                            em
                        };
                        assert!(
                            (along - expected).abs() <= expected * 0.3 + 3.0,
                            "{file} char {ci} ({:?}): along {along} vs uniform {expected} (em {em})",
                            chars.get(ci)
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked >= 80, "expected the full synth set, checked {checked}");
        assert!(
            position_checked >= 5,
            "expected at least 5 plain-glyph lines for the position check, got {position_checked}"
        );
    }

    /// Mobile `OcrEngine.reDecodeLineResult`: a line rebuilt from its cached
    /// raw alternatives without the model. Text and alternatives are
    /// rewritten, char boxes recomputed in the annotation's own frame, and the
    /// raw cache / sidecar / orientation carry over. `seq_len_total == 0`
    /// keeps the existing boxes (mobile's `cropW == 0` fallback), and a line
    /// with nothing cached comes back unchanged.
    #[test]
    fn annotation_re_decodes_from_raw_alternatives() {
        let step = |entries: &[(char, f32)]| entries.to_vec();
        let raw = vec![
            step(&[('あ', 0.9), ('い', 0.1)]),
            step(&[('あ', 0.8), ('い', 0.2)]),
            step(&[('\u{3000}', 0.95), ('あ', 0.4)]),
            step(&[('い', 0.7), ('あ', 0.3)]),
        ];
        let line = LineResult {
            text: "あ".into(),
            char_boxes: vec![BoundingBox::new(0, 0, 40, 40, 1.0)],
            alternatives: vec![vec![('あ', 0.9)]],
            raw_alternatives: raw.clone(),
            sample_txt: Some("/tmp/sample.txt".into()),
            is_vertical: false,
            chunk_boxes: vec![BoundingBox::new(0, 0, 160, 40, 1.0)],
        };
        let ann = DetectedAnnotation {
            bbox: BoundingBox::new(10, 20, 120, 40, 1.0),
            quad: None,
            line: Some(line),
        };

        // Geometry unknown: the walk still rewrites text/alternatives, but the
        // old boxes stay (there is nothing to recompute them from).
        let kept = ann.re_decode_line(0).line.unwrap();
        assert_eq!(kept.text, "あい");
        assert_eq!(kept.alternatives.len(), 2);
        assert_eq!(kept.char_boxes.len(), 1, "boxes kept without geometry");
        assert_eq!(kept.raw_alternatives, raw);
        assert_eq!(
            kept.sample_txt.as_deref(),
            Some(std::path::Path::new("/tmp/sample.txt"))
        );

        // Geometry known: one box per emitted character, in the bbox frame.
        let rebuilt = ann.re_decode_line(4);
        assert_eq!(rebuilt.bbox.x, 10);
        assert!(rebuilt.quad.is_none());
        let line = rebuilt.line.unwrap();
        assert_eq!(line.text, "あい");
        assert_eq!(line.alternatives.len(), 2);
        assert_eq!(line.alternatives[1][0], ('い', 0.7));
        assert_eq!(line.char_boxes.len(), 2, "one box per emitted character");
        for b in &line.char_boxes {
            assert!(b.x >= 10 && b.right() <= 130, "boxes stay in the crop frame: {b:?}");
        }
        assert_eq!(line.raw_alternatives, raw, "the cache is carried, not dropped");

        // Nothing cached: unchanged (mobile returns `oldLine`).
        let bare = DetectedAnnotation {
            line: Some(LineResult {
                raw_alternatives: vec![],
                ..ann.line.clone().unwrap()
            }),
            ..ann.clone()
        };
        let same = bare.re_decode_line(4).line.unwrap();
        assert_eq!(same.text, "あ");
        assert_eq!(same.char_boxes.len(), 1);
    }
}
