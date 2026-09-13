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
            raw_pairs.push((bbox, rot));
        }

        // Debug: print raw detected boxes
        eprintln!("[PP-OCR DET] raw {} boxes:", raw_pairs.len());
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
        // 1. Shrink vertical box widths by 10% (centered)
        for (b, _) in pp_pairs.iter_mut().filter(|(b, _)| b.h > b.w) {
            let shrink = (b.w as f32 * 0.05).round() as i32;
            b.x += shrink;
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
                        if text.is_empty() { continue; }

                        // Dataset collection: save the line crop + detected text
                        let sample_txt = save_line_sample(&job.crop, &text);

                        let n = char_cols.len();
                        let mut char_boxes = Vec::with_capacity(n);
                        if n > 0 && seq_len_total > 0 && !job.is_vertical {
                            // ---- HORIZONTAL: x-axis char boxes ----
                            let avg_col_w = job.crop_w as f32 / seq_len_total as f32;
                            let char_w = (job.crop_h as f32).max(3.0);
                            let mut cells: Vec<(f32, f32)> = char_cols.iter().map(|&t| {
                                let c = (t as f32 + 0.5) * avg_col_w;
                                let h = char_w / 2.0;
                                ((c - h).max(0.0), (c + h).min(job.crop_w as f32))
                            }).collect();
                            cells.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            for ci in 0..n.saturating_sub(1) {
                                if cells[ci].1 <= cells[ci + 1].0 { continue; }
                                let half = (cells[ci].1 - cells[ci + 1].0) / 2.0;
                                cells[ci].1 -= half; cells[ci + 1].0 += half;
                            }
                            for &(xl, xr) in &cells {
                                char_boxes.push(BoundingBox::new(
                                    (job.crop_x as f32 + xl).round() as i32, job.crop_y as i32,
                                    (xr - xl).max(1.0).round() as i32, job.crop_h as i32, 1.0,
                                ));
                            }
                        } else if n > 0 && seq_len_total > 0 {
                            // ---- VERTICAL: y-axis char boxes with punct handling ----
                            let avg_col_w = job.crop_h as f32 / seq_len_total as f32;
                            let avg_ch_h = if n > 1 {
                                let span = char_cols[n - 1] - char_cols[0];
                                (span / (n - 1) as f32 * avg_col_w).max(3.0)
                            } else { avg_col_w.max(3.0) };
                            let mut cells: Vec<(f32, f32)> = char_cols.iter().map(|&t| {
                                let c = (t as f32 + 0.5) * avg_col_w;
                                let h = avg_ch_h / 2.0;
                                ((c - h).max(0.0), (c + h).min(job.crop_h as f32))
                            }).collect();
                            cells.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            let is_cp: Vec<bool> = text.chars().map(|ch| matches!(ch, '\u{3002}'|'\u{002E}'|'\u{FF0E}'|'\u{3001}'|'\u{002C}'|'\u{FF0C}'|')'|'\u{FF09}'|'\u{3017}'|'\u{300D}'|'\u{300F}'|'\u{3015}'|'\u{3011}'|'\u{3009}'|']'|'\u{FF3D}')).collect();
                            let is_op: Vec<bool> = text.chars().map(|ch| matches!(ch, '('|'\u{FF08}'|'\u{300C}'|'\u{300E}'|'\u{3014}'|'\u{3010}'|'\u{300A}'|'\u{3008}'|'\u{3016}'|'['|'\u{FF3B}')).collect();
                            for ci in 0..n.saturating_sub(1) {
                                if cells[ci].1 <= cells[ci + 1].0 { continue; }
                                if is_cp[ci] { cells[ci].1 = cells[ci + 1].0; }
                                else if is_op[ci + 1] { cells[ci + 1].0 = cells[ci].1; }
                                else if is_cp[ci + 1] { cells[ci + 1].0 = cells[ci].1; }
                                else if is_op[ci] { cells[ci].1 = cells[ci + 1].0; }
                                else { let h = (cells[ci].1 - cells[ci + 1].0) / 2.0; cells[ci].1 -= h; cells[ci + 1].0 += h; }
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
                                char_boxes.push(BoundingBox::new(
                                    job.crop_x as i32, (job.crop_y as f32 + yt).round() as i32,
                                    job.crop_w as i32, ch.round() as i32, 1.0,
                                ));
                            }
                        }

                        // Apply glyph conversion only for vertical text
                        let (final_text, final_alts) = if job.is_vertical {
                            (text.chars().map(|c| crate::util::japanese::to_vertical_glyph(c)).collect::<String>(),
                             alternatives.into_iter().map(|alts| alts.into_iter().map(|(c, s)| (crate::util::japanese::to_vertical_glyph(c), s)).collect()).collect())
                        } else {
                            (text, alternatives)
                        };

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
}
