use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use imageproc::contours;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use fontdue::{Font, FontSettings, LineMetrics};
use std::path::Path;
use std::time::Instant;

use crate::models::*;
use crate::util::japanese::to_vertical_glyph;

// PP-OCRv6 detection constants
const PPOCR_DET_LONG_SIDE: u32 = 960;
const PPOCR_DET_THRESH: f32 = 0.3;
const PPOCR_DET_BOX_THRESH: f32 = 0.8;
const PPOCR_DET_UNCLIP_RATIO: f32 = 1.1;
const X_OVERLAP_THRESHOLD: f32 = 0.3;

// Box fill ratio when rendering glyphs inside detected boxes. 1.0 means match box height, <1.0 leave padding.
const BOX_FILL_RATIO: f32 = 0.9;

/// Number of sessions in the PP-OCRv6 pool.
/// Each session adds ~21 MB of RAM.
const PPOCR_SESSION_POOL_SIZE: usize = 4;
pub struct OcrEngine {
    pub detect_session: Session,
    /// PP-OCRv6 recognition sessions — used for both horizontal and vertical text.
    pub ppocr_session: RecognizeSessionPool,
    pub ppocr_vocab: Vec<String>,
    pub recognition_mode: RecognitionMode,
    pub batch_size: usize,
    /// BOOOCR sidecar (feature-vector recognizer). When present, recognition
    /// goes through it instead of the PP-OCR rec sessions.
    pub booocr: Option<std::sync::Arc<std::sync::Mutex<crate::booocr::BooOcrClient>>>,
    model_dir: String,
}

/// Lazily-initialized pool of recognition sessions.
///
/// On first access, all sessions are loaded in parallel.
pub struct RecognizeSessionPool {
    sessions: std::sync::OnceLock<Vec<std::sync::Arc<std::sync::Mutex<Session>>>>,
    model_path: std::path::PathBuf,
    pool_size: usize,
}

impl RecognizeSessionPool {
    fn new(model_path: std::path::PathBuf) -> Self {
        RecognizeSessionPool {
            sessions: std::sync::OnceLock::new(),
            model_path,
            pool_size: PPOCR_SESSION_POOL_SIZE,
        }
    }

    fn empty() -> Self {
        RecognizeSessionPool {
            sessions: std::sync::OnceLock::new(),
            model_path: std::path::PathBuf::new(),
            pool_size: 0,
        }
    }

    fn with_size(model_path: std::path::PathBuf, pool_size: usize) -> Self {
        RecognizeSessionPool {
            sessions: std::sync::OnceLock::new(),
            model_path,
            pool_size,
        }
    }

    /// Get the session pool, initializing it on first call.
    pub fn get(&self) -> &[std::sync::Arc<std::sync::Mutex<Session>>] {
        let n = self.pool_size;
        self.sessions.get_or_init(|| {
            let mut sessions = Vec::with_capacity(n);
            for _ in 0..n {
                let s = Session::builder()
                    .expect("Failed to create session builder")
                    .with_execution_providers([ort::ep::XNNPACK::default().build()])
                    .expect("Failed to configure XNNPACK execution provider")
                    .commit_from_file(&self.model_path)
                    .expect("Failed to load recognition model");
                sessions.push(std::sync::Arc::new(std::sync::Mutex::new(s)));
            }
            sessions
        })
    }
}

impl OcrEngine {
    pub fn new(model_dir: &str, recognition_mode: RecognitionMode, batch_size: usize) -> Result<Self> {
        let model_path = Path::new(model_dir);

        let detect_session = Session::builder()
            .expect("Failed to create session builder")
            .with_execution_providers([ort::ep::XNNPACK::default().build()])
            .expect("Failed to configure XNNPACK for detection")
            .commit_from_file(model_path.join("PP-OCRv6_small_det_onnx").join("inference.onnx"))
            .expect("Failed to load PP-OCRv6 detection model");

        println!("Detection model loaded. Recognition models will be loaded on first use.");

        // PP-OCRv6 recognition model
        let ppocr_model_path = model_path.join("PP-OCRv6_small_rec_onnx").join("inference.onnx");
        let ppocr_pool = if ppocr_model_path.exists() {
            RecognizeSessionPool::with_size(ppocr_model_path, PPOCR_SESSION_POOL_SIZE)
        } else {
            eprintln!("[PP-OCR] Model not found at {ppocr_model_path:?}, recognition disabled");
            RecognizeSessionPool::empty()
        };
        let ppocr_vocab_path = model_path.join("PP-OCRv6_small_rec_onnx").join("vocab.json");
        let ppocr_vocab: Vec<String> = match std::fs::read_to_string(&ppocr_vocab_path) {
            Ok(content) => serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse PP-OCR vocab"))?,
            Err(_) => {
                eprintln!("[PP-OCR] Vocab not found at {ppocr_vocab_path:?}");
                Vec::new()
            }
        };

        println!("Recognition mode: {:?}, batch size: {}", recognition_mode, batch_size);

        // BOOOCR sidecar: primary recognizer when available. Spawning it
        // blocks ~3s on the font-database load; on any failure we fall back
        // to the PP-OCR recognition sessions.
        let booocr = match crate::booocr::BooOcrClient::new(&crate::booocr::booocr_dir()) {
            Ok(c) => {
                println!("[BOOOCR] sidecar ready — line recognition via BOOOCR");
                Some(std::sync::Arc::new(std::sync::Mutex::new(c)))
            }
            Err(e) => {
                eprintln!("[BOOOCR] unavailable ({e}) — falling back to PP-OCR recognition");
                None
            }
        };

        Ok(OcrEngine {
            detect_session,
            ppocr_session: ppocr_pool,
            ppocr_vocab,
            recognition_mode,
            batch_size,
            booocr,
            model_dir: model_dir.to_string(),
        })
    }

    pub fn is_ready(&self) -> bool {
        !self.ppocr_vocab.is_empty()
    }

    /// Detects bounding boxes using PP-OCRv6 segmentation-based detection model.
    /// The model outputs a probability map [1,1,H,W]. Post-processing:
    /// threshold → find connected components → bounding boxes → merge → sort.
    pub fn detect(&mut self, image: &DynamicImage) -> Result<Vec<BoundingBox>> {
        let orig_w = image.width() as f32;
        let orig_h = image.height() as f32;

        // 1. Resize: keep aspect ratio, longest side = PPOCR_DET_LONG_SIDE, pad to 32
        let scale = PPOCR_DET_LONG_SIDE as f32 / orig_w.max(orig_h);
        let resize_w = (orig_w * scale).round() as u32;
        let resize_h = (orig_h * scale).round() as u32;
        let resize_w = resize_w.max(32);
        let resize_h = resize_h.max(32);
        let pad_w = ((resize_w + 31) / 32) * 32;
        let pad_h = ((resize_h + 31) / 32) * 32;

        let resized = image.resize_exact(resize_w, resize_h, image::imageops::FilterType::Triangle);
        let mut padded = RgbaImage::from_pixel(pad_w, pad_h, Rgba([128u8, 128u8, 128u8, 255u8]));
        for y in 0..resize_h {
            for x in 0..resize_w {
                let px = resized.get_pixel(x, y);
                padded.put_pixel(x, y, px);
            }
        }

        // 2. Convert to NCHW float with ImageNet normalization
        let num_elements = (3 * pad_h * pad_w) as usize;
        let mut data = vec![0.0f32; num_elements];
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let w = pad_w as usize;
        let h = pad_h as usize;
        for y in 0..h {
            for x in 0..w {
                let px = padded.get_pixel(x as u32, y as u32);
                let r = (px.0[0] as f32 / 255.0 - mean[0]) / std[0];
                let g = (px.0[1] as f32 / 255.0 - mean[1]) / std[1];
                let b = (px.0[2] as f32 / 255.0 - mean[2]) / std[2];
                let idx = y * w + x;
                data[idx] = r;
                data[h * w + idx] = g;
                data[2 * h * w + idx] = b;
            }
        }

        let input_tensor = Tensor::from_array((
            [1i64, 3, pad_h as i64, pad_w as i64],
            data.into_boxed_slice(),
        ))?;

        // 3. Build inputs
        let input_name = self
            .detect_session
            .inputs()
            .iter()
            .next()
            .context("No input found")?
            .name()
            .to_string();
        let inputs = ort::inputs! { input_name.as_str() => input_tensor };

        // 4. Extract output name before running (avoids double borrow)
        let output_name = self
            .detect_session
            .outputs()
            .iter()
            .next()
            .context("No output found")?
            .name()
            .to_string();

        // 5. Run inference, extract prob map [1,1,H,W]
        let (prob_map, out_w, out_h) = {
            let run_outputs = self.detect_session.run(inputs)?;
            let output_val = run_outputs
                .get(output_name.as_str())
                .context("Failed to get output")?;
            let arr = output_val.try_extract_array::<f32>()?.to_owned();
            eprintln!("[PP-OCR DET] output shape: {:?}", arr.shape());
            // Shape [1,1,out_h,out_w] — squeeze to 2D
            let out_h = if arr.ndim() == 4 { arr.shape()[2] as usize } else { pad_h as usize };
            let out_w = if arr.ndim() == 4 { arr.shape()[3] as usize } else { pad_w as usize };
            let raw: Vec<f32> = arr.iter().copied().collect();
            // Reshape to 2D grid
            let mut map = vec![0.0f32; out_h * out_w];
            if arr.ndim() == 4 {
                // [1,1,out_h,out_w]
                for y in 0..out_h {
                    for x in 0..out_w {
                        map[y * out_w + x] = raw[y * out_w + x];
                    }
                }
            } else {
                // fallback: 1D flat
                map = raw;
            }
            (map, out_w, out_h)
        };
        // Debug: print prob_map statistics to understand value range
        if !prob_map.is_empty() {
            let max_val = prob_map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_val = prob_map.iter().cloned().fold(f32::INFINITY, f32::min);
            let sum: f32 = prob_map.iter().sum();
            let mean = sum / prob_map.len() as f32;
            eprintln!("[PP-OCR DET] prob_map: min={min_val:.4} max={max_val:.4} mean={mean:.4}");
        }
        let out_w = out_w as u32;
        let out_h = out_h as u32;

        // Scale factor from model output (padded) to original image
        // The image content occupies resize_w × resize_h within the padded space
        let scale_w = orig_w / resize_w as f32;
        let scale_h = orig_h / resize_h as f32;

        // 5. Threshold → find contours → bounding boxes
        let mut binary = vec![0u8; (out_w * out_h) as usize];
        for y in 0..out_h {
            for x in 0..out_w {
                let idx = (y * out_w + x) as usize;
                if idx < prob_map.len() && prob_map[idx] > PPOCR_DET_THRESH {
                    binary[idx] = 255;
                }
            }
        }

        let binary_img = image::GrayImage::from_raw(out_w, out_h, binary)
            .context("Failed to create binary image")?;
        let contours = contours::find_contours_with_threshold::<i32>(&binary_img, 128);
        let mut raw_boxes: Vec<BoundingBox> = Vec::new();

        for contour in &contours {
            if contour.points.len() < 3 { continue; } // noise filter

            let mut min_x = out_w as i32;
            let mut min_y = out_h as i32;
            let mut max_x = 0i32;
            let mut max_y = 0i32;
            for pt in &contour.points {
                min_x = min_x.min(pt.x);
                min_y = min_y.min(pt.y);
                max_x = max_x.max(pt.x);
                max_y = max_y.max(pt.y);
            }

            // Unclip: expand box using proper PP-OCR formula: distance = area * ratio / perimeter
            let bw = (max_x - min_x) as f32;
            let bh = (max_y - min_y) as f32;
            let area = bw * bh;
            let perimeter = 2.0 * (bw + bh);
            let expand = if perimeter > 0.0 { area * PPOCR_DET_UNCLIP_RATIO / perimeter } else { 0.0 };
            let ux = (min_x as f32 - expand).max(0.0);
            let uy = (min_y as f32 - expand).max(0.0);
            let ux2 = (max_x as f32 + expand).min(out_w as f32 - 1.0);
            let uy2 = (max_y as f32 + expand).min(out_h as f32 - 1.0);

            // Scale back to original image coords
            let orig_x = (ux * scale_w).round() as i32;
            let orig_y = (uy * scale_h).round() as i32;
            let orig_w_box = ((ux2 - ux) * scale_w).round() as i32;
            let orig_h_box = ((uy2 - uy) * scale_h).round() as i32;

            // Average prob over the region as confidence score
            let _score = 0.0f32;

            if orig_w_box < 4 || orig_h_box < 4 { continue; }

            raw_boxes.push(BoundingBox::new(
                orig_x, orig_y, orig_w_box, orig_h_box, PPOCR_DET_BOX_THRESH,
            ));
        }

        // Debug: print raw detected boxes
        eprintln!("[PP-OCR DET] raw {} boxes:", raw_boxes.len());
        for (i, b) in raw_boxes.iter().enumerate() {
            eprintln!("  [{i}] x={} y={} w={} h={} c={:.3}", b.x, b.y, b.w, b.h, b.confidence);
        }

        // MERGE DISABLED for diagnosis
        let sorted = self.sort_detected_boxes(raw_boxes);
        // Post-processing
        let mut pp_boxes = sorted;
        // Filter out degenerate tiny boxes (noise specks)
        pp_boxes.retain(|b| b.w >= 10 && b.h >= 10);
        // 1. Shrink vertical box widths by 10% (centered)
        for b in pp_boxes.iter_mut().filter(|b| b.h > b.w) {
            let shrink = (b.w as f32 * 0.05).round() as i32;
            b.x += shrink;
        }
        // 2. For stacked overlapping horizontal boxes, split at overlap midpoint
        let h_indices: Vec<usize> = pp_boxes
                .iter()
                .enumerate()
                .filter(|(_, b)| b.w >= b.h)
                .map(|(i, _)| i)
                .collect();
            for i in 0..h_indices.len() {
                for j in (i + 1)..h_indices.len() {
                    let ai = h_indices[i];
                    let bi = h_indices[j];
                    // Only split if boxes are in the same column (horizontal overlap too)
                    let a_right = pp_boxes[ai].x + pp_boxes[ai].w;
                    let b_right = pp_boxes[bi].x + pp_boxes[bi].w;
                    let h_overlap = a_right.min(b_right) - pp_boxes[ai].x.max(pp_boxes[bi].x);
                    if h_overlap <= 0 {
                        continue;
                    }
                    let (upper, lower) = if pp_boxes[ai].y <= pp_boxes[bi].y {
                        (ai, bi)
                    } else {
                        (bi, ai)
                    };
                    let upper_bottom = pp_boxes[upper].y + pp_boxes[upper].h;
                    let lower_bottom = pp_boxes[lower].y + pp_boxes[lower].h;
                    // Check vertical overlap: upper box bottom > lower box top
                    if upper_bottom > pp_boxes[lower].y {
                        let overlap_mid = (pp_boxes[lower].y + upper_bottom.min(lower_bottom)) / 2;
                        pp_boxes[upper].h = (overlap_mid - pp_boxes[upper].y).max(1);
                        pp_boxes[lower].y = pp_boxes[upper].y + pp_boxes[upper].h;
                        pp_boxes[lower].h = (lower_bottom - pp_boxes[lower].y).max(1);
                    }
                }
            }
        eprintln!("[PP-OCR DET] final {} boxes:", pp_boxes.len());
        for (i, b) in pp_boxes.iter().enumerate() {
            eprintln!("  [{i}] x={} y={} w={} h={} c={:.3}", b.x, b.y, b.w, b.h, b.confidence);
        }
        Ok(pp_boxes)
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

    pub fn sort_detected_boxes(&self, mut boxes: Vec<BoundingBox>) -> Vec<BoundingBox> {
        // Separate by orientation
        let mut horizontal: Vec<BoundingBox> = Vec::new();
        let mut vertical: Vec<BoundingBox> = Vec::new();
        for b in boxes.drain(..) {
            if b.w >= b.h {
                horizontal.push(b);
            } else {
                vertical.push(b);
            }
        }

        // Horizontal: top-to-bottom, left-to-right
        horizontal.sort_by(|a, b| a.y.cmp(&b.y).then(a.x.cmp(&b.x)));

        // Vertical: right-to-left, top-to-bottom
        // Japanese vertical text is read right-to-left across columns.
        vertical.sort_by(|a, b| b.x.cmp(&a.x).then(a.y.cmp(&b.y)));

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
    pub fn detect_lines(&mut self, image: &DynamicImage) -> Result<Vec<BoundingBox>> {
        let boxes = self.detect(image)?;
        // Merge disabled — render ALL boxes
        let sorted = self.sort_detected_boxes(boxes);
        Ok(sorted)
    }
}


// ---------------------------------------------------------------------------
// Streaming recognition (standalone, Send-friendly)
// ---------------------------------------------------------------------------

/// Sequential ID for dataset line samples — unique across worker threads and runs.
static NEXT_LINE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Save a recognized line crop as `ocr_line_<ts>_<id>.png` in `out_dir`,
/// with a sidecar `<same-stem>.txt` containing the detected text. Used to
/// build a dataset of real OCR content for the next OCR project.
fn save_line_sample(crop: &image::DynamicImage, text: &str, out_dir: &std::path::Path) {
    let id = NEXT_LINE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let stem = out_dir.join(format!("ocr_line_{ts}_{id}"));
    if let Err(e) = crop.save(stem.with_extension("png")) {
        eprintln!("[dataset] failed to save line crop: {e}");
        return;
    }
    if let Err(e) = std::fs::write(stem.with_extension("txt"), text) {
        eprintln!("[dataset] failed to save line text: {e}");
    }
}

/// Run character recognition on pre-detected boxes and stream each result
/// over the channel as soon as it's ready. Designed to be called from a
/// background thread: all sessions are `Arc`-wrapped and `Send`.
/// Line crops + detected text are saved into `out_dir` for dataset collection.
pub fn recognize_boxes_streaming(
    image: &DynamicImage,
    sorted: &[BoundingBox],
    rec_sessions_ppocr: &[std::sync::Arc<std::sync::Mutex<Session>>],
    ppocr_vocab: &[String],
    batch_size: usize,
    recognition_mode: RecognitionMode,
    booocr: Option<std::sync::Arc<std::sync::Mutex<crate::booocr::BooOcrClient>>>,
    sender: std::sync::mpsc::Sender<(usize, DetectedAnnotation)>,
    out_dir: &std::path::Path,
) -> Result<()> {
    use std::time::Instant;
    let t_recognize = Instant::now();
    // Owned copy so worker threads (which require 'static captures) can use it.
    let out_dir = out_dir.to_path_buf();

    let horizontal_boxes: Vec<_> = sorted.iter().filter(|b| b.w >= b.h).cloned().collect();
    let vertical_boxes: Vec<_> = sorted.iter().filter(|b| b.h > b.w).cloned().collect();
    // Pre-compute sorted indices for each horizontal/vertical box
    let h_indices: Vec<usize> = sorted.iter().enumerate()
        .filter(|(_, b)| b.w >= b.h).map(|(i, _)| i).collect();
    let v_indices: Vec<usize> = sorted.iter().enumerate()
        .filter(|(_, b)| b.h > b.w).map(|(i, _)| i).collect();

    // Build a single job queue from ALL boxes (horizontal + vertical).
    // Each job carries its orientation so workers can choose the right
    // character-box computation and post-processing.
    struct Job {
        idx: usize,
        bbox: BoundingBox,
        crop: DynamicImage,
        crop_x: u32, crop_y: u32, crop_w: u32, crop_h: u32,
        is_vertical: bool,
    }
    let mut jobs: Vec<Job> = Vec::with_capacity(sorted.len());
    for (i, bbox) in sorted.iter().enumerate() {
        let crop = image.crop_imm(
            bbox.x.max(0) as u32, bbox.y.max(0) as u32,
            bbox.w.max(0) as u32, bbox.h.max(0) as u32,
        );
        if crop.width() < 4 || crop.height() < 4 { continue; }
        jobs.push(Job {
            idx: i, bbox: bbox.clone(), crop,
            crop_x: bbox.x.max(0) as u32, crop_y: bbox.y.max(0) as u32,
            crop_w: bbox.w.max(0) as u32, crop_h: bbox.h.max(0) as u32,
            is_vertical: bbox.h > bbox.w,
        });
    }

    if !jobs.is_empty() {
        use std::collections::VecDeque;
        let n_jobs = jobs.len();
        let job_queue = std::sync::Arc::new(std::sync::Mutex::new(VecDeque::from(jobs)));
        let mut handles = Vec::new();

        if let Some(client) = booocr {
            // BOOOCR backend: one worker, serialized through the shared
            // sidecar connection. Per-char boxes come straight from BOOOCR.
            let q = std::sync::Arc::clone(&job_queue);
            let client = client.clone();
            let snd = sender.clone();
            let od = out_dir.clone();
            handles.push(std::thread::spawn(move || loop {
                let job = {
                    let mut ql = q.lock().unwrap();
                    ql.pop_front()
                };
                let job = match job {
                    Some(j) => j,
                    None => return,
                };

                let png_path = match crate::booocr::save_crop_png(&job.crop, job.idx) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("[BOOOCR] crop save failed: {e}");
                        continue;
                    }
                };
                let line = {
                    let mut c = client.lock().unwrap();
                    c.recognize_crop(&png_path)
                };
                let _ = std::fs::remove_file(&png_path);

                let Ok(line) = line else {
                    eprintln!("[BOOOCR] recognition failed for line {}", job.idx);
                    continue;
                };
                if line.chars.is_empty() {
                    continue;
                }

                let text: String = line.chars.iter().map(|c| c.c).collect();
                let char_boxes: Vec<BoundingBox> = line
                    .chars
                    .iter()
                    .map(|c| {
                        BoundingBox::new(
                            job.crop_x as i32 + c.x,
                            job.crop_y as i32 + c.y,
                            c.w.max(1),
                            c.h.max(1),
                            c.alts.first().map(|(_, s)| *s).unwrap_or(1.0),
                        )
                    })
                    .collect();
                let alternatives: Vec<Vec<(char, f32)>> =
                    line.chars.iter().map(|c| c.alts.clone()).collect();

                // Dataset collection: save the line crop + detected text
                save_line_sample(&job.crop, &text, &od);

                // Apply glyph conversion only for vertical text
                let (final_text, final_alts) = if job.is_vertical {
                    (
                        text.chars()
                            .map(|c| crate::util::japanese::to_vertical_glyph(c))
                            .collect::<String>(),
                        alternatives
                            .into_iter()
                            .map(|alts| {
                                alts.into_iter()
                                    .map(|(c, s)| (crate::util::japanese::to_vertical_glyph(c), s))
                                    .collect()
                            })
                            .collect(),
                    )
                } else {
                    (text, alternatives)
                };

                let annotation = DetectedAnnotation {
                    bbox: job.bbox.clone(),
                    line: Some(LineResult {
                        text: final_text,
                        char_boxes,
                        alternatives: final_alts,
                        is_vertical: job.is_vertical,
                        chunk_boxes: vec![BoundingBox::new(
                            job.crop_x as i32,
                            job.crop_y as i32,
                            job.crop_w as i32,
                            job.crop_h as i32,
                            1.0,
                        )],
                    }),
                };
                if snd.send((job.idx, annotation)).is_err() {
                    return;
                }
                std::thread::yield_now();
            }));
        } else if !ppocr_vocab.is_empty() && !rec_sessions_ppocr.is_empty() {
            println!(
                "[PP-OCR] Processing {} boxes ({} workers)",
                n_jobs,
                rec_sessions_ppocr.len().min(n_jobs)
            );
            for worker_id in 0..rec_sessions_ppocr
                .len()
                .min(job_queue.lock().unwrap().len())
                .max(1)
            {
            let q = std::sync::Arc::clone(&job_queue);
            let sess = rec_sessions_ppocr[worker_id % rec_sessions_ppocr.len()].clone();
            let voc = ppocr_vocab.to_vec();
            let snd = sender.clone();
            let od = out_dir.clone();

            handles.push(std::thread::spawn(move || loop {
                let job = { let mut ql = q.lock().unwrap(); ql.pop_front() };
                let job = match job { Some(j) => j, None => return };

                let mut session = sess.lock().unwrap();
                let result = crate::ppocr::recognize_ppocr_batch(
                    &mut session, &[&job.crop], &voc,
                );
                drop(session);

                if let Ok(mut results) = result {
                    if let Some((text, alternatives, char_cols, seq_len_total)) = results.pop() {
                        if text.is_empty() { continue; }

                        // Dataset collection: save the line crop + detected text
                        save_line_sample(&job.crop, &text, &od);

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
                            line: Some(LineResult {
                                text: final_text,
                                char_boxes,
                                alternatives: final_alts,
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
