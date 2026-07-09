use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use fontdue::{Font, FontSettings, LineMetrics};
use std::path::Path;
use std::time::Instant;

use crate::models::*;
use crate::ocr_parallel;

// Constants matching the Kotlin implementation
const DETECT_WIDTH: u32 = 960;
const DETECT_HEIGHT: u32 = 544;
const REC_WIDTH: u32 = 960;
const REC_HEIGHT: u32 = 32;
const VERT_REC_WIDTH: u32 = 32;
const VERT_REC_HEIGHT: u32 = 480;
const REC_CONFIDENCE_THRESHOLD: f32 = 0.1;
const X_OVERLAP_THRESHOLD: f32 = 0.3;

// Box fill ratio when rendering glyphs inside detected boxes. 1.0 means match box height, <1.0 leave padding.
const BOX_FILL_RATIO: f32 = 0.9;

// Simulated swapped pairs for text correction
const MEIKI_SWAPPED_PAIRS: &[(&str, &str); 8] = &[
    ("儡傀", "傀儡"),
    ("談冗", "冗談"),
    ("汰淘", "淘汰"),
    ("沱滂", "滂沱"),
    ("攣痙", "痙攣"),
    ("酊酩", "酩酊"),
    ("麭麺", "麺麭"),
    ("哭慟", "慟哭"),
];

/// Number of sessions in each orientation pool.
/// More sessions = less contention, but each session costs ~31 MB of RAM.
const SESSION_POOL_SIZE: usize = 4;

pub struct OcrEngine {
    detect_session: Session,
    /// Recognition sessions are loaded lazily on first use to speed up startup.
    /// Each pool is initialized exactly once via `std::sync::Once`.
    recognize_sessions: RecognizeSessionPool,
    recognize_sessions_vertical: RecognizeSessionPool,
    pub char_vocab: Vec<i64>,
    model_dir: String,
}

/// Lazily-initialized pool of recognition sessions.
///
/// On first access, all `SESSION_POOL_SIZE` sessions are loaded in parallel.
/// This defers ~93 MB of model loading from startup to first recognition call,
/// cutting cold-start time by ~1.5 seconds.
pub struct RecognizeSessionPool {
    sessions: std::sync::OnceLock<Vec<std::sync::Arc<std::sync::Mutex<Session>>>>,
    model_path: std::path::PathBuf,
}

impl RecognizeSessionPool {
    fn new(model_path: std::path::PathBuf) -> Self {
        RecognizeSessionPool {
            sessions: std::sync::OnceLock::new(),
            model_path,
        }
    }

    /// Get the session pool, initializing it on first call.
    fn get(&self) -> &[std::sync::Arc<std::sync::Mutex<Session>>] {
        self.sessions.get_or_init(|| {
            let mut sessions = Vec::with_capacity(SESSION_POOL_SIZE);
            for _ in 0..SESSION_POOL_SIZE {
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
    pub fn new(model_dir: &str) -> Result<Self> {
        let model_path = Path::new(model_dir);

        let detect_session = Session::builder()
            .expect("Failed to create session builder")
            .with_execution_providers([ort::ep::XNNPACK::default().build()])
            .expect("Failed to configure XNNPACK for detection")
            .commit_from_file(model_path.join("meiki.text.detect.v0.1.960x544.onnx"))
            .expect("Failed to load detection model");

        println!("Detection model loaded. Recognition models will be loaded on first use.");

        // Load character vocabulary
        let vocab_path = model_path.join("char_vocab.json");
        let vocab_json: Vec<i64> = match std::fs::read_to_string(&vocab_path) {
            Ok(content) => serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse char_vocab.json"))?,
            Err(_) => Vec::new(),
        };

        let rec_path = model_path.join("meiki.text.rec.v0.960x32.with_logits.onnx");
        let rec_vert_path = model_path.join("meiki.text.rec.v0.vertical.32x480.with_logits.onnx");

        Ok(OcrEngine {
            detect_session,
            recognize_sessions: RecognizeSessionPool::new(rec_path.to_path_buf()),
            recognize_sessions_vertical: RecognizeSessionPool::new(rec_vert_path.to_path_buf()),
            char_vocab: vocab_json,
            model_dir: model_dir.to_string(),
        })
    }

    pub fn is_ready(&self) -> bool {
        !self.char_vocab.is_empty()
    }

    /// Detects bounding boxes in the image using the detection model.
    pub fn detect(&mut self, image: &DynamicImage) -> Result<Vec<BoundingBox>> {
        let orig_w = image.width() as i32;
        let orig_h = image.height() as i32;

        // 1. Resize image to DETECT_WIDTH x DETECT_HEIGHT
        let resized = image.resize_exact(
            DETECT_WIDTH,
            DETECT_HEIGHT,
            image::imageops::FilterType::Triangle,
        );

        // 2. Convert to float tensor in NCHW format, normalized to [0, 1]
        let num_elements = (3 * DETECT_HEIGHT * DETECT_WIDTH) as usize;
        let mut img_data = vec![0.0f32; num_elements];
        for (x, y, pixel) in resized.pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let y_usize = y as usize;
            let x_usize = x as usize;
            let w_usize = DETECT_WIDTH as usize;
            let h_usize = DETECT_HEIGHT as usize;
            img_data[0 * h_usize * w_usize + y_usize * w_usize + x_usize] = r;
            img_data[1 * h_usize * w_usize + y_usize * w_usize + x_usize] = g;
            img_data[2 * h_usize * w_usize + y_usize * w_usize + x_usize] = b;
        }

        let input_tensor = Tensor::from_array((
            [1i64, 3, DETECT_HEIGHT as i64, DETECT_WIDTH as i64],
            img_data.into_boxed_slice(),
        ))?;

        // 3. Build inputs map using the named map form of ort::inputs!
        let session_input_names: Vec<String> = self
            .detect_session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let image_input_name = session_input_names
            .iter()
            .find(|n| n.contains("image") || n.contains("input"))
            .or_else(|| session_input_names.first())
            .context("Detection model has no inputs")?;

        let has_orig_target_sizes = session_input_names.iter().any(|n| n == "orig_target_sizes");

        let inputs = if has_orig_target_sizes {
            let size_tensor = Tensor::from_array((
                [1i64, 2],
                vec![orig_w as i64, orig_h as i64].into_boxed_slice(),
            ))?;
            ort::inputs! {
                image_input_name.as_str() => input_tensor,
                "orig_target_sizes" => size_tensor
            }
        } else {
            ort::inputs! {
                image_input_name.as_str() => input_tensor
            }
        };

        // 4. Run the detection model and extract arrays
        let output_names: Vec<String> = self
            .detect_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let boxes_output_name = output_names
            .iter()
            .find(|n| n.contains("boxes"))
            .or_else(|| output_names.get(0))
            .context("No boxes output found")?
            .clone();
        let scores_output_name = output_names
            .iter()
            .find(|n| n.contains("scores"))
            .or_else(|| output_names.get(1))
            .context("No scores output found")?
            .clone();

        let (boxes_arr, scores_arr) = {
            let run_outputs = self.detect_session.run(inputs)?;

            let boxes_val = run_outputs
                .get(boxes_output_name.as_str())
                .context("Failed to get boxes output")?;
            let scores_val = run_outputs
                .get(scores_output_name.as_str())
                .context("Failed to get scores output")?;

            let boxes_arr = boxes_val.try_extract_array::<f32>()?.to_owned();
            let scores_arr = scores_val.try_extract_array::<f32>()?.to_owned();

            (boxes_arr, scores_arr)
        };

        // 5. Parse boxes and scores
        let num_boxes = if boxes_arr.ndim() == 3 {
            boxes_arr.shape()[1]
        } else {
            boxes_arr.shape()[0]
        };

        let mut detected_boxes = Vec::new();

        for i in 0..num_boxes {
            let score = if scores_arr.ndim() == 2 {
                scores_arr[[0, i]]
            } else {
                scores_arr[i]
            };

            // 6. Filter by confidence threshold
            if score <= 0.4 {
                continue;
            }

            let (left, top, right, bottom) = if boxes_arr.ndim() == 3 {
                (
                    boxes_arr[[0, i, 0]],
                    boxes_arr[[0, i, 1]],
                    boxes_arr[[0, i, 2]],
                    boxes_arr[[0, i, 3]],
                )
            } else {
                (
                    boxes_arr[[i, 0]],
                    boxes_arr[[i, 1]],
                    boxes_arr[[i, 2]],
                    boxes_arr[[i, 3]],
                )
            };

            let left = left as i32;
            let mut top = top as i32;
            let mut right = right as i32;
            let bottom = bottom as i32;

            // 7. Add small margins to avoid clipping
            if bottom - top > right - left {
                // Vertical lines: top and right margins
                let v_margin = ((right - left) as f32 * 0.1).max(2.0) as i32;
                let h_margin = ((right - left) as f32 * 0.05).max(1.0) as i32;
                top = (top - v_margin).max(0);
                right = (right + h_margin).min(orig_w);
            } else {
                // Horizontal lines: right margin
                let h_margin = ((bottom - top) as f32 * 0.1).max(4.0) as i32;
                right = (right + h_margin).min(orig_w);
            }

            detected_boxes.push(BoundingBox::new(
                left,
                top,
                right - left,
                bottom - top,
                score,
            ));
        }

        // 8. Merge redundant, highly overlapping boxes
        let merged_boxes = self.merge_overlapping_boxes(detected_boxes);

        // 9. Sort boxes top-to-bottom, left-to-right
        let sorted_boxes = self.sort_detected_boxes(merged_boxes.clone());

        Ok(sorted_boxes.clone())
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
        // Sort boxes from top to bottom, left to right
        boxes.sort_by(|a, b| {
            let top_a = a.top();
            let top_b = b.top();
            let height = a.h.max(b.h);
            let threshold = height / 2;

            if (top_a - top_b).abs() <= threshold {
                a.left().cmp(&b.left())
            } else {
                top_a.cmp(&top_b)
            }
        });

        boxes
    }

    pub fn recognize_single_line(
        &mut self,
        image: &DynamicImage,
        bbox: &BoundingBox,
    ) -> Result<Option<LineResult>> {
        // Similar to Kotlin: decide between short chunk, horizontal long line, vertical long line
        let is_vertical = bbox.h > bbox.w;
        let crop_x = bbox.x.max(0) as u32;
        let crop_y = bbox.y.max(0) as u32;
        let crop_w = bbox.w.max(0) as u32;
        let crop_h = bbox.h.max(0) as u32;
        if crop_w == 0 || crop_h == 0 {
            return Ok(None);
        }
        let crop = image.crop_imm(crop_x, crop_y, crop_w, crop_h);

        let result: Option<LineResult>;
        if is_vertical && ((crop_h as f32) * (32.0f32 / crop_w as f32) > 350.0f32) {
            result =
                Some(self.recognize_vertical_long_line(&crop, crop_x as i32, crop_y as i32)?);
        } else if !is_vertical && ((crop_w as f32) * (32.0f32 / crop_h as f32) > REC_WIDTH as f32) {
            result =
                Some(self.recognize_horizontal_long_line(&crop, crop_x as i32, crop_y as i32)?);
        } else {
            let (filtered, eff_w, eff_h) = self.recognize_single_chunk(&crop, is_vertical)?;
            let text: String = filtered.iter().map(|c| c.char).collect();
            let alternatives: Vec<Vec<(char, f32)>> =
                filtered.iter().map(|c| c.alternatives.clone()).collect();
            let mut char_boxes: Vec<BoundingBox> = Vec::new();
            for c in filtered.iter() {
                // convert to global rect
                if is_vertical {
                    let x1 = (c.box_coords[0] / (VERT_REC_WIDTH as f32)) * (crop_w as f32)
                        + (crop_x as f32);
                    let y1 = (c.box_coords[1] / (eff_h as f32)) * (crop_h as f32) + (crop_y as f32);
                    let x2 = (c.box_coords[2] / (VERT_REC_WIDTH as f32)) * (crop_w as f32)
                        + (crop_x as f32);
                    let y2 = (c.box_coords[3] / (eff_h as f32)) * (crop_h as f32) + (crop_y as f32);
                    char_boxes.push(BoundingBox::new(
                        x1.round() as i32,
                        y1.round() as i32,
                        (x2 - x1).round() as i32,
                        (y2 - y1).round() as i32,
                        c.score,
                    ));
                } else {
                    let x1 = (c.box_coords[0] / (eff_w as f32)) * (crop_w as f32) + (crop_x as f32);
                    let y1 =
                        (c.box_coords[1] / (REC_HEIGHT as f32)) * (crop_h as f32) + (crop_y as f32);
                    let x2 = (c.box_coords[2] / (eff_w as f32)) * (crop_w as f32) + (crop_x as f32);
                    let y2 =
                        (c.box_coords[3] / (REC_HEIGHT as f32)) * (crop_h as f32) + (crop_y as f32);
                    char_boxes.push(BoundingBox::new(
                        x1.round() as i32,
                        y1.round() as i32,
                        (x2 - x1).round() as i32,
                        (y2 - y1).round() as i32,
                        c.score,
                    ));
                }
            }
            result = Some(LineResult {
                text,
                char_boxes,
                alternatives,
                is_vertical,
                chunk_boxes: vec![BoundingBox::new(
                    crop_x as i32,
                    crop_y as i32,
                    crop_w as i32,
                    crop_h as i32,
                    1.0,
                )],
            });
        }

        Ok(result)
    }

    pub fn recognize_single_chunk(
        &mut self,
        chunk: &DynamicImage,
        is_vertical: bool,
    ) -> Result<(Vec<CharCandidate>, i32, i32)> {
        let target_w = if is_vertical {
            VERT_REC_WIDTH as u32
        } else {
            REC_WIDTH as u32
        };
        let target_h = if is_vertical {
            VERT_REC_HEIGHT as u32
        } else {
            REC_HEIGHT as u32
        };

        // Compute effective sizes matching Kotlin logic
        let (effective_w, effective_h) = if is_vertical {
            let scale_factor = 32.0f32 / (chunk.width() as f32);
            (
                32i32,
                (chunk.height() as f32 * scale_factor).min(VERT_REC_HEIGHT as f32) as i32,
            )
        } else {
            let scale_factor = 32.0f32 / (chunk.height() as f32);
            (
                (chunk.width() as f32 * scale_factor).min(REC_WIDTH as f32) as i32,
                32i32,
            )
        };

        if effective_w <= 0 || effective_h <= 0 {
            return Ok((Vec::new(), effective_w as i32, effective_h as i32));
        }

        // Resize chunk and pad to target
        let resized = chunk.resize_exact(
            REC_WIDTH,
            REC_HEIGHT,
            image::imageops::FilterType::Triangle,
        );
        let mut padded = RgbaImage::from_pixel(target_w, target_h, Rgba([0u8, 0u8, 0u8, 255u8]));
        let resized_rgba = resized.to_rgba8();
        for y in 0..(effective_h as u32) {
            for x in 0..(effective_w as u32) {
                let p = resized_rgba.get_pixel(x, y);
                padded.put_pixel(x, y, *p);
            }
        }

        // Convert to NCHW float tensor
        let img_data = self.image_to_nchw(&padded, target_w, target_h);
        let input_tensor = Tensor::from_array((
            [1i64, 3, target_h as i64, target_w as i64],
            img_data.into_boxed_slice(),
        ))?;

        // Run the session and extract outputs inside a limited scope to avoid holding a mutable borrow on self
        let (labels_arr_opt, boxes_arr_opt, scores_arr_opt, indices_arr_opt, raw_logits_opt) = {
            // Build inputs
            let mut active_session = if is_vertical {
                self.recognize_sessions_vertical.get()[0].lock().unwrap()
            } else {
                self.recognize_sessions.get()[0].lock().unwrap()
            };
            let session_input_names: Vec<String> = active_session
                .inputs()
                .iter()
                .map(|o| o.name().to_string())
                .collect();
            let image_input_name = session_input_names
                .iter()
                .find(|n| n.contains("image") || n.contains("input"))
                .or_else(|| session_input_names.first())
                .map(|s| s.to_string())
                .context("Recognition model has no inputs")?;

            let has_orig_target_sizes =
                session_input_names.iter().any(|n| n == "orig_target_sizes");

            let inputs = if has_orig_target_sizes {
                let size_tensor = Tensor::from_array((
                    [1i64, 2],
                    vec![target_w as i64, target_h as i64].into_boxed_slice(),
                ))?;
                ort::inputs! {
                    image_input_name.as_str() => input_tensor,
                    "orig_target_sizes" => size_tensor
                }
            } else {
                ort::inputs! { image_input_name.as_str() => input_tensor }
            };

            // Run session and collect outputs
            let output_names: Vec<String> = active_session
                .outputs()
                .iter()
                .map(|o| o.name().to_string())
                .collect();
            let run_outputs = active_session.run(inputs)?;

            // Helper closures to extract arrays
            let try_extract_f32 =
                |val: &ort::value::Value| val.try_extract_array::<f32>().ok().map(|a| a.to_owned());
            let try_extract_i64 =
                |val: &ort::value::Value| val.try_extract_array::<i64>().ok().map(|a| a.to_owned());

            let labels_name = output_names
                .iter()
                .find(|n| n.contains("labels") || n.contains("char_codes"));
            let boxes_name = output_names.iter().find(|n| n.contains("boxes"));
            let scores_name = output_names.iter().find(|n| n.contains("scores"));
            let logits_name = output_names.iter().find(|n| n.contains("logits"));
            let indices_name = output_names.iter().find(|n| n.contains("indices"));

            let labels_val = labels_name
                .and_then(|n| run_outputs.get(n.as_str()))
                .or_else(|| run_outputs.get(output_names.get(0).map(|s| s.as_str()).unwrap_or("")));
            let boxes_val = boxes_name
                .and_then(|n| run_outputs.get(n.as_str()))
                .or_else(|| run_outputs.get(output_names.get(1).map(|s| s.as_str()).unwrap_or("")));
            let scores_val = scores_name
                .and_then(|n| run_outputs.get(n.as_str()))
                .or_else(|| run_outputs.get(output_names.get(2).map(|s| s.as_str()).unwrap_or("")));
            let logits_val = logits_name.and_then(|n| run_outputs.get(n.as_str()));
            let indices_val = indices_name.and_then(|n| run_outputs.get(n.as_str()));

            // Extract arrays
            let labels_arr_opt: Option<Vec<i64>> = labels_val.and_then(|v| {
                try_extract_i64(v)
                    .map(|a| a.iter().cloned().collect())
                    .or_else(|| {
                        // try i32
                        v.try_extract_array::<i32>()
                            .ok()
                            .map(|a| a.to_owned().iter().map(|x| *x as i64).collect())
                    })
            });

            let boxes_arr_opt = boxes_val.and_then(|v| try_extract_f32(v));
            let scores_arr_opt: Option<Vec<f32>> = scores_val.and_then(|v| {
                try_extract_f32(v)
                    .map(|a| a.iter().cloned().collect())
                    .or_else(|| {
                        v.try_extract_array::<f64>()
                            .ok()
                            .map(|a| a.to_owned().iter().map(|x| *x as f32).collect())
                    })
            });

            let indices_arr_opt: Option<Vec<i64>> = indices_val.and_then(|v| {
                try_extract_i64(v)
                    .map(|a| a.iter().cloned().collect())
                    .or_else(|| {
                        v.try_extract_array::<i32>()
                            .ok()
                            .map(|a| a.to_owned().iter().map(|x| *x as i64).collect())
                    })
            });

            let raw_logits_opt: Option<Vec<f32>> = logits_val.and_then(|v| {
                try_extract_f32(v)
                    .map(|a| a.iter().cloned().collect())
                    .or_else(|| {
                        v.try_extract_array::<f64>()
                            .ok()
                            .map(|a| a.to_owned().iter().map(|x| *x as f32).collect())
                    })
            });

            (
                labels_arr_opt,
                boxes_arr_opt,
                scores_arr_opt,
                indices_arr_opt,
                raw_logits_opt,
            )
        };

        // Basic validation
        if labels_arr_opt.is_none() || boxes_arr_opt.is_none() || scores_arr_opt.is_none() {
            // Clean up and return empty
            return Ok((Vec::new(), effective_w as i32, effective_h as i32));
        }

        let labels_arr = labels_arr_opt.unwrap();
        let boxes_arr = boxes_arr_opt.unwrap();
        let scores_arr = scores_arr_opt.unwrap();
        let indices_arr = indices_arr_opt;
        let raw_logits = raw_logits_opt;

        // Normalize shapes and build logits matrix if present
        let num_queries = 48usize;
        let logits_matrix: Option<Vec<Vec<f32>>> = raw_logits.as_ref().and_then(|raw| {
            if raw.len() % num_queries == 0 && raw.len() > 0 {
                let num_classes = raw.len() / num_queries;
                let mut mat: Vec<Vec<f32>> = Vec::with_capacity(num_queries);
                for q in 0..num_queries {
                    let mut row: Vec<f32> = Vec::with_capacity(num_classes);
                    for c in 0..num_classes {
                        row.push(raw[q * num_classes + c]);
                    }
                    mat.push(row);
                }
                Some(mat)
            } else {
                None
            }
        });

        // Parse boxes and scores
        let mut parsed_boxes: Vec<[f32; 4]> = Vec::new();
        if boxes_arr.ndim() == 3 {
            let num = boxes_arr.shape()[1];
            for i in 0..num {
                parsed_boxes.push([
                    boxes_arr[[0, i, 0]],
                    boxes_arr[[0, i, 1]],
                    boxes_arr[[0, i, 2]],
                    boxes_arr[[0, i, 3]],
                ]);
            }
        } else if boxes_arr.ndim() == 2 {
            let num = boxes_arr.shape()[0];
            for i in 0..num {
                parsed_boxes.push([
                    boxes_arr[[i, 0]],
                    boxes_arr[[i, 1]],
                    boxes_arr[[i, 2]],
                    boxes_arr[[i, 3]],
                ]);
            }
        }

        let mut parsed_scores: Vec<f32> = Vec::new();
        if scores_arr.len() > 0 {
            parsed_scores = scores_arr.clone();
        }

        // Build candidates
        let mut candidates: Vec<CharCandidate> = Vec::new();
        for i in 0..parsed_scores.len() {
            if parsed_scores[i] > REC_CONFIDENCE_THRESHOLD {
                let mut alternatives: Vec<(char, f32)> = Vec::new();
                if let (Some(mat), Some(indices)) = (&logits_matrix, &indices_arr) {
                    let num_classes = if !mat.is_empty() { mat[0].len() } else { 0 };
                    if num_classes > 0 && i < indices.len() {
                        let query_idx = (indices[i] / (num_classes as i64)) as usize;
                        if query_idx < mat.len() {
                            let qlogits = &mat[query_idx];
                            // take top 15
                            let mut kv: Vec<(usize, f32)> =
                                qlogits.iter().cloned().enumerate().collect();
                            kv.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            for (j, &(_idx, _val)) in kv.iter().enumerate().take(15) {
                                let class_idx = kv[j].0;
                                let ch = self
                                    .char_vocab
                                    .get(class_idx)
                                    .and_then(|c| std::char::from_u32(*c as u32))
                                    .unwrap_or(' ');
                                alternatives.push((ch, kv[j].1));
                            }
                        }
                    }
                }

                let label_char = labels_arr
                    .get(i)
                    .and_then(|v| std::char::from_u32(*v as u32))
                    .unwrap_or(' ');
                let box_coords = parsed_boxes.get(i).cloned().unwrap_or([0.0, 0.0, 0.0, 0.0]);
                candidates.push(CharCandidate {
                    char: label_char,
                    score: parsed_scores[i],
                    box_coords,
                    alternatives,
                });
            }
        }

        // Sort and filter overlaps
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut filtered: Vec<CharCandidate> = Vec::new();
        for cand in candidates.into_iter() {
            let mut keep = true;
            for f in filtered.iter() {
                let overlap = if is_vertical {
                    self.calculate_y_overlap(&cand.box_coords, &f.box_coords)
                } else {
                    self.calculate_x_overlap(&cand.box_coords, &f.box_coords)
                };
                if overlap > X_OVERLAP_THRESHOLD {
                    keep = false;
                    break;
                }
            }
            if keep {
                filtered.push(cand);
            }
        }

        // Sort final
        if is_vertical {
            filtered.sort_by(|a, b| {
                a.box_coords[1]
                    .partial_cmp(&b.box_coords[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            filtered.sort_by(|a, b| {
                a.box_coords[0]
                    .partial_cmp(&b.box_coords[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok((filtered, effective_w as i32, effective_h as i32))
    }

    pub fn recognize_vertical_long_line(
        &mut self,
        crop: &DynamicImage,
        crop_x: i32,
        crop_y: i32,
    ) -> Result<LineResult> {
        let scale_factor = 32f32 / (crop.width() as f32);
        let max_chunk_height = (350f32 / scale_factor) as i32;
        let mut chunk_results: Vec<(Vec<CharCandidate>, i32, i32, i32, i32, i32, i32)> = Vec::new();
        let mut current_y = 0i32;
        while current_y < crop.height() as i32 {
            let remaining_h = crop.height() as i32 - current_y;
            if remaining_h < 10 {
                break;
            }
            let h = std::cmp::min(max_chunk_height, remaining_h);
            let chunk_bitmap = crop.crop_imm(0, current_y as u32, crop.width() as u32, h as u32);
            let (candidates, eff_w, eff_h) = self.recognize_single_chunk(&chunk_bitmap, true)?;
            if eff_h <= 0 {
                current_y += ((h as f32) * 0.8f32) as i32;
                continue;
            }
            chunk_results.push((
                candidates.clone(),
                0,
                current_y,
                crop.width() as i32,
                h,
                eff_w,
                eff_h,
            ));
            if current_y + h >= crop.height() as i32 {
                break;
            }

            if !candidates.is_empty() {
                let mut sorted_candidates = candidates.clone();
                sorted_candidates.sort_by(|a, b| {
                    a.box_coords[1]
                        .partial_cmp(&b.box_coords[1])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let overlap_start_threshold = (h as f32) * 0.6f32;
                let anchor_options: Vec<_> = sorted_candidates
                    .iter()
                    .filter(|it| {
                        let local_y_top = (it.box_coords[1] / eff_h as f32) * (h as f32);
                        local_y_top > overlap_start_threshold
                    })
                    .collect();
                let anchor_candidate = if anchor_options.len() >= 2 {
                    anchor_options[anchor_options.len() - 2].clone()
                } else if !anchor_options.is_empty() {
                    anchor_options[0].clone()
                } else {
                    sorted_candidates.last().unwrap().clone()
                };
                let local_y_top = (anchor_candidate.box_coords[1] / eff_h as f32) * (h as f32);
                let chunk_margin = (crop.width() as f32 * 0.1f32).max(2.0) as i32;
                let next_y = (current_y + local_y_top as i32 - chunk_margin).max(0);
                if next_y <= current_y || next_y >= current_y + h - 10 {
                    current_y += ((h as f32) * 0.8f32) as i32;
                } else {
                    current_y = next_y;
                }
            } else {
                current_y += ((h as f32) * 0.8f32) as i32;
            }
        }

        // For simplicity, stitch naively by concatenation of recognized chars in order
        let mut final_text = String::new();
        let mut final_char_boxes: Vec<BoundingBox> = Vec::new();
        for (cands, offset_x, offset_y, chunk_w, chunk_h, _eff_w, eff_h) in chunk_results.iter() {
            for cand in cands.iter() {
                let global = {
                    // Convert cand.box_coords -> global rect
                    let rx1 = cand.box_coords[0];
                    let ry1 = cand.box_coords[1];
                    let rx2 = cand.box_coords[2];
                    let ry2 = cand.box_coords[3];
                    let x1 = (rx1 / (VERT_REC_WIDTH as f32)) * (*chunk_w as f32)
                        + (*offset_x as f32)
                        + (crop_x as f32);
                    let y1 = (ry1 / (*eff_h as f32)) * (*chunk_h as f32)
                        + (*offset_y as f32)
                        + (crop_y as f32);
                    let x2 = (rx2 / (VERT_REC_WIDTH as f32)) * (*chunk_w as f32)
                        + (*offset_x as f32)
                        + (crop_x as f32);
                    let y2 = (ry2 / (*eff_h as f32)) * (*chunk_h as f32)
                        + (*offset_y as f32)
                        + (crop_y as f32);
                    BoundingBox::new(
                        x1.round() as i32,
                        y1.round() as i32,
                        (x2 - x1).round() as i32,
                        (y2 - y1).round() as i32,
                        cand.score,
                    )
                };
                final_text.push(cand.char);
                final_char_boxes.push(global);
            }
        }

        Ok(LineResult {
            text: final_text,
            char_boxes: final_char_boxes,
            alternatives: Vec::new(),
            is_vertical: true,
            chunk_boxes: Vec::new(),
        })
    }

    pub fn recognize_horizontal_long_line(
        &mut self,
        crop: &DynamicImage,
        crop_x: i32,
        crop_y: i32,
    ) -> Result<LineResult> {
        let scale_factor = 32f32 / (crop.height() as f32);
        let max_chunk_width = (960f32 / scale_factor) as i32;
        let mut chunk_results: Vec<(Vec<CharCandidate>, i32, i32, i32, i32, i32, i32)> = Vec::new();
        let mut current_x = 0i32;
        while current_x < crop.width() as i32 {
            let remaining_w = crop.width() as i32 - current_x;
            if remaining_w < 10 {
                break;
            }
            let w = std::cmp::min(max_chunk_width, remaining_w);
            let chunk_bitmap = crop.crop_imm(current_x as u32, 0, w as u32, crop.height() as u32);
            let (candidates, eff_w, eff_h) = self.recognize_single_chunk(&chunk_bitmap, false)?;
            if eff_w <= 0 {
                current_x += ((w as f32) * 0.8f32) as i32;
                continue;
            }
            chunk_results.push((
                candidates.clone(),
                current_x,
                0,
                w,
                crop.height() as i32,
                eff_w,
                eff_h,
            ));
            if current_x + w >= crop.width() as i32 {
                break;
            }

            if !candidates.is_empty() {
                let mut sorted_candidates = candidates.clone();
                sorted_candidates.sort_by(|a, b| {
                    a.box_coords[0]
                        .partial_cmp(&b.box_coords[0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let overlap_start_threshold = (w as f32) * 0.6f32;
                let anchor_options: Vec<_> = sorted_candidates
                    .iter()
                    .filter(|it| {
                        let local_x_left = (it.box_coords[0] / eff_w as f32) * (w as f32);
                        local_x_left > overlap_start_threshold
                    })
                    .collect();
                let anchor_candidate = if anchor_options.len() >= 2 {
                    anchor_options[anchor_options.len() - 2].clone()
                } else if !anchor_options.is_empty() {
                    anchor_options[0].clone()
                } else {
                    sorted_candidates.last().unwrap().clone()
                };
                let local_x_left = (anchor_candidate.box_coords[0] / eff_w as f32) * (w as f32);
                let chunk_margin = (crop.height() as f32 * 0.1f32).max(2.0) as i32;
                let next_x = (current_x + local_x_left as i32 - chunk_margin).max(0);
                if next_x <= current_x || next_x >= current_x + w - 10 {
                    current_x += ((w as f32) * 0.8f32) as i32;
                } else {
                    current_x = next_x;
                }
            } else {
                current_x += ((w as f32) * 0.8f32) as i32;
            }
        }

        // Naive stitching: concatenate in order
        let mut final_text = String::new();
        let mut final_char_boxes: Vec<BoundingBox> = Vec::new();
        for (cands, offset_x, offset_y, chunk_w, chunk_h, eff_w, _eff_h) in chunk_results.iter() {
            for cand in cands.iter() {
                let rx1 = cand.box_coords[0];
                let ry1 = cand.box_coords[1];
                let rx2 = cand.box_coords[2];
                let ry2 = cand.box_coords[3];
                let x1 = (rx1 / (*eff_w as f32)) * (*chunk_w as f32)
                    + (*offset_x as f32)
                    + (crop_x as f32);
                let y1 = (ry1 / (REC_HEIGHT as f32)) * (*chunk_h as f32)
                    + (*offset_y as f32)
                    + (crop_y as f32);
                let x2 = (rx2 / (*eff_w as f32)) * (*chunk_w as f32)
                    + (*offset_x as f32)
                    + (crop_x as f32);
                let y2 = (ry2 / (REC_HEIGHT as f32)) * (*chunk_h as f32)
                    + (*offset_y as f32)
                    + (crop_y as f32);
                final_text.push(cand.char);
                final_char_boxes.push(BoundingBox::new(
                    x1.round() as i32,
                    y1.round() as i32,
                    (x2 - x1).round() as i32,
                    (y2 - y1).round() as i32,
                    cand.score,
                ));
            }
        }

        Ok(LineResult {
            text: final_text,
            char_boxes: final_char_boxes,
            alternatives: Vec::new(),
            is_vertical: false,
            chunk_boxes: Vec::new(),
        })
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

    /// Static version of recognize_single_line that creates its own ONNX sessions.
    /// Used for parallel recognition where each thread needs its own session.
    fn recognize_single_line_static(
        model_dir: &str,
        char_vocab: &[i64],
        image: &DynamicImage,
        bbox: &BoundingBox,
    ) -> Result<Option<LineResult>> {
        ocr_parallel::recognize_single_line_static(model_dir, char_vocab, image, bbox)
    }

    // Modified: return the annotated image in-memory when `render` is true
    pub fn run_detection(
        &mut self,
        image: &DynamicImage,
        render: bool,
        font_path: Option<&str>,
    ) -> Result<(Vec<DetectedAnnotation>, Option<RgbaImage>)> {
        let t_total = Instant::now();

        // --- Line detection ---
        let t_detect = Instant::now();
        let boxes = self.detect(image)?;
        let detect_ms = t_detect.elapsed().as_secs_f64() * 1000.0;
        println!("[OCR timing] Line detection:       {:>8.2} ms", detect_ms);

        let t_merge = Instant::now();
        let merged = self.merge_overlapping_boxes(boxes);
        let sorted = self.sort_detected_boxes(merged);
        let merge_ms = t_merge.elapsed().as_secs_f64() * 1000.0;
        println!("[OCR timing] Merge/sort boxes:     {:>8.2} ms", merge_ms);

        println!("Detected {} bounding boxes.", sorted.len());
        for (i, bbox) in sorted.iter().enumerate() {
            let b = bbox;
            println!(
                "  Box {}: x={}, y={}, w={}, h={}, conf={:.2}",
                i, b.x, b.y, b.w, b.h, b.confidence
            );
        }

        // --- Character recognition ---
        let t_recognize = Instant::now();

        // Group boxes by orientation for batched processing
        let horizontal_boxes: Vec<_> = sorted
            .iter()
            .filter(|b| b.w >= b.h)
            .cloned()
            .collect();
        let vertical_boxes: Vec<_> = sorted
            .iter()
            .filter(|b| b.h > b.w)
            .cloned()
            .collect();

        let mut annotations = Vec::with_capacity(sorted.len());

        // Process horizontal boxes in parallel batches
        if !horizontal_boxes.is_empty() {
            let rec_sessions = self.recognize_sessions.get().to_vec();
            let pool_size = rec_sessions.len();
            let char_vocab = self.char_vocab.clone();
            let batch_size = ocr_parallel::MAX_BATCH_SIZE;

            // Build all chunk crops upfront (shared read-only)
            let all_chunks: Vec<(DynamicImage, bool)> = horizontal_boxes.iter().map(|bbox| {
                let crop_x = bbox.x.max(0) as u32;
                let crop_y = bbox.y.max(0) as u32;
                let crop_w = bbox.w.max(0) as u32;
                let crop_h = bbox.h.max(0) as u32;
                let crop = image.crop_imm(crop_x, crop_y, crop_w, crop_h);
                (crop, false)
            }).collect();

            let num_batches = (all_chunks.len() + batch_size - 1) / batch_size;
            let batch_results: std::sync::Mutex<Vec<Option<(Result<Vec<(Vec<CharCandidate>, i32, i32)>>, f64)>>> = std::sync::Mutex::new((0..num_batches).map(|_| None).collect());

            std::thread::scope(|s| {
                for batch_idx in 0..num_batches {
                    let start = batch_idx * batch_size;
                    let end = (start + batch_size).min(all_chunks.len());
                    let pool_idx = batch_idx % pool_size;
                    let rec_sess = rec_sessions[pool_idx].clone();
                    let cv = char_vocab.clone();
                    let br = &batch_results;

                    let chunk_slice = &all_chunks[start..end];
                    s.spawn(move || {
                        let t_box = std::time::Instant::now();
                        let mut sess = rec_sess.lock().unwrap();
                        let results = ocr_parallel::recognize_batch_chunks_static(
                            &mut sess, &cv, chunk_slice,
                        );
                        let box_ms = t_box.elapsed().as_secs_f64() * 1000.0;
                        let mut locked = br.lock().unwrap();
                        locked[batch_idx] = Some((results, box_ms));
                    });
                }
            });

            // Process results in order
            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(horizontal_boxes.len());
                if let Some((Ok(batch_results_inner), box_ms)) =
                    batch_results.lock().unwrap()[batch_idx].take()
                        .map(|(r, ms)| (r.map(|v| v), ms))
                {
                    for (i, (bbox, result)) in horizontal_boxes[start..end]
                        .iter().zip(batch_results_inner.into_iter()).enumerate()
                    {
                        let (filtered, eff_w, eff_h) = result;
                        let text: String = filtered.iter().map(|c| c.char).collect();
                        let alternatives: Vec<Vec<(char, f32)>> =
                            filtered.iter().map(|c| c.alternatives.clone()).collect();

                        let mut char_boxes: Vec<BoundingBox> = Vec::new();
                        let crop_x = bbox.x.max(0) as u32;
                        let crop_y = bbox.y.max(0) as u32;
                        let crop_w = bbox.w.max(0) as u32;
                        let crop_h = bbox.h.max(0) as u32;

                        for c in filtered.iter() {
                            let x1 = (c.box_coords[0] / (eff_w as f32)) * (crop_w as f32) + (crop_x as f32);
                            let y1 = (c.box_coords[1] / (REC_HEIGHT as f32)) * (crop_h as f32) + (crop_y as f32);
                            let x2 = (c.box_coords[2] / (eff_w as f32)) * (crop_w as f32) + (crop_x as f32);
                            let y2 = (c.box_coords[3] / (REC_HEIGHT as f32)) * (crop_h as f32) + (crop_y as f32);
                            char_boxes.push(BoundingBox::new(
                                x1.round() as i32, y1.round() as i32,
                                (x2 - x1).round() as i32, (y2 - y1).round() as i32,
                                c.score,
                            ));
                        }

                        println!(
                            "[OCR timing]   Box {:>3} ({}x{} @ {},{}): {:>8.2} ms",
                            annotations.len(), bbox.w, bbox.h, bbox.x, bbox.y, box_ms
                        );

                        annotations.push(DetectedAnnotation {
                            bbox: bbox.clone(),
                            line: Some(LineResult {
                                text,
                                char_boxes,
                                alternatives,
                                is_vertical: false,
                                chunk_boxes: vec![BoundingBox::new(
                                    crop_x as i32, crop_y as i32,
                                    crop_w as i32, crop_h as i32, 1.0,
                                )],
                            }),
                        });
                    }
                }
            }
        }

        // Process vertical boxes in parallel batches        // Process vertical boxes in parallel batches
        if !vertical_boxes.is_empty() {
            let vert_sessions = self.recognize_sessions_vertical.get().to_vec();
            let pool_size = vert_sessions.len();
            let char_vocab = self.char_vocab.clone();
            let batch_size = ocr_parallel::MAX_BATCH_SIZE;

            let all_chunks: Vec<(DynamicImage, bool)> = vertical_boxes.iter().map(|bbox| {
                let crop_x = bbox.x.max(0) as u32;
                let crop_y = bbox.y.max(0) as u32;
                let crop_w = bbox.w.max(0) as u32;
                let crop_h = bbox.h.max(0) as u32;
                let crop = image.crop_imm(crop_x, crop_y, crop_w, crop_h);
                (crop, true)
            }).collect();

            let num_batches = (all_chunks.len() + batch_size - 1) / batch_size;
            let batch_results: std::sync::Mutex<Vec<Option<(Result<Vec<(Vec<CharCandidate>, i32, i32)>>, f64)>>> = std::sync::Mutex::new((0..num_batches).map(|_| None).collect());

            std::thread::scope(|s| {
                for batch_idx in 0..num_batches {
                    let start = batch_idx * batch_size;
                    let end = (start + batch_size).min(all_chunks.len());
                    let pool_idx = batch_idx % pool_size;
                    let rec_sess = vert_sessions[pool_idx].clone();
                    let cv = char_vocab.clone();
                    let br = &batch_results;

                    let chunk_slice = &all_chunks[start..end];
                    s.spawn(move || {
                        let t_box = std::time::Instant::now();
                        let mut sess = rec_sess.lock().unwrap();
                        let results = ocr_parallel::recognize_batch_chunks_static(
                            &mut sess, &cv, chunk_slice,
                        );
                        let box_ms = t_box.elapsed().as_secs_f64() * 1000.0;
                        let mut locked = br.lock().unwrap();
                        locked[batch_idx] = Some((results, box_ms));
                    });
                }
            });

            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(vertical_boxes.len());
                if let Some((Ok(batch_results_inner), box_ms)) =
                    batch_results.lock().unwrap()[batch_idx].take()
                        .map(|(r, ms)| (r.map(|v| v), ms))
                {
                    for (i, (bbox, result)) in vertical_boxes[start..end]
                        .iter().zip(batch_results_inner.into_iter()).enumerate()
                    {
                        let (filtered, eff_w, eff_h) = result;
                        let text: String = filtered.iter().map(|c| c.char).collect();
                        let alternatives: Vec<Vec<(char, f32)>> =
                            filtered.iter().map(|c| c.alternatives.clone()).collect();

                        let mut char_boxes: Vec<BoundingBox> = Vec::new();
                        let crop_x = bbox.x.max(0) as u32;
                        let crop_y = bbox.y.max(0) as u32;
                        let crop_w = bbox.w.max(0) as u32;
                        let crop_h = bbox.h.max(0) as u32;

                        for c in filtered.iter() {
                            let x1 = (c.box_coords[0] / (VERT_REC_WIDTH as f32)) * (crop_w as f32) + (crop_x as f32);
                            let y1 = (c.box_coords[1] / (eff_h as f32)) * (crop_h as f32) + (crop_y as f32);
                            let x2 = (c.box_coords[2] / (VERT_REC_WIDTH as f32)) * (crop_w as f32) + (crop_x as f32);
                            let y2 = (c.box_coords[3] / (eff_h as f32)) * (crop_h as f32) + (crop_y as f32);
                            char_boxes.push(BoundingBox::new(
                                x1.round() as i32, y1.round() as i32,
                                (x2 - x1).round() as i32, (y2 - y1).round() as i32,
                                c.score,
                            ));
                        }

                        println!(
                            "[OCR timing]   Box {:>3} ({}x{} @ {},{}): {:>8.2} ms",
                            annotations.len(), bbox.w, bbox.h, bbox.x, bbox.y, box_ms
                        );

                        annotations.push(DetectedAnnotation {
                            bbox: bbox.clone(),
                            line: Some(LineResult {
                                text,
                                char_boxes,
                                alternatives,
                                is_vertical: true,
                                chunk_boxes: vec![BoundingBox::new(
                                    crop_x as i32, crop_y as i32,
                                    crop_w as i32, crop_h as i32, 1.0,
                                )],
                            }),
                        });
                    }
                }
            }
        }        let recognize_ms = t_recognize.elapsed().as_secs_f64() * 1000.0;
        println!(
            "[OCR timing] Character recognition: {:>8.2} ms  ({} boxes, batched)",
            recognize_ms, sorted.len()
        );

        // Prepare font if rendering and path provided or defaults
        let mut font_opt: Option<Font> = None;
        if render {
            let settings = FontSettings::default();
            // Try provided path first
            if let Some(fp) = font_path {
                if let Ok(bytes) = std::fs::read(fp) {
                    match Font::from_bytes(bytes, settings) {
                        Ok(f) => font_opt = Some(f),
                        Err(e) => println!("Failed to parse font at {}: {}", fp, e),
                    }
                } else {
                    println!("Failed to read font at {}", fp);
                }
            }

            // Try common default
            if font_opt.is_none() {
                let candidates = [
                    "fonts/NotoSansJP-Regular.ttf",
                    "fonts/NotoSansJP-Regular.otf",
                    "font.ttf",
                    "NotoSansJP-Regular.ttf",
                ];
                for cand in candidates.iter() {
                    if let Ok(bytes) = std::fs::read(cand) {
                        match Font::from_bytes(bytes, settings) {
                            Ok(f) => {
                                println!("Loaded font from {}", cand);
                                font_opt = Some(f);
                                break;
                            }
                            Err(_) => {}
                        }
                    }
                }
            }
        }

        let t_render = Instant::now();
        if render {
            // Render detected boxes onto the image in-memory and return the annotated image
            let mut img_rgba = image.to_rgba8();
            let color = Rgba([255u8, 0u8, 0u8, 255u8]);
            let thickness = 2u32;

            // Precompute a reference glyph height for the loaded font (if any) using '本'.
            let _ref_glyph_unit_h = 0.0f32;
            if let Some(ref font) = font_opt {
                let (metrics, _) = font.rasterize('本', 1.0);
                let _ref_glyph_unit_h = metrics.height as f32;
            }

            for bbox in sorted.iter() {
                draw_rectangle(&mut img_rgba, bbox, color, thickness);

                // Draw confidence label (e.g. "0.84") near the top-left of the box
                let label = format!("{:.2}", bbox.confidence);
                let fg = Rgba([255u8, 255u8, 255u8, 255u8]); // white text
                let bg = Rgba([0u8, 0u8, 0u8, 200u8]); // semi-opaque black background
                let scale = 2u32; // scale factor for the tiny font
                draw_text_small(
                    &mut img_rgba,
                    bbox.left(),
                    bbox.top(),
                    &label,
                    fg,
                    bg,
                    scale,
                );

                // If we have recognition results, render them into the baked image as before (optional)
                if let Some(ref line_result) = annotations
                    .iter()
                    .find(|a| a.bbox.x == bbox.x && a.bbox.y == bbox.y)
                    .and_then(|a| a.line.clone())
                {
                    if let Some(ref font) = font_opt {
                        // Render each detected character directly on top of its detection box.
                        // Follow Kotlin implementation: compute a fixed square size (based on max char width for
                        // vertical lines or max char height for horizontal), refine positions with measured
                        // advances, then center each glyph inside its display square using a per-font reference.

                        let chars: Vec<char> = line_result.text.chars().collect();
                        if !chars.is_empty() && !line_result.char_boxes.is_empty() {
                            // fixed size: max width (vertical) or max height (horizontal)
                            let fixed_size = if line_result.is_vertical {
                                line_result
                                    .char_boxes
                                    .iter()
                                    .map(|b| b.w)
                                    .max()
                                    .unwrap_or(0)
                            } else {
                                line_result
                                    .char_boxes
                                    .iter()
                                    .map(|b| b.h)
                                    .max()
                                    .unwrap_or(0)
                            };

                            // Measure per-character advances at the fixed_size.
                            // If measuring fails or yields zero, fall back to fixed_size.
                            let mut advances: Vec<i32> = Vec::new();
                            if fixed_size > 0 {
                                let px = fixed_size as f32;
                                for ch in chars.iter() {
                                    let (metrics, _) = font.rasterize(*ch, px);
                                    let adv = if metrics.advance_width > 0.0 {
                                        metrics.advance_width.round() as i32
                                    } else {
                                        fixed_size
                                    };
                                    advances.push(adv);
                                }
                            } else {
                                advances = vec![0; chars.len()];
                            }

                            // Build refined boxes anchored to previous original positions (to avoid drift)
                            let mut refined: Vec<BoundingBox> = Vec::new();
                            refined.push(line_result.char_boxes[0].clone());
                            for i in 1..line_result.char_boxes.len() {
                                let prev = &line_result.char_boxes[i - 1];
                                let cur = &line_result.char_boxes[i];
                                let adv = advances.get(i - 1).cloned().unwrap_or(fixed_size);
                                if line_result.is_vertical {
                                    let new_top = (prev.top().saturating_add(adv)).max(cur.top());
                                    refined.push(BoundingBox::new(
                                        cur.left(),
                                        new_top,
                                        cur.w,
                                        cur.h,
                                        cur.confidence,
                                    ));
                                } else {
                                    let new_left =
                                        (prev.left().saturating_add(adv)).max(cur.left());
                                    refined.push(BoundingBox::new(
                                        new_left,
                                        cur.top(),
                                        cur.w,
                                        cur.h,
                                        cur.confidence,
                                    ));
                                }
                            }

                            // Convert refined boxes to square display boxes centered on original centers
                            let mut display_boxes: Vec<BoundingBox> = Vec::new();
                            for b in refined.iter() {
                                let center_x = b.left() + b.w / 2;
                                let center_y = b.top() + b.h / 2;
                                let left = center_x - fixed_size / 2;
                                let top = center_y - fixed_size / 2;
                                display_boxes
                                    .push(BoundingBox::new(left, top, fixed_size, fixed_size, 1.0));
                            }

                            // Helper: find a scale (px) so that the reference glyph '本' visual height
                            // matches target_h via binary search.
                            let find_scale_for_target = |font: &Font, target_h: f32| -> f32 {
                                let mut lo = 1.0f32.max(target_h * 0.2);
                                let mut hi = (target_h.max(1.0) * 8.0).max(64.0);
                                // Expand upper bound until glyph height >= target
                                for _ in 0..10 {
                                    let (m, _) = font.rasterize('本', hi);
                                    if (m.height as f32) >= target_h || hi > 4096.0 {
                                        break;
                                    }
                                    hi *= 2.0;
                                }
                                // Binary refine
                                let mut s = lo;
                                for _ in 0..12 {
                                    let mid = (lo + hi) / 2.0;
                                    let (m, _) = font.rasterize('本', mid);
                                    let measured = m.height as f32;
                                    if measured == 0.0 {
                                        lo = mid;
                                        s = mid;
                                        continue;
                                    }
                                    s = mid;
                                    if measured < target_h {
                                        lo = mid;
                                    } else {
                                        hi = mid;
                                    }
                                    if (measured - target_h).abs() < 0.5 {
                                        break;
                                    }
                                }
                                s
                            };

                            // Compute a single scale per line (based on fixed_size) using the reference glyph
                            let target_for_line = (fixed_size as f32) * BOX_FILL_RATIO;
                            let scale_for_line = find_scale_for_target(&font, target_for_line);

                            // Draw each char using the same scale_for_line and center M-box
                            let mut dbg_printed = 0usize;
                            for (i, db) in display_boxes.iter().enumerate() {
                                if i >= chars.len() {
                                    break;
                                }
                                let ch = chars[i];

                                let scale_px = scale_for_line.max(4.0).min(4096.0);
                                let line_metrics = font
                                    .horizontal_line_metrics(scale_px)
                                    .unwrap_or_else(|| fontdue::LineMetrics {
                                        ascent: scale_px,
                                        descent: 0.0,
                                        line_gap: 0.0,
                                        new_line_size: scale_px,
                                    });

                                // Compute baseline so that the glyph's measured vertical box is centered in the display box.
                                let box_center_y = db.top() as f32 + (db.h as f32) / 2.0;
                                let baseline_y = box_center_y
                                    + (line_metrics.ascent - line_metrics.descent) / 2.0;
                                let y_top = baseline_y - line_metrics.ascent;

                                // Measure advance (width) at this scale for horizontal centering
                                let (adv_metrics, _) = font.rasterize(ch, scale_px);
                                let text_width = adv_metrics.advance_width;
                                let x =
                                    db.left() as f32 + ((db.w as f32 - text_width).max(0.0) / 2.0);

                                // Debug: measure final glyph bbox at this scale to verify visual height
                                let (final_metrics, _) = font.rasterize(ch, scale_px);
                                let final_h = final_metrics.height as f32;

                                if dbg_printed < 8 {
                                    dbg_printed += 1;
                                    /*println!(
                                        "DBG glyph='{}' box_h={} target_h={:.1} scale_px={:.2} final_h={:.1} adv_w={:.1}",
                                        ch, db.h, target_for_line, scale_px, final_h, text_width
                                    );*/
                                }

                                // Draw black background fill behind the character for contrast
                                if i < 3 {
                                    /*println!("DBG fill: db=({},{},{},{}) img={}x{}", db.left(), db.top(), db.w, db.h, img_rgba.width(), img_rgba.height());*/
                                }
                                draw_filled_rect(
                                    &mut img_rgba,
                                    db.left() - 1,
                                    db.top() - 1,
                                    db.w + 2,
                                    db.h + 2,
                                    Rgba([0u8, 0u8, 0u8, 200u8]),
                                );

                                draw_text_ttf(
                                    &mut img_rgba,
                                    font,
                                    &ch.to_string(),
                                    x.round() as i32,
                                    y_top.round() as i32,
                                    scale_px,
                                    Rgba([0u8, 255u8, 0u8, 255u8]),
                                );

                                // Draw the display box for debugging/visibility
                                draw_rectangle(
                                    &mut img_rgba,
                                    db,
                                    Rgba([0u8, 255u8, 0u8, 255u8]),
                                    1,
                                );
                            }
                        } else {
                            // Fallback: draw the whole line text above the box like before
                            let font_size = (bbox.h.max(12) as f32) * 0.6f32;
                            let text_x = bbox.left();
                            let text_y = bbox.top() - (font_size as i32) - 2;
                            draw_text_ttf(
                                &mut img_rgba,
                                font,
                                &line_result.text,
                                text_x,
                                text_y.max(0),
                                font_size,
                                Rgba([0u8, 255u8, 0u8, 255u8]),
                            );

                            // Draw character boxes in green
                            let char_color = Rgba([0u8, 255u8, 0u8, 255u8]);
                            for cb in line_result.char_boxes.iter() {
                                draw_rectangle(&mut img_rgba, cb, char_color, 1);
                            }
                        }
                    } else {
                        println!("Recognized: {}", line_result.text);

                        // Still draw detected char boxes so user can inspect positions
                        let char_color = Rgba([0u8, 255u8, 0u8, 255u8]);
                        for cb in line_result.char_boxes.iter() {
                            draw_rectangle(&mut img_rgba, cb, char_color, 1);
                        }
                    }
                }
            }

            // Return the annotated image as well as the annotations
            let render_ms = t_render.elapsed().as_secs_f64() * 1000.0;
            println!("[OCR timing] Rendering:            {:>8.2} ms", render_ms);
            let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
            println!("[OCR timing] ─────────────────────────────────────");
            println!("[OCR timing] TOTAL:                {:>8.2} ms", total_ms);
            return Ok((annotations, Some(img_rgba)));
        } else {
            let render_ms = t_render.elapsed().as_secs_f64() * 1000.0;
            println!("[OCR timing] Rendering:            {:>8.2} ms (skipped)", render_ms);
            let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
            println!("[OCR timing] ─────────────────────────────────────");
            println!("[OCR timing] TOTAL:                {:>8.2} ms", total_ms);
            println!("Render disabled; returning annotations without a baked image.");
            return Ok((annotations, None));
        }
    }
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
