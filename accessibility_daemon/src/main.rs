use anyhow::{Result, Context};

use ort::session::Session;
use ort::value::Tensor;
use image::DynamicImage;
use std::path::Path;
use image::GenericImageView;

// Constants matching the Kotlin implementation
const DETECT_WIDTH: u32 = 960;
const DETECT_HEIGHT: u32 = 544;
const REC_WIDTH: u32 = 960;
const REC_HEIGHT: u32 = 32;
const VERT_REC_WIDTH: u32 = 32;
const VERT_REC_HEIGHT: u32 = 480;
const REC_CONFIDENCE_THRESHOLD: f32 = 0.5;
const X_OVERLAP_THRESHOLD: f32 = 0.3;

// Simulated swapped pairs for text correction
const MEIKI_SWAPPED_PAIRS: &[(&str, &str)] = &[
    ("は", "ば"),
    ("ひ", "び"),
    ("ふ", "ぶ"),
    ("へ", "べ"),
    ("ほ", "ぼ"),
];

// Represents a bounding box with coordinates and a confidence score.
#[derive(Debug, Clone)]
struct BoundingBox {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    confidence: f32,
}

impl BoundingBox {
    fn new(x: i32, y: i32, w: i32, h: i32, confidence: f32) -> Self {
        BoundingBox { x, y, w, h, confidence }
    }

    fn left(&self) -> i32 { self.x }
    fn top(&self) -> i32 { self.y }
    fn right(&self) -> i32 { self.x + self.w }
    fn bottom(&self) -> i32 { self.y + self.h }

    fn area(&self) -> i32 {
        self.w * self.h
    }
}

struct OcrEngine {
    detect_session: Session,
    recognize_session: Session,
    recognize_session_vertical: Session,
    char_vocab: Vec<i64>,
}

impl OcrEngine {
    fn new(model_dir: &str) -> Result<Self> {
        let model_path = Path::new(model_dir);

        let detect_session = Session::builder()?.commit_from_file(model_path.join("meiki.text.detect.v0.1.960x544.onnx"))?;
        let recognize_session = Session::builder()?.commit_from_file(model_path.join("meiki.text.rec.v0.960x32.with_logits.onnx"))?;
        let recognize_session_vertical = Session::builder()?.commit_from_file(model_path.join("meiki.text.rec.v0.vertical.32x480.with_logits.onnx"))?;

        // Load character vocabulary
        let vocab_path = model_path.join("char_vocab.json");
        let vocab_json: Vec<i64> = match std::fs::read_to_string(&vocab_path) {
            Ok(content) => serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse char_vocab.json"))?,
            Err(_) => Vec::new(),
        };

        Ok(OcrEngine {
            detect_session,
            recognize_session,
            recognize_session_vertical,
            char_vocab: vocab_json,
        })
    }

    fn is_ready(&self) -> bool {
        !self.char_vocab.is_empty()
    }

    /// Detects bounding boxes in the image using the detection model.
    fn detect(&mut self, image: &DynamicImage) -> Result<Vec<BoundingBox>> {
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

        // In ort rc.12, Tensor::from_array takes (shape, data) where shape is a tuple of i64 dims
        // and data is a boxed slice.
        let input_tensor = Tensor::from_array((
            [1i64, 3, DETECT_HEIGHT as i64, DETECT_WIDTH as i64],
            img_data.into_boxed_slice(),
        ))?;

        // 3. Build inputs map using the named map form of ort::inputs!
        // Collect input names to avoid holding borrows into `self.detect_session` across a mutable run() call.
        let session_input_names: Vec<String> = self.detect_session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let image_input_name = session_input_names
            .iter()
            .find(|n| n.contains("image") || n.contains("input"))
            .or_else(|| session_input_names.first())
            .context("Detection model has no inputs")?;

        // Check if the model expects orig_target_sizes as an additional input
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

        // Collect output names before calling run() so we don't hold borrows from the session across the mutable call.
        let output_names: Vec<String> = self.detect_session
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

        // 4. Run the detection model and extract arrays. Keep run_outputs scoped so it drops before we borrow `self` again.
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





        // boxes shape: [batch, num_boxes, 4] or [num_boxes, 4]
        // scores shape: [batch, num_boxes] or [num_boxes]
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

    fn merge_overlapping_boxes(&self, boxes: Vec<BoundingBox>) -> Vec<BoundingBox> {
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

    fn should_merge_boxes(&self, a: &BoundingBox, b: &BoundingBox) -> bool {
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

    fn sort_detected_boxes(&self, mut boxes: Vec<BoundingBox>) -> Vec<BoundingBox> {
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

    fn run_detection(&mut self, image: &DynamicImage) -> Result<()> {
        let boxes = self.detect(image)?;
        let merged = self.merge_overlapping_boxes(boxes);
        let sorted = self.sort_detected_boxes(merged);

        println!("Detected {} bounding boxes.", sorted.len());
        for (i, bbox) in sorted.iter().enumerate() {
                    let b = bbox;
            println!("  Box {}: x={}, y={}, w={}, h={}, conf={:.2}",
                     i, b.x, b.y, b.w, b.h, b.confidence);
        }

        // TODO: Step 4: Crop sub-images based on bounding boxes.
        // TODO: Step 5: Feed cropped images into the recognition model.
        Ok(())
    }
}

fn main() -> Result<()> {
    env_logger::init();
    println!("Accessibility Daemon Starting...");

    let mut engine = OcrEngine::new("./models")?;
    println!("Models loaded successfully.");
    println!("Character vocabulary loaded: {} chars", engine.char_vocab.len());

    let test_image_path = "test_image.png";
    let image = image::open(test_image_path)
        .context(format!("Failed to open test image: {}", test_image_path))?;
    println!("Test image loaded successfully: {} ({}x{})", test_image_path, image.width(), image.height());

    engine.run_detection(&image)?;

    Ok(())
}
