//! Burn WGPU-based OCR engine.
//! 
//! This module provides a drop-in replacement for the ORT-based OcrEngine,
//! using the Burn framework with the WGPU compute backend.
//!
//! Key changes from ORT version:
//! - Uses Lanczos3 resize filter (matches PIL BILINEAR) instead of Triangle
//! - Loads model weights from .bpk files via BurnpackStore
//! - Runs inference on GPU via WGPU compute shaders
//! - Uses ConvStrategy::Direct and other precision fixes from burn/ submodule

use anyhow::{Context, Result};
use burn::backend::Wgpu;
use burn::tensor::{Device, Tensor, TensorData, Int};
use burn::module::Module;
use image::{DynamicImage, GenericImageView};
use std::path::Path;
use std::time::Instant;

use crate::models::*;
// Only import detection model for now — recognition models gated behind feature flag
use crate::models_burn::meiki_text_detect::Model as DetectModel;
#[cfg(feature = "burn-recognition")]
use crate::models_burn::meiki_text_rec_horizontal::Model as RecHModel;
#[cfg(feature = "burn-recognition")]
use crate::models_burn::meiki_text_rec_vertical::Model as RecVModel;

type Backend = Wgpu;

const DETECT_WIDTH: u32 = 960;
const DETECT_HEIGHT: u32 = 544;
const REC_WIDTH: u32 = 960;
const REC_HEIGHT: u32 = 32;
const VERT_REC_WIDTH: u32 = 32;
const VERT_REC_HEIGHT: u32 = 480;
const REC_CONFIDENCE_THRESHOLD: f32 = 0.1;
const X_OVERLAP_THRESHOLD: f32 = 0.3;
const BOX_FILL_RATIO: f32 = 0.9;

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

pub struct OcrEngine {
    device: Device,
    model_dir: String,
    pub char_vocab: Vec<i64>,
}

impl OcrEngine {
    pub fn new(model_dir: &str) -> Result<Self> {
        let model_path = Path::new(model_dir);
        let device: Device = Default::default();

        println!("Loading Burn WGPU OCR engine…");

        // Load character vocabulary
        let vocab_path = model_path.join("char_vocab.json");
        let vocab_json: Vec<i64> = match std::fs::read_to_string(&vocab_path) {
            Ok(content) => serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse char_vocab.json"))?,
            Err(_) => Vec::new(),
        };

        println!("Burn WGPU OCR engine ready. Vocab size: {}", vocab_json.len());

        Ok(OcrEngine {
            device,
            model_dir: model_dir.to_string(),
            char_vocab: vocab_json,
        })
    }

    pub fn is_ready(&self) -> bool {
        !self.char_vocab.is_empty()
    }

    fn image_to_nchw(&self, img: &DynamicImage) -> Vec<f32> {
        let w = img.width() as usize;
        let h = img.height() as usize;
        let mut data = vec![0.0f32; 3 * h * w];
        for (x, y, pixel) in img.pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let yu = y as usize;
            let xu = x as usize;
            data[0 * h * w + yu * w + xu] = r;
            data[1 * h * w + yu * w + xu] = g;
            data[2 * h * w + yu * w + xu] = b;
        }
        data
    }

    /// Detect bounding boxes using the Burn WGPU detection model.
    pub fn detect(&self, image: &DynamicImage) -> Result<Vec<BoundingBox>> {
        let orig_w = image.width() as i32;
        let orig_h = image.height() as i32;

        // 1. Resize with Lanczos3 (matches ORT/PIL BILINEAR)
        // FIX: Use Triangle filter which matches PIL's BILINEAR (2x2 linear interpolation).
        // Lanczos3 (4x4 sinc) produces slightly different pixel values that amplify through
        // 102 conv layers and cause ~5-9 missed detections vs ORT.
        let resized = image.resize_exact(
            DETECT_WIDTH,
            DETECT_HEIGHT,
            image::imageops::FilterType::Triangle,
        );

        // 2. Convert to NCHW float tensor
        let img_data = self.image_to_nchw(&resized);
        let input = Tensor::<4>::from_data(
            TensorData::new(img_data, [1, 3, DETECT_HEIGHT as usize, DETECT_WIDTH as usize]),
            &self.device,
        );

        eprintln!("DEBUG: input shape: {:?}", input.shape());

        // 3. Load detection model
        let model_path = Path::new(&self.model_dir).join("meiki.text.detect.v0.1.bpk");
        let model = DetectModel::from_file(&model_path, &self.device);

        // Build orig_target_sizes tensor
        let orig_sizes = Tensor::<2, burn::tensor::Int>::from_ints(
            [[orig_w as i64, orig_h as i64]],
            &self.device,
        );

        // 4. Run the full model
        let (labels, boxes, scores) = model.forward(input, orig_sizes);

        eprintln!("DEBUG: boxes shape: {:?}", boxes.shape());
        eprintln!("DEBUG: scores shape: {:?}", scores.shape());
        eprintln!("DEBUG: labels shape: {:?}", labels.shape());

        // 5. Extract output data
        let boxes_data: Vec<f32> = boxes.into_data().to_vec().unwrap();
        let scores_data: Vec<f32> = scores.into_data().to_vec().unwrap();

        // 6. Parse boxes and scores
        let num_boxes = boxes_data.len() / 4;
        let mut detected_boxes = Vec::new();

        for i in 0..num_boxes {
            let score = if i < scores_data.len() { scores_data[i] } else { 0.0 };
            if score <= 0.4 { continue; }

            let left = boxes_data[i * 4] as i32;
            let top = boxes_data[i * 4 + 1] as i32;
            let right = boxes_data[i * 4 + 2] as i32;
            let bottom = boxes_data[i * 4 + 3] as i32;

            let left = left.max(0);
            let mut top = top.max(0);
            let mut right = right.min(orig_w);
            let bottom = bottom.min(orig_h);

            if bottom - top > right - left {
                let v_margin = ((right - left) as f32 * 0.1).max(2.0) as i32;
                let h_margin = ((right - left) as f32 * 0.05).max(1.0) as i32;
                top = (top - v_margin).max(0);
                right = (right + h_margin).min(orig_w);
            } else {
                let h_margin = ((bottom - top) as f32 * 0.1).max(4.0) as i32;
                right = (right + h_margin).min(orig_w);
            }

            detected_boxes.push(BoundingBox::new(
                left, top, right - left, bottom - top, score,
            ));
        }

        // DISABLED: merge_overlapping_boxes for raw comparison
        // let merged = self.merge_overlapping_boxes(detected_boxes);
        // let sorted = self.sort_detected_boxes(merged);
        // Ok(sorted)
        let sorted = self.sort_detected_boxes(detected_boxes);
        Ok(sorted)
    }

    pub fn merge_overlapping_boxes(&self, boxes: Vec<BoundingBox>) -> Vec<BoundingBox> {
        // Same implementation as ORT version
        if boxes.is_empty() { return boxes; }
        let mut result = Vec::new();
        let mut used = vec![false; boxes.len()];
        for i in 0..boxes.len() {
            if used[i] { continue; }
            let mut current = boxes[i].clone();
            for j in (i + 1)..boxes.len() {
                if used[j] { continue; }
                let other = &boxes[j];
                let ix1 = current.left().max(other.left());
                let iy1 = current.top().max(other.top());
                let ix2 = current.right().min(other.right());
                let iy2 = current.bottom().min(other.bottom());
                if ix2 > ix1 && iy2 > iy1 {
                    let inter = (ix2 - ix1) * (iy2 - iy1);
                    let area_a = current.area();
                    let area_b = other.area();
                    let iou = inter as f32 / (area_a + area_b - inter).max(1) as f32;
                    if iou > X_OVERLAP_THRESHOLD {
                        used[j] = true;
                        if other.confidence > current.confidence {
                            current = other.clone();
                        }
                    }
                }
            }
            result.push(current);
        }
        result
    }

    pub fn sort_detected_boxes(&self, mut boxes: Vec<BoundingBox>) -> Vec<BoundingBox> {
        boxes.sort_by(|a, b| {
            let row_diff = (a.y / 20) - (b.y / 20);
            if row_diff != 0 {
                row_diff.cmp(&0)
            } else {
                a.x.cmp(&b.x)
            }
        });
        boxes
    }

    // TODO: Implement recognize_char() using Burn recognition models
    // For now, this is a stub that returns empty results
}
