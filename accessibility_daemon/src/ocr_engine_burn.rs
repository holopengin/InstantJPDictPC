use anyhow::{Context, Result};
use burn::backend::Wgpu;
use burn::tensor::{Tensor, Shape, Data, ElementConversion, Distribution};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, Recorder, FileRecorder};
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use std::path::Path;
use std::time::Instant;

use crate::models::*;

type Backend = Wgpu;

const DETECT_WIDTH: u32 = 960;
const DETECT_HEIGHT: u32 = 544;
const REC_WIDTH: u32 = 960;
const REC_HEIGHT: u32 = 32;
const VERT_REC_WIDTH: u32 = 32;
const VERT_REC_HEIGHT: u32 = 480;

/// Burn WGPU-based OCR engine — drop-in replacement for the ORT-based OcrEngine
pub struct OcrEngineBurn {
    device: burn::backend::wgpu::WgpuDevice,
    // Model weights loaded from .bpk files
    detect_weights: Vec<u8>,
    rec_h_weights: Vec<u8>,
    rec_v_weights: Vec<u8>,
    pub char_vocab: Vec<i64>,
}

impl OcrEngineBurn {
    pub fn new(model_dir: &str) -> Result<Self> {
        let model_path = Path::new(model_dir);
        let device = burn::backend::wgpu::WgpuDevice::default();

        println!("Loading Burn model weights…");

        // Load .bpk weight files
        let detect_weights = std::fs::read(model_path.join("meiki.text.detect.v0.1.bpk"))
            .context("Failed to read detection model weights")?;
        let rec_h_weights = std::fs::read(model_path.join("meiki.text.rec.v0.960x32.bpk"))
            .context("Failed to read horizontal recognition model weights")?;
        let rec_v_weights = std::fs::read(model_path.join("meiki.text.rec.v0.vertical.32x480.bpk"))
            .context("Failed to read vertical recognition model weights")?;

        // Load character vocabulary
        let vocab_path = model_path.join("char_vocab.json");
        let vocab_json: Vec<i64> = match std::fs::read_to_string(&vocab_path) {
            Ok(content) => serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse char_vocab.json"))?,
            Err(_) => Vec::new(),
        };

        println!("Burn WGPU OCR engine ready.");

        Ok(OcrEngineBurn {
            device,
            detect_weights,
            rec_h_weights,
            rec_v_weights,
            char_vocab: vocab_json,
        })
    }

    pub fn is_ready(&self) -> bool {
        !self.char_vocab.is_empty()
    }

    fn image_to_nchw(&self, img: &image::DynamicImage) -> Vec<f32> {
        let w = img.width() as usize;
        let h = img.height() as usize;
        let mut data = vec![0.0f32; 3 * h * w];
        for (x, y, pixel) in img.pixels() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let y_usize = y as usize;
            let x_usize = x as usize;
            data[0 * h * w + y_usize * w + x_usize] = r;
            data[1 * h * w + y_usize * w + x_usize] = g;
            data[2 * h * w + y_usize * w + x_usize] = b;
        }
        data
    }

    /// Detect bounding boxes using the Burn WGPU detection model.
    /// NOTE: This is a placeholder — the actual model architecture needs to be
    /// instantiated from the generated .rs files. For now, this demonstrates
    /// the preprocessing pipeline and tensor format.
    pub fn detect(&self, image: &DynamicImage) -> Result<Vec<BoundingBox>> {
        let orig_w = image.width() as i32;
        let orig_h = image.height() as i32;

        // 1. Resize with Lanczos3 (matches ORT/PIL BILINEAR)
        let resized = image.resize_exact(
            DETECT_WIDTH,
            DETECT_HEIGHT,
            image::imageops::FilterType::Lanczos3,
        );

        // 2. Convert to NCHW float tensor
        let img_data = self.image_to_nchw(&DynamicImage::ImageRgba8(resized.to_rgba8()));

        // 3. Create Burn tensor
        let input_tensor: Tensor<Backend, 1> = Tensor::from_data(
            Data::new(img_data, Shape::new([1, 3, DETECT_HEIGHT as usize, DETECT_WIDTH as usize])),
            &self.device,
        );

        // TODO: Load model from .bpk weights and run inference
        // The model architecture is in burn_models/detect/meiki.text.detect.v0.1.rs
        // Weights are in self.detect_weights
        //
        // For now, return empty (model loading + inference needs the generated Model struct)

        let _ = (input_tensor, orig_w, orig_h);

        Ok(Vec::new())
    }
}
