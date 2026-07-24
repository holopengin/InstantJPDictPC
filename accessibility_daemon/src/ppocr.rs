use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use ort::session::Session;
use ort::value::Tensor;

/// Run PP-OCRv6 recognition on a single image crop.
///
/// Returns (text, alternatives, char_cols, max_t) where char_cols are the
/// timestep indices (in model-output space) for each decoded character,
/// and max_t is the total number of timesteps in the model output.
///
/// This model uses CTC decoding, NOT NRTR...
/// "algorithm: NRTR" — the exported ONNX uses CTC label decoding).
///
/// Vocab layout (18710 classes total):
///   class 0    = CTC blank token
///   class 1-18708 = character_dict entries (18708 chars from vocab.json)
///   class 18709 = ' ' (space, appended by CTCLabelDecode with use_space_char=True)
///
/// Decoding: CTC greedy — argmax → collapse repeats → strip blank (0).
///
/// Input:  [1, 3, 48, W] — BGR, height 48, variable width
/// Output: [1, seq_len, 18710] — character logits per timestep
///
/// Preprocessing:
/// 1. If crop is portrait (height >= 1.5x width), rotate 90° CCW
/// 2. Lanczos resize to height 48, proportional width
/// 3. Grayscale, normalized as pixel/128 - 1, replicated to 3 channels
pub fn recognize_ppocr_vertical(
    sess: &mut Session,
    crop: &DynamicImage,
    vocab: &[String],
) -> Result<(String, Vec<Vec<(char, f32)>>, Vec<f32>, usize)> {
    // 1. Rotate 90° CCW if portrait (image crate: rotate90°=CW, rotate270°=CCW)
    let rotated = if crop.height() >= crop.width().saturating_mul(3) / 2 {
        crop.rotate270()
    } else {
        crop.clone()
    };
    let (rw, rh) = rotated.dimensions();
    if rw < 4 || rh < 4 {
        return Ok((String::new(), Vec::new(), Vec::new(), 0));
    }

    // 2. Lanczos resize to height 48, proportional width
    let target_h = 48u32;
    let target_w = (rw as f32 * target_h as f32 / rh as f32).round() as u32;
    let target_w = target_w.max(4).min(3200);
    let resized = rotated.resize_exact(
        target_w,
        target_h,
        image::imageops::FilterType::Lanczos3,
    );

    // 3. Grayscale → pixel/128 - 1 → replicate to 3 channels
    let gray = resized.to_luma8();
    let n = (target_h * target_w) as usize;
    let mut data = vec![0.0f32; 3 * n];

    for y in 0..target_h {
        for x in 0..target_w {
            let px = gray.get_pixel(x, y)[0];
            let val = (px as f32) / 128.0 - 1.0;
            let idx = (y * target_w + x) as usize;
            data[idx] = val;
            data[n + idx] = val;
            data[2 * n + idx] = val;
        }
    }

    // 4. Create tensor and run inference
    let input_shape = [1i64, 3, target_h as i64, target_w as i64];
    let tensor = Tensor::from_array((input_shape, data.into_boxed_slice()))?;
    let outputs = sess
        .run(ort::inputs!["x" => tensor])
        .context("PP-OCR inference failed")?;

    // 5. Extract output
    let output_val = outputs["fetch_name_0"]
        .try_extract_array::<f32>()
        .context("Failed to extract PP-OCR output")?
        .into_owned();

    let seq_len = (target_w / 4).max(1) as usize;
    let num_classes = 18710usize;
    let flat: Vec<f32> = output_val.iter().copied().collect();
    let max_t = seq_len.min(flat.len() / num_classes);

    // 6. CTC greedy decode
    //    class 0 = blank (skip, use for repeat-collapse)
    //    class 1..18708 = vocab[class-1]
    //    class 18709  = ' ' (space)
    let mut text = String::new();
    let mut alts: Vec<Vec<(char, f32)>> = Vec::new();
    let mut char_cols: Vec<f32> = Vec::new(); // x-positions in rotated image space
    let mut prev_class = 0i32; // 0 = blank

    for t in 0..max_t {
        let offset = t * num_classes;

        // Find best class
        let mut best_class = 0i32;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let val = flat[offset + c];
            if val > best_score {
                best_score = val;
                best_class = c as i32;
            }
        }

        // Top-5 alternatives (excluding blank=0)
        let mut top_k: Vec<(usize, f32)> = Vec::with_capacity(5);
        for c in 1..num_classes {
            let val = flat[offset + c];
            if top_k.len() < 5 {
                top_k.push((c, val));
            } else {
                let min_idx = top_k
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                if val > top_k[min_idx].1 {
                    top_k[min_idx] = (c, val);
                }
            }
        }
        top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Map alternatives to chars
        let frame_alts: Vec<(char, f32)> = top_k
            .iter()
            .take(5)
            .filter_map(|(c, s)| {
                let ch = class_to_char(*c as i32, vocab);
                ch.map(|ch| (ch, *s))
            })
            .collect();
        alts.push(frame_alts);

        // CTC collapse: skip blank (0) and consecutive repeats
        if best_class > 0 && best_class != prev_class {
            if let Some(ch) = class_to_char(best_class, vocab) {
                text.push(ch);
                // Model downsamples 4x. Center of this timestep's receptive
                // field in model-input coords: (t + 0.5) * 4
                // Store the TIMESTEP index so the caller can map t=0 → box top
                // and t=max_t → box bottom.
                char_cols.push(t as f32);
            }
        }

        if best_class > 0 {
            prev_class = best_class;
        } else {
            prev_class = 0;
        }
    }

    Ok((text, alts, char_cols, max_t))
}

/// Map model class index to character.
/// class 0        = blank (returns None)
/// class 1..18708 = vocab[class-1]
/// class 18709    = ' ' (space, appended by CTCLabelDecode)
fn class_to_char(class: i32, vocab: &[String]) -> Option<char> {
    if class <= 0 {
        return None;
    }
    if class as usize <= vocab.len() {
        // class 1..N maps to vocab index 0..N-1
        vocab.get(class as usize - 1).and_then(|s| s.chars().next())
    } else if class == 18709 {
        // Space
        Some(' ')
    } else {
        None
    }
}
