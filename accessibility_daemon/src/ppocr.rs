use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use ort::session::Session;
use ort::value::Tensor;

/// Run PP-OCRv6 recognition on multiple image crops in one batched
/// inference call.  Returns one result per crop.
///
/// Input:  [N, 3, 48, W] — N crops, height 48, variable width (padded to
///         the widest crop in the batch).
/// Output: [N, seq_len, 18710] — character logits per timestep.
///
/// Preprocessing (per crop):
/// 1. If crop is portrait (height >= 1.5× width), rotate 90° CCW
/// 2. Lanczos resize to height 48, proportional width
/// 3. Grayscale, pixel/128 - 1, replicated to 3 channels
///
/// Decoding: CTC greedy — argmax → collapse repeats → strip blank (0).
/// vocab layout: 0=blank, 1..18708=chars, 18709=space.
pub fn recognize_ppocr_vertical_batch(
    sess: &mut Session,
    crops: &[&DynamicImage],
    vocab: &[String],
) -> Result<Vec<(String, Vec<Vec<(char, f32)>>, Vec<f32>, usize)>> {
    let num_crops = crops.len();
    if num_crops == 0 {
        return Ok(Vec::new());
    }

    // ---- preprocessing ---------------------------------------------------
    struct PrepRes {
        width: u32,       // target_w after resize
        seq_len: usize,   // model timesteps for this crop
        data: Vec<f32>,   // 3 * 48 * width normalized pixels
    }

    let target_h = 48u32;
    let mut prepped: Vec<PrepRes> = Vec::with_capacity(num_crops);
    let mut max_w = 0u32;

    for crop in crops.iter().copied() {
        // 1. rotate CCW if portrait
        let rotated = if crop.height() >= crop.width().saturating_mul(3) / 2 {
            crop.rotate270()
        } else {
            crop.clone()
        };
        let (rw, rh) = rotated.dimensions();
        if rw < 4 || rh < 4 {
            prepped.push(PrepRes {
                width: 0,
                seq_len: 0,
                data: Vec::new(),
            });
            continue;
        }

        // 2. Lanczos resize to height 48
        let target_w = (rw as f32 * target_h as f32 / rh as f32).round() as u32;
        let target_w = target_w.max(4).min(3200);
        let resized = rotated.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);

        // 3. grayscale → pixel/128 - 1 → 3 channels
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

        let seq_len = ((target_w as f32 / 4.0).ceil() as usize).max(1);
        max_w = max_w.max(target_w);
        prepped.push(PrepRes { width: target_w, seq_len, data });
    }

    // ---- build batched input tensor --------------------------------------
    // Each crop is padded to max_w with zeros (which is neutral gray after
    // normalisation: pixel 128 → 128/128 - 1 = 0.0).
    let n_chan = 3usize;
    let total = n_chan * target_h as usize * max_w as usize;
    let mut batch_data = vec![0.0f32; num_crops * total];

    for (i, p) in prepped.iter().enumerate() {
        if p.width == 0 {
            continue;
        }
        let src_w = p.width as usize;
        let n = (target_h as usize) * src_w;
        // Copy per channel
        for c in 0..n_chan {
            for y in 0..target_h as usize {
                let src_off = c * n + y * src_w;
                let dst_off = i * total + c * (target_h as usize) * (max_w as usize)
                    + y * (max_w as usize);
                let src_slice = &p.data[src_off..src_off + src_w];
                let dst_slice = &mut batch_data[dst_off..dst_off + src_w];
                dst_slice.copy_from_slice(src_slice);
            }
        }
    }

    // ---- run inference ---------------------------------------------------
    let input_shape = [
        num_crops as i64,
        3,
        target_h as i64,
        max_w as i64,
    ];
    let tensor = Tensor::from_array((input_shape, batch_data.into_boxed_slice()))?;
    let outputs = sess
        .run(ort::inputs!["x" => tensor])
        .context("PP-OCR batched inference failed")?;

    let output_val = outputs["fetch_name_0"]
        .try_extract_array::<f32>()
        .context("Failed to extract PP-OCR output")?
        .into_owned();
    let flat: Vec<f32> = output_val.iter().copied().collect();

    let num_classes = 18710usize;
    // Expected output shape: [N, seq_len_total, 18710].
    // seq_len_total should be max_w/4, but infer from the flat array.
    let seq_len_total = flat.len() / (num_crops * num_classes);

    // ---- decode each item -------------------------------------------------
    let mut results: Vec<(String, Vec<Vec<(char, f32)>>, Vec<f32>, usize)> =
        Vec::with_capacity(num_crops);

    for i in 0..num_crops {
        let seq_len = prepped[i].seq_len.min(seq_len_total);
        let base_off = i * seq_len_total * num_classes;
        let mut text = String::new();
        let mut alts: Vec<Vec<(char, f32)>> = Vec::new();
        let mut char_cols: Vec<f32> = Vec::new();
        let mut prev_class = 0i32;

        for t in 0..seq_len {
            let offset = base_off + t * num_classes;
            let slice = &flat[offset..offset + num_classes];
            let (max_idx, &max_val) = slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            let class_idx = max_idx as i32;

            // Collect alternatives (top-5)
            let mut top5: Vec<(usize, f32)> = slice
                .iter()
                .enumerate()
                .filter(|(_, &v)| v.is_finite())
                .map(|(i, &v)| (i, v))
                .collect();
            top5.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top5_chars: Vec<(char, f32)> = top5[..top5.len().min(5)]
                .iter()
                .map(|(idx, score)| {
                    let ch = if *idx == 18709 {
                        ' '
                    } else if *idx > 0 && *idx <= 18708 {
                        std::char::from_u32(vocab[*idx - 1].chars().next().unwrap() as u32)
                            .unwrap_or('\u{FFFD}')
                    } else {
                        '\0'
                    };
                    (ch, *score)
                })
                .collect();
            alts.push(top5_chars);

            // CTC: skip blank (0). Collapse repeats (don't output if same as
            // previous non-blank class). Space (18709) can repeat.
            if class_idx == 0 {
                prev_class = 0;
                continue;
            }
            if class_idx == 18709 {
                text.push(' ');
                prev_class = 18709;
                // Track timestep position even for space
                char_cols.push(t as f32);
                continue;
            }
            if class_idx == prev_class {
                continue;
            }

            let ch = if class_idx > 0 && class_idx <= 18708 {
                let s = &vocab[(class_idx - 1) as usize];
                s.chars().next().unwrap_or('\u{FFFD}')
            } else {
                continue;
            };
            text.push(ch);
            char_cols.push(t as f32);
            prev_class = class_idx;
        }

        results.push((text, alts, char_cols, seq_len_total));
    }

    Ok(results)
}

/// Single-crop convenience wrapper around the batched function.
pub fn recognize_ppocr_vertical(
    sess: &mut Session,
    crop: &DynamicImage,
    vocab: &[String],
) -> Result<(String, Vec<Vec<(char, f32)>>, Vec<f32>, usize)> {
    let mut batch = recognize_ppocr_vertical_batch(sess, &[crop], vocab)?;
    Ok(batch.remove(0))
}
