//! Free functions for parallel OCR recognition.
//! Each function creates its own ONNX sessions, allowing true parallel execution.

use anyhow::{Context, Result};
use image::{DynamicImage, RgbaImage};
use ndarray as nd;
use ort::session::Session;
use ort::value::Tensor;

use crate::models::*;

// Constants copied from ocr_engine.rs
const REC_WIDTH: u32 = 960;
const REC_HEIGHT: u32 = 32;
const VERT_REC_WIDTH: u32 = 32;
const VERT_REC_HEIGHT: u32 = 480;
const REC_CONFIDENCE_THRESHOLD: f32 = 0.1;
const X_OVERLAP_THRESHOLD: f32 = 0.3;
const NUM_QUERIES: usize = 48;
pub const MAX_BATCH_SIZE: usize = 8;

/// Run recognition on a single chunk using the provided sessions.
pub fn recognize_single_chunk_static(
    session: &mut Session,
    char_vocab: &[i64],
    chunk: &DynamicImage,
    is_vertical: bool,
) -> Result<(Vec<CharCandidate>, i32, i32)> {
    let target_w = if is_vertical { VERT_REC_WIDTH } else { REC_WIDTH };
    let target_h = if is_vertical { VERT_REC_HEIGHT } else { REC_HEIGHT };

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
        return Ok((Vec::new(), effective_w, effective_h));
    }

    let resized = chunk.resize_exact(
        effective_w as u32,
        effective_h as u32,
        image::imageops::FilterType::Triangle,
    );
    let mut padded = RgbaImage::from_pixel(target_w, target_h, image::Rgba([0u8, 0u8, 0u8, 255u8]));
    let resized_rgba = resized.to_rgba8();
    for y in 0..(effective_h as u32) {
        for x in 0..(effective_w as u32) {
            let p = resized_rgba.get_pixel(x, y);
            padded.put_pixel(x, y, *p);
        }
    }

    let img_data = image_to_nchw_static(&padded, target_w, target_h);
    let input_tensor = Tensor::from_array((
        [1i64, 3, target_h as i64, target_w as i64],
        img_data.into_boxed_slice(),
    ))?;

    let (labels_arr_opt, boxes_arr_opt, scores_arr_opt, indices_arr_opt, raw_logits_opt) = {
        let active_session = session;
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

        let output_names: Vec<String> = active_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let run_outputs = active_session.run(inputs)?;

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

        let labels_arr_opt: Option<Vec<i64>> = labels_val.and_then(|v| {
            try_extract_i64(v)
                .map(|a| a.iter().cloned().collect())
                .or_else(|| {
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

    if labels_arr_opt.is_none() || boxes_arr_opt.is_none() || scores_arr_opt.is_none() {
        return Ok((Vec::new(), effective_w, effective_h));
    }

    let labels_arr = labels_arr_opt.unwrap();
    let boxes_arr = boxes_arr_opt.unwrap();
    let scores_arr = scores_arr_opt.unwrap();
    let indices_arr = indices_arr_opt;
    let raw_logits = raw_logits_opt;

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
                        let mut kv: Vec<(usize, f32)> =
                            qlogits.iter().cloned().enumerate().collect();
                        kv.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        for (j, &(_idx, _val)) in kv.iter().enumerate().take(15) {
                            let class_idx = kv[j].0;
                            let ch = char_vocab
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
                calculate_y_overlap_static(&cand.box_coords, &f.box_coords)
            } else {
                calculate_x_overlap_static(&cand.box_coords, &f.box_coords)
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

    Ok((filtered, effective_w, effective_h))
}

/// Run batched recognition on multiple chunks at once (max 8 per batch).
/// Returns a vec of (char_candidates, effective_w, effective_h) for each chunk.
pub fn recognize_batch_chunks_static(
    session: &mut Session,
    char_vocab: &[i64],
    chunks: &[(DynamicImage, bool)], // (chunk, is_vertical)
) -> Result<Vec<(Vec<CharCandidate>, i32, i32)>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = chunks.len().min(MAX_BATCH_SIZE);
    let is_vertical = chunks[0].1;

    let target_w = if is_vertical { VERT_REC_WIDTH } else { REC_WIDTH };
    let target_h = if is_vertical { VERT_REC_HEIGHT } else { REC_HEIGHT };

    // Prepare batch input tensor: [batch_size, 3, target_h, target_w]
    let t_pre = std::time::Instant::now();
    let mut t_infer = std::time::Duration::ZERO;
    let mut batch_data = vec![0.0f32; batch_size * 3 * target_h as usize * target_w as usize];

    let mut effective_sizes = Vec::with_capacity(batch_size);

    for (idx, (chunk, chunk_is_vertical)) in chunks.iter().take(batch_size).enumerate() {
        assert_eq!(*chunk_is_vertical, is_vertical, "Batch must have consistent orientation");

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
            effective_sizes.push((0, 0));
            continue;
        }

        effective_sizes.push((effective_w, effective_h));

        // Resize chunk and pad to target
        let resized = chunk.resize_exact(
            effective_w as u32,
            effective_h as u32,
            image::imageops::FilterType::Triangle,
            );
            let mut padded = RgbaImage::from_pixel(target_w, target_h, image::Rgba([0u8, 0u8, 0u8, 255u8]));
            let resized_rgba = resized.to_rgba8();
        for y in 0..(effective_h as u32) {
            for x in 0..(effective_w as u32) {
                let p = resized_rgba.get_pixel(x, y);
                padded.put_pixel(x, y, *p);
            }
        }

        // Convert to NCHW and copy into batch
        let img_data = image_to_nchw_static(&padded, target_w, target_h);
        let batch_offset = idx * 3 * target_h as usize * target_w as usize;
        batch_data[batch_offset..batch_offset + img_data.len()].copy_from_slice(&img_data);
    }

    let input_tensor = Tensor::from_array((
        [batch_size as i64, 3, target_h as i64, target_w as i64],
        batch_data.into_boxed_slice(),
    ))?;

    // Run session
    let (labels_arr_opt, boxes_arr_opt, scores_arr_opt, indices_arr_opt, raw_logits_opt) = {
        let session_input_names: Vec<String> = session
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

        let has_orig_target_sizes = session_input_names.iter().any(|n| n == "orig_target_sizes");

        let inputs = if has_orig_target_sizes {
            let size_tensor = Tensor::from_array((
                [batch_size as i64, 2],
                vec![target_w as i64, target_h as i64].repeat(batch_size).into_boxed_slice(),
            ))?;
            ort::inputs! {
                image_input_name.as_str() => input_tensor,
                "orig_target_sizes" => size_tensor
            }
        } else {
            ort::inputs! { image_input_name.as_str() => input_tensor }
        };

        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let run_outputs = session.run(inputs)?;
        let t_infer = t_pre.elapsed();

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

        let labels_arr_opt: Option<Vec<i64>> = labels_val.and_then(|v| {
            try_extract_i64(v)
                .map(|a| a.iter().cloned().collect())
                .or_else(|| {
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

    if labels_arr_opt.is_none() || boxes_arr_opt.is_none() || scores_arr_opt.is_none() {
        // Return empty for all items in batch
        return Ok(chunks
            .iter()
            .take(batch_size)
            .map(|_| (Vec::new(), 0, 0))
            .collect());
    }

    let boxes_arr = boxes_arr_opt.unwrap().iter().cloned().collect::<Vec<f32>>();
    let scores_arr = scores_arr_opt.unwrap().iter().cloned().collect::<Vec<f32>>();
    let labels_arr = labels_arr_opt.unwrap().iter().cloned().collect::<Vec<i64>>();
    let indices_arr = indices_arr_opt.map(|idx| idx.iter().cloned().collect::<Vec<i64>>());
    // Handle logits for alternatives
        let logits_matrix: Option<Vec<Vec<Vec<f32>>>> = raw_logits_opt.as_ref().and_then(|raw| {
            if raw.len() % (batch_size * NUM_QUERIES) == 0 && raw.len() > 0 {
                let num_classes = raw.len() / (batch_size * NUM_QUERIES);
                let mut batch_mat = Vec::with_capacity(batch_size);
                for b in 0..batch_size {
                    let mut mat: Vec<Vec<f32>> = Vec::with_capacity(NUM_QUERIES);
                    for q in 0..NUM_QUERIES {
                        let mut row: Vec<f32> = Vec::with_capacity(num_classes);
                        for c in 0..num_classes {
                            row.push(raw[(b * NUM_QUERIES + q) * num_classes + c]);
                        }
                        mat.push(row);
                    }
                    batch_mat.push(mat);
                }
                Some(batch_mat)
            } else {
                None
            }
        });

        // Parse outputs for each item in batch
        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let (effective_w, effective_h) = effective_sizes[b];
            if effective_w <= 0 || effective_h <= 0 {
                results.push((Vec::new(), 0, 0));
                continue;
            }

            // Extract this batch item's outputs (flat arrays)
            let labels_start = b * NUM_QUERIES;
            let labels_end = labels_start + NUM_QUERIES;
            let batch_labels = &labels_arr[labels_start..labels_end];

            let boxes_start = b * NUM_QUERIES * 4;
            let boxes_end = boxes_start + NUM_QUERIES * 4;
            let batch_boxes_flat = &boxes_arr[boxes_start..boxes_end];

            let scores_start = b * NUM_QUERIES;
            let scores_end = scores_start + NUM_QUERIES;
            let batch_scores = &scores_arr[scores_start..scores_end];

            let batch_indices = indices_arr.as_ref().map(|idx| {
                let idx_start = b * NUM_QUERIES;
                let idx_end = idx_start + NUM_QUERIES;
                &idx[idx_start..idx_end]
            });

            let mut candidates: Vec<CharCandidate> = Vec::new();
            for i in 0..batch_scores.len() {
                if batch_scores[i] > REC_CONFIDENCE_THRESHOLD {
                    let mut alternatives: Vec<(char, f32)> = Vec::new();
                    if let (Some(batch_mat), Some(indices)) = (&logits_matrix, &batch_indices) {
                        if let Some(mat) = batch_mat.get(b) {
                            let num_classes = if !mat.is_empty() { mat[0].len() } else { 0 };
                            if num_classes > 0 && i < indices.len() {
                                let query_idx = (indices[i] / (num_classes as i64)) as usize;
                                if query_idx < mat.len() {
                                    let qlogits = &mat[query_idx];
                                    let mut kv: Vec<(usize, f32)> =
                                        qlogits.iter().cloned().enumerate().collect();
                                    kv.sort_by(|a, b| {
                                        b.1.partial_cmp(&a.1)
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                    });
                                    for (j, &(_idx, _val)) in kv.iter().enumerate().take(15) {
                                        let class_idx = kv[j].0;
                                        let ch = char_vocab
                                            .get(class_idx)
                                            .and_then(|c| std::char::from_u32(*c as u32))
                                            .unwrap_or(' ');
                                        alternatives.push((ch, kv[j].1));
                                    }
                                }
                            }
                        }
                    }

                    let label_char = batch_labels
                        .get(i)
                        .and_then(|v| std::char::from_u32(*v as u32))
                        .unwrap_or(' ');
                    let box_coords = [
                        batch_boxes_flat[i * 4 + 0],
                        batch_boxes_flat[i * 4 + 1],
                        batch_boxes_flat[i * 4 + 2],
                        batch_boxes_flat[i * 4 + 3],
                    ];
                    candidates.push(CharCandidate {
                        char: label_char,
                        score: batch_scores[i],
                        box_coords,
                        alternatives,
                    });
                }
            }

        // Sort and filter overlaps (same logic as single chunk)
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
                    calculate_y_overlap_static(&cand.box_coords, &f.box_coords)
                } else {
                    calculate_x_overlap_static(&cand.box_coords, &f.box_coords)
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

        results.push((filtered, effective_w, effective_h));
    }

    println!("[BATCH timing] {:>2} items: pre={:.1}ms  infer={:.1}ms  total={:.1}ms",
        batch_size,
        t_pre.elapsed().as_secs_f64() * 1000.0 - t_infer.as_secs_f64() * 1000.0,
        t_infer.as_secs_f64() * 1000.0,
        t_pre.elapsed().as_secs_f64() * 1000.0);

    Ok(results)
}

/// Recognize a long line by splitting into chunks (vertical or horizontal).
pub fn recognize_long_line_static(
    session: &mut Session,
    char_vocab: &[i64],
    crop: &DynamicImage,
    crop_x: i32,
    crop_y: i32,
    is_vertical: bool,
) -> Result<LineResult> {
    // For long lines in parallel mode, fall back to a single-chunk approach.
    // We only have one session here, so we pass it as both the primary and
    // vertical session. This works because the session is selected by is_vertical
    // inside recognize_single_chunk_static, and we pass the right one as the
    // matching orientation.
    let (filtered, eff_w, eff_h) =
        recognize_single_chunk_static(session, char_vocab, crop, is_vertical)?;

    let text: String = filtered.iter().map(|c| c.char).collect();
    let mut char_boxes: Vec<BoundingBox> = Vec::new();

    for c in filtered.iter() {
        if is_vertical {
            let x1 = (c.box_coords[0] / (VERT_REC_WIDTH as f32)) * (crop.width() as f32)
                + (crop_x as f32);
            let y1 = (c.box_coords[1] / (eff_h as f32)) * (crop.height() as f32)
                + (crop_y as f32);
            let x2 = (c.box_coords[2] / (VERT_REC_WIDTH as f32)) * (crop.width() as f32)
                + (crop_x as f32);
            let y2 = (c.box_coords[3] / (eff_h as f32)) * (crop.height() as f32)
                + (crop_y as f32);
            char_boxes.push(BoundingBox::new(
                x1.round() as i32,
                y1.round() as i32,
                (x2 - x1).round() as i32,
                (y2 - y1).round() as i32,
                c.score,
            ));
        } else {
            let x1 = (c.box_coords[0] / (eff_w as f32)) * (crop.width() as f32) + (crop_x as f32);
            let y1 = (c.box_coords[1] / (REC_HEIGHT as f32)) * (crop.height() as f32)
                + (crop_y as f32);
            let x2 = (c.box_coords[2] / (eff_w as f32)) * (crop.width() as f32) + (crop_x as f32);
            let y2 = (c.box_coords[3] / (REC_HEIGHT as f32)) * (crop.height() as f32)
                + (crop_y as f32);
            char_boxes.push(BoundingBox::new(
                x1.round() as i32,
                y1.round() as i32,
                (x2 - x1).round() as i32,
                (y2 - y1).round() as i32,
                c.score,
            ));
        }
    }

    Ok(LineResult {
        text,
        char_boxes,
        alternatives: Vec::new(),
        is_vertical,
        chunk_boxes: Vec::new(),
    })
}

fn image_to_nchw_static(img: &RgbaImage, width: u32, height: u32) -> Vec<f32> {
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

fn calculate_x_overlap_static(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
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

fn calculate_y_overlap_static(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
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

/// Public entry point: recognize a single line by creating ONNX sessions from model files.
/// This is the function called from the parallel recognition loop.
pub fn recognize_single_line_static(
    model_dir: &str,
    char_vocab: &[i64],
    image: &DynamicImage,
    bbox: &BoundingBox,
) -> Result<Option<LineResult>> {
    use ort::session::Session;

    let mut recognize_session = Session::builder()?
        .commit_from_file(format!("{}/meiki.text.rec.v0.960x32.with_logits.onnx", model_dir))?;
    let mut recognize_session_vertical = Session::builder()?.commit_from_file(
        format!("{}/meiki.text.rec.v0.vertical.32x480.with_logits.onnx", model_dir),
    )?;

    let rec_session = std::sync::Arc::new(std::sync::Mutex::new(recognize_session));
    let rec_session_vert = std::sync::Arc::new(std::sync::Mutex::new(recognize_session_vertical));
    recognize_single_line_with_sessions(
        &rec_session,
        &rec_session_vert,
        char_vocab,
        image,
        bbox,
    )
}

/// Recognize a single line using pre-built ONNX sessions (avoids reloading models per box).
pub fn recognize_single_line_with_sessions(
    recognize_session: &std::sync::Arc<std::sync::Mutex<Session>>,
    recognize_session_vertical: &std::sync::Arc<std::sync::Mutex<Session>>,
    char_vocab: &[i64],
    image: &DynamicImage,
    bbox: &BoundingBox,
) -> Result<Option<LineResult>> {
    let is_vertical = bbox.h > bbox.w;
    let crop_x = bbox.x.max(0) as u32;
    let crop_y = bbox.y.max(0) as u32;
    let crop_w = bbox.w.max(0) as u32;
    let crop_h = bbox.h.max(0) as u32;
    if crop_w == 0 || crop_h == 0 {
        return Ok(None);
    }
    let crop = image.crop_imm(crop_x, crop_y, crop_w, crop_h);

    // Lock both sessions for the duration of this box's recognition.
    let mut rec_sess = recognize_session.lock().unwrap();
    let mut rec_sess_vert = recognize_session_vertical.lock().unwrap();

    let result = if is_vertical
        && ((crop_h as f32) * (32.0f32 / crop_w as f32) > 350.0f32)
    {
        Some(recognize_long_line_static(
            &mut rec_sess_vert,
            char_vocab,
            &crop,
            crop_x as i32,
            crop_y as i32,
            true,
        )?)
    } else if !is_vertical
        && ((crop_w as f32) * (32.0f32 / crop_h as f32) > REC_WIDTH as f32)
    {
        Some(recognize_long_line_static(
            &mut rec_sess,
            char_vocab,
            &crop,
            crop_x as i32,
            crop_y as i32,
            false,
        )?)
    } else {
        let (filtered, eff_w, eff_h) = if is_vertical {
            recognize_single_chunk_static(
                &mut rec_sess_vert,
                char_vocab,
                &crop,
                is_vertical,
            )?
        } else {
            recognize_single_chunk_static(
                &mut rec_sess,
                char_vocab,
                &crop,
                is_vertical,
            )?
        };
        let text: String = filtered.iter().map(|c| c.char).collect();
        let alternatives: Vec<Vec<(char, f32)>> =
            filtered.iter().map(|c| c.alternatives.clone()).collect();
        let mut char_boxes: Vec<BoundingBox> = Vec::new();
        for c in filtered.iter() {
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
                let y1 = (c.box_coords[1] / (REC_HEIGHT as f32)) * (crop_h as f32)
                    + (crop_y as f32);
                let x2 = (c.box_coords[2] / (eff_w as f32)) * (crop_w as f32) + (crop_x as f32);
                let y2 = (c.box_coords[3] / (REC_HEIGHT as f32)) * (crop_h as f32)
                    + (crop_y as f32);
                char_boxes.push(BoundingBox::new(
                    x1.round() as i32,
                    y1.round() as i32,
                    (x2 - x1).round() as i32,
                    (y2 - y1).round() as i32,
                    c.score,
                ));
            }
        }
        Some(LineResult {
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
        })
    };

    Ok(result)
}
