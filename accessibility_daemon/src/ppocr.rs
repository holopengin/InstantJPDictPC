//! PP-OCRv6 recognition — dynamic-width ncnn (rec_dyn).
//!
//! Ported from InstantJPDict's OcrEngine.inferResizedRec /
//! recognizePpocrBatch / ctcDecode[TopK]. The model, the input geometry
//! (rotate 270° portrait, resize to 48×W, mult-of-8 zero pad), the
//! `(gray/127.5)-1` normalisation, the lengthwise squish, the class remap
//! and the CTC greedy decode (with sub-column peak interpolation and top-15
//! alternatives) all match the mobile app — the inference itself runs the
//! shared native core (see `ppocr_ncnn.rs`), so the two apps can only drift
//! if a Kotlin-side change is not mirrored here.

use anyhow::{bail, Result};
use image::{DynamicImage, GenericImageView};
use std::borrow::Cow;

use crate::ppocr_ncnn::{top_k, RecNet};

pub const REC_TARGET_H: u32 = 48;
const REC_STRIDE: u32 = 8;
/// Single-pass targetW cap; the mobile long-line stitch gate. Beyond it lines
/// are chunked and stitched (see `stitch_long_line`).
const LONG_LINE_GATE: u32 = 2000;
/// Per-chunk targetW cap in the stitch paths.
const CHUNK_TARGET_MAX: u32 = 480;
/// Stitch best-pair window (last N stitched × first N current).
const STITCH_WINDOW: usize = 10;
/// Stitch best-pair gates: center distance and prediction overlap.
const STITCH_MAX_DIST_PX: f32 = 30.0;
const STITCH_MIN_PRED: f32 = 0.4;
/// Stitch append rule: chars past last center + this gap start a new tail.
const STITCH_APPEND_GAP_PX: f32 = 10.0;
/// Softmax row of the head: 0 = blank, 1..=18708 = chars, 18709 = space.
const SPACE_CLASS: i32 = 18709;
const FULLWIDTH_SPACE_CLASS: i32 = 18708;

/// One recognized line: text, per-character top-15 alternatives, CTC
/// timestep columns (fractional), and the model's timestep count.
#[derive(Debug, Clone)]
pub struct PpocrResult {
    pub text: String,
    pub alternatives: Vec<Vec<(char, f32)>>,
    pub char_cols: Vec<f32>,
    pub seq_len_total: usize,
}

impl PpocrResult {
    fn empty() -> Self {
        PpocrResult {
            text: String::new(),
            alternatives: Vec::new(),
            char_cols: Vec::new(),
            seq_len_total: 0,
        }
    }
}

/// Lengthwise squish factor (#24), same default as the mobile debug slider.
/// `REC_SQUISH=1.0` disables it. Clamped to the mobile's allowed 0.2..1.0.
fn rec_squish() -> f32 {
    std::env::var("REC_SQUISH")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5)
        .clamp(0.2, 1.0)
}

/// Squished content width for a target width: factor× length, min 8.
fn squish_target(target_w: u32, factor: f32) -> u32 {
    (target_w as f32 * factor).round().max(8.0) as u32
}

/// Pack gray pixels into the 3-channel rec input with PP-OCR normalise
/// `(gray/127.5 - 1)`. Zero-pads columns `[content_w, model_w)`. Mirrors the
/// mobile GRAY_LUT float math exactly (same terms, same order).
fn build_rec_input(pixels: &[u8], content_w: u32, target_h: u32, model_w: u32) -> Vec<f32> {
    let mut input = vec![0.0f32; 3 * (target_h * model_w) as usize];
    let hw = (target_h * model_w) as usize;
    let content_w = content_w as usize;
    for y in 0..target_h as usize {
        for x in 0..content_w {
            let px = (y * content_w + x) * 3;
            let r = pixels[px] as f32;
            let g = pixels[px + 1] as f32;
            let b = pixels[px + 2] as f32;
            let gray = r * 0.299f32 + g * 0.587f32 + b * 0.114f32;
            let v = gray / 127.5f32 - 1.0f32;
            let idx = y * model_w as usize + x;
            input[idx] = v;
            input[hw + idx] = v;
            input[2 * hw + idx] = v;
        }
    }
    input
}

/// Pruned-out id -> orig class id (#39); identity fallback if remap failed.
fn remap_class(remap: &[i32], pruned_idx: i32) -> i32 {
    remap.get(pruned_idx as usize).copied().unwrap_or(pruned_idx)
}

/// Orig class id -> char, mobile `decodeChar`.
fn decode_char(vocab: &[String], class_idx: i32) -> char {
    if class_idx == SPACE_CLASS {
        ' '
    } else if class_idx == FULLWIDTH_SPACE_CLASS {
        '\u{3000}'
    } else if (1..=FULLWIDTH_SPACE_CLASS).contains(&class_idx) {
        vocab
            .get((class_idx - 1) as usize)
            .and_then(|s| s.chars().next())
            .unwrap_or('\u{FFFD}')
    } else {
        '\u{3000}'
    }
}

/// Sub-column peak offset (#49): parabolic interpolation of the winning
/// class value across neighbouring timesteps. Returns 0 when the peak is
/// flat, at a boundary, or prominence is below the mobile gate.
fn peak_offset(v0: f32, v1: f32, v2: f32) -> f32 {
    let denom = v0 - 2.0f32 * v1 + v2;
    if denom >= -1e-6f32 {
        return 0.0f32;
    }
    if (v1 - v0).min(v1 - v2) <= 1.0f32 {
        return 0.0f32;
    }
    (0.5f32 * (v0 - v2) / denom).clamp(-0.5, 0.5)
}

/// Top-15 char alternatives for one timestep from full logits, descending.
fn top15_alternatives(vocab: &[String], remap: &[i32], slice: &[f32]) -> Vec<(char, f32)> {
    let mut scored: Vec<(usize, f32)> = slice.iter().copied().enumerate().collect();
    scored.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored
        .into_iter()
        .take(top_k())
        .map(|(idx, v)| (decode_char(vocab, remap_class(remap, idx as i32)), v))
        .collect()
}

/// Greedy CTC decode from native top-15 lists (#42): entry 0 is the argmax
/// (native emits descending, lowest-id wins ties); blank/space tests use the
/// remapped pruned indices. Mirrors mobile `ctcDecodeTopK` with
/// blankThreshold 0 (pure greedy).
fn ctc_decode_topk(
    vocab: &[String],
    remap: &[i32],
    top_pruned: &[Vec<i32>],
    top_chars: &[Vec<(char, f32)>],
    seq_len: usize,
) -> PpocrResult {
    let mut text = String::new();
    let mut alts: Vec<Vec<(char, f32)>> = Vec::new();
    let mut char_cols: Vec<f32> = Vec::new();
    let mut prev_class: i32 = 0;

    // Winner (entry 0) score at a neighbouring timestep, matched by pruned
    // class id; absent from top-15 → no interpolation for that side.
    let wval = |tt: usize, cls: i32| -> Option<f32> {
        let p = top_pruned.get(tt)?;
        let c = top_chars.get(tt)?;
        let k = p.iter().position(|&x| x == cls)?;
        c.get(k).map(|(_, s)| *s)
    };

    for t in 0..seq_len {
        let (Some(pruned), Some(indexed)) = (top_pruned.get(t), top_chars.get(t)) else {
            continue;
        };
        if pruned.is_empty() || indexed.is_empty() {
            continue;
        }
        let max_val = indexed[0].1;
        let class_idx = remap_class(remap, pruned[0]);
        let w0 = if t > 0 { wval(t - 1, pruned[0]) } else { None };
        let w2 = wval(t + 1, pruned[0]);
        let t_frac = if let (Some(w0), Some(w2)) = (w0, w2) {
            t as f32 + peak_offset(w0, max_val, w2)
        } else {
            t as f32
        };

        if class_idx == 0 {
            // blankThreshold is 0 on the single-pass path: stay pure greedy.
            prev_class = 0;
        } else if class_idx == SPACE_CLASS {
            text.push(' ');
            prev_class = SPACE_CLASS;
            char_cols.push(t_frac);
            alts.push(indexed.to_vec());
        } else if class_idx == prev_class {
            // collapse repeat
        } else {
            let ch = decode_char(vocab, class_idx);
            if ch != '\u{FFFD}' {
                text.push(ch);
                char_cols.push(t_frac);
                alts.push(indexed.to_vec());
                prev_class = class_idx;
            }
        }
    }

    PpocrResult {
        text,
        alternatives: alts,
        char_cols,
        seq_len_total: seq_len,
    }
}

/// Greedy CTC decode from full logits, mobile `ctcDecode` with
/// blankThreshold 0.
fn ctc_decode_full(
    vocab: &[String],
    remap: &[i32],
    logits: &[Vec<f32>],
    num_classes: usize,
    seq_len: usize,
) -> PpocrResult {
    let mut text = String::new();
    let mut alts: Vec<Vec<(char, f32)>> = Vec::new();
    let mut char_cols: Vec<f32> = Vec::new();
    let mut prev_class: i32 = 0;

    let wval = |tt: usize, cls: i32| -> Option<f32> {
        let slice = logits.get(tt)?;
        slice.get(cls as usize).copied()
    };

    for t in 0..seq_len {
        let Some(slice) = logits.get(t) else { continue };
        if slice.len() < num_classes {
            continue;
        }
        let (max_idx, &max_val) = slice
            .iter()
            .enumerate()
            .fold((0usize, &f32::NEG_INFINITY), |(bi, bv), (i, v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            });
        let w0 = if t > 0 { wval(t - 1, max_idx as i32) } else { None };
        let w2 = wval(t + 1, max_idx as i32);
        let t_frac = if let (Some(w0), Some(w2)) = (w0, w2) {
            t as f32 + peak_offset(w0, max_val, w2)
        } else {
            t as f32
        };
        let class_idx = remap_class(remap, max_idx as i32);
        let indexed = top15_alternatives(vocab, remap, slice);

        if class_idx == 0 {
            prev_class = 0;
        } else if class_idx == SPACE_CLASS {
            text.push(' ');
            prev_class = SPACE_CLASS;
            char_cols.push(t_frac);
            alts.push(indexed);
        } else if class_idx == prev_class {
            // collapse repeat
        } else {
            let ch = decode_char(vocab, class_idx);
            if ch != '\u{FFFD}' {
                text.push(ch);
                char_cols.push(t_frac);
                alts.push(indexed);
                prev_class = class_idx;
            }
        }
    }

    PpocrResult {
        text,
        alternatives: alts,
        char_cols,
        seq_len_total: seq_len,
    }
}

/// Recognizer output for one line crop. Mirrors mobile `recognizePpocrBatch`:
/// portrait crops rotate 270°, extreme-aspect lines (>2000 targetW) take the
/// chunk-and-stitch path (unsquished chunks), everything else takes the
/// single-pass path with the live squish factor. `remap` length is the
/// pruned CTC head width.
pub fn recognize_crop(rec: &RecNet, crop: &DynamicImage, vocab: &[String], remap: &[i32]) -> Result<PpocrResult> {
    recognize_crop_with_squish(rec, crop, vocab, remap, rec_squish())
}

fn recognize_crop_with_squish(
    rec: &RecNet,
    crop: &DynamicImage,
    vocab: &[String],
    remap: &[i32],
    squish: f32,
) -> Result<PpocrResult> {
    let (cw, ch) = crop.dimensions();
    if cw < 4 || ch < 4 {
        return Ok(PpocrResult::empty());
    }

    // Portrait crops rotate 270° so the model always sees horizontal text.
    let rotated: Cow<DynamicImage> = if ch >= cw * 3 / 2 {
        Cow::Owned(crop.rotate270())
    } else {
        Cow::Borrowed(crop)
    };
    let (rw, rh) = rotated.dimensions();

    // Long-line split for extreme aspects (targetW > 2000): very long lines
    // crush timesteps in one pass; split into overlapping exact-width chunks
    // and stitch them (#24). After the portrait rotation a long vertical line
    // is a long horizontal one, so the X path carries both in practice.
    let is_long_horiz = rw >= rh * 3 / 2
        && (rw as f32 * REC_TARGET_H as f32 / rh as f32) > LONG_LINE_GATE as f32;
    let is_long_vert = rh >= rw * 3 / 2
        && (rh as f32 * REC_TARGET_H as f32 / rw as f32) > LONG_LINE_GATE as f32;
    if is_long_horiz || is_long_vert {
        let axis = if is_long_horiz { Axis::X } else { Axis::Y };
        if let Some(stitched) = stitch_long_line(rec, &rotated, axis, vocab, remap)? {
            return Ok(stitched);
        }
        eprintln!("[PP-OCR] long-line stitch failed rw={rw} rh={rh} — falling through to crush");
    }

    // Dynamic width (#23): exact targetW capped at LONG_LINE_GATE (validated
    // #24), then squish (#24) applied pre-inference; stitch paths skip squish.
    // Model width snaps to mult-of-8 (≤7px pad).
    let target_w = ((rw as f32 * REC_TARGET_H as f32 / rh as f32).round() as u32)
        .max(4)
        .min(LONG_LINE_GATE);
    let squished = squish_target(target_w, squish);
    let target_w = if squished / REC_STRIDE < 32 { target_w } else { squished };
    infer_resized(rec, &rotated, target_w, vocab, remap)
}

/// Mobile `inferResizedRec` for one already-oriented crop: resize to
/// `target_w × 48`, mult-of-8 zero pad, grey `(gray/127.5)-1` 3-channel
/// input, then the native top-K decode with the full-logits fallback.
fn infer_resized(
    rec: &RecNet,
    src: &DynamicImage,
    target_w: u32,
    vocab: &[String],
    remap: &[i32],
) -> Result<PpocrResult> {
    let model_w = target_w.div_ceil(REC_STRIDE) * REC_STRIDE;
    let seq_len = (model_w / REC_STRIDE) as usize;

    let resized = src.resize_exact(target_w, REC_TARGET_H, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let input = build_rec_input(rgb.as_raw(), target_w, REC_TARGET_H, model_w);

    let k = top_k();
    let packed = rec.infer_topk(&input, model_w as usize, REC_TARGET_H as usize)?;
    let packed_ok = packed.len() == seq_len * k * 2
        && !remap.is_empty()
        && (0..seq_len * k).all(|i| {
            let id = packed[i * 2] as i32;
            id >= 0 && (id as usize) < remap.len()
        });

    if !packed_ok {
        // Full-logits fallback: downloads seqLen×numClasses floats, then
        // decodes identically (used when native top-K layout/ids disagree
        // with the loaded remap, e.g. after a head re-prune).
        if packed.len() != seq_len * k * 2 {
            eprintln!(
                "[PP-OCR] rec topK bad size {} (want {}) — trying full logits",
                packed.len(),
                seq_len * k * 2
            );
        }
        let flat = rec.infer(&input, model_w as usize, REC_TARGET_H as usize)?;
        let num_out = remap.len();
        if flat.len() != seq_len * num_out {
            bail!(
                "rec output {} floats, expected {} (seq {} × head {})",
                flat.len(),
                seq_len * num_out,
                seq_len,
                num_out
            );
        }
        let logits: Vec<Vec<f32>> = (0..seq_len)
            .map(|t| flat[t * num_out..(t + 1) * num_out].to_vec())
            .collect();
        return Ok(ctc_decode_full(vocab, remap, &logits, num_out, seq_len));
    }

    let mut top_pruned: Vec<Vec<i32>> = Vec::with_capacity(seq_len);
    let mut top_chars: Vec<Vec<(char, f32)>> = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let mut ids = Vec::with_capacity(k);
        let mut chars = Vec::with_capacity(k);
        for j in 0..k {
            let id = packed[(t * k + j) * 2] as i32;
            let val = packed[(t * k + j) * 2 + 1];
            ids.push(id);
            chars.push((decode_char(vocab, remap_class(remap, id)), val));
        }
        top_pruned.push(ids);
        top_chars.push(chars);
    }
    Ok(ctc_decode_topk(vocab, remap, &top_pruned, &top_chars, seq_len))
}

// ——— Long-line stitch (lines wider than 2000 @48px; CTC crush fix) ———

#[derive(Clone, Copy, PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

/// One inferred chunk: decoded text plus the geometry to place it globally.
/// `offset_px` is along the reading axis in full-res source pixels; `char_cols`
/// are chunk-local timesteps; `seq_len` trims mult-of-8 padding.
struct Chunk {
    chars: Vec<char>,
    char_cols: Vec<f32>,
    alts: Vec<Vec<(char, f32)>>,
    seq_len: usize,
    offset_px: f32,
}

/// Stitch a long line (>2000 targetW, mobile #24). Phase 1 chunks from the
/// reading start with a second-to-last-char anchor (10% fallback step); each
/// chunk is inferred unsquished at full resolution. Phase 2 aligns chunks by
/// identical timestep size (`cross/6` source px): the best pair in the
/// [STITCH_WINDOW] tail/head window with center distance ≤ 30px and
/// prediction overlap ≥ 0.4 wins (`score = 0.3·(1-dist/30) + 0.7·pred`); the
/// winner's alternatives merge via [interleave_alternatives] and later chars
/// re-base onto it. No winner → append chars past the last global center
/// (+10px), or single-char chunks unconditionally; total stall → +1-timestep
/// fallback offset. Double spaces collapse at the end. Mirrors mobile
/// `recognizeAndStitchLongHoriz` / `...Vert` (after the portrait rotation the
/// vertical variant is unreachable, kept for parity).
///
/// Returns None when there is nothing to stitch or a chunk failed to infer,
/// so the caller can fall through to the single-pass crush.
fn stitch_long_line(
    rec: &RecNet,
    rotated: &DynamicImage,
    axis: Axis,
    vocab: &[String],
    remap: &[i32],
) -> Result<Option<PpocrResult>> {
    let (rw, rh) = rotated.dimensions();
    let (along, cross) = if axis == Axis::X { (rw, rh) } else { (rh, rw) };
    let scale = REC_TARGET_H as f32 / cross as f32;
    let max_chunk_len = ((CHUNK_TARGET_MAX as f32 / scale) as i32).max(64) as u32;
    if max_chunk_len == 0 {
        return Ok(None);
    }
    let chunk_margin = ((cross as f32 * 0.1) as i32).max(2);

    // ——— Phase 1: chunk with anchor-driven next position ———
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut pos: i32 = 0;
    while pos < along as i32 {
        let len = (max_chunk_len as i32).min(along as i32 - pos);
        if len < 16 {
            break;
        }
        let chunk = match axis {
            Axis::X => rotated.crop_imm(pos as u32, 0, len as u32, cross),
            Axis::Y => rotated.crop_imm(0, pos as u32, cross, len as u32),
        };
        // Preserve aspect len→targetW (not stretch); cap at CHUNK_TARGET_MAX.
        let target_w = ((len as f32 * REC_TARGET_H as f32 / cross as f32).round() as i32)
            .clamp(4, CHUNK_TARGET_MAX as i32) as u32;
        let decoded = match infer_resized(rec, &chunk, target_w, vocab, remap) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[PP-OCR] rec stitch chunk w{target_w} infer failed: {e}");
                return Ok(None);
            }
        };
        let chars: Vec<char> = decoded.text.chars().collect();
        let anchor_idx = if chars.len() >= 2 { chars.len() - 2 } else { 0 };
        let anchor_t = decoded
            .char_cols
            .get(anchor_idx)
            .copied()
            .or_else(|| decoded.char_cols.last().copied());
        let seq_len = decoded.seq_len_total.max(1);
        let chars_empty = chars.is_empty();
        chunks.push(Chunk {
            chars,
            char_cols: decoded.char_cols,
            alts: decoded.alternatives,
            seq_len,
            offset_px: pos as f32,
        });
        if pos + len as i32 >= along as i32 {
            break;
        }
        // Anchor on the second-to-last char: it is fully observed, the last
        // char may be cut by the chunk edge.
        let next = match anchor_t {
            Some(anchor_t) if !chars_empty => {
                let local = ((anchor_t + 0.5) / seq_len as f32) * len as f32;
                (pos + local as i32 - chunk_margin).max(0)
            }
            _ => -1,
        };
        if next <= pos || next >= pos + len as i32 - 10 {
            pos += ((len as f32 * 0.9) as i32).max(16);
        } else {
            pos = next;
        }
    }
    if chunks.is_empty() {
        return Ok(None);
    }
    if chunks.len() == 1 {
        let c = &chunks[0];
        return Ok(Some(PpocrResult {
            text: c.chars.iter().collect(),
            alternatives: c.alts.clone(),
            char_cols: c.char_cols.clone(),
            seq_len_total: c.seq_len,
        }));
    }

    // ——— Phase 2: stitch via anchor alignment with identical timestep size ———
    // Each timestep is cross/6 px source (48px height / stride 8); all chunks
    // share the size. totalSeqLen rescales the result to the full line.
    let timestep_px = cross as f32 / 6.0;
    let total_seq_len = ((((along as f32 * REC_TARGET_H as f32) / cross as f32)
        / REC_STRIDE as f32)
        .ceil() as usize)
        .max(1);
    let centers: Vec<Vec<f32>> = chunks
        .iter()
        .map(|c| {
            c.char_cols
                .iter()
                .map(|t| c.offset_px + (t + 0.5) * timestep_px)
                .collect()
        })
        .collect();
    Ok(Some(stitch_phase2(
        &chunks,
        &centers,
        timestep_px,
        total_seq_len,
    )))
}

/// Prediction overlap 0–1 for a stitch candidate pair: 0.6–1.0 when top-1
/// agrees (scaled by top-5 overlap), 0.5 when either top-1 appears in the
/// other's top-3, else 0. Pairs below [STITCH_MIN_PRED] never stitch.
fn compare_prediction_vectors(alt1: &[(char, f32)], alt2: &[(char, f32)]) -> f32 {
    if alt1.is_empty() || alt2.is_empty() {
        return 0.0;
    }
    if alt1[0].0 == alt2[0].0 {
        let set2: Vec<char> = alt2.iter().take(5).map(|(c, _)| *c).collect();
        let m = alt1.iter().take(5).filter(|(c, _)| set2.contains(c)).count();
        return 0.6 + (m as f32 / 5.0) * 0.4;
    }
    let c1 = alt1[0].0;
    let c2 = alt2[0].0;
    if alt2.iter().take(3).any(|(c, _)| *c == c1) || alt1.iter().take(3).any(|(c, _)| *c == c2) {
        0.5
    } else {
        0.0
    }
}

/// Merge two alternative lists at a stitch anchor: shared chars average up
/// (×0.8), unique chars discount (×0.6), keep top 15 by score. First-seen
/// order is preserved for score ties (mobile LinkedHashMap + stable sort).
fn interleave_alternatives(alt1: &[(char, f32)], alt2: &[(char, f32)]) -> Vec<(char, f32)> {
    let mut merged: Vec<(char, f32)> = Vec::new();
    for &(ch, sc) in alt1 {
        match merged.iter_mut().find(|(c, _)| *c == ch) {
            Some(e) => e.1 = sc,
            None => merged.push((ch, sc)),
        }
    }
    for &(ch, sc) in alt2 {
        match merged.iter_mut().find(|(c, _)| *c == ch) {
            Some(e) if e.1 > 0.0 => e.1 = (e.1 + sc) * 0.8,
            Some(e) => e.1 = sc * 0.6,
            None => merged.push((ch, sc * 0.6)),
        }
    }
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged.truncate(top_k());
    merged
}

/// Phase 2 of the stitch: align every chunk after the first onto the
/// accumulated text (`stitchedGlobal` holds global pixel centers, `cols`
/// global timesteps). Mirrors the mobile loop statement-for-statement.
fn stitch_phase2(
    chunks: &[Chunk],
    centers: &[Vec<f32>],
    timestep_px: f32,
    total_seq_len: usize,
) -> PpocrResult {
    let mut text: Vec<char> = chunks[0].chars.clone();
    let mut alts: Vec<Vec<(char, f32)>> = chunks[0].alts.clone();
    let mut cols: Vec<f32> = chunks[0].char_cols.clone();
    let mut global: Vec<f32> = centers[0].clone();

    for i in 1..chunks.len() {
        let curr = &chunks[i];
        let curr_global = &centers[i];
        if curr.alts.is_empty() || alts.is_empty() {
            let last_px = global.last().copied().unwrap_or(-100.0);
            let offset_geom_t = curr.offset_px / timestep_px;
            for j in 0..curr.alts.len() {
                let Some(&cand_px) = curr_global.get(j) else { continue };
                if cand_px > last_px + STITCH_APPEND_GAP_PX {
                    let Some(&ch) = curr.chars.get(j) else { continue };
                    if text.last() == Some(&' ') && ch == ' ' {
                        continue;
                    }
                    let Some(&t) = curr.char_cols.get(j) else { continue };
                    text.push(ch);
                    alts.push(curr.alts[j].clone());
                    cols.push(offset_geom_t + t);
                    global.push(cand_px);
                }
            }
            continue;
        }

        let mut best_prev: i32 = -1;
        let mut best_curr: i32 = -1;
        let mut best_score = -1.0f32;
        let p_start = alts.len().saturating_sub(STITCH_WINDOW);
        let c_end = curr.alts.len().min(STITCH_WINDOW);
        for p_idx in (p_start..alts.len()).rev() {
            let p_gc = global[p_idx];
            for c_idx in 0..c_end {
                let c_gx = curr_global[c_idx];
                let dist = (p_gc - c_gx).abs();
                if dist > STITCH_MAX_DIST_PX {
                    continue;
                }
                let pred = compare_prediction_vectors(&alts[p_idx], &curr.alts[c_idx]);
                if pred < STITCH_MIN_PRED {
                    continue;
                }
                let score = (1.0 - dist / STITCH_MAX_DIST_PX) * 0.3 + pred * 0.7;
                if score > best_score {
                    best_score = score;
                    best_prev = p_idx as i32;
                    best_curr = c_idx as i32;
                }
            }
        }
        if best_prev != -1 {
            let (bp, bc) = (best_prev as usize, best_curr as usize);
            let merged = interleave_alternatives(&alts[bp], &curr.alts[bc]);
            let to_keep = bp + 1;
            text.truncate(to_keep);
            alts.truncate(to_keep);
            cols.truncate(to_keep);
            global.truncate(to_keep);
            alts[bp] = merged;
            let offset_t = cols[bp] - curr.char_cols[bc];
            let offset_px = global[bp] - curr_global[bc];
            for j in bc + 1..curr.alts.len() {
                let Some(&ch) = curr.chars.get(j) else { continue };
                if text.last() == Some(&' ') && ch == ' ' {
                    continue;
                }
                let Some(&t) = curr.char_cols.get(j) else { continue };
                text.push(ch);
                alts.push(curr.alts[j].clone());
                cols.push(t + offset_t);
                global.push(curr_global[j] + offset_px);
            }
        } else {
            let last_px = global.last().copied().unwrap_or(-100.0);
            let mut appended = 0usize;
            for j in 0..curr.alts.len() {
                let cand_px = curr_global[j];
                if cand_px > last_px + STITCH_APPEND_GAP_PX
                    || (appended == 0 && curr.alts.len() == 1)
                {
                    let Some(&ch) = curr.chars.get(j) else { continue };
                    if text.last() == Some(&' ') && ch == ' ' {
                        continue;
                    }
                    let Some(&t) = curr.char_cols.get(j) else { continue };
                    text.push(ch);
                    alts.push(curr.alts[j].clone());
                    cols.push(curr.offset_px / timestep_px + t);
                    global.push(cand_px);
                    appended += 1;
                }
            }
            if appended == 0 {
                let last_t = cols.last().copied().unwrap_or(0.0);
                let last_px2 = global.last().copied().unwrap_or(0.0);
                let fallback_offset_t = (last_t + 1.0) - curr.char_cols[0];
                let fallback_offset_px = (last_px2 + timestep_px) - curr_global[0];
                for j in 0..curr.alts.len() {
                    let Some(&ch) = curr.chars.get(j) else { continue };
                    if text.last() == Some(&' ') && ch == ' ' {
                        continue;
                    }
                    let Some(&t) = curr.char_cols.get(j) else { continue };
                    text.push(ch);
                    alts.push(curr.alts[j].clone());
                    cols.push(t + fallback_offset_t);
                    global.push(curr_global[j] + fallback_offset_px);
                }
            }
        }
    }

    // Collapse double spaces. Mobile rewrites the text only; we drop the
    // matching column/alt too, so charCols and the text stay index-aligned
    // for the char-box pass downstream.
    let mut i = 1;
    while i < text.len() {
        if text[i] == ' ' && text[i - 1] == ' ' {
            text.remove(i);
            if i < cols.len() {
                cols.remove(i);
            }
            if i < alts.len() {
                alts.remove(i);
            }
        } else {
            i += 1;
        }
    }
    PpocrResult {
        text: text.into_iter().collect(),
        alternatives: alts,
        char_cols: cols,
        seq_len_total: total_seq_len,
    }
}

/// Batched entry point kept for the existing streaming call sites: runs each
/// crop through the shared recognizer, in input order. The PC workers provide
/// the concurrency the mobile batch path gets from coroutines.
pub fn recognize_ppocr_batch(
    rec: &RecNet,
    crops: &[&DynamicImage],
    vocab: &[String],
    remap: &[i32],
) -> Result<Vec<PpocrResult>> {
    let mut results = Vec::with_capacity(crops.len());
    for crop in crops {
        results.push(recognize_crop(rec, crop, vocab, remap)?);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repo_path(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
    }

    struct Fixture {
        rec: RecNet,
        vocab: Vec<String>,
        remap: Vec<i32>,
    }

    fn fixture() -> Fixture {
        let dir = repo_path("assets/PP-OCRv6_small_ncnn");
        let rec = RecNet::create(&dir.join("rec_dyn.param"), &dir.join("rec_dyn.bin"), 64, 1)
            .expect("rec_dyn load");
        let vocab: Vec<String> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("vocab.json")).unwrap()).unwrap();
        let remap: Vec<i32> = std::fs::read_to_string(dir.join("rec_remap.txt"))
            .unwrap()
            .lines()
            .filter_map(|l| l.trim().parse::<i32>().ok())
            .collect();
        assert!(!vocab.is_empty() && !remap.is_empty());
        Fixture { rec, vocab, remap }
    }

    fn levenshtein(a: &str, b: &str) -> usize {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        let mut cur = vec![0usize; b.len() + 1];
        for i in 1..=a.len() {
            cur[0] = i;
            for j in 1..=b.len() {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        prev[b.len()]
    }

    /// Repeat a synth crop into a strip; the text inside repeats too.
    fn tile(src: &DynamicImage, count: u32, horizontal: bool) -> DynamicImage {
        let (w, h) = src.dimensions();
        let (ow, oh) = if horizontal { (w * count, h) } else { (w, h * count) };
        let mut out = image::RgbaImage::new(ow, oh);
        for i in 0..count {
            let (x, y) = if horizontal { (i * w, 0) } else { (0, i * h) };
            image::imageops::overlay(&mut out, &src.to_rgba8(), x as i64, y as i64);
        }
        DynamicImage::ImageRgba8(out)
    }

    /// Long-line stitch (#24): a strip past the 2000 targetW gate must take
    /// the chunk-and-stitch path and come back as one line, with `charCols`
    /// and the text the same length and `seqLenTotal` rescaled to the strip.
    #[test]
    fn synth_long_line_stitches() {
        let f = fixture();
        // 32 × 168px → targetW ≈ 2081, just past the 2000 gate.
        let line = image::open(repo_path("test_images/synth/line_00_h.png")).unwrap();
        let strip = tile(&line, 32, true);
        let target = strip.width() as f32 * REC_TARGET_H as f32 / strip.height() as f32;
        assert!(
            target > LONG_LINE_GATE as f32,
            "test strip must exceed the gate (got {target})"
        );
        let res = recognize_crop_with_squish(&f.rec, &strip, &f.vocab, &f.remap, 0.5).unwrap();
        let expected = "漢字".repeat(32);
        // Tile boundaries decode as spaces (the seam is wider than the
        // in-glyph gap); compare the glyphs only.
        let glyphs: String = res.text.chars().filter(|c| *c != ' ').collect();
        let dist = levenshtein(&glyphs, &expected);
        assert_eq!(
            res.char_cols.len(),
            res.text.chars().count(),
            "charCols/text lengths differ: {} vs {}",
            res.char_cols.len(),
            res.text.chars().count()
        );
        assert!(
            glyphs.chars().count() * 100 >= expected.chars().count() * 80,
            "stitched only {} of {} glyphs: {:?}",
            glyphs.chars().count(),
            expected.chars().count(),
            res.text
        );
        assert!(
            dist * 100 <= expected.chars().count() * 5,
            "stitch glyph CER {dist}/{}: {:?}",
            expected.chars().count(),
            res.text
        );
    }

    /// The overlay sizes every glyph from the line's measured pitch. On the
    /// synth set the drawn em is exactly 44px, so the char centers the
    /// recognizer produces must yield ~44 through the same normalization
    /// (halfwidth-aware median, mobile estimateEm).
    #[test]
    fn synth_pitch_recovers_drawn_em() {
        let f = fixture();
        let truth: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(repo_path("test_images/synth/truth.json")).unwrap())
                .unwrap();
        let mut checked = 0usize;
        for line in truth["lines"].as_array().unwrap() {
            if line["orientation"].as_str() != Some("H") {
                continue;
            }
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
            let img = image::open(repo_path(&format!("test_images/synth/{file}"))).unwrap();
            let crop = img.crop_imm(
                left.max(0) as u32,
                top.max(0) as u32,
                (right - left).max(1) as u32,
                (bottom - top).max(1) as u32,
            );
            let res = recognize_crop(&f.rec, &crop, &f.vocab, &f.remap).unwrap();
            assert_eq!(
                res.char_cols.len(),
                res.text.chars().count(),
                "{file}: cols/text mismatch"
            );
            let centers: Vec<f32> = res
                .char_cols
                .iter()
                .map(|&t| (t + 0.5) * (right - left) as f32 / res.seq_len_total as f32)
                .collect();
            let em = crate::util::japanese::estimate_em(&res.text, &centers);
            // 4+ char lines land within ~1% (the median averages the CTC
            // half-timestep quantization); a 2-char line has a single gap to
            // average, so it gets the wider bound.
            let tol = if res.text.chars().count() >= 4 { 0.15 } else { 0.25 };
            assert!(
                (em - 44.0).abs() <= 44.0 * tol,
                "{file}: em={em:.1} (drawn em 44) text={:?}",
                res.text
            );
            checked += 1;
        }
        assert_eq!(checked, 8, "expected 8 horizontal synth lines");
    }

    /// Mobile's androidTest synth set (truth.json + line crops): the shared
    /// ncnn backend on PC must read every line, with the per-line character
    /// error rate staying in the same band the device shows. Truth punctuation
    /// is not an exact-match target on the device either (the mobile test only
    /// aligns chars for position metrics), so this gates on CER, not equality.
    #[test]
    fn synth_rec_matches_truth() {
        let f = fixture();
        let truth: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(repo_path("test_images/synth/truth.json")).unwrap())
                .unwrap();
        let mut failures: Vec<String> = Vec::new();
        let mut report: Vec<String> = Vec::new();
        let mut total_dist = 0usize;
        let mut total_len = 0usize;
        let mut lines = 0usize;
        for line in truth["lines"].as_array().unwrap() {
            let file = line["file"].as_str().unwrap();
            let expect = line["text"].as_str().unwrap();
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
            let img = image::open(repo_path(&format!("test_images/synth/{file}"))).unwrap();
            let crop = img.crop_imm(
                left.max(0) as u32,
                top.max(0) as u32,
                (right - left).max(1) as u32,
                (bottom - top).max(1) as u32,
            );
            let res = recognize_crop(&f.rec, &crop, &f.vocab, &f.remap).unwrap();
            let dist = levenshtein(&res.text, expect);
            let len = expect.chars().count();
            total_dist += dist;
            total_len += len;
            lines += 1;
            report.push(format!("{file}: CER {dist}/{len} got={:?}", res.text));
            if res.text.is_empty() || dist * 100 > len * 30 {
                failures.push(format!("{file}: got {:?}, want {:?}", res.text, expect));
            }
        }
        assert_eq!(lines, 16, "expected the full synth set");
        assert!(
            failures.is_empty(),
            "rec parity failures ({}):\n{}",
            failures.len(),
            failures.join("\n")
        );
        assert!(
            total_dist * 100 <= total_len * 12,
            "mean CER {}% over {} chars:\n{}",
            total_dist as f64 * 100.0 / total_len as f64,
            total_len,
            report.join("\n")
        );
    }

    /// The native top-K path and the full-logits fallback must agree on text
    /// (the fallback is what runs when the remap and the model disagree).
    #[test]
    fn synth_topk_and_full_logits_agree() {
        let f = fixture();
        let img = image::open(repo_path("test_images/synth/line_03_h.png")).unwrap();
        let (w, h) = img.dimensions();
        let target_w = ((w as f32 * REC_TARGET_H as f32 / h as f32).round() as u32).max(4);
        let model_w = target_w.div_ceil(REC_STRIDE) * REC_STRIDE;
        let resized = img.resize_exact(target_w, REC_TARGET_H, image::imageops::FilterType::Triangle);
        let input = build_rec_input(resized.to_rgb8().as_raw(), target_w, REC_TARGET_H, model_w);

        // Squish off for a direct preprocessing comparison.
        let via_decode =
            recognize_crop_with_squish(&f.rec, &img, &f.vocab, &f.remap, 1.0).unwrap();

        let k = top_k();
        let seq_len = (model_w / REC_STRIDE) as usize;
        let packed = f.rec.infer_topk(&input, model_w as usize, REC_TARGET_H as usize).unwrap();
        assert_eq!(packed.len(), seq_len * k * 2, "native top-K layout");
        let flat = f.rec.infer(&input, model_w as usize, REC_TARGET_H as usize).unwrap();
        assert_eq!(flat.len(), seq_len * f.remap.len(), "full logits layout");

        let num_out = f.remap.len();
        let logits: Vec<Vec<f32>> = (0..seq_len)
            .map(|t| flat[t * num_out..(t + 1) * num_out].to_vec())
            .collect();
        let from_full = ctc_decode_full(&f.vocab, &f.remap, &logits, num_out, seq_len);
        assert_eq!(via_decode.text, from_full.text);
    }
}
