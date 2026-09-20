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
    /// Source pixels per timestep along the reading axis (of the oriented
    /// crop). Exact, unlike `seq_len_total` which counts padded model
    /// timesteps: the content only spans `target_w` of `model_w`.
    pub step_px: f32,
    /// Top-K alternatives for EVERY timestep, blanks included, descending by
    /// score — mobile `PPOcrResult.rawAlternatives`. One inner vec per
    /// timestep; blank class 0 decodes to `'\u{3000}'` like mobile, so
    /// `re_decode_raw_alternatives` can tell blank from space. The cache that
    /// makes a re-decode possible without re-running the model.
    pub raw_alternatives: Vec<Vec<(char, f32)>>,
}

impl PpocrResult {
    fn empty() -> Self {
        PpocrResult {
            text: String::new(),
            alternatives: Vec::new(),
            char_cols: Vec::new(),
            seq_len_total: 0,
            step_px: 0.0,
            raw_alternatives: Vec::new(),
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

/// Mobile `OcrEngine.top15Alternatives`' heap order: a Java `PriorityQueue`
/// (binary min-heap by score, lowest index not privileged) fed every class id
/// and polled whenever it exceeds TOP_K, then read in backing-array order.
/// Reproduced here because the final stable sort's tie order depends on it.
struct JavaMinHeap<'a> {
    data: Vec<usize>,
    score: &'a [f32],
}

impl JavaMinHeap<'_> {
    fn add(&mut self, k: usize) {
        self.data.push(k);
        let mut child = self.data.len() - 1;
        while child > 0 {
            let parent = (child - 1) / 2;
            // Java siftUp breaks on compare(key, parent) >= 0.
            if self.score[self.data[child]] >= self.score[self.data[parent]] {
                break;
            }
            self.data.swap(child, parent);
            child = parent;
        }
    }

    fn poll(&mut self) -> Option<usize> {
        let n = self.data.len();
        if n == 0 {
            return None;
        }
        let result = self.data[0];
        let last = self.data.pop().unwrap();
        if n > 1 {
            self.data[0] = last;
            let mut parent = 0usize;
            loop {
                let left = 2 * parent + 1;
                if left >= self.data.len() {
                    break;
                }
                let mut child = left;
                let right = left + 1;
                if right < self.data.len()
                    && self.score[self.data[right]] < self.score[self.data[left]]
                {
                    child = right;
                }
                // Java siftDown breaks on compare(key, child) <= 0.
                if self.score[self.data[parent]] <= self.score[self.data[child]] {
                    break;
                }
                self.data.swap(parent, child);
                parent = child;
            }
        }
        Some(result)
    }
}

/// Mobile `top15Alternatives` index order: heap selection + stable
/// descending sort (see `JavaMinHeap`).
fn java_topk_order(slice: &[f32], k: usize) -> Vec<usize> {
    let mut heap = JavaMinHeap { data: Vec::new(), score: slice };
    for i in 0..slice.len() {
        heap.add(i);
        if heap.data.len() > k {
            heap.poll();
        }
    }
    let mut order = heap.data;
    order.sort_by(|&a, &b| {
        slice[b]
            .partial_cmp(&slice[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(k);
    order
}

/// Top-15 char alternatives for one timestep from full logits, in mobile's
/// exact order: heap selection, then a stable descending sort — so equal
/// scores keep the Java heap's backing-array order (D6.1).
fn top15_alternatives(vocab: &[String], remap: &[i32], slice: &[f32]) -> Vec<(char, f32)> {
    java_topk_order(slice, top_k())
        .into_iter()
        .map(|idx| (decode_char(vocab, remap_class(remap, idx as i32)), slice[idx]))
        .collect()
}

/// Greedy CTC decode from native top-15 lists (#42): entry 0 is the argmax
/// (native emits descending, lowest-id wins ties); blank/space tests use the
/// remapped pruned indices. Mirrors mobile `ctcDecodeTopK` with
/// blankThreshold 0 (pure greedy). `top_chars` is kept as the result's
/// `raw_alternatives` (mobile passes the same lists to `ctcDecodeTopK` and
/// copies them onto the result), so it is taken by value.
fn ctc_decode_topk(
    vocab: &[String],
    remap: &[i32],
    top_pruned: &[Vec<i32>],
    top_chars: Vec<Vec<(char, f32)>>,
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
        step_px: 0.0,
        raw_alternatives: top_chars,
    }
}

/// Greedy CTC decode from full logits, mobile `ctcDecode` with
/// blankThreshold 0. Builds `raw_alternatives` per timestep exactly like the
/// mobile fallback (`top15Alternatives(cropLogits[t])`), so a re-decode works
/// the same on both paths.
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
    let mut raw_alternatives: Vec<Vec<(char, f32)>> = Vec::with_capacity(seq_len);
    let mut prev_class: i32 = 0;

    let wval = |tt: usize, cls: i32| -> Option<f32> {
        let slice = logits.get(tt)?;
        slice.get(cls as usize).copied()
    };

    for t in 0..seq_len {
        let Some(slice) = logits.get(t) else {
            raw_alternatives.push(Vec::new());
            continue;
        };
        if slice.len() < num_classes {
            raw_alternatives.push(Vec::new());
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
        raw_alternatives.push(indexed.clone());

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
        step_px: 0.0,
        raw_alternatives,
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
    infer_resized_with_topk(rec, src, target_w, vocab, remap, None)
}

/// `topk_override` is a test seam for the native top-K call (D5.1: a top-K
/// failure must degrade to full logits, not drop the line).
fn infer_resized_with_topk(
    rec: &RecNet,
    src: &DynamicImage,
    target_w: u32,
    vocab: &[String],
    remap: &[i32],
    topk_override: Option<Result<Vec<f32>>>,
) -> Result<PpocrResult> {
    let model_w = target_w.div_ceil(REC_STRIDE) * REC_STRIDE;
    let seq_len = (model_w / REC_STRIDE) as usize;
    // Exact source px per timestep: the resized content spans `target_w`
    // model px (not the zero-padded `model_w`), over `src.width()` source px.
    let step_px = src.width() as f32 * REC_STRIDE as f32 / target_w as f32;

    let resized = src.resize_exact(target_w, REC_TARGET_H, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let input = build_rec_input(rgb.as_raw(), target_w, REC_TARGET_H, model_w);

    let k = top_k();
    // Mobile degrades to full logits when the native top-K entry fails
    // (`UnsatisfiedLinkError` there); a top-K error must not drop the line.
    let packed = match topk_override {
        Some(result) => result,
        None => rec.infer_topk(&input, model_w as usize, REC_TARGET_H as usize),
    };
    let packed = match packed {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("[PP-OCR] rec topK failed ({e}) — trying full logits");
            None
        }
    };
    let packed_ok = packed.as_ref().is_some_and(|packed| {
        packed.len() == seq_len * k * 2
            && !remap.is_empty()
            && (0..seq_len * k).all(|i| {
                let id = packed[i * 2] as i32;
                id >= 0 && (id as usize) < remap.len()
            })
    });

    if !packed_ok {
        // Full-logits fallback: downloads seqLen×numClasses floats, then
        // decodes identically (used when native top-K fails or its
        // layout/ids disagree with the loaded remap, e.g. after a head
        // re-prune).
        if let Some(packed) = &packed {
            if packed.len() != seq_len * k * 2 {
                eprintln!(
                    "[PP-OCR] rec topK bad size {} (want {}) — trying full logits",
                    packed.len(),
                    seq_len * k * 2
                );
            }
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
        let mut decoded = ctc_decode_full(vocab, remap, &logits, num_out, seq_len);
        decoded.step_px = step_px;
        return Ok(decoded);
    }

    let packed = packed.expect("packed_ok checked above");
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
    let mut decoded = ctc_decode_topk(vocab, remap, &top_pruned, top_chars, seq_len);
    decoded.step_px = step_px;
    Ok(decoded)
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
    /// Full per-timestep top-K of the chunk (mobile `rawAltsPerTimestep`);
    /// phase 2 concatenates every chunk's list onto the result, like mobile.
    raw_alts: Vec<Vec<(char, f32)>>,
    seq_len: usize,
    /// Source px per chunk-local timestep (for the single-chunk fast path).
    step_px: f32,
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
            raw_alts: decoded.raw_alternatives,
            seq_len,
            step_px: decoded.step_px,
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
            step_px: c.step_px,
            raw_alternatives: c.raw_alts.clone(),
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
    // Mobile `stitchedRawAll`: per-chunk raw lists concatenated, never
    // truncated/re-based (the walk in `re_decode_raw_alternatives` simply
    // mismatches the stitched text and yields nothing, like mobile's).
    let mut stitched_raw: Vec<Vec<(char, f32)>> = chunks[0].raw_alts.clone();

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
            stitched_raw.extend(curr.raw_alts.iter().cloned());
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
        stitched_raw.extend(curr.raw_alts.iter().cloned());
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
        step_px: timestep_px,
        raw_alternatives: stitched_raw,
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

// ——— Re-decode from cached raw alternatives (no model) ———

/// Model-free re-decode output: text, per-emitted-character alternatives and
/// fractional CTC columns, ready to rebuild a `LineResult`.
#[allow(dead_code)] // only reached through `DetectedAnnotation::re_decode_line` (unwired yet)
#[derive(Debug, Clone)]
pub struct ReDecodedLine {
    pub text: String,
    pub alternatives: Vec<Vec<(char, f32)>>,
    pub char_cols: Vec<f32>,
}

/// Mobile `OcrEngine.reDecodeLineResult`'s walk, without the `LineResult`
/// plumbing: greedy CTC over the cached
/// [`raw_alternatives`](PpocrResult::raw_alternatives) — entry 0 is the
/// argmax, `'\u{3000}'` is the blank marker (so blank resets the collapse
/// state), a space never collapses, any other character collapses only
/// against the immediately preceding emitted character. Emits at most one
/// character per timestep; `alternatives` stays index-aligned with `text`.
/// Columns carry the winner's own score at the neighbouring timesteps
/// (matched by character, absent → integer column), like the live decode —
/// except that the live full-logits decode reads the neighbour's whole row,
/// so it can interpolate from a class outside the neighbour's top-K while
/// this walk cannot (mobile's `ctcDecode.wval` vs `reDecodeLineResult.wval`).
/// Vertical lines get the emit path's punctuation normalisation applied to
/// the text and to every alternative entry, so cached pre-fix raw data cannot
/// reintroduce ASCII `?`/horizontal `…`.
///
/// `None` when nothing is cached (mobile returns the line unchanged).
#[allow(dead_code)] // only reached through `DetectedAnnotation::re_decode_line` (unwired yet)
pub fn re_decode_raw_alternatives(
    raw: &[Vec<(char, f32)>],
    is_vertical: bool,
) -> Option<ReDecodedLine> {
    if raw.is_empty() {
        return None;
    }
    let mut text = String::new();
    let mut new_alts: Vec<Vec<(char, f32)>> = Vec::new();
    let mut char_cols: Vec<f32> = Vec::new();
    let mut prev_char: Option<char> = None;

    // Winner score at a neighbouring timestep, matched by character
    // (mobile `wval`; first match wins).
    let wval = |t: usize, ch: char| -> Option<f32> {
        raw.get(t)?
            .iter()
            .find(|(c, _)| *c == ch)
            .map(|(_, s)| *s)
    };
    let frac_for = |t: usize, ch: char, v1: f32| -> f32 {
        let w0 = if t > 0 { wval(t - 1, ch) } else { None };
        match (w0, wval(t + 1, ch)) {
            (Some(w0), Some(w2)) => peak_offset(w0, v1, w2),
            _ => 0.0,
        }
    };

    for (t, alts) in raw.iter().enumerate() {
        let Some(&(top_char, top_score)) = alts.first() else {
            continue;
        };
        if top_char == '\u{3000}' {
            // Blank: resets the repeat state but is never emitted.
            prev_char = None;
        } else if top_char == ' ' {
            text.push(' ');
            prev_char = Some(' ');
            char_cols.push(t as f32 + frac_for(t, ' ', top_score));
            new_alts.push(alts.clone());
        } else if prev_char == Some(top_char) {
            // collapse repeat
        } else {
            text.push(top_char);
            char_cols.push(t as f32 + frac_for(t, top_char, top_score));
            new_alts.push(alts.clone());
            prev_char = Some(top_char);
        }
    }

    if is_vertical {
        text = crate::util::japanese::vertical_punctuation(&text);
        for alts in new_alts.iter_mut() {
            for (c, _) in alts.iter_mut() {
                *c = crate::util::japanese::vertical_punctuation_char(*c);
            }
        }
    }
    Some(ReDecodedLine {
        text,
        alternatives: new_alts,
        char_cols,
    })
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
        // Stitched lines still cache raw alternatives (mobile concatenates
        // every chunk's per-timestep lists onto the result).
        assert!(
            !res.raw_alternatives.is_empty(),
            "stitched lines must keep the raw cache"
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
                .map(|&t| (t + 0.5) * res.step_px)
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

        // Both paths cache one descending top-K list per timestep, and the
        // re-decode walks it back to the same characters — the raw shape the
        // mobile re-decode contract assumes.
        assert_eq!(via_decode.raw_alternatives.len(), via_decode.seq_len_total);
        for raw in &via_decode.raw_alternatives {
            assert_eq!(raw.len(), k, "one top-K list per timestep");
            assert!(raw.windows(2).all(|w| w[0].1 >= w[1].1), "descending");
        }
        assert_eq!(from_full.raw_alternatives.len(), seq_len);
        let re = re_decode_raw_alternatives(&via_decode.raw_alternatives, false).unwrap();
        assert_eq!(re.text, via_decode.text);
        assert_eq!(re.alternatives, via_decode.alternatives);
        // The top-K live decode also matches neighbours inside the top-K, so
        // its columns round-trip exactly on this line.
        assert_eq!(re.char_cols, via_decode.char_cols);
        let re_full = re_decode_raw_alternatives(&from_full.raw_alternatives, false).unwrap();
        assert_eq!(re_full.text, from_full.text);
        assert_eq!(re_full.alternatives, from_full.alternatives);
        // The full-logits live decode reads the neighbour's WHOLE row, so it
        // can interpolate from a class the neighbour's top-15 dropped; the
        // re-decode only has the cache and falls back to the integer column
        // (mobile's `ctcDecode.wval` vs `reDecodeLineResult.wval` asymmetry).
        // Both columns still name the same winner timestep.
        assert_eq!(re_full.char_cols.len(), from_full.char_cols.len());
        for (a, b) in re_full.char_cols.iter().zip(&from_full.char_cols) {
            assert!((a - b).abs() < 1.0, "re-decode column {a} vs live {b}");
        }
    }

    /// D6.1: mobile's alternative tie order is the Java PriorityQueue's
    /// backing-array order after a stable descending sort. Expected orders
    /// were generated by running the exact Android code (PriorityQueue +
    /// `sortedByDescending`) on the JDK.
    #[test]
    fn topk_order_matches_java_priority_queue() {
        assert_eq!(java_topk_order(&[0.5, 0.5, 0.5], 2), vec![2, 1]);
        assert_eq!(java_topk_order(&[0.1, 0.9, 0.5, 0.5], 3), vec![1, 3, 2]);
        assert_eq!(
            java_topk_order(
                &[
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 0.9, 0.9,
                    0.5, 0.5, 0.5, 0.2, 0.2, 0.1
                ],
                15
            ),
            vec![9, 10, 11, 8, 12, 13, 7, 6, 5, 14, 16, 4, 15, 3, 2]
        );
        assert_eq!(
            java_topk_order(
                &[
                    0.7, 0.7, 0.3, 0.3, 0.9, 0.9, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.2, 0.8,
                    0.8, 0.4, 0.4, 0.6, 0.6, 0.6, 0.05, 0.95, 0.95, 0.35, 0.35, 0.75, 0.75,
                    0.15, 0.15, 0.55, 0.55, 0.25, 0.65, 0.65, 0.45, 0.45, 0.85, 0.85, 0.0,
                    1.0
                ],
                15
            ),
            vec![39, 21, 22, 4, 5, 36, 37, 13, 14, 26, 25, 1, 0, 33, 32]
        );
    }

    /// D5.1: a failing native top-K entry degrades to full logits and still
    /// decodes the line (mobile catches `UnsatisfiedLinkError` there).
    #[test]
    fn topk_failure_falls_back_to_full_logits() {
        let f = fixture();
        let img = image::open(repo_path("test_images/synth/line_03_h.png")).unwrap();
        let (w, h) = img.dimensions();
        let target_w = ((w as f32 * REC_TARGET_H as f32 / h as f32).round() as u32).max(4);
        let normal = infer_resized(&f.rec, &img, target_w, &f.vocab, &f.remap).unwrap();
        let fallback = infer_resized_with_topk(
            &f.rec,
            &img,
            target_w,
            &f.vocab,
            &f.remap,
            Some(Err(anyhow::anyhow!("simulated native top-K failure"))),
        )
        .unwrap();
        assert!(!fallback.text.is_empty(), "fallback dropped the line");
        assert_eq!(normal.text, fallback.text);
        // The fallback caches a full raw list too, and it re-decodes to the
        // same line (mobile builds `rawAlts` on both paths).
        assert_eq!(fallback.raw_alternatives.len(), fallback.seq_len_total);
        assert!(fallback.raw_alternatives.iter().all(|r| r.len() == top_k()));
        let re = re_decode_raw_alternatives(&fallback.raw_alternatives, false).unwrap();
        assert_eq!(re.text, fallback.text);
        assert_eq!(re.alternatives, fallback.alternatives);
    }

    // ——— raw alternatives + re-decode (no model) ———

    /// One synthetic CTC timestep: `(class, char, score)` entries, entry 0 the
    /// argmax. Class ids drive the live decode (blank 0, space 18709); the
    /// chars drive the re-decode walk — mobile's `topPruned`/`topChars` split.
    fn step(entries: &[(i32, char, f32)]) -> (Vec<i32>, Vec<(char, f32)>) {
        (
            entries.iter().map(|&(c, _, _)| c).collect(),
            entries.iter().map(|&(_, ch, s)| (ch, s)).collect(),
        )
    }

    const TEST_VOCAB: [&str; 5] = ["あ", "い", "う", "え", "お"];

    fn decode_topk(steps: &[(Vec<i32>, Vec<(char, f32)>)]) -> PpocrResult {
        let vocab: Vec<String> = TEST_VOCAB.iter().map(|s| s.to_string()).collect();
        let remap: Vec<i32> = (0..=TEST_VOCAB.len() as i32).collect();
        let top_pruned: Vec<Vec<i32>> = steps.iter().map(|(p, _)| p.clone()).collect();
        let top_chars: Vec<Vec<(char, f32)>> = steps.iter().map(|(_, c)| c.clone()).collect();
        ctc_decode_topk(&vocab, &remap, &top_pruned, top_chars, steps.len())
    }

    fn re_decode(steps: &[(Vec<i32>, Vec<(char, f32)>)], is_vertical: bool) -> ReDecodedLine {
        let raw: Vec<Vec<(char, f32)>> = steps.iter().map(|(_, c)| c.clone()).collect();
        re_decode_raw_alternatives(&raw, is_vertical).expect("raw is not empty")
    }

    /// The top-K decode's text/columns and the raw cache it leaves behind: one
    /// descending list per timestep, blanks included, entry 0 the argmax.
    #[test]
    fn topk_decode_pins_text_cols_and_raw_shape() {
        let steps = [
            step(&[(1, 'あ', 0.9), (2, 'い', 0.5), (0, '\u{3000}', 0.1)]),
            step(&[(1, 'あ', 0.8), (2, 'い', 0.3), (0, '\u{3000}', 0.2)]),
            step(&[(0, '\u{3000}', 0.95), (1, 'あ', 0.4), (2, 'い', 0.2)]),
            step(&[(1, 'あ', 0.7), (3, 'う', 0.6), (0, '\u{3000}', 0.1)]),
            step(&[(18709, ' ', 0.6), (1, 'あ', 0.5), (0, '\u{3000}', 0.1)]),
            step(&[(18709, ' ', 0.5), (2, 'い', 0.4), (0, '\u{3000}', 0.1)]),
            step(&[(2, 'い', 0.9), (0, '\u{3000}', 0.3), (1, 'あ', 0.1)]),
        ];
        let res = decode_topk(&steps);
        assert_eq!(res.text, "ああ  い");
        assert_eq!(res.char_cols, vec![0.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(res.alternatives.len(), 5);

        assert_eq!(res.raw_alternatives.len(), steps.len());
        for (t, raw) in res.raw_alternatives.iter().enumerate() {
            assert_eq!(raw.len(), steps[t].1.len(), "t{t} keeps every entry");
            assert!(raw.windows(2).all(|w| w[0].1 >= w[1].1), "t{t} descending");
        }
        assert_eq!(
            res.raw_alternatives[2][0],
            ('\u{3000}', 0.95),
            "the blank is cached as the ideographic space"
        );
        // Every emitted column's entry 0 is the character the decode emitted.
        let top0: Vec<char> = res.raw_alternatives.iter().map(|r| r[0].0).collect();
        for (ch, col) in res.text.chars().zip(&res.char_cols) {
            assert_eq!(top0[*col as usize], ch);
        }
    }

    /// The re-decode's fractional columns match the live decode's: the
    /// winner's own score at the neighbouring timesteps (mobile
    /// `wval`/`peakOffset`), absent on either side → integer column.
    #[test]
    fn re_decode_pins_fractional_columns() {
        let steps = [
            step(&[(2, 'い', 1.0), (1, 'あ', 1.0)]),
            step(&[(1, 'あ', 3.0), (2, 'い', 0.5)]),
            step(&[(1, 'あ', 0.0), (2, 'い', 0.5)]),
        ];
        let live = decode_topk(&steps);
        assert_eq!(live.text, "いあ");
        assert_eq!(live.char_cols, vec![0.0, 0.9]);

        let re = re_decode(&steps, false);
        assert_eq!(re.text, live.text);
        assert_eq!(re.char_cols, live.char_cols);
        assert_eq!(re.alternatives, live.alternatives);
    }

    /// Mobile's full-logits live decode reads the neighbour's whole row
    /// (`ctcDecode.wval` indexes `cropLogits[t][cls]`), so it can interpolate
    /// from a class the neighbour's top-K dropped; the re-decode only has the
    /// cached top-K and falls back to the integer column. Both columns still
    /// name the same winner timestep — the same asymmetry mobile ships.
    #[test]
    fn full_logits_interpolation_can_exceed_the_raw_cache() {
        let vocab: Vec<String> = TEST_VOCAB.iter().map(|s| s.to_string()).collect();
        let remap: Vec<i32> = (0..20).collect();
        let num_classes = 20;
        // t0: い wins; あ (0.49) ranks below fifteen 0.5 fillers → not cached.
        let mut t0 = vec![0.0f32; num_classes];
        t0[2] = 1.0;
        for c in 3..18 {
            t0[c] = 0.5;
        }
        t0[1] = 0.49;
        // t1: あ wins 3.0; its neighbour scores (0.49, 0.0) interpolate it.
        let mut t1 = vec![0.05f32; num_classes];
        t1[1] = 3.0;
        // t2: い wins; あ (0.0) is not in this row's top-K either.
        let mut t2 = vec![0.05f32; num_classes];
        t2[2] = 0.9;
        t2[0] = 0.1;
        t2[1] = 0.0;
        let logits = vec![t0, t1, t2];

        let live = ctc_decode_full(&vocab, &remap, &logits, num_classes, 3);
        assert_eq!(live.text, "いあい");
        assert!(
            live.raw_alternatives[0].iter().all(|(c, _)| *c != 'あ'),
            "あ must be outside t0's top-K for this pin"
        );
        let expected = 1.0 + peak_offset(0.49, 3.0, 0.0);
        assert_eq!(live.char_cols[1], expected, "live interpolates the whole row");

        let re = re_decode_raw_alternatives(&live.raw_alternatives, false).unwrap();
        assert_eq!(re.text, live.text);
        assert_eq!(re.char_cols[1], 1.0, "no neighbour entry → integer column");
    }

    /// Mobile's documented walk (the Kotlin `timestepColumns` fixture): a
    /// blank resets the repeat state, a space never collapses, and the
    /// alternatives stay one entry per emitted character.
    #[test]
    fn re_decode_walk_matches_mobile() {
        let steps = [
            step(&[(1, 'あ', 0.90), (0, '\u{3000}', 0.10)]),
            step(&[(1, 'あ', 0.80), (0, '\u{3000}', 0.10)]),
            step(&[(0, '\u{3000}', 0.90)]),
            step(&[(1, 'あ', 0.70), (0, '\u{3000}', 0.10)]),
            step(&[(18709, ' ', 0.60), (0, '\u{3000}', 0.10)]),
            step(&[(18709, ' ', 0.50), (0, '\u{3000}', 0.10)]),
            step(&[(2, 'い', 0.90), (0, '\u{3000}', 0.10)]),
        ];
        let re = re_decode(&steps, false);
        assert_eq!(re.text, "ああ  い");
        assert_eq!(re.char_cols, vec![0.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(re.alternatives.len(), 5);
        assert_eq!(re.alternatives[1], steps[3].1, "repeat after a blank is emitted");
        assert!(re_decode_raw_alternatives(&[], false).is_none());
    }

    /// Vertical re-decode normalises the text and every alternative entry
    /// (mobile maps the whole list), so cached pre-fix raw data cannot
    /// reintroduce ASCII `?` or horizontal `…`/`‥`.
    #[test]
    fn re_decode_normalises_vertical_punctuation() {
        let steps = [
            step(&[(3, '?', 0.9), (0, '\u{3000}', 0.1)]),
            step(&[(4, '…', 0.8), (0, '\u{3000}', 0.1)]),
            step(&[(1, 'あ', 0.9), (3, '?', 0.5)]),
        ];
        let horizontal = re_decode(&steps, false);
        assert_eq!(horizontal.text, "?…あ");

        let vertical = re_decode(&steps, true);
        assert_eq!(vertical.text, "？︙あ");
        assert_eq!(vertical.alternatives[0][0].0, '？');
        assert_eq!(vertical.alternatives[1][0].0, '︙');
        assert_eq!(
            vertical.alternatives[2][1].0,
            '？',
            "non-top entries are normalised too"
        );
    }
}
