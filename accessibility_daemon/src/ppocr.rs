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
/// Single-pass targetW cap; the mobile long-line stitch gate. PC has no
/// stitch path, so this is only a sanity clamp (a 2000-px-wide line at 48px
/// height is as far as exact-width CTC was validated).
const LONG_LINE_GATE: u32 = 2000;
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

/// Run the dynamic-width recognizer on a single line crop. Mirrors mobile
/// `inferResizedRec`: portrait crops rotate 270°, the resize target is the
/// aspect-preserving width, squish applies before inference (with the mobile
/// >=32-timestep floor), and the model width snaps up to a multiple of 8
/// (zero-padded). `remap` length is the pruned CTC head width.
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

    // Dynamic width (#23): exact targetW capped at LONG_LINE_GATE, then
    // squish (#24) applied pre-inference; model width snaps to mult-of-8.
    let target_w = ((rw as f32 * REC_TARGET_H as f32 / rh as f32).round() as u32)
        .max(4)
        .min(LONG_LINE_GATE);
    let squished = squish_target(target_w, squish);
    let target_w = if squished / REC_STRIDE < 32 { target_w } else { squished };
    let model_w = target_w.div_ceil(REC_STRIDE) * REC_STRIDE;
    let seq_len = (model_w / REC_STRIDE) as usize;

    let resized = rotated.resize_exact(target_w, REC_TARGET_H, image::imageops::FilterType::Triangle);
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
