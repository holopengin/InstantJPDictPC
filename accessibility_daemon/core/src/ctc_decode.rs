//! Pure PP-OCR CTC decoding over already-pruned candidate rows.
//!
//! The native recognizer still owns inference and candidate extraction. This
//! module owns everything after that boundary: class remapping, vocabulary
//! lookup, Java-compatible top-15 ordering, greedy blank/repeat/space collapse,
//! and fractional peak interpolation. Keeping those stages ungated lets Android
//! and the desktop call the same implementation without pulling ncnn into a
//! mobile build.

use crate::char_boxes::peak_offset;

/// The recognizer emits this many alternatives per timestep.
pub const TOP_K: usize = 15;
/// Softmax row 0 is blank.
const BLANK_CLASS: i32 = 0;
/// Last ordinary vocabulary-backed class; decoding it yields an ideographic space.
const FULLWIDTH_SPACE_CLASS: i32 = 18708;
/// The explicit half-width-space class.
const SPACE_CLASS: i32 = 18709;

/// The pure output of one CTC decode.
///
/// This deliberately has no image, model, or pixel geometry. The desktop
/// wrapper adds its source-pixels-per-timestep value after decoding.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcDecodeResult {
    pub text: String,
    /// Top-K alternatives for emitted characters only.
    pub alternatives: Vec<Vec<(char, f32)>>,
    /// Fractional timestep column for each emitted character.
    pub char_cols: Vec<f32>,
    pub seq_len_total: usize,
    /// Top-K alternatives for every timestep, including blanks.
    pub raw_alternatives: Vec<Vec<(char, f32)>>,
    /// The timestep each emitted character's alternatives came from, so
    /// `alternatives[j] == raw_alternatives[alt_rows[j]]`.
    ///
    /// The greedy walk pushes the row it just built into both lists in the
    /// same iteration, so the indices are recorded as it goes: strictly
    /// ascending, in range, and exact for any input (including the malformed
    /// ones that push an empty raw row). They let a boundary carry the emitted
    /// list as one integer per character instead of re-crossing `top_k` cells
    /// it already received — see the Android shim's flat result record.
    pub alt_rows: Vec<usize>,
}

impl CtcDecodeResult {
    fn empty(seq_len_total: usize) -> Self {
        Self {
            text: String::new(),
            alternatives: Vec::new(),
            char_cols: Vec::new(),
            seq_len_total,
            raw_alternatives: Vec::new(),
            alt_rows: Vec::new(),
        }
    }

    /// Vertical-line punctuation at emit time (#56, #63).
    ///
    /// PP-OCR emits ASCII `?` where Japanese text wants fullwidth `？`, and the
    /// horizontal `…`/`‥` where vertical text wants `︙`/`︰`; all three lack a
    /// `vert` alternate and mis-centre in the vertical em box. Dictionary lookup
    /// folds them back ([`crate::util::japanese::normalize`]), so the
    /// substitution is lookup-safe.
    ///
    /// The rule is a fixed **1:1** character substitution, so folding it in
    /// here is byte-identical to mapping the rows on the host after the decode
    /// and needs no second pass: `char_cols`, `seq_len_total`, `alt_rows` and
    /// every list's shape are untouched, and the mapping is idempotent (a
    /// vertical line whose rows are already normalised is left alone).
    /// Horizontal lines do not ask for this — a horizontal `?` stays `?`.
    pub fn apply_vertical_punctuation(&mut self) {
        self.text = crate::util::japanese::vertical_punctuation(&self.text);
        for row in self
            .alternatives
            .iter_mut()
            .chain(self.raw_alternatives.iter_mut())
        {
            for (c, _) in row.iter_mut() {
                *c = crate::util::japanese::vertical_punctuation_char(*c);
            }
        }
    }
}

/// Pruned class id -> original class id, with the shipped identity fallback.
#[inline]
pub fn remap_class(remap: &[i32], pruned_idx: i32) -> i32 {
    remap
        .get(pruned_idx as usize)
        .copied()
        .unwrap_or(pruned_idx)
}

/// Original class id -> its first vocabulary character.
pub fn decode_char(vocab: &[String], class_idx: i32) -> char {
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

/// Reproduces the exact selection/tie order of Android's
/// `PriorityQueue<Int>(compareBy { slice[it] })` followed by
/// `sortedByDescending`.
///
/// The stable final sort is not sufficient on its own: its input order is the
/// min-heap's backing-array order, which differs from a plain index sort when
/// scores tie.
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

/// Java `PriorityQueue` selection followed by its stable descending sort.
pub(crate) fn java_topk_order(slice: &[f32], k: usize) -> Vec<usize> {
    let mut heap = JavaMinHeap {
        data: Vec::new(),
        score: slice,
    };
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

/// Top-15 decoded alternatives for one full-logits timestep.
pub fn top15_alternatives(vocab: &[String], remap: &[i32], slice: &[f32]) -> Vec<(char, f32)> {
    java_topk_order(slice, TOP_K)
        .into_iter()
        .map(|idx| {
            (
                decode_char(vocab, remap_class(remap, idx as i32)),
                slice[idx],
            )
        })
        .collect()
}

/// The one greedy state machine shared by packed and full-logits callers.
fn ctc_decode_candidates<C, A, L, R>(
    vocab: &[String],
    seq_len: usize,
    mut class_at: C,
    mut alternatives_at: A,
    mut left_score_at: L,
    mut right_score_at: R,
) -> CtcDecodeResult
where
    C: FnMut(usize) -> Option<i32>,
    A: FnMut(usize) -> Option<Vec<(char, f32)>>,
    L: FnMut(usize) -> Option<f32>,
    R: FnMut(usize) -> Option<f32>,
{
    let mut text = String::new();
    let mut alternatives: Vec<Vec<(char, f32)>> = Vec::new();
    let mut char_cols: Vec<f32> = Vec::new();
    let mut raw_alternatives: Vec<Vec<(char, f32)>> = Vec::with_capacity(seq_len);
    let mut alt_rows: Vec<usize> = Vec::new();
    let mut prev_class = BLANK_CLASS;

    for t in 0..seq_len {
        let (Some(class_idx), Some(indexed)) = (class_at(t), alternatives_at(t)) else {
            raw_alternatives.push(Vec::new());
            continue;
        };
        if indexed.is_empty() {
            raw_alternatives.push(Vec::new());
            continue;
        }

        let max_val = indexed[0].1;
        let t_frac = if let (Some(w0), Some(w2)) = (left_score_at(t), right_score_at(t)) {
            t as f32 + peak_offset(w0, max_val, w2)
        } else {
            t as f32
        };
        raw_alternatives.push(indexed.clone());

        if class_idx == BLANK_CLASS {
            prev_class = BLANK_CLASS;
        } else if class_idx == SPACE_CLASS {
            text.push(' ');
            prev_class = SPACE_CLASS;
            char_cols.push(t_frac);
            alternatives.push(indexed);
            alt_rows.push(t);
        } else if class_idx == prev_class {
            // Collapse a repeated non-blank class.
        } else {
            let ch = decode_char(vocab, class_idx);
            if ch != '\u{FFFD}' {
                text.push(ch);
                char_cols.push(t_frac);
                alternatives.push(indexed);
                alt_rows.push(t);
                prev_class = class_idx;
            }
        }
    }

    CtcDecodeResult {
        text,
        alternatives,
        char_cols,
        seq_len_total: seq_len,
        raw_alternatives,
        alt_rows,
    }
}

fn packed_is_valid(packed: &[f32], expected_len: usize) -> bool {
    if packed.len() != expected_len {
        return false;
    }
    packed.iter().step_by(2).all(|&raw| {
        raw.is_finite() && raw.fract() == 0.0 && raw >= i32::MIN as f32 && raw < 2_147_483_648.0
        // i32::MAX is rounded up when cast to f32
    })
}

#[inline]
fn packed_score(packed: &[f32], row: usize, top_k: usize, class_idx: i32) -> Option<f32> {
    let base = row.checked_mul(top_k)?.checked_mul(2)?;
    if base.checked_add(top_k.checked_mul(2)?)? > packed.len() {
        return None;
    }
    for k in 0..top_k {
        let at = base + k * 2;
        if packed[at] as i32 == class_idx {
            return Some(packed[at + 1]);
        }
    }
    None
}

/// Greedy decode from the recognizer's packed candidate rows.
///
/// `packed` is exactly the native `inferTopK` layout: for each timestep,
/// `top_k` adjacent `(class-id-as-f32, logit)` pairs, descending by logit. The
/// packed input is read in place; only the returned character/alternative
/// vectors allocate. A malformed or non-integral class id fails closed to an
/// empty result rather than producing plausible garbage.
pub fn ctc_decode_topk(
    vocab: &[String],
    remap: &[i32],
    packed: &[f32],
    seq_len: usize,
    top_k: usize,
) -> CtcDecodeResult {
    let Some(row_floats) = seq_len.checked_mul(top_k) else {
        return CtcDecodeResult::empty(seq_len);
    };
    let Some(expected_len) = row_floats.checked_mul(2) else {
        return CtcDecodeResult::empty(seq_len);
    };
    if top_k == 0 || !packed_is_valid(packed, expected_len) {
        return CtcDecodeResult::empty(seq_len);
    }

    ctc_decode_candidates(
        vocab,
        seq_len,
        |t| {
            let raw = packed[t * top_k * 2] as i32;
            Some(remap_class(remap, raw))
        },
        |t| {
            let base = t * top_k * 2;
            Some(
                (0..top_k)
                    .map(|k| {
                        let at = base + k * 2;
                        let class = remap_class(remap, packed[at] as i32);
                        (decode_char(vocab, class), packed[at + 1])
                    })
                    .collect(),
            )
        },
        |t| {
            if t == 0 {
                None
            } else {
                let raw = packed[t * top_k * 2] as i32;
                packed_score(packed, t - 1, top_k, raw)
            }
        },
        |t| {
            let raw = packed[t * top_k * 2] as i32;
            packed_score(packed, t + 1, top_k, raw)
        },
    )
}

/// Full-logits greedy decode, used by the desktop fallback and ungated tests.
///
/// The only data that needs to remain large is the caller's logits. The decode
/// itself maps the selected top-15 entries into Rust-owned output vectors.
pub fn ctc_decode_full(
    vocab: &[String],
    remap: &[i32],
    logits: &[Vec<f32>],
    num_classes: usize,
    seq_len: usize,
) -> CtcDecodeResult {
    let winners: Vec<Option<usize>> = (0..seq_len)
        .map(|t| {
            let slice = logits.get(t)?;
            if slice.len() < num_classes {
                return None;
            }
            Some(
                slice
                    .iter()
                    .take(num_classes)
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
                        if v > best_v {
                            (i, v)
                        } else {
                            (best_i, best_v)
                        }
                    })
                    .0,
            )
        })
        .collect();

    ctc_decode_candidates(
        vocab,
        seq_len,
        |t| {
            winners
                .get(t)
                .copied()
                .flatten()
                .map(|idx| remap_class(remap, idx as i32))
        },
        |t| {
            let slice = logits.get(t)?;
            if slice.len() < num_classes {
                return None;
            }
            Some(top15_alternatives(vocab, remap, slice))
        },
        |t| {
            let winner = winners.get(t).copied().flatten()?;
            logits
                .get(t.checked_sub(1)?)
                .and_then(|row| row.get(winner))
                .copied()
        },
        |t| {
            let winner = winners.get(t).copied().flatten()?;
            logits
                .get(t.checked_add(1)?)
                .and_then(|row| row.get(winner))
                .copied()
        },
    )
}

/// Compact form of [`ctc_decode_full`] for a boundary that already has the
/// top-K table but not the full rows.
///
/// `left_scores[t]` and `right_scores[t]` carry the pruned winning class's score
/// in rows `t-1` and `t+1`; `NaN` means unavailable. Those two values per timestep
/// are the only information the full decoder reads outside its top-K table.
pub fn ctc_decode_full_packed(
    vocab: &[String],
    remap: &[i32],
    packed: &[f32],
    left_scores: &[f32],
    right_scores: &[f32],
    seq_len: usize,
    top_k: usize,
) -> CtcDecodeResult {
    let Some(row_floats) = seq_len.checked_mul(top_k) else {
        return CtcDecodeResult::empty(seq_len);
    };
    let Some(expected_len) = row_floats.checked_mul(2) else {
        return CtcDecodeResult::empty(seq_len);
    };
    if top_k == 0
        || left_scores.len() != seq_len
        || right_scores.len() != seq_len
        || !packed_is_valid(packed, expected_len)
    {
        return CtcDecodeResult::empty(seq_len);
    }
    ctc_decode_candidates(
        vocab,
        seq_len,
        |t| Some(remap_class(remap, packed[t * top_k * 2] as i32)),
        |t| {
            let base = t * top_k * 2;
            Some(
                (0..top_k)
                    .map(|k| {
                        let at = base + k * 2;
                        (
                            decode_char(vocab, remap_class(remap, packed[at] as i32)),
                            packed[at + 1],
                        )
                    })
                    .collect(),
            )
        },
        |t| finite_score(left_scores.get(t).copied()),
        |t| finite_score(right_scores.get(t).copied()),
    )
}

#[inline]
fn finite_score(value: Option<f32>) -> Option<f32> {
    value.filter(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    const TEST_VOCAB: [&str; 5] = ["あ", "い", "う", "え", "お"];

    fn vocab() -> Vec<String> {
        TEST_VOCAB.iter().map(|s| (*s).to_string()).collect()
    }

    fn packed_steps(steps: &[&[(i32, f32)]], top_k: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(steps.len() * top_k * 2);
        for step in steps {
            assert_eq!(step.len(), top_k);
            for &(class, score) in *step {
                out.push(class as f32);
                out.push(score);
            }
        }
        out
    }

    #[test]
    fn remap_and_decode_char_pin_mobile_fallbacks() {
        let remap = [0, 10, 20];
        assert_eq!(remap_class(&remap, 1), 10);
        assert_eq!(remap_class(&remap, 9), 9, "identity fallback");
        assert_eq!(decode_char(&vocab(), 0), '\u{3000}');
        assert_eq!(decode_char(&vocab(), 1), 'あ');
        assert_eq!(decode_char(&vocab(), 18708), '\u{3000}');
        assert_eq!(decode_char(&vocab(), 18709), ' ');
        assert_eq!(decode_char(&vocab(), 99999), '\u{3000}');
        assert_eq!(decode_char(&[], 1), '\u{FFFD}');
    }

    #[test]
    fn topk_order_matches_java_priority_queue() {
        assert_eq!(java_topk_order(&[0.5, 0.5, 0.5], 2), vec![2, 1]);
        assert_eq!(java_topk_order(&[0.1, 0.9, 0.5, 0.5], 3), vec![1, 3, 2]);
        assert_eq!(
            java_topk_order(
                &[
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 0.9, 0.9, 0.5, 0.5,
                    0.5, 0.2, 0.2, 0.1,
                ],
                15,
            ),
            vec![9, 10, 11, 8, 12, 13, 7, 6, 5, 14, 16, 4, 15, 3, 2],
        );
    }

    #[test]
    fn topk_decode_pins_text_columns_and_raw_shape() {
        let steps: &[&[(i32, f32)]] = &[
            &[(1, 0.9), (2, 0.5), (0, 0.1)],
            &[(1, 0.8), (2, 0.3), (0, 0.2)],
            &[(0, 0.95), (1, 0.4), (2, 0.2)],
            &[(1, 0.7), (3, 0.6), (0, 0.1)],
            &[(18709, 0.6), (1, 0.5), (0, 0.1)],
            &[(18709, 0.5), (2, 0.4), (0, 0.1)],
            &[(2, 0.9), (0, 0.3), (1, 0.1)],
        ];
        let v = vocab();
        let remap: Vec<i32> = (0..=TEST_VOCAB.len() as i32).collect();
        let res = ctc_decode_topk(&v, &remap, &packed_steps(steps, 3), steps.len(), 3);
        assert_eq!(res.text, "ああ  い");
        assert_eq!(res.char_cols, vec![0.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(res.alternatives.len(), 5);
        assert_eq!(res.raw_alternatives.len(), steps.len());
        assert!(res.raw_alternatives.iter().all(|row| row.len() == 3));
        assert_eq!(res.raw_alternatives[2][0], ('\u{3000}', 0.95));
    }

    /// The recorded `alt_rows` is exactly the identity a boundary relies on:
    /// `alternatives[j] == raw_alternatives[alt_rows[j]]`, ascending and in
    /// range, including for the collapsed repeats, the blanks and the emitted
    /// spaces. This is what lets the Android result carry the emitted list as
    /// one integer per character.
    #[test]
    fn alt_rows_index_back_into_the_raw_rows_exactly() {
        let steps: &[&[(i32, f32)]] = &[
            &[(1, 0.9), (2, 0.5), (0, 0.1)],
            &[(1, 0.8), (2, 0.3), (0, 0.2)], // repeat: collapsed, no row recorded
            &[(0, 0.95), (1, 0.4), (2, 0.2)], // blank: no row recorded
            &[(1, 0.7), (3, 0.6), (0, 0.1)],
            &[(18709, 0.6), (1, 0.5), (0, 0.1)], // space: recorded
            &[(18709, 0.5), (2, 0.4), (0, 0.1)], // space again: recorded
            &[(2, 0.9), (0, 0.3), (1, 0.1)],
        ];
        let v = vocab();
        let remap: Vec<i32> = (0..=TEST_VOCAB.len() as i32).collect();
        let res = ctc_decode_topk(&v, &remap, &packed_steps(steps, 3), steps.len(), 3);
        assert_eq!(res.alt_rows, vec![0, 3, 4, 5, 6]);
        assert_eq!(res.alt_rows.len(), res.alternatives.len());
        assert_eq!(res.alt_rows.len(), res.char_cols.len());
        for (j, &t) in res.alt_rows.iter().enumerate() {
            assert!(t < res.raw_alternatives.len());
            assert_eq!(res.alternatives[j], res.raw_alternatives[t]);
        }
        // Strictly ascending: an emitted row can never be recorded twice.
        assert!(res.alt_rows.windows(2).all(|w| w[0] < w[1]));
        // A failed-closed decode records nothing and stays self-consistent.
        let bad = ctc_decode_topk(&v, &remap, &[], 3, 3);
        assert!(bad.alt_rows.is_empty());
        assert_eq!(bad, ctc_decode_topk(&v, &remap, &[], 3, 3));
    }

    /// Folding the vertical punctuation into the decode is byte-identical to
    /// mapping the rows on the host afterwards, on every field the boundary
    /// carries: the text, both alternative lists, the columns and the length.
    /// The three substitutions are 1:1, so nothing shifts.
    #[test]
    fn vertical_punctuation_folds_into_the_rows_without_moving_anything() {
        // A vocabulary carrying all three source characters plus one that is
        // not substituted, so the no-op case is in the same fixture.
        let v: Vec<String> = ['?', '…', '‥', 'a', 'あ']
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        // class i+1 -> v[i]: 1='?', 2='…', 3='‥', 4='a', 5='あ'.
        let remap: Vec<i32> = (0..=5).collect();
        let steps: &[&[(i32, f32)]] = &[
            &[(1, 0.9), (4, 0.4), (0, 0.1)], // '?'
            &[(2, 0.8), (1, 0.3), (0, 0.2)], // '…'
            &[(3, 0.7), (2, 0.2), (0, 0.1)], // '‥'
            &[(4, 0.6), (3, 0.2), (0, 0.1)], // 'a' — not a substitution
            &[(5, 0.5), (4, 0.2), (0, 0.1)], // 'あ'
        ];
        let packed = packed_steps(steps, 3);
        let horizontal = ctc_decode_topk(&v, &remap, &packed, steps.len(), 3);
        assert_eq!(horizontal.text, "?…‥aあ");

        let mut vertical = horizontal.clone();
        vertical.apply_vertical_punctuation();
        assert_eq!(vertical.text, "？︙︰aあ");
        // Only the characters moved: columns, length and row bookkeeping.
        assert_eq!(vertical.char_cols, horizontal.char_cols);
        assert_eq!(vertical.seq_len_total, horizontal.seq_len_total);
        assert_eq!(vertical.alt_rows, horizontal.alt_rows);
        assert_eq!(vertical.alternatives.len(), horizontal.alternatives.len());
        assert_eq!(
            vertical.raw_alternatives.len(),
            horizontal.raw_alternatives.len()
        );
        // Every cell in both lists went through the same character map, and the
        // scores were left alone.
        let map = |r: &Vec<Vec<(char, f32)>>| -> Vec<Vec<char>> {
            r.iter().map(|row| row.iter().map(|(c, _)| *c).collect()).collect()
        };
        assert_eq!(
            map(&vertical.alternatives),
            vec![
                vec!['？', 'a', '\u{3000}'],
                vec!['︙', '？', '\u{3000}'],
                vec!['︰', '︙', '\u{3000}'],
                vec!['a', '︰', '\u{3000}'],
                vec!['あ', 'a', '\u{3000}'],
            ]
        );
        assert_eq!(
            map(&vertical.raw_alternatives),
            vec![
                vec!['？', 'a', '\u{3000}'],
                vec!['︙', '？', '\u{3000}'],
                vec!['︰', '︙', '\u{3000}'],
                vec!['a', '︰', '\u{3000}'],
                vec!['あ', 'a', '\u{3000}'],
            ]
        );
        for (row, orig) in vertical.raw_alternatives.iter().zip(&horizontal.raw_alternatives) {
            for ((c, s), (oc, os)) in row.iter().zip(orig) {
                assert_eq!(s, os, "a score must never move");
                assert_eq!(
                    *c,
                    crate::util::japanese::vertical_punctuation_char(*oc),
                    "each cell is its own source character mapped once"
                );
            }
        }

        // Idempotent, and identical to mapping the horizontal result by hand —
        // the exact operation the host used to run after the decode.
        let mut twice = vertical.clone();
        twice.apply_vertical_punctuation();
        assert_eq!(twice, vertical);
        let mut by_hand = horizontal.clone();
        by_hand.text = crate::util::japanese::vertical_punctuation(&by_hand.text);
        for row in by_hand.alternatives.iter_mut().chain(by_hand.raw_alternatives.iter_mut()) {
            for (c, _) in row.iter_mut() {
                *c = crate::util::japanese::vertical_punctuation_char(*c);
            }
        }
        assert_eq!(by_hand, vertical);
    }

    #[test]
    fn topk_decode_uses_only_neighbours_in_the_packed_slice() {
        let steps: &[&[(i32, f32)]] = &[
            &[(2, 1.0), (1, 1.0), (0, 0.0)],
            &[(1, 3.0), (2, 0.5), (0, 0.0)],
            &[(1, 0.0), (2, 0.5), (0, 0.0)],
        ];
        let v = vocab();
        let remap: Vec<i32> = (0..20).collect();
        let res = ctc_decode_topk(&v, &remap, &packed_steps(steps, 3), 3, 3);
        assert_eq!(res.text, "いあ");
        assert_eq!(res.char_cols, vec![0.0, 0.9]);
    }

    #[test]
    fn compact_full_decode_preserves_full_row_interpolation() {
        let v = vocab();
        let remap: Vec<i32> = (0..20).collect();
        let num_classes = 20;
        let mut t0 = vec![0.0f32; num_classes];
        t0[2] = 1.0;
        for c in 3..18 {
            t0[c] = 0.5;
        }
        t0[1] = 0.49;
        let mut t1 = vec![0.05f32; num_classes];
        t1[1] = 3.0;
        let mut t2 = vec![0.05f32; num_classes];
        t2[2] = 0.9;
        t2[0] = 0.1;
        t2[1] = 0.0;
        let logits = vec![t0, t1, t2];
        let full = ctc_decode_full(&v, &remap, &logits, num_classes, 3);

        let mut packed = Vec::new();
        for row in &logits {
            for idx in java_topk_order(row, TOP_K) {
                packed.push(idx as f32);
                packed.push(row[idx]);
            }
        }
        let left = [f32::NAN, 0.49, 3.0];
        let right = [3.0, 0.0, f32::NAN];
        let compact = ctc_decode_full_packed(&v, &remap, &packed, &left, &right, 3, TOP_K);
        assert_eq!(compact, full);
        assert_eq!(full.text, "いあい");
        assert!(full.raw_alternatives[0].iter().all(|(c, _)| *c != 'あ'));
        assert_eq!(full.char_cols[1], 1.0 + peak_offset(0.49, 3.0, 0.0));
    }

    #[test]
    fn malformed_packed_input_fails_closed() {
        let v = vocab();
        let remap = vec![0, 1, 2];
        assert_eq!(
            ctc_decode_topk(&v, &remap, &[], 1, 1),
            CtcDecodeResult::empty(1)
        );
        assert_eq!(
            ctc_decode_topk(&v, &remap, &[0.0, 1.0, 0.5, 1.0], 1, 1),
            CtcDecodeResult::empty(1),
        );
        assert_eq!(
            ctc_decode_topk(&v, &remap, &[f32::NAN, 1.0], 1, 1),
            CtcDecodeResult::empty(1),
        );
    }

    #[test]
    #[ignore = "release benchmark; run with --release -- --nocapture"]
    fn benchmark_realistic_topk_and_full_decode() {
        const CLASSES: usize = 18_710;
        let v: Vec<String> = (0..18_708)
            .map(|i| {
                char::from_u32(0x3000 + (i % 80) as u32)
                    .unwrap()
                    .to_string()
            })
            .collect();
        let remap: Vec<i32> = (0..CLASSES as i32).collect();

        for seq_len in [40usize, 60, 100] {
            let logits: Vec<Vec<f32>> = (0..seq_len)
                .map(|t| {
                    (0..CLASSES)
                        .map(|c| {
                            // Deterministic, finite, and varied without a random-number
                            // generator in either setup or measured regions.
                            ((t * 37 + c * 17) % 9973) as f32 / 100.0
                        })
                        .collect()
                })
                .collect();
            let mut packed = Vec::with_capacity(seq_len * TOP_K * 2);
            for row in &logits {
                for idx in java_topk_order(row, TOP_K) {
                    packed.push(idx as f32);
                    packed.push(row[idx]);
                }
            }

            for _ in 0..3 {
                std::hint::black_box(ctc_decode_topk(&v, &remap, &packed, seq_len, TOP_K));
            }
            let mut topk_us = Vec::with_capacity(100);
            for _ in 0..100 {
                let t0 = Instant::now();
                let out = ctc_decode_topk(&v, &remap, &packed, seq_len, TOP_K);
                topk_us.push(t0.elapsed().as_nanos() as f64 / 1_000.0);
                std::hint::black_box(out);
            }
            for _ in 0..1 {
                std::hint::black_box(ctc_decode_full(&v, &remap, &logits, CLASSES, seq_len));
            }
            let mut full_us = Vec::with_capacity(20);
            for _ in 0..20 {
                let t0 = Instant::now();
                let out = ctc_decode_full(&v, &remap, &logits, CLASSES, seq_len);
                full_us.push(t0.elapsed().as_nanos() as f64 / 1_000.0);
                std::hint::black_box(out);
            }
            topk_us.sort_by(f64::total_cmp);
            full_us.sort_by(f64::total_cmp);
            let p50 = |xs: &[f64]| xs[xs.len() / 2];
            let p95 = |xs: &[f64]| xs[(xs.len() * 95 / 100).min(xs.len() - 1)];
            println!(
                "CTC_BENCH seq={seq_len} classes={CLASSES} topk_p50_us={:.3} topk_p95_us={:.3} full_p50_us={:.3} full_p95_us={:.3} packed_input_bytes={} full_input_bytes={}",
                p50(&topk_us),
                p95(&topk_us),
                p50(&full_us),
                p95(&full_us),
                packed.len() * 4,
                logits.iter().map(Vec::len).sum::<usize>() * 4,
            );
        }
    }
}
