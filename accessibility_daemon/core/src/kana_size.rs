//! Kana small/large correction (#44), ported from the mobile app.
//!
//! Mobile reference: `util/KanaSizeEncoder.kt` (window encoding),
//! `util/KanaSizeFix.kt` (the ε policy), `KanaSizeNcnn.kt` +
//! `app/src/main/cpp/kana_size_ncnn.cpp` (the native model). The Kotlin tests
//! `KanaSizeEncoderTest` / `KanaSizeFixTest` are ported below, and the
//! numeric gate they describe (`KanaSizeNcnn.selfCheck`, ten published
//! logits) runs here against the PC's own ncnn.
//!
//! A 47,425-param byte-CNN decides whether a confusable kana position is the
//! small (っ) or the big (つ) form. Its input is the surrounding text only —
//! the target character is excluded from the 40-byte window, and the pair
//! identity travels through the base index. That is deliberate: if the target
//! were fed in, the model would mostly agree with whatever the OCR already
//! emitted, which is useless exactly where the OCR is wrong.
//!
//! The correction is unconditional (mobile removed the preference gate), and
//! ε = 0.01 is the only tunable. For each position whose character is a member
//! of a size pair, the model emits p(big):
//!
//! - flip small -> big when `p > 1 - ε`
//! - flip big -> small when `p < ε`
//! - leave the middle band alone
//!
//! Only `LineResult::text` is rewritten: the flip preserves the character
//! count, and mobile's `correctPage` likewise leaves `charBoxes` and the
//! recognizer's `alternatives` alone (it records the flip in the separate
//! `overrides` map, a mobile-only provenance concept this port does not have).

#[cfg(feature = "native")]
use std::path::Path;

#[cfg(feature = "native")]
use anyhow::{bail, Context, Result};

use crate::models::LineResult;

/// Certainty required to flip; the middle band is deliberately left untouched.
pub const EPSILON: f32 = 0.01;

// ---------------------------------------------------------------------------
// Native model (nb_all), through the same pinned ncnn fork as the OCR nets.
// Only compiled with the `native` feature; the window encoder and the ε
// policy below are pure Rust and always available.
// ---------------------------------------------------------------------------

#[cfg(feature = "native")]
#[allow(non_camel_case_types)]
mod ffi {
    use std::os::raw::{c_char, c_int};

    pub enum kana_size_t {}

    extern "C" {
        pub fn kana_size_create(
            param_path: *const c_char,
            bin_path: *const c_char,
        ) -> *mut kana_size_t;
        pub fn kana_size_destroy(model: *mut kana_size_t);
        pub fn kana_size_logits(
            model: *mut kana_size_t,
            win: *const c_int,
            bases: *const c_int,
            n: c_int,
            out_len: *mut c_int,
        ) -> *mut f32;
        pub fn kana_size_floats_free(data: *mut f32);
    }
}

/// The `nb_all` kana size model, loaded once and shared by the recognition
/// workers (ncnn `create_extractor()` gives every call its own extractor
/// state, so one loaded net can serve several threads — same as `RecNet`).
/// Requires the `native` feature.
#[cfg(feature = "native")]
pub struct KanaSizeNet {
    ptr: *mut ffi::kana_size_t,
}

// The C++ side owns an ncnn::Net, whose extractors are per-call; the handle is
// immutable after load.
#[cfg(feature = "native")]
unsafe impl Send for KanaSizeNet {}
#[cfg(feature = "native")]
unsafe impl Sync for KanaSizeNet {}

#[cfg(feature = "native")]
impl KanaSizeNet {
    /// Load `param`/`bin`. Fails rather than degrading: callers treat an error
    /// as "correction unavailable" (mobile's `KanaSizeNcnn.load` returns null).
    pub fn load(param: &Path, bin: &Path) -> Result<Self> {
        let param_c = std::ffi::CString::new(param.to_string_lossy().as_bytes())
            .with_context(|| format!("model path contains NUL: {}", param.display()))?;
        let bin_c = std::ffi::CString::new(bin.to_string_lossy().as_bytes())
            .with_context(|| format!("model path contains NUL: {}", bin.display()))?;
        let ptr = unsafe { ffi::kana_size_create(param_c.as_ptr(), bin_c.as_ptr()) };
        if ptr.is_null() {
            bail!(
                "failed to load kana size ncnn model {} / {}",
                param.display(),
                bin.display()
            );
        }
        Ok(KanaSizeNet { ptr })
    }

    /// Logits for `bases.len()` positions. `wins` is `n * 40` byte values
    /// (each 0..=255, from [`window`]) and `bases` is `n` pair indices.
    /// Fails closed: mobile's Kotlin wrapper rejects a base outside
    /// [`BASE_ORDER`] instead of letting the native clamp-to-0 score every
    /// such position as pair 0 — plausible logits, silently wrong.
    pub fn logits(&self, wins: &[i32], bases: &[i32]) -> Result<Vec<f32>> {
        let n = bases.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if wins.len() < n * WINDOW_BYTES {
            bail!(
                "kana logits: {} window bytes for {} positions",
                wins.len(),
                n
            );
        }
        if let Some(bad) = bases.iter().position(|&b| b < 0 || b as usize >= BASE_ORDER.len()) {
            bail!(
                "kana logits: base {} at position {} is outside BASE_ORDER ({})",
                bases[bad],
                bad,
                BASE_ORDER.len()
            );
        }
        let mut len: i32 = -1;
        let ptr = unsafe {
            ffi::kana_size_logits(
                self.ptr,
                wins.as_ptr(),
                bases.as_ptr(),
                n as i32,
                &mut len,
            )
        };
        if ptr.is_null() || len < 0 {
            bail!("kana size inference failed");
        }
        let out = unsafe { std::slice::from_raw_parts(ptr, len as usize).to_vec() };
        unsafe { ffi::kana_size_floats_free(ptr) };
        Ok(out)
    }
}

#[cfg(feature = "native")]
impl Drop for KanaSizeNet {
    fn drop(&mut self) {
        unsafe { ffi::kana_size_destroy(self.ptr) }
    }
}

/// Probability the pair is the *big* form, from the logit. The one sigmoid in
/// the project (#86/C1): mobile's policy calls `KanaSizeNcnn.probBig` rather
/// than carrying a private copy.
pub fn prob_big(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

// ---------------------------------------------------------------------------
// Window encoder
// ---------------------------------------------------------------------------

/// Pair index. The BIG form is the base, hiragana then katakana.
pub const BASE_ORDER: [char; 20] = [
    'あ', 'い', 'う', 'え', 'お', 'つ', 'や', 'ゆ', 'よ', 'わ', 'ア', 'イ', 'ウ', 'エ', 'オ', 'ツ',
    'ヤ', 'ユ', 'ヨ', 'ワ',
];

const RADIUS: usize = 5;
const CELL: usize = 4;
/// The model's input size: 2 × `RADIUS` UTF-8 cells of 4 bytes.
pub const WINDOW_BYTES: usize = 2 * RADIUS * CELL;

/// Characters that terminate context. The `nb_*` artifacts are trained on this
/// line-domain clip: the boundary character is not part of the window and
/// nothing beyond it is either.
pub const BOUNDARY: [char; 2] = ['。', '\n'];

/// The canonical big form for either member of a pair, or None if `ch` is not
/// part of one.
pub fn big_form_of(ch: char) -> Option<char> {
    match small_to_big(ch) {
        Some(big) => Some(big),
        None => base_position(ch).map(|_| ch),
    }
}

/// Pair index for the position holding `ch`, or None when `ch` is not part of
/// a size pair. Both っ and つ map to the same index: the pair is the class,
/// the size is the decision.
pub fn base_index_of(ch: char) -> Option<usize> {
    big_form_of(ch).and_then(base_position)
}

/// Whether `ch` is the small member of a pair, i.e. what the model corrects.
pub fn is_small(ch: char) -> bool {
    small_to_big(ch).is_some()
}

fn base_position(ch: char) -> Option<usize> {
    BASE_ORDER.iter().position(|&c| c == ch)
}

/// Small form -> big form. The app's own copy; the model's table is the
/// authority (mobile `KanaSizeEncoder.SMALL_TO_BIG`).
fn small_to_big(ch: char) -> Option<char> {
    Some(match ch {
        'ぁ' => 'あ',
        'ぃ' => 'い',
        'ぅ' => 'う',
        'ぇ' => 'え',
        'ぉ' => 'お',
        'ゎ' => 'わ',
        'っ' => 'つ',
        'ゃ' => 'や',
        'ゅ' => 'ゆ',
        'ょ' => 'よ',
        'ァ' => 'ア',
        'ィ' => 'イ',
        'ゥ' => 'ウ',
        'ェ' => 'エ',
        'ォ' => 'オ',
        'ヮ' => 'ワ',
        'ッ' => 'ツ',
        'ャ' => 'ヤ',
        'ュ' => 'ユ',
        'ョ' => 'ヨ',
        _ => return None,
    })
}

/// Big form -> small form, inverted from [`small_to_big`].
fn small_of(ch: char) -> Option<char> {
    const SMALL_FORMS: [char; 20] = [
        'ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ', 'ゎ', 'っ', 'ゃ', 'ゅ', 'ょ', 'ァ', 'ィ', 'ゥ', 'ェ', 'ォ',
        'ヮ', 'ッ', 'ャ', 'ュ', 'ョ',
    ];
    SMALL_FORMS
        .iter()
        .copied()
        .find(|&small| small_to_big(small) == Some(ch))
}

/// Left-align the character's UTF-8 bytes in a 4-byte cell, zero-padded.
fn write_cell(out: &mut [i32], ch: Option<char>) {
    let mut buf = [0u8; CELL];
    let n = match ch {
        Some(c) => {
            let s = c.encode_utf8(&mut buf);
            s.len()
        }
        None => 0,
    };
    for (i, cell) in out.iter_mut().enumerate() {
        *cell = if i < n { buf[i] as i32 } else { 0 };
    }
}

/// The 40-byte window for the position at `index` in `text`, with the
/// character at `index` excluded. Layout, exactly as the model's interface
/// document specifies:
///
/// ```text
///   cells 0..4  left context, leftmost first:  L5 L4 L3 L2 L1
///   cells 5..9  right context, nearest first:  R1 R2 R3 R4 R5
/// ```
///
/// Context stops at a [`BOUNDARY`] character and runs off the ends as zero
/// padding, which is what the app itself sees, since it recognises line by
/// line.
pub fn window(text: &str, index: usize) -> [i32; WINDOW_BYTES] {
    let chars: Vec<char> = text.chars().collect();
    window_chars(&chars, index)
}

/// [`window`] over an already-collected character slice (the policy scores
/// every candidate of a line, so it collects once).
fn window_chars(text: &[char], index: usize) -> [i32; WINDOW_BYTES] {
    // How far context actually reaches in each direction, stopping at a
    // boundary. The boundary character terminates the walk without being
    // counted, so it is never in the window and nothing beyond it is either.
    let mut left = 0usize;
    let mut j = index as isize - 1;
    while j >= 0 && left < RADIUS && !BOUNDARY.contains(&text[j as usize]) {
        left += 1;
        j -= 1;
    }
    let mut right = 0usize;
    let mut j = index + 1;
    while j < text.len() && right < RADIUS && !BOUNDARY.contains(&text[j]) {
        right += 1;
        j += 1;
    }

    let mut out = [0i32; WINDOW_BYTES];
    let mut w = 0usize;
    for off in (1..=RADIUS).rev() {
        // L5..L1, leftmost first
        let src = if off <= left { Some(text[index - off]) } else { None };
        write_cell(&mut out[w..w + CELL], src);
        w += CELL;
    }
    for off in 1..=RADIUS {
        // R1..R5, nearest first
        let src = if off <= right { Some(text[index + off]) } else { None };
        write_cell(&mut out[w..w + CELL], src);
        w += CELL;
    }
    out
}

// ---------------------------------------------------------------------------
// The ε policy
// ---------------------------------------------------------------------------

/// One position the model flipped, with the certainty that fired it.
#[derive(Debug, Clone, PartialEq)]
pub struct Flip {
    /// Line index in the corrected page.
    pub line: usize,
    /// Character index in the line's text.
    pub index: usize,
    pub from: char,
    pub to: char,
    pub p_big: f32,
}

/// One position left in the middle band, lowest confidence first.
#[derive(Debug, Clone, PartialEq)]
pub struct Declined {
    pub line: usize,
    pub index: usize,
    pub ch: char,
    /// Distance to the threshold that would have fired: `1 - p` for a small
    /// member, `p` for a big one.
    pub confidence: f32,
}

/// Result of one correction pass over a page.
#[derive(Debug, Clone)]
pub struct Correction {
    /// The page, with flips applied. Equal to the input when nothing fired.
    pub lines: Vec<LineResult>,
    pub flips: Vec<Flip>,
    /// Declined positions, lowest confidence first.
    pub declined: Vec<Declined>,
}

impl Correction {
    /// The positions the model declined to change, lowest confidence first, as
    /// `L<line>@<index> <char> p=<p>`. Deliberately no surrounding text: this
    /// is copied out of the app and shared, so it must not carry the book's
    /// own words (mobile `KanaSizeFix.lastDeclined`).
    pub fn declined_summary(&self) -> String {
        self.declined
            .iter()
            .take(5)
            .map(|d| format!("L{}@{} {} p={:.3}", d.line, d.index, d.ch, d.confidence))
            .collect::<Vec<_>>()
            .join(" / ")
    }
}

struct Candidate {
    line: usize,
    index: usize,
    ch: char,
    base: usize,
}

/// The policy, with scoring injected so it is testable without the native
/// layer. `score` takes `n * 40` window bytes and `n` pair indices and returns
/// `n` logits; a missing or wrong-length result leaves the page untouched,
/// exactly like mobile's `KanaSizeFix.apply`.
///
/// Candidates arrive in text order, and the pair set is wider than it looks —
/// `あ` belongs to the あ/ぁ pair, so a line of plain hiragana can still
/// present candidates.
pub fn correct_lines(
    lines: &[LineResult],
    mut score: impl FnMut(&[i32], &[i32]) -> Option<Vec<f32>>,
    epsilon: f32,
) -> Correction {
    let mut cands: Vec<Candidate> = Vec::new();
    for (li, line) in lines.iter().enumerate() {
        for (i, ch) in line.text.chars().enumerate() {
            if let Some(base) = base_index_of(ch) {
                cands.push(Candidate {
                    line: li,
                    index: i,
                    ch,
                    base,
                });
            }
        }
    }
    if cands.is_empty() {
        return Correction {
            lines: lines.to_vec(),
            flips: Vec::new(),
            declined: Vec::new(),
        };
    }

    let mut wins = vec![0i32; cands.len() * WINDOW_BYTES];
    let mut bases = vec![0i32; cands.len()];
    for (k, c) in cands.iter().enumerate() {
        let w = window(&lines[c.line].text, c.index);
        wins[k * WINDOW_BYTES..(k + 1) * WINDOW_BYTES].copy_from_slice(&w);
        bases[k] = c.base as i32;
    }

    let Some(logits) = score(&wins, &bases) else {
        return Correction {
            lines: lines.to_vec(),
            flips: Vec::new(),
            declined: Vec::new(),
        };
    };
    if logits.len() != cands.len() {
        return Correction {
            lines: lines.to_vec(),
            flips: Vec::new(),
            declined: Vec::new(),
        };
    }

    let mut flips: Vec<Flip> = Vec::new();
    let mut declined: Vec<Declined> = Vec::new();
    for (k, c) in cands.iter().enumerate() {
        let p = prob_big(logits[k]);
        let small = is_small(c.ch);
        let target = if small {
            big_form_of(c.ch)
        } else {
            small_of(c.ch)
        };
        let Some(target) = target else { continue };
        let flips_it = if small { p > 1.0 - epsilon } else { p < epsilon };
        if !flips_it {
            // The closest few by name: this tells a threshold problem apart
            // from the model simply agreeing with the recogniser.
            declined.push(Declined {
                line: c.line,
                index: c.index,
                ch: c.ch,
                confidence: if small { 1.0 - p } else { p },
            });
            continue;
        }
        flips.push(Flip {
            line: c.line,
            index: c.index,
            from: c.ch,
            to: target,
            p_big: p,
        });
    }
    declined.sort_by(|a, b| {
        a.confidence
            .partial_cmp(&b.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if flips.is_empty() {
        return Correction {
            lines: lines.to_vec(),
            flips,
            declined,
        };
    }

    let mut out = lines.to_vec();
    for f in &flips {
        let mut text: Vec<char> = out[f.line].text.chars().collect();
        if f.index < text.len() {
            text[f.index] = f.to;
            out[f.line].text = text.into_iter().collect();
        }
    }
    Correction {
        lines: out,
        flips,
        declined,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    #[cfg(feature = "native")]
    use std::path::PathBuf;

    fn line(text: &str) -> LineResult {
        LineResult {
            text: text.to_string(),
            char_boxes: Vec::new(),
            alternatives: Vec::new(),
            raw_alternatives: Vec::new(),
            sample_txt: None,
            is_vertical: false,
            chunk_boxes: Vec::new(),
        }
    }

    /// Logits in candidate order: one entry per size-pair position, in page
    /// text order. Non-pair characters are not scored.
    fn logits_for(texts: &[&str], f: impl Fn(char) -> f32) -> Vec<f32> {
        let mut out = Vec::new();
        for t in texts {
            for c in t.chars() {
                if base_index_of(c).is_some() {
                    out.push(f(c));
                }
            }
        }
        out
    }

    /// A scorer that ignores its input and returns precomputed logits.
    fn score_from(logits: Vec<f32>) -> impl FnMut(&[i32], &[i32]) -> Option<Vec<f32>> {
        move |_, _| Some(logits.clone())
    }

    /// A scorer that returns the same logit for every position.
    fn const_logits(logit: f32) -> impl FnMut(&[i32], &[i32]) -> Option<Vec<f32>> {
        move |_, bases| Some(vec![logit; bases.len()])
    }

    // か/き are not size pairs, so these lines present exactly one candidate:
    // the っ or つ.
    const SMALL_TEXT: &str = "かっき";
    const BIG_TEXT: &str = "かつき";

    // ——— Encoder (ported KanaSizeEncoderTest) ———

    /// The ten published validation vectors, verbatim from the artifact: the
    /// same bytes its own `validate_port.py` checks and the Kotlin
    /// `KanaSizeEncoderTest` pins. A wrong cell order, padding convention,
    /// missing context clip, or the target leaking into its own window would
    /// all produce plausible-looking windows that silently degrade every
    /// prediction. Four carry a boundary (。 or newline) within reach of the
    /// target, which is what pins the line-domain clip.
    struct Vector {
        text: &'static str,
        index: usize,
        /// Big-form outcome after the ε policy at the published logit.
        after: char,
        expected: f32,
        win: [i32; WINDOW_BYTES],
    }

    const VECTORS: [Vector; 10] = [
        Vector {
            text: "かれはいっとう。",
            index: 4,
            after: 'っ',
            expected: -0.863643,
            win: [
                0, 0, 0, 0, 227, 129, 139, 0, 227, 130, 140, 0, 227, 129, 175, 0, 227, 129, 132, 0,
                227, 129, 168, 0, 227, 129, 134, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        },
        Vector {
            text: "きょうはいいてんきですね、まつ。",
            index: 14,
            after: 'つ',
            expected: 0.265265,
            win: [
                227, 129, 167, 0, 227, 129, 153, 0, 227, 129, 173, 0, 227, 128, 129, 0, 227, 129,
                190, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        },
        Vector {
            text: "みんなでサッカーをするつもりです。",
            index: 5,
            after: 'ッ',
            expected: -4.868875,
            win: [
                227, 129, 191, 0, 227, 130, 147, 0, 227, 129, 170, 0, 227, 129, 167, 0, 227, 130,
                181, 0, 227, 130, 171, 0, 227, 131, 188, 0, 227, 130, 146, 0, 227, 129, 153, 0,
                227, 130, 139, 0,
            ],
        },
        Vector {
            text: "シーツをあらう。",
            index: 2,
            after: 'ツ',
            expected: 9.617793,
            win: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 227, 130, 183, 0, 227, 131, 188, 0, 227, 130,
                146, 0, 227, 129, 130, 0, 227, 130, 137, 0, 227, 129, 134, 0, 0, 0, 0, 0,
            ],
        },
        Vector {
            text: "きょうのてんきはいいですね。",
            index: 1,
            after: 'ょ',
            expected: -6.163191,
            win: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 227, 129, 141, 0, 227, 129, 134,
                0, 227, 129, 174, 0, 227, 129, 166, 0, 227, 130, 147, 0, 227, 129, 141, 0,
            ],
        },
        Vector {
            text: "キャンプにいく。",
            index: 1,
            after: 'ャ',
            expected: -4.094974,
            win: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 227, 130, 173, 0, 227, 131, 179,
                0, 227, 131, 151, 0, 227, 129, 171, 0, 227, 129, 132, 0, 227, 129, 143, 0,
            ],
        },
        Vector {
            text: "昌仙も、おもわず床几を立って、\n「あッ」\n　と、櫓",
            index: 12,
            after: 'っ',
            expected: -1.286511,
            win: [
                227, 129, 154, 0, 229, 186, 138, 0, 229, 135, 160, 0, 227, 130, 146, 0, 231, 171,
                139, 0, 227, 129, 166, 0, 227, 128, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        },
        Vector {
            text: "なろうかと……」\n　おえつは、片手に、腕白を抱きな",
            index: 12,
            after: 'つ',
            expected: 4.041704,
            win: [
                0, 0, 0, 0, 0, 0, 0, 0, 227, 128, 128, 0, 227, 129, 138, 0, 227, 129, 136, 0, 227,
                129, 175, 0, 227, 128, 129, 0, 231, 137, 135, 0, 230, 137, 139, 0, 227, 129, 171,
                0,
            ],
        },
        Vector {
            text: "は、変化多き世の中にもちょっと例の少ない並ならぬ三",
            index: 12,
            after: 'ょ',
            expected: -7.952802,
            win: [
                227, 129, 174, 0, 228, 184, 173, 0, 227, 129, 171, 0, 227, 130, 130, 0, 227, 129,
                161, 0, 227, 129, 163, 0, 227, 129, 168, 0, 228, 190, 139, 0, 227, 129, 174, 0,
                229, 176, 145, 0,
            ],
        },
        Vector {
            text: "。\n　そして、ザッザ、ザッザと、草の波を分けて、押",
            index: 12,
            after: 'ッ',
            expected: -1.359567,
            win: [
                227, 130, 182, 0, 227, 131, 131, 0, 227, 130, 182, 0, 227, 128, 129, 0, 227, 130,
                182, 0, 227, 130, 182, 0, 227, 129, 168, 0, 227, 128, 129, 0, 232, 141, 137, 0,
                227, 129, 174, 0,
            ],
        },
    ];

    #[test]
    fn every_published_vector_encodes_byte_for_byte() {
        for v in &VECTORS {
            assert_eq!(
                window(v.text, v.index),
                v.win,
                "window mismatch for {}@{}",
                v.text,
                v.index
            );
        }
    }

    #[test]
    fn base_index_follows_the_models_table() {
        assert_eq!(BASE_ORDER.len(), 20);
        assert_eq!(BASE_ORDER.iter().collect::<std::collections::HashSet<_>>().len(), 20);
        assert_eq!(base_index_of('あ'), Some(0));
        assert_eq!(base_index_of('つ'), Some(5));
        assert_eq!(base_index_of('よ'), Some(8));
        assert_eq!(base_index_of('ア'), Some(10));
        assert_eq!(base_index_of('ツ'), Some(15));
        assert_eq!(base_index_of('か'), None);
    }

    #[test]
    fn both_members_of_a_pair_share_one_index() {
        assert_eq!(base_index_of('つ'), base_index_of('っ'));
        assert_eq!(base_index_of('よ'), base_index_of('ょ'));
        assert_eq!(base_index_of('ツ'), base_index_of('ッ'));
        assert!(is_small('っ'));
        assert!(!is_small('つ'));
    }

    #[test]
    fn the_target_character_is_not_in_its_own_window() {
        // Same neighbours, different target: the windows must be identical, or
        // the model would be shown the OCR's answer at the one position it is
        // supposed to be deciding.
        assert_eq!(window("かれはっとう。", 3), window("かれはつとう。", 3));
    }

    #[test]
    fn context_stops_at_a_full_stop_or_a_newline() {
        // The boundary character terminates the walk and is not itself part of
        // the window, so neither the 。 nor anything before it is visible to a
        // position after it.
        let stop = window("あ。っあ", 2);
        assert_eq!(stop[..20], [0i32; 20], "left of the 。 must be empty");
        assert_eq!(
            stop[20..],
            [227, 129, 130, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "right context is the あ, then nothing"
        );

        // A newline clips the same way, and a character beyond it stays hidden
        // even though the window has room for it: without the clip this cell
        // would hold the あ.
        let nl = window("あ\nいっ", 3);
        assert_eq!(
            nl[..20],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 227, 129, 132, 0]
        );
        assert_eq!(nl[20..], [0i32; 20]);
    }

    #[test]
    fn context_runs_off_the_lines_ends_as_zeros() {
        let w = window("あい", 0);
        assert_eq!(w.len(), WINDOW_BYTES);
        assert_eq!(w[..20], [0i32; 20]);
        assert_eq!(w[20..24], [227, 129, 132, 0]);
        assert_eq!(w[24..], [0i32; 16]);
    }

    // ——— Policy (ported KanaSizeFixTest) ———

    #[test]
    fn flips_a_small_position_the_model_is_sure_is_big() {
        let calls = Cell::new(0usize);
        let bases = RefCell::new(Vec::new());
        let wins_len = Cell::new(0usize);
        let out = correct_lines(
            &[line(SMALL_TEXT)],
            |w, b| {
                calls.set(calls.get() + 1);
                bases.replace(b.to_vec());
                wins_len.set(w.len());
                Some(logits_for(&[SMALL_TEXT], |_| 10.0))
            },
            EPSILON,
        );
        assert_eq!(out.lines[0].text, "かつき");
        assert_eq!(out.flips.len(), 1);
        assert_eq!(out.flips[0].index, 1);
        assert_eq!(out.flips[0].from, 'っ');
        assert_eq!(out.flips[0].to, 'つ');
        assert_eq!(calls.get(), 1);
        assert_eq!(wins_len.get(), 40, "one window of 40 byte values");
        assert_eq!(bases.borrow()[0], 5, "っ travels as the つ pair index");
    }

    #[test]
    fn flips_a_big_position_the_model_is_sure_is_small() {
        let out = correct_lines(
            &[line(BIG_TEXT)],
            score_from(logits_for(&[BIG_TEXT], |_| -10.0)),
            EPSILON,
        );
        assert_eq!(out.lines[0].text, "かっき");
        assert_eq!(out.flips.len(), 1);
    }

    #[test]
    fn leaves_the_middle_band_alone() {
        // p(big) = 0.5 sits between ε and 1-ε, so neither direction fires.
        let out = correct_lines(
            &[line(SMALL_TEXT)],
            score_from(logits_for(&[SMALL_TEXT], |_| 0.0)),
            EPSILON,
        );
        assert_eq!(out.lines[0].text, SMALL_TEXT);
        assert!(out.flips.is_empty());
    }

    /// The posture the gate must have: a page too short to judge reads as
    /// modern, so the correction still applies.
    #[test]
    fn a_page_below_the_kana_floor_is_still_corrected() {
        let out = correct_lines(
            &[line(BIG_TEXT)],
            score_from(logits_for(&[BIG_TEXT], |_| -10.0)),
            EPSILON,
        );
        assert_eq!(out.lines[0].text, "かっき");
    }

    #[test]
    fn corrects_a_legacy_looking_page_because_era_handling_lives_in_the_artifact() {
        let mut text = String::new();
        text.push_str(&"あ".repeat(30));
        text.push('つ');
        text.push_str(&"あ".repeat(30));
        text.push('つ');
        text.push_str(&"あ".repeat(30));
        text.push('つ');
        text.push_str(&"あ".repeat(10));
        text.push('つ');
        text.push_str(&"あ".repeat(20));
        let out = correct_lines(
            &[line(&text)],
            score_from(logits_for(&[text.as_str()], |c| {
                if c == 'つ' {
                    -10.0
                } else {
                    0.0
                }
            })),
            EPSILON,
        );
        assert_eq!(
            out.lines[0].text,
            text.replace('つ', "っ"),
            "every large つ is flipped, since nothing withholds it"
        );
        assert_eq!(out.flips.len(), 4);
    }

    #[test]
    fn does_not_mutate_the_input_lines() {
        let input = line(SMALL_TEXT);
        let out = correct_lines(
            &[input.clone()],
            score_from(logits_for(&[SMALL_TEXT], |_| 10.0)),
            EPSILON,
        );
        assert_eq!(input.text, SMALL_TEXT, "the caller's line is untouched");
        assert_eq!(out.lines[0].text, "かつき");
    }

    #[test]
    fn positions_that_are_not_size_pairs_are_never_scored() {
        let calls = Cell::new(0usize);
        let text = "かきくけこさしすせそなにぬねの";
        let out = correct_lines(
            &[line(text)],
            |_, b| {
                calls.set(calls.get() + 1);
                Some(vec![0.0; b.len()])
            },
            EPSILON,
        );
        assert_eq!(calls.get(), 0, "no pair members in this line");
        assert_eq!(out.lines[0].text, text);
    }

    /// The threshold is tunable, and this pins that it actually acts: p(big)
    /// = 0.05 is far above the 0.01 default and comfortably inside a 0.10
    /// setting.
    #[test]
    fn a_looser_epsilon_flips_a_marginal_position() {
        let logit = (0.05f32 / 0.95).ln();
        let tight = correct_lines(
            &[line(BIG_TEXT)],
            score_from(logits_for(&[BIG_TEXT], |_| logit)),
            EPSILON,
        );
        assert_eq!(tight.lines[0].text, BIG_TEXT, "the 0.01 default declines this");
        let loose = correct_lines(
            &[line(BIG_TEXT)],
            score_from(logits_for(&[BIG_TEXT], |_| logit)),
            0.10,
        );
        assert_eq!(loose.lines[0].text, "かっき", "a 0.10 setting flips it");
    }

    /// Both directions of the strict inequality at the ε boundary: small→big
    /// fires only above 1-ε, big→small only below ε.
    #[test]
    fn epsilon_boundaries_are_strict() {
        let logit_for_p = |p: f32| (p / (1.0 - p)).ln();
        // small → big: p = 0.995 flips, p = 0.985 does not.
        let out = correct_lines(&[line(SMALL_TEXT)], const_logits(logit_for_p(0.995)), EPSILON);
        assert_eq!(out.lines[0].text, "かつき");
        let out = correct_lines(&[line(SMALL_TEXT)], const_logits(logit_for_p(0.985)), EPSILON);
        assert_eq!(out.lines[0].text, SMALL_TEXT);

        // big → small: p = 0.005 flips, p = 0.015 does not.
        let out = correct_lines(&[line(BIG_TEXT)], const_logits(logit_for_p(0.005)), EPSILON);
        assert_eq!(out.lines[0].text, "かっき");
        let out = correct_lines(&[line(BIG_TEXT)], const_logits(logit_for_p(0.015)), EPSILON);
        assert_eq!(out.lines[0].text, BIG_TEXT);
    }

    #[test]
    fn correcting_twice_is_idempotent() {
        let text = "きっゃと";
        let first = correct_lines(
            &[line(text)],
            score_from(logits_for(&[text], |_| 10.0)),
            EPSILON,
        );
        assert_eq!(first.lines[0].text, "きつやと");
        let second = correct_lines(
            &first.lines,
            score_from(logits_for(&[first.lines[0].text.as_str()], |_| 10.0)),
            EPSILON,
        );
        assert_eq!(second.lines[0].text, "きつやと");
        assert!(
            second.flips.is_empty(),
            "a corrected page must not flip again"
        );
    }

    #[test]
    fn declined_positions_are_reported_with_their_confidence() {
        let out = correct_lines(
            &[line(BIG_TEXT)],
            score_from(logits_for(&[BIG_TEXT], |_| 0.0)),
            EPSILON,
        );
        let d = out.declined_summary();
        assert!(d.contains("L0@1"), "should name line and index: {d}");
        assert!(d.contains("p=0.500"), "should give a confidence: {d}");
        assert!(!d.contains('か'), "must not carry book text: {d}");
    }

    #[test]
    fn a_failed_scorer_leaves_the_page_untouched() {
        let out = correct_lines(&[line(SMALL_TEXT)], |_, _| None, EPSILON);
        assert_eq!(out.lines[0].text, SMALL_TEXT);
        assert!(out.flips.is_empty());
    }

    #[test]
    fn a_short_result_array_is_rejected_rather_than_half_applied() {
        let out = correct_lines(
            &[line(SMALL_TEXT)],
            |_, bases| Some(vec![10.0; bases.len() + 1]),
            EPSILON,
        );
        assert_eq!(out.lines[0].text, SMALL_TEXT);
        assert!(out.flips.is_empty());
    }

    #[test]
    fn a_small_flip_and_a_big_flip_coexist_in_one_page() {
        let text = "きっゃと";
        let out = correct_lines(
            &[line(text)],
            score_from(logits_for(&[text], |_| 10.0)),
            EPSILON,
        );
        assert_eq!(out.lines[0].text, "きつやと", "both are flipped towards big");
    }

    /// Candidates are scored page-wide in text order; each line is rewritten
    /// on its own.
    #[test]
    fn flips_are_applied_per_line() {
        let out = correct_lines(
            &[line(SMALL_TEXT), line(BIG_TEXT)],
            score_from(logits_for(&[SMALL_TEXT, BIG_TEXT], |c| {
                if c == 'っ' {
                    10.0
                } else {
                    -10.0
                }
            })),
            EPSILON,
        );
        assert_eq!(out.lines[0].text, "かつき");
        assert_eq!(out.lines[1].text, "かっき");
        assert_eq!(out.flips.len(), 2);
        assert_eq!(out.flips[0].line, 0);
        assert_eq!(out.flips[1].line, 1);
    }

    // ——— The real model (the PC's own ncnn path, `native` feature only) ———

    #[cfg(feature = "native")]
    struct Fixture {
        net: KanaSizeNet,
    }

    #[cfg(feature = "native")]
    fn fixture() -> Fixture {
        let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/..")).join("assets/kana_size");
        Fixture {
            net: KanaSizeNet::load(&dir.join("nb_all.param"), &dir.join("nb_all.bin"))
                .expect("kana size model load"),
        }
    }

    /// The on-device self-check, on the PC: the model author's ten published
    /// vectors through this repo's own encoder and native wrapper. It proves
    /// the whole path (asset bytes, param/bin load, FFI marshalling, float
    /// behaviour) with ten positions, and it is the numeric gate the mobile
    /// app runs through JNI.
    #[cfg(feature = "native")]
    #[test]
    fn kana_model_self_check() {
        let f = fixture();
        let n = VECTORS.len();
        let mut wins = vec![0i32; n * WINDOW_BYTES];
        let mut bases = vec![0i32; n];
        for (i, v) in VECTORS.iter().enumerate() {
            wins[i * WINDOW_BYTES..(i + 1) * WINDOW_BYTES].copy_from_slice(&window(v.text, v.index));
            let ch = v.text.chars().nth(v.index).expect("target in text");
            bases[i] = base_index_of(ch).expect("target is a pair member") as i32;
        }
        let out = f.net.logits(&wins, &bases).expect("kana logits");
        assert_eq!(out.len(), n);
        let mut worst = 0f32;
        let mut worst_idx = 0usize;
        for i in 0..n {
            let d = (out[i] - VECTORS[i].expected).abs();
            if d > worst {
                worst = d;
                worst_idx = i;
            }
        }
        assert!(
            worst <= 1e-4,
            "kana model mismatch: case {} off by {:.3e} (got {}, want {})",
            worst_idx,
            worst,
            out[worst_idx],
            VECTORS[worst_idx].expected
        );
    }

    /// End to end through the real model: each published target either flips
    /// or stays exactly as its published logit says it must.
    #[cfg(feature = "native")]
    #[test]
    fn kana_model_end_to_end_keeps_or_flips_the_published_targets() {
        let f = fixture();
        for v in &VECTORS {
            let corrected = correct_lines(
                &[line(v.text)],
                |w, b| f.net.logits(w, b).ok(),
                EPSILON,
            );
            let got: Vec<char> = corrected.lines[0].text.chars().collect();
            let flipped = corrected.flips.iter().any(|fl| fl.index == v.index);
            assert_eq!(
                got[v.index], v.after,
                "target {}@{} of {:?} (published logit {})",
                v.text.chars().nth(v.index).unwrap(),
                v.index,
                v.text,
                v.expected
            );
            assert_eq!(
                flipped,
                v.after != v.text.chars().nth(v.index).unwrap(),
                "flip report disagrees for {}@{}",
                v.text,
                v.index
            );
        }
    }
}

