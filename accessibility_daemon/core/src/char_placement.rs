//! CTC-Anchored Placement (CAP) — the improved per-character box algorithm.
//!
//! Port of the normative Python reference
//! `tools/char_placement/place.py::proposed_char_boxes` (float64, shipped
//! `ProposedOptions` defaults) from the Android research branch, which the
//! Kotlin mirror (`CharPlacement.kt`, float32, ≤1.5 px) is pinned against.
//! The conformance spec for this stage is `docs/char-placement-conformance.md`
//! in the Android checkout (the handoff doc — one home for the prose).
//!
//! One pure function turns a decoded line + per-timestep CTC top-K + crop
//! luminance into one box per character, each spanning the whole cross axis
//! (`0..crop_h` horizontal, `0..crop_w` vertical), adjacent boxes sharing
//! exact boundary floats after the final pass.
//!
//! Pipeline: greedy CTC runs over the top-K → advance-class layout template
//! (Huber-reweighted fit) → per-character anchor (template when it agrees
//! with the run column, the run column otherwise) → robust ink refinement
//! (mid-quartile inside a Voronoi window, with the punctuation/bimodal
//! retries) → measured ink extents → translate-apart for infeasible pairs →
//! a shared boundary per pair, placed in the empty ink space between the
//! glyphs (spec §7: this is deliberately *not* the old midpoint cap).
//!
//! # Porting rules (each paid for by a real failure — spec §1)
//!
//! * **Profile walks clamp to the profile length.** Python's `prof[a:b]`
//!   slice caps at the array end; an indexing walk with the raw `b` overruns
//!   exactly when `hi == L`, which is the common case: the last glyph's
//!   Voronoi window clips to the line end on a tight det crop. On Android
//!   that threw inside an emit callback that swallowed it, and half a page
//!   of detected lines rendered blank. Every walk here clamps;
//!   `last_glyph_window_reaching_the_line_end_does_not_walk_past_the_profile`
//!   pins it (the `CharPlacementEdgeCaseTest` regression).
//! * **No silent catches on this path.** Placement runs inside the
//!   recognition emit path, so an input that cannot be handled falls back to
//!   the `char_cols` path instead of unwinding: `text` shorter than the run
//!   count, a missing/short pixel array, `crop < 8` (skips the ink pass),
//!   `n == 0` / `seq_len_total <= 0` (empty result) — never panic (spec
//!   §1.3.3).
//! * **Deterministic.** No RNG anywhere: same inputs, same boxes (the only
//!   RNG in the spec is the metric's jitter sampling, which lives in the
//!   harness, not here).
//!
//! Not ported, deliberately: the off-default end-anchored sweep
//! (`sweep_place`, a measured *look* option rejected as the default, spec
//! §7) and its pull-back — Kotlin did not port it either while it is not the
//! default. [`Options`] carries the shipped knobs that do load work.

use crate::util::japanese::is_half_width;

/// The decoder's blank (and the reference's `BLANK`), as a character.
pub const BLANK: char = '\u{3000}';

// ─────────────────────────────────────────────────────────────────────────────
// Japanese font-metric classes (jpfmt.py: advance + optical class)
// ─────────────────────────────────────────────────────────────────────────────

/// Characters whose ink is NOT centred in their advance box along the reading
/// axis. Only `Center` characters may anchor the layout template.
const PUNCT_CLOSING: &str = "、。，．,.)）〕》」』】〙〗〟’”］";
const PUNCT_OPENING: &str = "(（〔《「『【〘〖〝‘“［";
const PUNCT_MISC: &str = "：；！？…‥︙︰・ー～「」『』（）[]{}〈〉《》【】";
const SMALL_KANA: &str = "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ";
const OPTICAL_MARKS: &str = "゛゜ゝゞヽヾ々〆〇";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Optical {
    Center,
    Small,
    Punct,
}

fn optical_class(ch: char) -> Optical {
    if SMALL_KANA.contains(ch) {
        return Optical::Small;
    }
    if PUNCT_CLOSING.contains(ch)
        || PUNCT_OPENING.contains(ch)
        || PUNCT_MISC.contains(ch)
        || OPTICAL_MARKS.contains(ch)
        || ch.is_whitespace()
    {
        return Optical::Punct;
    }
    Optical::Center
}

/// `advance_units`: 0.5 em for ASCII + halfwidth katakana, 1.0 otherwise.
/// (Identical rule to `japanese::is_half_width`, the mobile `isHalfWidth`.)
fn advance_units(ch: char) -> f64 {
    if is_half_width(ch) {
        0.5
    } else {
        1.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Options (ProposedOptions, the normative dataclass — shipped defaults)
// ─────────────────────────────────────────────────────────────────────────────

/// The reference's `ProposedOptions` with its shipped defaults. The
/// load-bearing knobs (spec §1): `translate_max_em`, `final_pass`,
/// `translate_overlap`, `bimodal_retry`, `bimodal_valley_em`,
/// `bimodal_min_frac`. `sweep_place` and its family are absent on purpose
/// (see the module docs).
#[derive(Clone, Debug)]
pub struct Options {
    // Fusion
    pub ink_max_pull_em: f64,
    pub window_em: f64,
    pub window_stride: f64,
    // Anchor selection: template when it agrees with the CTC column within
    // these bounds, the CTC column otherwise (1.2 strides is just past the
    // ±0.5-stride quantization bound).
    pub anchor_tol_em: f64,
    pub anchor_tol_stride: f64,
    // Punctuation / small kana: a smeared window retries around the CTC run
    // centre, taken only when it measures a compact blob near the anchor.
    pub punct_spread_fallback: bool,
    // Neighbour-blob retry for center-class windows split by a real valley.
    pub bimodal_retry: bool,
    pub bimodal_valley_em: f64,
    pub bimodal_min_frac: f64,
    pub punct_fallback_window_em: f64,
    pub punct_fallback_max_em: f64,
    // Template fit
    pub huber_em: f64,
    pub ridge_ls: f64,
    // Profile
    pub profile_band: f64,
    pub profile_smooth: i32,
    // Reliability gates
    pub min_mass_frac: f64,
    pub max_spread_em: f64,
    pub conf_floor: f64,
    // Second pass
    pub refine_passes: i32,
    // Overlap resolution: translate the pair into its neighbours' positive
    // slack before splitting, each glyph's move capped at `translate_max_em`.
    pub translate_overlap: bool,
    pub translate_max_em: f64,
    pub translate_passes: i32,
    pub translate_gate_cut: bool,
    // Final pass: shared boundaries in the empty ink space, not midpoints.
    pub final_pass: bool,
    pub extent_pad_px: f64,
    pub extent_floor: f64,
    pub extent_window_em: f64,
    pub extent_grow_frac: f64,
    pub split_floor_frac: f64,
    pub min_half_px: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            ink_max_pull_em: 0.45,
            window_em: 0.6,
            window_stride: 0.4,
            anchor_tol_em: 0.4,
            anchor_tol_stride: 1.2,
            punct_spread_fallback: true,
            bimodal_retry: true,
            bimodal_valley_em: 0.20,
            bimodal_min_frac: 0.12,
            punct_fallback_window_em: 0.5,
            punct_fallback_max_em: 0.6,
            huber_em: 0.35,
            ridge_ls: 1e-3,
            profile_band: 0.2,
            profile_smooth: 2,
            min_mass_frac: 0.12,
            max_spread_em: 0.80,
            conf_floor: 1.0,
            refine_passes: 1,
            translate_overlap: true,
            translate_max_em: 0.04,
            translate_passes: 2,
            translate_gate_cut: false,
            final_pass: true,
            extent_pad_px: 1.0,
            extent_floor: 0.5,
            extent_window_em: 0.15,
            extent_grow_frac: 0.3,
            split_floor_frac: 1.0,
            min_half_px: 2.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers (mirror the reference exactly, including its rounding rules)
// ─────────────────────────────────────────────────────────────────────────────

/// Python's `_clamp`: `lo if v < lo else hi if v > hi else v` (never panics,
/// even for a degenerate `lo > hi`).
fn clampv(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// Python's `int(np.clip(round(v), lo, hi))`: clamp a rounded value.
fn clamp_round(v: f64, lo: usize, hi: usize) -> usize {
    let r = round_half_even(v);
    if r < lo as f64 {
        lo
    } else if r > hi as f64 {
        hi
    } else {
        r as usize
    }
}

/// Python's `round()` on a float: banker's rounding (half to even). Rust's
/// `f64::round` rounds halves away from zero, which differs exactly on `x.5`
/// centres — the reference is normative, so mirror it.
fn round_half_even(v: f64) -> f64 {
    let floor = v.floor();
    let diff = v - floor;
    if diff > 0.5 {
        floor + 1.0
    } else if diff < 0.5 || (floor as i64) % 2 == 0 {
        // Under the tie: always down; on the tie: to the even neighbour.
        floor
    } else {
        floor + 1.0
    }
}

/// `np.searchsorted(a, v)`: leftmost index `i` with `a[i] >= v` (ascending).
fn searchsorted_left(a: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = a.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if a[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `np.median` for an even count (mean of the two middle values), 0.0 when
/// empty (the reference's `_median` guard).
fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// `_weighted_median`: sort by value, cumulate weights, take the first value
/// whose cumsum reaches half the total (0-weight totals fall back to the
/// plain median).
fn weighted_median(xs: &[f64], ws: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap_or(std::cmp::Ordering::Equal));
    let xs_s: Vec<f64> = order.iter().map(|&i| xs[i]).collect();
    let ws_s: Vec<f64> = order.iter().map(|&i| ws[i]).collect();
    let mut cw = 0.0f64;
    let mut cums = Vec::with_capacity(xs_s.len());
    for &w in &ws_s {
        cw += w;
        cums.push(cw);
    }
    if cw <= 0.0 {
        return median(xs);
    }
    let i = searchsorted_left(&cums, cw / 2.0);
    xs_s[i.min(xs_s.len() - 1)]
}

// ─────────────────────────────────────────────────────────────────────────────
// Luminance / ink profile (shared by the whole stage)
// ─────────────────────────────────────────────────────────────────────────────

/// The reference's channel mean in float32 for one run of RGB8 samples; a
/// trailing partial sample is dropped. Published so a boundary reading a
/// *border sample* uses the crate's own arithmetic (it is the per-pixel form
/// of what [`luminance_from_rgb`] does to a whole crop).
pub fn luminance_from_rgb_samples(rgb: &[u8]) -> Vec<f32> {
    rgb.chunks_exact(3)
        .map(|px| (px[0] as f32 + px[1] as f32 + px[2] as f32) / 3.0)
        .collect()
}

/// RGB8 → luminance, the reference's `luminance_from_argb` uint8 path
/// (channel mean in float32). Row-major, `w * h` samples.
pub fn luminance_from_rgb(pixels: &[u8], w: u32, h: u32) -> Vec<f32> {
    let n = (w as usize).saturating_mul(h as usize);
    let mut out = vec![0.0f32; n];
    // zip stops at the shorter side: n samples, or whatever the buffer holds.
    for (o, v) in out.iter_mut().zip(luminance_from_rgb_samples(pixels)) {
        *o = v;
    }
    out
}

/// The border sample, at stride 7: the top/bottom row pairs first, then the
/// left/right column pairs. The order is the median's only input, so it cannot
/// change the result — but a boundary that measures the evidence samples the
/// same pixels in the same order.
pub fn ink_border_lum(lum: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut border: Vec<f32> = Vec::new();
    if h > 0 && w > 0 {
        let mut j = 0usize;
        while j < w {
            border.push(lum[j]);
            border.push(lum[(h - 1) * w + j]);
            j += 7;
        }
        let mut i = 0usize;
        while i < h {
            border.push(lum[i * w]);
            border.push(lum[i * w + w - 1]);
            i += 7;
        }
    }
    border
}

/// Background polarity from a border sample: the *true* median (the mean of
/// the two middle values on an even count) against 128.0 — CAP's own rule,
/// which differs from `char_boxes::snap_polarity`'s upper median; the two
/// stages are independent, so they keep their own. Published so a boundary
/// measuring the evidence does not restate the median.
pub fn ink_polarity(border_lum: &[f32]) -> bool {
    let mut sorted = border_lum.to_vec();
    sorted.sort_by(f32::total_cmp);
    median(&sorted.iter().map(|&v| v as f64).collect::<Vec<_>>()) > 128.0
}

/// Ink mask + background polarity: the border sample decides light-on-dark
/// vs dark-on-light, then a fixed threshold pair builds the mask.
/// Returns `(mask, background_is_light)`.
fn ink_mask(lum: &[f32], w: usize, h: usize) -> (Vec<bool>, bool) {
    let border = ink_border_lum(lum, w, h);
    let bg_light = ink_polarity(&border);
    let mask: Vec<bool> = lum
        .iter()
        .map(|&v| if bg_light { v < 110.0 } else { v > 145.0 })
        .collect();
    (mask, bg_light)
}

/// The ink mask CAP's ink pass reduces, in whichever form the caller has it.
///
/// Both consumers of the mask ([`dominant_cross_band`] and [`ink_profile`])
/// only ever *read* it, and both read the same pixels the luminance path would
/// have thresholded, so the two forms are interchangeable by construction: the
/// owned one is what [`ink_mask`] builds from a crop, the packed one is the
/// 1-bit-per-pixel form a boundary can ship (8x smaller than the ARGB crop and
/// cheaper to marshal than a single boxed float list of the same length).
#[derive(Clone, Debug)]
pub enum InkMask {
    /// One `bool` per pixel, row-major.
    Owned { mask: Vec<bool>, w: usize, h: usize },
    /// One bit per pixel over the **global** row-major pixel index
    /// `i = y * w + x`, **MSB-first within each byte**: pixel `i` is bit
    /// `0x80 >> (i % 8)` of byte `i / 8`. A byte can straddle a row boundary
    /// (when `w` is not a multiple of 8), so the bit comes from `i % 8`, never
    /// from `x % 8` — the rows are contiguous, not byte-aligned.
    Packed { bits: Vec<u8>, w: usize, h: usize },
}

impl InkMask {
    /// Bytes a packed mask of `w × h` needs: one per 8 pixels, last byte
    /// partially used. A short buffer is refused rather than read past.
    pub fn packed_len(w: usize, h: usize) -> usize {
        w.saturating_mul(h).div_ceil(8)
    }

    fn get(&self, x: usize, y: usize) -> bool {
        match self {
            InkMask::Owned { mask, w, .. } => mask[y * w + x],
            InkMask::Packed { bits, w, .. } => {
                let i = y * w + x;
                bits[i / 8] & (0x80u8 >> (i % 8)) != 0
            }
        }
    }
}

/// CAP's ink evidence as a boundary ships it: the background polarity plus the
/// ink mask at 1 bit per pixel.
///
/// This is the *whole* image input of the ink pass — `dominant_cross_band` and
/// `ink_profile` read nothing else — so a packed mask is not an approximation
/// of the crop, it is the crop's ink reduced losslessly. The minimum further
/// reduction is [`InkProfile`] (polarity + the band-restricted profile), which
/// costs one extra crossing because the band is derived from the crop.
#[derive(Clone, Debug, PartialEq)]
pub struct InkEvidence {
    /// `true` when the border sample's median says the background is light.
    pub bg_light: bool,
    /// The mask, [`InkMask::Packed`]'s layout: one bit per pixel over the
    /// global row-major index, MSB-first in each byte.
    pub bits: Vec<u8>,
    pub w: usize,
    pub h: usize,
}

impl InkEvidence {
    /// An evidence record, or `None` when `bits` is too short for `w × h`
    /// (the spec §1.3.3 "short array skips the ink pass" rule, on the packed
    /// form) or the frame is degenerate.
    pub fn from_packed(bg_light: bool, bits: &[u8], w: u32, h: u32) -> Option<Self> {
        let (w, h) = (w as usize, h as usize);
        if w == 0 || h == 0 || bits.len() < InkMask::packed_len(w, h) {
            return None;
        }
        Some(InkEvidence {
            bg_light,
            bits: bits.to_vec(),
            w,
            h,
        })
    }

    fn mask(&self) -> InkMask {
        InkMask::Packed {
            bits: self.bits.clone(),
            w: self.w,
            h: self.h,
        }
    }
}

/// CAP's ink evidence reduced to its minimum: the background polarity and the
/// **band-restricted, un-smoothed** reading-axis ink profile. The blur, the
/// mid-quartile walks, the extents and the boundary pass are all unchanged, so
/// a correct profile gives identical boxes to the crop path.
///
/// The band is *not* part of the record because the crop path does not use it
/// after [`ink_profile`] either — it only decides which rows the counts came
/// from. Callers that cannot afford the second crossing want
/// [`InkEvidence`] instead.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InkProfile {
    /// `true` when the border sample's median says the background is light.
    pub bg_light: bool,
    /// Ink count per reading-axis position, inside the band, un-smoothed.
    /// Length: `crop_w` horizontal, `crop_h` vertical.
    pub band: Vec<f32>,
}

/// The constants the evidence measurement is defined against, so a caller that
/// reduces the crop itself (the mobile FFI boundary) reads them from here
/// instead of hardcoding them: the two ink thresholds, the polarity threshold,
/// the border stride and the band's minimum cross fraction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InkSpec {
    pub ink_below: f32,
    pub ink_above: f32,
    pub bg_median_above: f32,
    pub border_stride: usize,
    /// The band's minimum cross fraction (`dominant_cross_band`'s `min_frac`).
    pub band_min_frac: f64,
    /// The profile box-blur radius.
    pub profile_smooth: i32,
    /// The largest count a count series can carry.
    pub max_count: usize,
    /// The exclusive upper bound of the ink channel sums on a light
    /// background: ink is `channel sum < ink_below_sum`, so a caller can count
    /// with an integer compare instead of a float division per pixel.
    pub ink_below_sum: i32,
    /// The inclusive lower bound of the ink channel sums on a dark background.
    pub ink_above_sum: i32,
}

/// The largest count a count series can carry. A cross axis thicker than this
/// cannot be reduced to counts and must ship the mask instead.
pub const INK_MAX_COUNT: usize = 255;

/// The shipped [`InkSpec`].
pub fn ink_spec() -> InkSpec {
    InkSpec {
        ink_below: 110.0,
        ink_above: 145.0,
        bg_median_above: 128.0,
        border_stride: 7,
        band_min_frac: 0.35,
        profile_smooth: 2,
        max_count: INK_MAX_COUNT,
        ink_below_sum: (110.0f32 * 3.0) as i32,
        ink_above_sum: (145.0f32 * 3.0) as i32 + 1,
    }
}

/// The integer ink test this crate's own float test is equivalent to, as one
/// predicate, so a caller can share it verbatim. Pinned over every possible
/// channel sum by `ink_spec_integer_agrees_with_float`.
pub fn is_ink_sum(sum: i32, bg_light: bool) -> bool {
    if bg_light {
        sum < ink_spec().ink_below_sum
    } else {
        sum >= ink_spec().ink_above_sum
    }
}

/// The band [`dominant_cross_band`] picks from these cross-axis ink counts:
/// row indices (horizontal) or column indices (vertical), `[lo, hi)` in the
/// same half-open convention [`ink_profile`] slices with. Published so a caller
/// that measures the profile itself measures it in the band the algorithm will
/// use — one crossing to ask, one to place.
pub fn cross_band(cross: &[f64], min_frac: f64) -> (usize, usize) {
    let n = cross.len();
    if n == 0 {
        return (0, 0);
    }
    let (lo_f, hi_f) = dominant_cross_band_from_counts(cross, min_frac);
    // The same truncation (and the same non-empty floor) `ink_profile` applies.
    let lo = (n as f64 * lo_f) as usize;
    let hi = (n as f64 * hi_f) as usize;
    (lo, hi.max(lo + 1))
}

/// Fractions `[lo, hi]` of the cross axis that hold the body text: the
/// dominant ink run (the one containing the ink-mass median), widened until
/// it covers `min_frac` of the cross so a sparse row is not mistaken for a
/// thin strip. Dodges ruby and neighbouring-line ink before it can
/// contaminate the reading-axis profile.
fn dominant_cross_band(
    mask: &InkMask,
    w: usize,
    h: usize,
    vertical: bool,
    min_frac: f64,
) -> (f64, f64) {
    let (cross, _n) = if vertical {
        // per column
        let mut c = vec![0.0f64; w];
        for x in 0..w {
            let mut m = 0.0;
            for y in 0..h {
                if mask.get(x, y) {
                    m += 1.0;
                }
            }
            c[x] = m;
        }
        (c, w)
    } else {
        // per row
        let mut c = vec![0.0f64; h];
        for y in 0..h {
            let mut m = 0.0;
            for x in 0..w {
                if mask.get(x, y) {
                    m += 1.0;
                }
            }
            c[y] = m;
        }
        (c, h)
    };
    dominant_cross_band_from_counts(&cross, min_frac)
}

/// [`dominant_cross_band`] on ink counts the caller measured, split out so the
/// band is a pure function of the cross-axis reduction. That is what lets a
/// boundary ask for the band in one crossing ([`cross_band`]) and measure the
/// profile in the band the answer names.
fn dominant_cross_band_from_counts(cross: &[f64], min_frac: f64) -> (f64, f64) {
    let n = cross.len();
    if n == 0 {
        return (0.0, 1.0);
    }
    let max = cross.iter().cloned().fold(0.0f64, f64::max);
    if max <= 0.0 {
        return (0.0, 1.0);
    }
    let thr = (0.15 * max).max(1.0);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &v) in cross.iter().enumerate() {
        if v >= thr && start.is_none() {
            start = Some(i);
        } else if v < thr && start.is_some() {
            runs.push((start.unwrap(), i - 1));
            start = None;
        }
    }
    if let Some(s) = start {
        runs.push((s, n - 1));
    }
    if runs.is_empty() {
        return (0.0, 1.0);
    }
    let total: f64 = cross.iter().sum();
    let mut cum = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &v in cross {
        acc += v;
        cum.push(acc);
    }
    let median_pos = searchsorted_left(&cum, total / 2.0);
    let mut ci = runs
        .iter()
        .position(|r| r.0 <= median_pos && median_pos <= r.1)
        .unwrap_or(0);
    let mut lo_i = ci;
    let mut hi_i = ci;
    let need = min_frac * n as f64;
    while ((runs[hi_i].1 - runs[lo_i].0 + 1) as f64) < need {
        let left = if lo_i > 0 { Some(runs[lo_i - 1]) } else { None };
        let right = if hi_i + 1 < runs.len() {
            Some(runs[hi_i + 1])
        } else {
            None
        };
        if left.is_none() && right.is_none() {
            break;
        }
        let gap_l = left.map(|l| (runs[lo_i].0 as i64 - l.1 as i64) as f64).unwrap_or(1e9);
        let gap_r = right.map(|r| (r.0 as i64 - runs[hi_i].1 as i64) as f64).unwrap_or(1e9);
        if gap_l <= gap_r && left.is_some() {
            lo_i -= 1;
        } else if right.is_some() {
            hi_i += 1;
        } else {
            break;
        }
    }
    ci = lo_i; // the merged range spans lo_i..=hi_i
    let lo = runs[ci].0;
    let hi = runs[hi_i].1;
    ((lo as f64 / n as f64).max(0.0), (((hi + 1) as f64) / n as f64).min(1.0))
}

/// Reading-axis ink projection: ink pixel count per column/row inside the
/// given cross band, smoothed with a *zero-padded* box blur — numpy's
/// `np.convolve(prof, k, mode="same")` pads with zeros, it does not clamp at
/// the edges (clamping would change the profile at the line ends, where the
/// last glyph's window lives).
fn ink_profile(
    mask: &InkMask,
    w: usize,
    h: usize,
    vertical: bool,
    band: (f64, f64),
    smooth: i32,
) -> Vec<f32> {
    let (lo_f, hi_f) = band;
    let (a, mut b) = if vertical {
        let a = (w as f64 * lo_f) as usize;
        let b = (w as f64 * hi_f) as usize;
        (a, b.max(a + 1))
    } else {
        let a = (h as f64 * lo_f) as usize;
        let b = (h as f64 * hi_f) as usize;
        (a, b.max(a + 1))
    };
    let len = if vertical { h } else { w };
    let mut prof = vec![0.0f32; len];
    if a >= if vertical { w } else { h } {
        return prof; // Python's empty slice sums to zero
    }
    b = b.min(if vertical { w } else { h });
    if vertical {
        for y in 0..h {
            let mut m = 0.0f32;
            for x in a..b {
                if mask.get(x, y) {
                    m += 1.0;
                }
            }
            prof[y] = m;
        }
    } else {
        for x in 0..w {
            let mut m = 0.0f32;
            for y in a..b {
                if mask.get(x, y) {
                    m += 1.0;
                }
            }
            prof[x] = m;
        }
    }
    smooth_profile(prof, smooth)
}

/// The profile's *zero-padded* box blur, split out because [`InkProfile`]
/// ships a band-restricted, un-smoothed series that has to be blurred exactly
/// the same way (the profile is the only thing downstream reads).
fn smooth_profile(mut prof: Vec<f32>, smooth: i32) -> Vec<f32> {
    let len = prof.len();
    if smooth > 0 {
        let m = (2 * smooth + 1) as usize;
        let half = smooth as usize;
        let k = 1.0f32 / m as f32;
        let mut out = vec![0.0f32; len];
        for (i, o) in out.iter_mut().enumerate() {
            // numpy 'same' for an odd kernel: centre-aligned, zero outside.
            let mut acc = 0.0f32;
            for j in 0..m {
                let src = i as isize + j as isize - half as isize;
                if src >= 0 && (src as usize) < len {
                    acc += prof[src as usize] * k;
                }
            }
            *o = acc;
        }
        prof = out;
    }
    prof
}

// ─────────────────────────────────────────────────────────────────────────────
// Profile statistics (every walk clamped to the profile length — spec §1.3.1)
// ─────────────────────────────────────────────────────────────────────────────

/// Mass, mid-quartile centre and IQR spread of the profile inside `[lo, hi]`.
fn mid_quartile(prof: &[f32], lo: f64, hi: f64) -> (f64, f64, f64) {
    let n = prof.len();
    let a = clampv(lo, 0.0, n as f64).floor() as usize;
    let b = clampv(hi, 0.0, n as f64).ceil() as usize;
    if b <= a {
        return (0.0, 0.5 * (lo + hi), 0.0);
    }
    let seg = &prof[a..b];
    let mass: f64 = seg.iter().map(|&v| v as f64).sum();
    if mass <= 0.0 {
        return (0.0, 0.5 * (lo + hi), 0.0);
    }
    let mut cum = Vec::with_capacity(seg.len());
    let mut acc = 0.0;
    for &v in seg {
        acc += v as f64;
        cum.push(acc);
    }
    let q1 = searchsorted_left(&cum, mass * 0.25) + a;
    let q3 = searchsorted_left(&cum, mass * 0.75) + a;
    (mass, 0.5 * (q1 as f64 + q3 as f64), (q3 - q1) as f64)
}

/// Trimmed (5%..95%) mass span of the profile around `centre`, clipped to the
/// limits — the glyph's own ink extent for the final pass. A quantile span,
/// not a floor walk: the profile is smoothed, so a contiguous run would walk
/// straight into the neighbour's strokes.
fn ink_run(
    prof: &[f32],
    centre: f64,
    lo_lim: f64,
    hi_lim: f64,
    _floor: f64,
) -> (f64, f64) {
    let n = prof.len();
    let a = clampv(lo_lim, 0.0, n as f64).floor() as usize;
    let b = clampv(hi_lim, 0.0, n as f64).ceil() as usize;
    if b <= a {
        return (centre, centre);
    }
    let seg = &prof[a..b];
    let mass: f64 = seg.iter().map(|&v| v as f64).sum();
    if mass <= 0.0 {
        return (centre, centre);
    }
    let mut cum = Vec::with_capacity(seg.len());
    let mut acc = 0.0;
    for &v in seg {
        acc += v as f64;
        cum.push(acc);
    }
    let lo = (searchsorted_left(&cum, mass * 0.05) + a) as f64;
    let hi = (searchsorted_left(&cum, mass * 0.95) + a) as f64;
    let lo = if hi < lo { hi } else { lo };
    (lo, hi)
}

/// Midpoint of the emptiest run under the profile floor inside `[lo, hi]`,
/// else the lightest point (ties toward the interval centre). Only whole
/// pixels inside the interval are candidates, so the result never leaves it.
fn emptiest_point(prof: &[f32], lo: f64, hi: f64, floor: f64) -> f64 {
    let n = prof.len();
    let a = clampv(lo, 0.0, n as f64).ceil() as usize;
    let b = clampv(hi, 0.0, n as f64).floor() as usize + 1;
    if b <= a {
        return 0.5 * (lo + hi);
    }
    let end = b.min(n); // Python's slice caps at the array end
    if end <= a {
        return 0.5 * (lo + hi);
    }
    let seg = &prof[a..end];
    let thr = floor.max(0.10 * seg.iter().cloned().fold(0.0f32, f32::max) as f64);
    let mut best_len = 0usize;
    let mut best_end = 0usize;
    let mut cur = 0usize;
    for (k, &v) in seg.iter().enumerate() {
        if (v as f64) <= thr {
            cur += 1;
            if cur > best_len {
                best_len = cur;
                best_end = k;
            }
        } else {
            cur = 0;
        }
    }
    if best_len >= 1 {
        return a as f64 + best_end as f64 - (best_len - 1) as f64 / 2.0;
    }
    let size = seg.len() as f64;
    let mut best_i = 0usize;
    let mut best_cost = f64::INFINITY;
    for (k, &v) in seg.iter().enumerate() {
        let cost = v as f64 + 1e-3 * (k as f64 - size / 2.0).abs();
        if cost < best_cost {
            best_cost = cost;
            best_i = k;
        }
    }
    a as f64 + best_i as f64
}

/// True when a window holds two ink blobs separated by a real valley: the
/// mid-quartile then straddles the valley and lands *between* the glyphs
/// (the dakuten case: の sat 9.5 px high until this retry).
fn window_is_bimodal(prof: &[f32], lo: f64, hi: f64, em: f64, opts: &Options) -> bool {
    let n = prof.len();
    let a = clampv(lo, 0.0, n as f64).ceil() as usize;
    let b = clampv(hi, 0.0, n as f64).floor() as usize + 1;
    if b <= a {
        return false;
    }
    let end = b.min(n); // Python's `prof[a:b]` caps at the array end
    if end <= a {
        return false;
    }
    let seg = &prof[a..end];
    let total: f64 = seg.iter().map(|&v| v as f64).sum();
    if total <= 0.0 {
        return false;
    }
    let thr = opts
        .extent_floor
        .max(0.15 * seg.iter().cloned().fold(0.0f32, f32::max) as f64);
    let valley_len = 2usize.max((opts.bimodal_valley_em * em) as usize);
    let mut run = 0usize;
    let mut best = 0usize;
    let mut start = 0usize;
    let mut best_start = 0usize;
    for (k, &v) in seg.iter().enumerate() {
        if (v as f64) <= thr {
            if run == 0 {
                start = k;
            }
            run += 1;
            if run > best {
                best = run;
                best_start = start;
            }
        } else {
            run = 0;
        }
    }
    if best < valley_len {
        return false;
    }
    let left: f64 = seg[..best_start].iter().map(|&v| v as f64).sum();
    let right: f64 = seg[best_start + best..].iter().map(|&v| v as f64).sum();
    left.min(right) / total >= opts.bimodal_min_frac
}

// ─────────────────────────────────────────────────────────────────────────────
// CTC activations (step 0)
// ─────────────────────────────────────────────────────────────────────────────

/// One decoded character's CTC support: the argmax run plus its mean top-1
/// confidence and mean top1−top2 margin (raw logit units).
struct CtcChar {
    run_start: i64,
    run_end: i64,
    conf: f64,
    margin: f64,
}

/// Replay the decoder's greedy walk over the per-timestep top-K (blank resets
/// the repeat state, a repeated argmax extends the run, a new argmax opens
/// the next one) and attach a run to each decoded character. Returns an empty
/// vec on a count mismatch — the caller then falls back to one-timestep runs
/// straight from `char_cols` (the reference's safety net).
fn runs_from_steps(text: &[char], steps: &[Vec<(char, f64)>]) -> Vec<CtcChar> {
    let mut runs: Vec<(i64, i64)> = Vec::new();
    let mut prev: Option<char> = None;
    for (t, alts) in steps.iter().enumerate() {
        if alts.is_empty() {
            continue;
        }
        let top = alts[0].0;
        if top == BLANK {
            prev = None;
            continue;
        }
        if prev == Some(top) {
            if let Some(last) = runs.last_mut() {
                last.1 = t as i64;
            }
            continue;
        }
        runs.push((t as i64, t as i64));
        prev = Some(top);
    }
    if runs.is_empty() || runs.len() != text.len() {
        return Vec::new();
    }
    runs.into_iter()
        .map(|(s, e)| {
            let mut conf = 0.0;
            let mut conf_n = 0.0;
            let mut margin = 0.0;
            let mut margin_n = 0.0;
            for t in s..=e {
                if let Some(alts) = steps.get(t as usize) {
                    if alts.is_empty() {
                        continue;
                    }
                    conf += alts[0].1;
                    conf_n += 1.0;
                    let second = if alts.len() > 1 { alts[1].1 } else { 0.0 };
                    margin += alts[0].1 - second;
                    margin_n += 1.0;
                }
            }
            CtcChar {
                run_start: s,
                run_end: e,
                conf: if conf_n > 0.0 { conf / conf_n } else { 0.0 },
                margin: if margin_n > 0.0 { margin / margin_n } else { 0.0 },
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1 — the advance-class layout template
// ─────────────────────────────────────────────────────────────────────────────

/// Fit `centre_i = X0 + em*(P_i + u_i/2) + ls*i` over center-class characters
/// by Huber-reweighted least squares in column-standardised space (the `ls`
/// column is collinear with the advance column on uniform-class lines, so it
/// is fitted on the residuals afterwards with a ridge prior). Returns
/// `(em, X0, ls)`.
fn robust_template_fit(
    centers: &[f64],
    units: &[f64],
    classes: &[Optical],
    confs: &[f64],
    cross: f64,
    opts: &Options,
) -> (f64, f64, f64) {
    let n = centers.len();
    let mut idx: Vec<usize> = (0..n).filter(|&i| classes[i] == Optical::Center).collect();
    if idx.len() < 2 {
        idx = (0..n).collect();
    }
    if idx.len() < 2 {
        return (
            cross,
            centers.first().copied().unwrap_or(0.0),
            0.0,
        );
    }

    // Pitch scale: weighted median of width-normalized adjacent gaps.
    let mut pitches: Vec<f64> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    for i in 0..n.saturating_sub(1) {
        let gap = centers[i + 1] - centers[i];
        let units_pair = 0.5 * (units[i] + units[i + 1]);
        if gap > 0.0 && units_pair > 0.0 {
            pitches.push(gap / units_pair);
            weights.push((0.5 * (confs[i] + confs[i + 1]) / 8.0).max(0.2));
        }
    }
    let mut pitch = if pitches.is_empty() {
        0.0
    } else {
        weighted_median(&pitches, &weights)
    };
    if pitch <= 0.0 {
        let span = centers[n - 1] - centers[0];
        let total_units = if n > 1 {
            units[1..].iter().sum::<f64>() + 0.5 * (units[0] + units[n - 1])
        } else {
            1.0
        };
        pitch = if span > 0.0 { span / total_units } else { cross };
    }

    // Design rows [P_i + u_i/2, 1] for the fitted subset.
    let mut row_head = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &u in units.iter().take(n) {
        row_head.push(acc + 0.5 * u);
        acc += u;
    }
    let rows: Vec<[f64; 2]> = idx.iter().map(|&i| [row_head[i], 1.0]).collect();
    let ys: Vec<f64> = idx.iter().map(|&i| centers[i]).collect();
    let m = rows.len() as f64;
    let mut scale = [
        (rows.iter().map(|r| r[0] * r[0]).sum::<f64>() / m).sqrt(),
        (rows.iter().map(|r| r[1] * r[1]).sum::<f64>() / m).sqrt(),
    ];
    for s in &mut scale {
        if *s <= 0.0 {
            *s = 1.0;
        }
    }
    let a_std: Vec<[f64; 2]> = rows
        .iter()
        .map(|r| [r[0] / scale[0], r[1] / scale[1]])
        .collect();

    let mut em = pitch;
    let mut x0 = ys[0];
    let mut w: Vec<f64> = Vec::new();
    for it in 0..3 {
        if it == 0 {
            w = idx
                .iter()
                .map(|&i| clampv(confs[i] / 4.0, 0.15, 1.0))
                .collect();
        } else {
            let s = opts.huber_em * pitch;
            w = ys
                .iter()
                .zip(a_std.iter())
                .map(|(&y, a)| {
                    let r = y - (a[0] * (em * scale[0]) + a[1] * x0);
                    clampv(s / r.abs().max(1e-6), 0.05, 1.0)
                })
                .collect();
        }
        let mut ata = [[0.0f64; 2]; 2];
        let mut atb = [0.0f64; 2];
        for (k, a) in a_std.iter().enumerate() {
            let wk = w[k];
            for c in 0..2 {
                atb[c] += wk * a[c] * ys[k];
                for d in 0..2 {
                    ata[c][d] += wk * a[c] * a[d];
                }
            }
        }
        let det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0];
        if !det.is_finite() || det == 0.0 {
            return (pitch, ys[0], 0.0);
        }
        let sol0 = (ata[1][1] * atb[0] - ata[0][1] * atb[1]) / det;
        let sol1 = (ata[0][0] * atb[1] - ata[1][0] * atb[0]) / det;
        em = sol0 / scale[0];
        x0 = sol1;
        if !em.is_finite() || em <= 0.0 {
            return (pitch, ys[0], 0.0);
        }
    }
    // Sanity clamps: the line's own pitch is the scale reference; the cross
    // extent is only a loose backstop (a crop can be much taller than one em).
    em = clampv(em, 0.6 * pitch, 1.7 * pitch);
    em = clampv(em, 0.15 * cross, 3.0 * cross);

    // Letter-spacing only survives when mixed advance classes make the index
    // column non-collinear; ridge prior |ls| <= 0.1·pitch, kept only above
    // 0.02·pitch.
    let mut ls = 0.0;
    let mut distinct: Vec<f64> = idx.iter().map(|&i| units[i]).collect();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    if distinct.len() > 1 && idx.len() >= 4 {
        let pred: Vec<f64> = a_std
            .iter()
            .map(|a| a[0] * (em * scale[0]) + a[1] * x0)
            .collect();
        let resid: Vec<f64> = ys.iter().zip(pred.iter()).map(|(&y, &p)| y - p).collect();
        let xs: Vec<f64> = idx.iter().map(|&i| i as f64).collect();
        let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
        let xc: Vec<f64> = xs.iter().map(|&x| x - mean_x).collect();
        let wx2: f64 = w.iter().zip(xc.iter()).map(|(&wk, &x)| wk * x * x).sum();
        let mut slope: f64 = w
            .iter()
            .zip(xc.iter())
            .zip(resid.iter())
            .map(|((&wk, &x), &r)| wk * r * x)
            .sum::<f64>()
            / wx2.max(1e-9);
        let ridge = (0.35 * pitch / (0.1 * pitch)).powi(2);
        slope = slope * wx2 / (wx2 + ridge);
        if slope.abs() > 0.02 * pitch {
            ls = slope;
        }
    }
    (em, x0, ls)
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4 — translate overlapping pairs apart (before splitting)
// ─────────────────────────────────────────────────────────────────────────────

/// Push pairs whose required half-widths do not fit between their centres
/// apart into the *positive slack* of their flanking pairs, in place. Slack is
/// recomputed after every move, is never driven negative (a move can shrink a
/// neighbour pair's interval but never make it infeasible), and each glyph's
/// move is capped at `translate_max_em`. Whatever deficit remains is left for
/// the split path.
fn translate_apart(
    centers: &mut [f64],
    need: &[f64],
    ink_half: &[f64],
    desired: &[f64],
    em: f64,
    opts: &Options,
) {
    let n = centers.len();
    if n < 2 || !opts.translate_overlap {
        return;
    }
    let cap = opts.translate_max_em * em;

    let slacks = |centers: &[f64]| -> Vec<f64> {
        (0..n - 1)
            .map(|i| (centers[i + 1] - centers[i]) - need[i] - need[i + 1])
            .collect()
    };

    // Would the split path's floor leave ink outside this glyph's box?
    let would_split_cut = |centers: &[f64], i: usize| -> bool {
        for j in [i, i + 1] {
            if j >= n {
                continue;
            }
            let mut h = if j > 0 {
                desired[j].min(0.49 * (centers[j] - centers[j - 1]))
            } else {
                desired[j]
            };
            if j < n - 1 {
                h = h.min(0.49 * (centers[j + 1] - centers[j]));
            }
            if ink_half[j] > h {
                return true;
            }
        }
        false
    };

    for _ in 0..opts.translate_passes.max(1) {
        let mut slack = slacks(centers);
        let mut moved = false;
        for i in 0..n - 1 {
            let d = centers[i + 1] - centers[i];
            let deficit = need[i] + need[i + 1] - d;
            if deficit <= 0.0 {
                continue;
            }
            if opts.translate_gate_cut && !would_split_cut(centers, i) {
                continue;
            }
            let room_l = if i > 0 {
                slack[i - 1].max(0.0).min(cap)
            } else {
                0.0
            };
            let room_r = if i + 1 < n - 1 {
                slack[i + 1].max(0.0).min(cap)
            } else {
                0.0
            };
            let mv = deficit.min(room_l + room_r);
            if mv <= 0.0 {
                continue;
            }
            let total = room_l + room_r;
            let dl = mv * (room_l / total);
            centers[i] -= dl;
            centers[i + 1] += mv - dl;
            slack = slacks(centers);
            moved = true;
        }
        if !moved {
            break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The stage
// ─────────────────────────────────────────────────────────────────────────────

/// Place one line's characters with the shipped defaults. See [`place_with`].
///
/// Returns one `[left, top, right, bottom]` box per character of `text` in
/// crop pixels; each box spans the whole cross axis by contract.
#[allow(clippy::too_many_arguments)]
pub fn place(
    text: &str,
    char_cols: &[f32],
    seq_len_total: usize,
    crop_w: u32,
    crop_h: u32,
    vertical: bool,
    lum: Option<(&[f32], u32, u32)>,
    steps: Option<&[Vec<(char, f64)>]>,
) -> Vec<[f64; 4]> {
    place_with(
        text,
        char_cols,
        seq_len_total,
        crop_w,
        crop_h,
        vertical,
        lum,
        steps,
        &Options::default(),
    )
}

/// The reference `proposed_char_boxes`, stage by stage.
///
/// | input | meaning |
/// |---|---|
/// | `text` | the recognized line, one char per box |
/// | `char_cols` | one fractional column per character, in timesteps (the references apply any `+0.5` convention internally — do not pre-adjust) |
/// | `seq_len_total` | timesteps in the line's decode grid |
/// | `crop_w` / `crop_h` | crop size in pixels |
/// | `vertical` | reading axis = y (`"v"`) or x (`"h"`) |
/// | `lum` | luminance source, `crop_w * crop_h` row-major, dims as given (any other shape, a short array, or `crop < 8` skips the ink pass — spec §1.3.3) |
/// | `steps` | optional per-timestep top-K `[char, score]`, same convention the `gap` kind uses; absent/shorter than `text` falls back to one-timestep runs at `char_cols` |
#[allow(clippy::too_many_arguments)]
pub fn place_with(
    text: &str,
    char_cols: &[f32],
    seq_len_total: usize,
    crop_w: u32,
    crop_h: u32,
    vertical: bool,
    lum: Option<(&[f32], u32, u32)>,
    steps: Option<&[Vec<(char, f64)>]>,
    opts: &Options,
) -> Vec<[f64; 4]> {
    // Guard rails (spec §1.3.3): wrong-shaped or short pixels skip the ink
    // pass; so does a crop too small for a meaningful profile.
    let prof = lum.and_then(|(p, w, h)| {
        if w != crop_w || h != crop_h || w < 8 || h < 8 {
            return None;
        }
        let (w, h) = (w as usize, h as usize);
        if p.len() != w.saturating_mul(h) {
            return None;
        }
        let (mask, _bg_light) = ink_mask(p, w, h);
        let mask = InkMask::Owned { mask, w, h };
        let band = dominant_cross_band(&mask, w, h, vertical, 0.35);
        Some(ink_profile(
            &mask,
            w,
            h,
            vertical,
            band,
            opts.profile_smooth,
        ))
    });
    place_core(
        text,
        char_cols,
        seq_len_total,
        crop_w,
        crop_h,
        vertical,
        prof,
        steps,
        opts,
    )
}

/// [`place_with`] on an [`InkEvidence`]: the packed ink mask instead of the
/// crop's luminance. One crossing instead of `crop_w * crop_h` ARGB ints (the
/// mask is 1 bit per pixel and the image is the ink pass's *only* input, so
/// the boxes are identical — the same claim
/// `ink_profile`+`dominant_cross_band` already rest on), which is what the
/// mobile boundary ships. `steps` and every knob behave as in [`place_with`].
#[allow(clippy::too_many_arguments)]
pub fn place_with_evidence(
    text: &str,
    char_cols: &[f32],
    seq_len_total: usize,
    crop_w: u32,
    crop_h: u32,
    vertical: bool,
    evidence: Option<&InkEvidence>,
    steps: Option<&[Vec<(char, f64)>]>,
    opts: &Options,
) -> Vec<[f64; 4]> {
    // The same guard rails: a missing record, a frame mismatch, a sub-8px crop
    // or a short mask buffer all skip the ink pass instead of reading past.
    let prof = evidence.and_then(|ev| {
        if ev.w != crop_w as usize
            || ev.h != crop_h as usize
            || crop_w < 8
            || crop_h < 8
            || ev.bits.len() < InkMask::packed_len(ev.w, ev.h)
        {
            return None;
        }
        let mask = ev.mask();
        let band = dominant_cross_band(&mask, ev.w, ev.h, vertical, 0.35);
        Some(ink_profile(
            &mask,
            ev.w,
            ev.h,
            vertical,
            band,
            opts.profile_smooth,
        ))
    });
    place_core(
        text,
        char_cols,
        seq_len_total,
        crop_w,
        crop_h,
        vertical,
        prof,
        steps,
        opts,
    )
}

/// [`place_with`] on an [`InkProfile`] — the ink evidence reduced all the way
/// to what the stage reads: the background polarity and the band-restricted,
/// un-smoothed reading-axis counts. The blur, the anchor refinement, the
/// measured extents and the boundary pass are [`place_with`]'s own.
///
/// A correct profile gives identical boxes; the profile's length must be the
/// reading axis (`crop_w` horizontal, `crop_h` vertical) and a sub-8px crop
/// skips the ink pass, exactly as above.
#[allow(clippy::too_many_arguments)]
pub fn place_with_profile(
    text: &str,
    char_cols: &[f32],
    seq_len_total: usize,
    crop_w: u32,
    crop_h: u32,
    vertical: bool,
    profile: Option<&InkProfile>,
    steps: Option<&[Vec<(char, f64)>]>,
    opts: &Options,
) -> Vec<[f64; 4]> {
    let prof_len = if vertical { crop_h } else { crop_w } as usize;
    let prof = profile.and_then(|p| {
        if crop_w < 8 || crop_h < 8 || p.band.len() != prof_len {
            return None;
        }
        if p.band.iter().any(|&c| c >= INK_MAX_COUNT as f32) {
            // A saturated count means the band was too thick to reduce; the
            // crop path would have used real pixels, so refuse rather than
            // place on wrong counts.
            return None;
        }
        Some(smooth_profile(p.band.clone(), opts.profile_smooth))
    });
    place_core(
        text,
        char_cols,
        seq_len_total,
        crop_w,
        crop_h,
        vertical,
        prof,
        steps,
        opts,
    )
}

/// The stage body, with the ink pass already resolved into `prof` (the
/// smoothed reading-axis profile, or `None` for the no-ink pass). This is the
/// reference `proposed_char_boxes` from step 1 on; every entry above differs
/// only in how it produced `prof`.
#[allow(clippy::too_many_arguments)]
fn place_core(
    text: &str,
    char_cols: &[f32],
    seq_len_total: usize,
    crop_w: u32,
    crop_h: u32,
    vertical: bool,
    prof: Option<Vec<f32>>,
    steps: Option<&[Vec<(char, f64)>]>,
    opts: &Options,
) -> Vec<[f64; 4]> {
    let text_chars: Vec<char> = text.chars().collect();
    let n = text_chars.len();
    if n == 0 || seq_len_total == 0 {
        return Vec::new();
    }
    let l = if vertical { crop_h as f64 } else { crop_w as f64 };
    let cross = if vertical { crop_w as f64 } else { crop_h as f64 };
    let px_per_t = l / seq_len_total as f64;

    // 1. CTC activations -----------------------------------------------------
    let mut ctcs = steps
        .filter(|s| !s.is_empty())
        .map(|s| runs_from_steps(&text_chars, s))
        .unwrap_or_default();
    if ctcs.len() != n {
        // Fallback: single-timestep runs straight from `char_cols`.
        ctcs = (0..n)
            .map(|i| {
                let t = char_cols
                    .get(i)
                    .map(|&c| round_half_even(c as f64) as i64)
                    .unwrap_or(0);
                CtcChar {
                    run_start: t,
                    run_end: t,
                    conf: 1.0,
                    margin: 1.0,
                }
            })
            .collect();
    }
    let ctc_centers: Vec<f64> = ctcs
        .iter()
        .map(|c| {
            (c.run_start + c.run_end) as f64 / 2.0 * px_per_t + 0.5 * px_per_t
        })
        .collect();
    let units: Vec<f64> = text_chars.iter().map(|&ch| advance_units(ch)).collect();
    let classes: Vec<Optical> = text_chars.iter().map(|&ch| optical_class(ch)).collect();
    let confs: Vec<f64> = ctcs.iter().map(|c| c.conf).collect();

    // 2. Robust layout template ---------------------------------------------
    let (em, x0, ls) = robust_template_fit(&ctc_centers, &units, &classes, &confs, cross, opts);
    let mut prefixes = vec![0.0f64; n];
    for i in 1..n {
        prefixes[i] = prefixes[i - 1] + units[i - 1];
    }
    let origins: Vec<f64> = (0..n)
        .map(|i| x0 + em * prefixes[i] + ls * i as f64)
        .collect();
    let tmpl: Vec<f64> = (0..n)
        .map(|i| origins[i] + 0.5 * units[i] * em)
        .collect();
    let stride = px_per_t;
    let tol = (opts.anchor_tol_em * em).max(opts.anchor_tol_stride * stride);
    let anchors: Vec<f64> = (0..n)
        .map(|i| {
            if (ctc_centers[i] - tmpl[i]).abs() <= tol {
                tmpl[i]
            } else {
                ctc_centers[i]
            }
        })
        .collect();

    // 3. Ink evidence is already resolved into `prof` (step 3 of the
    //    reference, run by the entry point) -------------------------------
    let mut centers = anchors.clone();
    if let Some(prof) = &prof {
        // Voronoi window masses (before refinement) for the mass gate.
        let mut masses = vec![0.0f64; n];
        for i in 0..n {
            let mut lo = if i == 0 {
                0.0
            } else {
                0.5 * (centers[i - 1] + centers[i])
            };
            let mut hi = if i == n - 1 {
                l
            } else {
                0.5 * (centers[i] + centers[i + 1])
            };
            // Clip the Voronoi window to the character's own cell.
            lo = lo.max(origins[i] - opts.window_em * em);
            hi = hi.min(origins[i] + units[i] * em + opts.window_em * em);
            if hi <= lo {
                lo = origins[i];
                hi = origins[i] + units[i] * em;
            }
            masses[i] = mid_quartile(prof, lo, hi).0;
        }
        let positives: Vec<f64> = masses.iter().copied().filter(|&m| m > 0.0).collect();
        let med_mass = {
            let m = median(&positives);
            if m > 0.0 {
                m
            } else {
                1.0
            }
        };

        let win_center = (opts.window_em * em).max(opts.window_stride * stride);
        for _pass in 0..=opts.refine_passes.max(0) {
            for i in 0..n {
                let lo0 = if i == 0 {
                    0.0
                } else {
                    0.5 * (centers[i - 1] + centers[i])
                };
                let hi0 = if i == n - 1 {
                    l
                } else {
                    0.5 * (centers[i] + centers[i + 1])
                };
                let lo = lo0.max(anchors[i] - win_center);
                let hi = hi0.min(anchors[i] + win_center);
                if hi <= lo {
                    continue;
                }
                // Robust ink centre: mid-quartile of the profile in the
                // window (a neighbour's stroke leaking in moves a centroid,
                // not the quartiles).
                let (mut mass, mut ink_c, mut spread) = mid_quartile(prof, lo, hi);
                if mass < (6.0f64).max(opts.min_mass_frac * med_mass) {
                    continue;
                }
                let smeared = spread > opts.max_spread_em * em;
                let mut retry = false;
                if smeared {
                    if classes[i] != Optical::Center {
                        // Punctuation/small kana: a neighbour's stroke can
                        // dominate their small window; retry around the CTC
                        // run centre (which sits on the glyph).
                        retry = opts.punct_spread_fallback;
                    } else {
                        // A center-class window that is simply smeared: drop
                        // the pull (the reference's behaviour).
                        continue;
                    }
                } else if classes[i] == Optical::Center
                    && opts.bimodal_retry
                    && window_is_bimodal(prof, lo, hi, em, opts)
                {
                    // Two blobs with a real valley: the mid-quartile lands
                    // between them. Retry around the CTC run centre.
                    retry = true;
                }
                if retry {
                    let win_fb =
                        (opts.punct_fallback_window_em * em).max(opts.window_stride * stride);
                    let lo2 = lo0.max(ctc_centers[i] - win_fb);
                    let hi2 = hi0.min(ctc_centers[i] + win_fb);
                    if hi2 > lo2 {
                        let (mass2, ink_c2, spread2) = mid_quartile(prof, lo2, hi2);
                        if mass2 >= (6.0f64).max(opts.min_mass_frac * med_mass)
                            && spread2 <= opts.max_spread_em * em
                            && (ink_c2 - anchors[i]).abs() <= opts.punct_fallback_max_em * em
                        {
                            mass = mass2;
                            ink_c = ink_c2;
                            spread = spread2;
                        }
                    }
                }
                let pull = clampv(
                    ink_c - centers[i],
                    -opts.ink_max_pull_em * em,
                    opts.ink_max_pull_em * em,
                );
                let wgt = if ctcs[i].margin < opts.conf_floor {
                    0.5
                } else {
                    1.0
                };
                centers[i] += wgt * pull;
                let _ = (mass, spread); // measurements gated the pull above
            }
        }
    }

    // 4. Boundaries and boxes ------------------------------------------------
    // Measured ink extents decide how much room each glyph wants; a shared
    // boundary per adjacent pair then goes into the empty ink space between
    // the two glyphs. Extents are trusted only when the centre sits on its
    // own ink (profile above the floor): an offset-ink glyph's "extent" would
    // otherwise be the neighbour's stroke.
    let desired: Vec<f64> = units.iter().map(|&u| 0.5 * u * em).collect();
    let mut need = desired.clone();
    let mut ink_half = desired.clone();
    let mut boundaries: Vec<f64> = Vec::new();
    if !opts.final_pass {
        // Ablation: the previous behaviour — one midpoint boundary per pair.
        boundaries = (0..n.saturating_sub(1))
            .map(|i| 0.5 * (centers[i] + centers[i + 1]))
            .collect();
    }
    if opts.final_pass {
        if let Some(prof) = &prof {
            for i in 0..n {
                let ci = clamp_round(centers[i], 0, prof.len().saturating_sub(1));
                if prof[ci] <= opts.extent_floor as f32 {
                    continue;
                }
                let pad = 0.5 * units[i] * em + opts.extent_window_em * em;
                let mut lo_lim = centers[i] - pad;
                let mut hi_lim = centers[i] + pad;
                if i > 0 {
                    lo_lim = lo_lim.max(0.5 * (centers[i] + centers[i - 1]));
                }
                if i < n - 1 {
                    hi_lim = hi_lim.min(0.5 * (centers[i] + centers[i + 1]));
                }
                let (span_lo, span_hi) = ink_run(prof, centers[i], lo_lim, hi_lim, opts.extent_floor);
                ink_half[i] = (centers[i] - span_lo).max(span_hi - centers[i]);
                let half_need = ink_half[i] + opts.extent_pad_px;
                need[i] = clampv(
                    half_need,
                    desired[i],
                    desired[i] * (1.0 + opts.extent_grow_frac),
                );
            }
        }
        // Translate first: the boundary pass then gives both glyphs their
        // full room in genuinely empty space; only the residue is split.
        translate_apart(&mut centers, &need, &ink_half, &desired, em, opts);
        // h_old: what the midpoint cap allowed — the pass's monotone floor.
        let mut h_old = desired.clone();
        for i in 0..n {
            if i > 0 {
                let d = centers[i] - centers[i - 1];
                if d > 0.0 {
                    h_old[i] = h_old[i].min(0.49 * d);
                }
            }
            if i + 1 < n {
                let d = centers[i + 1] - centers[i];
                if d > 0.0 {
                    h_old[i] = h_old[i].min(0.49 * d);
                }
            }
        }
        let req: Vec<f64> = (0..n).map(|i| need[i].max(h_old[i])).collect();
        for i in 0..n.saturating_sub(1) {
            let d = centers[i + 1] - centers[i];
            let mid = 0.5 * (centers[i] + centers[i + 1]);
            if d <= 0.0 || prof.is_none() {
                boundaries.push(mid);
                continue;
            }
            let prof = prof.as_ref().unwrap();
            let lo = centers[i] + req[i];
            let hi = centers[i + 1] - req[i + 1];
            if hi >= lo {
                // Both glyphs fit: the boundary sits in the emptiest spot of
                // the empty span between their measured inks.
                boundaries.push(emptiest_point(prof, lo, hi, opts.extent_floor));
                continue;
            }
            // They do not fit (tight tracking, merged strokes, a wide glyph
            // beside a narrow one): split at the emptiest point, keeping each
            // box at least `split_floor_frac` of its old half-width.
            let lo = centers[i] + (opts.split_floor_frac * h_old[i]).max(opts.min_half_px);
            let hi = centers[i + 1] - (opts.split_floor_frac * h_old[i + 1]).max(opts.min_half_px);
            boundaries.push(if hi > lo {
                emptiest_point(prof, lo, hi, opts.extent_floor)
            } else {
                mid
            });
        }
    }

    let mut boxes: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let mut half = need[i];
        if !opts.final_pass {
            if i > 0 {
                let d = centers[i] - centers[i - 1];
                if d > 0.0 {
                    half = half.min(0.49 * d);
                }
            }
            if i + 1 < n {
                let d = centers[i + 1] - centers[i];
                if d > 0.0 {
                    half = half.min(0.49 * d);
                }
            }
        } else {
            let mid_prev = if i > 0 {
                0.5 * (centers[i - 1] + centers[i])
            } else {
                0.0
            };
            let mid_next = if i + 1 < n {
                0.5 * (centers[i] + centers[i + 1])
            } else {
                0.0
            };
            if i > 0 {
                let b = boundaries
                    .get(i - 1)
                    .copied()
                    .unwrap_or(mid_prev);
                half = half.min(centers[i] - b);
            }
            if i + 1 < n {
                let b = boundaries.get(i).copied().unwrap_or(mid_next);
                half = half.min(b - centers[i]);
            }
        }
        half = half.max(1.0);
        let mut a = clampv(centers[i] - half, 0.0, l);
        let mut b = clampv(centers[i] + half, 0.0, l);
        if b - a < 1.0 {
            b = (a + 1.0).min(l);
            a = (b - 1.0).max(0.0);
        }
        boxes.push((a, b));
    }
    if vertical {
        boxes.into_iter().map(|(a, b)| [0.0, a, cross, b]).collect()
    } else {
        boxes.into_iter().map(|(a, b)| [a, 0.0, b, cross]).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The device regression (`CharPlacementEdgeCaseTest`): the LAST glyph's
    /// Voronoi window is clipped to the line length `L`, and the bimodal walk
    /// reads `prof[a .. floor(hi)+1]` — one past the end when `hi == L`, the
    /// common case for a det crop that hugs the final glyph. Python never
    /// showed it (`prof[a:b]` slices past the end); Kotlin threw inside a
    /// callback that swallowed it and half a page rendered blank. The
    /// profile is white background with one ink blob per glyph so the mass
    /// gate passes, the window is not smeared, and the bimodal walk runs for
    /// the last glyph exactly as on device.
    #[test]
    fn last_glyph_window_reaching_the_line_end_does_not_walk_past_the_profile() {
        let crop_w = 80usize;
        let crop_h = 40usize;
        let mut lum = vec![255.0f32; crop_w * crop_h]; // white background
        let mut ink = |x0: usize, x1: usize| {
            for y in 4..36 {
                for x in x0..x1 {
                    lum[y * crop_w + x] = 32.0; // 0x202020 channel mean
                }
            }
        };
        ink(24, 36); // glyph 0 (centre ~30t * 20px/t)
        ink(64, 76); // glyph 1, the LAST one (centre ~70; window clips to L=80)

        let boxes = place(
            "あい",
            &[1.0, 3.0],
            4,
            crop_w as u32,
            crop_h as u32,
            false,
            Some((&lum, crop_w as u32, crop_h as u32)),
            None,
        );
        assert_eq!(boxes.len(), 2, "both glyphs get a box");
        let last = boxes[1];
        assert!(
            last[0] <= 70.0 && last[2] >= 70.0,
            "last box must cover its ink centre (x=70): {last:?}"
        );
        // The contract: full cross axis, box order along the reading axis.
        for b in &boxes {
            assert_eq!(b[1], 0.0, "horizontal boxes start at the top edge");
            assert_eq!(b[3], crop_h as f64, "…and span the whole cross axis");
        }
        assert!(boxes[0][2] <= boxes[1][0] + 1e-9 || boxes[0][0] < boxes[1][0]);
    }

    /// Guard rails (spec §1.3.3): degenerate inputs fall back or return
    /// empty, never panic.
    #[test]
    fn guard_rails_degrade_instead_of_panicking() {
        assert!(place("", &[], 0, 100, 50, false, None, None).is_empty());
        assert!(place("あ", &[0.0], 0, 100, 50, false, None, None).is_empty());
        // Text longer than the top-K run count falls back to char_cols.
        let steps: Vec<Vec<(char, f64)>> = vec![vec![('あ', 2.0)]];
        let boxes = place("あいう", &[0.0, 1.0, 2.0], 4, 100, 50, false, None, Some(&steps));
        assert_eq!(boxes.len(), 3, "the char_cols fallback still places every char");
        // Short pixel array: no ink pass, boxes still come out.
        let short = vec![0.0f32; 10];
        let boxes = place(
            "あい",
            &[0.0, 2.0],
            4,
            100,
            50,
            false,
            Some((&short, 100, 50)),
            None,
        );
        assert_eq!(boxes.len(), 2, "a short pixel array skips the ink pass");
        // Crop < 8 skips the ink pass.
        let tiny = vec![0.0f32; 4 * 4];
        let boxes = place("あい", &[0.0, 1.0], 2, 4, 4, false, Some((&tiny, 4, 4)), None);
        assert_eq!(boxes.len(), 2, "a sub-8px crop skips the ink pass");
    }

    /// Boxes span the whole cross axis on both orientations, and the count
    /// follows `text`.
    #[test]
    fn boxes_span_the_cross_axis_in_both_orientations() {
        let lum = vec![255.0f32; 40 * 120];
        let h = place("あい", &[0.0, 1.0], 2, 40, 120, false, Some((&lum, 40, 120)), None);
        assert!(h.iter().all(|b| b[1] == 0.0 && b[3] == 120.0));
        let v = place("あい", &[0.0, 1.0], 2, 40, 120, true, Some((&lum, 40, 120)), None);
        assert!(v.iter().all(|b| b[0] == 0.0 && b[2] == 40.0));
    }

    // ─── ink evidence: crop luminance vs. the packed mask vs. the profile ────

    /// A line crop's luminance: paper background, one ink blob per glyph, plus
    /// a ruby-ish strip outside the body band so the band's choice is
    /// load-bearing. `invert` gives the light-on-dark polarity.
    fn cap_crop(w: usize, h: usize, glyphs: &[(usize, usize)], height: usize, invert: bool) -> Vec<f32> {
        let mut lum = vec![255.0f32; w * h];
        let paint = |lum: &mut Vec<f32>, x: usize, y: usize, v: f32| lum[y * w + x] = v;
        for &(c0, c1) in glyphs {
            for x in c0..(c1 + 3).min(w) {
                for y in 2..(2 + height).min(h) {
                    paint(&mut lum, x, y, 32.0);
                }
            }
        }
        // Ink in the extreme edges (ruby / the neighbouring line): the band has
        // to exclude it, so a wrong band shows up as different boxes.
        for x in 0..w {
            paint(&mut lum, x, 0, 32.0);
            paint(&mut lum, x, h - 1, 32.0);
        }
        if invert {
            for v in lum.iter_mut() {
                *v = 255.0 - *v;
            }
        }
        lum
    }

    /// What a boundary measures: the border sample's polarity and the mask at
    /// 1 bit per pixel. Deliberately written as the *caller's* job (luminance
    /// in, packed bits out) rather than by calling the crate's own helpers, so
    /// the test pins the reduction itself and not a tautology.
    fn measure_evidence(lum: &[f32], w: usize, h: usize) -> (bool, Vec<u8>) {
        let spec = ink_spec();
        let mut border: Vec<f32> = Vec::new();
        let mut j = 0usize;
        while j < w {
            border.push(lum[j]);
            border.push(lum[(h - 1) * w + j]);
            j += spec.border_stride;
        }
        let mut i = 0usize;
        while i < h {
            border.push(lum[i * w]);
            border.push(lum[i * w + w - 1]);
            i += spec.border_stride;
        }
        let mut sorted = border.clone();
        sorted.sort_by(f32::total_cmp);
        // The crate's median: the mean of the two middle values on an even count.
        let n = sorted.len();
        let med = if n % 2 == 1 {
            sorted[n / 2] as f64
        } else {
            0.5 * (sorted[n / 2 - 1] as f64 + sorted[n / 2] as f64)
        };
        let bg_light = med > spec.bg_median_above as f64;
        let mut bits = vec![0u8; InkMask::packed_len(w, h)];
        for y in 0..h {
            for x in 0..w {
                let v = lum[y * w + x];
                let ink = if bg_light {
                    v < spec.ink_below
                } else {
                    v > spec.ink_above
                };
                if ink {
                    let i = y * w + x;
                    bits[i / 8] |= 0x80u8 >> (i % 8);
                }
            }
        }
        (bg_light, bits)
    }

    /// And the minimum reduction: the same polarity, the cross-axis counts, and
    /// (in the band the crate's own `cross_band` names) the reading-axis
    /// counts. Two crossings on a real boundary, one function here.
    fn measure_profile(
        lum: &[f32],
        w: usize,
        h: usize,
        vertical: bool,
    ) -> (InkProfile, (usize, usize)) {
        let (bg_light, bits) = measure_evidence(lum, w, h);
        let mask = InkMask::Packed { bits, w, h };
        let (cross_len, read_len) = if vertical { (w, h) } else { (h, w) };
        let mut cross = vec![0.0f64; cross_len];
        for k in 0..cross_len {
            let mut m = 0.0f64;
            for j in 0..read_len {
                let ink = if vertical {
                    mask.get(k, j)
                } else {
                    mask.get(j, k)
                };
                if ink {
                    m += 1.0;
                }
            }
            cross[k] = m;
        }
        let (lo, hi) = cross_band(&cross, ink_spec().band_min_frac);
        let mut band = vec![0.0f32; read_len];
        for r in 0..read_len {
            let mut m = 0.0f32;
            for c in lo..hi.min(cross_len) {
                let ink = if vertical {
                    mask.get(c, r)
                } else {
                    mask.get(r, c)
                };
                if ink {
                    m += 1.0;
                }
            }
            band[r] = m;
        }
        (InkProfile { bg_light, band }, (lo, hi))
    }

    /// The evidence entries are pure re-plumbing of the crop entry: a packed
    /// mask measured from the crop must place *identical* boxes, and so must
    /// the minimum profile. The sweep covers both orientations, both
    /// polarities, edge ink that forces a real band, crop shapes from
    /// single-glyph to many, with and without `steps`.
    #[test]
    fn evidence_and_profile_entries_match_the_crop_entry_bit_for_bit() {
        let mut state = 0xc0ffee_1234u64;
        let lcg = move |n: &mut u64| -> u32 {
            *n = n.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*n >> 33) as u32
        };
        let mut checked = 0;
        for case in 0..64usize {
            let vertical = case % 4 == 0;
            let (read, cross) = if vertical {
                (40 + (lcg(&mut state) as usize % 200), 16 + (lcg(&mut state) as usize % 60))
            } else {
                (40 + (lcg(&mut state) as usize % 200), 16 + (lcg(&mut state) as usize % 60))
            };
            let (w, h) = if vertical { (cross, read) } else { (read, cross) };
            let n = 2 + (lcg(&mut state) as usize % 9);
            let pitch = (read / n).max(6);
            let glyphs: Vec<(usize, usize)> = (0..n)
                .map(|i| (i * pitch + 1 + (lcg(&mut state) as usize % 3), i * pitch + pitch / 2))
                .collect();
            let lum = cap_crop(read, cross, &glyphs, (cross * 2 / 3).max(3), case % 2 == 1);
            let text: String = "あいうえおかきくけこさしすせそ".chars().take(n).collect();
            let cols: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let steps: Option<Vec<Vec<(char, f64)>>> = if case % 3 == 0 {
                None
            } else {
                Some(
                    (0..(n + 2))
                        .map(|t| {
                            let ch = if t % 3 == 2 {
                                BLANK
                            } else {
                                text.chars().nth(t % n).unwrap_or(BLANK)
                            };
                            vec![(ch, 4.0 - (t % 3) as f64 * 0.5), ('あ', 1.0)]
                        })
                        .collect(),
                )
            };
            let seq = n + 2;
            let opts = Options::default();
            let (bg_light, bits) = measure_evidence(&lum, w, h);
            let ev = InkEvidence::from_packed(bg_light, &bits, w as u32, h as u32)
                .expect("a full frame yields evidence");
            let (prof, band) = measure_profile(&lum, w, h, vertical);
            let via_crop = place_with(
                &text, &cols, seq, w as u32, h as u32, vertical,
                Some((&lum, w as u32, h as u32)), steps.as_deref(), &opts,
            );
            let via_mask = place_with_evidence(
                &text, &cols, seq, w as u32, h as u32, vertical,
                Some(&ev), steps.as_deref(), &opts,
            );
            assert_eq!(
                via_crop, via_mask,
                "packed mask diverged: case {case} w={w} h={h} n={n} vert={vertical} band={band:?}"
            );
            let via_prof = place_with_profile(
                &text, &cols, seq, w as u32, h as u32, vertical,
                Some(&prof), steps.as_deref(), &opts,
            );
            assert_eq!(
                via_crop, via_prof,
                "minimum profile diverged: case {case} w={w} h={h} n={n} vert={vertical} band={band:?}"
            );
            // The fixture must actually be exercising the ink pass, or the
            // equality above is vacuous.
            let no_ink = place_with(
                &text, &cols, seq, w as u32, h as u32, vertical, None, steps.as_deref(), &opts,
            );
            if via_crop != no_ink {
                checked += 1;
            }
        }
        assert!(checked >= 16, "only {checked} cases moved boxes; sweep too weak");
    }

    /// The guards: a missing record, a short mask buffer, a wrong-shaped
    /// profile and a sub-8px crop all skip the ink pass (the template-only
    /// boxes), never read past the buffer and never panic.
    #[test]
    fn evidence_entries_keep_the_crop_guards() {
        let (w, h) = (200usize, 44usize);
        let glyphs = vec![(10usize, 14usize), (50, 54), (90, 94), (130, 134), (170, 174)];
        let lum = cap_crop(w, h, &glyphs, 24, false);
        let text = "あいうえお";
        let cols = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let opts = Options::default();
        let inked = place_with(
            text, &cols, 6, w as u32, h as u32, false,
            Some((&lum, w as u32, h as u32)), None, &opts,
        );
        let bare = place_with(text, &cols, 6, w as u32, h as u32, false, None, None, &opts);
        assert_ne!(inked, bare, "the fixture must move boxes");
        let (bg_light, bits) = measure_evidence(&lum, w, h);
        let good = InkEvidence::from_packed(bg_light, &bits, w as u32, h as u32).unwrap();
        // A short buffer: no ink pass.
        let short = InkEvidence::from_packed(bg_light, &bits[..bits.len() - 1], w as u32, h as u32);
        assert!(short.is_none(), "a short mask is refused outright");
        // A frame mismatch: no ink pass (the record is for another crop).
        assert_eq!(
            place_with_evidence(
                text, &cols, 6, w as u32, h as u32, false,
                Some(&good), None, &opts,
            ),
            inked
        );
        // (The comparison frame has to be the *wider* one: the box geometry
        // reads `crop_w` through the CTC stride, so the template-only boxes of
        // a 202px frame are not the template-only boxes of a 200px one.)
        let bare_wide =
            place_with(text, &cols, 6, (w + 2) as u32, h as u32, false, None, None, &opts);
        assert_eq!(
            place_with_evidence(
                text, &cols, 6, (w + 2) as u32, h as u32, false,
                Some(&good), None, &opts,
            ),
            bare_wide,
            "an evidence record for another frame skips the ink pass"
        );
        // A wrong-shaped profile: no ink pass. A saturated count: refused.
        let mut p = InkProfile {
            bg_light,
            band: vec![1.0; w],
        };
        p.band.pop();
        assert_eq!(
            place_with_profile(text, &cols, 6, w as u32, h as u32, false, Some(&p), None, &opts),
            bare
        );
        let mut sat = InkProfile {
            bg_light,
            band: vec![1.0; w],
        };
        sat.band[3] = INK_MAX_COUNT as f32;
        assert_eq!(
            place_with_profile(text, &cols, 6, w as u32, h as u32, false, Some(&sat), None, &opts),
            bare,
            "a saturated count is refused rather than placed on"
        );
        // Sub-8px crops skip the ink pass on every entry.
        let tiny = vec![0.0f32; 4 * 4];
        let tiny_ev = InkEvidence::from_packed(true, &[0u8; 2], 4, 4);
        assert!(tiny_ev.is_some(), "packing still describes a 4x4 frame");
        assert_eq!(
            place_with("あい", &[0.0, 1.0], 2, 4, 4, false, Some((&tiny, 4, 4)), None, &opts),
            place_with_evidence(
                "あい", &[0.0, 1.0], 2, 4, 4, false, tiny_ev.as_ref(), None, &Options::default(),
            )
        );
        assert_eq!(
            place_with("あい", &[0.0, 1.0], 2, 4, 4, false, Some((&tiny, 4, 4)), None, &opts),
            place_with_profile(
                "あい", &[0.0, 1.0], 2, 4, 4, false,
                Some(&InkProfile { bg_light: true, band: vec![0.0; 4] }), None, &Options::default(),
            )
        );
    }

    /// The published integer ink test is the float one, exactly, for every
    /// possible channel sum: the facade can pack the mask with integer
    /// compares and still place the same boxes.
    #[test]
    fn ink_spec_integer_agrees_with_float() {
        let spec = ink_spec();
        assert_eq!(spec.ink_below_sum, 330);
        assert_eq!(spec.ink_above_sum, 436);
        for sum in 0..=765i32 {
            let lum = sum as f32 / 3.0;
            assert_eq!(
                is_ink_sum(sum, true),
                lum < spec.ink_below,
                "light background, sum {sum}"
            );
            assert_eq!(
                is_ink_sum(sum, false),
                lum > spec.ink_above,
                "dark background, sum {sum}"
            );
        }
    }

    /// The published measurement constants, the packed-mask length arithmetic
    /// and the band probe — the numbers a boundary reads instead of hardcoding.
    #[test]
    fn ink_spec_and_band_probe_are_the_documented_reduction() {
        let spec = ink_spec();
        assert_eq!(spec.ink_below, 110.0);
        assert_eq!(spec.ink_above, 145.0);
        assert_eq!(spec.bg_median_above, 128.0);
        assert_eq!(spec.border_stride, 7);
        assert_eq!(spec.band_min_frac, 0.35);
        assert_eq!(spec.profile_smooth, 2);
        assert_eq!(spec.max_count, INK_MAX_COUNT);
        assert_eq!(InkMask::packed_len(0, 0), 0);
        assert_eq!(InkMask::packed_len(1, 1), 1);
        assert_eq!(InkMask::packed_len(8, 1), 1);
        assert_eq!(InkMask::packed_len(9, 1), 2);
        assert_eq!(InkMask::packed_len(568, 60), 4260);
        // A mask with an even number of pixels still gets a final partial byte.
        let mut bits = vec![0u8; InkMask::packed_len(3, 1)];
        bits[0] = 0b1010_0000;
        let ev = InkEvidence::from_packed(false, &bits, 3, 1).unwrap();
        let m = ev.mask();
        assert!(m.get(0, 0) && !m.get(1, 0) && m.get(2, 0), "MSB-first per pixel");
        // A byte straddles a row boundary when w is not a multiple of 8, so the
        // bit comes from the *global* index, not from x: 10-wide rows put row 1's
        // first pixels in the same byte as row 0's last ones.
        let bits = {
            let mut b = vec![0u8; InkMask::packed_len(10, 2)];
            b[0] |= 0x80u8 >> 0; // (0,0)
            b[1] |= 0x80u8 >> 2; // i = 10 -> byte 1, bit 2
            b
        };
        let m = InkEvidence::from_packed(false, &bits, 10, 2).unwrap().mask();
        assert!(m.get(0, 0) && !m.get(9, 0) && m.get(0, 1) && !m.get(9, 1));
        // The band probe: a full cross axis has no ink anywhere → the whole
        // axis (the crate's own "no runs" bail), and a centred blob gives the
        // band the mask's own ink occupies.
        assert_eq!(cross_band(&[], 0.35), (0, 0));
        assert_eq!(cross_band(&[0.0; 40], 0.35), (0, 40));
        assert_eq!(
            cross_band(&[0.0, 0.0, 8.0, 9.0, 0.0], 0.35),
            (2, 4),
            "the band is the blob's own run, widened only until min_frac is met"
        );
        // Two runs and a gap narrower than the minimum: they merge, so the
        // band spans the gap (the ruby/Neighbouring-line dodge is a *merge*,
        // never a trim of the dominant run).
        assert_eq!(cross_band(&[9.0, 0.0, 0.0, 0.0, 8.0], 0.35), (0, 5));
    }
}
