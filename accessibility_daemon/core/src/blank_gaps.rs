//! Mobile `BlankGaps` / `GapDetector` (#44 Feature 2): the spacing-ratio
//! blank detector and the placeholder materialiser that follows it, pure so
//! they are host-tested and reachable from a nav-only build.
//!
//! Detection is the measured one: a character dropped by the recogniser
//! leaves roughly **twice** the median spacing between the neighbouring
//! emitted characters (measured 2.00× at deletion sites versus 1.00× for
//! ordinary adjacent pairs over 231 paired lines). The trigger is
//! **per-orientation** — vertical fires at 1.6 (measured recall 1.00, no false
//! positives), horizontal at 1.8 and still weak on its own — and the two
//! thresholds are configurable rather than hardcoded.
//!
//! Geometry is taken from, in order: `char_boxes` centres (pixels),
//! `char_cols` when it matches the text length (timesteps, converted with the
//! crop length), and finally a walk of `raw_alternatives` that reproduces the
//! recogniser's own CTC collapse rules.
//!
//! Materialisation inserts the [`GAP_CHAR`] placeholder right-to-left so the
//! detector's indices stay valid, growing every parallel per-character list
//! together, and is idempotent.

use crate::models::{BoundingBox, LineResult, GAP_CHAR};
use std::collections::BTreeMap;

/// Mobile `GapDetector.DEFAULT_VERTICAL_RATIO`: measured recall 1.00 and no
/// false positives on the vertical bench.
pub const DEFAULT_VERTICAL_RATIO: f32 = 1.6;

/// Mobile `GapDetector.DEFAULT_HORIZONTAL_RATIO`: the best available point for
/// horizontal lines, which are weak on their own (a quarter of their
/// deletions leave no wider gap at all) and need the component signal before
/// showing anything.
pub const DEFAULT_HORIZONTAL_RATIO: f32 = 1.8;

/// Mobile `GapDetector.DEFAULT_TIMESTEP_STRIDE_PX`: the model's own
/// downsampling stride, used when the crop length was never cached.
pub const DEFAULT_TIMESTEP_STRIDE_PX: f32 = 8.0;

/// Mobile `GapDetector.MIN_EMITTED_CHARS`: fewer characters than this cannot
/// produce a spacing pair.
pub const MIN_EMITTED_CHARS: usize = 2;

/// Mobile `TIMESTEP_BLANK_CHAR`: blank timesteps are stored in
/// `raw_alternatives` as the ideographic space. Distinct from [`GAP_CHAR`],
/// the reversible placeholder.
pub const TIMESTEP_BLANK_CHAR: char = '\u{3000}';

/// The measured vertical threshold, kept under the name the desktop and the
/// spec page already use (mobile `GapDetector.DEFAULT_VERTICAL_RATIO`).
pub const BLANK_GAP_RATIO: f32 = DEFAULT_VERTICAL_RATIO;

/// Mobile `Gap`: one detected deletion site.
///
/// `insert_at` is the **character index in [`LineResult::text`]** the
/// placeholder belongs at — the gap sits between `text[insert_at - 1]` and
/// `text[insert_at]`. `ratio` is this pair's spacing divided by the line's own
/// median spacing (scale-invariant, so it carries across every geometry
/// source), and `span_px` is the pair's spacing in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gap {
    pub insert_at: usize,
    pub ratio: f32,
    pub span_px: f32,
}

impl Gap {
    /// The plan document's name for [`Gap::ratio`]; same value, so downstream
    /// call sites can use either spelling.
    pub fn pitch_ratio(&self) -> f32 {
        self.ratio
    }
}

/// Mobile `GapDetector`: spacing-ratio detection over a recognised line.
#[derive(Debug, Clone, Copy)]
pub struct GapDetector {
    /// Trigger for vertical (tategumi) lines.
    pub vertical_threshold: f32,
    /// Trigger for horizontal lines.
    pub horizontal_threshold: f32,
}

impl Default for GapDetector {
    fn default() -> Self {
        GapDetector {
            vertical_threshold: DEFAULT_VERTICAL_RATIO,
            horizontal_threshold: DEFAULT_HORIZONTAL_RATIO,
        }
    }
}

impl GapDetector {
    /// Mobile `GapDetector(verticalThreshold, horizontalThreshold)`.
    pub fn new(vertical_threshold: f32, horizontal_threshold: f32) -> Self {
        GapDetector {
            vertical_threshold,
            horizontal_threshold,
        }
    }

    /// The threshold that applies to a line of this orientation.
    pub fn threshold_for(&self, is_vertical: bool) -> f32 {
        if is_vertical {
            self.vertical_threshold
        } else {
            self.horizontal_threshold
        }
    }

    /// Detect gaps using the line's own orientation threshold.
    pub fn detect(&self, line: &LineResult) -> Vec<Gap> {
        self.detect_with(line, self.threshold_for(line.is_vertical))
    }

    /// Detect gaps using an explicit `threshold`, so a caller can sweep the
    /// curve without rebuilding the detector. A pair triggers when
    /// `spacing / median_spacing >= threshold` — the `>=` form is the one the
    /// measured sweep is quoted in ("ratio ≥ 1.6").
    pub fn detect_with(&self, line: &LineResult, threshold: f32) -> Vec<Gap> {
        let n = line.text.chars().count();
        if n < MIN_EMITTED_CHARS {
            return Vec::new();
        }
        let Some((centres, px_per_unit)) = geometry_of(line) else {
            return Vec::new();
        };
        if centres.len() != n {
            return Vec::new();
        }

        let spacings: Vec<f32> = centres
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();
        let pitch = median_of(&mut spacings.clone());
        if pitch <= 0.0 {
            return Vec::new(); // degenerate geometry: everything on one pixel
        }

        spacings
            .iter()
            .enumerate()
            .filter_map(|(k, s)| {
                let ratio = s / pitch;
                (ratio >= threshold).then_some(Gap {
                    insert_at: k + 1, // between chars k and k+1
                    ratio,
                    span_px: s * px_per_unit,
                })
            })
            .collect()
    }
}

/// Mobile `GapDetector.geometryOf`: `(centres, pixels per unit)`, or `None`
/// when no source describes this text.
fn geometry_of(line: &LineResult) -> Option<(Vec<f32>, f32)> {
    let n = line.text.chars().count();
    if n == 0 {
        return None;
    }

    if line.char_boxes.len() >= n {
        // Boxes are already pixels. Centres are integer maths, matching the
        // mobile `JpDictRect.centerY()` / `centerX()` exactly.
        let centres = line
            .char_boxes
            .iter()
            .take(n)
            .map(|b| {
                if line.is_vertical {
                    (b.y + b.h / 2) as f32
                } else {
                    (b.x + b.w / 2) as f32
                }
            })
            .collect();
        return Some((centres, 1.0));
    }

    let px_per_timestep = pixel_per_timestep(line);
    if line.char_cols.len() == n {
        return Some((line.char_cols.clone(), px_per_timestep));
    }

    // Last resort: walk the raw per-timestep lists. Reached only under
    // [`needs_raw_alternatives`], which is the same predicate a host asks
    // before handing those lists over.
    if needs_raw_alternatives(&line.text, line.char_boxes.len(), line.char_cols.len()) {
        let derived = timestep_columns(&line.raw_alternatives);
        if derived.len() == n {
            return Some((derived, px_per_timestep));
        }
    }
    None
}

/// Whether the detector's geometry reaches the `raw_alternatives` walk: the
/// char boxes do not cover the text and the CTC columns do not either, so the
/// per-timestep top-K lists are the only source left.
///
/// This is the detector's source *precedence*, not its threshold: a host that
/// wants to know whether it must hand those lists over (they are the largest
/// per-line payload a recognised line carries) can ask first and skip building
/// them when the answer is `false` — which is every line whose char boxes the
/// layout stage already produced.
pub fn needs_raw_alternatives(
    text: &str,
    char_boxes_len: usize,
    char_cols_len: usize,
) -> bool {
    let n = text.chars().count();
    n > 0 && char_boxes_len < n && char_cols_len != n
}

/// Mobile `GapDetector.pixelPerTimestep`: pixels per CTC timestep along the
/// reading axis (`crop_h / seq_len_total` vertical, `crop_w / seq_len_total`
/// horizontal), falling back to the model's own stride when the crop geometry
/// was never cached.
fn pixel_per_timestep(line: &LineResult) -> f32 {
    let len = if line.is_vertical {
        line.crop_h
    } else {
        line.crop_w
    };
    if len > 0 && line.seq_len_total > 0 {
        len as f32 / line.seq_len_total as f32
    } else {
        DEFAULT_TIMESTEP_STRIDE_PX
    }
}

/// Mobile `timestepColumns`: emitted character → CTC timestep column,
/// recovered from the raw per-timestep top-K lists.
///
/// The walk reproduces the recogniser's own collapse rules, and the unit test
/// pins each assumption:
///
/// - `raw[t]` is the top-N list for timestep `t`, and the emitted character
///   for that timestep is its **first** entry (the argmax).
/// - Blank is stored as [`TIMESTEP_BLANK_CHAR`] and **resets** the previous
///   character, so a repeat after a blank is emitted.
/// - A space is always emitted and never collapses.
/// - Any other character collapses only against the immediately preceding
///   emitted character.
///
/// Returns an empty list for input that derives nothing; the caller compares
/// the length against the text length and gives up when they disagree.
pub fn timestep_columns(raw: &[Vec<(char, f32)>]) -> Vec<f32> {
    let mut cols: Vec<f32> = Vec::with_capacity(raw.len());
    let mut prev: Option<char> = None;
    for (t, alts) in raw.iter().enumerate() {
        let Some(top) = alts.first() else {
            continue;
        };
        let ch = top.0;
        if ch == TIMESTEP_BLANK_CHAR {
            prev = None;
        } else if ch == ' ' {
            cols.push(t as f32);
            prev = Some(' ');
        } else if Some(ch) == prev {
            // CTC repeat collapse.
        } else {
            cols.push(t as f32);
            prev = Some(ch);
        }
    }
    cols
}

/// Mobile `medianOf`: even counts average the two middles.
pub fn median_of(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

/// Mobile `interpolateGapBox`: a placeholder box centred between the two
/// neighbours it was dropped from, sized as the mean of their extents along
/// the reading axis. At either end of the line there is only one neighbour,
/// and its box is reused.
pub fn interpolate_gap_box(boxes: &[BoundingBox], index: usize, is_vertical: bool) -> BoundingBox {
    let before = index.checked_sub(1).and_then(|i| boxes.get(i));
    let after = boxes.get(index);
    let (Some(before), Some(after)) = (before, after) else {
        return before
            .or(after)
            .cloned()
            .unwrap_or_else(|| BoundingBox::new(0, 0, 0, 0, 1.0));
    };
    // Integer maths, matching the mobile `JpDictRect` arithmetic exactly.
    let (bl, bt, br, bb) = (before.left(), before.top(), before.right(), before.bottom());
    let (al, at, ar, ab) = (after.left(), after.top(), after.right(), after.bottom());
    if is_vertical {
        let centre_y = ((bt + bb) / 2 + (at + ab) / 2) / 2;
        let height = ((bb - bt + (ab - at)) / 2).max(1);
        BoundingBox::new(
            bl.min(al),
            centre_y - height / 2,
            br.max(ar) - bl.min(al),
            height,
            before.confidence,
        )
    } else {
        let centre_x = ((bl + br) / 2 + (al + ar) / 2) / 2;
        let width = ((br - bl + (ar - al)) / 2).max(1);
        BoundingBox::new(
            centre_x - width / 2,
            bt.min(at),
            width,
            bb.max(ab) - bt.min(at),
            before.confidence,
        )
    }
}

/// Mobile `BlankGaps.columnFor`: the CTC timestep column for a new position —
/// midway between the neighbours' columns when they are known, else the
/// preceding column, else the following one, else 0.
pub fn column_for(line: &LineResult, insert_at: usize) -> f32 {
    let cols = &line.char_cols;
    if cols.is_empty() || cols.len() != line.text.chars().count() {
        return 0.0;
    }
    let before = insert_at.checked_sub(1).and_then(|i| cols.get(i));
    let after = cols.get(insert_at);
    match (before, after) {
        (Some(b), Some(a)) => (b + a) / 2.0,
        (Some(b), None) => *b,
        (None, Some(a)) => *a,
        (None, None) => 0.0,
    }
}

/// Mobile `LineResult.withGapCharAt`: insert the placeholder at `index`,
/// growing **every** parallel per-character list together so they still
/// describe the same characters at the same indices.
///
/// - `text` gains [`GAP_CHAR`] at `index`.
/// - `char_boxes` gains an interpolated box when the list was full-length and
///   non-empty.
/// - `alternatives` gains `gap_alternatives` (default `[(GAP_CHAR, 0.0)]`)
///   under the same condition.
/// - `char_cols` gains `column` under the same condition.
/// - `overrides` keys move: every key `>= index` shifts by +1, so a filled
///   blank or an applied correction keeps pointing at its own character. The
///   placeholder itself is **not** inserted as an override.
///
/// Lists that carry no data are left empty rather than given a single
/// mismatched entry — "empty" means "unknown". An index past the end is a
/// no-op.
pub fn with_gap_char_at(
    line: &LineResult,
    index: usize,
    column: f32,
    gap_alternatives: Option<Vec<(char, f32)>>,
) -> LineResult {
    let chars: Vec<char> = line.text.chars().collect();
    if index > chars.len() {
        return line.clone();
    }
    materialise_gap_plan(
        line,
        &single_gap_plan(
            line,
            chars.len(),
            GapInsertion {
                index,
                column,
                placeholder_box: interpolate_gap_box(&line.char_boxes, index, line.is_vertical),
                ratio: 0.0,
                span_px: 0.0,
                gap_alternatives,
            },
        ),
    )
}

/// Mobile `LineResult.withGapAt`: alias for [`with_gap_char_at`] under the
/// name the gap subtask used. Same behaviour, same arguments.
pub fn with_gap_at(line: &LineResult, index: usize, column: f32) -> LineResult {
    with_gap_char_at(line, index, column, None)
}

/// The original desktop entry point: [`with_gap_char_at`] with an unknown
/// column and the default placeholder alternatives.
pub fn with_gap_char(line: &LineResult, index: usize) -> LineResult {
    with_gap_char_at(line, index, 0.0, None)
}

/// One placeholder [`materialise_gap_plan`] will insert, carrying everything a
/// host needs so it can grow its own parallel lists without the line coming
/// back over the boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct GapInsertion {
    /// Character index in the **original** [`LineResult::text`] the
    /// placeholder belongs at, as [`Gap::insert_at`] reports it.
    pub index: usize,
    /// The CTC timestep column the placeholder takes ([`column_for`]).
    pub column: f32,
    /// The interpolated placeholder box ([`interpolate_gap_box`]), read only
    /// when [`GapPlan::grows_char_boxes`].
    pub placeholder_box: BoundingBox,
    /// This pair's spacing over the line's median, and the spacing in pixels —
    /// the [`Gap`] diagnostics, carried so a plan is self-describing. Neither
    /// affects the insertion.
    pub ratio: f32,
    pub span_px: f32,
    /// The synthetic alternatives entry, or `None` for the placeholder
    /// convention. Read only when [`GapPlan::grows_alternatives`].
    pub gap_alternatives: Option<Vec<(char, f32)>>,
}

/// The measured gaps on one line, plus which of its parallel per-character
/// lists a materialisation grows.
///
/// The flags are the "was this list full-length and non-empty?" guards
/// [`with_gap_char_at`] applied per insertion, hoisted so a host can make the
/// same decision from one crossing instead of recomputing lengths in its own
/// character units (Kotlin counts UTF-16 units, Rust code points, and they
/// differ on the supplementary-plane vocabulary entries).
#[derive(Debug, Clone, PartialEq)]
pub struct GapPlan {
    /// Ascending by `index`; materialise right-to-left.
    pub insertions: Vec<GapInsertion>,
    pub grows_char_boxes: bool,
    pub grows_alternatives: bool,
    pub grows_char_cols: bool,
}

impl GapPlan {
    /// The plan of a line that gets no placeholder.
    pub fn none() -> Self {
        Self {
            insertions: Vec::new(),
            grows_char_boxes: false,
            grows_alternatives: false,
            grows_char_cols: false,
        }
    }
}

/// The one-insertion plan behind [`with_gap_char_at`]: the same three
/// full-length guards, evaluated against the line as it stands.
fn single_gap_plan(line: &LineResult, n: usize, insertion: GapInsertion) -> GapPlan {
    GapPlan {
        insertions: vec![insertion],
        grows_char_boxes: line.char_boxes.len() == n && !line.char_boxes.is_empty(),
        grows_alternatives: line.alternatives.len() == n && !line.alternatives.is_empty(),
        grows_char_cols: !line.char_cols.is_empty() && line.char_cols.len() == n,
    }
}

/// Mobile `BlankGaps.apply` split in two: [`blank_gap_plan`] decides *where*
/// the placeholders go and [`materialise_gap_plan`] performs the insertion.
///
/// The plan is a handful of integers per line, where a full `LineResult` round
/// trip is every cell of every per-character list: the alternatives and the raw
/// per-timestep lists come back only to be cloned and grown by one entry, and
/// the geometry the detector read is already the answer. A host that keeps its
/// own lists can therefore ask for the plan and insert into them directly.
pub fn blank_gap_plan(line: &LineResult) -> GapPlan {
    blank_gap_plan_against(line, line.alternatives.len())
}

/// [`blank_gap_plan`] for a host that keeps its own lists, so only their
/// **lengths** cross.
///
/// `alternatives_len` is the only thing read about the per-character
/// alternatives, and it is read solely to decide whether inserting a
/// placeholder would keep that list index-aligned — never as content.
pub fn blank_gap_plan_against(line: &LineResult, alternatives_len: usize) -> GapPlan {
    if !line.is_vertical || line.text.contains(GAP_CHAR) {
        return GapPlan::none();
    }
    let gaps = GapDetector::default().detect(line);
    if gaps.is_empty() {
        return GapPlan::none();
    }
    let n = line.text.chars().count();
    let insertions = gaps
        .iter()
        .map(|g| GapInsertion {
            index: g.insert_at,
            // Every gap's column is the midpoint of two *original* columns, so
            // computing them all against the line as it stands gives the same
            // values the right-to-left materialisation computes against the
            // progressively grown one.
            column: column_for(line, g.insert_at),
            placeholder_box: interpolate_gap_box(&line.char_boxes, g.insert_at, line.is_vertical),
            ratio: g.ratio,
            span_px: g.span_px,
            gap_alternatives: None,
        })
        .collect();
    GapPlan {
        insertions,
        grows_char_boxes: line.char_boxes.len() == n && !line.char_boxes.is_empty(),
        grows_alternatives: alternatives_len == n && alternatives_len > 0,
        grows_char_cols: !line.char_cols.is_empty() && line.char_cols.len() == n,
    }
}

/// Apply a [`GapPlan`] to `line`, growing every list the plan says grows.
///
/// The one implementation of the insertion, shared by [`apply_blank_gaps`] and
/// [`with_gap_char_at`], so the two cannot drift: right-to-left, so the plan's
/// indices — computed against the original text — stay valid as the text grows.
pub fn materialise_gap_plan(line: &LineResult, plan: &GapPlan) -> LineResult {
    if plan.insertions.is_empty() {
        return line.clone();
    }
    let mut out = line.clone();
    for insertion in plan.insertions.iter().rev() {
        // A plan built by `blank_gap_plan` can never be out of range; a
        // hand-built one is clamped rather than panicking at the boundary.
        let index = insertion.index.min(out.text.chars().count());
        let chars: Vec<char> = out.text.chars().collect();
        let mut text: String = chars[..index].iter().collect();
        text.push(GAP_CHAR);
        text.extend(chars[index..].iter());
        out.text = text;

        if plan.grows_char_boxes {
            let mut boxes = out.char_boxes.clone();
            boxes.insert(index, insertion.placeholder_box.clone());
            out.char_boxes = boxes;
        }
        if plan.grows_alternatives {
            let mut alts = out.alternatives.clone();
            alts.insert(
                index,
                insertion
                    .gap_alternatives
                    .clone()
                    .unwrap_or_else(|| vec![(GAP_CHAR, 0.0)]),
            );
            out.alternatives = alts;
        }
        if plan.grows_char_cols {
            let mut cols = out.char_cols.clone();
            cols.insert(index, insertion.column);
            out.char_cols = cols;
        }
        let mut overrides: BTreeMap<i32, (char, f32)> = BTreeMap::new();
        for (key, value) in &out.overrides {
            let key = if *key >= index as i32 { *key + 1 } else { *key };
            overrides.insert(key, *value);
        }
        out.overrides = overrides;
    }
    out
}

/// Mobile `BlankGaps.apply`: materialise every measured gap as a placeholder.
/// Vertical lines only, idempotent, insertions right-to-left so the
/// detector's indices (computed against the original text) stay valid.
pub fn apply_blank_gaps(line: &LineResult) -> LineResult {
    materialise_gap_plan(line, &blank_gap_plan(line))
}

/// The insertion indices of [`apply_blank_gaps`] without materialising them.
/// Vertical lines only, matching the policy the desktop pipeline applies.
pub fn blank_gap_positions(line: &LineResult) -> Vec<usize> {
    if !line.is_vertical {
        return Vec::new();
    }
    GapDetector::default()
        .detect(line)
        .into_iter()
        .map(|g| g.insert_at)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── fixtures (mirroring the Android test helpers) ───────────────────────

    const TEXT5: &str = "あいうえお";

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    /// Android `GapDetectorTest.verticalBoxes`: 40px wide, 20px tall boxes
    /// centred on the given y coordinates.
    fn vertical_boxes(centres: &[i32]) -> Vec<BoundingBox> {
        vertical_boxes_h(centres, 20)
    }

    fn vertical_boxes_h(centres: &[i32], height: i32) -> Vec<BoundingBox> {
        centres
            .iter()
            .map(|&c| BoundingBox::new(0, c - height / 2, 40, height, 1.0))
            .collect()
    }

    /// Android `GapDetectorTest.horizontalBoxes`: 20px wide, 40px tall boxes
    /// centred on the given x coordinates.
    fn horizontal_boxes(centres: &[i32]) -> Vec<BoundingBox> {
        horizontal_boxes_wh(centres, 20, 40)
    }

    fn horizontal_boxes_wh(centres: &[i32], width: i32, height: i32) -> Vec<BoundingBox> {
        centres
            .iter()
            .map(|&c| BoundingBox::new(c - width / 2, 0, width, height, 1.0))
            .collect()
    }

    /// Android `GapDetectorTest.line` / `BlankGapsTest.line`: alternatives are
    /// one `(char, 1.0)` entry per character.
    fn make_line(text: &str, boxes: Vec<BoundingBox>, is_vertical: bool) -> LineResult {
        LineResult {
            text: text.to_string(),
            char_boxes: boxes,
            alternatives: text.chars().map(|c| vec![(c, 1.0)]).collect(),
            is_vertical,
            ..Default::default()
        }
    }

    /// [`make_line`] plus a raw per-timestep cache and crop geometry
    /// (Android `GapDetectorTest.line`'s `raw` / `cropW` / `cropH` /
    /// `seqLenTotal` parameters).
    fn raw_line(
        text: &str,
        raw: Vec<Vec<(char, f32)>>,
        is_vertical: bool,
        crop_w: i32,
        crop_h: i32,
        seq_len_total: i32,
    ) -> LineResult {
        LineResult {
            text: text.to_string(),
            alternatives: text.chars().map(|c| vec![(c, 1.0)]).collect(),
            raw_alternatives: raw,
            is_vertical,
            crop_w,
            crop_h,
            seq_len_total,
            ..Default::default()
        }
    }

    fn geom(b: &BoundingBox) -> (i32, i32, i32, i32) {
        (b.x, b.y, b.w, b.h)
    }

    /// Field-for-field line equality: `models::LineResult` is deliberately not
    /// `PartialEq` (it carries a `PathBuf` and a `ChunkBox` list), so the plan
    /// parity rows compare the fields the gap pipeline can move.
    fn same_line(a: &LineResult, b: &LineResult) -> bool {
        a.text == b.text
            && a.is_vertical == b.is_vertical
            && a.char_cols == b.char_cols
            && a.crop_w == b.crop_w
            && a.crop_h == b.crop_h
            && a.crop_x == b.crop_x
            && a.crop_y == b.crop_y
            && a.seq_len_total == b.seq_len_total
            && a.raw_alternatives == b.raw_alternatives
            && a.overrides == b.overrides
            && a.alternatives == b.alternatives
            && a.char_boxes.len() == b.char_boxes.len()
            && a.char_boxes.iter().zip(&b.char_boxes).all(|(x, y)| geom(x) == geom(y))
            && a.char_boxes.iter().zip(&b.char_boxes).all(|(x, y)| x.confidence == y.confidence)
    }

    /// One timestep's top-N list; entry 0 is the head's emitted character.
    fn step(ch: char, score: f32) -> Vec<(char, f32)> {
        vec![(ch, score)]
    }

    /// A blank timestep, stored as the `'\u3000'` entry.
    fn blank_step() -> Vec<(char, f32)> {
        vec![(TIMESTEP_BLANK_CHAR, 0.99)]
    }

    /// 11 timesteps that decode to [`TEXT5`] with a deleted character between
    /// the third and fourth emitted characters: emitted columns 0, 2, 4, 8, 10.
    fn raw_with_deletion() -> Vec<Vec<(char, f32)>> {
        vec![
            step('あ', 1.0), // t0
            blank_step(),    // t1
            step('い', 1.0), // t2
            blank_step(),    // t3
            step('う', 1.0), // t4
            blank_step(),    // t5 ┐ three blanks where a character was dropped
            blank_step(),    // t6 │
            blank_step(),    // t7 ┘
            step('え', 1.0), // t8
            blank_step(),    // t9
            step('お', 1.0), // t10
        ]
    }

    // ── GapDetectorTest (spacing-ratio detection) ───────────────────────────

    /// Android `double_spacing_in_the_middle_reports_exactly_one_gap_with_ratio_two`:
    /// spacings 20, 20, 40, 20 → median 20 → the third pair is 2.00×.
    #[test]
    fn double_spacing_in_the_middle_reports_exactly_one_gap_with_ratio_two() {
        let gaps = GapDetector::default().detect(&make_line(
            TEXT5,
            vertical_boxes(&[10, 30, 50, 90, 110]),
            true,
        ));

        assert_eq!(gaps.len(), 1);
        let gap = gaps[0];
        assert_eq!(gap.insert_at, 3); // between chars 2 and 3
        assert!(approx(gap.ratio, 2.0));
        assert!(approx(gap.span_px, 40.0));
        assert_eq!(gap.pitch_ratio(), gap.ratio); // plan-document spelling
    }

    /// Android `uniform_vertical_line_reports_none`.
    #[test]
    fn uniform_vertical_line_reports_none() {
        let gaps = GapDetector::default().detect(&make_line(
            TEXT5,
            vertical_boxes(&[10, 30, 50, 70, 90]),
            true,
        ));

        assert!(gaps.is_empty());
    }

    /// Android `uniform_horizontal_line_reports_none`.
    #[test]
    fn uniform_horizontal_line_reports_none() {
        let gaps = GapDetector::default().detect(&make_line(
            TEXT5,
            horizontal_boxes(&[10, 30, 50, 70, 90]),
            false,
        ));

        assert!(gaps.is_empty());
    }

    /// Android `a_1_7_ratio_fires_vertically_and_not_horizontally`: spacings
    /// 20, 20, 34, 20 → 1.70×. Vertical fires at 1.6; horizontal does not
    /// reach its own 1.8.
    #[test]
    fn a_1_7_ratio_fires_vertically_and_not_horizontally() {
        let centres = [10, 30, 50, 84, 104];
        let detector = GapDetector::default();

        let vertical = detector.detect(&make_line(TEXT5, vertical_boxes(&centres), true));
        assert_eq!(vertical.len(), 1);
        assert!(approx(vertical[0].ratio, 1.7));
        assert_eq!(vertical[0].insert_at, 3);

        let horizontal = detector.detect(&make_line(TEXT5, horizontal_boxes(&centres), false));
        assert!(horizontal.is_empty(), "1.70 < the 1.8 horizontal threshold");
    }

    /// Android `a_1_9_ratio_fires_horizontally_too`: spacings 20, 20, 38, 20.
    #[test]
    fn a_1_9_ratio_fires_horizontally_too() {
        let gaps = GapDetector::default().detect(&make_line(
            TEXT5,
            horizontal_boxes(&[10, 30, 50, 88, 108]),
            false,
        ));

        assert_eq!(gaps.len(), 1);
        assert!(approx(gaps[0].ratio, 1.9));
    }

    /// Android `the_vertical_threshold_is_inclusive_at_1_6`: spacings
    /// 20, 20, 32, 20 → 1.60× exactly. The measured curve is quoted as
    /// "ratio ≥ 1.6".
    #[test]
    fn the_vertical_threshold_is_inclusive_at_1_6() {
        let gaps = GapDetector::default().detect(&make_line(
            TEXT5,
            vertical_boxes(&[10, 30, 50, 82, 102]),
            true,
        ));

        assert_eq!(gaps.len(), 1);
        assert!(approx(gaps[0].ratio, 1.6));
    }

    /// Android `thresholds_are_configurable_per_orientation`: never hardcoded,
    /// and a caller may sweep the curve without rebuilding the detector.
    #[test]
    fn thresholds_are_configurable_per_orientation() {
        let detector = GapDetector::default();
        assert!(approx(detector.threshold_for(true), 1.6));
        assert!(approx(detector.threshold_for(false), 1.8));

        let ratio_two = make_line(TEXT5, vertical_boxes(&[10, 30, 50, 90, 110]), true);
        assert!(GapDetector::new(2.5, 1.8).detect(&ratio_two).is_empty());
        assert_eq!(GapDetector::new(1.2, 1.8).detect(&ratio_two).len(), 1);

        assert!(detector.detect_with(&ratio_two, 2.5).is_empty());
        assert_eq!(detector.detect_with(&ratio_two, 2.0).len(), 1);
    }

    /// The measured constants the spec page quotes.
    #[test]
    fn the_detector_defaults_are_the_measured_constants() {
        assert_eq!(BLANK_GAP_RATIO, 1.6);
        assert_eq!(DEFAULT_VERTICAL_RATIO, 1.6);
        assert_eq!(DEFAULT_HORIZONTAL_RATIO, 1.8);
        assert_eq!(DEFAULT_TIMESTEP_STRIDE_PX, 8.0);
        assert_eq!(MIN_EMITTED_CHARS, 2);
    }

    /// Android `empty_char_boxes_falls_back_to_raw_alternatives`: the walk
    /// derives the emitted columns and the span uses the model stride when no
    /// crop length was cached.
    #[test]
    fn empty_char_boxes_falls_back_to_raw_alternatives() {
        let gaps = GapDetector::default().detect(&raw_line(TEXT5, raw_with_deletion(), true, 0, 0, 0));

        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].insert_at, 3);
        assert!(approx(gaps[0].ratio, 2.0));
        // 4 timesteps × the model stride (8 px), since no crop length was cached
        assert!(approx(gaps[0].span_px, 32.0));
    }

    /// Android `fallback_span_uses_the_crop_length_when_it_is_known`.
    #[test]
    fn fallback_span_uses_the_crop_length_when_it_is_known() {
        let detector = GapDetector::default();

        // crop_h / seq_len_total = 5 px per timestep, spacing = 4 timesteps
        let vertical = detector.detect(&raw_line(TEXT5, raw_with_deletion(), true, 0, 55, 11));
        assert!(approx(vertical[0].span_px, 20.0));

        // crop_w / seq_len_total = 10 px per timestep
        let horizontal = detector.detect(&raw_line(TEXT5, raw_with_deletion(), false, 110, 0, 11));
        assert!(approx(horizontal[0].span_px, 40.0));
    }

    /// Android `char_cols_geometry_is_used_when_boxes_are_missing`: spacings
    /// are in timesteps, and the ratio is scale-invariant.
    #[test]
    fn char_cols_geometry_is_used_when_boxes_are_missing() {
        let mut line = make_line(TEXT5, vec![], true);
        line.char_cols = vec![0.0, 2.0, 4.0, 8.0, 10.0];

        let gaps = GapDetector::default().detect(&line);

        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].insert_at, 3);
        assert!(approx(gaps[0].ratio, 2.0));
    }

    /// Android `text_that_does_not_match_the_timestep_walk_detects_nothing`:
    /// 5 emitted columns derived, but the line claims 2 characters.
    #[test]
    fn text_that_does_not_match_the_timestep_walk_detects_nothing() {
        let gaps = GapDetector::default().detect(&raw_line("あい", raw_with_deletion(), true, 0, 0, 0));

        assert!(gaps.is_empty());
    }

    /// The desktop counterpart: fewer char boxes than characters cannot be
    /// walked, and boxes beyond the text are ignored rather than mismatched.
    #[test]
    fn text_and_geometry_mismatch_detects_nothing() {
        let detector = GapDetector::default();

        let too_few = make_line(TEXT5, vertical_boxes(&[10, 30]), true);
        assert!(detector.detect(&too_few).is_empty());

        let extra = make_line("あい", vertical_boxes(&[10, 30, 50, 90, 110]), true);
        assert!(detector.detect(&extra).is_empty());
    }

    /// Android `degenerate_geometry_detects_nothing`: every character on the
    /// same pixel (median spacing 0) and a single character both report none.
    #[test]
    fn degenerate_geometry_detects_nothing() {
        let detector = GapDetector::default();

        let same_pixel = make_line(TEXT5, vertical_boxes(&[50, 50, 50, 50, 50]), true);
        assert!(detector.detect(&same_pixel).is_empty());

        let one_char = LineResult {
            text: "あ".to_string(),
            ..Default::default()
        };
        assert!(detector.detect(&one_char).is_empty());
    }

    /// Android `three_char_line_median_is_contaminated_by_the_gap`: spacings
    /// 20, 40 → median 30 → the double gap reads as 1.33×, below both
    /// thresholds. The pitch estimator needs a few ordinary pairs; documented,
    /// not a bug to "fix" by lowering the threshold.
    #[test]
    fn three_char_line_median_is_contaminated_by_the_gap() {
        let line = make_line("あいう", vertical_boxes(&[10, 30, 70]), true);

        assert!(GapDetector::default().detect(&line).is_empty());
    }

    /// Android `timestep_walk_matches_the_documented_assumptions`: entry 0 is
    /// the argmax, `'\u3000'` is the blank entry, blank resets the repeat
    /// state, spaces never collapse.
    #[test]
    fn timestep_walk_matches_the_documented_assumptions() {
        let raw = vec![
            step('あ', 0.90), // t0 emitted
            step('あ', 0.80), // t1 CTC repeat → collapsed
            blank_step(),     // t2 blank → resets the previous char
            step('あ', 0.70), // t3 emitted again after the blank
            step(' ', 0.60),  // t4 space
            step(' ', 0.50),  // t5 space again — spaces never collapse
            step('い', 0.90), // t6
        ];

        assert_eq!(timestep_columns(&raw), vec![0.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// Android `blank_marker_and_placeholder_are_different_characters`.
    #[test]
    fn blank_marker_and_placeholder_are_different_characters() {
        assert_eq!(TIMESTEP_BLANK_CHAR, '\u{3000}'); // blank, as raw_alternatives stores it
        assert_eq!(GAP_CHAR, '\u{25CC}'); // the reversible placeholder
        assert_ne!(TIMESTEP_BLANK_CHAR, GAP_CHAR);
    }

    /// Android `median_helper_handles_odd_even_and_empty_input`.
    #[test]
    fn median_helper_handles_odd_even_and_empty_input() {
        assert_eq!(median_of(&mut []), 0.0);
        assert_eq!(median_of(&mut [5.0]), 5.0);
        assert_eq!(median_of(&mut [2.0, 1.0]), 1.5);
        assert_eq!(median_of(&mut [3.0, 1.0, 2.0]), 2.0);
    }

    // ── BlankGapsTest (the materialisation policy) ──────────────────────────

    /// The fixture geometry the detector measured: y-centres
    /// 10/30/50/90/110 → spacings 20/20/40/20 → one gap at index 3.
    fn gapped_centres() -> [i32; 5] {
        [10, 30, 50, 90, 110]
    }

    /// Android `a_measured_vertical_gap_becomes_a_placeholder_between_its_neighbours`:
    /// the placeholder lands between う and え, every character keeps its
    /// identity at its shifted index, and the synthetic alternatives entry is
    /// the placeholder itself.
    #[test]
    fn a_measured_vertical_gap_becomes_a_placeholder_between_its_neighbours() {
        let out = apply_blank_gaps(&make_line(TEXT5, vertical_boxes(&gapped_centres()), true));

        assert_eq!(out.text, format!("あいう{GAP_CHAR}えお"));
        assert_eq!(out.char_boxes.len(), TEXT5.chars().count() + 1);
        assert_eq!(out.alternatives[3], vec![(GAP_CHAR, 0.0)]);
        assert_eq!(out.alternatives[4], vec![('え', 1.0)]);
    }

    /// Android `horizontal_lines_are_never_touched`: the same geometry that
    /// fires on a vertical line leaves a horizontal one alone. (Mobile checks
    /// reference identity; the PC always clones, so equality is the observable
    /// contract.)
    #[test]
    fn horizontal_lines_are_never_touched() {
        let horizontal = make_line(TEXT5, horizontal_boxes(&gapped_centres()), false);

        let out = apply_blank_gaps(&horizontal);
        assert_eq!(out.text, horizontal.text);
        assert_eq!(out.char_boxes.len(), horizontal.char_boxes.len());
    }

    /// Android `a_line_that_already_has_a_placeholder_is_unchanged`: applied
    /// twice must not insert twice — a second placeholder would shift the
    /// user's filled override.
    #[test]
    fn a_line_that_already_has_a_placeholder_is_unchanged() {
        let once = apply_blank_gaps(&make_line(TEXT5, vertical_boxes(&gapped_centres()), true));
        let twice = apply_blank_gaps(&once);

        assert_eq!(twice.text, once.text);
        assert_eq!(twice.char_boxes.len(), once.char_boxes.len());
    }

    /// Android `an_evenly_spaced_line_is_unchanged`.
    #[test]
    fn an_evenly_spaced_line_is_unchanged() {
        let even = make_line(TEXT5, vertical_boxes(&[10, 30, 50, 70, 90]), true);

        let out = apply_blank_gaps(&even);
        assert_eq!(out.text, even.text);
        assert_eq!(out.char_boxes.len(), even.char_boxes.len());
    }

    /// Android `the_placeholder_column_sits_between_its_neighbours`: charCols
    /// drives box recomputation (#49); the new column is the midpoint,
    /// matching the geometry the placeholder box itself is interpolated from.
    #[test]
    fn the_placeholder_column_sits_between_its_neighbours() {
        let mut plain = make_line(TEXT5, vertical_boxes(&gapped_centres()), true);
        plain.char_cols = vec![0.0, 2.0, 4.0, 8.0, 10.0];

        let out = apply_blank_gaps(&plain);

        assert_eq!(out.char_cols.len(), 6);
        assert!(approx(out.char_cols[3], 6.0));
        assert!(approx(out.char_cols[5], 10.0));
    }

    /// Android `a_filled_blank_is_still_an_override_and_survives_the_shift`:
    /// an override already on a later character must move with it, or a fill
    /// would land on the wrong character.
    #[test]
    fn a_filled_blank_is_still_an_override_and_survives_the_shift() {
        let mut with_override = make_line(TEXT5, vertical_boxes(&gapped_centres()), true);
        with_override.overrides.insert(4, ('ぇ', 1.0));

        let out = apply_blank_gaps(&with_override);

        assert_eq!(out.overrides.get(&5).map(|v| v.0), Some('ぇ'));
        assert_eq!(out.overrides.get(&3), None);
    }

    /// Right-to-left insertion (`BlankGaps.apply`): with two detected gaps the
    /// placeholders go in from the end, so the detector's indices — computed
    /// against the original text — stay valid. Left-to-right insertion would
    /// put the second placeholder between え and お instead of お and か.
    #[test]
    fn multiple_gaps_insert_right_to_left_and_stay_index_aligned() {
        // 7 chars, spacings 20, 20, 40, 20, 40, 20 → median 20 → gaps at 3 and 5.
        let out = apply_blank_gaps(&make_line(
            "あいうえおかき",
            vertical_boxes(&[10, 30, 50, 90, 110, 150, 170]),
            true,
        ));

        assert_eq!(out.text, format!("あいう{GAP_CHAR}えお{GAP_CHAR}かき"));
        assert_eq!(out.char_boxes.len(), 9);
        assert_eq!(out.alternatives.len(), 9);
        assert_eq!(out.alternatives[3], vec![(GAP_CHAR, 0.0)]);
        assert_eq!(out.alternatives[4], vec![('え', 1.0)]);
        assert_eq!(out.alternatives[5], vec![('お', 1.0)]);
        assert_eq!(out.alternatives[6], vec![(GAP_CHAR, 0.0)]);
        // The interpolated boxes sit at the two gap centres: 70 and 130.
        assert_eq!(out.char_boxes[3].y + out.char_boxes[3].h / 2, 70);
        assert_eq!(out.char_boxes[6].y + out.char_boxes[6].h / 2, 130);
    }

    // ── LineResultGapTest (synthetic insertion) ─────────────────────────────

    /// Android `LineResultGapTest.fixture`: vertical boxes at y-centres
    /// 10, 30, 50, 70, 90, overrides on the first and last character, a full
    /// charCols row, and crop geometry attached so the carry-over assertions
    /// have something to carry.
    fn fixture() -> LineResult {
        let mut line = make_line(TEXT5, vertical_boxes(&[10, 30, 50, 70, 90]), true);
        line.sample_txt = Some(std::path::PathBuf::from("/tmp/gap-fixture.txt"));
        line.raw_alternatives = vec![vec![('あ', 0.9)], vec![('い', 0.8)]];
        line.overrides.insert(0, ('Z', 1.0));
        line.overrides.insert(4, ('X', 1.0));
        line.char_cols = (0..5).map(|i| i as f32).collect();
        line.crop_w = 40;
        line.crop_h = 100;
        line
    }

    /// Android `inserts_the_placeholder_at_the_requested_index`: the
    /// placeholder lands at the index and everything else about the line is
    /// carried over.
    #[test]
    fn inserts_the_placeholder_at_the_requested_index() {
        let line = fixture();

        let out = with_gap_char_at(&line, 3, 3.5, None);

        assert_eq!(out.text.chars().count(), 6);
        assert_eq!(out.text.chars().nth(3), Some(GAP_CHAR));
        assert_eq!(out.text, format!("あいう{GAP_CHAR}えお"));
        assert_eq!(out.is_vertical, line.is_vertical);
        assert_eq!(out.crop_w, line.crop_w);
        assert_eq!(out.crop_h, line.crop_h);
        assert_eq!(out.sample_txt, line.sample_txt);
        assert_eq!(out.raw_alternatives, line.raw_alternatives);
    }

    /// Android `all_per_character_lists_stay_the_same_length`: text, char
    /// boxes, alternatives and charCols all grow together, and the receiver is
    /// not mutated.
    #[test]
    fn all_per_character_lists_stay_the_same_length() {
        let out = with_gap_char_at(&fixture(), 3, 3.0, None);

        assert_eq!(out.text.chars().count(), 6);
        assert_eq!(out.char_boxes.len(), out.text.chars().count());
        assert_eq!(out.alternatives.len(), out.text.chars().count());
        assert_eq!(out.char_cols.len(), out.text.chars().count());

        let line = fixture();
        let _ = with_gap_char_at(&line, 3, 3.0, None);
        assert_eq!(line.text.chars().count(), 5);
        assert_eq!(line.char_boxes.len(), 5);
        assert_eq!(line.alternatives.len(), 5);
        // not re-keyed in place
        assert_eq!(line.overrides.keys().copied().collect::<Vec<_>>(), vec![0, 4]);
    }

    /// Android `a_character_at_index_n_moves_to_n_plus_one`.
    #[test]
    fn a_character_at_index_n_moves_to_n_plus_one() {
        let line = fixture();

        let out = with_gap_char_at(&line, 3, 3.0, None);

        for i in 0..3 {
            assert_eq!(out.text.chars().nth(i), line.text.chars().nth(i));
            assert_eq!(geom(&out.char_boxes[i]), geom(&line.char_boxes[i]));
            assert_eq!(out.alternatives[i], line.alternatives[i]);
            assert_eq!(out.char_cols[i], line.char_cols[i]);
        }
        for i in 3..5 {
            assert_eq!(out.text.chars().nth(i + 1), line.text.chars().nth(i));
            assert_eq!(geom(&out.char_boxes[i + 1]), geom(&line.char_boxes[i]));
            assert_eq!(out.alternatives[i + 1], line.alternatives[i]);
            assert_eq!(out.char_cols[i + 1], line.char_cols[i]);
        }
    }

    /// Android `overrides_follow_the_characters_they_were_applied_to`.
    #[test]
    fn overrides_follow_the_characters_they_were_applied_to() {
        let out = with_gap_char_at(&fixture(), 3, 3.0, None);

        assert_eq!(out.overrides.len(), 2);
        assert_eq!(out.overrides.get(&0), Some(&('Z', 1.0))); // before: unchanged
        assert_eq!(out.overrides.get(&5), Some(&('X', 1.0))); // was index 4, now shifted
        assert!(!out.overrides.contains_key(&4));
        assert!(!out.overrides.contains_key(&3)); // the placeholder carries no override
    }

    /// Android `every_override_shifts_when_inserting_at_the_start`.
    #[test]
    fn every_override_shifts_when_inserting_at_the_start() {
        let out = with_gap_char_at(&fixture(), 0, 0.0, None);

        assert_eq!(out.overrides.get(&1), Some(&('Z', 1.0)));
        assert_eq!(out.overrides.get(&5), Some(&('X', 1.0)));
        assert_eq!(out.overrides.len(), 2);
    }

    /// Android `the_gap_box_sits_between_its_neighbours`: the placeholder box
    /// is centred at the midpoint of its neighbours' centres and sized as the
    /// mean of their extents along the reading axis.
    #[test]
    fn the_gap_box_sits_between_its_neighbours() {
        let line = fixture(); // vertical centres 10, 30, 50, 70, 90

        let out = with_gap_char_at(&line, 3, 3.0, None);

        let before = &out.char_boxes[2];
        let inserted = &out.char_boxes[3];
        let after = &out.char_boxes[4];
        assert_eq!(before.y + before.h / 2, 50);
        assert_eq!(inserted.y + inserted.h / 2, 60); // midpoint of 50 and 70
        assert_eq!(after.y + after.h / 2, 70);
        assert_eq!(inserted.h, 20); // mean of the neighbours' heights
        assert_eq!(inserted.x, 0);
        assert_eq!(inserted.x + inserted.w, 40);
        // The neighbours keep their own boxes.
        assert_eq!(geom(before), geom(&line.char_boxes[2]));
        assert_eq!(geom(after), geom(&line.char_boxes[3]));
    }

    /// Android `the_gap_box_interpolates_on_the_x_axis_for_horizontal_lines`.
    #[test]
    fn the_gap_box_interpolates_on_the_x_axis_for_horizontal_lines() {
        let line = make_line(TEXT5, horizontal_boxes(&[10, 30, 50, 70, 90]), false);

        let out = with_gap_char_at(&line, 1, 1.0, None);

        let inserted = &out.char_boxes[1];
        assert_eq!(inserted.x + inserted.w / 2, 20); // midpoint of 10 and 30
        assert_eq!(inserted.w, 20);
        assert_eq!(inserted.y, 0);
        assert_eq!(inserted.y + inserted.h, 40);
    }

    /// Android `inserting_at_the_ends_reuses_the_single_neighbour`.
    #[test]
    fn inserting_at_the_ends_reuses_the_single_neighbour() {
        let line = fixture();

        let at_start = with_gap_char_at(&line, 0, 0.0, None);
        assert_eq!(at_start.text.chars().next(), Some(GAP_CHAR));
        assert_eq!(geom(&at_start.char_boxes[0]), geom(&line.char_boxes[0]));
        assert_eq!(at_start.char_boxes.len(), at_start.text.chars().count());

        let at_end = with_gap_char_at(&line, line.text.chars().count(), 5.0, None);
        assert_eq!(at_end.text.chars().nth(5), Some(GAP_CHAR));
        // Single neighbour reused.
        assert_eq!(geom(&at_end.char_boxes[5]), geom(&line.char_boxes[4]));
        assert_eq!(at_end.char_boxes.len(), at_end.text.chars().count());
    }

    /// Android `the_synthetic_alternatives_entry_defaults_to_the_reversible_blank`.
    #[test]
    fn the_synthetic_alternatives_entry_defaults_to_the_reversible_blank() {
        let out = with_gap_char_at(&fixture(), 3, 3.0, None);

        assert_eq!(out.alternatives[3].len(), 1);
        assert_eq!(out.alternatives[3][0], (GAP_CHAR, 0.0));
    }

    /// Android `a_caller_can_supply_its_own_gap_alternatives`.
    #[test]
    fn a_caller_can_supply_its_own_gap_alternatives() {
        let custom = vec![(GAP_CHAR, 0.5), ('あ', 0.4)];

        let out = with_gap_char_at(&fixture(), 2, 2.0, Some(custom.clone()));

        assert_eq!(out.alternatives[2], custom);
        assert_eq!(out.alternatives.len(), 6);
    }

    /// Android `the_inserted_placeholder_is_unlookupable`: the placeholder
    /// occupies index 3 and the neighbours keep their identity. (The actual
    /// lookup guard lives in the overlay controller.)
    #[test]
    fn the_inserted_placeholder_is_unlookupable() {
        let out = with_gap_char_at(&fixture(), 3, 3.0, None);

        assert_eq!(out.text.chars().nth(3), Some(GAP_CHAR));
        assert_eq!(out.text.chars().nth(2), Some('う'));
        assert_eq!(out.text.chars().nth(4), Some('え'));
    }

    /// Android `an_index_out_of_range_is_a_no_op`: an index past the end
    /// leaves the line unchanged (mobile also guards negative indices; Rust's
    /// `usize` cannot express one). Inserting exactly at the end is valid.
    #[test]
    fn an_index_out_of_range_is_a_no_op() {
        let line = fixture();

        let out = with_gap_char_at(&line, 6, 0.0, None);
        assert_eq!(out.text, line.text);
        assert_eq!(out.char_boxes.len(), line.char_boxes.len());

        let at_end = with_gap_char_at(&line, 5, 5.0, None);
        assert_eq!(at_end.text.chars().nth(5), Some(GAP_CHAR));
    }

    /// Android `empty_box_and_column_lists_stay_empty`: no geometry to
    /// interpolate, so text and alternatives grow while the box and column
    /// lists stay empty.
    #[test]
    fn empty_box_and_column_lists_stay_empty() {
        let line = make_line(TEXT5, vec![], true);

        let out = with_gap_char_at(&line, 2, 2.0, None);

        assert_eq!(out.text.chars().count(), 6);
        assert_eq!(out.alternatives.len(), 6);
        assert!(out.char_boxes.is_empty());
        assert!(out.char_cols.is_empty());
    }

    /// Android `withGapAt_is_the_same_operation_as_withGapCharAt`.
    #[test]
    fn with_gap_at_is_the_same_operation_as_with_gap_char_at() {
        let line = fixture();

        let a = with_gap_char_at(&line, 2, 2.0, None);
        let b = with_gap_at(&line, 2, 2.0);

        assert_eq!(a.text, b.text);
        assert_eq!(a.char_boxes, b.char_boxes);
        assert_eq!(a.alternatives, b.alternatives);
        assert_eq!(a.overrides, b.overrides);
        assert_eq!(a.char_cols, b.char_cols);
    }

    /// Android `a_detected_gap_can_be_materialised_and_consumes_the_spacing`:
    /// detect → materialise → the placeholder takes up the spacing, so the
    /// line reads as uniform again, and a later override survives the shift.
    #[test]
    fn a_detected_gap_can_be_materialised_and_consumes_the_spacing() {
        let mut line = make_line(TEXT5, vertical_boxes(&gapped_centres()), true);
        line.overrides.insert(4, ('X', 1.0));
        line.char_cols = (0..5).map(|i| i as f32).collect();

        let detected = GapDetector::default().detect(&line);
        assert_eq!(detected.len(), 1);
        assert_eq!(detected[0].insert_at, 3);

        let out = with_gap_char_at(&line, detected[0].insert_at, 3.0, None);

        assert_eq!(out.text.chars().count(), 6);
        assert_eq!(out.char_boxes.len(), 6);
        assert_eq!(out.alternatives.len(), 6);
        assert_eq!(out.char_cols.len(), 6);
        assert_eq!(out.text.chars().nth(3), Some(GAP_CHAR));
        assert_eq!(out.overrides.get(&5), Some(&('X', 1.0)));
        // the placeholder's box takes up the space, so the line reads as uniform again
        assert!(GapDetector::default().detect(&out).is_empty());
    }

    /// The legacy `with_gap_char` entry point still materialises with the
    /// default alternatives and an unknown column.
    #[test]
    fn with_gap_char_stays_the_default_column_and_alternatives() {
        let out = with_gap_char(&fixture(), 3);

        assert_eq!(out.text, format!("あいう{GAP_CHAR}えお"));
        assert_eq!(out.alternatives[3], vec![(GAP_CHAR, 0.0)]);
        assert_eq!(out.char_cols[3], 0.0); // no column supplied
    }

    // ── the plan/materialise split (a boundary keeps its own lists) ─────────

    /// The plan carries exactly what the materialised line gained: the index,
    /// the column and the box, plus which lists grew. Applying the plan through
    /// [`materialise_gap_plan`] is the same line [`apply_blank_gaps`] builds,
    /// which is what lets a host insert into its own lists instead of crossing
    /// the line back and forth.
    #[test]
    fn the_plan_reproduces_the_materialised_line_exactly() {
        // Two gaps, every list full-length, an override on each side of them.
        let mut line = make_line(
            "あいうえおかき",
            vertical_boxes(&[10, 30, 50, 90, 110, 150, 170]),
            true,
        );
        line.char_cols = (0..7).map(|i| i as f32).collect();
        line.overrides.insert(0, ('Z', 1.0));
        line.overrides.insert(6, ('X', 1.0));
        line.crop_w = 40;
        line.crop_h = 200;
        line.seq_len_total = 25;

        let plan = blank_gap_plan(&line);
        assert_eq!(
            plan.insertions.iter().map(|i| i.index).collect::<Vec<_>>(),
            vec![3, 5],
            "ascending, so a host can insert right-to-left"
        );
        assert_eq!(plan.insertions[0].ratio, 2.0);
        assert!(approx(plan.insertions[0].span_px, 40.0));
        assert!(plan.grows_char_boxes && plan.grows_alternatives && plan.grows_char_cols);
        assert!(plan.insertions.iter().all(|i| i.gap_alternatives.is_none()));

        let applied = materialise_gap_plan(&line, &plan);
        assert!(same_line(&applied, &apply_blank_gaps(&line)));
        assert_eq!(applied.text, format!("あいう{GAP_CHAR}えお{GAP_CHAR}かき"));
        // The plan's own numbers are the line's numbers. `index` addresses the
        // *original* text, so the position a placeholder ends up at is that
        // index plus one per earlier insertion — which is why the insertions
        // run right-to-left.
        for (j, insertion) in plan.insertions.iter().enumerate() {
            let at = [3, 6][j];
            assert_eq!(insertion.index, [3, 5][j]);
            assert_eq!(applied.text.chars().nth(at), Some(GAP_CHAR));
            assert_eq!(applied.char_boxes[at], insertion.placeholder_box);
            assert_eq!(applied.char_cols[at], insertion.column);
            assert_eq!(applied.alternatives[at], vec![(GAP_CHAR, 0.0)]);
        }
        // Midpoint columns, right-to-left safe: gap 3 sits between cols 2 and 3,
        // gap 5 between cols 4 and 5 — both original columns, so computing them
        // all up front is the same as growing the line one insertion at a time.
        assert!(approx(plan.insertions[0].column, 2.5));
        assert!(approx(plan.insertions[1].column, 4.5));
        // Overrides follow their characters: the one before both gaps is
        // untouched, the one after both moved by two, and neither placeholder
        // carries one.
        assert_eq!(applied.overrides.get(&0).map(|v| v.0), Some('Z'));
        assert_eq!(applied.overrides.get(&8).map(|v| v.0), Some('X'));
        assert_eq!(applied.overrides.get(&7), None);
        assert_eq!(applied.overrides.len(), 2);
        // Untouched fields carry over.
        assert_eq!(applied.crop_h, 200);
        assert_eq!(applied.seq_len_total, 25);
    }

    /// An empty plan is the whole answer for every line the policy declines:
    /// horizontal, already carrying a placeholder, no measured gap, and
    /// unusable geometry.
    #[test]
    fn a_declined_line_has_an_empty_plan() {
        let mut horizontal = make_line(TEXT5, vertical_boxes(&gapped_centres()), false);
        assert!(blank_gap_plan(&horizontal).insertions.is_empty());
        horizontal.is_vertical = true;

        let once = apply_blank_gaps(&horizontal);
        assert!(!once.text.contains(GAP_CHAR) == false); // it did insert
        assert!(
            blank_gap_plan(&once).insertions.is_empty(),
            "idempotent: a line that already has a placeholder gets no plan"
        );

        let even = make_line(TEXT5, vertical_boxes(&[10, 30, 50, 70, 90]), true);
        assert!(blank_gap_plan(&even).insertions.is_empty());

        let mut degenerate = even.clone();
        degenerate.char_boxes = (0..5).map(|_| BoundingBox::new(0, 50, 40, 20, 1.0)).collect();
        assert!(blank_gap_plan(&degenerate).insertions.is_empty());
        assert!(same_line(&apply_blank_gaps(&degenerate), &degenerate));
    }

    /// The three growth flags are the "full-length and non-empty" guards,
    /// hoisted out of the materialiser so a host applies the same decision
    /// instead of recomputing lengths in its own character units.
    #[test]
    fn the_growth_flags_mirror_the_full_length_guards() {
        // Boxes only, no alternatives, no columns: only the boxes grow.
        let mut line = make_line(TEXT5, vertical_boxes(&gapped_centres()), true);
        line.alternatives = Vec::new();
        line.char_cols = Vec::new();
        let plan = blank_gap_plan(&line);
        assert!(plan.grows_char_boxes);
        assert!(!plan.grows_alternatives);
        assert!(!plan.grows_char_cols);
        let applied = materialise_gap_plan(&line, &plan);
        assert_eq!(applied.alternatives, line.alternatives);
        assert!(applied.char_cols.is_empty());
        assert_eq!(applied.char_boxes.len(), 6);
        // The host-supplied length is the only thing read about the
        // alternatives, and it is the host's own count: a host with no
        // alternatives list (or a short one) gets a plan that does not grow
        // it, from the same measured gaps.
        let full = make_line(TEXT5, vertical_boxes(&gapped_centres()), true);
        assert!(blank_gap_plan(&full).grows_alternatives);
        let claimed_short = blank_gap_plan_against(&full, 2);
        assert!(!claimed_short.grows_alternatives);
        assert_eq!(claimed_short.insertions, blank_gap_plan(&full).insertions);
        assert_eq!(claimed_short.grows_char_boxes, blank_gap_plan(&full).grows_char_boxes);
        assert_eq!(claimed_short.grows_char_cols, blank_gap_plan(&full).grows_char_cols);

        // The single-insertion entry point reads the same flags.
        let synth = with_gap_char_at(&line, 2, 7.0, None);
        assert_eq!(synth.char_boxes.len(), 6);
        assert!(synth.alternatives.is_empty());
        assert!(synth.char_cols.is_empty());
        assert_eq!(synth.char_cols.len(), 0);
        // and the custom-alternatives path still only shows up when the
        // alternatives list was full-length to begin with
        let with_alts = make_line(TEXT5, vertical_boxes(&[10, 30, 50, 70, 90]), true);
        let custom = with_gap_char_at(&with_alts, 2, 1.0, Some(vec![('X', 0.5)]));
        assert_eq!(custom.alternatives[2], vec![('X', 0.5)]);
    }

    /// The detector's source *precedence*, as a predicate a host can ask before
    /// handing over the raw per-timestep lists. Only the last-resort walk reads
    /// them, so every line whose char boxes (or columns) describe the text can
    /// skip building them.
    #[test]
    fn needs_raw_alternatives_is_the_detector_source_precedence() {
        let text = TEXT5; // 5 characters
        // Boxes cover the text: the walk is unreachable.
        assert!(!needs_raw_alternatives(text, 5, 0));
        assert!(!needs_raw_alternatives(text, 9, 5));
        // No boxes, columns cover the text.
        assert!(!needs_raw_alternatives(text, 0, 5));
        // Neither: the walk is the only source left.
        assert!(needs_raw_alternatives(text, 0, 0));
        assert!(needs_raw_alternatives(text, 3, 4));
        // An empty line needs nothing at all.
        assert!(!needs_raw_alternatives("", 0, 0));
        assert!(!needs_raw_alternatives("", 5, 5));

        // Cross-check against the detector itself: the predicate is exactly the
        // set of shapes whose gaps only the raw walk can find.
        let from_raw = raw_line(TEXT5, raw_with_deletion(), true, 0, 0, 0);
        assert!(needs_raw_alternatives(TEXT5, 0, 0));
        assert_eq!(GapDetector::default().detect(&from_raw).len(), 1);
        // The same geometry with the boxes filled in is read from the boxes, so
        // the predicate is false and no plan needs the raw lists.
        let mut from_boxes = from_raw.clone();
        from_boxes.char_boxes = vertical_boxes(&[0, 20, 40, 80, 100]);
        assert!(!needs_raw_alternatives(TEXT5, 5, 0));
        assert_eq!(
            GapDetector::default().detect(&from_boxes).len(),
            1,
            "and the box geometry finds the same gap"
        );
        // The plan is unchanged when the raw lists are withheld, because the
        // walk is not read.
        let mut without_raw = from_boxes.clone();
        without_raw.raw_alternatives = Vec::new();
        assert_eq!(blank_gap_plan(&without_raw), blank_gap_plan(&from_boxes));
    }
}
