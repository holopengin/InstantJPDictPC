//! Mobile `BlankGaps` / `GapDetector` (#44 Feature 2): the spacing-ratio
//! blank detector and the placeholder materialiser that follows it, pure so
//! they are host-tested and reachable from a nav-only build.
//!
//! Detection is vertical-only: a character index is a gap when the spacing to
//! the next character is at least [`BLANK_GAP_RATIO`]× the line's median
//! spacing. Mobile's detector also carries a horizontal threshold
//! (`DEFAULT_HORIZONTAL_RATIO = 1.8f`); the PC never asks for it
//! (`BlankGaps.apply` returns early on horizontal lines). Materialisation
//! inserts the [`GAP_CHAR`] placeholder right-to-left so the detector's
//! indices stay valid, growing text, char boxes and alternatives together,
//! and is idempotent.

use crate::models::{BoundingBox, LineResult, GAP_CHAR};

/// Mobile `GapDetector.DEFAULT_VERTICAL_RATIO`: measured recall 1.00 and no
/// false positives on the vertical bench; horizontal lines are never
/// eligible (`BlankGaps.apply` returns early on them).
pub const BLANK_GAP_RATIO: f32 = 1.6;

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

/// Mobile `GapDetector.detect` on the char-box geometry: character indices
/// where the spacing to the next character is at least [BLANK_GAP_RATIO]×
/// the median spacing of the line. Vertical lines only.
pub fn blank_gap_positions(line: &LineResult) -> Vec<usize> {
    if !line.is_vertical {
        return Vec::new();
    }
    let n = line.text.chars().count();
    if n < 2 || line.char_boxes.len() < n {
        return Vec::new();
    }
    let centres: Vec<f32> = line
        .char_boxes
        .iter()
        .take(n)
        .map(|b| b.y as f32 + b.h as f32 / 2.0)
        .collect();
    let spacings: Vec<f32> = centres
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .collect();
    let pitch = median_of(&mut spacings.clone());
    if pitch <= 0.0 {
        return Vec::new();
    }
    spacings
        .iter()
        .enumerate()
        .filter(|(_, s)| **s / pitch >= BLANK_GAP_RATIO)
        .map(|(k, _)| k + 1)
        .collect()
}

/// Mobile `interpolateGapBox`: a placeholder box centred between the two
/// neighbours it was dropped from, sized as the mean of their extents along
/// the reading axis.
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

/// Mobile `LineResult.withGapCharAt`: insert the placeholder at `index`,
/// growing text, char boxes and alternatives together so every parallel list
/// still describes the same characters at the same indices. PC lines carry
/// no CTC columns or overrides, so those mobile-list steps have no analogue.
pub fn with_gap_char(line: &LineResult, index: usize) -> LineResult {
    let chars: Vec<char> = line.text.chars().collect();
    if index > chars.len() {
        return line.clone();
    }
    let mut text: String = chars[..index].iter().collect();
    text.push(GAP_CHAR);
    text.extend(chars[index..].iter());

    let char_boxes = if line.char_boxes.len() == chars.len() && !line.char_boxes.is_empty() {
        let mut out = line.char_boxes.clone();
        out.insert(
            index,
            interpolate_gap_box(&line.char_boxes, index, line.is_vertical),
        );
        out
    } else {
        line.char_boxes.clone()
    };
    let alternatives = if line.alternatives.len() == chars.len() && !line.alternatives.is_empty() {
        let mut out = line.alternatives.clone();
        out.insert(index, vec![(GAP_CHAR, 0.0)]);
        out
    } else {
        line.alternatives.clone()
    };
    LineResult {
        text,
        char_boxes,
        alternatives,
        is_vertical: line.is_vertical,
        ..line.clone()
    }
}

/// Mobile `BlankGaps.apply`: materialise every measured gap as a placeholder.
/// Vertical lines only, idempotent, insertions right-to-left so the
/// detector's indices (computed against the original text) stay valid.
pub fn apply_blank_gaps(line: &LineResult) -> LineResult {
    if !line.is_vertical || line.text.contains(GAP_CHAR) {
        return line.clone();
    }
    let gaps = blank_gap_positions(line);
    if gaps.is_empty() {
        return line.clone();
    }
    let mut out = line.clone();
    for &index in gaps.iter().rev() {
        out = with_gap_char(&out, index);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── fixtures (mirroring the Android test helpers) ───────────────────────

    const TEXT5: &str = "あいうえお";

    /// Android `GapDetectorTest.verticalBoxes`: 40px wide, 20px tall boxes
    /// centred on the given y coordinates.
    fn vertical_boxes(centres: &[i32]) -> Vec<BoundingBox> {
        centres
            .iter()
            .map(|&c| BoundingBox::new(0, c - 10, 40, 20, 1.0))
            .collect()
    }

    /// Android `GapDetectorTest.horizontalBoxes`: 20px wide, 40px tall boxes
    /// centred on the given x coordinates.
    fn horizontal_boxes(centres: &[i32]) -> Vec<BoundingBox> {
        centres
            .iter()
            .map(|&c| BoundingBox::new(c - 10, 0, 20, 40, 1.0))
            .collect()
    }

    /// Android `GapDetectorTest.line` / `BlankGapsTest.line`: alternatives are
    /// one `(char, 1.0)` entry per character.
    fn make_line(text: &str, boxes: Vec<BoundingBox>, is_vertical: bool) -> LineResult {
        LineResult {
            text: text.to_string(),
            char_boxes: boxes,
            alternatives: text.chars().map(|c| vec![(c, 1.0)]).collect(),
            raw_alternatives: vec![],
            sample_txt: None,
            is_vertical,
            chunk_boxes: vec![],
            ..Default::default()
        }
    }

    fn geom(b: &BoundingBox) -> (i32, i32, i32, i32) {
        (b.x, b.y, b.w, b.h)
    }

    // ── GapDetectorTest (spacing-ratio detection) ───────────────────────────
    //
    // The Android tests also pin the mobile detector surface the PC does not
    // expose: per-orientation configurable thresholds, the `ratio` / `spanPx`
    // of a result, the raw-alternatives and char-cols fallbacks, and the
    // timestep walk. Those stay pinned by the Kotlin tests; the fixtures here
    // cover everything `blank_gap_positions` answers (the insertion indices),
    // plus the PC contract for the fallback cases (no detection).

    /// Android `double_spacing_in_the_middle_reports_exactly_one_gap_with_ratio_two`:
    /// spacings 20, 20, 40, 20 → median 20 → the third pair is 2.00×. (`ratio`
    /// and `spanPx` are mobile-detector surface; the index is what the PC
    /// exposes.)
    #[test]
    fn double_spacing_in_the_middle_reports_exactly_one_gap() {
        let line = make_line(TEXT5, vertical_boxes(&[10, 30, 50, 90, 110]), true);

        assert_eq!(blank_gap_positions(&line), vec![3]); // between chars 2 and 3
    }

    /// Android `uniform_vertical_line_reports_none`.
    #[test]
    fn uniform_vertical_line_reports_none() {
        let line = make_line(TEXT5, vertical_boxes(&[10, 30, 50, 70, 90]), true);

        assert!(blank_gap_positions(&line).is_empty());
    }

    /// Android `uniform_horizontal_line_reports_none`.
    #[test]
    fn uniform_horizontal_line_reports_none() {
        let line = make_line(TEXT5, horizontal_boxes(&[10, 30, 50, 70, 90]), false);

        assert!(blank_gap_positions(&line).is_empty());
    }

    /// Android `a_1_7_ratio_fires_vertically_and_not_horizontally`: spacings
    /// 20, 20, 34, 20 → 1.70×. The vertical half fires; the horizontal half
    /// is stronger on the PC — horizontal lines are never eligible at all,
    /// not merely below a second threshold.
    #[test]
    fn a_1_7_ratio_fires_vertically_and_not_horizontally() {
        let centres = [10, 30, 50, 84, 104];

        let vertical = make_line(TEXT5, vertical_boxes(&centres), true);
        assert_eq!(blank_gap_positions(&vertical), vec![3]);

        let horizontal = make_line(TEXT5, horizontal_boxes(&centres), false);
        assert!(
            blank_gap_positions(&horizontal).is_empty(),
            "horizontal is not eligible"
        );
    }

    /// Android `a_1_9_ratio_fires_horizontally_too`: the PC counterpart. The
    /// mobile detector has a second, higher horizontal threshold (1.8) and
    /// this fixture fires there; the PC never asks for horizontal detection
    /// (`BlankGaps.apply` returns early), so nothing fires.
    #[test]
    fn a_1_9_ratio_reports_nothing_horizontally() {
        let line = make_line(TEXT5, horizontal_boxes(&[10, 30, 50, 88, 108]), false);

        assert!(blank_gap_positions(&line).is_empty());
    }

    /// Android `the_vertical_threshold_is_inclusive_at_1_6`: spacings
    /// 20, 20, 32, 20 → 1.60× exactly.
    #[test]
    fn the_vertical_threshold_is_inclusive_at_1_6() {
        let line = make_line(TEXT5, vertical_boxes(&[10, 30, 50, 82, 102]), true);

        assert_eq!(blank_gap_positions(&line), vec![3]);
    }

    /// Android `thresholds_are_configurable_per_orientation`: the PC has no
    /// per-call threshold surface, just the fixed measured vertical constant
    /// (mobile `GapDetector.DEFAULT_VERTICAL_RATIO`).
    #[test]
    fn the_threshold_is_the_fixed_vertical_constant() {
        assert_eq!(BLANK_GAP_RATIO, 1.6);
    }

    /// Android `empty_char_boxes_falls_back_to_raw_alternatives`: mobile falls
    /// back to walking `rawAlternatives`; the PC has no timestep walk, so
    /// char boxes are the only geometry and a boxless line detects nothing.
    #[test]
    fn empty_char_boxes_detect_nothing() {
        let line = make_line(TEXT5, vec![], true);

        assert!(blank_gap_positions(&line).is_empty());
    }

    /// Android `text_that_does_not_match_the_timestep_walk_detects_nothing`:
    /// the PC counterpart — fewer char boxes than characters cannot be
    /// walked, and boxes beyond the text are ignored rather than mismatched.
    #[test]
    fn text_and_geometry_mismatch_detects_nothing() {
        let too_few = make_line(TEXT5, vertical_boxes(&[10, 30]), true);
        assert!(blank_gap_positions(&too_few).is_empty());

        let extra = make_line("あい", vertical_boxes(&[10, 30, 50, 90, 110]), true);
        assert!(blank_gap_positions(&extra).is_empty());
    }

    /// Android `degenerate_geometry_detects_nothing`: every character on the
    /// same pixel (median spacing 0) and a single character both report none.
    #[test]
    fn degenerate_geometry_detects_nothing() {
        let same_pixel = make_line(TEXT5, vertical_boxes(&[50, 50, 50, 50, 50]), true);
        assert!(blank_gap_positions(&same_pixel).is_empty());

        let one_char = make_line("あ", vec![], true);
        assert!(blank_gap_positions(&one_char).is_empty());
    }

    /// Android `three_char_line_median_is_contaminated_by_the_gap`: spacings
    /// 20, 40 → median 30 → the double gap reads as 1.33×, below the
    /// threshold. The pitch estimator needs a few ordinary pairs; documented,
    /// not a bug to "fix" by lowering the threshold.
    #[test]
    fn three_char_line_median_is_contaminated_by_the_gap() {
        let line = make_line("あいう", vertical_boxes(&[10, 30, 70]), true);

        assert!(blank_gap_positions(&line).is_empty());
    }

    /// Android `blank_marker_and_placeholder_are_different_characters`: the
    /// placeholder is U+25CC, not the timestep blank U+3000 the mobile
    /// `rawAlternatives` walk stores (that walk is not ported).
    #[test]
    fn the_placeholder_is_not_the_timestep_blank_marker() {
        assert_eq!(GAP_CHAR, '\u{25CC}');
        assert_ne!(GAP_CHAR, '\u{3000}');
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

    /// Android `a_measured_vertical_gap_becomes_a_placeholder_between_its_neighbours`:
    /// the placeholder lands between う and え, every character keeps its
    /// identity at its shifted index, and the synthetic alternatives entry is
    /// the placeholder itself — nothing is claimed about a character nobody
    /// has evidence for, and the panel still has a selection.
    #[test]
    fn a_measured_vertical_gap_becomes_a_placeholder_between_its_neighbours() {
        let out = apply_blank_gaps(&make_line(TEXT5, vertical_boxes(&[10, 30, 50, 90, 110]), true));

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
        let horizontal = make_line(TEXT5, horizontal_boxes(&[10, 30, 50, 90, 110]), false);

        let out = apply_blank_gaps(&horizontal);
        assert_eq!(out.text, horizontal.text);
        assert_eq!(out.char_boxes.len(), horizontal.char_boxes.len());
        assert!(blank_gap_positions(&horizontal).is_empty());
    }

    /// Android `a_line_that_already_has_a_placeholder_is_unchanged`: applied
    /// twice must not insert twice — a second placeholder would shift the
    /// user's filled override.
    #[test]
    fn a_line_that_already_has_a_placeholder_is_unchanged() {
        let once = apply_blank_gaps(&make_line(TEXT5, vertical_boxes(&[10, 30, 50, 90, 110]), true));
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
    /// 10, 30, 50, 70, 90, with a sample path and raw alternatives attached so
    /// the carry-over assertions have something to carry. (The PC line has no
    /// overrides or charCols.)
    fn fixture() -> LineResult {
        let mut line = make_line(TEXT5, vertical_boxes(&[10, 30, 50, 70, 90]), true);
        line.sample_txt = Some(std::path::PathBuf::from("/tmp/gap-fixture.txt"));
        line.raw_alternatives = vec![vec![('あ', 0.9)], vec![('い', 0.8)]];
        line
    }

    /// Android `inserts_the_placeholder_at_the_requested_index`: the
    /// placeholder lands at the index and everything else about the line is
    /// carried over (mobile checks the crop sizes; the PC its sample/raw
    /// fields).
    #[test]
    fn inserts_the_placeholder_at_the_requested_index() {
        let line = fixture();

        let out = with_gap_char(&line, 3);

        assert_eq!(out.text.chars().count(), 6);
        assert_eq!(out.text.chars().nth(3), Some(GAP_CHAR));
        assert_eq!(out.text, format!("あいう{GAP_CHAR}えお"));
        assert_eq!(out.is_vertical, line.is_vertical);
        assert_eq!(out.sample_txt, line.sample_txt);
        assert_eq!(out.raw_alternatives, line.raw_alternatives);
    }

    /// Android `all_per_character_lists_stay_the_same_length`: text, char
    /// boxes and alternatives all grow together, and the receiver is not
    /// mutated. (Mobile's overrides list has no PC analogue.)
    #[test]
    fn all_per_character_lists_stay_the_same_length() {
        let out = with_gap_char(&fixture(), 3);

        assert_eq!(out.text.chars().count(), 6);
        assert_eq!(out.char_boxes.len(), out.text.chars().count());
        assert_eq!(out.alternatives.len(), out.text.chars().count());

        let line = fixture();
        let _ = with_gap_char(&line, 3);
        assert_eq!(line.text.chars().count(), 5);
        assert_eq!(line.char_boxes.len(), 5);
        assert_eq!(line.alternatives.len(), 5);
    }

    /// Android `a_character_at_index_n_moves_to_n_plus_one`.
    #[test]
    fn a_character_at_index_n_moves_to_n_plus_one() {
        let line = fixture();

        let out = with_gap_char(&line, 3);

        for i in 0..3 {
            assert_eq!(out.text.chars().nth(i), line.text.chars().nth(i));
            assert_eq!(geom(&out.char_boxes[i]), geom(&line.char_boxes[i]));
            assert_eq!(out.alternatives[i], line.alternatives[i]);
        }
        for i in 3..5 {
            assert_eq!(out.text.chars().nth(i + 1), line.text.chars().nth(i));
            assert_eq!(geom(&out.char_boxes[i + 1]), geom(&line.char_boxes[i]));
            assert_eq!(out.alternatives[i + 1], line.alternatives[i]);
        }
    }

    /// Android `the_gap_box_sits_between_its_neighbours`: the placeholder box
    /// is centred at the midpoint of its neighbours' centres and sized as the
    /// mean of their extents along the reading axis.
    #[test]
    fn the_gap_box_sits_between_its_neighbours() {
        let line = fixture(); // vertical centres 10, 30, 50, 70, 90

        let out = with_gap_char(&line, 3);

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

        let out = with_gap_char(&line, 1);

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

        let at_start = with_gap_char(&line, 0);
        assert_eq!(at_start.text.chars().nth(0), Some(GAP_CHAR));
        assert_eq!(geom(&at_start.char_boxes[0]), geom(&line.char_boxes[0]));
        assert_eq!(at_start.char_boxes.len(), at_start.text.chars().count());

        let at_end = with_gap_char(&line, line.text.chars().count());
        assert_eq!(at_end.text.chars().nth(5), Some(GAP_CHAR));
        // Single neighbour reused.
        assert_eq!(geom(&at_end.char_boxes[5]), geom(&line.char_boxes[4]));
        assert_eq!(at_end.char_boxes.len(), at_end.text.chars().count());
    }

    /// Android `the_synthetic_alternatives_entry_defaults_to_the_reversible_blank`.
    #[test]
    fn the_synthetic_alternatives_entry_defaults_to_the_reversible_blank() {
        let out = with_gap_char(&fixture(), 3);

        assert_eq!(out.alternatives[3].len(), 1);
        assert_eq!(out.alternatives[3][0], (GAP_CHAR, 0.0));
    }

    /// Android `the_inserted_placeholder_is_unlookupable`: the placeholder
    /// occupies index 3 and the neighbours keep their identity. (The actual
    /// lookup guard lives in the Android overlay controller.)
    #[test]
    fn the_inserted_placeholder_is_unlookupable() {
        let out = with_gap_char(&fixture(), 3);

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

        let out = with_gap_char(&line, 6);
        assert_eq!(out.text, line.text);
        assert_eq!(out.char_boxes.len(), line.char_boxes.len());

        let at_end = with_gap_char(&line, 5);
        assert_eq!(at_end.text.chars().nth(5), Some(GAP_CHAR));
    }

    /// Android `empty_box_and_column_lists_stay_empty`: no geometry to
    /// interpolate, so text and alternatives grow while the box list stays
    /// empty. (The PC line carries no charCols.)
    #[test]
    fn empty_box_and_column_lists_stay_empty() {
        let line = make_line(TEXT5, vec![], true);

        let out = with_gap_char(&line, 2);

        assert_eq!(out.text.chars().count(), 6);
        assert_eq!(out.alternatives.len(), 6);
        assert!(out.char_boxes.is_empty());
    }

    /// Android `a_detected_gap_can_be_materialised_and_consumes_the_spacing`:
    /// detect → materialise → the placeholder takes up the spacing, so the
    /// line reads as uniform again. (Mobile also checks the override shift;
    /// the PC line has no overrides.)
    #[test]
    fn a_detected_gap_can_be_materialised_and_consumes_the_spacing() {
        let line = make_line(TEXT5, vertical_boxes(&[10, 30, 50, 90, 110]), true);

        let detected = blank_gap_positions(&line);
        assert_eq!(detected, vec![3]);

        let out = with_gap_char(&line, detected[0]);

        assert_eq!(out.text.chars().count(), 6);
        assert_eq!(out.char_boxes.len(), 6);
        assert_eq!(out.alternatives.len(), 6);
        assert_eq!(out.text.chars().nth(3), Some(GAP_CHAR));
        assert!(blank_gap_positions(&out).is_empty());
    }
}
