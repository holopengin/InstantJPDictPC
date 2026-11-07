//! Mobile `FuriganaRule` (#28/#99): the furigana (ruby) geometry rules, pure
//! so they are host-tested.
//!
//! Matching runs on RAW contour geometry (pre-unclip: unclip padding
//! fabricates overlap for stacked fragments); only the gap and short-side
//! tests use UNCLIPPED boxes (raw gutters are real pixels, unclip closes them
//! to ruby distance). Orientation gating (vertical vs horizontal, near-square
//! counts as both) lives here too: [`filter_furigana`] is the index-aligned
//! pass over a page, shared by the desktop engine and the Android binding (one
//! call instead of one per box pair).
//!
//! Both rules require a significant size difference in BOTH dimensions:
//! vertical needs a much shorter height AND a narrower width; horizontal
//! (since #99) needs thinness AND a much shorter width. Thinness alone used to
//! be enough horizontally, and it ate real short lines on a receipt — detail
//! text under a heading, same-ish width — as if they were ruby.

use crate::models::{BoundingBox, VERTICAL_MIN_ASPECT};

/// Vertical: small long-side < 30% of large long-side.
const SIZE_RATIO: f32 = 0.3;
/// Horizontal ruby runs long but thin; measured ~0.65-0.70 of main height here
/// (vertical ruby stays short-only).
const THIN_RATIO: f32 = 0.75;
/// Horizontal unclipped short-side ceiling: thin ruby here runs ~0.7 of main
/// height; full-height short lines (≈1.0) must survive.
const HSHORT_RATIO: f32 = 0.85;
/// Small short-side < 65% of large short-side: ruby glyphs run smaller;
/// full-width short lines (か？」) and short columns survive this.
const WIDTH_RATIO: f32 = 0.65;
/// #99 horizontal long-side ceiling, checked on the unclipped boxes like
/// [`WIDTH_RATIO`]: a thin line that is nearly as wide as its neighbour is
/// real text, not ruby. Ruby strips measured at or below ~0.65 of their line
/// width still pass.
const HWIDTH_RATIO: f32 = 0.65;
/// Gap <= 50% of large short-side.
const GAP_RATIO: f32 = 0.5;
/// Overlap >= 50% of small long-side.
const OVERLAP_RATIO: f32 = 0.5;
/// Absolute ceiling: real short columns (e.g. 458px) dwarf ruby runs even when
/// the ratio matches — ruby longer than 12% of the image side is not ruby.
const MAX_FRAC: f32 = 0.12;
/// Absolute floor on the ANNOTATED box: ruby hugs full-size body text, not
/// compact blocks (logo boxes, badges). Catches caption strips above logo
/// blocks.
const BIG_MIN_FRAC: f32 = 0.2;

/// Tiny vertical box hugging a much larger vertical box (either side) (#28).
/// The center must lie OUTSIDE the big box: stacked column fragments (tail of
/// the column above/below, overlapping only via unclip padding) share its
/// x-range. Size/center/overlap use RAW contour geometry; gap uses UNCLIPPED
/// (raw gutters are real pixels, unclip closes them to ruby distance).
pub fn is_ruby_vertical(
    s_raw: &BoundingBox,
    b_raw: &BoundingBox,
    s_un: &BoundingBox,
    b_un: &BoundingBox,
    img_h: i32,
) -> bool {
    let img_h = img_h as f32;
    if (b_raw.h as f32) < img_h * BIG_MIN_FRAC {
        return false;
    }
    if (s_raw.h as f32) >= (b_raw.h as f32) * SIZE_RATIO {
        return false;
    }
    if (s_raw.h as f32) >= img_h * MAX_FRAC {
        return false;
    }
    if (s_un.w as f32) >= (b_un.w as f32) * WIDTH_RATIO {
        return false;
    }
    let cx = (s_raw.left() + s_raw.right()) / 2;
    if cx >= b_raw.left() && cx <= b_raw.right() {
        return false;
    }
    if (gap_len(s_un.left(), s_un.right(), b_un.left(), b_un.right()) as f32)
        > (b_un.w as f32) * GAP_RATIO
    {
        return false;
    }
    if (overlap_len(s_raw.top(), s_raw.bottom(), b_raw.top(), b_raw.bottom()) as f32)
        < (s_raw.h as f32) * OVERLAP_RATIO
    {
        return false;
    }
    true
}

/// Tiny horizontal box right above a much larger horizontal box (#28 (same
/// split), #99 both dimensions). The candidate must be smaller in BOTH
/// dimensions: thinness alone let a short receipt line — detail text under a
/// heading, same-ish width — be dropped as if it were ruby. Horizontal ruby is
/// thin AND much shorter than the annotated line; a thin but wide line is real
/// text. (Vertical has the same both-axes shape: much shorter and much
/// narrower.)
pub fn is_ruby_horizontal(
    s_raw: &BoundingBox,
    b_raw: &BoundingBox,
    s_un: &BoundingBox,
    b_un: &BoundingBox,
    img_w: i32,
    img_h: i32,
) -> bool {
    let (img_w, img_h) = (img_w as f32, img_h as f32);
    if (b_raw.w as f32) < img_w * BIG_MIN_FRAC {
        return false;
    }
    if (s_raw.h as f32) >= (b_raw.h as f32) * THIN_RATIO {
        return false;
    }
    if (s_raw.h as f32) >= img_h * MAX_FRAC {
        return false;
    }
    if (s_un.h as f32) >= (b_un.h as f32) * HSHORT_RATIO {
        return false;
    }
    if (s_un.w as f32) >= (b_un.w as f32) * HWIDTH_RATIO {
        return false;
    }
    // Above-ness on RAW geometry: unclip grows both boxes toward each other
    // (~18px mutual encroachment here), flipping genuinely-above ruby to
    // overlapping.
    if s_raw.bottom() > b_raw.top() + 2 {
        return false;
    }
    if (b_un.top() - s_un.bottom()) as f32 > (b_un.h as f32) * GAP_RATIO {
        return false;
    }
    if (overlap_len(s_raw.left(), s_raw.right(), b_raw.left(), b_raw.right()) as f32)
        < (s_raw.w as f32) * OVERLAP_RATIO
    {
        return false;
    }
    true
}

fn overlap_len(a1: i32, a2: i32, b1: i32, b2: i32) -> i32 {
    (a2.min(b2) - a1.max(b1)).max(0)
}

fn gap_len(a1: i32, a2: i32, b1: i32, b2: i32) -> i32 {
    (a1.max(b1) - a2.min(b2)).max(0)
}

/// Mobile shared orientation rule (#28): near-square boxes count as vertical
/// for the ruby checks, so lone upright characters are tested against both
/// rules.
pub fn is_vertical_box(b: &BoundingBox) -> bool {
    b.h as f32 >= b.w as f32 * VERTICAL_MIN_ASPECT
}

/// Near-square (single-kanji-like) box: checked against both furigana rules.
pub fn is_square_box(b: &BoundingBox) -> bool {
    let (w, h) = (b.w as f32, b.h as f32);
    w.min(h) >= w.max(h) / VERTICAL_MIN_ASPECT
}

/// Mobile `filterFurigana` (#28): keep-flags for likely-furigana boxes.
/// `raw`/`uncl` are index-aligned (raw contour AABBs vs unclipped boxes).
///
/// One call for the whole page: the O(n²) pair walk runs here, so a binding
/// crosses the FFI once instead of once per pair.
pub fn filter_furigana(raw: &[BoundingBox], uncl: &[BoundingBox], img_w: i32, img_h: i32) -> Vec<bool> {
    if raw.len() < 2 {
        return vec![true; raw.len()];
    }
    (0..raw.len())
        .map(|i| {
            let small = &raw[i];
            let check_vert = is_vertical_box(small) || is_square_box(small);
            let check_horiz = !is_vertical_box(small) || is_square_box(small);
            !raw.iter().enumerate().any(|(j, big)| {
                j != i
                    && ((check_vert
                        && is_vertical_box(big)
                        && is_ruby_vertical(&raw[i], big, &uncl[i], &uncl[j], img_h))
                        || (check_horiz
                            && !is_vertical_box(big)
                            && is_ruby_horizontal(&raw[i], big, &uncl[i], &uncl[j], img_w, img_h)))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mobile `FuriganaRuleTest`: shapes pinned on the cases that matter. The
    /// receipt regression (#99): the horizontal rule used to accept any thin
    /// box above a bigger line, so a short real line — detail text under a
    /// heading, nearly as wide as the line it relates to — was dropped as
    /// ruby. Horizontal candidates now need both dimensions, exactly as the
    /// vertical rule always did.
    const IMG: i32 = 1000;

    fn rect(l: i32, t: i32, r: i32, b: i32) -> BoundingBox {
        BoundingBox::new(l, t, r - l, b - t, 1.0)
    }

    #[test]
    fn thin_ruby_strip_above_its_line_is_ruby() {
        let big = rect(100, 300, 600, 360); // 500x60 line
        let small = rect(200, 276, 320, 300); // 120x24: thin and much shorter
        let small_un = rect(188, 266, 332, 308);
        let big_un = rect(90, 290, 610, 370);
        assert!(is_ruby_horizontal(
            &small, &big, &small_un, &big_un, IMG, IMG
        ));
    }

    #[test]
    fn thin_but_wide_real_line_is_not_ruby() {
        // The receipt shape: small detail text above a heading — thin, but
        // nearly as wide as the line below it. Must survive as a real line.
        let big = rect(100, 300, 600, 360);
        let small = rect(100, 270, 600, 300); // 500x30
        let small_un = rect(85, 258, 615, 312);
        let big_un = rect(90, 290, 610, 370);
        assert!(!is_ruby_horizontal(
            &small, &big, &small_un, &big_un, IMG, IMG
        ));
    }

    #[test]
    fn narrow_short_column_beside_its_column_is_ruby() {
        let big = rect(300, 50, 360, 850); // 60x800 column
        let small = rect(276, 200, 300, 300); // 24x100: much shorter and narrower
        let small_un = rect(266, 190, 310, 310);
        let big_un = rect(290, 40, 370, 860);
        assert!(is_ruby_vertical(&small, &big, &small_un, &big_un, IMG));
    }

    #[test]
    fn full_width_short_column_is_not_ruby() {
        let big = rect(300, 50, 360, 850);
        let small = rect(300, 200, 360, 300); // same glyph width, just shorter
        let small_un = rect(290, 190, 370, 310);
        let big_un = rect(290, 40, 370, 860);
        assert!(!is_ruby_vertical(&small, &big, &small_un, &big_un, IMG));
    }
}
