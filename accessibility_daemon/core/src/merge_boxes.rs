//! Pure straight-box overlap merge shared by the desktop engine and the
//! mobile UniFFI shim.
//!
//! Everything here is data in / data out over [`crate::models`] geometry —
//! no `image` types, no ncnn handles — so this module is **not** gated on
//! the `native` feature, unlike [`crate::ocr_engine`]. Moved here verbatim
//! from that module (behaviour unchanged; its caller now delegates via a
//! `pub(crate)` re-export); the only additions are the threshold-explicit
//! [`should_merge`] / [`merge_overlapping_boxes`] rect-level form, which is
//! mobile `OcrEngine.mergeOverlappingBoxes` / `shouldMerge` over plain
//! `BoundingBox`es so the Android shim can pass its live `xOverlapThresh`
//! pref instead of the desktop default.
//!
//! Shape reconciliation: Android merges straight rects (`JpDictRect`, i.e.
//! `x/y/w/h` with no confidence), while the desktop merges
//! `(BoundingBox, RotatedBox)` pairs. The pure rule — intersection over the
//! smaller box plus the vertical-centre gate — only ever reads the AABB, so
//! it lives here once at rect level ([`should_merge`], [`union_boxes`],
//! [`merge_overlapping_boxes`]); the desktop-only frame concerns (skip
//! rotated frames, rebuild the merged frame axis-aligned) stay in the
//! frame-level [`merge_straight_boxes`], which is this same rect rule
//! applied to the pairs' AABBs at the default threshold.

use crate::models::{BoundingBox, RotatedBox};

/// Mobile `xOverlapThresh` pref default: union two straight boxes when their
/// intersection covers at least this fraction of the smaller box.
pub const X_OVERLAP_THRESHOLD: f32 = 0.40;

/// Mobile `OcrEngine.shouldMerge` with an explicit overlap threshold: the
/// intersection must cover at least `x_overlap_thresh` of the smaller box
/// and the vertical centres must sit within one average height of each
/// other. The desktop calls this at [`X_OVERLAP_THRESHOLD`]; Android passes
/// its live pref (same 0.40 default).
pub fn should_merge(a: &BoundingBox, b: &BoundingBox, x_overlap_thresh: f32) -> bool {
    let (ax1, ay1, ax2, ay2) = (a.x, a.y, a.x + a.w, a.y + a.h);
    let (bx1, by1, bx2, by2) = (b.x, b.y, b.x + b.w, b.y + b.h);
    let (ix1, iy1) = (ax1.max(bx1), ay1.max(by1));
    let (ix2, iy2) = (ax2.min(bx2), ay2.min(by2));
    if ix1 >= ix2 || iy1 >= iy2 {
        return false;
    }
    let inter = (ix2 - ix1) as f32 * (iy2 - iy1) as f32;
    let min_area = (a.w * a.h).min(b.w * b.h) as f32;
    if min_area <= 0.0 {
        return false;
    }
    if inter / min_area < x_overlap_thresh {
        return false;
    }
    let y_diff = ((ay1 + ay2) as f32 / 2.0 - (by1 + by2) as f32 / 2.0).abs();
    let avg_h = (a.h + b.h) as f32 / 2.0;
    y_diff <= avg_h
}

/// Desktop shorthand at the default threshold (what `detect_lines` uses).
pub(crate) fn should_merge_straight(a: &BoundingBox, b: &BoundingBox) -> bool {
    should_merge(a, b, X_OVERLAP_THRESHOLD)
}

/// Union of two AABBs, keeping the stronger confidence.
fn union_boxes(a: &BoundingBox, b: &BoundingBox) -> BoundingBox {
    let x1 = a.x.min(b.x);
    let y1 = a.y.min(b.y);
    let x2 = (a.x + a.w).max(b.x + b.w);
    let y2 = (a.y + a.h).max(b.y + b.h);
    BoundingBox::new(x1, y1, x2 - x1, y2 - y1, a.confidence.max(b.confidence))
}

/// Mobile `OcrEngine.mergeOverlappingBoxes` over plain rects: largest box
/// first, greedily unioning every later box the running union should merge
/// with. Bit-for-bit the Kotlin loop (stable area-descending order, forward
/// inner scan, progressive `current`), with the live threshold as a
/// parameter. The confidence channel is carried through `union_boxes`' max
/// rule; the Android shim passes 1.0 and drops it on return.
pub fn merge_overlapping_boxes(
    boxes: Vec<BoundingBox>,
    x_overlap_thresh: f32,
) -> Vec<BoundingBox> {
    if boxes.len() < 2 {
        return boxes;
    }
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    // Largest box first, like mobile (`sortedByDescending { area }`).
    // Stable, like Kotlin's sort.
    order.sort_by_key(|&i| std::cmp::Reverse(boxes[i].w as i64 * boxes[i].h as i64));
    let mut handled = vec![false; boxes.len()];
    let mut out = Vec::with_capacity(boxes.len());
    for (pos, &i) in order.iter().enumerate() {
        if handled[i] {
            continue;
        }
        let mut cur = boxes[i].clone();
        handled[i] = true;
        for &j in &order[pos + 1..] {
            if handled[j] {
                continue;
            }
            if should_merge(&cur, &boxes[j], x_overlap_thresh) {
                cur = union_boxes(&cur, &boxes[j]);
                handled[j] = true;
            }
        }
        out.push(cur);
    }
    out
}

/// Mobile `OcrEngine.mergeOverlappingBoxes`, restricted to straight frames
/// (no quad): the rotated path has no merge on mobile, so only boxes that
/// would have been axis-aligned there take part. A merged box becomes a
/// plain axis-aligned frame (angle 0).
/// Visible to the conformance corpus runner (`crate::conformance`): these are
/// pipeline stages both sides share, pinned by cases in
/// `tests/conformance/cases/`.
pub(crate) fn merge_straight_boxes(
    pairs: Vec<(BoundingBox, RotatedBox)>,
) -> Vec<(BoundingBox, RotatedBox)> {
    if pairs.len() < 2 {
        return pairs;
    }
    let straight: Vec<bool> = pairs.iter().map(|(_, r)| !r.is_rotated()).collect();
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    // Largest box first, like mobile (`sortedByDescending { area }`).
    order.sort_by_key(|&i| std::cmp::Reverse(pairs[i].0.w as i64 * pairs[i].0.h as i64));
    let mut handled = vec![false; pairs.len()];
    let mut out = Vec::with_capacity(pairs.len());
    for &i in &order {
        if handled[i] {
            continue;
        }
        handled[i] = true;
        if !straight[i] {
            out.push(pairs[i].clone());
            continue;
        }
        let mut cur = pairs[i].0.clone();
        for &j in &order {
            if handled[j] || !straight[j] {
                continue;
            }
            if should_merge_straight(&cur, &pairs[j].0) {
                cur = union_boxes(&cur, &pairs[j].0);
                handled[j] = true;
            }
        }
        let frame = RotatedBox::new(
            cur.x as f32 + cur.w as f32 / 2.0,
            cur.y as f32 + cur.h as f32 / 2.0,
            cur.w as f32,
            cur.h as f32,
            0.0,
            cur.confidence,
        );
        out.push((cur, frame));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bb(x: i32, y: i32, w: i32, h: i32) -> BoundingBox {
        BoundingBox::new(x, y, w, h, 0.8)
    }

    #[test]
    fn overlapping_halves_union_at_default_threshold() {
        // Overlap 50x20 of a 100x20 box: IoM 0.50 >= 0.40, union.
        let merged = merge_overlapping_boxes(vec![bb(0, 0, 100, 20), bb(50, 0, 100, 20)], 0.40);
        assert_eq!(merged.len(), 1);
        assert_eq!((merged[0].x, merged[0].y, merged[0].w, merged[0].h), (0, 0, 150, 20));
    }

    #[test]
    fn small_overlap_stays_split() {
        // Overlap 30x20: IoM 0.30 < 0.40.
        let apart = merge_overlapping_boxes(vec![bb(0, 0, 100, 20), bb(70, 0, 100, 20)], 0.40);
        assert_eq!(apart.len(), 2);
    }

    #[test]
    fn threshold_is_live() {
        // IoM 0.30 merges at 0.25 but not at 0.40.
        let boxes = vec![bb(0, 0, 100, 20), bb(70, 0, 100, 20)];
        assert_eq!(merge_overlapping_boxes(boxes.clone(), 0.25).len(), 1);
        assert_eq!(merge_overlapping_boxes(boxes, 0.40).len(), 2);
    }

    #[test]
    fn vertical_gate_blocks_stacked_lines() {
        // Full x-overlap but centres 40px apart with avg height 20: no merge.
        let stacked = merge_overlapping_boxes(vec![bb(0, 0, 100, 20), bb(0, 40, 100, 20)], 0.40);
        assert_eq!(stacked.len(), 2);
        assert!(!should_merge(&bb(0, 0, 100, 20), &bb(0, 40, 100, 20), 0.40));
    }

    #[test]
    fn disjoint_boxes_never_merge() {
        assert!(!should_merge(&bb(0, 0, 50, 20), &bb(100, 0, 50, 20), 0.0));
    }

    #[test]
    fn degenerate_area_never_merges() {
        assert!(!should_merge(&bb(0, 0, 0, 20), &bb(0, 0, 50, 20), 0.0));
    }
}
