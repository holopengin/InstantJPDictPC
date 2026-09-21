# Rotated geometry

## Rule

- The stored frame angle is the frame's **local-x-axis angle** from +x
  (y-down, positive = visually clockwise), kept in code as radians on
  `RotatedBox.angle` (`core/src/models.rs:62`, `RotatedBox::new`).
- **Glyph tilt equals that angle in both orientations.** For a vertical frame
  it is *not* `angle + 90°`: `atan2(-y.x, y.y)` on the rotated y-axis reduces
  to the same x-axis angle (`src/viewer.rs:706`, `glyph_tilt`). Only frames
  where `is_rotated()` tilt at all; the tilt rotates about char-box centres.
- Verticality is the 1.25 aspect rule (`VERTICAL_MIN_ASPECT`,
  `core/src/models.rs:30`; `is_vertical`, `:115`). Near-square stays horizontal.
- Axis-aligned is the 1° band **widened by fit quantization**:
  `max(1°, atan(1.5px / long side))` (`AXIS_ALIGNED_TOL_RAD` `:35`,
  `AXIS_ALIGNED_QUANT_TOL_PX` `:43`, `is_axis_aligned` `:132`). A 20px frame
  at 3° still counts as axis-aligned and takes the rect path.
- `fit_quad` fits the boundary-walk points (`core/src/models.rs:278`); degenerate
  point sets (single point, pair, collinear) return `None`.
- `unclip` expands along the box's own axes keeping the centre
  (`core/src/models.rs:140`); `map_local_rect` maps upright local crops back
  through the quad (`:171`).
- Detection runs the real chain in pipeline order — `fit_components` →
  furigana/min-size/enclosing-blob filter → straight-box merge →
  reading-order sort — with the **case's** `det_thresh`/`det_unclip`, never
  platform defaults. Defaults are converged: PC 0.25 / 0.7
  (`PPOCR_DET_THRESH`, `PPOCR_DET_UNCLIP_RATIO`, `core/src/ocr_engine.rs:21-23`,
  env overrides `DET_THRESH`/`DET_UNCLIP` at `:63`/`:72`) = mobile 0.25 /
  0.70 (mobile commit `d7f1575`, tuning #101, superseding #97's 0.65 / 1.2).

## Mirrored Android source

- `RotatedGeometry.kt` (`AXIS_ALIGNED_TOL_DEG`, `:32`); `OcrEngine.kt`
  (`:685` axis-aligned comment, `:805` rect-vs-quad path, `:1013`
  `isVerticalLineBox`).
- `LineOverlayView.kt:127` (`line.tiltDeg`).
- DECISION 2026-09-22: the formula is required behavior —
  `max(1°, atan(1.5px / long side))`. The pre-emptive fix was reverted off
  the Android conformance branch so the corpus enforces it: `geometry-01`
  frame 8 fails there until the behavior lands (Android
  `conformance/01-android-runner` @ `82d77bd`,
  `conformance/02-graduated-rules` @ `e24e057`). Do not weaken the PC side
  toward plain-1° in the meantime: `geometry-01` frame 8 pins the widening.

## Traps

- Adding 90° for vertical frames (treating the angle as the long axis):
  turned a −1.1° column into ~89° sideways glyphs (commit `12c8f26`).
- Sorting vertical lines on the left edge instead of the AABB **right** edge
  (see [reading-order](reading-order.md)).
- Probing the overlap rule with fully-contained box pairs: they cannot feel
  `OVERLAP_RATIO`, so the perturbation proof needed the extra
  `furigana-05-overlap-boundary` case.
- Retuning detection defaults without updating cases: every `detection` case
  states its values explicitly so a retune fails loudly (TOLERANCES.md §3).

## Pinning tests

- PC unit (`core/src/models.rs` tests): `rotated_horizontal_rect_fits_the_rect_it_was_made_from`,
  `rotated_vertical_column_reads_top_to_bottom`,
  `axis_aligned_component_fits_its_pixel_extents`,
  `near_square_boxes_stay_horizontal`,
  `interior_and_duplicate_points_do_not_disturb_the_fit`,
  `degenerate_point_sets_have_no_fit`,
  `unclip_expands_along_the_boxes_own_axes`,
  `short_noisy_fits_snap_to_straight`,
  `axis_aligned_frame_maps_local_boxes_one_to_one`,
  `rotated_frame_maps_local_corners_onto_the_fitted_corners`,
  `aabb_rounds_the_frame_corners`.
- PC viewer: `glyph_tilt_follows_the_reading_axis` (`src/viewer.rs:4097`).
- Conformance: `geometry-01-angle-conventions` (angle/vertical/axis-aligned
  per frame), `detection-01-h-bars`, `detection-02-v-columns`,
  `detection-03-ruby-strip` (full chain + order), via `geometry_cases` /
  `detection_cases` (`core/src/conformance.rs:288` / `:237`).
- Mobile: `RotatedGeometryTest` (`anAxisAlignedComponentFitsItsPixelExtents`,
  `aRotatedHorizontalRectFitsTheSameRectItWasMadeFrom`,
  `aRotatedVerticalColumnReadsTopToBottomAndKeepsTheColumnsOwnRotation`,
  `nearSquareBoxesStayHorizontalLikeTheAxisAlignedRule`, …);
  `ConformanceCorpusTest.geometryCases` (unmerged branch).
