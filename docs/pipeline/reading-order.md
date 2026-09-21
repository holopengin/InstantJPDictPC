# Reading order and navigation graph

## Rule

Sort (`sort_detected_boxes`, `src/ocr_engine.rs:1545`):

- Split by the frame's own sizes (mobile `isVerticalLineBox`;
  near-square counts as horizontal), then horizontals first, verticals after.
- Horizontals: top-to-bottom, left-to-right.
- Verticals: right-edge-to-left (AABB **right** edge, not left),
  then top-to-bottom.

Nav graph (`src/nav_graph.rs`): per global char index, `[north, south,
east, west]` targets from char-box centres normalised by the page extents
(`NavGraph::build`, `:24`):

- Phase 1 (strict local): candidates within 0.05 Euclidean, 45° cone,
  cost = primary + 10 × off-axis; greedy assignment.
- Phase 2 (island connecting): unlimited distance, 45° cone, 1.5 × off-axis
  penalty, then connectivity enforcement.
- Phase 3 (wrap fill): wrapping-only candidates from the opposite
  half-plane fill remaining empty slots.
- Fewer than 5 nodes take the `fallback` (`:262`); `navigate(idx, dir)`
  (`:255`) resolves the four directions. The viewer leaves the graph dirty
  after a batch and rebuilds on demand (`build_nav_graph`,
  `src/overlay_state.rs:263`; `apply_ocr_batch` seeds the cursor on the
  first non-empty line, never on a placeholder).

## Mirrored Android source

- `OcrEngine.kt` (`sortDetectedBoxes` `:992`, `isVerticalLineBox` `:1013`,
  `:660` call site, `:2367` same comparators for line grouping).
- The three-phase torus/greedy/connectivity nav graph is PC-side structure
  with no direct mobile mirror (mobile `nav_graph_core` has drifted: W2 3.0
  vs 1.5, `DIR_MIN`, no cone + wrap bonus, different conflict resolution, no
  `enforce_connectivity` — re-sync or spec-as-intentional is open);
  ordering semantics (which the graph consumes) mirror mobile.

## Traps

- Sorting verticals on the left edge: mis-orders columns whose widths differ.
- Sorting verticals before horizontals, or mixing the two groups in one
  comparator: mobile concatenates horizontals-then-verticals.
- Using AABB sizes instead of the frame's own `w`/`h` for the vertical split:
  rotated frames misclassify.
- Seeding the cursor on a placeholder or an empty line after a batch.

## Pinning tests

- Conformance: `reading-order-01-mixed`, `detection-01/02/03` (expectations
  in reading order), via `reading_order_cases` / `detection_cases`
  (`src/conformance.rs:327` / `:237`).
- PC viewer: `apply_ocr_batch_fills_all_slots_in_one_pass`
  (`src/viewer.rs:4195`), `apply_ocr_batch_without_text_leaves_cursor_unset`
  (`:4258`); main: `detection_annotations_keep_indices_and_quads`.
- Mobile: `ConformanceCorpusTest.readingOrderCases` (unmerged branch; runs
  the real `OcrEngine.sortDetectedBoxes`, exposed internal-companion for
  host tests). **No `OcrEngineTest.readingOrder` unit test exists** — case
  JSON `mobile_mirror` values naming it are aspirational; the verified
  mirrors are `sortDetectedBoxes` itself and the branch runner.
- Gap: `NavGraph::build`/`navigate` have no dedicated unit tests on either
  side; only batch-seeding behaviour is pinned. A navigation-regression case
  is wanted.
