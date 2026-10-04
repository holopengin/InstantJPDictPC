# Reading order and navigation graph

## Rule

Sort (`sort_detected_boxes`, `core/src/ocr_engine.rs:1545`):

- Split by the frame's own sizes (mobile `isVerticalLineBox`;
  near-square counts as horizontal), then horizontals first, verticals after.
- Horizontals: top-to-bottom, left-to-right.
- Verticals: right-edge-to-left (AABB **right** edge, not left),
  then top-to-bottom.

Nav graph (`core/src/nav_graph.rs`): per global char index, `[north, south,
east, west]` targets from char-box centres normalised by the page extents
(`NavGraph::build`, `:24`):

- Phase 1 (strict local): candidates within 0.05 Euclidean, 45° cone,
  cost = primary + 10 × off-axis; greedy assignment.
- Phase 2 (island connecting): unlimited distance, 45° cone, 1.5 × off-axis
  penalty, then connectivity enforcement.
- Connectivity enforcement (`enforce_connectivity`, PC-side): the graph is
  made strongly connected in one O(n + E) pass over the SCC condensation — a
  directed cycle is closed through every component, adding each repair link to
  an empty slot (best-aligned direction) wherever possible. Runs before Phase 3
  (which then only fills the remaining empty slots, preserving connectivity).
- Phase 3 (wrap fill): wrapping-only candidates from the opposite
  half-plane fill remaining empty slots.
- Fewer than 5 nodes take the `fallback` (`:262`); `navigate(idx, dir)`
  (`:255`) resolves the four directions. The viewer leaves the graph dirty
  after a batch and rebuilds on demand (`build_nav_graph`,
  `core/src/overlay_state.rs:263`; `apply_ocr_batch` seeds the cursor on the
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
  (`core/src/conformance.rs:327` / `:237`).
- PC viewer: `apply_ocr_batch_fills_all_slots_in_one_pass`
  (`src/viewer.rs:4195`), `apply_ocr_batch_without_text_leaves_cursor_unset`
  (`:4258`); main: `detection_annotations_keep_indices_and_quads`.
- PC core: `nav_graph_01..08` (`core/src/nav_graph.rs`) pin the exact
  neighbour tables / `initial_edges`; `nav_graph_09/10` are differential
  oracle tests for the #107 top-k enumerators; `nav_graph_11` asserts strong
  connectivity after build on every layout; `nav_graph_12` proves the #107
  enforcement fix leaves already-connected graphs byte-identical and never
  sacrifices more Phase-1 links than the old repair; `nav_graph_13` covers
  large layouts the old swap loop could not connect. The pre-#107 build and
  list construction are kept verbatim under `cfg(test)` as the oracle.
- Mobile: `ConformanceCorpusTest.readingOrderCases` (unmerged branch; runs
  the real `OcrEngine.sortDetectedBoxes`, exposed internal-companion for
  host tests). **No `OcrEngineTest.readingOrder` unit test exists** — case
  JSON `mobile_mirror` values naming it are aspirational; the verified
  mirrors are `sortDetectedBoxes` itself and the branch runner.
- Gap: the nav graph has no conformance-corpus case (the corpus has no
  nav-graph kind); the Rust unit tests above are the pin, mirrored on the
  Android side by `nav_graph_core/src/lib.rs` and `NavGraphCoreTest.kt`.
