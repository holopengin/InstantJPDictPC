# Blank gaps and blank candidates

## Rule

Detection (the detector reports both orientations; the materialisation
*policy* below is vertical-only):

- A gap is a char spacing ≥ the line's **orientation threshold** × the line's
  own median spacing (`GapDetector::detect`, `core/src/blank_gaps.rs:111`;
  mobile `GapDetector.detect`, `util/GapDetector.kt:80`). Vertical fires at
  1.6 (`DEFAULT_VERTICAL_RATIO`, `core/src/blank_gaps.rs:27`) — measured
  recall 1.00, no false positives — and horizontal at 1.8
  (`DEFAULT_HORIZONTAL_RATIO`, `:33`), which is still weak on its own (a
  quarter of horizontal deletions leave no wider gap at all, so the
  horizontal path needs the component signal before showing anything). The
  boundary is **inclusive** (`ratio >= threshold`, the form the measured
  sweep is quoted in), and both thresholds are configurable constructor
  parameters, never hardcoded (`GapDetector::new`, `:88`;
  `threshold_for`; `detect_with` for sweeping the curve without rebuilding,
  `:119`). Median averages the two middles for even counts (`median_of`,
  `:250`; mobile `medianOf`, `util/GapDetector.kt:224`).
- Geometry comes from, in order (`geometry_of`, `core/src/blank_gaps.rs:150`):
  1. `char_boxes` centres — `center_y` for vertical, `center_x` for
     horizontal, integer maths matching mobile `JpDictRect`; spacings are
     already pixels.
  2. `char_cols` when its length equals the text length — the cached CTC
     timestep column per emitted character (#49); spacings are in timesteps
     and are converted by `pixel_per_timestep` (`crop_h / seq_len_total`
     vertical, `crop_w / seq_len_total` horizontal, else the model's own
     `DEFAULT_TIMESTEP_STRIDE_PX = 8.0`).
  3. `timestep_columns(raw_alternatives)` (`:226`) — the last resort. The
     emitted character for a timestep is the list's **first** entry (the
     argmax); a blank timestep is stored as `'\u3000'` (`TIMESTEP_BLANK_CHAR`)
     and **resets** the previous character, so a repeat after a blank is
     emitted; a space is always emitted and never collapses; any other
     character collapses only against the immediately preceding emitted one.
     A derived column count that disagrees with the text length is not
     geometry — a line that already carries a `◌` placeholder derives fewer
     columns than its text, which is the expected mismatch.
  `span_px` is the pair's spacing in pixels; the ratio is scale-invariant, so
  the threshold carries across all three sources.
- Detection **refuses unusable geometry** rather than inventing a gap: fewer
  than two emitted characters (`MIN_EMITTED_CHARS`, `:41`), a median spacing
  of `<= 0` (everything on one pixel), and the contaminated median of a
  three-character line (spacings 20, 40 → median 30 → the doubled gap reads
  as 1.33×, below both thresholds) all report nothing. The pitch estimator
  needs a few ordinary pairs; documented, not a bug to "fix" by lowering the
  threshold.
- **The materialisation policy is vertical-only** — `BlankGaps.apply` returns
  early on horizontal lines (mobile `util/BlankGaps.kt:56`; PC
  `apply_blank_gaps`, `core/src/blank_gaps.rs:413`), because 13% of ordinary
  horizontal intervals look like gaps. A horizontal line is therefore never
  given a placeholder even though the detector can report a gap on it.
- Each gap materialises as a `◌` (`GAP_CHAR`, U+25CC, `core/src/models.rs:385`;
  mobile `OcrEngine.GAP_CHAR`) placeholder inserted right-to-left so detector
  indices stay valid, growing **every** parallel per-character list together —
  text, char boxes, alternatives, `char_cols` — and shifting `overrides` keys
  `>= index` by +1 so a filled blank or an applied correction keeps pointing
  at its own character (`with_gap_char_at`, `core/src/blank_gaps.rs:337`;
  mobile `LineResult.withGapCharAt`, `util/LineResultGap.kt:48`). Lists that
  carry no data stay empty ("empty" means "unknown"); the placeholder itself
  is not inserted as an override. Idempotent: a line already containing the
  placeholder is unchanged. The placeholder box is interpolated between its
  neighbours (`interpolate_gap_box`, `:267`; mobile `interpolateGapBox`,
  `util/LineResultGap.kt:127`) and the new column is the midpoint of its
  neighbours' columns (`column_for`, `:305`; mobile `BlankGaps.columnFor`,
  `util/BlankGaps.kt:74`).

Candidates for a blank:

- The pool comes from **every** per-timestep top-K list in discovery order,
  deduped, capped at 15 (`MAX`, `core/src/util/gap_candidates.rs:16`;
  `generate` `:30`; mobile `GapCandidates.MAX` / `generate`,
  `util/GapCandidates.kt:19` / `:44`).
- The CharLM reorders the pool by line context (`CharLm::rank`,
  `core/src/util/char_lm.rs:149`; mobile `CharLm.rank`, `util/CharLm.kt:81`).
  The LM itself: packed table, orders 1–4 (`MAX_ORDER`, `:25`), binary
  search (`count` `:80`), longest-context back-off (`log_prob` `:124`).
- Empty pool falls back to **punctuation then kana**
  (`PUNCT_DEFAULTS` `、。「」…ー`, `KANA_DEFAULTS` `はのをに と`,
  `fallback` `core/src/util/gap_candidates.rs:62`; mobile
  `util/GapCandidates.kt:63-81`, LM-ranked per class when a model is loaded).
- Context stops at the placeholder and is clipped to `MAX_ORDER - 1` chars
  (`context_before` `:77`; mobile `contextBefore`,
  `util/GapCandidates.kt:90`).
- After a fill, **the list is retained** — a filled blank stays an override
  and keeps its candidate list (mobile `BlankGapsTest`, PC
  `filling_a_blank_keeps_its_candidate_list`).

## Mirrored Android source

- `util/GapDetector.kt` (`detect` `:80`, `thresholdFor` `:76`,
  `timestepColumns` `:198`, `medianOf` `:224`),
  `util/BlankGaps.kt` (`apply` `:56`, `applyIfEnabled` `:49`,
  `columnFor` `:74`), `util/LineResultGap.kt` (`withGapCharAt` `:48`,
  `withGapAt` `:112`, `interpolateGapBox` `:127`),
  `util/GapCandidates.kt`, `util/CharLm.kt`.

## Traps

- Offering candidates from only the argmax timestep: misses the pool the
  model actually ranked.
- Dropping the placeholder from the context instead of clipping there: the
  LM then scores with text from the wrong side of the gap.
- Re-sorting ties instead of keeping discovery order (`rank` keeps pool order
  on ties — pinned by `charlm-01`).
- Regenerating the list after a fill: loses the override provenance.
- Inserting left-to-right in `apply_blank_gaps`: the detector's indices are
  computed against the original text, so a later insertion lands one
  character off (pinned by `multiple_gaps_insert_right_to_left_and_stay_index_aligned`).
- Treating a blank timestep as an ordinary character in the walk: the repeat
  after a blank then collapses and the derived column count silently misses a
  character.
- Lowering the threshold to "fix" the three-character-line median: that
  breaks the measured false-positive rate.

## Pinning tests

- PC unit: `blank_gaps.rs` tests — `GapDetectorTest` mirrors
  (`double_spacing_in_the_middle_reports_exactly_one_gap_with_ratio_two`,
  `a_1_7_ratio_fires_vertically_and_not_horizontally`,
  `a_1_9_ratio_fires_horizontally_too`,
  `the_vertical_threshold_is_inclusive_at_1_6`,
  `thresholds_are_configurable_per_orientation`,
  `empty_char_boxes_falls_back_to_raw_alternatives`,
  `fallback_span_uses_the_crop_length_when_it_is_known`,
  `char_cols_geometry_is_used_when_boxes_are_missing`,
  `text_that_does_not_match_the_timestep_walk_detects_nothing`,
  `timestep_walk_matches_the_documented_assumptions`,
  `degenerate_geometry_detects_nothing`,
  `three_char_line_median_is_contaminated_by_the_gap`,
  `median_helper_handles_odd_even_and_empty_input`), `BlankGapsTest` mirrors
  (`the_placeholder_column_sits_between_its_neighbours`,
  `a_filled_blank_is_still_an_override_and_survives_the_shift`, …),
  `LineResultGapTest` mirrors (`overrides_follow_the_characters_they_were_applied_to`,
  `every_override_shifts_when_inserting_at_the_start`,
  `a_caller_can_supply_its_own_gap_alternatives`,
  `with_gap_at_is_the_same_operation_as_with_gap_char_at`,
  `a_detected_gap_can_be_materialised_and_consumes_the_spacing`, …), plus the
  desktop wrapper `blank_gaps_detect_wide_vertical_spacing`
  (`src/viewer.rs`); `gap_candidates.rs` tests
  (`the_pool_comes_from_every_timestep_in_discovery_order`,
  `the_model_reorders_the_pool_by_context`,
  `an_empty_pool_falls_back_to_punctuation_then_kana`,
  `the_context_drops_the_placeholder_and_older_characters`,
  `offerable_rejects_the_placeholder_the_blank_and_controls`);
  `char_lm.rs` tests (`count_finds_entries_of_every_order_and_misses_cleanly`,
  `log_prob_backs_off_from_the_longest_known_context`,
  `rank_prefers_what_the_context_predicts`,
  `shipped_model_parses_and_knows_japanese`); overlay state
  (`a_blank_offers_more_than_the_placeholder`,
  `a_blank_without_evidence_falls_back_to_punctuation_then_kana`,
  `filling_a_blank_keeps_its_candidate_list`,
  `an_installed_model_ranks_the_blank_list_by_context`).
- Conformance: `gap-detection-01-threshold-pair`,
  `gap-detection-02-char-cols-fallback`,
  `gap-detection-03-timestep-walk-fallback`,
  `gap-detection-04-degenerate-and-contaminated` (kind `gap_detection`,
  schema in `tests/conformance/FORMAT.md`); `gap-01-discovery-order`,
  `gap-02-lm-reorder`, `gap-03-fallback-order`, `gap-04-context-clip`,
  `charlm-01-counts-and-rank`, via `gap_detection_cases` / `gap_cases` /
  `char_lm_cases` (`core/src/conformance.rs`).
- Mobile: `GapDetectorTest` (thresholds, geometry fallbacks, the timestep
  walk, the median), `BlankGapsTest` (placeholder geometry, idempotence,
  fill retention), `LineResultGapTest` (parallel-list growth, override
  shifts, interpolation), `GapCandidatesTest`
  (`the_pool_keeps_kana_and_punctuation_and_deduplicates`,
  `the_model_reorders_the_pool_by_the_line_context`,
  `the_context_stops_at_the_placeholder_and_the_model_order`, …),
  `CharLmAssetTest` (`the_shipped_model_ranks_a_gap_pool`, …).
