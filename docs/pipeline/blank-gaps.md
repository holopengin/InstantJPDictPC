# Blank gaps and blank candidates

## Rule

Detection (vertical lines only):

- A gap is a char spacing ≥ 1.6× the line's median spacing
  (`BLANK_GAP_RATIO`, `src/viewer.rs:835`; mobile
  `GapDetector.DEFAULT_VERTICAL_RATIO = 1.6f`,
  `util/GapDetector.kt:156`). Median averages the two middles for even
  counts (`median_of`, `src/viewer.rs:839`; mobile `medianOf`,
  `util/GapDetector.kt:224`).
- **Horizontal lines are never eligible** — `BlankGaps.apply` returns early
  on them (mobile `util/BlankGaps.kt:56`; PC `blank_gap_positions`,
  `src/viewer.rs:853`). Note mobile's detector also carries a horizontal
  threshold (`DEFAULT_HORIZONTAL_RATIO = 1.8f`); the PC never asks for it.
- Each gap materialises as a `◌` (`GAP_CHAR`, U+25CC, `core/src/models.rs:385`;
  mobile `OcrEngine.GAP_CHAR`) placeholder inserted right-to-left so
  detector indices stay valid, growing text/char-boxes/alternatives together
  (`with_gap_char` `:925`, `apply_blank_gaps` `:963`; mobile
  `LineResult.withGapCharAt`, `util/LineResultGap.kt:55`). Idempotent: a line
  already containing the placeholder is unchanged. The placeholder box is
  interpolated between its neighbours (`interpolate_gap_box` `:886`; mobile
  `interpolateGapBox`, `util/LineResultGap.kt:127`).

Candidates for a blank:

- The pool comes from **every** per-timestep top-K list in discovery order,
  deduped, capped at 15 (`MAX`, `src/util/gap_candidates.rs:16`;
  `generate` `:30`; mobile `GapCandidates.MAX` / `generate`,
  `util/GapCandidates.kt:19` / `:44`).
- The CharLM reorders the pool by line context (`CharLm::rank`,
  `src/util/char_lm.rs:149`; mobile `CharLm.rank`, `util/CharLm.kt:81`).
  The LM itself: packed table, orders 1–4 (`MAX_ORDER`, `:25`), binary
  search (`count` `:80`), longest-context back-off (`log_prob` `:124`).
- Empty pool falls back to **punctuation then kana**
  (`PUNCT_DEFAULTS` `、。「」…ー`, `KANA_DEFAULTS` `はのをに と`,
  `fallback` `src/util/gap_candidates.rs:62`; mobile
  `util/GapCandidates.kt:63-81`, LM-ranked per class when a model is loaded).
- Context stops at the placeholder and is clipped to `MAX_ORDER - 1` chars
  (`context_before` `:77`; mobile `contextBefore`,
  `util/GapCandidates.kt:90`).
- After a fill, **the list is retained** — a filled blank stays an override
  and keeps its candidate list (mobile `BlankGapsTest`, PC
  `filling_a_blank_keeps_its_candidate_list`).

## Mirrored Android source

- `util/GapDetector.kt` (`detect` `:80`, `thresholdFor` `:76`),
  `util/BlankGaps.kt` (`apply` `:56`, `applyIfEnabled` `:49`),
  `util/LineResultGap.kt`, `util/GapCandidates.kt`, `util/CharLm.kt`.

## Traps

- Offering candidates from only the argmax timestep: misses the pool the
  model actually ranked.
- Dropping the placeholder from the context instead of clipping there: the
  LM then scores with text from the wrong side of the gap.
- Re-sorting ties instead of keeping discovery order (`rank` keeps pool
  order on ties — pinned by `charlm-01`).
- Regenerating the list after a fill: loses the override provenance.

## Pinning tests

- PC unit: `blank_gaps_detect_wide_vertical_spacing`
  (`src/viewer.rs:3800`); `gap_candidates.rs` tests
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
- Conformance: `gap-01-discovery-order`, `gap-02-lm-reorder`,
  `gap-03-fallback-order`, `gap-04-context-clip`, `charlm-01-counts-and-rank`,
  via `gap_cases` / `char_lm_cases` (`core/src/conformance.rs:403` / `:464`).
- Mobile: `BlankGapsTest` (placeholder geometry/idempotence/fill-retention),
  `GapCandidatesTest`
  (`the_pool_keeps_kana_and_punctuation_and_deduplicates`,
  `the_model_reorders_the_pool_by_the_line_context`,
  `the_context_stops_at_the_placeholder_and_the_model_order`, …),
  `GapDetectorTest`, `LineResultGapTest`, `CharLmAssetTest`
  (`the_shipped_model_ranks_a_gap_pool`, …).
