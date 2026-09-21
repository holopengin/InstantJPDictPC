# CTC decode and alternatives

## Rule

- CTC blank (class 0) is stored as the **ideographic space U+3000** so the
  walk can tell blank from real space (`decode_char`, `core/src/ppocr.rs:114`;
  mobile same convention into `ctcDecodeTopK`, `OcrEngine.kt:1907`).
- Every timestep caches its full top-K list (`raw_alternatives`,
  `PpocrResult`, `core/src/ppocr.rs:53`; built per timestep at `:320-380`;
  mobile passes the same lists to `ctcDecodeTopK`).
- Re-decode rebuilds text, char columns and per-character alternatives from
  the cache **without re-running the model**
  (`re_decode_raw_alternatives`, `:949`; entry 0 is the argmax, U+3000
  resets the collapse). PC entry point: `DetectedAnnotation::re_decode_line`
  (`core/src/ocr_engine.rs:1136`), which recomputes boxes in the emit frame's
  geometry and keeps old boxes when `seq_len_total == 0` (mobile's
  `cropW == 0` fallback); mobile `reDecodeLineResult` (`OcrEngine.kt:2410`).
- Top-K ordering matches the Java priority queue; top-K failure falls back
  to full logits (`java_topk_order` `:203`, `top15_alternatives` `:224`).
- Alternatives panel assembly (`assemble`, `src/util/oov_suggestions.rs:53`;
  mobile `OovSuggestions.assemble`, `util/OovSuggestions.kt:52`), in order:
  1. the head's own ranking first, unchanged (preferred whenever right);
  2. the current character if the head omitted it (override provenance);
  3. component neighbours by descending IDF mass, only below the
     `MIN_IDF_FRACTION = 0.7` tier cut and only when the character has
     discriminating components, capped at 15 (`MAX_COMPONENT_CANDIDATES`);
  4. obsolete variant forms of the character itself (offered, never applied),
     capped at 15 (`MAX_VARIANT_CANDIDATES`).
  Duplicates drop across groups. Panel tints non-head entries by source
  (`Source::Head/Components/Variant/Lm`; blank-path LM entries tag `Lm`
  when a model is loaded, `Head` in discovery order).

## Mirrored Android source

- `OcrEngine.kt` (`ctcDecode` `:1839`, `ctcDecodeTopK` `:1907`,
  `reDecodeLineResult` `:2410`),
  `util/OovSuggestions.kt` (`MIN_IDF_FRACTION` `:27`, caps `:35-36`),
  `util/OovCandidates.kt` (`neighbours_of`, majority/idf),
  `util/ComponentTable.kt`.

## Traps

- Decoding blank to a real space: merges words the collapse must keep apart.
- Re-running inference for a re-decode (e.g. after a settings change):
  wasteful and nondeterministic across hosts — use the cache.
- Applying variant forms automatically: they are offered, never applied.
- Letting component neighbours outrank the head list: the head is preferred
  whenever right; neighbours append after it
  (`component_neighbour_is_appended_after_the_head_list`).

## Pinning tests

- PC unit (`core/src/ppocr.rs` tests):
  `topk_decode_pins_text_cols_and_raw_shape` (blank cached as U+3000),
  `re_decode_pins_fractional_columns`, `re_decode_walk_matches_mobile`,
  `re_decode_normalises_vertical_punctuation`,
  `topk_order_matches_java_priority_queue`,
  `topk_failure_falls_back_to_full_logits`,
  `synth_topk_and_full_logits_agree`; OOV
  (`neighbours_of_emitted_char_find_the_measured_substitution`,
  `neighbours_are_sorted_by_evidence_and_never_contain_the_emitted_character`,
  `component_neighbour_is_appended_after_the_head_list`,
  `head_order_is_preserved_and_a_repeated_candidate_is_not_appended_twice`,
  `the_variant_group_walks_the_form_space_from_the_current_selection`,
  `component_group_is_capped`,
  `the_bundled_asset_still_has_12156_entries`).
- Conformance: the `gap` kind pins pool discovery order from recorded
  top-K; the `recognition` kind pins end-to-end texts (`recognition_cases`,
  `core/src/conformance.rs:648`).
- Mobile: `BlankAlternativesTest`, `OovCandidatesTest`,
  `OovSuggestionsTest`, `ComponentTableTest`, `RotatedLineResultTest`.
