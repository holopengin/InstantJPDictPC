# Dictionary structured content

## Rule (the #88 rendering model)

Jitendex packs every sense into one row's glossary; the walker
(`format_dictionary_results`, `core/src/overlay_state.rs:893`) renders:

- One glossary per repeated headword form — a repeated reading glossary is
  not rendered twice; identical headword rows render senses once.
- Entries split per dictionary with a source label (per-dictionary grouping).
- A row's glossary split for numbering: senses numbered across the entry,
  never all under "1"; a lone sense is not echoed into the header.
- Ruby runs are one word/sentence with separators preserved; a malformed
  ruby node still renders its content.
- Example boxes split into Japanese and English parts (`:1459`); extra-info
  boxes, sense notes and source lines are blocks of their own (never
  comma-joined into neighbours); cross-references render as text without
  comma noise; unknown tags render content rather than dropping it.
- Lookup splices redirect targets below their source; matches span line
  boundaries; dictionary enabled flag honoured; term-meta pitch banks
  imported idempotently; pitch line renders only with real data and ships off
  (toggle ports mobile parity #43/#86).
- Deinflected matches carry their chain into the chain row (see
  [deinflection](deinflection.md)); plain terms render exactly as before.

## Ticket-06 departure: body ruby is white regular (DELIBERATE)

Body (mini) ruby is **white regular**, term ruby stays **bold cyan**
(`jpdict_core::ruby_style::ruby_style`, `core/src/ruby_style.rs`; one shared palette between
aligned `ruby_view` `:2958` and fallback `full_ruby_view` `:3010` paths).
Mobile changed together: `RubyBaseStyle.forMini`
(`RubyBaseStyle.kt:36`), shipped as Android commit `1719405`; the PC side as
`2f5dd62`, graduated to the corpus by `c7dc78b`.
**Do NOT "parity-fix" body ruby back toward cyan+bold** — that reintroduces
the ticket-06 symptom by decision on both sides. The `ruby_style`
conformance kind states the departure inline in `FORMAT.md` for the same
reason.

## Mirrored Android source

- Definition walker / `OcrOverlayView` definition rendering;
  `RubyBaseStyle.kt`; `DictionaryRedirects`, pitch (`PitchAccentLine`).

## Traps

- Numbering every sense "1" (rendering rows instead of the split glossary).
- Dropping unknown tags/nodes instead of rendering their content (blanks a
  whole sense on dictionary updates).
- Painting body ruby from the term display style (the ticket-06 symptom).

## Pinning tests

- PC unit (`core/src/overlay_state.rs` tests):
  `a_repeated_reading_glossary_is_not_rendered_twice`,
  `identical_headword_rows_render_senses_once`,
  `a_lone_sense_is_not_echoed_into_the_header`,
  `senses_are_numbered_across_a_jitendex_entry_not_all_under_1`,
  `entries_are_split_per_dictionary_with_a_source_label`,
  `a_jitendex_example_splits_into_japanese_and_english_parts`,
  `a_malformed_ruby_still_renders_its_content`,
  `an_unknown_tag_renders_its_content_rather_than_dropping_it`,
  `lookup_splices_redirect_targets_below_their_source`,
  `matched_term_spans_line_boundaries`,
  `deinflected_matches_carry_their_chain`; viewer
  (`body_ruby_uses_body_typeface_not_term_display` `:4359`,
  `term_ruby_keeps_full_size_term_display` `:4371`,
  `pitch_line_renders_only_when_the_switch_is_on`,
  `pitch_line_ships_off`).
- Conformance: `dictionary-01-jitendex` via `dictionary_cases`
  (`core/src/conformance.rs:887`); `ruby-style-01-modes` via `ruby_style_cases`
  (`:822`).
- Mobile: `JitendexStructuredContentTest`, `FormatPerDictTest`,
  `DictionaryRedirectsTest`, `RubyBaseStyleTest` (`bodyRuby_isWhiteRegular`,
  `termRuby_isBoldCyan`, `parsedDefinitionRuby_alwaysTakesBodyPath`).
