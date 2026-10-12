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

## Panel face: name the bundled family, never a generic one

Every panel widget renders in the **bundled** face, and must ask iced for it
*by family name*. `init_panel_font` captures the resolved face's family at
startup (`loaded_face.family_name()` — the same face the overlay glyph cache
rasterizes and the same bytes registered with iced); `bold_font` (the
term-display weight, used by `headword_block`, `tag_chip`, `ruby_view`,
`full_ruby_view`) is that family at `Weight::Bold`.

Why the family name matters, because iced gives no way to forbid the system
database: `FontSystem::new_with_fonts` calls `load_system_fonts()`, so every
font installed on the machine is in the DB. A generic
`Family::SansSerif` request then resolves to the family cosmic-text
configured ("Open Sans"), and the kanji come from cosmic-text's per-glyph
fallback list — `han_unification(locale)`, which is `Noto Sans CJK SC`
unless the process locale is exactly `ja`. That is the "headwords render in
Chinese/ancient forms while the rest of the UI is modern Japanese" symptom.
With only the bundled Regular registered, a `Weight::Bold` request *within*
the bundled family lands on that Regular — the weight is aspirational (see
the ticket-06 note), but the glyph forms are always the bundled Japanese
ones.

Residual, accepted: (a) cosmic-text's per-glyph fallback still applies to
codepoints the bundled face lacks (e.g. the settings window's `✕`), and
(b) a *system* font installed under the literal family name `Noto Sans JP` /
`Noto Serif JP` could tie-break ahead of the bundled face, since system faces
are loaded first and `fontdb::find_best_match` breaks ties by first-seen.
Most distributions expose CJK as `Noto Sans CJK JP`, so this rarely bites.

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
- Requesting a generic family (or `IcedFont::default()`, which is
  `Family::SansSerif`) for panel text: it silently renders in system fonts,
  and CJK in Chinese forms. Name the bundled family (see above).
- **Splitting a tag field with `str::split_whitespace`.** The field is ASCII
  space-joined, but an individual tag *name* carries NBSP internally —
  `old kanji form` is one tag, per Jitendex's `tag_bank_1.json`, where 7 of
  8 canonical names are shaped that way. NBSP is Unicode whitespace, so
  generic splitting shreds the name into `old` / `kanji` / `form` and it can
  never match the tag table. Use `yomitan_parse::split_space_separated`.
- **Rendering a deinflection pair.** A definitions array may carry
  `[uninflectedTerm, [inflectionRule, …]]` beside its definitions — 131,789
  such rows in Jitendex rev `2026.02.05`, all on redirects. It is stored
  verbatim, so a generic walker renders the raw metadata (`redirected from X`)
  as if it were a gloss. Filter it with `definition_format`'s
  `is_deinflection_pair`.

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
  `pitch_line_ships_off`); panel face (`src/viewer.rs`
  `panel_fonts_never_request_a_generic_family`).
- Conformance: `dictionary-01-jitendex`, `dictionary-02-tag-names-with-nbsp`,
  `dictionary-03-redirect-deinflection-pair` via `dictionary_cases`
  (`core/src/conformance.rs:887`); `ruby-style-01-modes` via `ruby_style_cases`
  (`:822`).
- Mobile: `JitendexStructuredContentTest`, `FormatPerDictTest`,
  `DictionaryRedirectsTest`, `RubyBaseStyleTest` (`bodyRuby_isWhiteRegular`,
  `termRuby_isBoldCyan`, `parsedDefinitionRuby_alwaysTakesBodyPath`).
