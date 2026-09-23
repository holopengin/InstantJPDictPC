# Lookup normalization (width → variant fold → combining characters)

## Rule

`normalize` is the query-side pipeline that turns raw OCR text (or a manual
override) into the form dictionary lookup keys on. Three stages run in a fixed
order, and the order is part of the contract:

1. **Width conversion** — `convert_width`
   (`core/src/util/japanese.rs`); mobile `JapaneseUtil.convertWidth`
   (`util/JapaneseUtil.kt`). Halfwidth katakana widen to fullwidth, including
   the voiced (`ｶﾞ` → `ガ`) and semi-voiced (`ﾊﾟ` → `パ`) digraphs; fullwidth
   ASCII (`FF01`–`FF5E`) folds to ASCII; the ideographic space folds to a
   space; the vertical presentation forms `︙` (U+FE19) and `︰` (U+FE30) fold
   back to the horizontal `…`/`‥`, so a vertical line's lookup key matches
   horizontal dictionary text.
2. **Lookup-variant fold** — `fold_lookup_variants`; mobile
   `foldLookupVariants`. See below.
3. **Combining-character normalization** — `normalize_combining_characters`;
   mobile `normalizeCombiningCharacters`. The 25 decomposed hiragana
   dakuten/handakuten pairs compose (`か` + U+3099 → `が`, `は` + U+309A →
   `ぱ`). This is not a general NFC pass: the katakana pairs and the う pair
   stay decomposed, and the stage runs *after* the fold, so a mark following a
   combining dakuten sees the mark rather than the composed kana
   (`か` U+3099 `ゞ` → `がゞ`).

Query-side only: callers keep the raw text for display and keep using the raw
prefix length for match alignment, so a fold may change the query's length
without changing what is shown or which prefix a match corresponds to.

### The variant fold

The fold walks the string left to right and is a single character pass.

**Iteration marks.** `ゝ`/`ヽ` repeat the last character already emitted when it
is kana of the matching script (hiragana `U+3041`–`U+3096`, katakana
`U+30A1`–`U+30F6`); `ゞ`/`ヾ` voice the repeat through `HIRAGANA_VOICED` /
`KATAKANA_VOICED` (21 pairs each: か→が … う→ゔ, カ→ガ … ウ→ヴ) and fall back to
a plain repeat when the kana has no voiced form or is already voiced
(`まゞ` → `まま`, `がゞ` → `がが`, `ンヽ` → `ンン`). A mark whose preceding
character is not kana of the matching script — line-initial, after a kanji, or
after punctuation — is left as-is rather than folded into a guess, and marks do
not cross scripts (`カゝ`, `あヽ` stay). A run of marks repeats the expanded run
(`こゝゝ` → `こここ`). `々`/`〻` are **not** iteration marks here: dictionary
headwords contain them (`日々`), so expanding would lose matches.

**`LOOKUP_VARIANT_MAP`** (192 entries: the curated 27 plus the measured 165)
folds:

- Roman numerals — `Ⅰ`–`Ⅻ` and the small forms expand to their ASCII spelling
  (`Ⅶ` → `VII`, `ⅸ` → `ix`; NFKC behaviour), so a numeral inside a word folds
  too (`第Ⅻ章` → `第XII章`).
- The compatibility form `℃` → `°C`.
- Obsolete kana the recogniser can emit — `ゑ` → `え`, `ヰ` → `イ`.
- Chinese-only forms the recogniser emits in place of the Japanese one —
  `况` → `況`, `查` → `査`. Only the emittable half of a pair is in the table:
  `调` is not folded, so `调查④` → `调査④`.

**`MEASURED_VARIANT_FOLD`** (165 single-character pairs) is the Unihan-derived
half. Direction is variant → canonical, where canonical is the side the
recogniser's vocabulary can emit and variant is the side it cannot. Two
measured guards select the subset: a pair ships only if its variant side
actually occurs in real text, and only when its canonical side occurs at least
as often as its variant in the same corpus — Unihan's `kSemanticVariant` is
loose and also lists pairs whose "canonical" side is the rarer form, and a fold
*replaces* the lookup key, so folding those would rewrite a query that used to
resolve into one that does not. That guard keeps `壜` (the commoner side) and
`坂` (the corpus prefers 坂, as in 大阪) out of the table, and where several
canonicals exist it picks the corpus-dominant one (`葢` → `蓋`, `悋` → `吝`,
`冫` → `氷`). No canonical is itself a key, so one pass reaches the terminal
form (`冩` → `写`). All pairs are single characters, so unlike the Roman
numerals they never change query length.

**Drift guard.** Every measured pair must exist in
`assets/variants/kanji_variants.txt` in the same direction, and no canonical may
be a fold key; both sides pin this in their own unit test.

## Mirrored Android source

- `util/JapaneseUtil.kt` — `normalize`, `convertWidth`, `foldLookupVariants`,
  `normalizeCombiningCharacters`, and the four tables (`HIRAGANA_VOICED`,
  `KATAKANA_VOICED`, `LOOKUP_VARIANT_MAP`, `MEASURED_VARIANT_FOLD`).
- `assets/variants/kanji_variants.txt` — the variant-table asset the drift
  guard reads (byte-identical on both sides).

## Traps

- Running the fold before width conversion: `ｶヽ` would leave the mark with
  nothing to repeat; width first gives `カカ`.
- Running combining normalization before the fold: `か` U+3099 `ゞ` would read
  `がが` instead of `がゞ`.
- Expanding `々`/`〻`, which breaks headwords like `日々`.
- Folding backwards (canonical → variant) or chaining one step short
  (`冩` must reach `写` in one pass, never stop at `寫`).
- Treating `normalize` as NFC: katakana (`カ` U+3099) and う U+3099 stay
  decomposed.
- Re-deriving the measured pairs by hand instead of porting the table; the
  subset and its direction are corpus measurements.
- Blessing an expectation because the code emits it: `normalize` output is
  exact, no tolerance, and the corpus expectations are the composed result.

## Pinning tests

- PC unit (`core/src/util/japanese.rs`): `fold_expands_voiced_iteration_mark`,
  `fold_leaves_mark_that_cannot_be_repeated`,
  `fold_folds_compatibility_and_chinese_only_forms`,
  `fold_folds_measured_unihan_variants`,
  `unihan_fold_picks_the_corpus_dominant_canonical`,
  `unihan_fold_does_not_run_backwards`,
  `measured_variant_fold_matches_the_committed_asset`,
  `normalize_applies_the_fold`, `normalize_widens_halfwidth_kana_before_folding`,
  `normalize_stages_run_in_pipeline_order`,
  `normalize_existing_behaviour_is_unchanged`.
- Conformance: `normalize-01-variant-fold` via `normalize_cases`
  (`core/src/conformance.rs`); schema in
  `accessibility_daemon/tests/conformance/FORMAT.md`.
- Mobile: `JapaneseUtilVariantFoldTest` (20 tests).
