# Furigana rules

## Rule

Both `is_ruby_vertical` (`src/furigana.rs:52`) and `is_ruby_horizontal`
(`:96`) require a size difference in **both dimensions** (mobile #99):

- Vertical: small height < 30% of big height (`SIZE_RATIO`, `:19`),
  small width < 65% of big unclipped width (`WIDTH_RATIO`, `:29`).
- Horizontal: small height < 75% of big raw height (`THIN_RATIO`, `:22`),
  small unclipped height < 85% of big (`HSHORT_RATIO`, `:26`), **and**
  small unclipped width < 65% of big (`HWIDTH_RATIO`, `:35`).
- Shared: big box ≥ 20% of the image side (`BIG_MIN_FRAC`, `:47`; ruby hugs
  full-size body text, not logo blocks), small box < 12% of the image side
  (`MAX_FRAC`, `:44`), gap ≤ 50% of the big short side (`GAP_RATIO`, `:39`),
  overlap ≥ 50% of the small long side (`OVERLAP_RATIO`, `:41`).
- **Raw vs unclipped split**: size/centre/overlap/above-ness run on RAW
  contour geometry (unclip padding fabricates overlap for stacked fragments;
  ~18px mutual encroachment flips genuinely-above ruby to overlapping);
  only the gap and short-side tests use UNCLIPPED boxes (raw gutters are real
  pixels; unclip closes them to ruby distance).
- Vertical-only centre rule: the small centre must lie OUTSIDE the big box's
  x-range (stacked column fragments share its x-range).
- Horizontal above-ness on raw geometry with +2px slack
  (`s_raw.bottom() > b_raw.top() + 2` fails).
- The rule is off by default behind a settings switch (mobile #100); the PC
  `detection`/`recognition` cases state `furigana_filter` explicitly.

## Mirrored Android source

- `FuriganaRule.kt` (`SIZE_RATIO` `:21`, `HWIDTH_RATIO` `:35`,
  `GAP_RATIO` `:37`, `OVERLAP_RATIO` `:39`, `MAX_FRAC` `:42`,
  `isRubyVertical` `:53`, `isRubyHorizontal` `:73`).
- Caller-side orientation gating lives in `OcrEngine.filter_furigana`
  (PC: `ocr_engine`'s `filter_furigana` owns the index-aligned pass).

## Traps

- Thinness-alone horizontally (pre-#99): ate real short lines — receipt
  detail text under a heading, nearly as wide as its line — as ruby.
- Running everything on unclipped boxes: unclip padding fabricates overlap
  between stacked column fragments, hiding real lines inside big ones.
- Running everything on raw boxes: raw gutters read as ruby-distance gaps.
- Probing `OVERLAP_RATIO` with fully-contained pairs: no effect; the
  `furigana-05-overlap-boundary` case (strip overlapping exactly half its
  width) exists because the first perturbation probe was insensitive.

## Pinning tests

- PC unit (`src/furigana.rs` tests):
  `thin_ruby_strip_above_its_line_is_ruby`,
  `thin_but_wide_real_line_is_not_ruby` (the #99 receipt regression),
  `narrow_short_column_beside_its_column_is_ruby`,
  `full_width_short_column_is_not_ruby`.
- Conformance: `furigana-01-horizontal-ruby`,
  `furigana-02-receipt-line`, `furigana-03-vertical-ruby`,
  `furigana-04-vertical-fullwidth`, `furigana-05-overlap-boundary`, via
  `furigana_cases` (`src/conformance.rs:376`).
- Mobile: `FuriganaRuleTest` (`thinRubyStripAboveItsLine_isRuby`,
  `thinButWideRealLine_isNotRuby`, `narrowShortColumnBesideItsColumn_isRuby`,
  `fullWidthShortColumn_isNotRuby`); `ConformanceCorpusTest.furiganaCases`
  (unmerged branch). Note: case JSON `mobile_mirror` names use a
  no-underscore form; the real Kotlin names above are authoritative.
