# Glyph sizing and placement

## Rule

- The line size basis is mobile's `glyphSizePx`: the tallest char box by
  default — detector box height for horizontal, 1em cell for vertical — and
  for a **rotated** line the upright frame's **cross axis**
  (`isVertical ? w : h`), because char boxes of rotated cells are AABBs and
  would oversize the glyphs (`line_glyph_px`, `src/viewer.rs`; mobile
  `OcrOverlayStateController.glyphSizePx`, `:58`, consumed at
  `OcrOverlayView.kt:997`). Zero (no measurable box) draws nothing, like
  mobile skipping the line.
- The em is then sized so the glyph **ink fills that basis** — see the
  departure below (`line_text_px`, `GLYPH_FILL_RATIO`,
  `GlyphCache::ink_height_ratio`).
- Ink centring is per orientation (`glyph_ink_origin`): vertical centres the
  char's own ink with x on the reference column; horizontal centres x with y
  on the reference ink centre (natural baseline vs `あ`).
- Per-glyph shrink-to-box measures **along the reading axis** (height for
  vertical, width for horizontal) against char box × 0.92, doubled for
  halfwidth ink whose advance is 0.5em but whose face may overflow it
  (`glyph_fit_scale`, `GLYPH_FIT_RATIO`; mobile
  `ASCII_GLYPH_SCALE = 0.9f`, `LineOverlayView.kt:62`, `:185`). One factor
  per glyph; there is no line-wide cross-axis cap.
- Vertical glyphs resolve through the font's GSUB vert/vrt2
  (`gsub_vert_glyph`); zero-ink glyphs are skipped.
- A selected word is the **same glyph in yellow** (`OVERLAY_HL`), never a
  bold one. The overlay bundles no bold face and synthesizes no bold weight.
- Neighbor/alternatives chips take vertical presentation forms only in
  landscape (`chip_text`; mobile `OcrOverlayView` chip rule).

## Departures from mobile (DELIBERATE, maintainer decision)

Two, both PC-side appearance decisions. Do NOT "parity-fix" either back.

1. **Ink fills the box (`GLYPH_FILL_RATIO` / `ink_height_ratio`).** Mobile
   sets `textSize = fixedSize * 0.90`, which sizes the *em square*: a CJK face
   inks only ~0.85 em (`あ` measures 0.86 of the em in Noto Sans JP, 0.84 in
   Noto Serif JP), so every line drew at `0.90 × 0.86 ≈ 77%` of its box height
   — the visible padding above and below each line. The overlay instead
   measures the resolved face's ink ratio once at load and divides the box by
   it (`0.98 / 0.86 ≈ 1.14` × box), so the ink lands at ~98% of the box: the
   glyph is ~1.27× taller than before. The basis itself (`glyphSizePx`) is
   still mobile's; only the em→ink conversion differs. The correction is
   neutral where the per-glyph reading-axis fit binds instead (a char box
   much narrower than the glyph aspect): both old and new then land on
   `0.92 × box width`, so this can only help, never hurt. Pinned by
   `ink_ratio_is_measured_from_the_reference_glyph` and
   `ink_correction_grows_the_glyph_past_the_old_em_scaling`.
2. **Highlights are a tint, not a weight.** Mobile fake-bolds its highlighted
   glyphs (`paint.isFakeBoldText`; the PC port first bundled a sans bold
   companion face, then synthesized a stroke for faces without one). The
   selected word now turns yellow on the same regular face, so
   `fonts/NotoSansJP-Bold.ttf`, `overlay_font::find_bold_font_path` and the
   synthetic-embolden helpers are gone. Pinned by
   `highlight_only_changes_the_tint_not_the_weight`.

## Mirrored Android source

- `OcrOverlayStateController.kt:40-58`, `LineOverlayView.kt:127` (tilt),
  `OcrOverlayView.kt:997`.

## Traps

- Sizing rotated lines from the reading length (char-box AABB long side):
  oversizes every glyph on tilted lines — always the cross axis.
- Sizing the em instead of the ink: whatever constant multiplies the box
  (`×0.90`) leaves the face's em shortfall as padding. The correction has to
  divide by the *measured* ink ratio — a hardcoded constant is wrong for the
  serif face, which inks a little less than the sans one.
- A line-wide cross-axis cap: mobile scales per glyph on the reading axis
  only; a cap clips narrow punctuation.
- Forgetting the halfwidth doubling: halfwidth faces overflow their 0.5em
  advance and get shrunk to nothing.
- Sizing from a content scale computed before the image size is known
  (the stale-scale pre-warm freeze: `content_scale` returns `None` until
  `img_w`/`img_h` are real).
- Bolder the highlight because that is what mobile does: the weight is the
  thing the departure removed, and re-adding it needs the bundled bold face
  back.

## Pinning tests

- PC unit (`src/viewer.rs` tests):
  `line_text_size_fills_the_box_with_the_glyph_ink`,
  `ink_correction_grows_the_glyph_past_the_old_em_scaling`,
  `ink_ratio_is_measured_from_the_reference_glyph`,
  `highlight_only_changes_the_tint_not_the_weight`,
  `glyph_fit_scales_only_the_overflowing_axis`,
  `glyph_tilt_follows_the_reading_axis`,
  `vertical_glyphs_centre_their_own_ink`,
  `vertical_glyphs_come_from_gsub`,
  `glyph_ink_origin_is_identity_for_the_reference_glyph`,
  `blank_glyphs_report_no_ink`,
  `content_scale_needs_a_real_image_size`.
- No dedicated conformance kind: sizing and weight are rasterizer-dependent
  by design (TOLERANCES.md §1 — Skia vs fontdue, and neither Android's paint
  nor iced's text is hostable in the harness); the corpus pins geometry
  inputs, never rendered pixels or glyph weights.
- Mobile: `RotatedLineResultTest`, `UniformEmTest`, `OverlayFontTest`.