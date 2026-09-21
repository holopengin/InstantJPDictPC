# Glyph sizing and placement

## Rule

- Line size is mobile's `glyphSizePx`: the tallest char box by default —
  detector box height for horizontal, 1em cell for vertical — and for a
  **rotated** line the upright frame's **cross axis** (`isVertical ? w : h`),
  because char boxes of rotated cells are AABBs and would oversize the glyphs
  (`line_glyph_px`, `src/viewer.rs:720`; mobile
  `OcrOverlayStateController.glyphSizePx`, `:58`, consumed at
  `OcrOverlayView.kt:997`). Zero (no measurable box) draws nothing, like
  mobile skipping the line.
- Raster size is that × 0.90 in source pixels, scaled by the content
  transform (`TEXT_SIZE_RATIO`; mobile `fixedSize * 0.90`).
- Ink centring is per orientation (`glyph_ink_origin`, `:672`): vertical
  centres the char's own ink with x on the reference column; horizontal
  centres x with y on the reference ink centre (natural baseline vs `あ`).
- Per-glyph shrink-to-box measures **along the reading axis** (height for
  vertical, width for horizontal) against char box × 0.92, doubled for
  halfwidth ink whose advance is 0.5em but whose face may overflow it
  (`glyph_fit_scale`, `:816`, `GLYPH_FIT_RATIO` `:197`; mobile
  `ASCII_GLYPH_SCALE = 0.9f`, `LineOverlayView.kt:62`, `:185`). One factor
  per glyph; there is no line-wide cross-axis cap.
- Vertical glyphs resolve through the font's GSUB vert/vrt2
  (`gsub_vert_glyph`, `:547`); zero-ink glyphs are skipped.
- Neighbor/alternatives chips take vertical presentation forms only in
  landscape (`chip_text`; mobile `OcrOverlayView` chip rule).

## Mirrored Android source

- `OcrOverlayStateController.kt:40-58`, `LineOverlayView.kt:127` (tilt),
  `OcrOverlayView.kt:997`.

## Traps

- Sizing rotated lines from the reading length (char-box AABB long side):
  oversizes every glyph on tilted lines — always the cross axis.
- A line-wide cross-axis cap: mobile scales per glyph on the reading axis
  only; a cap clips narrow punctuation.
- Forgetting the halfwidth doubling: halfwidth faces overflow their 0.5em
  advance and get shrunk to nothing.
- Sizing from a content scale computed before the image size is known
  (the stale-scale pre-warm freeze: `content_scale` returns `None` until
  `img_w`/`img_h` are real).

## Pinning tests

- PC unit (`src/viewer.rs` tests):
  `line_text_size_uses_box_height_like_mobile` (`:4008`),
  `glyph_fit_scales_only_the_overflowing_axis` (`:4076`),
  `glyph_tilt_follows_the_reading_axis` (`:4097`),
  `vertical_glyphs_centre_their_own_ink` (`:4124`),
  `vertical_glyphs_come_from_gsub` (`:4164`),
  `glyph_ink_origin_is_identity_for_the_reference_glyph` (`:3771`),
  `blank_glyphs_report_no_ink` (`:3876`),
  `content_scale_needs_a_real_image_size`.
- No dedicated conformance kind: sizing is rasterizer-dependent by design
  (TOLERANCES.md §1 — Skia vs fontdue); the corpus pins geometry inputs,
  never rendered pixels.
- Mobile: `RotatedLineResultTest`, `UniformEmTest`, `OverlayFontTest`.
