# Char placement (CAP)

## Rule

- Per-character boxes come from **CAP**, the CTC-anchored placement
  (`char_placement::place`, `jpdict-core/src/char_placement.rs`; mobile
  `CharPlacement.place`, adopted behind `PREF_BOX_PLACEMENT_CAP`): CTC run
  centres anchor each glyph, a robust template fit (em / X0 / letter-spacing,
  Huber-reweighted, ridge on `ls`) seeds the line, an ink-profile pass refines
  centres, and one shared final boundary pass emits boxes that **tile the
  reading axis and span the full cross axis** (horizontal boxes are
  `crop_h` tall, vertical boxes `crop_w` wide).
- Evidence comes from the decode's raw per-timestep top-K
  (`LineResult::raw_alternatives`, also fed to `re_decode_line`). A count
  mismatch or absent steps falls back to single-timestep runs at `char_cols`
  — the reference's safety net. Degenerate input (empty text, `seq_len == 0`,
  `crop < 8` skipping the ink pass, a short pixel buffer) degrades to the
  fallback, never throws; profile walks clamp at the profile length
  (spec §1.3.1's blank-lines rule).
- The integer rounding to `BoundingBox` happens once, in
  `compute_char_boxes_line` (`jpdict-core/src/ocr_engine.rs`), with the shipped
  chain's conventions: horizontal `(crop + edge).round()` with the line bbox's
  full cross span, vertical likewise on x.
- **A line's final closing punctuation takes the advance cell after its
  predecessor.** The overlay draws every glyph centred in its character box,
  so a box is the glyph's advance cell. An ink-tight detection crop ends at a
  trailing `。`'s ink, so CAP's CTC column and template put it close to the
  previous glyph (its ink is offset to the left of its em centre, and the ink
  pass can only centre the box in the gap between the two glyphs). Drawn
  centred, the dot lands on the previous character. The post-pass
  `char_boxes::place_trailing_punct_cell` runs after `place` and gives the
  last fullwidth closing mark (`。`, `、`, closing brackets) the cell that
  follows its predecessor, extending that predecessor to the shared boundary
  so the line still tiles. `place` itself stays bit-identical to the Python
  reference; the new last box can extend past the crop's reading-axis end by
  up to one cell (the crop ends at the ink, not at the cell). Lines ending in
  any other character are untouched, and the legacy chain
  (`BOX_PLACEMENT_CAP=0`) keeps its boxes.
- **Kill switch**: `BOX_PLACEMENT_CAP=0` restores the shipped
  `legacyCells → snapCells → resolveInkCollisions → uniformCells` chain
  (`compute_char_boxes`) bit-for-bit; `BOX_LAYOUT_MODE` / `BOX_UNIFORM_SIZE`
  apply to that chain again. CAP is the default.

## Mirrored Android source

- Prose home (normative): `docs/char-placement-conformance.md` in the mobile
  checkout — algorithm, guard rails (§1.3), Tier-1 fixtures (§3), Tier-2
  gates (§4–5), FORMAT/TOLERANCES additions (§6).
- Reference implementation: `tools/char_placement/place.py::
  proposed_char_boxes` (float64, normative); `CharPlacement.kt` (float32) is
  the passing Android port, mirrored by the PC Rust port.

## Traps

- Python's `round()` is half-to-even; `f64::round` rounds `x.5` away from
  zero — mirrored by `round_half_even`, or the two implementations diverge
  exactly on tie centres.
- The box blur / profile smoothing is numpy `convolve(..., "same")`:
  **zero-padded**, not edge-clamped.
- The profile walk must clamp its window at the profile length (the
  device-measured bug: the last glyph's window reaching `hi == L` overran).
- Ink thresholds are polarity-aware from the border sample (bg light → mask
  `lum < 110`, else `lum > 145`); a hand-rolled single threshold drifts on
  dark-mode crops.
- The line sweep options (`sweep_place` / `sweep_pull_back`) are off by both
  defaults and both helpers return immediately — not ported on purpose.
- Never bless this kind's expectations from a PC dump: `expect_boxes` are the
  Python reference's output, a DUMP of the port would be circular.

## Pinning tests

- **Tier-1 (parity)**: conformance kind `char_placement`, cases
  `char-placement-01..06` (six byte-shared fixtures), `box_px` override 1.5,
  `mobile_mirror: CharPlacementTest.kotlinPortMatchesPythonReference`; the
  runner also pins the Android tap contract
  (`boxesContainTheirCharactersInkCentre`, ±0.5 px) and reports Tier-2-style
  metrics per case under `CONFORMANCE_DUMP=1`. Format: `FORMAT.md`
  (`char_placement` section), tolerance authority `TOLERANCES.md`.
- **Unit** (`jpdict-core/src/char_boxes.rs`): the trailing-cell post-pass
  (`trailing_period_takes_the_cell_after_its_predecessor`,
  `trailing_period_takes_the_cell_below_in_vertical_text`,
  `trailing_kana_is_left_alone`), plus the edge-window regression
  (`char_placement::last_glyph_window_reaching_the_line_end_does_not_walk_past_the_profile`),
  the guard-rail set, full-cross-axis spans in both orientations.
- **Pipeline**: `synth_char_boxes_cap_span_and_position`
  (`jpdict-core/src/ocr_engine.rs`) — one box per character, cross span, frame
  containment (a trailing closing mark's cell may reach past an ink-tight
  frame), and synth truth-centre positions end-to-end, including a trailing
  mark's cell centre; the `recognition` kind's machine-generated `char_boxes`
  pins (regenerate with `CONFORMANCE_DUMP=1` after an intentional algorithm
  change, spot-check the texts first).
- **Tier-2 (quality, spec §4–5)**: dump a `synthesize.py` corpus
  (seed defaults to 7) with
  `cargo run -p jpdict-core --example cap_dump -- <data_dir> <out.jsonl>`,
  then score it in the mobile checkout with
  `eval.py --data <data_dir> --only clean --boxes-from <out.jsonl>`. Measured
  2026-09-23 (this environment): the `imported` row equals the reference
  `proposed` row on every metric (centre err mean 1.48 px, p90 3.00 px,
  cover 0.970, tap 0.995) and passes all nine hard gates with margin.
