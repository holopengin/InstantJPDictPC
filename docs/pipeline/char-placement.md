# Char placement (CAP)

## Rule

- Per-character boxes come from **CAP**, the CTC-anchored placement
  (`char_placement::place`, `core/src/char_placement.rs`; mobile
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
  `compute_char_boxes_line` (`core/src/ocr_engine.rs`), with the shipped
  chain's conventions: horizontal `(crop + edge).round()` with the line bbox's
  full cross span, vertical likewise on x.
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
- **Unit** (`core/src/char_placement.rs`): the edge-window regression
  (`last_glyph_window_reaching_the_line_end_does_not_walk_past_the_profile`),
  the guard-rail set, full-cross-axis spans in both orientations.
- **Pipeline**: `synth_char_boxes_cap_span_and_position`
  (`core/src/ocr_engine.rs`) — one box per character, cross span, frame
  containment, and synth truth-centre positions end-to-end; the `recognition`
  kind's machine-generated `char_boxes` pins (regenerate wholesale with
  `CONFORMANCE_DUMP=1` after an intentional algorithm change, spot-check the
  texts first).
- **Tier-2 (quality, spec §4–5)**: dump a `synthesize.py` corpus
  (seed defaults to 7) with
  `cargo run -p jpdict_core --example cap_dump -- <data_dir> <out.jsonl>`,
  then score it in the mobile checkout with
  `eval.py --data <data_dir> --only clean --boxes-from <out.jsonl>`. Measured
  2026-09-23 (this environment): the `imported` row equals the reference
  `proposed` row on every metric (centre err mean 1.48 px, p90 3.00 px,
  cover 0.970, tap 0.995) and passes all nine hard gates with margin.
