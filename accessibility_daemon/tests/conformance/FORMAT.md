# Conformance corpus: case format

One corpus of pipeline cases that both codebases run from their own test
suites (ticket `pipeline-sharing/01`). Each case is a single JSON file in
`cases/` plus, for image-level cases, a synthetic input PNG in `images/`.
The PC harness is `src/conformance.rs` (runs under `cargo test`); the
Android runner is a follow-up (see the ticket comments).

## Why this directory

`accessibility_daemon/tests/conformance/` (not top-level): the cases belong
to the PC crate's test suite, cargo resolves `tests/` helpers and data by
convention, and `CARGO_MANIFEST_DIR` gives the harness a stable path to the
files at test time. The Android side reads the same files from its own
checkout; the format uses only plain data (ints, floats, strings, string
lists) so a Kotlin `kotlinx.serialization` mirror needs no Rust types.

## Envelope (every case)

```json
{
  "id": "furigana-02-receipt-line",
  "kind": "furigana",
  "description": "what pipeline semantic this pins, and why",
  "mobile_mirror": "FuriganaRuleTest.theReceiptLineSurvives (or null)",
  "tolerances": { "box_px": 2, "angle_deg": 1.0 },
  "case": { ...kind-specific... }
}
```

- `id` is the file name stem, unique. The file must be named `<id>.json`
  (enforced by the harness: a mismatch fails `every_case_has_a_runner`).
- `kind` selects the harness runner (below).
- `mobile_mirror` names the Android test that must agree, or `null` when the
  case is PC-hosted-only (reason in `description`).
- `tolerances` overrides the defaults in `TOLERANCES.md` for this case only.

## Kinds

### `detection` — image bytes in, line boxes out

```json
"case": {
  "image": "images/h-bars.png",
  "det_thresh": 0.25, "det_unclip": 0.7, "furigana_filter": false,
  "expect_lines": [
    { "box": [x, y, w, h], "vertical": false },
    ...
  ]
}
```

The harness builds the detector stand-in prob map from the image (dark pixel
`< 128` → `0.9`, else `0.0`) and runs the real post-processing chain in
pipeline order: `fit_components` → furigana/min-size/enclosing-blob filter →
straight-box merge → reading-order sort. `expect_lines` is in reading order,
so these cases pin detection boxes *and* reading order together. `box` is the
final axis-aligned `BoundingBox` (`[x, y, w, h]`, integers, compared within
`box_px`). `vertical` pins the frame's own orientation flag. No `text` field:
recognition needs the hosted inference net on each side (an accepted platform
substitution, see `TOLERANCES.md`), so line texts are pinned one level down,
at the `gap`/`kana` decode stages, from recorded decoder inputs.

### `geometry` — rotated-frame conventions

```json
"case": {
  "frames": [
    { "cx": 0, "cy": 0, "w": 200, "h": 40, "angle_deg": 162.0,
      "expect": { "vertical": false, "axis_aligned": false,
                  "norm_angle_deg": -18.0 } }
  ]
}
```

`angle_deg` is the frame's local-x angle from +x, y-down, positive =
clockwise visually. `norm_angle_deg` pins the `[-90°, 90°)` normalization
(`162° ≡ -18°`). `vertical` pins the 1.25 aspect rule, `axis_aligned` the 1°
rule plus the short-frame quantization widening. Angles compare within
`angle_deg`.

### `reading_order` — box order without pixels

```json
"case": {
  "boxes": [ { "box": [x, y, w, h], "w_local": 200, "h_local": 40 } ],
  "expect_order": [2, 0, 1]
}
```

Runs the real `sort_detected_boxes` (horizontal top-to-bottom/left-to-right,
vertical right-edge-to-left, horizontals before verticals). `w_local` /
`h_local` set the frame's own sizes that decide orientation. Indices into
`boxes`; exact match, no tolerance.

### `furigana` — ruby drop rules on box pairs

```json
"case": {
  "img": [1000, 1000],
  "pairs": [
    { "name": "receipt detail line",
      "small_raw": [l, t, r, b], "big_raw": [l, t, r, b],
      "small_un": [...], "big_un": [...],
      "orientation": "horizontal", "expect_ruby": false }
  ]
}
```

Rects are `[left, top, right, bottom]`. `raw` = contour geometry,
`un` = unclipped geometry. `expect_ruby` pins `is_ruby_horizontal` /
`is_ruby_vertical` exactly (pure predicates, no tolerance).

### `gap` — blank-gap candidates from decoder evidence

```json
"case": {
  "alternatives": [[["、", 0.9], ["の", 0.5]], ...],
  "context": ["私"], "limit": 15,
  "lm": { "mass": 500, "entries": [["私", 100], ["私の", 40]] },
  "expect": ["の", ...], "expect_fallback_first": "、"
}
```

`alternatives` is the line's per-timestep top-K (shape evidence proposes).
`lm` is an optional toy CharLm table in the shipped packed-format semantics
(the text prior disposes); `null` pins discovery order. `expect` is the exact
candidate list. `expect_fallback_first` optionally pins the punctuation-first
fallback class order.

### `char_lm` — the text prior itself

```json
"case": {
  "mass": 500, "entries": [["私", 100], ["私の", 40]],
  "counts": [{ "ngram": ["私", "の"], "expect": 40 }],
  "ranks": [{ "context": ["私"], "pool": ["を", "の"], "expect": ["の", "を"] }]
}
```

Exact match: counts, and rank order (ties keep pool order).

### `kana` — small/large correction from recorded logits

```json
"case": {
  "encoder": [{ "text": "かれはいっとう。", "index": 4, "expect_window_head": [0, 0, 0, 0, 227, 129, 139, 0] }],
  "policy": { "lines": ["かっき"], "logits": [-8.0],
              "epsilon": 0.01, "expect_text": ["かっき"], "expect_flips": 0 }
}
```

`encoder` pins the 40-int window bytes (exact) against the published
validation vectors. `policy` runs `correct_lines` with the logits injected
(no native net): `expect_text` exact, `expect_flips` the flip count, and each
flip's `from`/`to` when present.

### `dictionary` — Jitendex structured-content formatting

```json
"case": {
  "term": "お前", "reading": "おまえ",
  "definitions_ref": "../../data/jitendex/entries.json",
  "expect": {
    "headwords": ["お前"], "readings": ["おまえ"],
    "min_sense_groups": 1, "example_ja": "…", "example_en": "…"
  }
}
```

The definitions payload is read from the pinned `tests/data/jitendex/`
fixture (shared with the Android `JitendexStructuredContentTest` inputs), not
duplicated. Expectations pin the formatted DOM shape: headword/reading lists
exact, sense-group counts as lower bounds (dictionaries grow), example
Japanese/English split as substring match.

## Adding a case (the parity-bug rule)

A parity bug fix adds a conformance case: write the JSON, run
`cargo test conformance`, paste the actuals only after checking they are the
*correct* semantics (not just what the code happens to emit), and name the
`mobile_mirror` or record why there is none.
