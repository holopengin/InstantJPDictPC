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

### `recognition` — vendored photo/screenshot in, line texts out

```json
"case": {
  "image": "images/recognition/phone-photo.jpg",
  "det_thresh": 0.25, "det_unclip": 0.7, "furigana_filter": false,
  "recognition_mode": "both",
  "expect_lines": [
    { "i": 0, "box": [x, y, w, h], "vertical": false,
      "text": "…", "char_boxes": [[x, y, w, h], ...] },
    ...
  ]
}
```

The harness runs the full hosted pipeline in pipeline order —
`detect_lines` (real ncnn detector + merge + reading-order sort) →
`recognize_boxes_collect` (real ncnn recognizer + kana-size correction) —
with the case's `det_thresh` / `det_unclip` / `furigana_filter`, never the
platform defaults, and `recognition_mode` `both`. `expect_lines` is in
detection (reading) order. `i` is the detection box index: boxes the crop
stage skips (un-croppable quads) have no entry, and the pinned `i` fails
loudly if that set ever changes. `box` is the final axis-aligned
`BoundingBox` (compared within `box_px`); `vertical` is the recognized
line's own orientation flag (exact); `text` is the recognized line text
(exact, `null` when the box yields no line); `char_boxes` pins every
character box in order (count exact, each within `box_px`). Rotated quads
are not pinned per-line here (`quad` is `None` for these fixtures);
rotated-frame conventions stay covered by the `geometry` kind.

Regenerating: `CONFORMANCE_DUMP=1 cargo test recognition_cases` prints one
`DUMP-JSON <id> {...}` record per recognized line for every case without
failing, so a single run refreshes the whole kind — paste the records into
the case files only after spot-checking the texts against the images
(mis-recognition is a finding to record in the ticket, never something to
hand-edit away). Plain `cargo test` enforces every pin. This is the one
kind whose expectations are machine-generated wholesale; the other kinds
keep the dump-then-verify-one-case pattern from the parity-bug rule below.

### `deinflection` — surface form in, dictionary term + reason labels out

```json
"case": {
  "surface": "食べた",
  "expect_term": "食べる",
  "expect_reasons": ["past"]
}
```

The harness loads the shipped `assets/deinflect.json` (the same file the
app and the Android asset copy use — verified byte-identical at graduation)
and runs the real `Deinflector::deinflect`. Derivations are deduplicated by
term (first derivation wins), so terms are unique per surface: the runner
finds `expect_term` and compares its reasons exactly against
`expect_reasons` (ordered, outermost step first). The group key IS the
reason (`past`, `-te`, …), filled in at load from the map key on both sides
— a loader that drops the keys yields kana fragments or empty lists and
fails these cases. The no-op case pins a dictionary-form surface whose
identity candidate carries no reasons (no chain, no viewer row), so direct
matches render exactly as before.

Regenerating: `CONFORMANCE_DUMP=1 cargo test deinflection_cases` prints the
candidate list per surface (`DUMP <id> term=… reasons=…`, first 25) — paste
values only after checking they are the *correct* grammar (a wrong rule in
`deinflect.json` is a finding, never something to bless by copying).

### `ruby_style` — ruby base treatment per display mode (labels, not pixels)

```json
"case": {
  "modes": [
    { "mode": "body", "base": "white", "bold": false },
    { "mode": "term", "base": "cyan", "bold": true }
  ]
}
```

`mode` selects the renderer input both harnesses share without a UI
framework: `body` is PC `OcrViewer::ruby_style(is_mini = true)` /
Android `RubyBaseStyle::forMini(true)` — every ruby node `inline_line`
and `renderDefinition` build; `term` is the `false` path both headword
flows use. `base` is a color label (`white` = `Color::WHITE` /
`0xFFFFFFFF`, `cyan` = `(0, 1, 1)` / `0xFF00FFFF`) so a recolor fails with
the values attached; anything unmapped fails the runner loudly instead of
drifting. `bold` pins the weight decision.

Deliberately NOT pinned here: rendered pixels (painting an iced `Text` or
a `TextView` needs the UI framework — Robolectric is not an Android test
dependency), the point sizes, and the gray ruby row. Those stay pinned in
per-side unit tests (PC `viewer.rs`: `body_ruby_uses_body_typeface_not_term_display`,
`term_ruby_keeps_full_size_term_display`; Android `RubyBaseStyleTest` pins
the mapping and the parser fact that every definition ruby takes the body
path).

DELIBERATE DEPARTURE (ticket 06, 2026-09-21, maintainer decision, both
codebases changed together): body ruby is white/regular BY DECISION, even
though mobile historically painted mini ruby bold cyan. Do NOT "correct"
body toward cyan+bold — that reintroduces the ticket-06 symptom by design
on both sides.

## Adding a case (the parity-bug rule)

A parity bug fix adds a conformance case: write the JSON, run
`cargo test conformance`, paste the actuals only after checking they are the
*correct* semantics (not just what the code happens to emit), and name the
`mobile_mirror` or record why there is none.
