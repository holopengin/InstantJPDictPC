# Tolerances: what is drift and what is platform

Default per-case tolerances (overridable in the case's `tolerances` object):

| quantity | default | means |
|---|---|---|
| `box_px` | 2 | each of x/y/w/h within ±2 px |
| `angle_deg` | 1.0 | frame angle within ±1° |
| text / labels / order | exact | no tolerance — strings and sequences match byte-for-byte |

## Known-acceptable platform substitutions (NOT drift)

These differ by host on purpose. Cases are shaped so they never fire on
them; if one does, the case is wrong, not the code.

1. **Text raster metrics — Skia (Android) vs fontdue (PC).** Per-glyph boxes
   from rendered text differ by a pixel or two between rasterizers, so the
   corpus never pins char boxes produced from rendered glyphs (the
   PC-hosted `recognition` kind is the exception: it pins char boxes from
   real inference, exact on the PC host, Android-manual). Char-box
   geometry cases (`gap`, `kana`) work from *recorded decoder inputs* (timestep
   top-K lists, injected logits), which are identical integers and floats on
   both sides.
2. **Inference hosting — JNI ncnn (Android) vs Rust ncnn (PC).** Low bits of
   the detector prob map may differ across hosts, so `detection` cases build
   the prob stand-in from image bytes with an explicit `det_thresh` in the
   case, and `box_px` absorbs single boundary-pixel flips. End-to-end
   pixel→text cases exist as the PC-hosted `recognition` kind (vendored
   photos/screenshots, `CONFORMANCE_DUMP=1` regeneration); they stay
   Android-manual until the Android side can host inference in its tests.
3. **Detection defaults are converged: PC 0.25 / 0.7, mobile 0.25 / 0.70
   (#101, superseding #97's 0.65 / 1.2).** Every `detection` case still
   states its `det_thresh`/`det_unclip` explicitly and both runners must
   use the case values, not their platform defaults — so a future
   re-tuning on either side fails loudly instead of drifting silently.
4. **Synthetic images are pre-rendered PNGs.** No font, renderer, or OS text
   stack is involved in loading them; `image` (PC) and `BitmapFactory`
   (Android) must agree on pixels for these files, which they do for 8-bit
   grayscale PNG.
5. **Inference scheduling — worker completion order vs content.** The
   recognizer fans lines out over `PPOCR_REC_WORKERS` threads, so the raw
   streaming emission order varies run to run. The harness compares the
   index-sorted batch (`recognize_boxes_collect`), which normalizes that
   away: per-line box/text/confidence/char-boxes are byte-identical across
   runs on the same host (verified: two release-binary runs plus two
   in-harness runs over all 9 recognition fixtures, zero content diffs).
   Same-host inference nondeterminism, if it ever appears, is real drift
   until proven otherwise — record the variance and its tolerance here,
   don't widen `box_px` to hide it. Cross-host low-bit inference differences
   (JNI ncnn vs Rust ncnn) stay an accepted substitution, which is why the
   `recognition` kind is PC-hosted-only (`mobile_mirror: null`,
   Android-manual) until the Android side can host inference in its tests.

## Real drift (fails the suite)

Anything outside the above: changed box merge/sort order, changed ruby
ratios or gap/overlap fractions, changed fallback class order, changed LM
back-off chain, changed kana epsilon policy or window layout, changed
example ja/en split or sense-group numbering. The perturbation proof for
each harness addition (flip one semantic, watch the case fail, revert) is
recorded in the ticket comments when the corpus is extended.
