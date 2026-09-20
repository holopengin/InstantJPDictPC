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
   corpus never pins char boxes produced from rendered glyphs. Char-box
   geometry cases (`gap`, `kana`) work from *recorded decoder inputs* (timestep
   top-K lists, injected logits), which are identical integers and floats on
   both sides.
2. **Inference hosting — JNI ncnn (Android) vs Rust ncnn (PC).** Low bits of
   the detector prob map may differ across hosts, so `detection` cases build
   the prob stand-in from image bytes with an explicit `det_thresh` in the
   case, and `box_px` absorbs single boundary-pixel flips. End-to-end
   pixel→text cases stay out of the corpus until both sides can host
   inference inside their test suites.
3. **Detection unclip default differs today: PC 0.7, mobile 1.2 (#97).**
   Every `detection` case states its `det_unclip` explicitly and both runners
   must use the case value, not their platform default. The default gap
   itself is recorded here as accepted-for-now, not as drift — converging it
   is a separate ticket.
4. **Synthetic images are pre-rendered PNGs.** No font, renderer, or OS text
   stack is involved in loading them; `image` (PC) and `BitmapFactory`
   (Android) must agree on pixels for these files, which they do for 8-bit
   grayscale PNG.

## Real drift (fails the suite)

Anything outside the above: changed box merge/sort order, changed ruby
ratios or gap/overlap fractions, changed fallback class order, changed LM
back-off chain, changed kana epsilon policy or window layout, changed
example ja/en split or sense-group numbering. The perturbation proof for
each harness addition (flip one semantic, watch the case fail, revert) is
recorded in the ticket comments when the corpus is extended.
