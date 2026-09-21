# Pipeline invariant specs

Short, reviewed spec pages for the pipeline behaviours the PC port had to
reverse-engineer out of the Android app — the traps that cost the most time.
Each page states the rule, the Android source it mirrors (file + symbol),
the traps a re-implementer gets wrong, and the tests and conformance cases
that pin it.

Android paths are relative to
`app/src/main/java/com/holopengin/instantjpdict/` in the mobile repo;
PC paths are relative to `accessibility_daemon/` — pipeline modules live
under `core/src/` (the UI-free `jpdict_core` crate, ticket 03), the desktop
UI under `src/`.

Pages:

- [rotated-geometry.md](rotated-geometry.md) — frame angle convention, glyph
  tilt, axis-aligned band + quantization, unclip, detection defaults.
- [furigana.md](furigana.md) — ruby drop rules, raw vs unclipped geometry.
- [blank-gaps.md](blank-gaps.md) — vertical-only gap detection, placeholder
  insertion, candidate pool, LM ranking, fallback order, list retention.
- [glyph-sizing.md](glyph-sizing.md) — cross-axis sizing, `glyphSizePx`, ink
  centring, halfwidth handling.
- [kana-correction.md](kana-correction.md) — small/large window, encoder,
  epsilon policy.
- [ctc-decode.md](ctc-decode.md) — blank as ideographic space, raw cache,
  re-decode, alternatives panel + OOV assembly.
- [reading-order.md](reading-order.md) — sort rules and the nav graph.
- [dictionary.md](dictionary.md) — the #88 structured-content model, grouping,
  dedup, ruby, and the ticket-06 body-ruby departure.
- [deinflection.md](deinflection.md) — chain plumbing and reason labels
  (ticket-07).

Authority and tolerances: numeric tolerances and the accepted platform
substitutions (Skia-vs-fontdue metrics, JNI-vs-Rust ncnn low bits, converged
detection defaults) live in `accessibility_daemon/tests/conformance/` —
`TOLERANCES.md` is the authority and these pages link it rather than
duplicating it. Case format: `FORMAT.md` in the same directory.

Maintenance rule: **a parity bug fix updates the spec and adds a conformance
case.** (Also recorded in `AGENTS.md`; closes ticket 01's AGENTS.md
follow-up.)
