# Kana small/large correction

## Rule

Applied over ordered lines before layout (after recognition, before
`compute_char_boxes` consumers):

- Encoder: per-candidate 40-int window (`WINDOW_BYTES = 2·RADIUS·CELL`,
  `RADIUS = 5`, `CELL = 4`; `src/kana_size.rs:164-167`), byte-packed base
  indices over `BASE_ORDER` (20 pair families, big form first, hiragana then
  katakana; `:159`). The target character is NOT in its own window; context
  stops at `。`/newline (`BOUNDARY`, `:176`) and runs off line ends as zeros
  (`window`, `:266`).
- Policy (`correct_lines`, `:374`, scoring injected so it is testable
  without the native net): candidates in text order; flip a small form iff
  `p_big > 1 − ε`, a big form iff `p_big < ε`, with `EPSILON = 0.01` (`:37`).
  A missing/wrong-length score leaves the page untouched, like mobile
  `KanaSizeFix.apply`. Declined positions are reported lowest-confidence
  first, without surrounding text (`declined_summary`, `:349`; mobile
  `KanaSizeFix.lastDeclined`).
- Mobile measurement behind ε = 0.01: 7,620 confusable bench positions,
  grounded rule at three epsilon bands, reproducing +8/+10/+5
  (`util/KanaSizeFix.kt:25`).

## Mirrored Android source

- `util/KanaSizeEncoder.kt` (`window`), `KanaSizeNcnn.kt`
  (`WINDOW_BYTES = KanaSizeEncoder.WINDOW_BYTES`, `:91`),
  `util/KanaSizeFix.kt` (`EPSILON` `:52`, tunable `PREF_EPSILON`/`DEF_EPSILON`
  `:55-56`, `apply` `:103`, `flipsIt` `:147`, `lastDeclined` `:74`).
- PC has no prefs-tunable epsilon copy; the harness passes ε per case.

## Traps

- Including the target char in its own window: shifts every published
  vector by one cell and silently halves flip precision.
- Widening ε to "fix" misses: the middle band is declined by design
  (`kana-03`); a looser ε flips marginal positions (`a_looser_epsilon_flips_a_marginal_position`).
- Mutating input lines or half-applying a short score array: both return the
  page untouched instead.

## Pinning tests

- PC unit (`src/kana_size.rs` tests): published-vector table `VECTORS`
  (`:548`), window-shape/encoder tests, epsilon-band tests including
  `a_page_below_the_kana_floor_is_still_corrected` and the declined-summary
  test.
- Conformance: `kana-01-encoder-windows` (40-int window bytes exact),
  `kana-02-policy-flips`, `kana-03-policy-declines`, via `kana_cases`
  (`src/conformance.rs:519`).
- Mobile: `KanaSizeEncoderTest`
  (``every published vector encodes byte for byte``,
  ``base index follows the model's table``,
  ``the target character is not in its own window``,
  ``context stops at a full stop or a newline``,
  ``context runs off the line's ends as zeros``);
  `KanaSizeFixTest` (``flips a small/big position the model is sure about``,
  ``leaves the middle band alone``, ``a looser epsilon flips a marginal
  position``, ``declined positions are reported with their confidence``,
  ``a failed scorer leaves the page untouched``,
  ``a short result array is rejected rather than half-applied``, …).
