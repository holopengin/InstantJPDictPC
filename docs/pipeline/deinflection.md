# Deinflection reasons (ticket-07)

## Rule

Chain plumbing and labels, graduated to the corpus by `c7dc78b`:

- The rules file's **group key IS the reason**: at load, each rule's key
  (`past`, `-te`, …) is filled into `rule.reason` (PC
  `Deinflector::from_json_file`, `src/util/deinflector.rs:40`; mobile
  `Deinflector` load, `util/Deinflector.kt:46`). A loader that drops the keys
  yields kana fragments or empty lists.
- Derivation BFS from the surface form, identity candidate first with no
  reasons; derivations dedupe by term, first wins (`deinflect`, `:75`;
  mobile `deinflect`, `util/Deinflector.kt:55`).
- The derivation guard is a **character** count: a term shorter than two
  characters is never deinflected (`deinflect`, `core/src/util/deinflector.rs`;
  mobile `term.length < 2`, `util/Deinflector.kt:62`). A byte count lets every
  single non-ASCII character through — 「た」 would derive 「る」, and lookups
  deinflect prefixes down to one character, so the spurious candidate would
  reach the dictionary.
- Each step appends the human-readable group label — never the kana
  fragment — outermost step first (`reasons = current.reasons + rule.reason`,
  `util/Deinflector.kt:74`; PC `:75-125`). An empty reason (bare-array JSON
  with no group keys) appends nothing, so the identity guard still filters it.
- Display: deinflected matches carry their chain into the viewer chain row
  (`deinflection_row`, `src/viewer.rs:2748`); direct matches render exactly
  as before (no chain, no row). Lookup also tries pre-reform and
  sound-changed kana variants.

## Mirrored Android source

- `util/Deinflector.kt` (`reason` `:14`, `reasons` `:20`, `label` `:32`,
  load `:46`, `deinflect` `:55`).

## Traps

- Displaying `kanaIn`/`kanaOut` fragments as the "reason" (the ticket-07
  symptom the graduation fixed: reason labels instead of kana fragments,
  PC `b601d5e`).
- Keying the viewer row off term identity instead of chain presence: direct
  matches would gain a spurious empty row.
- Blessing a wrong rule by copying dump output into a case: the regenerate
  procedure (`CONFORMANCE_DUMP=1`) requires checking the grammar first
  (FORMAT.md `deinflection` section).

## Pinning tests

- PC unit (`src/util/deinflector.rs` tests): `reasons_carry_rule_names`,
  `identity_result_has_no_reasons`, `single_character_terms_are_not_deinflected`;
  viewer
  (`deinflected_entry_carries_reasons_for_the_chain_row`,
  `src/viewer.rs:4461`).
- Conformance: `deinflection-01-past-verb`, `deinflection-02-te-form`,
  `deinflection-03-adjective-past`, `deinflection-04-dictionary-form-noop`
  (identity carries no reasons), `deinflection-05-single-char-noop`
  (`expect_absent` pins the one-character guard), via `deinflection_cases`
  (`core/src/conformance.rs:767`) against the shipped `assets/deinflect.json`
  (byte-identical to the Android asset copy at graduation).
- Mobile: `DeinflectionChainTest` (`reasons_carryRuleNames`,
  `reasons_accumulateOutermostFirst_multiStep`, `identityResult_hasNoReasons`,
  `label_singleStep`, `label_multiStep`,
  `candidates_directVariantsHaveNoChain_deinflectedDo`,
  `processResults_attachesChainToDeinflectedTerm_only`,
  `formatDictionaryResults_propagatesChain_directStaysNull`).
