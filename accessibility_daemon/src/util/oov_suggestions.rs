//! Mobile `OovSuggestions` (#44, step 1): component-derived alternatives for a
//! recognised character.
//!
//! The head can only emit characters it has a class for, so its own top-K can
//! never contain the character that was actually in the book when that
//! character is outside its vocabulary. At the measured tier — neighbours
//! sharing IDF mass ≥ 0.7 with the emitted character — the right character
//! lands in a ~14-entry list for 12 of 59 measured substitutions.
//!
//! **Nothing here changes recognised text.** These are extra entries in a
//! popup the user already opens, so there is no over-correction risk.

use crate::util::oov_candidates::OovCandidates;

/// IDF mass a candidate must share with the emitted character (measured tier).
pub const MIN_IDF_FRACTION: f32 = 0.7;

/// Caps on the generated groups. The measured ranking still decides the
/// order, so the useful entries stay at the front; tapping a generated entry
/// rebuilds the list around it, which is what makes the kanji form space
/// walkable.
pub const MAX_COMPONENT_CANDIDATES: usize = 15;
pub const MAX_VARIANT_CANDIDATES: usize = 15;

/// Where a popup entry came from. The panel tints non-head entries by
/// source so the provenance is visible at a glance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Head,
    Components,
    Variant,
    /// The blank path's LM-ranked entries (mobile tags them; the PC blank
    /// path tags its ranked list `Lm` when a model is loaded, `Head` when
    /// the pool keeps discovery order).
    Lm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Suggestion {
    pub ch: char,
    pub source: Source,
}

/// The popup list for one character: the head's own ranking first and
/// unchanged (it is preferred whenever it is right), then component
/// neighbours by descending IDF mass, then the obsolete variant forms of the
/// character itself (offered, never applied).
///
/// Duplicates are dropped across groups, the current character is always
/// present so the panel can mark it, and each generated group is capped.
/// Pass `oov: None` when the component table is unavailable: the list is then
/// the head list unchanged.
pub fn assemble(
    current: char,
    head_alternatives: &[char],
    oov: Option<&OovCandidates>,
    variant_forms: &dyn Fn(char) -> Vec<char>,
) -> Vec<Suggestion> {
    let mut out: Vec<Suggestion> =
        Vec::with_capacity(head_alternatives.len() + MAX_COMPONENT_CANDIDATES + 1);
    let mut seen: Vec<char> = Vec::with_capacity(head_alternatives.len() * 2 + 8);

    for c in head_alternatives {
        if !seen.contains(c) {
            seen.push(*c);
            out.push(Suggestion { ch: *c, source: Source::Head });
        }
    }
    // An override can put a character in the text that the head never ranked
    // here, and the panel needs it present to show which entry is current.
    if !seen.contains(&current) {
        seen.push(current);
        out.push(Suggestion { ch: current, source: Source::Head });
    }

    if let Some(oov) = oov {
        if oov.has_discriminating_components(current) {
            let mut added = 0;
            for candidate in oov.neighbours_of(current) {
                if added >= MAX_COMPONENT_CANDIDATES {
                    break;
                }
                // neighbours_of is ordered by descending IDF mass, so the
                // first candidate below the tier ends the group.
                if candidate.idf_fraction < MIN_IDF_FRACTION {
                    break;
                }
                if !seen.contains(&candidate.ch) {
                    seen.push(candidate.ch);
                    out.push(Suggestion { ch: candidate.ch, source: Source::Components });
                    added += 1;
                }
            }
        }
    }

    let mut variants = 0;
    for form in variant_forms(current) {
        if variants >= MAX_VARIANT_CANDIDATES {
            break;
        }
        if !seen.contains(&form) {
            seen.push(form);
            out.push(Suggestion { ch: form, source: Source::Variant });
            variants += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::component_table::ComponentTable;

    /// The mobile fixture, sized so the arithmetic is real, not incidental:
    /// 50 kanji, `化` carried by two of them and `中` by twenty, which puts a
    /// candidate sharing only `化` at ln(50/2)/(ln(50/2)+ln(50/20)) ≈ 0.78 —
    /// clear of the measured 0.7 tier — while a candidate sharing only the
    /// common `中` sits at ≈ 0.22 and must be excluded.
    fn fixture_oov() -> OovCandidates {
        let mut text = String::from("仲:化 中\n伜:化 九 十\n");
        for i in 0..19 {
            text.push(char::from_u32(0x4E00 + i).expect("BMP"));
            text.push_str(":中\n");
        }
        for i in 0..29 {
            text.push(char::from_u32(0x5E00 + i).expect("BMP"));
            text.push_str(":水\n");
        }
        OovCandidates::new(ComponentTable::parse(&text))
    }

    fn chars(out: &[Suggestion]) -> Vec<char> {
        out.iter().map(|s| s.ch).collect()
    }

    fn sources(out: &[Suggestion]) -> Vec<Source> {
        out.iter().map(|s| s.source).collect()
    }

    #[test]
    fn component_neighbour_is_appended_after_the_head_list() {
        let oov = fixture_oov();
        let out = assemble('仲', &['仲'], Some(&oov), &|_| Vec::new());
        assert_eq!(chars(&out), vec!['仲', '伜']);
        assert_eq!(sources(&out), vec![Source::Head, Source::Components]);
    }

    #[test]
    fn head_order_is_preserved_and_a_repeated_candidate_is_not_appended_twice() {
        let oov = fixture_oov();
        let out = assemble('仲', &['伜', '仲'], Some(&oov), &|_| Vec::new());
        assert_eq!(chars(&out), vec!['伜', '仲']);
        assert!(sources(&out).iter().all(|s| *s == Source::Head));
    }

    #[test]
    fn the_variant_group_walks_the_form_space_from_the_current_selection() {
        // Suggestions are assembled from whatever character is *current*, so
        // choosing a generated entry rebuilds the candidates around it:
        // 摑 -> 掴 -> 摑 is reachable by tapping through the popup (#44).
        // Pinned so a refactor cannot quietly make the list depend on the
        // character the recogniser originally emitted instead.
        let from_old = assemble('摑', &['摑'], None, &|_| vec!['掴']);
        assert_eq!(from_old.last().map(|s| s.source), Some(Source::Variant));
        assert_eq!(from_old.last().map(|s| s.ch), Some('掴'));
        let from_modern = assemble('掴', &['掴'], None, &|_| vec!['摑']);
        assert_eq!(from_modern.last().map(|s| s.source), Some(Source::Variant));
        assert_eq!(from_modern.last().map(|s| s.ch), Some('摑'));
    }

    #[test]
    fn a_single_component_character_gets_no_component_suggestions() {
        // The IDF fraction is relative to the emitted character, so a
        // one-component character makes every one of its carriers a full
        // match (fraction 1.0): the tier would admit hundreds of unrelated
        // characters and the cap would then choose between them by codepoint.
        // No discriminating evidence, no suggestions.
        let oov = fixture_oov();
        let one = char::from_u32(0x4E00).expect("BMP"); // carries 中 only
        let out = assemble(one, &[one], Some(&oov), &|_| Vec::new());
        assert_eq!(chars(&out), vec![one]);
    }

    #[test]
    fn variant_forms_are_offered_last_and_capped() {
        let oov = fixture_oov();
        // Twenty forms, so the cap — not the supply — is what bounds the group.
        let forms: Vec<char> = (0..20).map(|i| char::from_u32(0x5F00 + i).expect("BMP")).collect();
        let out = assemble('仲', &['仲'], Some(&oov), &|_| forms.clone());
        let variant_entries: Vec<&Suggestion> =
            out.iter().filter(|s| s.source == Source::Variant).collect();
        assert_eq!(MAX_VARIANT_CANDIDATES, variant_entries.len());
        let first = out.iter().position(|s| s.source == Source::Variant).expect("variants");
        assert_eq!(MAX_VARIANT_CANDIDATES, out.len() - first);
        // A set smaller than the cap is offered whole, in order.
        let few = vec!['囘', '欝'];
        let small = assemble('回', &['回'], Some(&oov), &|_| few.clone());
        let got: Vec<char> =
            small.iter().filter(|s| s.source == Source::Variant).map(|s| s.ch).collect();
        assert_eq!(few, got);
    }

    #[test]
    fn component_group_is_capped() {
        // Emitted 仲 = 化 + 中, with 化 carried by 17 of 911 kanji and 中 by
        // 895: a candidate sharing only 化 scores
        // ln(911/17)/(ln(911/17)+ln(911/895)) ≈ 0.995, clear of the tier, so
        // sixteen qualify and the cap — not the tier — bounds the list.
        let mut wide = String::from("仲:化 中\n");
        for i in 0..16 {
            wide.push(char::from_u32(0x6C00 + i).expect("BMP"));
            wide.push_str(":化\n");
        }
        for i in 0..894 {
            wide.push(char::from_u32(0x4E00 + i).expect("BMP"));
            wide.push_str(":中\n");
        }
        let oov = OovCandidates::new(ComponentTable::parse(&wide));
        let out = assemble('仲', &['仲'], Some(&oov), &|_| Vec::new());
        assert_eq!(
            MAX_COMPONENT_CANDIDATES,
            out.iter().filter(|s| s.source == Source::Components).count()
        );
    }

    #[test]
    fn without_a_component_table_the_list_is_the_head_list_unchanged() {
        let head = vec!['あ', '仲', 'い'];
        let out = assemble('仲', &head, None, &|_| Vec::new());
        assert_eq!(chars(&out), head);
        assert!(sources(&out).iter().all(|s| *s == Source::Head));
    }

    #[test]
    fn the_current_character_is_present_even_when_the_head_list_omits_it() {
        // An override can put a character in the text that the head never
        // ranked here.
        let oov = fixture_oov();
        let out = assemble('伜', &['仲'], Some(&oov), &|_| Vec::new());
        assert!(out.iter().any(|s| s.ch == '伜'), "current character must be listed");
        assert_eq!(out.first().map(|s| s.ch), Some('仲'));
    }
}
