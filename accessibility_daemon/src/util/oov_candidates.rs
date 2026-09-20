//! Mobile `OovCandidates` (#44): the component-level candidate policy for
//! out-of-vocabulary characters.
//!
//! Two shapes of the OOV failure are served here, both measured against the
//! vendored KRADFILE table ([`ComponentTable`]) rather than guessed:
//!
//! - **deletion** — the head emits blank where a character should be.
//!   [`OovCandidates::majority_components`] turns the head's top-K free
//!   evidence into the components a candidate must carry.
//! - **substitution** — the head is confidently wrong and emits a
//!   component-space neighbour: 45 of 51 measured substitutions (88%) share a
//!   component with the truth. [`OovCandidates::neighbours_of`] generates
//!   that neighbour set from the character the head actually emitted.
//!
//! Ranking is deliberately *not* done here — a text n-gram cannot separate
//! `と呟いて` from `と咲いて`, so the ordering exposed is component evidence
//! only, and the caller adds whatever LM it has.

use crate::util::component_table::ComponentTable;

pub struct OovCandidates {
    table: ComponentTable,
}

impl OovCandidates {
    pub fn new(table: ComponentTable) -> Self {
        Self { table }
    }

    /// Borrow the underlying table (tests; mirrors direct mobile access).
    #[allow(dead_code)]
    pub fn table(&self) -> &ComponentTable {
        &self.table
    }

    /// Candidate characters for a character the head **emitted** (the
    /// substitution mode), ordered by [`Candidate::idf_fraction`] descending,
    /// ties broken by codepoint so the list is deterministic.
    ///
    /// The pool is every kanji sharing **at least one** component with
    /// `emitted` (excluding `emitted` itself). The fraction is the
    /// **intersected** share of the emitted character's IDF mass that the
    /// candidate also carries — never the candidate's own total mass, which
    /// rewards carrying many common components and scored 0/45 on the ranking
    /// task. An unknown character, or one whose components carry no IDF mass,
    /// yields an empty list rather than an error.
    pub fn neighbours_of(&self, emitted: char) -> Vec<Candidate> {
        let emitted_components = self.table.components_of(emitted);
        if emitted_components.is_empty() {
            return Vec::new();
        }
        // De-duplicated for the pool walk, mirroring the reference set use;
        // the denominator sums the stored decomposition as-is.
        let emitted_set: Vec<char> = {
            let mut seen = Vec::with_capacity(emitted_components.len());
            for c in emitted_components {
                if !seen.contains(c) {
                    seen.push(*c);
                }
            }
            seen
        };
        let denominator = idf_mass(&emitted_components.iter().map(|c| self.table.idf_of(*c)).collect::<Vec<_>>());
        if denominator <= 0.0 {
            return Vec::new();
        }

        // Component -> kanji is a precomputed posting list, so the pool is
        // the union of a handful of lists, not a scan of the table.
        let mut pool: Vec<char> = Vec::new();
        for component in &emitted_set {
            for k in self.table.kanji_with(&[*component]) {
                if !pool.contains(&k) {
                    pool.push(k);
                }
            }
        }

        let mut out: Vec<Candidate> = Vec::with_capacity(pool.len());
        for k in pool {
            if k == emitted {
                continue;
            }
            let comps = self.table.components_of(k);
            let shared: f64 = comps
                .iter()
                .filter(|c| emitted_set.contains(c))
                .map(|c| self.table.idf_of(*c) as f64)
                .sum();
            // `shares_all_components`: the candidate carries **every**
            // component of the emitted character — the near-identity relation.
            let shares_all = emitted_set.iter().all(|c| comps.contains(c));
            out.push(Candidate {
                ch: k,
                idf_fraction: (shared / denominator) as f32,
                shares_all_components: shares_all,
            });
        }
        out.sort_by(|a, b| {
            b.idf_fraction
                .total_cmp(&a.idf_fraction)
                .then_with(|| a.ch.cmp(&b.ch))
        });
        out
    }

    /// Whether component evidence can discriminate at all for this character.
    /// The IDF fraction is measured **relative to the emitted character**, so
    /// a single-component character makes every one of its carriers a full
    /// match (fraction 1.0): the tier then admits hundreds of unrelated
    /// characters and a cap picks between them by codepoint — noise presented
    /// as evidence. Two or more components are what the measurements rely on.
    pub fn has_discriminating_components(&self, emitted: char) -> bool {
        self.table.components_of(emitted).len() >= 2
    }

    /// The components a top-K of characters **agree on**, by majority vote: a
    /// component carried by at least `need_fraction` of `top_k` (at least one
    /// character). Ordered strongest first — by how many of the top-K carry
    /// it, then by codepoint.
    ///
    /// **Majority voting, never a strict intersection.** For the measured
    /// top-5 `咳 咬 啦 哮 眩` the strict intersection is empty, so
    /// intersecting all five filters every candidate away. `counts` counts
    /// *characters*, not component occurrences. If the vote leaves nothing
    /// the single strongest component is returned rather than an empty set;
    /// an empty `top_k`, or one whose characters are all unknown, returns an
    /// empty list. The default 0.5 is the measured threshold.
    ///
    /// (Deletion-path policy: not wired into the panel yet — the blank path
    /// keeps its evidence-pool rules — but ported and pinned by tests so the
    /// two implementations stay comparable.)
    #[allow(dead_code)]
    pub fn majority_components(&self, top_k: &[char], need_fraction: f64) -> Vec<char> {
        if top_k.is_empty() {
            return Vec::new();
        }
        let mut counts: Vec<(char, usize)> = Vec::new();
        for ch in top_k {
            let comps = self.table.components_of(*ch);
            let mut seen: Vec<char> = Vec::with_capacity(comps.len());
            for c in comps {
                if !seen.contains(c) {
                    seen.push(*c);
                }
            }
            for c in seen {
                match counts.iter_mut().find(|(k, _)| *k == c) {
                    Some(slot) => slot.1 += 1,
                    None => counts.push((c, 1)),
                }
            }
        }
        if counts.is_empty() {
            return Vec::new();
        }
        // `int(len(chars) * need_frac + 0.9999)` in the reference: ceil for
        // the usual fractional thresholds (5 * 0.5 -> 3), truncation for exact
        // integers.
        let need = ((top_k.len() as f64) * need_fraction + 0.9999) as usize;
        let need = need.max(1);
        let mut picked: Vec<char> = counts
            .iter()
            .filter(|(_, n)| *n >= need)
            .map(|(c, _)| *c)
            .collect();
        if picked.is_empty() {
            // Unreachable for need_fraction <= 1 but kept because the
            // reference falls back here.
            let strongest = counts.iter().max_by_key(|(_, n)| *n).expect("non-empty");
            picked = vec![strongest.0];
        }
        picked.sort_by(|a, b| {
            let ca = counts.iter().find(|(k, _)| k == a).map(|(_, n)| *n).unwrap_or(0);
            let cb = counts.iter().find(|(k, _)| k == b).map(|(_, n)| *n).unwrap_or(0);
            cb.cmp(&ca).then_with(|| a.cmp(b))
        });
        picked
    }
}

/// A character that could stand where the emitted character was, with the
/// strength of the visual relation to it.
pub struct Candidate {
    pub ch: char,
    /// Share of the emitted character's component information (IDF mass)
    /// that this candidate also carries, in `[0, 1]`. Intersected, never the
    /// candidate's whole mass.
    pub idf_fraction: f32,
    /// The candidate carries **every** component of the emitted character —
    /// the near-identity relation. (Pinned by tests; no panel reader yet.)
    #[allow(dead_code)]
    pub shares_all_components: bool,
}

fn idf_mass(weights: &[f32]) -> f64 {
    weights.iter().map(|w| *w as f64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real asset, parsed (mirrors mobile `OovCandidatesTest`).
    fn real() -> Option<OovCandidates> {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/components/krad_components.txt");
        let text = std::fs::read_to_string(&path).ok()?;
        Some(OovCandidates::new(ComponentTable::parse(&text)))
    }

    /// The measured top-5 the recogniser emitted at the deleted `呟`.
    const MEASURED_TOP5: [char; 5] = ['咳', '咬', '啦', '哮', '眩'];

    #[test]
    fn majority_components_of_the_measured_top5_matches_fukan() {
        let Some(candidates) = real() else {
            eprintln!("skipping: components asset not present");
            return;
        };
        // 口 is carried by 4 of 5, 亠 by 3 — both >= ceil(5/2) — and both are
        // components of the dropped 呟 (亠 口 幺 玄).
        assert_eq!(candidates.majority_components(&MEASURED_TOP5, 0.5), vec!['口', '亠']);
        // The strict intersection really is empty: this is what the vote replaces.
        let mut strict: Option<Vec<char>> = None;
        for ch in MEASURED_TOP5 {
            let set = candidates.table().components_of(ch).to_vec();
            strict = Some(match strict {
                None => set,
                Some(acc) => acc.into_iter().filter(|c| set.contains(c)).collect(),
            });
        }
        assert_eq!(strict.unwrap_or_default(), Vec::<char>::new());
    }

    #[test]
    fn majority_components_falls_back_to_the_strongest_and_handles_empty_input() {
        let Some(candidates) = real() else {
            eprintln!("skipping: components asset not present");
            return;
        };
        // need_fraction 1.0 demands unanimity, which this top-K does not
        // have; the fallback is the single strongest component (口, 4 of 5).
        assert_eq!(candidates.majority_components(&MEASURED_TOP5, 1.0), vec!['口']);
        assert!(candidates.majority_components(&[], 0.5).is_empty());
        assert!(candidates.majority_components(&['\u{E000}', '\u{E001}'], 0.5).is_empty());
    }

    #[test]
    fn neighbours_of_emitted_char_find_the_measured_substitution() {
        let Some(candidates) = real() else {
            eprintln!("skipping: components asset not present");
            return;
        };
        // 曇 (二 厶 日 雨) -> 壜 (二 厶 土 日 雨): the head's component-space
        // neighbour from the Aozora bench, carrying all four of 曇's
        // components — the near-identity relation, with full IDF fraction.
        let neighbours = candidates.neighbours_of('曇');
        let ban = neighbours.iter().find(|c| c.ch == '壜').expect("壜 is a neighbour");
        assert!(ban.shares_all_components);
        assert!((ban.idf_fraction - 1.0).abs() < 1e-6);

        // And the relation is discriminating: 但 (一 化 日) shares only the common 日.
        let ta = neighbours.iter().find(|c| c.ch == '但').expect("但 is a neighbour");
        assert!(!ta.shares_all_components);
        assert!((ta.idf_fraction - 0.1798134).abs() < 1e-4, "但 = {}", ta.idf_fraction);

        // Exactly the two kanji carrying every component of 曇, ordered by
        // codepoint when the evidence ties: 壜 < 罎.
        let full: Vec<char> = neighbours
            .iter()
            .filter(|c| c.shares_all_components)
            .map(|c| c.ch)
            .collect();
        assert_eq!(full, vec!['壜', '罎']);
        assert_eq!(neighbours.first().map(|c| c.ch), Some('壜'));
    }

    #[test]
    fn idf_fraction_intersects_the_emitted_characters_components() {
        let Some(candidates) = real() else {
            eprintln!("skipping: components asset not present");
            return;
        };
        // 呟 = 亠 口 幺 玄 (13.064 nats); 咳 = ノ 丶 亠 人 口 shares 亠 + 口
        // only: (4.2185 / 13.0642) = 0.32291. The rejected variant divided by
        // *咳's* whole mass instead, giving 0.35765.
        let neighbours = candidates.neighbours_of('呟');
        let kai = neighbours.iter().find(|c| c.ch == '咳').expect("咳 is a neighbour");
        assert!((kai.idf_fraction - 0.3229068).abs() < 1e-4, "咳 = {}", kai.idf_fraction);
        assert!(!kai.shares_all_components, "呟's 幺 and 玄 are missing");
        assert!(neighbours.iter().all(|c| c.ch != '呟'), "never its own candidate");
    }

    #[test]
    fn neighbours_are_sorted_by_evidence_and_never_contain_the_emitted_character() {
        let Some(candidates) = real() else {
            eprintln!("skipping: components asset not present");
            return;
        };
        let neighbours = candidates.neighbours_of('曇');
        assert!(neighbours.len() > 1000, "the pool is the wide 'shares any component' rule");
        for pair in neighbours.windows(2) {
            assert!(pair[0].idf_fraction >= pair[1].idf_fraction);
        }
        let emitted = candidates.table().components_of('曇').to_vec();
        for c in &neighbours {
            assert_ne!(c.ch, '曇');
            assert!(c.idf_fraction > 0.0 && c.idf_fraction <= 1.0);
            assert!(
                candidates.table().components_of(c.ch).iter().any(|x| emitted.contains(x)),
                "{} shares nothing with 曇",
                c.ch
            );
        }
    }

    #[test]
    fn unknown_characters_yield_no_candidates_and_no_majority() {
        let Some(candidates) = real() else {
            eprintln!("skipping: components asset not present");
            return;
        };
        assert!(candidates.neighbours_of('\u{E000}').is_empty());
        assert!(candidates.neighbours_of('あ').is_empty());
        assert!(candidates.majority_components(&['あ'], 0.5).is_empty());
    }
}
