//! Mobile `KanjiVariants` (#44): the vendored Unihan
//! `kSemanticVariant`/`kZVariant` pairs plus the JMdict old-orthography half,
//! loaded from `assets/variants/kanji_variants.txt` (source, licence and
//! SHA-256 are in the `PROVENANCE.txt` beside it; regenerate with mobile's
//! `tools/build_kanji_variants.py`).
//!
//! **Direction:** `variant -> canonical`, where the *canonical* side is the
//! one the shipped recogniser's dictionary carries and the *variant* is the
//! one it does not. The table is exposed in both directions for callers that
//! want it — e.g. [`KanjiVariantTable::obsolete_forms_of`] offering the
//! obsolete forms of a character as alternatives. Until a table is loaded
//! every lookup is the identity function (a missing entry is not an error —
//! an unknown character folds to itself).

use std::collections::HashMap;

pub struct KanjiVariantTable {
    canonical_by_variant: HashMap<char, char>,
    // Mobile API surface, pinned by tests (the multi-candidate Unihan pairs
    // the lookup fold resolves on the corpus); no production reader yet.
    #[allow(dead_code)]
    canonicals_by_variant: HashMap<char, Vec<char>>,
    variants_by_canonical: HashMap<char, Vec<char>>,
}

impl KanjiVariantTable {
    /// An empty table: every lookup is the identity function.
    /// (Mobile API surface, pinned by tests.)
    #[allow(dead_code)]
    pub fn empty() -> Self {
        Self {
            canonical_by_variant: HashMap::new(),
            canonicals_by_variant: HashMap::new(),
            variants_by_canonical: HashMap::new(),
        }
    }

    /// Parse the committed asset: one `variant<TAB>canonical` pair per line,
    /// `#` comments and blank lines ignored, malformed lines skipped. Every
    /// pair is kept — Unihan gives some variants more than one canonical
    /// candidate, and [`KanjiVariantTable::canonicals_of`] exposes them —
    /// while [`KanjiVariantTable::canonical`] answers with the **first** line
    /// for that variant (the generator sorts by variant then canonical, so
    /// first is the lowest codepoint: deterministic, but arbitrary).
    /// Supplementary-plane pairs are skipped: this API is BMP-based, like the
    /// mobile reference.
    pub fn parse(text: &str) -> Self {
        let mut pairs: Vec<(char, char)> = Vec::new();
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split('\t');
            let (Some(v), Some(c), None) = (parts.next(), parts.next(), parts.next()) else {
                continue;
            };
            let (Some(variant), Some(canonical)) = (single_bmp(v), single_bmp(c)) else {
                continue;
            };
            if variant == canonical {
                continue;
            }
            pairs.push((variant, canonical));
        }

        let mut canonical_by_variant: HashMap<char, char> = HashMap::new();
        let mut canonicals_by_variant: HashMap<char, Vec<char>> = HashMap::new();
        let mut variants_by_canonical: HashMap<char, Vec<char>> = HashMap::new();
        for (variant, canonical) in pairs {
            canonical_by_variant.entry(variant).or_insert(canonical);
            canonicals_by_variant.entry(variant).or_default().push(canonical);
            variants_by_canonical.entry(canonical).or_default().push(variant);
        }
        for list in canonicals_by_variant.values_mut().chain(variants_by_canonical.values_mut()) {
            list.sort();
            list.dedup();
        }
        Self {
            canonical_by_variant,
            canonicals_by_variant,
            variants_by_canonical,
        }
    }

    /// Load the shipped table; `None` when the file is missing, unreadable or
    /// parses to nothing (callers then offer no variant forms).
    pub fn load(path: &std::path::Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        let table = Self::parse(&text);
        if table.entry_count() == 0 {
            return None;
        }
        Some(table)
    }

    /// Number of distinct variants installed; 0 when nothing is loaded.
    pub fn entry_count(&self) -> usize {
        self.canonical_by_variant.len()
    }

    /// The canonical form of `ch`, or `ch` itself when the table has no entry
    /// for it. Never fails, including on an empty table.
    /// (Mobile API surface, pinned by tests; the panel only needs the
    /// reverse direction.)
    #[allow(dead_code)]
    pub fn canonical(&self, ch: char) -> char {
        self.canonical_by_variant.get(&ch).copied().unwrap_or(ch)
    }

    /// Every canonical candidate the table lists for `ch`, sorted; empty when
    /// there are none. (Mobile API surface, pinned by tests.)
    #[allow(dead_code)]
    pub fn canonicals_of(&self, ch: char) -> &[char] {
        self.canonicals_by_variant.get(&ch).map(Vec::as_slice).unwrap_or(&[])
    }

    /// The variant forms that fold to `ch`, sorted; empty when there are none.
    pub fn obsolete_forms_of(&self, ch: char) -> Vec<char> {
        self.variants_by_canonical.get(&ch).cloned().unwrap_or_default()
    }
}

/// Exactly one BMP character, or nothing (mirrors Kotlin `singleOrNull` plus
/// the mobile generator's BMP-only rule).
fn single_bmp(token: &str) -> Option<char> {
    let mut chars = token.chars();
    let c = chars.next()?;
    if chars.next().is_some() || (c as u32) > 0xFFFF {
        return None;
    }
    Some(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed asset, parsed (mirrors mobile `KanjiVariantsTest`).
    fn table() -> Option<KanjiVariantTable> {
        let path =
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/..")).join("assets/variants/kanji_variants.txt");
        Some(KanjiVariantTable::parse(&std::fs::read_to_string(&path).ok()?))
    }

    #[test]
    fn parses_the_committed_asset() {
        let path =
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/..")).join("assets/variants/kanji_variants.txt");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("skipping: {} not present", path.display());
            return;
        };
        assert_eq!(669, text.lines().filter(|l| !l.trim().is_empty()).count());
        // 600 distinct variants: Unihan gives 58 of them more than one
        // canonical candidate, and the table keeps every pair.
        assert_eq!(600, table().expect("asset").entry_count());
    }

    #[test]
    fn the_jmdict_half_reaches_old_orthography_the_vocab_rule_could_not() {
        let Some(t) = table() else {
            eprintln!("skipping: variants asset not present");
            return;
        };
        assert_eq!('掴', t.canonical('摑'));
        assert_eq!('国', t.canonical('國'));
        assert_eq!('会', t.canonical('會'));
        assert_eq!('灯', t.canonical('燈'));
        assert_eq!('当', t.canonical('當'));
        // Traps the intersection exists to exclude: Unihan links each of
        // these, but they are different words in Japanese.
        assert_eq!('誌', t.canonical('誌'));
        assert_eq!('製', t.canonical('製'));
        assert_eq!('長', t.canonical('長'));
        assert_eq!('階', t.canonical('階'));
    }

    #[test]
    fn chains_resolve_to_the_form_a_dictionary_indexes() {
        let Some(t) = table() else {
            eprintln!("skipping: variants asset not present");
            return;
        };
        // 冩 -> 寫 -> 写: both hops now reach 写.
        assert_eq!('写', t.canonical('冩'));
        assert_eq!('写', t.canonical('寫'));
        // A genuine cycle (干 <-> 乾) is kept as-is rather than guessed at.
        assert_eq!('干', t.canonical('乾'));
        assert_eq!('乾', t.canonical('干'));
    }

    #[test]
    fn folds_variant_onto_the_dictionary_form() {
        let Some(t) = table() else {
            eprintln!("skipping: variants asset not present");
            return;
        };
        assert_eq!('回', t.canonical('囘'));
        assert_eq!('鬱', t.canonical('欝'));
        assert_eq!('罈', t.canonical('壜'));
        assert_eq!('逃', t.canonical('迯'));
        assert_eq!('器', t.canonical('噐'));
        assert_eq!('慚', t.canonical('慙'));
    }

    #[test]
    fn a_variant_with_several_canonical_candidates_keeps_them_all() {
        let Some(t) = table() else {
            eprintln!("skipping: variants asset not present");
            return;
        };
        assert_eq!(&['盖', '蓋'], t.canonicals_of('葢'));
        assert_eq!('盖', t.canonical('葢'));
        assert_eq!(&['冰', '氷'], t.canonicals_of('冫'));
        assert_eq!('冰', t.canonical('冫'));
        assert_eq!(&['回'], t.canonicals_of('囘'));
        assert!(t.canonicals_of('あ').is_empty());
    }

    #[test]
    fn unknown_character_returns_itself() {
        let Some(t) = table() else {
            eprintln!("skipping: variants asset not present");
            return;
        };
        assert_eq!('漢', t.canonical('漢'));
        assert_eq!('あ', t.canonical('あ'));
        assert_eq!('A', t.canonical('A'));
        assert_eq!('回', t.canonical('回'));
    }

    #[test]
    fn obsolete_forms_are_the_reverse_direction() {
        let Some(t) = table() else {
            eprintln!("skipping: variants asset not present");
            return;
        };
        assert!(t.obsolete_forms_of('回').contains(&'囘'));
        assert!(t.obsolete_forms_of('鬱').contains(&'欝'));
        assert_eq!(vec!['墰', '壜'], t.obsolete_forms_of('罈'));
        let forms = t.obsolete_forms_of('回');
        let mut sorted = forms.clone();
        sorted.sort();
        assert_eq!(forms, sorted, "sorted");
        assert!(t.obsolete_forms_of('あ').is_empty());
        assert!(t.obsolete_forms_of('漢').is_empty());
    }

    #[test]
    fn empty_table_is_the_identity() {
        let t = KanjiVariantTable::parse("");
        assert_eq!(0, t.entry_count());
        assert_eq!('囘', t.canonical('囘'));
        assert!(t.obsolete_forms_of('回').is_empty());
    }

    #[test]
    fn parse_skips_malformed_lines_and_keeps_the_first_canonical() {
        let t = KanjiVariantTable::parse(
            "# a comment\n你\t你\n囘\t回\n囘\t囬\n囘\t回\textra\n囘回\n回\t回\n",
        );
        // Only the well-formed single-character pairs survive (你 is
        // supplementary-plane, so it is skipped); the duplicate variant keeps
        // the first line.
        assert_eq!(1, t.entry_count());
        assert_eq!('回', t.canonical('囘'));
    }

    #[test]
    fn load_rejects_a_missing_or_empty_asset() {
        assert!(
            KanjiVariantTable::load(std::path::Path::new("/nonexistent/kanji_variants.txt")).is_none()
        );
        let empty = std::env::temp_dir().join("instantjpdict_empty_variants.txt");
        std::fs::write(&empty, "# nothing\n").expect("write fixture");
        assert!(KanjiVariantTable::load(&empty).is_none(), "empty parses to nothing");
        let _ = std::fs::remove_file(&empty);
    }
}
