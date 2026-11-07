//! Mobile `GapCandidates` (#44): what to offer for a blank.
//!
//! A gap is only worth filling when there is evidence for what went there, so
//! the pool is the set of characters the recogniser itself offered along the
//! line — its per-timestep top-K ([`crate::models::LineResult::raw_alternatives`],
//! blanks included) — filtered to what a real character can be. The character
//! LM then ranks that pool in the line's own context (shape evidence proposes,
//! the text prior disposes); without the model the pool keeps its discovery
//! order. The fallback list (punctuation first, then kana) is never empty: a
//! blank with nothing to choose from is worse than a guess.

use crate::models::GAP_CHAR;
use crate::util::char_lm::{CharLm, MAX_ORDER};

/// Mobile `GapCandidates.MAX`.
pub const MAX: usize = 15;

/// Whether a character the recogniser proposed is worth offering. Kanji, kana
/// and punctuation alike: the gap in vertical Japanese text is very often a
/// 読点 or a bracket, so a kanji-only pool comes back empty exactly where the
/// evidence was there. The CTC blank (decoded as the ideographic space) and
/// the placeholder itself are never offered.
pub fn is_offerable(ch: char) -> bool {
    ch != GAP_CHAR && !ch.is_whitespace() && !ch.is_control()
}

/// Candidates for the blank, best first, from the line's per-timestep top-K.
/// Duplicates collapse; the LM reorders the pool by `context` when one is
/// loaded, else the pool keeps its discovery order. Capped at `limit`.
pub fn generate(
    alternatives: &[Vec<(char, f32)>],
    limit: usize,
    lm: Option<&CharLm>,
    context: &[char],
) -> Vec<char> {
    let mut pool: Vec<char> = Vec::new();
    for alts in alternatives {
        for (ch, _) in alts {
            if is_offerable(*ch) && !pool.contains(ch) {
                pool.push(*ch);
            }
        }
    }
    let ordered = match lm {
        Some(lm) => lm.rank(context, &pool),
        None => pool,
    };
    ordered.into_iter().take(limit).collect()
}

/// Punctuation a gap most often holds. Class order is fixed; see [`fallback`].
pub const PUNCT_DEFAULTS: [char; 6] = ['、', '。', '「', '」', '…', 'ー'];

/// The kana it most often holds when it is not punctuation.
pub const KANA_DEFAULTS: [char; 5] = ['は', 'の', 'を', 'に', 'と'];

/// The fallback list, punctuation first, never empty. The class order is
/// deliberately *not* the model's: mobile measured that an order-4 character
/// model asked to prefer punctuation over a continuation is really running a
/// frequency contest, so the model orders *within* a class and the classes stay
/// in this order.
pub fn fallback(limit: usize, lm: Option<&CharLm>, context: &[char]) -> Vec<char> {
    let punct = match lm {
        Some(lm) => lm.rank(context, &PUNCT_DEFAULTS),
        None => PUNCT_DEFAULTS.to_vec(),
    };
    let kana = match lm {
        Some(lm) => lm.rank(context, &KANA_DEFAULTS),
        None => KANA_DEFAULTS.to_vec(),
    };
    punct.into_iter().chain(kana).take(limit).collect()
}

/// The characters before the gap, the context the back-off chain can use: the
/// model is order 4, so anything longer is ignored and the placeholder itself
/// is dropped rather than read as a real character.
pub fn context_before(text: &str, index: usize) -> Vec<char> {
    let chars: Vec<char> = text.chars().collect();
    let end = index.min(chars.len());
    let mut start = end;
    while start > 0 && chars[start - 1] != GAP_CHAR && end - start < MAX_ORDER - 1 {
        start -= 1;
    }
    chars[start..end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy table packed exactly like `tools/pack_char_lm.py` writes it.
    fn toy_lm() -> CharLm {
        let entries: &[(&str, u16)] = &[
            ("私", 100),
            ("の", 200),
            ("を", 50),
            ("は", 30),
            ("、", 80),
            ("。", 70),
            ("私の", 40),
            ("私を", 30),
            ("のは", 60),
            ("今日", 90),
            ("日は", 50),
        ];
        let mut records: Vec<([u16; 4], u16)> = entries
            .iter()
            .map(|(ngram, count)| {
                let mut units = [0u16; 4];
                for (i, ch) in ngram.chars().enumerate() {
                    units[i] = ch as u16;
                }
                (units, *count)
            })
            .collect();
        records.sort_by(|a, b| a.0.cmp(&b.0));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x314D_4C43u32.to_le_bytes());
        bytes.extend_from_slice(&(records.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&500u32.to_le_bytes());
        for (units, count) in records {
            for unit in units {
                bytes.extend_from_slice(&unit.to_le_bytes());
            }
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        CharLm::from_bytes(bytes).expect("toy table")
    }

    /// The pool spans every timestep in first-seen order; the CTC blank (the
    /// ideographic space) and the placeholder are dropped, duplicates collapse.
    #[test]
    fn the_pool_comes_from_every_timestep_in_discovery_order() {
        let raw = vec![
            vec![('、', 0.9), ('の', 0.5)],
            vec![('\u{3000}', 0.9), ('の', 0.8), ('私', 0.3)],
            vec![('\u{25CC}', 0.1), ('。', 0.2)],
        ];
        assert_eq!(
            generate(&raw, MAX, None, &[]),
            vec!['、', 'の', '私', '。']
        );
    }

    /// With a model, the same pool is reordered by the line context: after 私
    /// the model prefers の over を even though を came first from the
    /// recogniser.
    #[test]
    fn the_model_reorders_the_pool_by_context() {
        let lm = toy_lm();
        let raw = vec![vec![('を', 0.9)], vec![('の', 0.8), ('私', 0.1)]];
        assert_eq!(generate(&raw, MAX, None, &[]), vec!['を', 'の', '私']);
        assert_eq!(
            generate(&raw, MAX, Some(&lm), &['私']),
            vec!['の', 'を', '私'],
            "the context prior leads"
        );
    }

    /// An empty pool is the one case the fallback exists for: punctuation
    /// leads, kana follows, and the model only orders within a class.
    #[test]
    fn an_empty_pool_falls_back_to_punctuation_then_kana() {
        assert!(generate(&[], MAX, None, &[]).is_empty());
        let fb = fallback(MAX, None, &[]);
        assert_eq!(fb[0], '、');
        let kana_at = fb.iter().position(|c| *c == 'は').expect("kana in fallback");
        let last_punct = fb
            .iter()
            .rposition(|c| PUNCT_DEFAULTS.contains(c))
            .expect("punctuation in fallback");
        assert!(last_punct < kana_at, "punctuation precedes kana: {fb:?}");

        // With a model the classes stay in order; only their members move.
        let lm = toy_lm();
        let fb = fallback(MAX, Some(&lm), &['私']);
        assert!(PUNCT_DEFAULTS.contains(&fb[0]), "punctuation still leads");
        let last_punct = fb
            .iter()
            .rposition(|c| PUNCT_DEFAULTS.contains(c))
            .expect("punctuation in fallback");
        let kana_at = fb.iter().position(|c| KANA_DEFAULTS.contains(c)).expect("kana");
        assert!(last_punct < kana_at, "punctuation precedes kana: {fb:?}");
    }

    /// The context is the characters before the gap, capped at the model's
    /// order; a placeholder directly before the gap blocks the back-off
    /// (it is never read as a real character).
    #[test]
    fn the_context_drops_the_placeholder_and_older_characters() {
        assert_eq!(context_before("あいうえお", 5), vec!['う', 'え', 'お']);
        assert_eq!(context_before("あいうえお", 2), vec!['あ', 'い']);
        assert_eq!(context_before(&format!("あ{GAP_CHAR}う"), 2), Vec::<char>::new());
        assert_eq!(context_before("", 0), Vec::<char>::new());
    }

    #[test]
    fn offerable_rejects_the_placeholder_the_blank_and_controls() {
        assert!(!is_offerable('\u{25CC}'), "the placeholder itself");
        assert!(!is_offerable('\u{3000}'), "the CTC blank");
        assert!(!is_offerable('\n'));
        assert!(is_offerable('、'));
        assert!(is_offerable('私'));
        assert!(is_offerable('a'));
    }
}
