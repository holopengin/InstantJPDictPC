//! Mobile `GapCandidates` (#44): what to offer for a blank.
//!
//! A gap is only worth filling when there is evidence for what went there, so
//! the pool is the set of characters the recogniser itself offered along the
//! line — its per-timestep top-K ([`crate::models::LineResult::raw_alternatives`],
//! blanks included) — filtered to what a real character can be. Mobile then
//! reorders the pool with a bundled character language model; that asset is not
//! shipped on the PC, so the pool keeps its discovery order (recorded in
//! `04-full-parity-deferrals`). The fallback list (punctuation first, then
//! kana) is never empty: a blank with nothing to choose from is worse than a
//! guess.

use crate::models::GAP_CHAR;

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
/// Order is first-seen across the timesteps (the mobile model's context
/// ranking is not ported yet); duplicates collapse and the pool is capped at
/// `limit`.
pub fn generate(alternatives: &[Vec<(char, f32)>], limit: usize) -> Vec<char> {
    let mut pool: Vec<char> = Vec::new();
    for alts in alternatives {
        for (ch, _) in alts {
            if is_offerable(*ch) && !pool.contains(ch) {
                pool.push(*ch);
            }
        }
    }
    pool.truncate(limit);
    pool
}

/// Punctuation a gap most often holds. Class order is fixed; see [`fallback`].
pub const PUNCT_DEFAULTS: [char; 6] = ['、', '。', '「', '」', '…', 'ー'];

/// The kana it most often holds when it is not punctuation.
pub const KANA_DEFAULTS: [char; 5] = ['は', 'の', 'を', 'に', 'と'];

/// The fallback list, punctuation first, never empty. The class order is
/// deliberately *not* a model's: mobile measured that an order-4 character
/// model asked to prefer punctuation over a continuation is really running a
/// frequency contest, so the model orders *within* a class (when one is
/// loaded) and the classes stay in this order.
pub fn fallback(limit: usize) -> Vec<char> {
    PUNCT_DEFAULTS
        .iter()
        .chain(KANA_DEFAULTS.iter())
        .copied()
        .take(limit)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pool spans every timestep in first-seen order; the CTC blank (the
    /// ideographic space) and the placeholder are dropped, duplicates collapse.
    #[test]
    fn the_pool_comes_from_every_timestep_in_discovery_order() {
        let raw = vec![
            vec![('、', 0.9), ('の', 0.5)],
            vec![('\u{3000}', 0.9), ('の', 0.8), ('私', 0.3)],
            vec![('\u{25CC}', 0.1), ('。', 0.2)],
        ];
        assert_eq!(generate(&raw, MAX), vec!['、', 'の', '私', '。']);
    }

    /// An empty pool is the one case the fallback exists for: punctuation
    /// leads, kana follows.
    #[test]
    fn an_empty_pool_falls_back_to_punctuation_then_kana() {
        assert!(generate(&[], MAX).is_empty());
        let fb = fallback(MAX);
        assert_eq!(fb[0], '、');
        let kana_at = fb.iter().position(|c| *c == 'は').expect("kana in fallback");
        let last_punct = fb
            .iter()
            .rposition(|c| PUNCT_DEFAULTS.contains(c))
            .expect("punctuation in fallback");
        assert!(last_punct < kana_at, "punctuation precedes kana: {fb:?}");
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
