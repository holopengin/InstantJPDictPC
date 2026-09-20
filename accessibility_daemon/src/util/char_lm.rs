//! Mobile `CharLm` (#44): the character n-gram text prior that ranks blank
//! candidates where shape alone cannot — the deletion pools are hundreds of
//! candidates and the component ordering over them is degenerate, so the LM is
//! the only ranker available.
//!
//! ## File format (`assets/lm/char_lm.bin`, written by mobile's
//! `tools/pack_char_lm.py`)
//!
//! ```text
//! header  16 bytes  magic "CLM1", entry count (u32), max order (u32),
//!                   unigram mass (u32)                      — little endian
//! record  10 bytes  n-gram: up to 4 UTF-16 code units, zero-padded on the
//!                   right, then the count saturated at 65535
//! ```
//!
//! Records sort by the n-gram's own code-unit order (zero-padded), so a shorter
//! n-gram sorts immediately before its extensions and one binary search finds
//! an entry of any order. U+0000 never occurs in Japanese text, which is what
//! makes the padding unambiguous. Counts are saturated at 16 bits because they
//! are only ever used as ratios.
//!
//! The shipped model: Aozora Bunko, 5,000,000 characters, order 4, min count
//! ≥ 5 at the highest order — see `assets/lm/PROVENANCE.txt`.

pub const MAX_ORDER: usize = 4;
const HEADER: usize = 16;
const RECORD: usize = 10;
/// "CLM1" little endian.
const MAGIC: u32 = 0x314D_4C43;
/// Unknown characters sort last rather than tie with real evidence.
const UNSEEN: f32 = -30.0;

pub struct CharLm {
    data: Vec<u8>,
    entries: usize,
    /// Corpus length, from the header: the unigram prior needs it and a scan to
    /// recover it would cost 1.4M records per candidate.
    unigram_mass: u32,
}

impl CharLm {
    /// Wrap packed bytes. `None` when the header or length does not describe a
    /// table.
    pub fn from_bytes(data: Vec<u8>) -> Option<Self> {
        if data.len() < HEADER {
            return None;
        }
        let word = |at: usize| u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
        let magic = word(0);
        let count = word(4) as usize;
        let order = word(8);
        let mass = word(12);
        if magic != MAGIC || !(1..=MAX_ORDER as u32).contains(&order) || count == 0 || mass == 0 {
            return None;
        }
        if data.len() < HEADER + count * RECORD {
            return None;
        }
        Some(Self {
            data,
            entries: count,
            unigram_mass: mass,
        })
    }

    /// Load the shipped model; `None` when the file is missing or malformed
    /// (the blank's list then keeps its discovery order).
    pub fn load(path: &std::path::Path) -> Option<Self> {
        Self::from_bytes(std::fs::read(path).ok()?)
    }

    /// Entry count from the header (for logs and tests).
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// Occurrences of `ngram`, or 0 when it is unknown (or longer than the
    /// model's order). Supplementary-plane characters were skipped by the
    /// packer, so they read as unknown too.
    pub fn count(&self, ngram: &[char]) -> u32 {
        if ngram.is_empty() || ngram.len() > MAX_ORDER {
            return 0;
        }
        let mut wanted = [0u16; MAX_ORDER];
        for (i, ch) in ngram.iter().enumerate() {
            let code = *ch as u32;
            if code > 0xFFFF {
                return 0;
            }
            wanted[i] = code as u16;
        }
        let (mut lo, mut hi) = (0usize, self.entries - 1);
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let base = HEADER + mid * RECORD;
            let mut cmp = std::cmp::Ordering::Equal;
            for i in 0..MAX_ORDER {
                let got = u16::from_le_bytes([self.data[base + i * 2], self.data[base + i * 2 + 1]]);
                cmp = got.cmp(&wanted[i]);
                if cmp != std::cmp::Ordering::Equal {
                    break;
                }
            }
            match cmp {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => {
                    // A target smaller than every record: mid is already 0.
                    if mid == 0 {
                        break;
                    }
                    hi = mid - 1;
                }
                std::cmp::Ordering::Equal => {
                    return u16::from_le_bytes([self.data[base + 8], self.data[base + 9]]) as u32;
                }
            }
        }
        0
    }

    /// `log P(ch | context)`: the longest context the model knows, backed off
    /// one character at a time, and a unigram prior when nothing longer is
    /// known. Unknown characters score [`UNSEEN`] so they sort last.
    pub fn log_prob(&self, context: &[char], ch: char) -> f32 {
        let max_ctx = MAX_ORDER - 1;
        let start = context.len().saturating_sub(max_ctx);
        let ctx = &context[start..];
        for len in (1..=ctx.len()).rev() {
            let joint: Vec<char> = ctx[ctx.len() - len..].iter().copied().chain([ch]).collect();
            let joint_count = self.count(&joint);
            if joint_count > 0 {
                let denom = self.count(&ctx[ctx.len() - len..]);
                if denom > 0 {
                    return (joint_count as f32 / denom as f32).ln();
                }
            }
        }
        let uni = self.count(&[ch]);
        if uni > 0 && self.unigram_mass > 0 {
            (uni as f32 / self.unigram_mass as f32).ln()
        } else {
            UNSEEN
        }
    }

    /// `candidates` ordered by how well `context` predicts them, best first.
    /// Ties keep the input order, so a caller's own ranking still breaks them
    /// deterministically.
    pub fn rank(&self, context: &[char], candidates: &[char]) -> Vec<char> {
        let mut scored: Vec<(usize, char, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, *c, self.log_prob(context, *c)))
            .collect();
        scored.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
        scored.into_iter().map(|(_, c, _)| c).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy table packed exactly like `tools/pack_char_lm.py` writes it:
    /// header + 10-byte records sorted by zero-padded code units.
    fn packed(entries: &[(&str, u16)], mass: u32) -> Vec<u8> {
        let mut records: Vec<([u16; MAX_ORDER], u16)> = entries
            .iter()
            .map(|(ngram, count)| {
                let mut units = [0u16; MAX_ORDER];
                for (i, ch) in ngram.chars().enumerate() {
                    units[i] = ch as u16;
                }
                (units, *count)
            })
            .collect();
        records.sort_by(|a, b| a.0.cmp(&b.0));
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&(records.len() as u32).to_le_bytes());
        out.extend_from_slice(&(MAX_ORDER as u32).to_le_bytes());
        out.extend_from_slice(&mass.to_le_bytes());
        for (units, count) in records {
            for unit in units {
                out.extend_from_slice(&unit.to_le_bytes());
            }
            out.extend_from_slice(&count.to_le_bytes());
        }
        out
    }

    fn toy() -> CharLm {
        // A tiny "私の…" corpus: 私 100, の 200, を 50; 私の 40, 私を 5, のは 60.
        CharLm::from_bytes(packed(
            &[
                ("私", 100),
                ("の", 200),
                ("を", 50),
                ("は", 30),
                ("私の", 40),
                ("私を", 5),
                ("のは", 60),
                ("私は", 20),
            ],
            500,
        ))
        .expect("toy table")
    }

    #[test]
    fn count_finds_entries_of_every_order_and_misses_cleanly() {
        let lm = toy();
        assert_eq!(lm.count(&['私']), 100);
        assert_eq!(lm.count(&['私', 'の']), 40);
        assert_eq!(lm.count(&['私', 'の', 'は']), 0, "unknown trigram");
        assert_eq!(lm.count(&[]), 0, "empty");
        assert_eq!(lm.count(&['私', 'の', 'は', 'を', 'に']), 0, "too long");
    }

    /// The back-off chain: the longest known context wins, and a context the
    /// model does not know falls back to the unigram prior.
    #[test]
    fn log_prob_backs_off_from_the_longest_known_context() {
        let lm = toy();
        let bigram = lm.log_prob(&['私'], 'の'); // 40/100
        let unigram = lm.log_prob(&['誰'], 'の'); // 200/500
        assert!((bigram - (40.0f32 / 100.0).ln()).abs() < 1e-6);
        assert!((unigram - (200.0f32 / 500.0).ln()).abs() < 1e-6);
        assert!(lm.log_prob(&['誰'], '漢') <= UNSEEN + 1e-6, "unknown scores last");
    }

    /// Ranking follows the context: after 私 the model prefers の over を.
    #[test]
    fn rank_prefers_what_the_context_predicts() {
        let lm = toy();
        assert_eq!(lm.rank(&['私'], &['を', 'の']), vec!['の', 'を']);
        assert_eq!(lm.rank(&[], &['の', 'は']), vec!['の', 'は'], "unigram prior");
    }

    #[test]
    fn malformed_headers_are_rejected() {
        assert!(CharLm::from_bytes(Vec::new()).is_none());
        let mut short = packed(&[("の", 1)], 1);
        short.truncate(HEADER + RECORD - 1);
        assert!(CharLm::from_bytes(short).is_none(), "truncated records");
        let mut wrong_magic = packed(&[("の", 1)], 1);
        wrong_magic[0] = b'X';
        assert!(CharLm::from_bytes(wrong_magic).is_none());
    }

    /// The shipped Aozora model: sanity-check the header and that common
    /// Japanese is known while rare/unseen text scores last. Skipped when the
    /// asset is absent (e.g. a checkout without it).
    #[test]
    fn shipped_model_parses_and_knows_japanese() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/lm/char_lm.bin");
        let Some(lm) = CharLm::load(&path) else {
            eprintln!("skipping: {} not present", path.display());
            return;
        };
        // Header sanity (the shipped table is 1.43M entries).
        assert!(lm.entries > 1_000_000, "entries = {}", lm.entries);
        assert!(lm.unigram_mass > 1_000_000, "mass = {}", lm.unigram_mass);
        // の is the most common character in Japanese (its count saturates at
        // the 16-bit cap); a nonsense char is unseen.
        assert!(lm.count(&['の']) > 50_000, "の = {}", lm.count(&['の']));
        assert!(lm.count(&['私']) > 1_000, "私 = {}", lm.count(&['私']));
        assert_eq!(lm.count(&['𐐷']), 0, "supplementary plane is skipped");
        assert!(lm.log_prob(&['私'], 'の') > lm.log_prob(&['私'], '𠮟'));
        // The context prior works on real text: after 「今日」 the model prefers
        // は, after 「だから」 the comma, after 「定期船」 the の.
        assert_eq!(lm.rank(&['今', '日'], &['を', 'は']), vec!['は', 'を']);
        assert_eq!(lm.rank(&['だ', 'か', 'ら'], &['を', '、']), vec!['、', 'を']);
        assert_eq!(lm.rank(&['定', '期', '船'], &['を', 'の']), vec!['の', 'を']);
    }
}
