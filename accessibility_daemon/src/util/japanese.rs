//! Japanese text normalization utilities.
//! Mirrors `JapaneseUtil` from the Kotlin implementation.

use std::collections::HashMap;

/// Normalize Japanese text: convert full-width to half-width, standardize kana, etc.
pub fn normalize(text: &str) -> String {
    normalize_combining_characters(&convert_width(text))
}

/// Convert full-width characters to their standard equivalents.
fn convert_width(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1);

        // Check for half-width kana with dakuten (ﾞ)
        if next == Some(&'\u{FF9E}') {
            if let Some(mapped) = HALFWIDTH_VOICED_MAPPING.get(&c) {
                result.push(*mapped);
                i += 2;
                continue;
            }
        }

        // Check for half-width kana with handakuten (ﾟ)
        if next == Some(&'\u{FF9F}') {
            if let Some(mapped) = HALFWIDTH_SEMI_VOICED_MAPPING.get(&c) {
                result.push(*mapped);
                i += 2;
                continue;
            }
        }

        // Check basic half-width kana mapping
        if let Some(mapped) = HALFWIDTH_KANA_MAPPING.get(&c) {
            result.push(*mapped);
            i += 1;
            continue;
        }

        // Full-width ASCII range (FF01-FF5E) to standard ASCII
        if ('\u{FF01}'..='\u{FF5E}').contains(&c) {
            result.push((c as u32 - 0xFEE0) as u8 as char);
            i += 1;
            continue;
        }

        // Ideographic space to regular space
        if c == '\u{3000}' {
            result.push(' ');
            i += 1;
            continue;
        }

        result.push(c);
        i += 1;
    }

    result
}

/// Normalize combining characters (e.g. か + ゙ = が).
fn normalize_combining_characters(text: &str) -> String {
    text.replace("\u{304B}\u{3099}", "が")
        .replace("\u{304D}\u{3099}", "ぎ")
        .replace("\u{304F}\u{3099}", "ぐ")
        .replace("\u{3051}\u{3099}", "げ")
        .replace("\u{3053}\u{3099}", "ご")
        .replace("\u{3055}\u{3099}", "ざ")
        .replace("\u{3057}\u{3099}", "じ")
        .replace("\u{3059}\u{3099}", "ず")
        .replace("\u{305B}\u{3099}", "ぜ")
        .replace("\u{305D}\u{3099}", "ぞ")
        .replace("\u{305F}\u{3099}", "だ")
        .replace("\u{3061}\u{3099}", "ぢ")
        .replace("\u{3064}\u{3099}", "づ")
        .replace("\u{3066}\u{3099}", "で")
        .replace("\u{3068}\u{3099}", "ど")
        .replace("\u{306F}\u{3099}", "ば")
        .replace("\u{3072}\u{3099}", "び")
        .replace("\u{3075}\u{3099}", "ぶ")
        .replace("\u{3078}\u{3099}", "べ")
        .replace("\u{307B}\u{3099}", "ぼ")
        .replace("\u{306F}\u{309A}", "ぱ")
        .replace("\u{3072}\u{309A}", "ぴ")
        .replace("\u{3075}\u{309A}", "ぷ")
        .replace("\u{3078}\u{309A}", "ぺ")
        .replace("\u{307B}\u{309A}", "ぽ")
}

/// Convert katakana to hiragana, handling prolonged sound marks (ー).
pub fn katakana_to_hiragana(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len());

    for (i, &c) in chars.iter().enumerate() {
        if ('\u{30A1}'..='\u{30F6}').contains(&c) {
            result.push((c as u32 - 0x60) as u8 as char);
        } else if c == 'ー' && i > 0 {
            result.push(get_prolonged_hiragana(chars[i - 1]));
        } else {
            result.push(c);
        }
    }

    result
}

fn get_prolonged_hiragana(prev: char) -> char {
    match prev {
        'あ' | 'か' | 'さ' | 'た' | 'な' | 'は' | 'ま' | 'や' | 'ら' | 'わ' | 'ァ' | 'カ' | 'サ'
        | 'タ' | 'ナ' | 'ハ' | 'マ' | 'ヤ' | 'ラ' | 'ワ' => 'あ',
        'い' | 'き' | 'し' | 'ち' | 'に' | 'ひ' | 'み' | 'り' | 'ィ' | 'キ' | 'シ' | 'チ' | 'ニ'
        | 'ヒ' | 'ミ' | 'リ' => 'い',
        'う' | 'く' | 'す' | 'つ' | 'ぬ' | 'ふ' | 'む' | 'ゆ' | 'る' | 'ゥ' | 'ク' | 'ス' | 'ツ'
        | 'ヌ' | 'フ' | 'ム' | 'ユ' | 'ル' | 'ヴ' => 'う',
        'え' | 'け' | 'せ' | 'て' | 'ね' | 'へ' | 'め' | 'れ' | 'ェ' | 'ケ' | 'セ' | 'テ' | 'ネ'
        | 'ヘ' | 'メ' | 'レ' => 'え',
        'お' | 'こ' | 'そ' | 'と' | 'の' | 'ほ' | 'も' | 'よ' | 'ろ' | 'ォ' | 'コ' | 'ソ' | 'ト'
        | 'ノ' | 'ホ' | 'モ' | 'ヨ' | 'ロ' => 'う',
        _ => 'う',
    }
}

/// Collapse consecutive emphatic characters (っ, ッ, ー, ～).
pub fn collapse_emphatic(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len());
    result.push(chars[0]);

    for &c in &chars[1..] {
        let last = result.chars().last().unwrap();
        if (c == 'っ' || c == 'ッ' || c == 'ー' || c == '～') && c == last {
            continue;
        }
        result.push(c);
    }

    result
}

// ---------------------------------------------------------------------------
// Half-width kana mappings
// ---------------------------------------------------------------------------

lazy_static::lazy_static! {
    static ref HALFWIDTH_KANA_MAPPING: HashMap<char, char> = {
        let pairs = [
            ('｡', '。'), ('｢', '「'), ('｣', '」'), ('､', '、'), ('･', '・'),
            ('ｦ', 'ヲ'), ('ｧ', 'ァ'), ('ｨ', 'ィ'), ('ｩ', 'ゥ'), ('ｪ', 'ェ'), ('ｫ', 'ォ'),
            ('ｬ', 'ャ'), ('ｭ', 'ュ'), ('ｮ', 'ョ'), ('ｯ', 'ッ'), ('ｰ', 'ー'),
            ('ｱ', 'ア'), ('ｲ', 'イ'), ('ｳ', 'ウ'), ('ｴ', 'エ'), ('ｵ', 'オ'),
            ('ｶ', 'カ'), ('ｷ', 'キ'), ('ｸ', 'ク'), ('ｹ', 'ケ'), ('ｺ', 'コ'),
            ('ｻ', 'サ'), ('ｼ', 'シ'), ('ｽ', 'ス'), ('ｾ', 'セ'), ('ｿ', 'ソ'),
            ('ﾀ', 'タ'), ('ﾁ', 'チ'), ('ﾂ', 'ツ'), ('ﾃ', 'テ'), ('ﾄ', 'ト'),
            ('ﾅ', 'ナ'), ('ﾆ', 'ニ'), ('ﾇ', 'ヌ'), ('ﾈ', 'ネ'), ('ﾉ', 'ノ'),
            ('ﾊ', 'ハ'), ('ﾋ', 'ヒ'), ('ﾌ', 'フ'), ('ﾍ', 'ヘ'), ('ﾎ', 'ホ'),
            ('ﾏ', 'マ'), ('ﾐ', 'ミ'), ('ﾑ', 'ム'), ('ﾒ', 'メ'), ('ﾓ', 'モ'),
            ('ﾔ', 'ヤ'), ('ﾕ', 'ユ'), ('ﾖ', 'ヨ'),
            ('ﾗ', 'ラ'), ('ﾘ', 'リ'), ('ﾙ', 'ル'), ('ﾚ', 'レ'), ('ﾛ', 'ロ'),
            ('ﾜ', 'ワ'), ('ﾝ', 'ン'),
        ];
        pairs.iter().copied().collect()
    };

    static ref HALFWIDTH_VOICED_MAPPING: HashMap<char, char> = {
        let pairs = [
            ('ｶ', 'ガ'), ('ｷ', 'ギ'), ('ｸ', 'グ'), ('ｹ', 'ゲ'), ('ｺ', 'ゴ'),
            ('ｻ', 'ザ'), ('ｼ', 'ジ'), ('ｽ', 'ズ'), ('ｾ', 'ゼ'), ('ｿ', 'ゾ'),
            ('ﾀ', 'ダ'), ('ﾁ', 'ヂ'), ('ﾂ', 'ヅ'), ('ﾃ', 'デ'), ('ﾄ', 'ド'),
            ('ﾊ', 'バ'), ('ﾋ', 'ビ'), ('ﾌ', 'ブ'), ('ﾍ', 'ベ'), ('ﾎ', 'ボ'),
            ('ｳ', 'ヴ'),
        ];
        pairs.iter().copied().collect()
    };

    static ref HALFWIDTH_SEMI_VOICED_MAPPING: HashMap<char, char> = {
        let pairs = [
            ('ﾊ', 'パ'), ('ﾋ', 'ピ'), ('ﾌ', 'プ'), ('ﾍ', 'ペ'), ('ﾎ', 'ポ'),
        ];
        pairs.iter().copied().collect()
    };
}

/// Convert a CJK character to its vertical-mode presentation form.
/// Unicode vertical presentation form (U+FE10–U+FE48), used by the overlay
/// only as a fallback for characters the font's GSUB `vert`/`vrt2` feature
/// does not cover (e.g. ！ ？ … ；). GSUB-covered characters (、。「」 etc.)
/// never reach this table — the font's own vertical glyph wins.
pub fn to_vertical_glyph(ch: char) -> char {
    match ch {
        // ── FE10-FE19: Vertical Forms block ──
        ','          | '\u{FF0C}' => '\u{FE10}', // ,  ， -> VERTICAL COMMA
        '\u{3001}'                => '\u{FE11}', // 、   -> VERTICAL IDEOGRAPHIC COMMA
        '\u{3002}'  | '\u{FF0E}' => '\u{FE12}', // 。 ． -> VERTICAL IDEOGRAPHIC FULL STOP
        ':'         | '\u{FF1A}' => '\u{FE13}', // :  ： -> VERTICAL COLON
        ';'         | '\u{FF1B}' => '\u{FE14}', // ;  ； -> VERTICAL SEMICOLON
        '!'         | '\u{FF01}' => '\u{FE15}', // !  ！ -> VERTICAL EXCLAMATION MARK
        '?'         | '\u{FF1F}' => '\u{FE16}', // ?  ？ -> VERTICAL QUESTION MARK
        '\u{3016}'                => '\u{FE17}', // 〖   -> VERTICAL LEFT WHITE LENTICULAR BRACKET
        '\u{3017}'                => '\u{FE18}', // 〗   -> VERTICAL RIGHT WHITE LENTICULAR BRACKET
        '\u{2026}'                => '\u{FE19}', // …   -> VERTICAL HORIZONTAL ELLIPSIS

        // ── FE30-FE48: CJK Compatibility Forms (vertical variants) ──
        '\u{2025}'                => '\u{FE30}', // ‥   -> VERTICAL TWO DOT LEADER
        '\u{2014}' | '\u{30FC}'  => '\u{FE31}', // — ー -> VERTICAL EM DASH
        '\u{2013}'                => '\u{FE32}', // –   -> VERTICAL EN DASH
        '_'                       => '\u{FE33}', // _   -> VERTICAL LOW LINE
        '('         | '\u{FF08}' => '\u{FE35}', // (  （ -> VERTICAL LEFT PARENTHESIS
        ')'         | '\u{FF09}' => '\u{FE36}', // )  ） -> VERTICAL RIGHT PARENTHESIS
        '{'                       => '\u{FE37}', // {   -> VERTICAL LEFT CURLY BRACKET
        '}'                       => '\u{FE38}', // }   -> VERTICAL RIGHT CURLY BRACKET
        '\u{3014}'                => '\u{FE39}', // 〔   -> VERTICAL LEFT TORTOISE SHELL BRACKET
        '\u{3015}'                => '\u{FE3A}', // 〕   -> VERTICAL RIGHT TORTOISE SHELL BRACKET
        '\u{3010}'                => '\u{FE3B}', // 【   -> VERTICAL LEFT BLACK LENTICULAR BRACKET
        '\u{3011}'                => '\u{FE3C}', // 】   -> VERTICAL RIGHT BLACK LENTICULAR BRACKET
        '\u{300A}'                => '\u{FE3D}', // 《   -> VERTICAL LEFT DOUBLE ANGLE BRACKET
        '\u{300B}'                => '\u{FE3E}', // 》   -> VERTICAL RIGHT DOUBLE ANGLE BRACKET
        '\u{3008}'                => '\u{FE3F}', // 〈   -> VERTICAL LEFT ANGLE BRACKET
        '\u{3009}'                => '\u{FE40}', // 〉   -> VERTICAL RIGHT ANGLE BRACKET
        '\u{300C}'                => '\u{FE41}', // 「   -> VERTICAL LEFT CORNER BRACKET
        '\u{300D}'                => '\u{FE42}', // 」   -> VERTICAL RIGHT CORNER BRACKET
        '\u{300E}'                => '\u{FE43}', // 『   -> VERTICAL LEFT WHITE CORNER BRACKET
        '\u{300F}'                => '\u{FE44}', // 』   -> VERTICAL RIGHT WHITE CORNER BRACKET
        '['         | '\u{FF3B}' => '\u{FE47}', // [  ［ -> VERTICAL LEFT SQUARE BRACKET
        ']'         | '\u{FF3D}' => '\u{FE48}', // ]  ］ -> VERTICAL RIGHT SQUARE BRACKET

        _ => ch,
    }
}

/// Mobile `OcrEngine.isHalfWidth` (#49): ASCII + halfwidth katakana advance
/// at 0.5 em; everything else (JP, fullwidth latin, punctuation) at 1.0 em.
pub fn is_half_width(ch: char) -> bool {
    let cp = ch as u32;
    cp <= 0x7E || (0xFF61..=0xFFDC).contains(&cp)
}

/// Mobile `OcrEngine.estimateEm` (#49): the line's em from width-normalized
/// pitches. Every gap is divided by the mean advance of its two characters
/// (0.5 halfwidth, 1.0 fullwidth), so ASCII-majority mixed lines cannot drag
/// the estimate to ~0.5×; the median of the normalized gaps is the em.
/// Returns 0 when unestimable (fewer than 2 chars, mismatched lengths).
pub fn estimate_em(text: &str, centers: &[f32]) -> f32 {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() != centers.len() || centers.len() < 2 {
        return 0.0;
    }
    let mut norm: Vec<f32> = Vec::with_capacity(centers.len() - 1);
    for i in 0..centers.len() - 1 {
        let gap = (centers[i + 1] - centers[i]).abs();
        if gap <= 0.0 {
            continue;
        }
        let units = ((if is_half_width(chars[i]) { 0.5 } else { 1.0 })
            + (if is_half_width(chars[i + 1]) { 0.5 } else { 1.0 }))
            / 2.0;
        norm.push(gap / units);
    }
    if norm.is_empty() {
        return 0.0;
    }
    norm.sort_by(f32::total_cmp);
    norm[norm.len() / 2]
}

/// Split a space-separated KANJIDIC kana list into readings, stripping the
/// leading `-` from suffix-only readings. Mirrors `JapaneseUtil.splitKanaList`.
pub fn split_kana_list(raw: &str) -> Vec<String> {
    raw.split([' ', '\u{3000}'])
        .map(|s| s.trim_start_matches('-').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// One furigana run: `base` surface text with optional `ruby` above it.
#[derive(Debug, Clone, PartialEq)]
pub struct RubySegment {
    pub base: String,
    pub ruby: Option<String>,
}

/// True for CJK ideographs (and the iteration marks 々/〻).
pub fn is_kanji_char(c: char) -> bool {
    c == '々'
        || c == '〻'
        || ('\u{3400}'..='\u{4DBF}').contains(&c)
        || ('\u{4E00}'..='\u{9FFF}').contains(&c)
        || ('\u{F900}'..='\u{FAFF}').contains(&c)
}

/// Align a dictionary reading against its headword so ruby is shown only over
/// kanji spans, with okurigana/kana rendered as plain base text (#55).
/// Mirrors `FuriganaAligner.align`; None when the reading cannot be
/// unambiguously aligned (callers fall back to full-reading ruby).
pub fn align_furigana(term: &str, reading: &str) -> Option<Vec<RubySegment>> {
    if term.is_empty() || reading.is_empty() {
        return None;
    }
    if term == reading {
        return Some(vec![RubySegment {
            base: term.to_string(),
            ruby: None,
        }]);
    }
    // katakana_to_hiragana is 1:1 per char, so indices transfer to `reading`.
    let norm_reading = katakana_to_hiragana(reading);
    if norm_reading.chars().count() != reading.chars().count() {
        return None;
    }
    let norm: Vec<char> = norm_reading.chars().collect();
    let read: Vec<char> = reading.chars().collect();
    let term_chars: Vec<char> = term.chars().collect();

    let mut segments: Vec<RubySegment> = Vec::new();
    let mut ti = 0usize;
    let mut ri = 0usize;
    while ti < term_chars.len() {
        let c = term_chars[ti];
        if !is_kanji_char(c) {
            // Kana literal: must match the reading at the current position.
            if ri >= norm.len() {
                return None;
            }
            let c_hira = katakana_to_hiragana(&c.to_string());
            if c_hira.chars().next() != Some(norm[ri]) {
                return None;
            }
            match segments.last_mut() {
                Some(last) if last.ruby.is_none() => last.base.push(c),
                _ => segments.push(RubySegment {
                    base: c.to_string(),
                    ruby: None,
                }),
            }
            ti += 1;
            ri += 1;
        } else {
            // Kanji run: ruby is everything up to the next kana anchor.
            let mut tj = ti;
            while tj < term_chars.len() && is_kanji_char(term_chars[tj]) {
                tj += 1;
            }
            if tj < term_chars.len() {
                let anchor = katakana_to_hiragana(&term_chars[tj].to_string());
                let anchor = anchor.chars().next()?;
                let idx = norm[ri..].iter().position(|&x| x == anchor)? + ri;
                let ruby: String = read[ri..idx].iter().collect();
                if ruby.is_empty() {
                    return None;
                }
                segments.push(RubySegment {
                    base: term_chars[ti..tj].iter().collect(),
                    ruby: Some(ruby),
                });
                ri = idx;
            } else {
                let ruby: String = read[ri..].iter().collect();
                if ruby.is_empty() {
                    return None;
                }
                segments.push(RubySegment {
                    base: term_chars[ti..tj].iter().collect(),
                    ruby: Some(ruby),
                });
                ri = read.len();
            }
            ti = tj;
        }
    }
    if ri != norm.len() {
        return None;
    }
    Some(segments)
}

/// Small kana (拗音) fuse with the preceding kana into one mora.
const FUSING_KANA: &str = "ぁぃぅぇぉゃゅょゎゕゖァィゥェォャュョヮヵヶ";

/// Split a reading into morae (きょう = 2; がっこう = 4; コーヒー = 4).
pub fn morae_of(reading: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in reading.chars() {
        if !out.is_empty() && FUSING_KANA.contains(c) {
            out.last_mut().unwrap().push(c);
        } else {
            out.push(c.to_string());
        }
    }
    out
}

/// High/low per mora for a Yomitan downstep position (0 = heiban).
pub fn pitch_pattern(mora_count: usize, position: i32) -> Vec<bool> {
    if mora_count == 0 {
        return Vec::new();
    }
    let pos = position as usize;
    if position <= 0 {
        (0..mora_count).map(|i| i >= 1).collect()
    } else if position == 1 {
        (0..mora_count).map(|i| i == 0).collect()
    } else {
        (0..mora_count).map(|i| i >= 1 && i < pos).collect()
    }
}

/// True when the downstep lands past the final mora (odaka): the following
/// particle carries the fall, so the renderer draws a fall arrow.
pub fn falls_beyond_word(mora_count: usize, position: i32) -> bool {
    mora_count > 0 && position > 0 && position as usize >= mora_count
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirror of mobile `UniformEmTest`: true em 20px in every case.
    #[test]
    fn estimate_em_normalizes_halfwidth_advances() {
        let assert_em = |got: f32| assert!((got - 20.0).abs() < 1e-3, "em={got}");
        assert_em(estimate_em("AB日本CD", &[5.0, 15.0, 30.0, 50.0, 65.0, 75.0]));
        assert_em(estimate_em("日本語", &[10.0, 30.0, 50.0]));
        assert_em(estimate_em("ABCD", &[5.0, 15.0, 25.0, 35.0]));
        assert_em(estimate_em("ＡＢ", &[10.0, 30.0]));
        assert_em(estimate_em("ｱｲ", &[5.0, 15.0]));
        assert_eq!(estimate_em("あ", &[10.0]), 0.0);
        assert_eq!(estimate_em("", &[]), 0.0);
        assert_eq!(estimate_em("あい", &[10.0]), 0.0);
    }

    #[test]
    fn halfwidth_classification_matches_mobile() {
        assert!(is_half_width('A'));
        assert!(is_half_width('~'));
        assert!(is_half_width('\u{FF76}')); // ｶ halfwidth katakana
        assert!(!is_half_width('あ'));
        assert!(!is_half_width('漢'));
        assert!(!is_half_width('。'));
    }

    /// Mirrors Android `FuriganaAligner` examples: ruby over kanji runs only,
    /// okurigana as plain base text, fallback None when unalignable.
    #[test]
    fn furigana_aligner_covers_kanji_only() {
        assert_eq!(
            align_furigana("食べる", "たべる").unwrap(),
            vec![
                RubySegment { base: "食".into(), ruby: Some("た".into()) },
                RubySegment { base: "べる".into(), ruby: None },
            ]
        );
        assert_eq!(
            align_furigana("大きい", "おおきい").unwrap(),
            vec![
                RubySegment { base: "大".into(), ruby: Some("おお".into()) },
                RubySegment { base: "きい".into(), ruby: None },
            ]
        );
        assert_eq!(
            align_furigana("申し込む", "もうしこむ").unwrap(),
            vec![
                RubySegment { base: "申".into(), ruby: Some("もう".into()) },
                RubySegment { base: "し".into(), ruby: None },
                RubySegment { base: "込".into(), ruby: Some("こ".into()) },
                RubySegment { base: "む".into(), ruby: None },
            ]
        );
        // No kana anchor: whole ruby, as before.
        assert_eq!(
            align_furigana("今日", "きょう").unwrap(),
            vec![RubySegment { base: "今日".into(), ruby: Some("きょう".into()) }]
        );
        // Unalignable (reading shorter than the term).
        assert!(align_furigana("食べる", "たべ").is_none());
    }

    #[test]
    fn kana_list_split_strips_suffix_dashes() {
        assert_eq!(split_kana_list("しじぐい しじくい"), vec!["しじぐい", "しじくい"]);
        assert_eq!(split_kana_list("ぶん -ぶん"), vec!["ぶん", "ぶん"]);
        assert_eq!(split_kana_list("  "), Vec::<String>::new());
    }

    /// Mirrors `PitchAccent`: small kana fuse; っ/ー/ん keep their own mora.
    #[test]
    fn pitch_morae_and_contour() {
        assert_eq!(morae_of("きょう"), vec!["きょ", "う"]);
        assert_eq!(morae_of("がっこう").len(), 4);
        assert_eq!(morae_of("コーヒー").len(), 4);
        assert_eq!(pitch_pattern(4, 0), vec![false, true, true, true]);
        assert_eq!(pitch_pattern(4, 1), vec![true, false, false, false]);
        assert_eq!(pitch_pattern(4, 3), vec![false, true, true, false]);
        assert!(falls_beyond_word(4, 4));
        assert!(!falls_beyond_word(4, 3));
        assert!(!falls_beyond_word(0, 0));
    }
}
