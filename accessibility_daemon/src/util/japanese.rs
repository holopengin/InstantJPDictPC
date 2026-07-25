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
/// Replaces standard CJK punctuation and fullwidth forms with Unicode
/// vertical presentation forms (U+FE10–U+FE48) so the font renders
/// them in the correct vertical writing position (periods/comma at top-right,
/// brackets rotated 90°, etc.).
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
