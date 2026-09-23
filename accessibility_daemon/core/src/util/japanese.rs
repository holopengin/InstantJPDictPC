//! Japanese text normalization utilities.
//! Mirrors `JapaneseUtil` from the Kotlin implementation.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Normalize Japanese text: convert full-width to half-width, standardize kana, etc.
///
/// Stage order is part of the contract: width conversion first (so a mark sees
/// the widened kana), then the lookup-variant fold, then combining-character
/// normalization.
pub fn normalize(text: &str) -> String {
    normalize_combining_characters(&fold_lookup_variants(&convert_width(text)))
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

        // Vertical presentation forms fold back to the horizontal forms the
        // recogniser emits, so lookup of a vertical line matches dictionary
        // text. Outside the fullwidth-ASCII range, so they need explicit cases.
        if c == '\u{FE19}' {
            result.push('\u{2026}'); // ︙ -> …
            i += 1;
            continue;
        }
        if c == '\u{FE30}' {
            result.push('\u{2025}'); // ︰ -> ‥
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

/// Fold the variant characters of [`LOOKUP_VARIANT_MAP`] and expand the
/// iteration marks `ゝ`/`ゞ`/`ヽ`/`ヾ`, which repeat the preceding kana
/// (`こゝろ` → `こころ`, `たゞ` → `ただ`); a query containing one of them
/// matches nothing in a modern dictionary.
///
/// A voiced mark voices the repeat when the preceding kana has a voiced form
/// and falls back to a plain repeat otherwise (`まゞ` → `まま`, which is also
/// what an already-voiced kana needs: `がゞ` → `がが`). A mark whose preceding
/// character is not kana of the matching script (line-initial, after a kanji
/// or punctuation) is left as-is rather than folded into a guess, and marks do
/// not cross scripts (`カゝ` and `あヽ` stay). The preceding character is the
/// last one already emitted, so a run of marks repeats the expanded run
/// (`こゝゝ` → `こここ`).
///
/// Query-side only: callers keep using the raw text for display and for the
/// prefix lengths a match corresponds to, so a fold may change the query's
/// length without affecting what is shown.
pub fn fold_lookup_variants(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    for c in text.chars() {
        let last = result.chars().last();
        let is_hira = last.map_or(false, |l| ('\u{3041}'..='\u{3096}').contains(&l));
        let is_kata = last.map_or(false, |l| ('\u{30A1}'..='\u{30F6}').contains(&l));
        match c {
            'ゝ' => result.push(if is_hira { last.unwrap() } else { c }),
            'ゞ' => result.push(match last {
                Some(l) if is_hira => HIRAGANA_VOICED.get(&l).copied().unwrap_or(l),
                _ => c,
            }),
            'ヽ' => result.push(if is_kata { last.unwrap() } else { c }),
            'ヾ' => result.push(match last {
                Some(l) if is_kata => KATAKANA_VOICED.get(&l).copied().unwrap_or(l),
                _ => c,
            }),
            _ => match LOOKUP_VARIANT_MAP.get(&c) {
                Some(mapped) => result.push_str(mapped),
                None => result.push(c),
            },
        }
    }
    result
}

/// Convert katakana to hiragana, handling prolonged sound marks (ー).
pub fn katakana_to_hiragana(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len());

    for (i, &c) in chars.iter().enumerate() {
        if ('\u{30A1}'..='\u{30F6}').contains(&c) {
            // The hiragana block sits exactly 0x60 below the katakana block;
            // never narrow through u8 (that truncated ア to 'B').
            result.push(
                char::from_u32(c as u32 - 0x60).expect("katakana offset stays in the hiragana block"),
            );
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

// ---------------------------------------------------------------------------
// Lookup-variant fold tables
// ---------------------------------------------------------------------------

lazy_static::lazy_static! {
    /// Kana and their voiced counterparts, for the iteration marks `ゞ`/`ヾ`.
    static ref HIRAGANA_VOICED: HashMap<char, char> = {
        let pairs = [
            ('か', 'が'), ('き', 'ぎ'), ('く', 'ぐ'), ('け', 'げ'), ('こ', 'ご'),
            ('さ', 'ざ'), ('し', 'じ'), ('す', 'ず'), ('せ', 'ぜ'), ('そ', 'ぞ'),
            ('た', 'だ'), ('ち', 'ぢ'), ('つ', 'づ'), ('て', 'で'), ('と', 'ど'),
            ('は', 'ば'), ('ひ', 'び'), ('ふ', 'ぶ'), ('へ', 'べ'), ('ほ', 'ぼ'),
            ('う', 'ゔ'),
        ];
        pairs.iter().copied().collect()
    };

    static ref KATAKANA_VOICED: HashMap<char, char> = {
        let pairs = [
            ('カ', 'ガ'), ('キ', 'ギ'), ('ク', 'グ'), ('ケ', 'ゲ'), ('コ', 'ゴ'),
            ('サ', 'ザ'), ('シ', 'ジ'), ('ス', 'ズ'), ('セ', 'ゼ'), ('ソ', 'ゾ'),
            ('タ', 'ダ'), ('チ', 'ヂ'), ('ツ', 'ヅ'), ('テ', 'デ'), ('ト', 'ド'),
            ('ハ', 'バ'), ('ヒ', 'ビ'), ('フ', 'ブ'), ('ヘ', 'ベ'), ('ホ', 'ボ'),
            ('ウ', 'ヴ'),
        ];
        pairs.iter().copied().collect()
    };

    /// Unihan variant forms that real Japanese text uses, folded onto the form
    /// the shipped dictionary carries. Direction rule: canonical = the side the
    /// recogniser's vocabulary can emit, variant = the side it cannot.
    ///
    /// Subset rule — measured, not guessed: a pair ships only if its variant
    /// side actually occurs in real text, and only when its canonical side
    /// occurs at least as often as its variant in the same corpus. Unihan's
    /// `kSemanticVariant` is loose — it also lists pairs whose "canonical" side
    /// is the rarer form — and a fold REPLACES the lookup key, so folding those
    /// would rewrite a query that used to resolve into one that does not. No
    /// canonical is itself a key, so the fold stays idempotent: one pass
    /// reaches the terminal form. All pairs are single-character, so unlike the
    /// Roman numerals they never change query length.
    static ref MEASURED_VARIANT_FOLD: HashMap<char, &'static str> = {        let pairs = [
            ('㕞', "刷"), ('㘅', "啣"), ('㝵', "碍"), ('䖟', "蝱"), ('䙝', "褻"),
            ('䬒', "颼"), ('䯻', "髻"), ('䰗', "鬮"), ('乾', "干"), ('亻', "人"),
            ('來', "来"), ('俠', "侠"), ('册', "冊"), ('冩', "写"), ('冫', "氷"),
            ('准', "準"), ('凉', "涼"), ('凴', "憑"), ('凾', "函"), ('刋', "刊"),
            ('剝', "剥"), ('劒', "劍"), ('勹', "包"), ('匳', "奩"), ('匵', "櫝"),
            ('卭', "卬"), ('厶', "某"), ('后', "後"), ('噐', "器"), ('噓', "嘘"),
            ('嚮', "向"), ('囑', "嘱"), ('囘', "回"), ('國', "国"), ('堭', "隍"),
            ('壽', "寿"), ('娬', "嫵"), ('學', "学"), ('寫', "写"), ('寶', "宝"),
            ('將', "将"), ('尸', "屍"), ('屆', "届"), ('屬', "属"), ('峽', "峡"),
            ('巤', "鬣"), ('帋', "紙"), ('帒', "袋"), ('并', "併"), ('彌', "弥"),
            ('悋', "吝"), ('慙', "慚"), ('懜', "懵"), ('戀', "恋"), ('挾', "挟"),
            ('捬', "撫"), ('摑', "掴"), ('无', "無"), ('晝', "昼"), ('會', "会"),
            ('朙', "明"), ('栖', "棲"), ('樓', "楼"), ('樷', "叢"), ('樸', "朴"),
            ('欝', "鬱"), ('氵', "水"), ('涶', "唾"), ('渊', "淵"), ('潛', "潜"),
            ('濵', "濱"), ('灑', "洒"), ('灣', "湾"), ('烟', "煙"), ('燈', "灯"),
            ('犭', "犬"), ('甎', "磚"), ('甤', "蕤"), ('畄', "留"), ('畆', "畝"),
            ('當', "当"), ('癢', "痒"), ('皃', "貌"), ('眎', "視"), ('瞹', "曖"),
            ('碯', "瑙"), ('礟', "礮"), ('祿', "禄"), ('禀', "稟"), ('禦', "御"),
            ('禪', "禅"), ('禮', "礼"), ('禱', "祷"), ('秇', "藝"), ('秌', "秋"),
            ('穪', "稱"), ('竆', "窮"), ('竒', "奇"), ('笋', "筍"), ('簞', "箪"),
            ('粮', "糧"), ('糓', "穀"), ('纎', "纖"), ('缻', "缶"), ('网', "網"),
            ('羮', "羹"), ('耻', "恥"), ('耼', "聃"), ('聲', "声"), ('脉', "脈"),
            ('膓', "腸"), ('舊', "旧"), ('艪', "櫓"), ('苢', "苡"), ('莖', "茎"),
            ('萬', "万"), ('著', "着"), ('葢', "蓋"), ('薑', "姜"), ('蘯', "蕩"),
            ('號', "号"), ('蚦', "蚺"), ('蜹', "蚋"), ('蟬', "蝉"), ('蟲', "虫"),
            ('蠶', "蚕"), ('襍', "雜"), ('覔', "覓"), ('觧', "解"), ('註', "注"),
            ('誐', "哦"), ('賍', "贓"), ('賷', "齎"), ('軆', "体"), ('輓', "挽"),
            ('辶', "辵"), ('迯', "逃"), ('迹', "跡"), ('遉', "偵"), ('遙', "遥"),
            ('釐', "厘"), ('鍫', "鍬"), ('鏁', "鎖"), ('閙', "鬧"), ('隂', "陰"),
            ('隖', "塢"), ('雙', "双"), ('頣', "頤"), ('颱', "台"), ('飃', "飄"),
            ('餘', "余"), ('駞', "駝"), ('髗', "顱"), ('髩', "鬢"), ('鬂', "鬢"),
            ('鮧', "鯷"), ('鵶', "鴉"), ('鶽', "隼"), ('鸎', "鶯"), ('麄', "粗"),
            ('麴', "麹"), ('麸', "麩"), ('點', "点"), ('齅', "嗅"), ('﨑', "崎"),
        ];
        pairs.iter().copied().collect()
    };

    /// The curated half of the fold table: characters the recogniser can emit
    /// that dictionaries do not use — Roman numerals (NFKC behaviour), the
    /// compatibility form `℃`, obsolete kana, and the Chinese-only forms it
    /// emits in place of the Japanese ones — composed with
    /// [`MEASURED_VARIANT_FOLD`], the Unihan-derived half.
    static ref LOOKUP_VARIANT_MAP: HashMap<char, &'static str> = {
        let curated = [
            // Roman numerals (NFKC behaviour).
            ('Ⅰ', "I"), ('Ⅱ', "II"), ('Ⅲ', "III"), ('Ⅳ', "IV"), ('Ⅴ', "V"), ('Ⅵ', "VI"),
            ('Ⅶ', "VII"), ('Ⅷ', "VIII"), ('Ⅸ', "IX"), ('Ⅹ', "X"), ('Ⅺ', "XI"), ('Ⅻ', "XII"),
            ('ⅰ', "i"), ('ⅱ', "ii"), ('ⅲ', "iii"), ('ⅳ', "iv"), ('ⅴ', "v"), ('ⅵ', "vi"),
            ('ⅶ', "vii"), ('ⅷ', "viii"), ('ⅸ', "ix"), ('ⅹ', "x"),
            // Compatibility form.
            ('℃', "°C"),
            // Obsolete kana the recogniser can emit.
            ('ゑ', "え"), ('ヰ', "イ"),
            // Chinese-only forms emitted in place of the Japanese one.
            ('况', "況"), ('查', "査"),
        ];
        let mut map: HashMap<char, &'static str> = curated.iter().copied().collect();
        map.extend(MEASURED_VARIANT_FOLD.iter().map(|(c, s)| (*c, *s)));
        map
    };
}

/// The Unihan-derived fold table as owned data, for hosts that need the table
/// itself (a binding cannot carry a `HashMap`, and the mobile drift guard
/// checks the table against the committed asset). All pairs are
/// single-character. Ordered for a stable boundary; the table is read-only.
pub fn measured_variant_fold() -> Vec<(char, &'static str)> {
    let mut out: Vec<(char, &'static str)> = MEASURED_VARIANT_FOLD
        .iter()
        .map(|(c, s)| (*c, *s))
        .collect();
    out.sort_unstable();
    out
}

/// Mobile `JapaneseUtil.verticalPunctuation` (#56, #63): PP-OCR emits ASCII
/// `?` where JP wants fullwidth `？`, and horizontal `…`/`‥` where vertical
/// text wants the vertical presentation forms `︙`/`︰`. Applied at emit time
/// for vertical lines only, before char boxes. Lookup-safe: `normalize` folds
/// them back, so dictionary search is unaffected.
pub fn vertical_punctuation(text: &str) -> String {
    text.chars().map(vertical_punctuation_char).collect()
}

/// Mobile `JapaneseUtil.verticalPunctuationChar`.
pub fn vertical_punctuation_char(c: char) -> char {
    match c {
        '?' => '？',
        '…' => '︙', // U+2026 → U+FE19 vertical ellipsis
        '‥' => '︰', // U+2035 → U+FE30 vertical two-dot leader
        _ => c,
    }
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
///
/// The overlay no longer sizes from a pitch (it uses the box height, like
/// mobile); the ppocr synth tests still pin the formula.
#[allow(dead_code)]
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

// ---------------------------------------------------------------------------
// Pre-reform kana normalisation (#75, #81)
// ---------------------------------------------------------------------------

/// The committed `variants/kana_variants.txt` pairs (variant -> modern), in
/// file order. Query-side only: callers search the raw form alongside this.
const KANA_VARIANT_PAIRS: &[(char, char)] = &[
    // wagyou ワ行
    ('ゐ', 'い'), ('ゑ', 'え'), ('ヰ', 'イ'), ('ヱ', 'エ'),
    // dakugyou だ行
    ('ぢ', 'じ'), ('づ', 'ず'), ('ヂ', 'ジ'), ('ヅ', 'ズ'),
    // sokuon 促音
    ('つ', 'っ'), ('ツ', 'ッ'),
    // yoon 拗音
    ('や', 'ゃ'), ('ゆ', 'ゅ'), ('よ', 'ょ'), ('ヤ', 'ャ'), ('ユ', 'ュ'), ('ヨ', 'ョ'),
];

/// A field that must be exactly one character, mirroring Kotlin
/// `singleOrNull()` — an empty or multi-character field is dropped, not
/// truncated. The unit is Kotlin's: UTF-16 code units, so a supplementary-plane
/// character counts as two and is rejected here too.
fn single_char_field(field: &str) -> Option<char> {
    if field.encode_utf16().count() != 1 {
        return None;
    }
    field.chars().next()
}

/// Parsed pre-reform kana orthography table (#75): the `variant<TAB>canonical`
/// pairs, in the file format of the shipped `variants/kana_variants.txt`.
///
/// The desktop uses [`KanaOrthographyTable::builtin`] (the committed
/// [`KANA_VARIANT_PAIRS`]); the binding host parses the same text at runtime so
/// the asset stays the source of truth. Parsing mirrors the Kotlin
/// `KanaOrthography.Table.parse`: lines are trimmed, blank and `#` lines
/// skipped, the line split on tabs, malformed lines (wrong field count, a side
/// that is not a single character) dropped, a pair whose sides are equal
/// skipped, and the first mapping for a variant wins.
#[derive(Debug, Clone)]
pub struct KanaOrthographyTable {
    canonical_by_variant: HashMap<char, char>,
}

impl KanaOrthographyTable {
    /// The committed table (the shipped `variants/kana_variants.txt` rows).
    pub fn builtin() -> Self {
        Self {
            canonical_by_variant: KANA_VARIANT_PAIRS.iter().copied().collect(),
        }
    }

    /// The identity table: [`canonical`](Self::canonical) returns its input.
    pub fn empty() -> Self {
        Self {
            canonical_by_variant: HashMap::new(),
        }
    }

    /// Parse the committed text format (see the type docs).
    pub fn parse(text: &str) -> Self {
        let mut canonical_by_variant = HashMap::new();
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 2 {
                continue;
            }
            let (Some(variant), Some(canonical)) =
                (single_char_field(parts[0]), single_char_field(parts[1]))
            else {
                continue;
            };
            if variant == canonical {
                continue;
            }
            canonical_by_variant.entry(variant).or_insert(canonical);
        }
        Self {
            canonical_by_variant,
        }
    }

    /// Number of distinct variants in this table.
    pub fn entry_count(&self) -> usize {
        self.canonical_by_variant.len()
    }

    /// The modern form of `variant`, or `variant` itself when the table has no
    /// entry for it.
    pub fn canonical(&self, variant: char) -> char {
        self.canonical_by_variant
            .get(&variant)
            .copied()
            .unwrap_or(variant)
    }

    /// #75: rewrite a lookup query from pre-reform orthography (旧仮名遣い) onto
    /// the modern form the dictionary keys on. Identity when nothing is a
    /// variant. Mirrors `KanaOrthography.modernise`; the context rules are
    /// [`orthography_applies`].
    pub fn modernise(&self, query: &str) -> String {
        if query.is_empty() {
            return query.to_string();
        }
        let chars: Vec<char> = query.chars().collect();
        let mut out = String::with_capacity(query.len());
        for i in 0..chars.len() {
            let c = chars[i];
            let mapped = self.canonical(c);
            if mapped == c {
                out.push(c);
                continue;
            }
            let prev = if i > 0 { Some(chars[i - 1]) } else { None };
            let next = chars.get(i + 1).copied();
            out.push(if orthography_applies(c, prev, next) {
                mapped
            } else {
                c
            });
        }
        out
    }
}

/// Whether a table pair applies in context (see `KanaOrthography.applies`).
fn orthography_applies(variant: char, prev: Option<char>, next: Option<char>) -> bool {
    const WAGYOU: &str = "ゐゑヰヱ";
    const DAKUGYOU: &str = "ぢづヂヅ";
    const SOKUON: &str = "つツ";
    const YOON: &str = "やゆよヤユヨ";
    const DAKUTEN_PREV: &str = "つちツチ";
    const SOKUON_TRIGGERS: &str =
        "たちつてとさしすせそぱぴぷぺぽタチツテトサシスセソパピプペポ";
    const YOON_BASE: &str = "きしちにひみりぎじびぴキシチニヒミリギジビピ";

    if WAGYOU.contains(variant) {
        true
    } else if DAKUGYOU.contains(variant) {
        prev.map_or(true, |p| !DAKUTEN_PREV.contains(p))
    } else if SOKUON.contains(variant) {
        next.map_or(false, |n| SOKUON_TRIGGERS.contains(n))
    } else if YOON.contains(variant) {
        prev.map_or(false, |p| YOON_BASE.contains(p))
    } else {
        false
    }
}

/// #75: rewrite a lookup query from pre-reform orthography (旧仮名遣い) onto
/// the modern form the dictionary keys on. Identity when nothing is a variant.
/// Mirrors `KanaOrthography.modernise`.
///
/// Delegates to the builtin table, built once; the binding host uses
/// [`KanaOrthographyTable`] directly.
pub fn kana_orthography_modernise(query: &str) -> String {
    builtin_orthography_table().modernise(query)
}

/// The builtin orthography table, built once for the free-function callers.
fn builtin_orthography_table() -> &'static KanaOrthographyTable {
    static BUILTIN: OnceLock<KanaOrthographyTable> = OnceLock::new();
    BUILTIN.get_or_init(KanaOrthographyTable::builtin)
}

/// The committed `variants/kana_sound_changes.txt` rows.
const KANA_HA: &[(char, char)] = &[('は', 'わ'), ('ひ', 'い'), ('ふ', 'う'), ('へ', 'え')];
const KANA_AU: &[(char, char)] = &[
    ('か', 'こ'), ('が', 'ご'), ('さ', 'そ'), ('ざ', 'ぞ'), ('た', 'と'), ('だ', 'ど'),
    ('な', 'の'), ('は', 'ほ'), ('ば', 'ぼ'), ('ぱ', 'ぽ'), ('ゃ', 'ょ'), ('や', 'よ'),
    ('ら', 'ろ'), ('わ', 'お'),
];
const KANA_EU: &[(char, &str)] = &[
    ('え', "よ"), ('け', "きょ"), ('げ', "ぎょ"), ('せ', "しょ"), ('ぜ', "じょ"),
    ('て', "ちょ"), ('で', "じょ"), ('ね', "にょ"), ('へ', "ひょ"), ('べ', "びょ"),
    ('ぺ', "ぴょ"), ('め', "みょ"), ('れ', "りょ"),
];

/// Parsed historical kana sound-change table (#81): rows `ha` (ハ行転呼),
/// `au` (アウ->オウ) and `eu` (エウ->ヨウ), in the file format of the shipped
/// `variants/kana_sound_changes.txt`.
///
/// The desktop uses [`KanaSoundTable::builtin`] (the committed [`KANA_HA`] /
/// [`KANA_AU`] / [`KANA_EU`] rows); the binding host parses the same text at
/// runtime so the asset stays the source of truth. Parsing mirrors the Kotlin
/// `KanaSoundChanges.Table.parse`: lines are trimmed, blank and `#` lines
/// skipped, the line split on tabs, malformed lines (wrong field count, a
/// variant that is not a single character, an empty or unchanged modern form, a
/// modern form of the wrong length for its row) dropped, unknown rows ignored,
/// and the first mapping for a variant wins.
#[derive(Debug, Clone)]
pub struct KanaSoundTable {
    ha: HashMap<char, char>,
    au: HashMap<char, char>,
    eu: HashMap<char, String>,
}

impl KanaSoundTable {
    /// The committed table (the shipped `variants/kana_sound_changes.txt`
    /// rows).
    pub fn builtin() -> Self {
        Self {
            ha: KANA_HA.iter().copied().collect(),
            au: KANA_AU.iter().copied().collect(),
            eu: KANA_EU
                .iter()
                .map(|(variant, modern)| (*variant, (*modern).to_string()))
                .collect(),
        }
    }

    /// The identity table: every lookup misses and
    /// [`modernise`](Self::modernise) returns its input.
    pub fn empty() -> Self {
        Self {
            ha: HashMap::new(),
            au: HashMap::new(),
            eu: HashMap::new(),
        }
    }

    /// Parse the committed text format (see the type docs).
    pub fn parse(text: &str) -> Self {
        let mut ha = HashMap::new();
        let mut au = HashMap::new();
        let mut eu = HashMap::new();
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 3 {
                continue;
            }
            let Some(variant) = single_char_field(parts[0]) else {
                continue;
            };
            let modern = parts[1];
            if modern.is_empty() || modern == variant.to_string() {
                continue;
            }
            match parts[2] {
                "ha" => {
                    if let Some(c) = single_char_field(modern) {
                        ha.entry(variant).or_insert(c);
                    }
                }
                "au" => {
                    if let Some(c) = single_char_field(modern) {
                        au.entry(variant).or_insert(c);
                    }
                }
                "eu" => {
                    if (1..=2).contains(&modern.encode_utf16().count()) {
                        eu.entry(variant).or_insert_with(|| modern.to_string());
                    }
                }
                _ => {}
            }
        }
        Self { ha, au, eu }
    }

    /// Number of distinct pairs in this table.
    pub fn entry_count(&self) -> usize {
        self.ha.len() + self.au.len() + self.eu.len()
    }

    /// The ハ行転呼 modern form of `variant`, when the table carries the pair.
    pub fn ha(&self, variant: char) -> Option<char> {
        self.ha.get(&variant).copied()
    }

    /// The アウ->オウ modern form of `variant`, when the table carries the pair.
    pub fn au(&self, variant: char) -> Option<char> {
        self.au.get(&variant).copied()
    }

    /// The エウ->ヨウ modern form of `variant` (1–2 characters), when the table
    /// carries the pair.
    pub fn eu(&self, variant: char) -> Option<String> {
        self.eu.get(&variant).cloned()
    }

    /// #81: the two passes over this table, in the 告示's order — ハ行転呼,
    /// then the vowel changes before `う`. Identity when nothing is a rule's
    /// input. Mirrors `KanaSoundChanges.modernise`.
    pub fn modernise(&self, query: &str) -> String {
        if query.is_empty() {
            return query.to_string();
        }
        self.vowel_changes(&self.ha_gyouten(query))
    }

    /// Pass 1: 語中・語尾のハ行 -> ワ行, under each character's ending condition.
    fn ha_gyouten(&self, query: &str) -> String {
        let chars: Vec<char> = query.chars().collect();
        let mut out = String::with_capacity(query.len());
        for i in 0..chars.len() {
            let c = chars[i];
            let Some(mapped) = self.ha(c) else {
                out.push(c);
                continue;
            };
            let prev = if i > 0 { Some(chars[i - 1]) } else { None };
            let next = chars.get(i + 1).copied();
            out.push(if ha_gyouten_applies(c, prev, next) {
                mapped
            } else {
                c
            });
        }
        out
    }

    /// Pass 2: 母音の変化 before `う` — `au` replaces one character, `eu`
    /// replaces the エ段 character with its イ段 counterpart + small ょ and
    /// keeps the `う` (`け` + `う` -> `きょ` + `う`).
    fn vowel_changes(&self, query: &str) -> String {
        let chars: Vec<char> = query.chars().collect();
        let mut out = String::with_capacity(query.len());
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            let next = chars.get(i + 1).copied();
            if next == Some('う') {
                if let Some(mapped) = self.au(c) {
                    out.push(mapped);
                    out.push('う');
                    i += 2;
                    continue;
                }
                if let Some(mapped) = self.eu(c) {
                    out.push_str(&mapped);
                    out.push('う');
                    i += 2;
                    continue;
                }
            }
            out.push(c);
            i += 1;
        }
        out
    }
}

fn is_kana_char(c: char) -> bool {
    ('\u{3041}'..='\u{3096}').contains(&c) || ('\u{30A1}'..='\u{30F6}').contains(&c)
}

/// Whether a ハ行 character folds at this grammatical ending.
fn ha_gyouten_applies(variant: char, prev: Option<char>, next: Option<char>) -> bool {
    const PARTICLES: &str = "はがのにをともやかぞなよねだでどばへこそしかまでより";
    const HI_SUFFIX: &str = "てつな";
    const HE_SUFFIX: &str = "したてばどきけるれりま";
    const HA_SUFFIX: &str = "ずぬむば";
    match variant {
        // 終止形・連体形: word-final or before a particle. Not after ウ.
        'ふ' => {
            prev.is_some()
                && prev != Some('う')
                && next.map_or(true, |n| !is_kana_char(n) || PARTICLES.contains(n))
        }
        // 連用形.
        'ひ' => next.map_or(false, |n| HI_SUFFIX.contains(n)),
        // 仮定形・已然形 and the 下二段 連用形.
        'へ' => next.map_or(false, |n| HE_SUFFIX.contains(n)),
        // 未然形 + ず/ぬ/む/ば.
        'は' => prev.is_some() && next.map_or(false, |n| HA_SUFFIX.contains(n)),
        _ => false,
    }
}

/// #81: the historical sound changes JMdict's entry-local variants cannot
/// reach (やう→よう, けふ→きょう, 思ふ→思う). Compose after
/// [kana_orthography_modernise]. Mirrors `KanaSoundChanges.modernise`.
///
/// Delegates to the builtin table, built once; the binding host uses
/// [`KanaSoundTable`] directly.
pub fn kana_sound_changes_modernise(query: &str) -> String {
    builtin_sound_table().modernise(query)
}

/// The builtin sound table, built once for the free-function callers.
fn builtin_sound_table() -> &'static KanaSoundTable {
    static BUILTIN: OnceLock<KanaSoundTable> = OnceLock::new();
    BUILTIN.get_or_init(KanaSoundTable::builtin)
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

    /// Mirrors `KanaOrthographyTest`: the committed 16-pair table, pinned.
    #[test]
    fn kana_orthography_normalises_pre_reform_forms() {
        assert_eq!(kana_orthography_modernise("いつしよ"), "いっしょ");
        assert_eq!(kana_orthography_modernise("しゆつぱつ"), "しゅっぱつ");
        assert_eq!(kana_orthography_modernise("ちよつと"), "ちょっと");
        // きやう only reaches the 拗音 half of the change: やう->よう is #81.
        assert_eq!(kana_orthography_modernise("きやう"), "きゃう");
        assert_eq!(kana_orthography_modernise("ゐる"), "いる");
        assert_eq!(kana_orthography_modernise("植ゑた"), "植えた");
        assert_eq!(kana_orthography_modernise("あづま"), "あずま");
        assert_eq!(kana_orthography_modernise("おのづから"), "おのずから");
        assert_eq!(kana_orthography_modernise("はなぢ"), "はなじ");
        assert_eq!(kana_orthography_modernise("ウヰスキー"), "ウイスキー");
        assert_eq!(kana_orthography_modernise("ヱビス"), "エビス");
        // 連濁 is the modern form itself; を and the 小書き row are excluded.
        assert_eq!(kana_orthography_modernise("つづく"), "つづく");
        assert_eq!(kana_orthography_modernise("ちぢむ"), "ちぢむ");
        assert_eq!(kana_orthography_modernise("ツヅク"), "ツヅク");
        assert_eq!(kana_orthography_modernise("本を読む"), "本を読む");
        assert_eq!(kana_orthography_modernise("ヲタク"), "ヲタク");
        assert_eq!(kana_orthography_modernise("ああ"), "ああ");
        assert_eq!(kana_orthography_modernise("そうそう"), "そうそう");
        // 促音 fires only before た行/さ行/ぱ行.
        assert_eq!(kana_orthography_modernise("つかう"), "つかう");
        assert_eq!(kana_orthography_modernise("つくえ"), "つくえ");
        assert_eq!(kana_orthography_modernise("あつた"), "あった");
        assert_eq!(kana_orthography_modernise("いつて"), "いって");
        assert_eq!(kana_orthography_modernise("いつぱい"), "いっぱい");
        for modern in ["", "あ", "ABC", "日本語", "コーヒー", "12月", "、。"] {
            assert_eq!(kana_orthography_modernise(modern), modern);
        }
    }

    /// Mirrors `KanaSoundChangesTest`: the shipped composition
    /// (orthography fold, then sound changes), with every headline form pinned.
    #[test]
    fn kana_sound_changes_normalise_legacy_grammar() {
        fn normalise(q: &str) -> String {
            kana_sound_changes_modernise(&kana_orthography_modernise(q))
        }
        assert_eq!(normalise("やう"), "よう");
        assert_eq!(normalise("きやう"), "きょう");
        assert_eq!(normalise("けふ"), "きょう");
        assert_eq!(normalise("てふ"), "ちょう");
        assert_eq!(normalise("しやう"), "しょう");
        assert_eq!(normalise("でせう"), "でしょう");
        assert_eq!(normalise("だらう"), "だろう");
        assert_eq!(normalise("ありがたう"), "ありがとう");
        assert_eq!(normalise("たふとし"), "とうとし");
        assert_eq!(normalise("といふのは"), "というのは");
        // ハ行転呼 only at its grammatical ending.
        assert_eq!(normalise("思ふ"), "思う");
        assert_eq!(normalise("思ふが"), "思うが");
        assert_eq!(normalise("思ふ人"), "思う人");
        assert_eq!(normalise("思ひつ"), "思いつ");
        assert_eq!(normalise("戦ひながら"), "戦いながら");
        assert_eq!(normalise("数へて"), "数えて");
        assert_eq!(normalise("言へば"), "言えば");
        assert_eq!(normalise("添へた"), "添えた");
        assert_eq!(normalise("言はず"), "言わず");
        assert_eq!(normalise("思はぬ"), "思わぬ");
        assert_eq!(normalise("思はば"), "思わば");
        // ...and nowhere else.
        assert_eq!(normalise("ふね"), "ふね");
        assert_eq!(normalise("吹く"), "吹く");
        assert_eq!(normalise("ふうふ"), "ふうふ");
        assert_eq!(normalise("ひる"), "ひる");
        assert_eq!(normalise("へや"), "へや");
        assert_eq!(normalise("本は"), "本は");
        assert_eq!(normalise("はな"), "はな");
        // Withheld アウ rows fold to themselves.
        assert_eq!(normalise("あう"), "あう");
        assert_eq!(normalise("まう"), "まう");
        assert_eq!(normalise("おほ"), "おほ");
        assert_eq!(normalise("本を読む"), "本を読む");
        // Modern text is the identity.
        for modern in [
            "", "あ", "ABC", "日本語", "つくえ", "にっぽん", "がっこう", "コーヒー",
            "12月", "、。", "つづく", "ちぢむ", "買う", "会う", "クラウン",
        ] {
            assert_eq!(normalise(modern), modern);
        }
    }

    // ── Kana tables (the parsed forms the binding host installs) ────────────

    /// Mirrors mobile
    /// `KanaOrthographyTest.parse_skips_malformed_lines_and_keeps_the_first_canonical`:
    /// comments/blanks, wrong field counts, non-single-character sides and
    /// equal pairs are dropped, and the first mapping for a variant wins.
    #[test]
    fn kana_orthography_parse_skips_malformed_lines_and_keeps_the_first_canonical() {
        let table = KanaOrthographyTable::parse(
            "# a comment\n\
             ゐ\tい\n\
             ゐ\tゑ\n\
             ゐい\n\
             い\tい\n\
             ゐ\tい\textra\n\
             𝄞\tう\n",
        );
        assert_eq!(1, table.entry_count());
        assert_eq!('い', table.canonical('ゐ'));
    }

    /// Mirrors mobile `KanaOrthographyTest.empty_table_is_the_identity`.
    #[test]
    fn kana_orthography_empty_table_is_the_identity() {
        let table = KanaOrthographyTable::empty();
        assert_eq!(0, table.entry_count());
        assert_eq!('ゐ', table.canonical('ゐ'));
        assert_eq!("ゐる", table.modernise("ゐる"));
    }

    /// The builtin table is the committed [`KANA_VARIANT_PAIRS`], and a table
    /// parsed from their text format reproduces it row for row. A custom table
    /// drives the fold: only pairs the table carries can fold.
    #[test]
    fn kana_orthography_builtin_and_parsed_tables_agree() {
        let builtin = KanaOrthographyTable::builtin();
        assert_eq!(16, builtin.entry_count());
        let text: String = KANA_VARIANT_PAIRS
            .iter()
            .map(|(variant, canonical)| format!("{variant}\t{canonical}\n"))
            .collect();
        let parsed = KanaOrthographyTable::parse(&text);
        assert_eq!(builtin.entry_count(), parsed.entry_count());
        for (variant, canonical) in KANA_VARIANT_PAIRS {
            assert_eq!(*canonical, builtin.canonical(*variant));
            assert_eq!(*canonical, parsed.canonical(*variant));
        }
        let custom = KanaOrthographyTable::parse("ゐ\tゑ\n");
        assert_eq!(1, custom.entry_count());
        assert_eq!("ゑる", custom.modernise("ゐる"));
    }

    /// Mirrors mobile
    /// `KanaSoundChangesTest.parse_skips_malformed_lines_and_keeps_the_first_mapping`:
    /// comments/blanks, wrong field counts, non-single-character variants,
    /// equal pairs, wrong-length modern forms and unknown rows are dropped, and
    /// the first mapping for a variant wins.
    #[test]
    fn kana_sound_parse_skips_malformed_lines_and_keeps_the_first_mapping() {
        let table = KanaSoundTable::parse(
            "# a comment\n\
             ふ\tう\tha\n\
             ふ\tひ\tha\n\
             ふう\n\
             う\tう\tha\n\
             ふ\tう\tha\textra\n\
             ふう\tう\tha\n\
             け\tきょ\teu\n\
             け\tきょう\teu\n\
             け\tきょ\tunknown-row\n",
        );
        assert_eq!(2, table.entry_count());
        assert_eq!(Some('う'), table.ha('ふ'));
        assert_eq!(Some("きょ".to_string()), table.eu('け'));
    }

    /// Mirrors mobile `KanaSoundChangesTest.empty_table_is_the_identity`.
    #[test]
    fn kana_sound_empty_table_is_the_identity() {
        let table = KanaSoundTable::empty();
        assert_eq!(0, table.entry_count());
        assert_eq!("けふ", table.modernise("けふ"));
    }

    /// The builtin table is the committed rows, and a table parsed from their
    /// text format reproduces them. The withheld rows stay absent, and a custom
    /// table drives both passes.
    #[test]
    fn kana_sound_builtin_and_parsed_tables_agree() {
        let builtin = KanaSoundTable::builtin();
        assert_eq!(31, builtin.entry_count());
        assert_eq!(None, builtin.ha('ほ'));
        assert_eq!(None, builtin.au('あ'));
        assert_eq!(None, builtin.au('ま'));
        let mut text = String::new();
        for (variant, modern) in KANA_HA {
            text.push_str(&format!("{variant}\t{modern}\tha\n"));
        }
        for (variant, modern) in KANA_AU {
            text.push_str(&format!("{variant}\t{modern}\tau\n"));
        }
        for (variant, modern) in KANA_EU {
            text.push_str(&format!("{variant}\t{modern}\teu\n"));
        }
        let parsed = KanaSoundTable::parse(&text);
        assert_eq!(builtin.entry_count(), parsed.entry_count());
        for (variant, modern) in KANA_HA {
            assert_eq!(Some(*modern), parsed.ha(*variant));
        }
        for (variant, modern) in KANA_AU {
            assert_eq!(Some(*modern), parsed.au(*variant));
        }
        for (variant, modern) in KANA_EU {
            assert_eq!(Some((*modern).to_string()), parsed.eu(*variant));
        }
        let custom = KanaSoundTable::parse("い\tよ\teu\n");
        assert_eq!("よう", custom.modernise("いう"));
        assert_eq!("いう", builtin.modernise("いう"));
    }

    // ── Lookup-variant fold (`JapaneseUtilVariantFoldTest`) ─────────────────

    /// Mirrors mobile `expands_hiragana_iteration_mark`.
    #[test]
    fn fold_expands_hiragana_iteration_mark() {
        assert_eq!(fold_lookup_variants("こゝろ"), "こころ");
        assert_eq!(fold_lookup_variants("こゝ"), "ここ");
        assert_eq!(fold_lookup_variants("あゝ"), "ああ");
    }

    /// Mirrors mobile `expands_voiced_iteration_mark`.
    #[test]
    fn fold_expands_voiced_iteration_mark() {
        assert_eq!(fold_lookup_variants("たゞ"), "ただ");
        assert_eq!(fold_lookup_variants("かゞ"), "かが");
        assert_eq!(fold_lookup_variants("はゞ"), "はば");
        // no voiced form (or already voiced): the mark is still a plain repeat
        assert_eq!(fold_lookup_variants("まゞ"), "まま");
        assert_eq!(fold_lookup_variants("がゞ"), "がが");
        assert_eq!(fold_lookup_variants("ナヾ"), "ナナ");
    }

    /// Mirrors mobile `expands_katakana_iteration_marks`.
    #[test]
    fn fold_expands_katakana_iteration_marks() {
        assert_eq!(fold_lookup_variants("カヽ"), "カカ");
        assert_eq!(fold_lookup_variants("カヾ"), "カガ");
        assert_eq!(fold_lookup_variants("ハヾ"), "ハバ");
    }

    /// Mirrors mobile `repeats_run_of_marks`.
    #[test]
    fn fold_repeats_run_of_marks() {
        assert_eq!(fold_lookup_variants("こゝゝ"), "こここ");
    }

    /// Mirrors mobile `leaves_mark_that_cannot_be_repeated`.
    #[test]
    fn fold_leaves_mark_that_cannot_be_repeated() {
        // line-initial, after a kanji, after punctuation
        assert_eq!(fold_lookup_variants("ゝあ"), "ゝあ");
        assert_eq!(fold_lookup_variants("日ゝ"), "日ゝ");
        assert_eq!(fold_lookup_variants("、ゝ"), "、ゝ");
        assert_eq!(fold_lookup_variants("。ヾ"), "。ヾ");
        // iteration marks do not cross scripts
        assert_eq!(fold_lookup_variants("カゝ"), "カゝ");
        assert_eq!(fold_lookup_variants("あヽ"), "あヽ");
    }

    /// Mirrors mobile `folds_obsolete_kana_the_head_can_emit`.
    #[test]
    fn fold_folds_obsolete_kana() {
        assert_eq!(fold_lookup_variants("こゑ"), "こえ");
        assert_eq!(fold_lookup_variants("ヰロ"), "イロ");
    }

    /// Mirrors mobile `folds_roman_numerals`.
    #[test]
    fn fold_folds_roman_numerals() {
        assert_eq!(fold_lookup_variants("Ⅶ"), "VII");
        assert_eq!(fold_lookup_variants("Ⅷ"), "VIII");
        assert_eq!(fold_lookup_variants("第Ⅻ章"), "第XII章");
        assert_eq!(fold_lookup_variants("ⅸ"), "ix");
    }

    /// Mirrors mobile `folds_compatibility_and_chinese_only_forms`.
    #[test]
    fn fold_folds_compatibility_and_chinese_only_forms() {
        assert_eq!(fold_lookup_variants("20℃"), "20°C");
        assert_eq!(fold_lookup_variants("状况"), "状況");
        // 查 folds (emittable) while 调 does not (pruned) — same word, and only
        // the emittable half of the pair is worth an entry
        assert_eq!(fold_lookup_variants("调查④"), "调査④");
    }

    /// Mirrors mobile `does_not_fold_characters_we_pruned_ourselves`.
    #[test]
    fn fold_leaves_pruned_characters_alone() {
        // 调 has no Unihan Japanese reading, so the head cannot emit it, and an
        // entry would be dead code. Emittability is checked against the pruned
        // head's remap, never the unpruned vocabulary, which still lists the
        // pruned classes.
        assert_eq!(fold_lookup_variants("调"), "调");
    }

    /// Mirrors mobile `leaves_kanji_iteration_mark_alone`.
    #[test]
    fn fold_leaves_kanji_iteration_mark_alone() {
        // dictionary headwords contain 々 (日々), so expanding would lose matches
        assert_eq!(fold_lookup_variants("日々"), "日々");
        assert_eq!(normalize("日々"), "日々");
    }

    /// Mirrors mobile `folds_measured_unihan_variants`.
    #[test]
    fn fold_folds_measured_unihan_variants() {
        // the pair the direction rule was validated on, and the corpus's most
        // frequent variant forms
        assert_eq!(fold_lookup_variants("囘"), "回");
        assert_eq!(fold_lookup_variants("欝"), "鬱");
        // frequency guard: 壜 is the *commoner* side, so folding it would
        // rewrite a resolvable query into a dead one
        assert_eq!(fold_lookup_variants("壜"), "壜");
        assert_eq!(fold_lookup_variants("劒"), "劍");
        assert_eq!(fold_lookup_variants("慙"), "慚");
        // and inside a word, which is how the fold is actually reached
        assert_eq!(fold_lookup_variants("囘想"), "回想");
        assert_eq!(fold_lookup_variants("欝々"), "鬱々");
        assert_eq!(fold_lookup_variants("慙愧"), "慚愧");
        assert_eq!(fold_lookup_variants("迯げる"), "逃げる");
        assert_eq!(fold_lookup_variants("噐械"), "器械");
        assert_eq!(fold_lookup_variants("迯"), "逃");
    }

    /// Mirrors mobile `unihan_fold_picks_the_corpus_dominant_canonical`.
    #[test]
    fn unihan_fold_picks_the_corpus_dominant_canonical() {
        // several variants have multiple canonical candidates in Unihan; the
        // fold takes the form that dominates the corpus, not an arbitrary first
        // (葢 -> 蓋; 悋 -> 吝; 冫 -> 氷)
        assert_eq!(fold_lookup_variants("葢"), "蓋");
        assert_eq!(fold_lookup_variants("悋"), "吝");
        assert_eq!(fold_lookup_variants("冫"), "氷");
        assert_eq!(fold_lookup_variants("秇"), "藝");
        assert_eq!(fold_lookup_variants("﨑"), "崎");
    }

    /// Mirrors mobile `unihan_fold_does_not_run_backwards`.
    #[test]
    fn unihan_fold_does_not_run_backwards() {
        // the canonical side is what the dictionary already keys on: folding it
        // would move the query to a form the recogniser cannot emit
        assert_eq!(fold_lookup_variants("回"), "回");
        assert_eq!(fold_lookup_variants("鬱"), "鬱");
        assert_eq!(fold_lookup_variants("蓋"), "蓋");
        assert_eq!(normalize("回想"), "回想");
    }

    /// Mirrors mobile `unihan_fold_is_idempotent_and_leaves_unknown_characters_alone`.
    #[test]
    fn unihan_fold_is_idempotent_and_leaves_unknown_characters_alone() {
        let plain = "日本語のテキストです。";
        assert_eq!(fold_lookup_variants(plain), plain);
        for s in ["囘想", "欝々", "迯げる", "噐械", "壜", "﨑", "囘囘回"] {
            let once = normalize(s);
            assert_eq!(normalize(&once), once, "not idempotent for {s:?}");
        }
    }

    /// Mirrors mobile `measured_variant_fold_matches_the_committed_asset`.
    ///
    /// Drift guard: every pair folded here must exist in
    /// `assets/variants/kanji_variants.txt` in the same direction (and no
    /// canonical may itself be a key, or the fold would not be idempotent).
    /// Multi-candidate variants are checked against the asset's *whole*
    /// candidate set, because the fold's choice among them is a corpus
    /// measurement the asset does not carry.
    #[test]
    fn measured_variant_fold_matches_the_committed_asset() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../assets/variants/kanji_variants.txt"
        );
        let text = std::fs::read_to_string(path).expect("kanji_variants.txt reads");
        let mut asset: HashMap<char, std::collections::HashSet<char>> = HashMap::new();
        for line in text.lines() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 2
                || parts[0].chars().count() != 1
                || parts[1].chars().count() != 1
            {
                continue;
            }
            asset
                .entry(parts[0].chars().next().unwrap())
                .or_default()
                .insert(parts[1].chars().next().unwrap());
        }
        assert_eq!(165, MEASURED_VARIANT_FOLD.len());
        assert_eq!(600, asset.len());
        let fold_keys: std::collections::HashSet<char> =
            MEASURED_VARIANT_FOLD.keys().copied().collect();
        for (variant, canonical) in MEASURED_VARIANT_FOLD.iter() {
            let target = canonical
                .chars()
                .next()
                .expect("fold targets are single characters");
            assert_eq!(
                canonical.chars().count(),
                1,
                "'{variant}' folds to a multi-character string"
            );
            let candidates = asset
                .get(variant)
                .unwrap_or_else(|| panic!("'{variant}' is not in kanji_variants.txt"));
            assert!(
                candidates.contains(&target),
                "asset has '{variant}' -> {candidates:?}, not '{target}'"
            );
            // Idempotence is a property of the FOLD, not of the asset: what
            // must not happen is a canonical that is itself a key *here*, which
            // would leave the query one step short of the form dictionaries
            // index.
            assert!(
                !fold_keys.contains(&target),
                "canonical '{target}' is itself a fold key — fold would chain"
            );
        }
    }

    /// Mirrors mobile `folds_the_jmdict_half_and_leaves_modern_forms_alone`.
    #[test]
    fn fold_folds_the_jmdict_half_and_leaves_modern_forms_alone() {
        // Old orthography the head can emit — both forms are in the vocabulary,
        // which is exactly why the original direction rule dropped these pairs.
        // The queried old form resolves to the headword a dictionary indexes.
        assert_eq!(fold_lookup_variants("摑"), "掴");
        assert_eq!(fold_lookup_variants("國"), "国");
        assert_eq!(fold_lookup_variants("會"), "会");
        assert_eq!(fold_lookup_variants("燈"), "灯");
        // The modern side is a key for nothing, so it must resolve to itself:
        // folding it would rewrite the form the dictionaries actually index.
        assert_eq!(fold_lookup_variants("掴"), "掴");
        assert_eq!(fold_lookup_variants("国"), "国");
        // A chain (冩 -> 寫 -> 写) reaches the terminal form in one pass.
        assert_eq!(fold_lookup_variants("冩"), "写");
        // 坂 is NOT folded to 阪: the corpus prefers 坂 (大阪), so that pair
        // failed the direction guard and stays out of the fold.
        assert_eq!(fold_lookup_variants("坂"), "坂");
    }

    /// Mirrors mobile `normalize_applies_the_fold`.
    #[test]
    fn normalize_applies_the_fold() {
        assert_eq!(normalize("こゝろ"), "こころ");
        assert_eq!(normalize("たゞ"), "ただ");
        assert_eq!(normalize("Ⅶ"), "VII");
        assert_eq!(normalize("状况"), "状況");
    }

    /// Mirrors mobile `halfwidth_kana_is_widened_before_folding`.
    #[test]
    fn normalize_widens_halfwidth_kana_before_folding() {
        // order matters: convert_width runs first, so the mark sees fullwidth カ
        assert_eq!(normalize("ｶヽ"), "カカ");
    }

    /// Mirrors mobile `fold_is_idempotent_and_noop_on_plain_text`.
    #[test]
    fn fold_is_idempotent_and_noop_on_plain_text() {
        let plain = "日本語のテキストです。";
        assert_eq!(fold_lookup_variants(plain), plain);
        assert_eq!(normalize(plain), plain);
        for s in ["こゝろ", "たゞ", "Ⅶ", "状况", "カヾ"] {
            assert_eq!(normalize(&normalize(s)), normalize(s));
        }
    }

    /// Mirrors mobile `existing_normalize_behaviour_is_unchanged`.
    #[test]
    fn normalize_existing_behaviour_is_unchanged() {
        // fullwidth ASCII folds to ASCII, ideographic space to space, vertical
        // presentation forms back to their horizontal forms
        assert_eq!(normalize("？"), "?");
        assert_eq!(normalize("Ａ１"), "A1");
        assert_eq!(normalize("あ︙"), "あ…");
        assert_eq!(normalize("あ　い"), "あ い");
    }

    /// The composed stage order: width conversion → lookup-variant fold →
    /// combining-character normalization.
    #[test]
    fn normalize_stages_run_in_pipeline_order() {
        // Width conversion first: the mark sees the widened カ, not ｶ.
        assert_eq!(normalize("ｶヽ"), "カカ");
        assert_eq!(normalize("ｶﾞヾ"), "ガガ");
        // Fold before combining normalization: the mark sees the combining
        // dakuten, so it is left alone rather than voicing the repeat. Were
        // combining normalized first, this would read がが.
        assert_eq!(normalize("か\u{3099}ゞ"), "がゞ");
        // Combining normalization runs last: the decomposed pair composes.
        assert_eq!(normalize("は\u{309A}"), "ぱ");
    }

    /// Mobile parity (`FuriganaAlignerTest.katakanaReading_normalized`): the
    /// katakana block folds exactly 0x60 down onto hiragana. The earlier port
    /// narrowed the code point through `u8`, which turned every conversion
    /// into garbage (ア → 'B') and silently broke the lookup's katakana
    /// variants.
    #[test]
    fn katakana_to_hiragana_folds_the_block() {
        assert_eq!(katakana_to_hiragana("タベル"), "たべる");
        assert_eq!(katakana_to_hiragana("カタカナ"), "かたかな");
        assert_eq!(katakana_to_hiragana("ヴァイオリン"), "ゔぁいおりん");
        // ー takes the preceding kana's vowel; a leading ー is left alone.
        assert_eq!(katakana_to_hiragana("コーヒー"), "こうひい");
        assert_eq!(katakana_to_hiragana("ー"), "ー");
        // Mixed scripts: only the katakana block folds.
        assert_eq!(katakana_to_hiragana("食べル"), "食べる");
        assert_eq!(katakana_to_hiragana("ABC"), "ABC");
    }

    /// The binding accessor exposes the exact table the fold reads, in a stable
    /// order and with single-character targets — the mobile drift guard checks
    /// this data against the committed asset instead of carrying its own copy.
    #[test]
    fn measured_variant_fold_accessor_matches_the_table() {
        let pairs = measured_variant_fold();
        assert_eq!(pairs.len(), 165);
        assert_eq!(pairs.len(), MEASURED_VARIANT_FOLD.len());
        for &(variant, canonical) in &pairs {
            assert_eq!(MEASURED_VARIANT_FOLD.get(&variant), Some(&canonical));
            assert_eq!(canonical.chars().count(), 1, "{variant} -> {canonical}");
        }
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        assert_eq!(pairs, sorted, "stable order for the boundary");
    }
}
