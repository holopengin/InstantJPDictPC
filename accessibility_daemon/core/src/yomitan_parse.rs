//! Pure parsing of Yomitan dictionary bank documents.
//!
//! The ZIP walk and the database writes live with their respective clients
//! (`data::importer` on the desktop and `DictionaryImporter` on Android).
//! This module owns the format-facing half instead: deciding which bank a
//! filename represents, ordering numbered banks, and turning bank text into
//! the shared dictionary row models.  It deliberately has no `db` or `zip`
//! dependency, so the Android/nav-only build can use exactly the same parser.

use anyhow::{Context, Result};
use serde_json::Value;

use crate::data::models::{DictionaryEntry, DictionaryTag};

/// The entry-bank kinds understood by the shared importer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BankType {
    /// A term bank (`term_bank_N.json`).
    Term,
    /// A kanji bank (`kanji_bank_N.json`).
    Kanji,
    /// A term metadata bank (`term_meta_bank_N.json`).
    TermMeta,
}

/// Classify an entry-bank ZIP name using the same prefixes as the desktop
/// importer.  Tag banks have their own helper because they are not counted as
/// entry banks.
///
/// The `.json` suffix and the bank prefix are both required.  The importer
/// still decides what to do with a row whose numeric suffix is malformed;
/// classification only identifies the file kind.
pub fn classify_bank(name: &str) -> Option<BankType> {
    if is_bank_named(name, "term_bank_") {
        Some(BankType::Term)
    } else if is_bank_named(name, "kanji_bank_") {
        Some(BankType::Kanji)
    } else if is_bank_named(name, "term_meta_bank_") {
        Some(BankType::TermMeta)
    } else {
        None
    }
}

/// Whether `name` is a tag bank.  Tag banks are processed in their own pass,
/// so they deliberately do not appear in the entry-bank [`BankType`].
pub fn is_tag_bank(name: &str) -> bool {
    is_bank_named(name, "tag_bank_")
}

/// Extract the numeric suffix from a bank filename.
///
/// `term_bank_12.json` and `term_meta_bank_3.json` yield `Some(12)` and
/// `Some(3)`.  A missing suffix, a non-`.json` name, or a non-numeric suffix
/// yields `None`; callers that need the old sort fallback can use
/// [`extract_bank_number`].
pub fn bank_number(name: &str) -> Option<u64> {
    let without_json = name.strip_suffix(".json")?;
    let suffix = without_json.rsplitn(2, '_').next()?;
    if suffix.is_empty() {
        None
    } else {
        suffix.parse().ok()
    }
}

/// Numeric bank suffix with the importer's historical sort fallback.
///
/// Unknown suffixes sort as zero.  The desktop importer uses this for its
/// stable natural ordering (`term_bank_2` before `term_bank_10`).
pub fn extract_bank_number(name: &str) -> u64 {
    bank_number(name).unwrap_or(0)
}

fn is_bank_named(name: &str, prefix: &str) -> bool {
    name.starts_with(prefix) && name.ends_with(".json")
}

/// Read the declared dictionary title from an `index.json` document.
///
/// A malformed document, a non-object document, or a missing/non-string title
/// has no title; callers can then retain their filename-derived fallback.  This
/// is the same best-effort contract as both ZIP clients.
pub fn parse_index_title(content: &str) -> Option<String> {
    let map: serde_json::Map<String, Value> = serde_json::from_str(content).ok()?;
    map.get("title")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

/// Turn a Yomitan string-or-array field into the importer's flat string form.
///
/// Arrays contribute only their string elements, joined with one space.  Null
/// is empty; other scalar/object values use their JSON spelling.  This is the
/// PC implementation that the Android reader previously duplicated.
pub fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Array(values) => values
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" "),
        Value::Null => String::new(),
        _ => value.to_string(),
    }
}

/// Parse a term bank into shared dictionary-entry rows.
///
/// A malformed top-level document is an error.  Individual rows that do not
/// have the required shape, or whose first two fields are not strings, are
/// skipped just as the old streaming readers skipped them.
pub fn parse_term_bank(content: &str, dictionary_id: i64) -> Result<Vec<DictionaryEntry>> {
    let data: Vec<Value> =
        serde_json::from_str(content).context("Failed to parse term bank JSON")?;
    let mut entries = Vec::new();

    for item in data {
        let arr = match item.as_array() {
            Some(values) if values.len() >= 5 => values,
            _ => continue,
        };

        // Android's `reader.nextString()` throws on a non-string row, so the
        // row is dropped; fail the same way instead of inserting an empty entry.
        let Some(kanji) = arr[0].as_str() else {
            continue;
        };
        let Some(reading) = arr[1].as_str() else {
            continue;
        };
        let kanji = kanji.to_string();
        let reading = reading.to_string();
        let tags1 = value_to_string(&arr[2]);
        let rules = value_to_string(&arr[3]);
        let popularity = arr[4].as_i64().unwrap_or(0) as i32;

        let definitions = if arr.len() > 5 {
            serde_json::to_string(&arr[5]).unwrap_or_default()
        } else {
            String::new()
        };

        // Sequence is part of the format but is not stored in the shared row.
        let _sequence = if arr.len() > 6 {
            arr[6].as_i64().unwrap_or(0)
        } else {
            0
        };
        let tags2 = if arr.len() > 7 {
            value_to_string(&arr[7])
        } else {
            String::new()
        };

        let combined_rules = format!("{} | {}", tags1, rules);
        let combined_rules = if tags2.is_empty() {
            combined_rules
        } else {
            format!("{} | {}", combined_rules, tags2)
        };

        entries.push(DictionaryEntry::new(
            kanji,
            reading,
            definitions,
            combined_rules,
            popularity,
            dictionary_id,
        ));
    }

    Ok(entries)
}

/// Parse a kanji bank into shared dictionary-entry rows.
pub fn parse_kanji_bank(content: &str, dictionary_id: i64) -> Result<Vec<DictionaryEntry>> {
    let data: Vec<Value> =
        serde_json::from_str(content).context("Failed to parse kanji bank JSON")?;
    let mut entries = Vec::new();

    for item in data {
        let arr = match item.as_array() {
            Some(values) if values.len() >= 4 => values,
            _ => continue,
        };

        let kanji = value_to_string(&arr[0]);
        let onyomi = value_to_string(&arr[1]);
        let kunyomi = value_to_string(&arr[2]);
        let grade_freq = value_to_string(&arr[3]);

        let definitions = if arr.len() > 4 {
            serde_json::to_string(&arr[4]).unwrap_or_default()
        } else {
            String::new()
        };

        let jlpt = if arr.len() > 5 {
            if let Some(meta) = arr[5].as_object() {
                meta.get("jlpt")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let rules = format!("grade:{}", grade_freq);

        entries.push(DictionaryEntry {
            id: 0,
            kanji: kanji.clone(),
            reading: onyomi.clone(),
            definitions,
            rules,
            popularity: 0,
            dictionary_id,
            onyomi: if onyomi.is_empty() {
                None
            } else {
                Some(onyomi)
            },
            kunyomi: if kunyomi.is_empty() {
                None
            } else {
                Some(kunyomi)
            },
            jlpt: if jlpt.is_empty() { None } else { Some(jlpt) },
        });
    }

    Ok(entries)
}

/// Parse the pitch-bearing rows of a term-meta bank.
///
/// Yomitan term-meta rows are `[term, type, data]`.  Only `pitch` rows with a
/// non-empty reading are retained; frequency and IPA rows are intentionally
/// skipped.  The payload is stored verbatim in `definitions` so render-time
/// pitch parsing sees the original integer positions.
pub fn parse_term_meta_bank(content: &str, dictionary_id: i64) -> Result<Vec<DictionaryEntry>> {
    let data: Vec<Value> =
        serde_json::from_str(content).context("Failed to parse term meta bank JSON")?;
    let mut entries = Vec::new();

    for item in data {
        let Some(arr) = item.as_array() else {
            continue;
        };
        if arr.len() < 3 {
            continue;
        }
        let term = arr[0].as_str().unwrap_or("");
        let meta_type = arr[1].as_str().unwrap_or("");
        let payload = &arr[2];
        if meta_type != "pitch" || term.is_empty() {
            continue;
        }
        let reading = payload
            .as_object()
            .and_then(|metadata| metadata.get("reading"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if reading.is_empty() {
            continue;
        }
        entries.push(DictionaryEntry::new(
            term.to_string(),
            reading.to_string(),
            payload.to_string(),
            String::new(),
            0,
            dictionary_id,
        ));
    }

    Ok(entries)
}

/// Parse a tag bank into shared dictionary-tag rows.
pub fn parse_tag_bank(content: &str, dictionary_id: i64) -> Result<Vec<DictionaryTag>> {
    let data: Vec<Value> =
        serde_json::from_str(content).context("Failed to parse tag bank JSON")?;
    let mut tags = Vec::new();

    for item in data {
        let arr = match item.as_array() {
            Some(values) if values.len() >= 5 => values,
            _ => continue,
        };

        let name = arr[0].as_str().unwrap_or("").to_string();
        let category = arr[1].as_str().unwrap_or("").to_string();
        let order = arr[2].as_i64().unwrap_or(0) as i32;
        let notes = arr[3].as_str().unwrap_or("").to_string();
        let popularity = arr[4].as_i64().unwrap_or(0) as i32;

        tags.push(DictionaryTag {
            id: 0,
            name,
            category,
            order,
            notes,
            popularity,
            dictionary_id,
        });
    }

    Ok(tags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_banks_and_extracts_natural_sort_numbers() {
        assert_eq!(classify_bank("term_bank_2.json"), Some(BankType::Term));
        assert_eq!(classify_bank("kanji_bank_12.json"), Some(BankType::Kanji));
        assert_eq!(
            classify_bank("term_meta_bank_3.json"),
            Some(BankType::TermMeta)
        );
        assert_eq!(classify_bank("tag_bank_1.json"), None);
        assert_eq!(classify_bank("term_bank_1.txt"), None);
        assert_eq!(classify_bank("other.json"), None);
        assert!(is_tag_bank("tag_bank_4.json"));

        assert_eq!(bank_number("term_bank_12.json"), Some(12));
        assert_eq!(bank_number("term_meta_bank_3.json"), Some(3));
        assert_eq!(bank_number("term_bank_2.txt"), None);
        assert_eq!(bank_number("term_bank_.json"), None);
        assert_eq!(extract_bank_number("term_bank_bad.json"), 0);

        let mut names = vec!["term_bank_10.json", "term_bank_2.json"];
        names.sort_by_key(|name| extract_bank_number(name));
        assert_eq!(names, vec!["term_bank_2.json", "term_bank_10.json"]);
    }

    #[test]
    fn term_rows_keep_definitions_and_flatten_rules() {
        let bank = r#"[
            ["吃","たべる","", "v1", 7, ["to eat"], 3, "common"],
            [null,"drop","", "", 0, ["drop"], 1, ""],
            ["欠","かく",["tag-a","tag-b"],"v5",2,{"type":"structured-content"},4,""]
        ]"#;
        let rows = parse_term_bank(bank, 41).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].kanji, "吃");
        assert_eq!(rows[0].reading, "たべる");
        assert_eq!(rows[0].definitions, r#"["to eat"]"#);
        assert_eq!(rows[0].rules, " | v1 | common");
        assert_eq!(rows[0].popularity, 7);
        assert_eq!(rows[0].dictionary_id, 41);
        assert_eq!(rows[1].rules, "tag-a tag-b | v5");
        assert_eq!(rows[1].definitions, r#"{"type":"structured-content"}"#);
    }

    #[test]
    fn kanji_rows_preserve_readings_and_optional_metadata() {
        let bank = r#"[
            ["漢",["カン","かん"],["た.","た.し"], "8", ["kanji"], {"jlpt":"N5"}],
            ["空", [], [], "", [], {}]
        ]"#;
        let rows = parse_kanji_bank(bank, 9).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].reading, "カン かん");
        assert_eq!(rows[0].onyomi.as_deref(), Some("カン かん"));
        assert_eq!(rows[0].kunyomi.as_deref(), Some("た. た.し"));
        assert_eq!(rows[0].rules, "grade:8");
        assert_eq!(rows[0].jlpt.as_deref(), Some("N5"));
        assert_eq!(rows[1].onyomi, None);
        assert_eq!(rows[1].kunyomi, None);
        assert_eq!(rows[1].jlpt, None);
    }

    #[test]
    fn term_meta_keeps_only_pitch_rows_with_a_reading() {
        let bank = r#"[
            ["分","pitch",{"reading":"ぶん","pitches":[{"position":1}]}],
            ["分","freq",{"value":100}],
            ["分","pitch",{"pitches":[{"position":2}]}],
            ["ふん","pitch",{"reading":"ふん","pitches":[{"position":2}]}]
        ]"#;
        let rows = parse_term_meta_bank(bank, 12).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].kanji, "分");
        assert_eq!(rows[0].reading, "ぶん");
        assert_eq!(
            rows[0].definitions,
            r#"{"reading":"ぶん","pitches":[{"position":1}]}"#
        );
        assert_eq!(rows[1].reading, "ふん");
        assert_eq!(rows[0].rules, "");
        assert_eq!(rows[0].popularity, 0);
    }

    #[test]
    fn tag_rows_and_index_title_follow_the_shared_shape() {
        let bank = r#"[["common","partOfSpeech",4,"keep",12],["x"]]"#;
        let rows = parse_tag_bank(bank, 3).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "common");
        assert_eq!(rows[0].category, "partOfSpeech");
        assert_eq!(rows[0].order, 4);
        assert_eq!(rows[0].notes, "keep");
        assert_eq!(rows[0].popularity, 12);
        assert_eq!(rows[0].dictionary_id, 3);
        assert_eq!(
            parse_index_title(r#"{"title":"Jitendex.org"}"#).as_deref(),
            Some("Jitendex.org")
        );
        assert_eq!(parse_index_title(r#"{"title":7}"#), None);
        assert_eq!(parse_index_title("not json"), None);
    }

    #[test]
    fn malformed_bank_documents_are_errors_but_bad_rows_are_skipped() {
        assert!(parse_term_bank("not json", 1).is_err());
        assert!(parse_kanji_bank("not json", 1).is_err());
        assert!(parse_term_meta_bank("not json", 1).is_err());
        assert!(parse_tag_bank("not json", 1).is_err());
        assert!(parse_term_bank(r#"[["ok","よみ",0,0,0]]"#, 1).is_ok());
        assert!(parse_kanji_bank(r#"[[]]"#, 1).unwrap().is_empty());
    }

    /// The shipped Jitendex fixture is already used by the definition-format
    /// tests.  Reuse it here as a realistic bank corpus: every structured
    /// definition must survive the parser's JSON boundary unchanged.
    #[test]
    fn jitendex_fixture_definitions_survive_term_bank_parsing() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../tests/data/jitendex/entries.json")).unwrap();
        let source_rows = fixture["entries"].as_array().unwrap();
        let bank_rows: Vec<Value> = source_rows
            .iter()
            .map(|row| {
                serde_json::json!([
                    row["term"],
                    row["reading"],
                    "",
                    "",
                    0,
                    row["definitions"],
                    0,
                    ""
                ])
            })
            .collect();
        let bank = serde_json::to_string(&bank_rows).unwrap();
        let parsed = parse_term_bank(&bank, 77).unwrap();
        assert_eq!(parsed.len(), source_rows.len());
        for (row, source) in parsed.iter().zip(source_rows) {
            assert_eq!(
                row.definitions,
                serde_json::to_string(&source["definitions"]).unwrap()
            );
        }
    }
}
