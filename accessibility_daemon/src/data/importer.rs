//! Yomitan dictionary ZIP file importer.
//! Mirrors `DictionaryImporter` from the Kotlin implementation.

use anyhow::{Context, Result};
use serde_json::Value;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::data::db::DictionaryDatabase;
use crate::data::models::{DictionaryEntry, DictionaryTag};

/// Imports Yomitan-format dictionary ZIP files into the database.
pub struct DictionaryImporter<'a> {
    db: &'a DictionaryDatabase,
}

impl<'a> DictionaryImporter<'a> {
    pub fn new(db: &'a DictionaryDatabase) -> Self {
        Self { db }
    }

    /// Import a Yomitan dictionary ZIP file.
    /// Returns the number of entries imported, or an error.
    pub fn import_zip<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        let path = path.as_ref();
        let file = File::open(path).context("Failed to open ZIP file")?;
        let reader = BufReader::new(file);
        let mut archive = zip::ZipArchive::new(reader).context("Failed to read ZIP archive")?;

        let file_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Imported Dictionary");
        let mut dict_title = file_name.to_string();
        let mut dictionary_id: Option<i64> = None;
        let mut total_processed = 0;

        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).context("Failed to read ZIP entry")?;
            let name = entry.name().to_string();

            if name == "index.json" {
                let mut content = String::new();
                entry.read_to_string(&mut content)?;
                if let Ok(map) = serde_json::from_str::<serde_json::Map<String, Value>>(&content) {
                    if let Some(title) = map.get("title").and_then(|v| v.as_str()) {
                        dict_title = title.to_string();
                        if let Some(id) = dictionary_id {
                            self.db.update_dictionary_name(id, &dict_title)?;
                        }
                    }
                }
            } else if name.starts_with("term_bank_") && name.ends_with(".json") {
                if dictionary_id.is_none() {
                    let max_priority = self.db.get_max_priority()?.unwrap_or(-1);
                    dictionary_id = Some(self.db.insert_dictionary(&dict_title, max_priority + 1)?);
                }
                let did = dictionary_id.unwrap();
                let mut content = String::new();
                entry.read_to_string(&mut content)?;
                let entries = Self::parse_term_bank(&content, did)?;
                let count = entries.len();
                self.db.insert_entries(&entries)?;
                total_processed += count;
            } else if name.starts_with("kanji_bank_") && name.ends_with(".json") {
                if dictionary_id.is_none() {
                    let max_priority = self.db.get_max_priority()?.unwrap_or(-1);
                    dictionary_id = Some(self.db.insert_dictionary(&dict_title, max_priority + 1)?);
                }
                let did = dictionary_id.unwrap();
                let mut content = String::new();
                entry.read_to_string(&mut content)?;
                let entries = Self::parse_kanji_bank(&content, did)?;
                let count = entries.len();
                self.db.insert_entries(&entries)?;
                total_processed += count;
            } else if name.starts_with("tag_bank_") && name.ends_with(".json") {
                if dictionary_id.is_none() {
                    let max_priority = self.db.get_max_priority()?.unwrap_or(-1);
                    dictionary_id = Some(self.db.insert_dictionary(&dict_title, max_priority + 1)?);
                }
                let did = dictionary_id.unwrap();
                let mut content = String::new();
                entry.read_to_string(&mut content)?;
                let tags = Self::parse_tag_bank(&content, did)?;
                self.db.insert_tags(&tags)?;
            }
        }

        println!(
            "Imported {} entries from '{}'",
            total_processed, dict_title
        );
        Ok(total_processed)
    }

    fn parse_term_bank(content: &str, dictionary_id: i64) -> Result<Vec<DictionaryEntry>> {
        let data: Vec<Value> = serde_json::from_str(content).context("Failed to parse term bank JSON")?;
        let mut entries = Vec::new();

        for item in data {
            let arr = match item.as_array() {
                Some(a) if a.len() >= 5 => a,
                _ => continue,
            };

            let kanji = arr[0].as_str().unwrap_or("").to_string();
            let reading = arr[1].as_str().unwrap_or("").to_string();
            let tags1 = Self::value_to_string(&arr[2]);
            let rules = Self::value_to_string(&arr[3]);
            let popularity = arr[4].as_i64().unwrap_or(0) as i32;

            let definitions = if arr.len() > 5 {
                serde_json::to_string(&arr[5]).unwrap_or_default()
            } else {
                String::new()
            };

            let _sequence = if arr.len() > 6 {
                arr[6].as_i64().unwrap_or(0)
            } else {
                0
            };
            let tags2 = if arr.len() > 7 {
                Self::value_to_string(&arr[7])
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

    fn parse_kanji_bank(content: &str, dictionary_id: i64) -> Result<Vec<DictionaryEntry>> {
        let data: Vec<Value> = serde_json::from_str(content).context("Failed to parse kanji bank JSON")?;
        let mut entries = Vec::new();

        for item in data {
            let arr = match item.as_array() {
                Some(a) if a.len() >= 4 => a,
                _ => continue,
            };

            let kanji = Self::value_to_string(&arr[0]);
            let onyomi = Self::value_to_string(&arr[1]);
            let kunyomi = Self::value_to_string(&arr[2]);
            let grade_freq = Self::value_to_string(&arr[3]);

            let definitions = if arr.len() > 4 {
                serde_json::to_string(&arr[4]).unwrap_or_default()
            } else {
                String::new()
            };

            let jlpt = if arr.len() > 5 {
                if let Some(meta) = arr[5].as_object() {
                    meta.get("jlpt").and_then(|v| v.as_str()).unwrap_or("").to_string()
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
                onyomi: if onyomi.is_empty() { None } else { Some(onyomi) },
                kunyomi: if kunyomi.is_empty() { None } else { Some(kunyomi) },
                jlpt: if jlpt.is_empty() { None } else { Some(jlpt) },
            });
        }

        Ok(entries)
    }

    fn parse_tag_bank(content: &str, dictionary_id: i64) -> Result<Vec<DictionaryTag>> {
        let data: Vec<Value> = serde_json::from_str(content).context("Failed to parse tag bank JSON")?;
        let mut tags = Vec::new();

        for item in data {
            let arr = match item.as_array() {
                Some(a) if a.len() >= 5 => a,
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

    fn value_to_string(v: &Value) -> String {
        match v {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            Value::Null => String::new(),
            _ => v.to_string(),
        }
    }
}
