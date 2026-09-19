//! Yomitan dictionary ZIP file importer.
//! Mirrors `DictionaryImporter` from the Kotlin implementation.

use anyhow::{Context, Result};
use serde_json::Value;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::data::db::DictionaryDatabase;
use crate::data::models::{DictionaryEntry, DictionaryTag};

/// Progress callback for import operations.
/// Called periodically with entries imported so far and bank progress.
pub type ImportProgressFn = Box<dyn Fn(ImportProgress)>;

/// Progress state reported during import.
#[derive(Debug, Clone)]
pub struct ImportProgress {
    /// Total entries imported so far across all banks.
    pub entries_imported: usize,
    /// Number of term bank files fully processed.
    pub banks_done: usize,
    /// Total number of term/kanji bank files in the archive.
    pub banks_total: usize,
    /// Name of the current file being processed.
    pub current_file: String,
}

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
    /// If `progress` is provided, it is called periodically with progress updates.
    pub fn import_zip<P: AsRef<Path>>(
        &self,
        path: P,
        progress: Option<ImportProgressFn>,
    ) -> Result<usize> {
        let path = path.as_ref();
        let file = File::open(path).context("Failed to open ZIP file")?;
        let reader = BufReader::new(file);
        let mut archive =
            zip::ZipArchive::new(reader).context("Failed to read ZIP archive")?;

        let file_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Imported Dictionary");
        let mut dict_title = file_name.to_string();
        let mut dictionary_id: Option<i64> = None;
        let mut total_processed = 0;

        // First pass: collect term/kanji bank files for progress reporting.
        // Sort them by the numeric suffix so they import in natural order
        // (term_bank1, term_bank2, ..., term_bank10, ...).
        let mut bank_indices: Vec<(usize, String, BankType)> = Vec::new();
        for i in 0..archive.len() {
            let entry = archive.by_index(i).context("Failed to read ZIP entry")?;
            let name = entry.name().to_string();
            if name == "index.json" {
                // handled below
            } else if name.starts_with("term_bank_") && name.ends_with(".json") {
                bank_indices.push((i, name, BankType::Term));
            } else if name.starts_with("kanji_bank_") && name.ends_with(".json") {
                bank_indices.push((i, name, BankType::Kanji));
            } else if name.starts_with("term_meta_bank_") && name.ends_with(".json") {
                // #43: pitch-accent (and frequency, skipped) term metadata.
                bank_indices.push((i, name, BankType::TermMeta));
            }
        }

        // Sort by numeric suffix, treating the number as if zero-padded
        // so that e.g. term_bank2 comes before term_bank10.
        bank_indices.sort_by(|a, b| {
            let num_a = Self::extract_bank_number(&a.1);
            let num_b = Self::extract_bank_number(&b.1);
            num_a.cmp(&num_b)
        });

        let banks_total = bank_indices.len();
        let mut banks_done = 0;

        // Process index.json first if present.
        let mut declared_title: Option<String> = None;
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).context("Failed to read ZIP entry")?;
            let name = entry.name().to_string();
            if name == "index.json" {
                let mut content = String::new();
                entry.read_to_string(&mut content)?;
                if let Ok(map) =
                    serde_json::from_str::<serde_json::Map<String, Value>>(&content)
                {
                    if let Some(title) = map.get("title").and_then(|v| v.as_str()) {
                        dict_title = title.to_string();
                        declared_title = Some(title.to_string());
                        if let Some(id) = dictionary_id {
                            self.db.update_dictionary_name(id, &dict_title)?;
                        }
                    }
                }
            }
        }

        // Re-importing a dictionary replaces the copy that declares the same
        // title instead of stacking a duplicate (#43/#71).
        if let Some(title) = &declared_title {
            self.db.delete_dictionary_by_name(title)?;
        }

        // Process term/kanji banks.
        for (_idx, name, bank_type) in &bank_indices {
            let mut entry = archive
                .by_index(*_idx)
                .context("Failed to read ZIP entry")?;

            if dictionary_id.is_none() {
                let max_priority = self.db.get_max_priority()?.unwrap_or(-1);
                dictionary_id = Some(
                    self.db
                        .insert_dictionary(&dict_title, max_priority + 1)?,
                );
            }
            let did = dictionary_id.unwrap();

            let mut content = String::new();
            entry.read_to_string(&mut content)?;

            let count = match bank_type {
                BankType::Term => {
                    let entries = Self::parse_term_bank(&content, did)?;
                    let c = entries.len();
                    self.db.insert_entries(&entries)?;
                    c
                }
                BankType::Kanji => {
                    let entries = Self::parse_kanji_bank(&content, did)?;
                    let c = entries.len();
                    self.db.insert_entries(&entries)?;
                    c
                }
                BankType::TermMeta => {
                    let entries = Self::parse_term_meta_bank(&content, did)?;
                    let c = entries.len();
                    self.db.insert_entries(&entries)?;
                    c
                }
            };

            total_processed += count;
            banks_done += 1;

            if let Some(ref cb) = progress {
                cb(ImportProgress {
                    entries_imported: total_processed,
                    banks_done,
                    banks_total,
                    current_file: name.clone(),
                });
            }
        }

        // Process tag banks (not counted in progress).
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).context("Failed to read ZIP entry")?;
            let name = entry.name().to_string();
            if name.starts_with("tag_bank_") && name.ends_with(".json") {
                if dictionary_id.is_none() {
                    let max_priority = self.db.get_max_priority()?.unwrap_or(-1);
                    dictionary_id = Some(
                        self.db
                            .insert_dictionary(&dict_title, max_priority + 1)?,
                    );
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


    /// Extract the numeric suffix from a bank filename.
    /// E.g. "term_bank_12.json" -> 12, "kanji_bank_3.json" -> 3.
    fn extract_bank_number(name: &str) -> u64 {
        // Find the last underscore before ".json" and parse the number after it.
        let without_json = name.strip_suffix(".json").unwrap_or(name);
        without_json
            .rsplitn(2, '_')
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    }

    fn parse_term_bank(content: &str, dictionary_id: i64) -> Result<Vec<DictionaryEntry>> {
        let data: Vec<Value> = serde_json::from_str(content)
            .context("Failed to parse term bank JSON")?;
        let mut entries = Vec::new();

        for item in data {
            let arr = match item.as_array() {
                Some(a) if a.len() >= 5 => a,
                _ => continue,
            };

            // Android's `reader.nextString()` throws on a non-string row, so
            // the row is dropped; fail the same way instead of inserting an
            // empty entry.
            let Some(kanji) = arr[0].as_str() else {
                continue;
            };
            let Some(reading) = arr[1].as_str() else {
                continue;
            };
            let kanji = kanji.to_string();
            let reading = reading.to_string();
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
        let data: Vec<Value> = serde_json::from_str(content)
            .context("Failed to parse kanji bank JSON")?;
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
                    meta.get("jlpt")
                        .and_then(|v| v.as_str())
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

    /// #43: Yomitan term-meta bank rows are `[term, type, data]`. Only `pitch`
    /// rows with a reading are kept (freq/ipa skipped); each becomes an entry
    /// keyed on the term with its reading and the data stored verbatim as the
    /// definition payload so render-time parsing sees exact integers.
    fn parse_term_meta_bank(content: &str, dictionary_id: i64) -> Result<Vec<DictionaryEntry>> {
        let data: Vec<Value> = serde_json::from_str(content)
            .context("Failed to parse term meta bank JSON")?;
        let mut entries = Vec::new();

        for item in data {
            let Some(arr) = item.as_array() else { continue };
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
                .and_then(|m| m.get("reading"))
                .and_then(|v| v.as_str())
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

    fn parse_tag_bank(content: &str, dictionary_id: i64) -> Result<Vec<DictionaryTag>> {
        let data: Vec<Value> = serde_json::from_str(content)
            .context("Failed to parse tag bank JSON")?;
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

#[derive(Debug, Clone, Copy)]
enum BankType {
    Term,
    Kanji,
    TermMeta,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay_state::OcrOverlayState;
    use std::io::Write;
    use std::path::PathBuf;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ijd_importer_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_zip(path: &Path, files: &[(&str, &str)]) {
        let file = File::create(path).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::FileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, content) in files {
            writer.start_file(*name, options).unwrap();
            writer.write_all(content.as_bytes()).unwrap();
        }
        writer.finish().unwrap();
    }

    #[test]
    fn reimporting_the_same_zip_replaces_the_dictionary() {
        let dir = scratch_dir("idempotent");
        let db = DictionaryDatabase::open(dir.join("dict.db")).unwrap();
        let zip_path = dir.join("jitendex.zip");
        write_zip(
            &zip_path,
            &[
                ("index.json", r#"{"title":"Jitendex.org"}"#),
                (
                    "term_bank_1.json",
                    r#"[["支持杭","しじぐい","","",0,["bearing pile"],1,""]]"#,
                ),
            ],
        );
        let importer = DictionaryImporter::new(&db);
        importer.import_zip(&zip_path, None).unwrap();
        importer.import_zip(&zip_path, None).unwrap();
        assert_eq!(db.get_entry_count().unwrap(), 1);
        assert_eq!(db.get_all_dictionaries().unwrap().len(), 1);
    }

    #[test]
    fn term_meta_pitch_rows_are_imported_and_verifiable() {
        let dir = scratch_dir("pitch");
        let db = DictionaryDatabase::open(dir.join("dict.db")).unwrap();
        let zip_path = dir.join("kanjium.zip");
        write_zip(
            &zip_path,
            &[
                ("index.json", r#"{"title":"Kanjium"}"#),
                (
                    "term_meta_bank_1.json",
                    r#"[["分","pitch",{"reading":"ぶん","pitches":[{"position":1}]}],
                       ["分","freq",{"value":100}],
                       ["ふん","pitch",{"reading":"ふん","pitches":[{"position":2}]}]]"#,
                ),
            ],
        );
        DictionaryImporter::new(&db).import_zip(&zip_path, None).unwrap();
        let rows = db.find_by_texts(&["分".to_string()]).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].reading, "ぶん");
        assert_eq!(
            OcrOverlayState::pitch_positions_of(&rows[0].definitions),
            Some(vec![1])
        );
    }

    #[test]
    fn malformed_term_rows_are_dropped() {
        let dir = scratch_dir("malformed");
        let db = DictionaryDatabase::open(dir.join("dict.db")).unwrap();
        let zip_path = dir.join("odd.zip");
        write_zip(
            &zip_path,
            &[
                ("index.json", r#"{"title":"Odd"}"#),
                (
                    "term_bank_1.json",
                    r#"[[null,"x","","",0,["dropped"],1,""],
                        [123,"y","","",0,["dropped"],2,""],
                        ["ok","よみ","","",0,["kept"],3,""]]"#,
                ),
            ],
        );
        DictionaryImporter::new(&db).import_zip(&zip_path, None).unwrap();
        let rows = db.find_by_texts(&["ok".to_string()]).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kanji, "ok");
        assert_eq!(db.get_entry_count().unwrap(), 1);
    }
}
