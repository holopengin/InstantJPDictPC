//! Yomitan dictionary ZIP file importer.
//! Mirrors `DictionaryImporter` from the Kotlin implementation.

use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::data::db::DictionaryDatabase;
use crate::yomitan_parse::{
    classify_bank, extract_bank_number, is_tag_bank, parse_index_title, parse_kanji_bank,
    parse_tag_bank, parse_term_bank, parse_term_meta_bank, BankType,
};

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

/// Provenance for an import beyond the plain file-picker path.
/// Mirrors Android's `importZipStream` keyword arguments.
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// True for the dictionary bundled with the app. The flag is written to
    /// the meta row only after every bank has landed (Android's completion
    /// marker), so an import killed part-way is retried next start, and the
    /// settings manager keeps the completed row from being deleted.
    pub built_in: bool,
    /// Stable id of the catalog entry this import came from, if any.
    /// `None` for the file picker.
    pub catalog_id: Option<String>,
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
        self.import_zip_with(path, progress, ImportOptions::default())
    }

    /// Import a Yomitan dictionary ZIP file, stamping the meta row with the
    /// provenance in `options` (catalog id). `import_zip` is
    /// this with [`ImportOptions::default`].
    pub fn import_zip_with<P: AsRef<Path>>(
        &self,
        path: P,
        progress: Option<ImportProgressFn>,
        options: ImportOptions,
    ) -> Result<usize> {
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

        // First pass: collect term/kanji/term-meta bank files for progress
        // reporting.  Bank classification and the numeric sort key live in
        // the ungated parser so the filename contract is shared with Android.
        let mut bank_indices: Vec<(usize, String, BankType)> = Vec::new();
        for i in 0..archive.len() {
            let entry = archive.by_index(i).context("Failed to read ZIP entry")?;
            let name = entry.name().to_string();
            if let Some(bank_type) = classify_bank(&name) {
                bank_indices.push((i, name, bank_type));
            }
        }

        // Sort by numeric suffix so that e.g. term_bank_2 comes before
        // term_bank_10.  Unknown suffixes retain the historical zero fallback.
        bank_indices.sort_by(|a, b| extract_bank_number(&a.1).cmp(&extract_bank_number(&b.1)));

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
                if let Some(title) = parse_index_title(&content) {
                    dict_title = title.clone();
                    declared_title = Some(title);
                    if let Some(id) = dictionary_id {
                        self.db.update_dictionary_name(id, &dict_title)?;
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
                dictionary_id = Some(self.db.insert_dictionary_with(
                    &dict_title,
                    max_priority + 1,
                    false,
                    options.catalog_id.as_deref(),
                )?);
            }
            let did = dictionary_id.unwrap();

            let mut content = String::new();
            entry.read_to_string(&mut content)?;

            let count = match bank_type {
                BankType::Term => {
                    let entries = parse_term_bank(&content, did)?;
                    let c = entries.len();
                    self.db.insert_entries(&entries)?;
                    c
                }
                BankType::Kanji => {
                    let entries = parse_kanji_bank(&content, did)?;
                    let c = entries.len();
                    self.db.insert_entries(&entries)?;
                    c
                }
                BankType::TermMeta => {
                    let entries = parse_term_meta_bank(&content, did)?;
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
            if is_tag_bank(&name) {
                if dictionary_id.is_none() {
                    let max_priority = self.db.get_max_priority()?.unwrap_or(-1);
                    dictionary_id = Some(self.db.insert_dictionary_with(
                        &dict_title,
                        max_priority + 1,
                        false,
                        options.catalog_id.as_deref(),
                    )?);
                }
                let did = dictionary_id.unwrap();
                let mut content = String::new();
                entry.read_to_string(&mut content)?;
                let tags = parse_tag_bank(&content, did)?;
                self.db.insert_tags(&tags)?;
            }
        }

        // `built_in` is the completion marker, not a label. Flipping it only
        // after every bank is written means an import killed part-way leaves a
        // non-built-in row, so the next launch imports again instead of
        // trusting a half-present dictionary.
        if options.built_in {
            if let Some(id) = dictionary_id {
                self.db.set_dictionary_built_in(id, true)?;
            }
        }

        println!("Imported {} entries from '{}'", total_processed, dict_title);
        Ok(total_processed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay_state::OcrOverlayState;
    use std::io::Write;
    use std::path::PathBuf;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ijd_importer_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_zip(path: &Path, files: &[(&str, &str)]) {
        let file = File::create(path).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
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
        DictionaryImporter::new(&db)
            .import_zip(&zip_path, None)
            .unwrap();
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
        DictionaryImporter::new(&db)
            .import_zip(&zip_path, None)
            .unwrap();
        let rows = db.find_by_texts(&["ok".to_string()]).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kanji, "ok");
        assert_eq!(db.get_entry_count().unwrap(), 1);
    }
}
