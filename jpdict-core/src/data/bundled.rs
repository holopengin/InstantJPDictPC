//! Bundled dictionary auto-install.
//!
//! Mirrors Android's first-launch install of the vendored pitch dictionary
//! (#43): the Kanjium pitch accents zip ships in `assets/dictionaries/` and is
//! imported once per install, on a worker thread, before any GUI work. It is
//! the *only* bundled dictionary — Jitendex and KANJIDIC are one-click catalog
//! downloads (`src/data/catalog.rs`).
//!
//! Idempotent: the zip is imported only when its stable title family is absent
//! from the database, and a completed install is marked `built_in` so later
//! starts skip it after one cheap query. The `built_in` flag is also what
//! keeps the settings manager from deleting the row. Nothing is downloaded —
//! the zip travels next to the binary.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::data::db::DictionaryDatabase;
use crate::data::importer::{DictionaryImporter, ImportOptions};

/// One zip shipped in `assets/dictionaries/`.
pub struct BundledDictionary {
    /// File name inside the assets `dictionaries/` directory.
    pub file: &'static str,
    /// Stable title family the installed `dictionary_meta.name` is matched
    /// against: the zip's declared title may carry a revision suffix
    /// (`Kanjium Pitch Accents` has none, but the rule is the catalog's).
    pub family: &'static str,
    /// Id stamped on `dictionary_meta.catalog_id`. Doubles as the retry
    /// marker: a family whose only rows carry this id with `built_in = 0` is
    /// our own interrupted install and is replaced.
    pub id: &'static str,
    /// Upstream pin, written to the install log for provenance.
    pub source: &'static str,
}

/// The zip this build ships, pinned in `assets/dictionaries/PROVENANCE.txt`.
pub static BUNDLED_DICTIONARIES: &[BundledDictionary] = &[BundledDictionary {
    file: "kanjium_pitch_accents.zip",
    family: "Kanjium Pitch Accents",
    id: "kanjium-pitch-accents",
    source: "mifunetoshiro/kanjium @ 685d4d723d6d",
}];

/// Install every bundled zip missing from `db`, reading the zips from
/// `dict_dir`. Returns how many imports were actually performed.
///
/// Idempotent and fast when everything is present: each dictionary costs one
/// family query against `dictionary_meta`. A zip that is missing from disk is
/// logged and skipped — this runs at start-up, so it must never abort it.
pub fn ensure_bundled_dictionaries(
    db: &DictionaryDatabase,
    dict_dir: &Path,
    specs: &[BundledDictionary],
) -> Result<usize> {
    let mut installed = 0;
    for spec in specs {
        if dictionary_is_present(db, spec)? {
            continue;
        }
        let path = dict_dir.join(spec.file);
        if !path.exists() {
            eprintln!(
                "[BundledDicts] {} not found at {}; skipping",
                spec.file,
                path.display()
            );
            continue;
        }
        let t_start = std::time::Instant::now();
        let entries = DictionaryImporter::new(db)
            .import_zip_with(
                &path,
                None,
                ImportOptions {
                    built_in: true,
                    catalog_id: Some(spec.id.to_string()),
                },
            )
            .with_context(|| format!("bundled import failed for {}", spec.file))?;
        installed += 1;
        println!(
            "[BundledDicts] Installed '{}' ({} entries) from {} in {:.0} ms",
            spec.family,
            entries,
            spec.source,
            t_start.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(installed)
}

/// Whether a bundled zip has nothing to do: a completed built-in copy is
/// present, or the family is occupied by something we did not install (the
/// file picker, and the catalog) — the bundled zip must never stack a
/// duplicate title next to a user's copy. Only rows that are ours and not yet
/// marked complete mean "retry".
fn dictionary_is_present(db: &DictionaryDatabase, spec: &BundledDictionary) -> Result<bool> {
    let rows: Vec<_> = db
        .get_all_dictionaries()?
        .into_iter()
        .filter(|d| crate::data::db::name_matches_family(&d.name, spec.family))
        .collect();
    if rows.iter().any(|d| d.built_in) {
        return Ok(true);
    }
    Ok(rows
        .iter()
        .any(|d| d.catalog_id.as_deref() != Some(spec.id)))
}

/// Run [`ensure_bundled_dictionaries`] on a worker thread, so a first-run
/// import never delays the GUI or the bootstrap thread. Lookups read the
/// database live, so no completion message is needed; failures are logged and
/// the next start retries.
pub fn spawn_bundled_install(db: Arc<DictionaryDatabase>, assets_dir: PathBuf) {
    let _ = std::thread::Builder::new()
        .name("bundled-dicts".into())
        .spawn(move || {
            let dict_dir = assets_dir.join("dictionaries");
            match ensure_bundled_dictionaries(&db, &dict_dir, BUNDLED_DICTIONARIES) {
                Ok(0) => {}
                Ok(n) => println!("[BundledDicts] {n} dictionary(ies) installed"),
                Err(e) => eprintln!("[BundledDicts] install failed: {e:#}"),
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::models::DictionaryEntry;
    use std::io::Write;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ijd_bundled_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_zip(path: &Path, files: &[(&str, &str)]) {
        let file = std::fs::File::create(path).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::FileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, content) in files {
            writer.start_file(*name, options).unwrap();
            writer.write_all(content.as_bytes()).unwrap();
        }
        writer.finish().unwrap();
    }

    /// Two specs: one title exactly the family, one carrying a revision —
    /// together they pin every match form the install uses.
    const TEST_SPECS: &[BundledDictionary] = &[
        BundledDictionary {
            file: "exact.zip",
            family: "Exact Family",
            id: "exact",
            source: "test",
        },
        BundledDictionary {
            file: "revised.zip",
            family: "Revised Family",
            id: "revised",
            source: "test",
        },
    ];

    fn write_test_zips(dir: &Path) {
        write_zip(
            &dir.join("exact.zip"),
            &[
                ("index.json", r#"{"title":"Exact Family"}"#),
                (
                    "term_bank_1.json",
                    r#"[["分","ぶん","","",0,["exact"],1,""]]"#,
                ),
            ],
        );
        write_zip(
            &dir.join("revised.zip"),
            &[
                ("index.json", r#"{"title":"Revised Family [2026-01-01]"}"#),
                (
                    "term_bank_1.json",
                    r#"[["時","とき","","",0,["revised"],1,""]]"#,
                ),
            ],
        );
    }

    #[test]
    fn first_run_installs_each_missing_zip_as_built_in() {
        let dir = scratch_dir("first_run");
        write_test_zips(&dir);
        let db = DictionaryDatabase::open(dir.join("dict.db")).unwrap();

        let installed = ensure_bundled_dictionaries(&db, &dir, TEST_SPECS).unwrap();
        assert_eq!(installed, 2);
        assert_eq!(db.get_entry_count().unwrap(), 2);

        let metas = db.get_all_dictionaries().unwrap();
        assert_eq!(metas.len(), 2);
        for meta in &metas {
            assert!(meta.built_in, "{} should be built in", meta.name);
        }
        let exact = metas.iter().find(|m| m.name == "Exact Family").unwrap();
        assert_eq!(exact.catalog_id.as_deref(), Some("exact"));
        let revised = metas
            .iter()
            .find(|m| m.name == "Revised Family [2026-01-01]")
            .unwrap();
        assert_eq!(revised.catalog_id.as_deref(), Some("revised"));
    }

    #[test]
    fn second_run_skips_present_dictionaries_without_duplicating() {
        let dir = scratch_dir("skip_present");
        write_test_zips(&dir);
        let db = DictionaryDatabase::open(dir.join("dict.db")).unwrap();

        assert_eq!(ensure_bundled_dictionaries(&db, &dir, TEST_SPECS).unwrap(), 2);
        assert_eq!(ensure_bundled_dictionaries(&db, &dir, TEST_SPECS).unwrap(), 0);

        assert_eq!(db.get_all_dictionaries().unwrap().len(), 2);
        assert_eq!(db.get_entry_count().unwrap(), 2);
    }

    #[test]
    fn a_user_imported_family_blocks_the_bundled_install() {
        let dir = scratch_dir("user_copy");
        write_test_zips(&dir);
        let db = DictionaryDatabase::open(dir.join("dict.db")).unwrap();

        // What the file picker leaves behind: same family, no catalog id.
        let user_id = db
            .insert_dictionary("Revised Family [2020-05-05]", 0)
            .unwrap();
        db.insert_entries(&[DictionaryEntry::new(
            "時".into(),
            "とき".into(),
            r#"["mine"]"#.into(),
            String::new(),
            0,
            user_id,
        )])
        .unwrap();

        let installed = ensure_bundled_dictionaries(&db, &dir, TEST_SPECS).unwrap();
        assert_eq!(installed, 1, "only the family with no user copy installs");

        let metas = db.get_all_dictionaries().unwrap();
        let revised = metas
            .iter()
            .find(|d| d.name.starts_with("Revised Family"))
            .unwrap();
        assert!(!revised.built_in);
        assert_eq!(revised.catalog_id, None);
        assert!(metas.iter().any(|d| d.name == "Exact Family" && d.built_in));

        assert_eq!(db.get_entry_count().unwrap(), 2);
        let rows = db.find_by_texts(&["時".to_string()]).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].definitions.contains("mine"));
    }

    #[test]
    fn an_interrupted_bundled_install_is_replaced() {
        let dir = scratch_dir("interrupted");
        write_test_zips(&dir);
        let db = DictionaryDatabase::open(dir.join("dict.db")).unwrap();

        // What a crash mid-import leaves behind: our catalog id, no built-in
        // flag, partial rows.
        let partial_id = db
            .insert_dictionary_with("Revised Family [2026-01-01]", 0, false, Some("revised"))
            .unwrap();
        db.insert_entries(&[DictionaryEntry::new(
            "時".into(),
            "とき".into(),
            r#"["stale"]"#.into(),
            String::new(),
            0,
            partial_id,
        )])
        .unwrap();

        assert_eq!(ensure_bundled_dictionaries(&db, &dir, TEST_SPECS).unwrap(), 2);

        let metas = db.get_all_dictionaries().unwrap();
        assert_eq!(metas.len(), 2);
        assert!(metas.iter().all(|d| d.built_in));
        assert_eq!(db.get_entry_count().unwrap(), 2);
        let rows = db.find_by_texts(&["時".to_string()]).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].definitions.contains("revised"));
    }
}
