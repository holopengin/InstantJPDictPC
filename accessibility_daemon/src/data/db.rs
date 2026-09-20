//! SQLite database connection and schema management.
//! Mirrors `AppDatabase` from the Kotlin implementation.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::data::models::{DictionaryEntry, DictionaryMeta, DictionaryTag};

/// Thread-safe SQLite database handle.
/// Equivalent to `AppDatabase` in Kotlin.
pub struct DictionaryDatabase {
    conn: Arc<Mutex<Connection>>,
}

impl DictionaryDatabase {
    /// Open or create a dictionary database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path).context("Failed to open dictionary database")?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        db.init_schema()?;
        Ok(db)
    }


    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS dictionary_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                enabled INTEGER NOT NULL DEFAULT 1,
                catalog_id TEXT
            );

            CREATE TABLE IF NOT EXISTS dictionary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kanji TEXT NOT NULL,
                reading TEXT NOT NULL,
                definitions TEXT NOT NULL DEFAULT '',
                rules TEXT NOT NULL DEFAULT '',
                popularity INTEGER NOT NULL DEFAULT 0,
                dictionary_id INTEGER NOT NULL,
                onyomi TEXT,
                kunyomi TEXT,
                jlpt TEXT,
                FOREIGN KEY (dictionary_id) REFERENCES dictionary_meta(id)
            );

            CREATE INDEX IF NOT EXISTS idx_dictionary_kanji ON dictionary(kanji);
            CREATE INDEX IF NOT EXISTS idx_dictionary_reading ON dictionary(reading);
            CREATE INDEX IF NOT EXISTS idx_dictionary_dict_id ON dictionary(dictionary_id);

            CREATE TABLE IF NOT EXISTS dictionary_tag (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT '',
                order_val INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                popularity INTEGER NOT NULL DEFAULT 0,
                dictionary_id INTEGER NOT NULL,
                FOREIGN KEY (dictionary_id) REFERENCES dictionary_meta(id)
            );

            CREATE INDEX IF NOT EXISTS idx_tag_name ON dictionary_tag(name);
            CREATE INDEX IF NOT EXISTS idx_tag_dict_id ON dictionary_tag(dictionary_id);
            ",
        )?;

        // #71 follow-up (D1.8): databases created before the download catalog
        // existed have no catalog_id. CREATE TABLE IF NOT EXISTS is a no-op on
        // those, so the column is added by hand — the same ALTER-shape as
        // Android's catalogId DDL, chosen over a destructive rebuild so a
        // populated dictionary.sqlite keeps every imported row. (The old
        // `built_in` column is left in place where it exists and is no longer
        // read or written: nothing is bundled any more.)
        Self::ensure_column(
            &conn,
            "dictionary_meta",
            "catalog_id",
            "ALTER TABLE dictionary_meta ADD COLUMN catalog_id TEXT",
        )?;
        Ok(())
    }

    /// Add `column` to `table` when an older database is missing it.
    /// `CREATE TABLE IF NOT EXISTS` cannot alter an existing table, and
    /// `PRAGMA table_info` is what tells the two cases apart.
    fn ensure_column(conn: &Connection, table: &str, column: &str, ddl: &str) -> Result<()> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let name: String = row.get(1)?;
            if name == column {
                return Ok(());
            }
        }
        conn.execute_batch(ddl)?;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Dictionary meta CRUD
    // -------------------------------------------------------------------------

    /// Test convenience wrapper for a plain dictionary row.
    #[cfg(test)]
    pub fn insert_dictionary(&self, name: &str, priority: i32) -> Result<i64> {
        self.insert_dictionary_with(name, priority, None)
    }

    /// Insert a meta row with the full provenance the importer tracks.
    pub fn insert_dictionary_with(
        &self,
        name: &str,
        priority: i32,
        catalog_id: Option<&str>,
    ) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO dictionary_meta (name, priority, catalog_id)
             VALUES (?1, ?2, ?3)",
            rusqlite::params![name, priority, catalog_id],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn get_all_dictionaries(&self) -> Result<Vec<DictionaryMeta>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, name, priority, enabled, catalog_id
             FROM dictionary_meta ORDER BY priority ASC",
        )?;
        let dicts = stmt
            .query_map([], |row| {
                Ok(DictionaryMeta {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    priority: row.get(2)?,
                    enabled: row.get::<_, i32>(3)? != 0,
                    catalog_id: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(dicts)
    }

    pub fn delete_dictionary(&self, dictionary_id: i64) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM dictionary WHERE dictionary_id = ?1",
            [&dictionary_id],
        )?;
        conn.execute(
            "DELETE FROM dictionary_tag WHERE dictionary_id = ?1",
            [&dictionary_id],
        )?;
        conn.execute(
            "DELETE FROM dictionary_meta WHERE id = ?1",
            [&dictionary_id],
        )?;
        Ok(())
    }

    /// Remove any existing dictionary with this declared title, so a re-import
    /// replaces it instead of stacking a duplicate.
    /// Mirrors Android's `DictionaryImporter.replaceExisting`.
    pub fn delete_dictionary_by_name(&self, name: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let existing: Option<i64> = conn
            .query_row(
                "SELECT id FROM dictionary_meta WHERE name = ?1",
                [name],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(id) = existing {
            conn.execute(
                "DELETE FROM dictionary WHERE dictionary_id = ?1",
                [&id],
            )?;
            conn.execute(
                "DELETE FROM dictionary_tag WHERE dictionary_id = ?1",
                [&id],
            )?;
            conn.execute("DELETE FROM dictionary_meta WHERE id = ?1", [&id])?;
        }
        Ok(())
    }

    pub fn update_dictionary_name(&self, dictionary_id: i64, name: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE dictionary_meta SET name = ?1 WHERE id = ?2",
            [name, &dictionary_id.to_string()],
        )?;
        Ok(())
    }

    pub fn update_dictionary_priority(&self, dictionary_id: i64, priority: i32) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE dictionary_meta SET priority = ?1 WHERE id = ?2",
            [&priority.to_string(), &dictionary_id.to_string()],
        )?;
        Ok(())
    }

    pub fn set_dictionary_enabled(&self, dictionary_id: i64, enabled: bool) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE dictionary_meta SET enabled = ?1 WHERE id = ?2",
            [&enabled as &dyn rusqlite::ToSql, &dictionary_id as &dyn rusqlite::ToSql],
        )?;
        Ok(())
    }

    pub fn get_max_priority(&self) -> Result<Option<i32>> {
        let conn = self.conn.lock().unwrap();
        let val: Option<i32> = conn
            .query_row("SELECT MAX(priority) FROM dictionary_meta", [], |row| {
                // MAX() returns NULL when table is empty; handle as None
                let val: Option<i64> = row.get(0)?;
                Ok(val.map(|v| v as i32))
            })
            .optional()?
            .and_then(|x| x);
        Ok(val)
    }

    /// id → display name for every dictionary, used for per-entry source
    /// captions. Mirrors Android's `DictionaryProvider.dictionaryNames()`.
    pub fn dictionary_names(&self) -> Result<std::collections::HashMap<i64, String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, name FROM dictionary_meta")?;
        let names = stmt
            .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)))?
            .collect::<Result<std::collections::HashMap<_, _>, _>>()?;
        Ok(names)
    }

    // -------------------------------------------------------------------------
    // Entry CRUD
    // -------------------------------------------------------------------------

    pub fn insert_entries(&self, entries: &[DictionaryEntry]) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        for entry in entries {
            tx.execute(
                "INSERT OR REPLACE INTO dictionary
                 (kanji, reading, definitions, rules, popularity, dictionary_id, onyomi, kunyomi, jlpt)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    &entry.kanji,
                    &entry.reading,
                    &entry.definitions,
                    &entry.rules,
                    entry.popularity,
                    entry.dictionary_id,
                    &entry.onyomi,
                    &entry.kunyomi,
                    &entry.jlpt,
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Find entries matching any of the given texts (against kanji or reading).
    /// Results are ordered by dictionary priority ascending, then entry popularity descending.
    /// Equivalent to `DictionaryDao.findByTexts()` in Kotlin.
    pub fn find_by_texts(&self, texts: &[String]) -> Result<Vec<DictionaryEntry>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = texts.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT d.id, d.kanji, d.reading, d.definitions, d.rules, d.popularity,
                    d.dictionary_id, d.onyomi, d.kunyomi, d.jlpt
             FROM dictionary d
             JOIN dictionary_meta m ON d.dictionary_id = m.id
             WHERE m.enabled = 1
               AND (d.kanji IN ({}) OR d.reading IN ({}))
             ORDER BY m.priority ASC, d.popularity DESC",
            placeholders, placeholders
        );

        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&sql)?;

        // Bind texts twice — once for kanji IN, once for reading IN
        let params: Vec<&dyn rusqlite::ToSql> = texts
            .iter()
            .map(|t| t as &dyn rusqlite::ToSql)
            .chain(texts.iter().map(|t| t as &dyn rusqlite::ToSql))
            .collect();

        let entries = stmt
            .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                Ok(DictionaryEntry {
                    id: row.get(0)?,
                    kanji: row.get(1)?,
                    reading: row.get(2)?,
                    definitions: row.get(3)?,
                    rules: row.get(4)?,
                    popularity: row.get(5)?,
                    dictionary_id: row.get(6)?,
                    onyomi: row.get(7)?,
                    kunyomi: row.get(8)?,
                    jlpt: row.get(9)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(entries)
    }

    pub fn get_entry_count(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM dictionary", [], |row| row.get(0))?;
        Ok(count)
    }



    // -------------------------------------------------------------------------
    // Tag CRUD
    // -------------------------------------------------------------------------

    pub fn insert_tags(&self, tags: &[DictionaryTag]) -> Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        for tag in tags {
            tx.execute(
                "INSERT OR REPLACE INTO dictionary_tag
                 (name, category, order_val, notes, popularity, dictionary_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    &tag.name,
                    &tag.category,
                    tag.order,
                    &tag.notes,
                    tag.popularity,
                    tag.dictionary_id,
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }


}

/// Whether `name` is the stable title `family` itself or a bracketed revision
/// of it (`family [2026-08-11]`) — the catalog's "already installed" match,
/// which does not care which upstream revision a row carries. Mirrors the
/// family rule mobile's `InstalledDictionary` uses.
///
/// The comparison is done in Rust rather than SQL `LIKE`: `_` and `%` in a
/// title family would otherwise be wildcard characters.
pub fn name_matches_family(name: &str, family: &str) -> bool {
    name == family
        || name
            .strip_prefix(family)
            .is_some_and(|rest| rest.starts_with(" ["))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db(name: &str) -> DictionaryDatabase {
        let dir = std::env::temp_dir().join(format!("ijd_db_{}_{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        DictionaryDatabase::open(dir.join("d.db")).unwrap()
    }

    #[test]
    fn lookup_skips_disabled_dictionaries() {
        let db = temp_db("enabled");
        let on = db.insert_dictionary("On", 0).unwrap();
        let off = db.insert_dictionary("Off", 1).unwrap();
        db.insert_entries(&[
            DictionaryEntry::new(
                "分".into(), "ぶん".into(), r#"["on"]"#.into(), String::new(), 0, on,
            ),
            DictionaryEntry::new(
                "分".into(), "ぶん".into(), r#"["off"]"#.into(), String::new(), 0, off,
            ),
        ])
        .unwrap();
        assert_eq!(db.find_by_texts(&["分".to_string()]).unwrap().len(), 2);
        db.set_dictionary_enabled(off, false).unwrap();
        let rows = db.find_by_texts(&["分".to_string()]).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].definitions.contains("on"));
        db.set_dictionary_enabled(off, true).unwrap();
        assert_eq!(db.find_by_texts(&["分".to_string()]).unwrap().len(), 2);
    }

    #[test]
    fn dictionary_names_maps_ids_to_titles() {
        let db = temp_db("names");
        let id = db.insert_dictionary("Jitendex.org", 0).unwrap();
        let names = db.dictionary_names().unwrap();
        assert_eq!(names.get(&id).map(String::as_str), Some("Jitendex.org"));
    }

    #[test]
    fn old_schema_gains_catalog_id_column() {
        let dir = std::env::temp_dir().join(format!("ijd_db_migrate_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.db");
        {
            // A pre-catalog database: dictionary_meta has only the original
            // four columns, plus a row that must survive the upgrade.
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE dictionary_meta (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT NOT NULL,
                     priority INTEGER NOT NULL DEFAULT 0,
                     enabled INTEGER NOT NULL DEFAULT 1
                 );
                 INSERT INTO dictionary_meta (name, priority) VALUES ('KANJIDIC [old]', 0);",
            )
            .unwrap();
        }

        let db = DictionaryDatabase::open(&path).unwrap();
        let dicts = db.get_all_dictionaries().unwrap();
        assert_eq!(dicts.len(), 1, "migration must not touch existing rows");
        assert_eq!(dicts[0].name, "KANJIDIC [old]");
        assert_eq!(dicts[0].catalog_id, None);
    }

    #[test]
    fn family_match_accepts_exact_title_and_bracketed_revision() {
        assert!(name_matches_family("Jitendex.org [2026-08-11]", "Jitendex.org"));
        assert!(name_matches_family("Jitendex.org", "Jitendex.org"));
        assert!(name_matches_family("KANJIDIC", "KANJIDIC"));
        assert!(name_matches_family("KANJIDIC [2026-258]", "KANJIDIC"));

        // A different family, and a longer title that only shares the prefix,
        // are not revisions of "Jitendex.org".
        assert!(!name_matches_family("Jitendex", "Jitendex.org"));
        assert!(!name_matches_family("Jitendex.org Extra", "Jitendex.org"));
        assert!(!name_matches_family("Jitendex.org[old]", "Jitendex.org"));
    }
}
