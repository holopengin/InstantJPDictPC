//! The download catalog (#71): one-click Yomitan zips for the settings window.
//!
//! The pinned entries and the rules for resolving an installed dictionary live
//! here, in the UI-free core.  The same source is consumed by the desktop and
//! by the Android UniFFI shim, so the catalog cannot silently drift between
//! the two clients.  The file is embedded at build time — it is small and the
//! settings window needs it even when the assets directory is not next to the
//! binary.

use serde::Deserialize;
use std::collections::HashSet;
use std::sync::OnceLock;

use crate::data::models::DictionaryMeta;

/// Schema version understood by the catalog reader.
pub const SCHEMA: i64 = 1;

/// One downloadable dictionary, as pinned in `assets/catalog/dictionaries.json`.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct CatalogEntry {
    /// Stable id, also written to `dictionary_meta.catalog_id`.
    pub id: String,
    /// Display name.
    pub name: String,
    /// What the dictionary is, for the row's tooltip-ish subtitle.
    pub description: String,
    /// Dated release URL (never `/releases/latest/`).
    pub url: String,
    /// Exact size the download must have.
    pub bytes: u64,
    /// Lowercase hex SHA-256 the download must have.
    pub sha256: String,
    /// Stable title family the installed `dictionary_meta.name` is matched
    /// against: the zip's declared title may carry a revision suffix
    /// (`Jitendex.org [2026-08-11]`, `KANJIDIC [2026-258]`).
    pub title: String,
    /// Whether the catalog recommends this entry to new users.
    #[serde(default)]
    pub recommended: bool,
    /// Licence label (the full texts ship in `assets/licenses/`).
    pub license: String,
    /// Upstream pin, for provenance in logs.
    pub source: String,
}

#[derive(Deserialize)]
struct CatalogFile {
    schema: i64,
    entries: Vec<CatalogEntry>,
}

impl CatalogFile {
    fn validate(&self) -> Result<(), String> {
        if self.schema != SCHEMA {
            return Err(format!(
                "dictionary catalog: schema {} is not the supported {SCHEMA}",
                self.schema
            ));
        }
        if self.entries.is_empty() {
            return Err("dictionary catalog: lists no dictionaries".to_string());
        }

        let mut seen_ids = HashSet::with_capacity(self.entries.len());
        for (index, entry) in self.entries.iter().enumerate() {
            if !seen_ids.insert(entry.id.as_str()) {
                return Err(format!(
                    "dictionary catalog: duplicate id `{}` at entry {index}",
                    entry.id
                ));
            }
            if entry.bytes == 0 {
                return Err(format!(
                    "dictionary catalog: entry {index} (`{}`) must have positive bytes",
                    entry.id
                ));
            }
            if !is_lowercase_sha256(&entry.sha256) {
                return Err(format!(
                    "dictionary catalog: entry {index} (`{}`) has an invalid SHA-256: {}",
                    entry.id, entry.sha256
                ));
            }
            if !entry.url.starts_with("https://") {
                return Err(format!(
                    "dictionary catalog: entry {index} (`{}`) URL is not https: {}",
                    entry.id, entry.url
                ));
            }
            if entry.url.contains("/releases/latest/") {
                return Err(format!(
                    "dictionary catalog: entry {index} (`{}`) points at /releases/latest/: {}",
                    entry.id, entry.url
                ));
            }
        }
        Ok(())
    }
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Parse and strictly validate a catalog document.
///
/// The embedded catalog is a build-time input, so [`entries`] turns an error
/// into a panic.  Keeping this function public also lets the Android boundary
/// validate its shipped fixture with the exact same rules instead of keeping a
/// second parser there.
pub fn parse_catalog(json: &str) -> Result<Vec<CatalogEntry>, String> {
    let file: CatalogFile = serde_json::from_str(json)
        .map_err(|error| format!("dictionary catalog: invalid JSON: {error}"))?;
    file.validate()?;
    Ok(file.entries)
}

/// The catalog, parsed and validated once. A malformed embedded file is a
/// build-time mistake, so the test below pins it rather than degrading at
/// runtime.
pub fn entries() -> &'static [CatalogEntry] {
    static ENTRIES: OnceLock<Vec<CatalogEntry>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        parse_catalog(include_str!("../../../assets/catalog/dictionaries.json"))
            .expect("embedded dictionary catalog must parse and validate")
    })
}

/// The stable title of an imported dictionary, with any bracketed revision
/// stripped.  This is the same normalization as mobile's `baseTitle`:
/// `JMdict [2026-09-15]` becomes `JMdict`, while surrounding whitespace is
/// ignored.
pub fn base_title(name: &str) -> String {
    let trimmed = name.trim();
    match trimmed.find(" [") {
        Some(index) if index > 0 => trimmed[..index].trim().to_string(),
        _ => trimmed.to_string(),
    }
}

/// Whether `name` belongs to the stable title `family`, including a bracketed
/// revision suffix (`family [2026-08-11]`).
///
/// The comparison is done in Rust rather than SQL `LIKE`: `_` and `%` in a
/// title family would otherwise be wildcard characters.
pub fn name_matches_family(name: &str, family: &str) -> bool {
    base_title(name) == family.trim()
}

/// Resolve installed dictionaries to catalog ids using the same rules as
/// mobile's `installedIds`: an exact catalog id wins; otherwise the first
/// entry whose title family matches the installed name is selected.  The
/// first-entry rule makes mutually exclusive title variants (the two JMdict
/// builds) resolve to one default when a title-only install has no id.
pub fn installed_ids(
    installed_names: &[String],
    installed_catalog_ids: &[Option<String>],
) -> Vec<String> {
    let entries = entries();
    let mut ids = Vec::new();
    let mut seen = HashSet::new();

    for (name, catalog_id) in installed_names.iter().zip(installed_catalog_ids) {
        let matched = catalog_id
            .as_deref()
            .and_then(|id| entries.iter().find(|entry| entry.id == id))
            .or_else(|| {
                entries
                    .iter()
                    .find(|entry| name_matches_family(name, &entry.title))
            });
        if let Some(entry) = matched {
            if seen.insert(entry.id.clone()) {
                ids.push(entry.id.clone());
            }
        }
    }
    ids
}

/// Whether `entry` is already installed: a row whose stable title family
/// matches, or whose `catalog_id` is this entry's id (which is how a
/// previously bundled or catalog-installed copy is recognised). Mirrors
/// mobile's `InstalledDictionary(name, catalogId)` match.
///
/// This is deliberately ungated: it is pure string/row matching and is part of
/// the shared catalog surface consumed by nav-only clients.
pub fn is_installed(dicts: &[DictionaryMeta], entry: &CatalogEntry) -> bool {
    let names: Vec<String> = dicts.iter().map(|dict| dict.name.clone()).collect();
    let catalog_ids: Vec<Option<String>> =
        dicts.iter().map(|dict| dict.catalog_id.clone()).collect();
    installed_ids(&names, &catalog_ids)
        .iter()
        .any(|installed_id| installed_id == &entry.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    fn row(id: &str, title: &str, url: &str, sha256: &str, bytes: u64) -> String {
        format!(
            r#"{{"id":"{id}","name":"{title}","description":"d","license":"l","source":"s","title":"{title}","bytes":{bytes},"sha256":"{sha256}","url":"{url}","recommended":true}}"#
        )
    }

    /// The embedded catalog is the canonical four-entry Android catalog and
    /// carries well-formed pins.
    #[test]
    fn the_catalog_pins_the_four_canonical_dictionaries() {
        let entries = entries();
        assert_eq!(entries.len(), 4);
        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "jitendex",
                "kanjidic-english",
                "jmdict-english",
                "jmdict-english-with-examples",
            ]
        );
        assert_eq!(
            entries
                .iter()
                .filter(|entry| entry.recommended)
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            vec!["jitendex", "kanjidic-english"],
        );
        for entry in entries {
            assert!(!entry.title.is_empty(), "{} has a title", entry.name);
            assert!(entry.url.starts_with("https://"), "{}", entry.url);
            assert!(!entry.url.contains("/releases/latest/"), "{}", entry.url);
            assert!(entry.bytes > 0, "{} has a size pin", entry.name);
            assert!(is_lowercase_sha256(&entry.sha256), "{}", entry.name);
        }
    }

    #[test]
    fn strict_parser_rejects_invalid_catalog_documents() {
        let valid = row("x", "X", "https://example.com/a.zip", VALID_SHA, 10);
        assert_eq!(
            parse_catalog(&format!(r#"{{"schema":1,"entries":[{valid}]}}"#))
                .unwrap()
                .len(),
            1
        );

        let cases = [
            r#"{"schema":2,"entries":[]}"#.to_string(),
            format!(r#"{{"entries":[{valid}]}}"#),
            format!(
                r#"{{"schema":1,"entries":[{},{}]}}"#,
                row("x", "X", "https://example.com/a.zip", VALID_SHA, 10),
                row("x", "X", "https://example.com/a.zip", VALID_SHA, 10),
            ),
            format!(
                r#"{{"schema":1,"entries":[{}]}}"#,
                row("x", "X", "https://example.com/a.zip", VALID_SHA, 0),
            ),
            format!(
                r#"{{"schema":1,"entries":[{}]}}"#,
                row(
                    "x",
                    "X",
                    "https://example.com/a.zip",
                    "A".repeat(64).as_str(),
                    10
                ),
            ),
            format!(
                r#"{{"schema":1,"entries":[{}]}}"#,
                row("x", "X", "http://example.com/a.zip", VALID_SHA, 10),
            ),
            format!(
                r#"{{"schema":1,"entries":[{}]}}"#,
                row(
                    "x",
                    "X",
                    "https://example.com/releases/latest/download/a.zip",
                    VALID_SHA,
                    10,
                ),
            ),
        ];
        for case in cases {
            assert!(
                parse_catalog(&case).is_err(),
                "accepted invalid catalog: {case}"
            );
        }
    }

    #[test]
    fn missing_required_fields_and_wrong_recommended_type_fail() {
        let missing_title = r#"{"schema":1,"entries":[{"id":"x","name":"X","description":"d","license":"l","source":"s","bytes":1,"sha256":"0000000000000000000000000000000000000000000000000000000000000000","url":"https://example.com/a.zip"}]}"#;
        assert!(parse_catalog(missing_title).is_err());

        let wrong_recommended = r#"{"schema":1,"entries":[{"id":"x","name":"X","description":"d","license":"l","source":"s","title":"X","bytes":1,"sha256":"0000000000000000000000000000000000000000000000000000000000000000","url":"https://example.com/a.zip","recommended":"yes"}]}"#;
        assert!(parse_catalog(wrong_recommended).is_err());
    }

    #[test]
    fn title_matching_and_installed_ids_follow_the_catalog_rules() {
        assert_eq!(base_title("  JMdict [2026-09-15]  "), "JMdict");
        assert_eq!(base_title("JMdict Forms"), "JMdict Forms");
        assert!(name_matches_family(
            "Jitendex.org [2026-08-11]",
            "Jitendex.org"
        ));
        assert!(name_matches_family("Jitendex.org", "Jitendex.org"));
        assert!(!name_matches_family("Jitendex.org Extra", "Jitendex.org"));
        assert!(!name_matches_family("Jitendex.org[old]", "Jitendex.org"));

        let names = vec![
            "JMdict [2026-09-15]".to_string(),
            "KANJIDIC [2026-258]".to_string(),
        ];
        let ids = vec![None, Some("kanjidic-english".to_string())];
        assert_eq!(
            installed_ids(&names, &ids),
            vec!["jmdict-english", "kanjidic-english"],
        );

        let meta = |name: &str, catalog_id: Option<&str>| DictionaryMeta {
            id: 1,
            name: name.to_string(),
            priority: 0,
            enabled: true,
            built_in: false,
            catalog_id: catalog_id.map(str::to_string),
        };
        let jmdict = entries()
            .iter()
            .find(|entry| entry.id == "jmdict-english-with-examples")
            .unwrap();
        let jitendex = entries()
            .iter()
            .find(|entry| entry.id == "jitendex")
            .unwrap();
        assert!(is_installed(
            &[meta("Jitendex.org [2026-08-11]", None)],
            jitendex,
        ));
        assert!(is_installed(
            &[meta("Jitendex", Some("jitendex"))],
            jitendex,
        ));
        assert!(!is_installed(&[meta("Jitendex Extra", None)], jitendex));
        assert!(is_installed(
            &[meta(
                "JMdict [2026-09-15]",
                Some("jmdict-english-with-examples")
            )],
            jmdict,
        ));
        assert!(!is_installed(&[meta("JMdict [2026-09-15]", None)], jmdict));
    }
}
