//! The download catalog (#71): one-click Yomitan zips for the settings window.
//!
//! Mirrors mobile's `assets/catalog/dictionaries.json`: every entry is pinned
//! to a dated upstream release URL plus a byte size and SHA-256, so an install
//! is integrity-checked before a single row reaches the database. The catalog
//! is embedded at build time — it is small and the settings window needs it
//! even when the assets directory is not next to the binary.

use serde::Deserialize;
use std::sync::OnceLock;

/// One downloadable dictionary, as pinned in `assets/catalog/dictionaries.json`.
#[derive(Clone, Debug, Deserialize)]
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
    pub family: String,
    /// Licence label (the full texts ship in `assets/licenses/`).
    pub license: String,
    /// Upstream pin, for provenance in logs.
    pub source: String,
}

#[derive(Deserialize)]
struct CatalogFile {
    entries: Vec<CatalogEntry>,
}

/// The catalog, parsed once. A malformed embedded file is a build-time
/// mistake, so the test below pins it rather than degrading at runtime.
pub fn entries() -> &'static [CatalogEntry] {
    static ENTRIES: OnceLock<Vec<CatalogEntry>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        let file: CatalogFile = serde_json::from_str(include_str!(
            "../../../assets/catalog/dictionaries.json"
        ))
        .expect("embedded dictionary catalog must parse");
        file.entries
    })
}

/// Whether `entry` is already installed: a row whose stable title family
/// matches, or whose `catalog_id` is this entry's id (which is how a
/// previously bundled or catalog-installed copy is recognised). Mirrors
/// mobile's `InstalledDictionary(name, catalogId)` match.
///
/// Requires the `db` feature (it matches against `db` title families).
#[cfg(feature = "db")]
pub fn is_installed(dicts: &[crate::data::models::DictionaryMeta], entry: &CatalogEntry) -> bool {
    dicts.iter().any(|d| {
        d.catalog_id.as_deref() == Some(entry.id.as_str())
            || crate::data::db::name_matches_family(&d.name, &entry.family)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded catalog parses and carries the two rows mobile recommends,
    /// with well-formed pins.
    #[test]
    fn the_catalog_pins_the_two_recommended_dictionaries() {
        let entries = entries();
        assert_eq!(entries.len(), 2, "Jitendex + KANJIDIC");
        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, vec!["jitendex", "kanjidic-english"]);
        for e in entries {
            assert!(e.url.starts_with("https://"), "{}", e.url);
            assert!(!e.url.contains("/releases/latest/"), "pinned URL: {}", e.url);
            assert!(e.bytes > 0, "{} has a size pin", e.name);
            assert_eq!(e.sha256.len(), 64, "{} has a SHA-256 pin", e.name);
            assert!(
                e.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{} SHA-256 is lowercase hex",
                e.name
            );
            assert!(!e.family.is_empty());
        }
    }

    /// Installed-state matching: family with a revision suffix, exact family,
    /// or the catalog id on a differently named row.
    #[cfg(feature = "db")]
    #[test]
    fn installed_state_matches_family_and_catalog_id() {
        use crate::data::models::DictionaryMeta;
        let jitendex = &entries()[0];
        let meta = |name: &str, catalog_id: Option<&str>| DictionaryMeta {
            id: 1,
            name: name.to_string(),
            priority: 0,
            enabled: true,
            built_in: false,
            catalog_id: catalog_id.map(str::to_string),
        };
        assert!(!is_installed(&[], jitendex));
        assert!(is_installed(
            &[meta("Jitendex.org [2026-08-11]", None)],
            jitendex
        ));
        assert!(is_installed(&[meta("Jitendex.org", None)], jitendex));
        assert!(is_installed(&[meta("Jitendex", Some("jitendex"))], jitendex));
        assert!(!is_installed(&[meta("Jitendex Extra", None)], jitendex));
        assert!(!is_installed(&[meta("JMdict", None)], jitendex));
    }
}
