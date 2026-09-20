//! Database schema models for dictionary storage.
//! Mirrors the Room entity classes from the Kotlin implementation.

use serde::{Deserialize, Serialize};

/// A single dictionary entry (term or kanji).
/// Equivalent to `DictionaryEntry` entity in Kotlin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryEntry {
    pub id: i64,
    pub kanji: String,
    pub reading: String,
    pub definitions: String, // JSON string
    pub rules: String,
    pub popularity: i32,
    pub dictionary_id: i64,
    pub onyomi: Option<String>,
    pub kunyomi: Option<String>,
    pub jlpt: Option<String>,
}

/// Metadata for an imported dictionary.
/// Equivalent to `DictionaryMeta` entity in Kotlin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryMeta {
    pub id: i64,
    pub name: String,
    pub priority: i32,
    pub enabled: bool,
    /// True for dictionaries bundled with the app (the vendored zips in
    /// `assets/dictionaries/`). Mirrors Android's `builtIn`: it is written
    /// only after every bank of a bundled import has landed, so it doubles as
    /// the completion marker, and the settings manager uses it to tell
    /// app-owned dictionaries from user ones.
    pub built_in: bool,
    /// Catalog entry this dictionary was installed from, when it came through
    /// the catalog — else `None` (file picker, bundled install). Mirrors
    /// Android's `catalogId`. Bundled installs record their own stable id so a
    /// half-finished install can be told apart from a user-imported copy.
    pub catalog_id: Option<String>,
}

/// A tag from a dictionary tag bank.
/// Equivalent to `DictionaryTag` entity in Kotlin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictionaryTag {
    pub id: i64,
    pub name: String,
    pub category: String,
    pub order: i32,
    pub notes: String,
    pub popularity: i32,
    pub dictionary_id: i64,
}

impl DictionaryEntry {
    pub fn new(
        kanji: String,
        reading: String,
        definitions: String,
        rules: String,
        popularity: i32,
        dictionary_id: i64,
    ) -> Self {
        Self {
            id: 0,
            kanji,
            reading,
            definitions,
            rules,
            popularity,
            dictionary_id,
            onyomi: None,
            kunyomi: None,
            jlpt: None,
        }
    }
}

