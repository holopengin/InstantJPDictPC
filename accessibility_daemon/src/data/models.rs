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

