//! Japanese deinflection engine.
//! Mirrors `Deinflector` from the Kotlin implementation.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;

/// A single deinflection rule.
#[derive(Debug, Clone, Deserialize)]
pub struct DeinflectionRule {
    #[serde(rename = "kanaIn")]
    pub kana_in: String,
    #[serde(rename = "kanaOut")]
    pub kana_out: String,
    #[serde(rename = "rulesOut")]
    pub rules_out: Vec<String>,
}

/// Result of a deinflection: a candidate term with its grammatical type.
#[derive(Debug, Clone)]
pub struct DeinflectionResult {
    pub term: String,
    pub reasons: Vec<String>,
    pub rule_types: Vec<String>,
}

/// Deinflects Japanese text by applying known conjugation rules.
pub struct Deinflector {
    rules: Vec<DeinflectionRule>,
}

impl Deinflector {
    /// Load deinflection rules from a JSON file.
    /// The JSON format is: `[{ "kanaIn": "...", "kanaOut": "...", "rulesIn": [...], "rulesOut": [...] }, ...]`
    pub fn from_json_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read deinflection rules: {:?}", path.as_ref()))?;

        // The deinflection rules file may be a JSON array or an object keyed by category.
        // Try array first, then object.
        let rules: Vec<DeinflectionRule> = if content.trim().starts_with('[') {
            serde_json::from_str(&content).context("Failed to parse deinflection rules JSON array")?
        } else {
            let map: HashMap<String, Vec<DeinflectionRule>> = serde_json::from_str(&content)
                .context("Failed to parse deinflection rules JSON object")?;
            map.into_values().flatten().collect()
        };

        println!("Loaded {} deinflection rules", rules.len());
        Ok(Self { rules })
    }


    /// Create an empty deinflector (no rules).
    pub fn empty() -> Self {
        Self { rules: Vec::new() }
    }

    /// Deinflect the given text, returning all possible base forms.
    /// The first result is always the original text with no deinflection.
    pub fn deinflect(&self, text: &str) -> Vec<DeinflectionResult> {
        let mut results = Vec::new();
        results.push(DeinflectionResult {
            term: text.to_string(),
            reasons: Vec::new(),
            rule_types: Vec::new(),
        });

        let mut i = 0;
        while i < results.len() {
            let current = results[i].clone();
            if current.term.len() < 2 {
                i += 1;
                continue;
            }

            for rule in &self.rules {
                if current.term.ends_with(&rule.kana_in) {
                    let root = format!(
                        "{}{}",
                        &current.term[..current.term.len() - rule.kana_in.len()],
                        rule.kana_out
                    );

                    if !root.is_empty() {
                        let new_result = DeinflectionResult {
                            term: root,
                            reasons: {
                                let mut r = current.reasons.clone();
                                r.push(rule.kana_in.clone());
                                r
                            },
                            rule_types: rule.rules_out.clone(),
                        };

                        if !results.iter().any(|r| r.term == new_result.term) {
                            results.push(new_result);
                        }
                    }
                }
            }
            i += 1;
        }

        results
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}
