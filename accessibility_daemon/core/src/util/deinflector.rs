//! Japanese deinflection engine.
//! Mirrors `Deinflector` from the Kotlin implementation.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;

/// A single deinflection rule.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DeinflectionRule {
    #[serde(rename = "kanaIn")]
    pub kana_in: String,
    #[serde(rename = "kanaOut")]
    pub kana_out: String,
    #[serde(rename = "rulesOut")]
    pub rules_out: Vec<String>,
    /// Top-level deinflect.json group key (e.g. "past").
    /// Not in the JSON payloads — filled in at load time from the map key,
    /// mirroring mobile's `Deinflector` (`reason = reason` copy on load).
    #[serde(default)]
    pub reason: String,
}

/// Result of a deinflection: a candidate term with its grammatical type.
#[derive(Debug, Clone, PartialEq)]
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
        Self::from_json_str(&content)
    }

    /// Load deinflection rules from JSON text.
    ///
    /// The parse half of [`Deinflector::from_json_file`], split out so a
    /// binding host that has the asset bytes (an APK asset is not a filesystem
    /// path) can pass them across without a temp file. Identical behaviour to
    /// the file loader.
    pub fn from_json_str(content: &str) -> Result<Self> {
        // The deinflection rules file may be a JSON array or an object keyed by category.
        // Try array first, then object.
        let rules: Vec<DeinflectionRule> = if content.trim().starts_with('[') {
            serde_json::from_str(content).context("Failed to parse deinflection rules JSON array")?
        } else {
            let map: HashMap<String, Vec<DeinflectionRule>> = serde_json::from_str(content)
                .context("Failed to parse deinflection rules JSON object")?;
            // The group key IS the human-readable reason ("past",
            // "causative", …) — mobile fills it in at load time the same way.
            map.into_iter()
                .flat_map(|(reason, list)| {
                    list.into_iter().map(move |mut rule| {
                        rule.reason = reason.clone();
                        rule
                    })
                })
                .collect()
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
                                // Human-readable group label ("past"), not the
                                // kana fragment — mobile pushes `rule.reason`.
                                // An empty reason (bare-array JSON with no
                                // group keys) adds nothing, so the identity
                                // guard in `push_deinflections` still filters it.
                                if !rule.reason.is_empty() {
                                    r.push(rule.reason.clone());
                                }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Mobile parity (`DeinflectionChainTest.reasons_carryRuleNames`): reasons
    /// are readable group labels, not kana fragments.
    #[test]
    fn reasons_carry_rule_names() {
        let deinflector = Deinflector::from_json_file(concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/deinflect.json")).unwrap();
        let hit = deinflector
            .deinflect("食べた")
            .into_iter()
            .find(|r| r.term == "食べる")
            .expect("a deinflected 食べる candidate");
        assert_eq!(hit.reasons, vec!["past"]);
    }

    /// Mobile parity (`DeinflectionChainTest.identityResult_hasNoReasons`):
    /// the identity result carries no reasons, so direct matches get no chain.
    #[test]
    fn identity_result_has_no_reasons() {
        let deinflector = Deinflector::from_json_file(concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/deinflect.json")).unwrap();
        let identity = deinflector
            .deinflect("食べた")
            .into_iter()
            .find(|r| r.term == "食べた")
            .expect("the identity candidate");
        assert!(identity.reasons.is_empty());
    }

    /// The binding split (ticket pipeline-sharing/06): parsing from text is
    /// the same load as parsing from the file, so a host that passes asset
    /// bytes across the FFI gets the desktop behaviour. Rule order is
    /// compared sorted: the object-keyed JSON loads through a `HashMap`, so
    /// iteration order is not stable across loads.
    #[test]
    fn from_json_str_matches_from_json_file() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/deinflect.json");
        let from_file = Deinflector::from_json_file(path).expect("file load");
        let from_str = Deinflector::from_json_str(include_str!("../../../assets/deinflect.json"))
            .expect("string load");

        let mut file_rules = from_file.rules.clone();
        let mut str_rules = from_str.rules.clone();
        let key = |r: &DeinflectionRule| (r.kana_in.clone(), r.kana_out.clone(), r.reason.clone());
        file_rules.sort_by_key(key);
        str_rules.sort_by_key(key);
        assert_eq!(file_rules, str_rules);
        assert_eq!(from_file.rule_count(), from_str.rule_count());
    }
}
