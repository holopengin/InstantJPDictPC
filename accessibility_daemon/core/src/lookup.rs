//! Database-bound dictionary lookup.
//!
//! The pure query preparation, row grouping, and formatting stages live in the
//! ungated [`crate::lookup_core`] module. This file keeps only the SQLite query
//! and the bounded redirect traversal, then delegates the owned result to the
//! desktop overlay state.

use std::collections::{HashMap, HashSet};

use crate::data::db::DictionaryDatabase;
use crate::data::models::DictionaryEntry;
use crate::models::{DeinflectionChain, TermMatch};
use crate::overlay_state::OcrOverlayState;
use crate::util::deinflector::Deinflector;

use crate::lookup_core::{expand_max_len, prepare_search_candidates as prepare_candidates};
pub use crate::lookup_core::{
    following_text, format_dictionary_results, process_results, LookupOutcome,
};

/// Legacy tuple-shaped facade retained for the desktop overlay module's
/// existing call sites. The algorithm and candidate records live in
/// [`crate::lookup_core`]; this only adapts the owned result.
pub fn prepare_search_candidates(
    following_text: &str,
    deinflector: &Deinflector,
) -> (
    HashSet<String>,
    Vec<(
        usize,
        Vec<(String, Option<Vec<String>>, Option<DeinflectionChain>)>,
    )>,
) {
    let prepared = prepare_candidates(following_text, deinflector);
    let by_length = prepared
        .by_length
        .into_iter()
        .map(|group| {
            (
                group.length,
                group
                    .candidates
                    .into_iter()
                    .map(|candidate| (candidate.term, candidate.required_types, candidate.chain))
                    .collect(),
            )
        })
        .collect();
    (prepared.terms, by_length)
}

/// Look up an already-extracted `following_text`, returning `None` for an empty
/// query or a search with no matches.
pub fn lookup_term(
    active_all_chars: &[String],
    global_idx: usize,
    following_text: &str,
    db: &DictionaryDatabase,
    deinflector: &Deinflector,
) -> Option<LookupOutcome> {
    if following_text.is_empty() {
        return None;
    }

    let prepared = prepare_candidates(following_text, deinflector);
    if prepared.terms.is_empty() {
        return None;
    }

    let all_terms: Vec<String> = prepared.terms.iter().cloned().collect();
    let db_results = db.find_by_texts(&all_terms).unwrap_or_default();
    let dict_names = db.dictionary_names().unwrap_or_default();
    let processed = process_results(&db_results, &prepared, following_text);

    // The highlight includes placeholder cells removed from following_text.
    let max_len = expand_max_len(active_all_chars, global_idx, processed.max_len);
    if processed.matches.is_empty() {
        return None;
    }

    // ── Redirect pass (#65): resolve pointer-only rows breadth-first. The
    // visited set and hop cap make cycles and redirect graphs finite.
    let mut redirect_via: HashMap<String, String> = HashMap::new();
    let mut resolved_by_term: HashMap<String, TermMatch> = HashMap::new();
    let mut children_of: HashMap<String, Vec<String>> = HashMap::new();
    {
        let mut visited: HashSet<String> = processed
            .matches
            .iter()
            .map(|term| term.term.clone())
            .collect();
        let mut queue: std::collections::VecDeque<TermMatch> =
            processed.matches.iter().cloned().collect();
        let mut hops = 0;
        while !queue.is_empty() && hops < 3 {
            for _ in 0..queue.len() {
                let Some(term_match) = queue.pop_front() else {
                    break;
                };
                for entry in &term_match.entries {
                    for target in OcrOverlayState::extract_redirect_targets(&entry.definitions, 3) {
                        if !visited.insert(target.clone()) {
                            continue;
                        }
                        let target_results =
                            db.find_by_texts(&[target.clone()]).unwrap_or_default();
                        if target_results.is_empty() {
                            continue;
                        }
                        let mut seen = HashSet::new();
                        let entries: Vec<DictionaryEntry> = target_results
                            .into_iter()
                            .filter(|entry| seen.insert(entry.id))
                            .collect();
                        redirect_via.insert(target.clone(), term_match.term.clone());
                        let resolved = TermMatch {
                            term: target.clone(),
                            entries,
                            chain: None,
                        };
                        resolved_by_term.insert(target.clone(), resolved.clone());
                        children_of
                            .entry(term_match.term.clone())
                            .or_default()
                            .push(target.clone());
                        queue.push_back(resolved);
                    }
                }
            }
            hops += 1;
        }
    }

    fn emit(
        term: &str,
        originals: &[TermMatch],
        resolved: &HashMap<String, TermMatch>,
        children: &HashMap<String, Vec<String>>,
        out: &mut Vec<TermMatch>,
    ) {
        if let Some(term_match) = originals.iter().find(|item| item.term == term) {
            out.push(term_match.clone());
        } else if let Some(term_match) = resolved.get(term) {
            out.push(term_match.clone());
        }
        if let Some(child_terms) = children.get(term) {
            for child in child_terms {
                emit(child, originals, resolved, children, out);
            }
        }
    }

    let mut resolved_matches = Vec::new();
    for term_match in &processed.matches {
        emit(
            &term_match.term,
            &processed.matches,
            &resolved_by_term,
            &children_of,
            &mut resolved_matches,
        );
    }

    let mut formatted = format_dictionary_results(&resolved_matches, &dict_names);
    for entry in &mut formatted {
        if entry.deinflection.is_none() {
            if let Some(via) = redirect_via.get(&entry.term) {
                entry.deinflection = Some(DeinflectionChain {
                    surface: via.clone(),
                    steps: vec!["redirect".to_string()],
                });
            }
        }
    }

    Some(LookupOutcome {
        matches: formatted,
        max_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bound lookup crosses a worker thread with its database, deinflector,
    /// and owned output. This is the integration pin for the ungated seam.
    #[test]
    fn lookup_term_runs_on_a_worker_thread() {
        let directory =
            std::env::temp_dir().join(format!("ijd_lookup_send_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let db = DictionaryDatabase::open(directory.join("d.db")).unwrap();
        let dictionary_id = db.insert_dictionary("JMdict", 0).unwrap();
        db.insert_entries(&[DictionaryEntry::new(
            "食べる".into(),
            "たべる".into(),
            r#"["to eat"]"#.into(),
            String::new(),
            0,
            dictionary_id,
        )])
        .unwrap();

        let active = ["食", "べ", "る"]
            .iter()
            .map(|character| character.to_string())
            .collect::<Vec<_>>();
        let following = following_text(&active, 0);
        let outcome = std::thread::spawn(move || {
            lookup_term(&active, 0, &following, &db, &Deinflector::empty())
        })
        .join()
        .expect("worker")
        .expect("outcome");

        assert!(outcome.max_len >= 1);
        assert!(format!("{:?}", outcome.matches).contains("食べる"));
        let _ = std::fs::remove_dir_all(&directory);
    }
}
