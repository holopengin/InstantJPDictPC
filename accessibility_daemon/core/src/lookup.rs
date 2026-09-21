//! Pure dictionary-lookup pipeline (the binding surface).
//!
//! This module is the `Send` half of the old `OcrOverlayState` lookup: it owns
//! no viewport state, takes plain inputs, and returns owned data. The desktop
//! `overlay_state::OcrOverlayState` keeps its `Rc` cache and `Cell` viewport
//! and delegates here; a Kotlin/Android binding exposes these free functions
//! (see `core/UNIFFI_READINESS.md`) without ever touching the `!Send` state.
//!
//! Data in / data out: active OCR text + database rows in; formatted entries
//! out. No `Rc`, no `Cell`, no interior mutability.

use std::collections::{HashMap, HashSet};

use crate::data::db::DictionaryDatabase;
use crate::data::models::DictionaryEntry;
use crate::models::{
    DeinflectionChain, FormattedEntry, FormattedHeadword, FormattedReadingGroup, FormattedSense,
    FormattedSenseGroup, TermMatch,
};
use crate::overlay_state::OcrOverlayState;
use crate::util::deinflector::Deinflector;
use crate::util::japanese;

/// Owned result of a dictionary lookup.
///
/// The desktop view keeps an `Rc<Vec<FormattedEntry>>` cache for O(1) clones
/// into `view()`; that cache is desktop-only. The pure pipeline returns plain
/// owned data so it can cross an FFI/thread boundary.
pub struct LookupOutcome {
    pub matches: Vec<FormattedEntry>,
    pub max_len: usize,
}

/// The search text for a tap: the next 20 global characters with full-width
/// spaces (the blank placeholder) stripped. Shared by [`lookup_term`] and the
/// desktop `OcrOverlayState::lookup` so the seam has one implementation.
pub fn following_text(active_all_chars: &[String], global_idx: usize) -> String {
    let end_idx = (global_idx + 20).min(active_all_chars.len());
    let text: String = active_all_chars[global_idx..end_idx].join("");
    // Strip full-width spaces (null/void character placeholder) before lookup
    text.chars().filter(|&c| c != '\u{3000}').collect()
}

/// The pure lookup pipeline for an already-extracted `following_text`.
///
/// Returns `None` when there is nothing to search (`following_text` empty or no
/// candidates) or when the search produced no matches; the caller decides how
/// to cache the miss. The desktop method owns the cursor/cache bookkeeping.
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

    // Build search candidates
    let (all_terms, candidates_by_length) = prepare_search_candidates(following_text, deinflector);

    if all_terms.is_empty() {
        return None;
    }

    // Query database
    let all_terms_vec: Vec<String> = all_terms.into_iter().collect();
    let db_results = db.find_by_texts(&all_terms_vec).unwrap_or_default();
    let dict_names = db.dictionary_names().unwrap_or_default();

    // Process results
    let (matches, max_len) =
        process_results(&db_results, &candidates_by_length, &all_terms_vec, following_text);

    // Expand max_len to count U+3000 chars in the original text that were
    // filtered out, so the highlight spans the correct visual range.
    let max_len = expand_max_len(active_all_chars, global_idx, max_len);

    if matches.is_empty() {
        return None;
    }

    // ── Redirect pass (#65): JMdict pointer entries (variant spellings)
    // carry only `?query=` links and render as dead "⟶, X" text. Resolve
    // them breadth-first (visited set + hop cap, so A→B→A cycles always
    // terminate), then splice each target directly below its source entry
    // so a redirect reads as pointer → target.
    let mut redirect_via: HashMap<String, String> = HashMap::new();
    let mut resolved_by_term: HashMap<String, TermMatch> = HashMap::new();
    let mut children_of: HashMap<String, Vec<String>> = HashMap::new();
    {
        let mut visited: HashSet<String> = matches.iter().map(|m| m.term.clone()).collect();
        let mut queue: std::collections::VecDeque<TermMatch> = matches.iter().cloned().collect();
        let mut hops = 0;
        while !queue.is_empty() && hops < 3 {
            for _ in 0..queue.len() {
                let Some(m) = queue.pop_front() else { break };
                for entry in &m.entries {
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
                            .filter(|e| seen.insert(e.id))
                            .collect();
                        redirect_via.insert(target.clone(), m.term.clone());
                        let resolved = TermMatch {
                            term: target.clone(),
                            entries,
                            chain: None,
                        };
                        resolved_by_term.insert(target.clone(), resolved.clone());
                        children_of
                            .entry(m.term.clone())
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
        if let Some(m) = originals.iter().find(|m| m.term == term) {
            out.push(m.clone());
        } else if let Some(m) = resolved.get(term) {
            out.push(m.clone());
        }
        if let Some(kids) = children.get(term) {
            for child in kids {
                emit(child, originals, resolved, children, out);
            }
        }
    }
    let mut resolved_matches: Vec<TermMatch> = Vec::new();
    for m in &matches {
        emit(
            &m.term,
            &matches,
            &resolved_by_term,
            &children_of,
            &mut resolved_matches,
        );
    }

    // Format results
    let mut formatted = format_dictionary_results(&resolved_matches, &dict_names);
    for entry in formatted.iter_mut() {
        // The redirect hop joins the deinflection chain ("via → term
        // · redirect") instead of a separate caption.
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

/// Count U+3000 chars in the original text that were filtered out, so the
/// highlight spans the correct visual range.
fn expand_max_len(active_all_chars: &[String], global_idx: usize, max_len: usize) -> usize {
    let original = &active_all_chars[global_idx..];
    let mut seen_non_space = 0usize;
    let mut expanded = 0usize;
    for ch in original.iter() {
        if seen_non_space >= max_len {
            break;
        }
        expanded += 1;
        if ch != "\u{3000}" {
            seen_non_space += 1;
        }
    }
    expanded
}

/// Prepare search candidates from the following text.
/// Returns (set of all terms to search, candidates grouped by length).
pub fn prepare_search_candidates(
    following_text: &str,
    deinflector: &Deinflector,
) -> (
    HashSet<String>,
    Vec<(usize, Vec<(String, Option<Vec<String>>, Option<DeinflectionChain>)>)>,
) {
    let mut all_terms = HashSet::new();
    let mut candidates_by_length = Vec::new();

    let max_len = following_text.chars().count();
    for len in (1..=max_len).rev() {
        let query_text_raw: String = following_text.chars().take(len).collect();
        let query_text = japanese::normalize(&query_text_raw);

        // #75: pre-reform orthography. The modern form of the prefix is
        // searched as one more variant — the raw prefix stays in the list,
        // so this can only add a reachable headword, never take one away.
        let modernised = japanese::kana_orthography_modernise(&query_text);
        // #81: the historical sound changes JMdict's entry-local variants
        // cannot reach (やう→よう, けふ→きょう, 思ふ→思う). Composed after
        // #75, so きやう → きゃう → きょう; both forms stay in the
        // candidate set.
        let sound_changed = japanese::kana_sound_changes_modernise(&modernised);

        // The RAW prefix is searched alongside its folded form, so an old
        // form's own entries are never lost to the fold.
        let mut variants: Vec<String> = Vec::new();
        for v in [
            query_text_raw.clone(),
            query_text.clone(),
            japanese::katakana_to_hiragana(&query_text),
            japanese::collapse_emphatic(&query_text),
            modernised.clone(),
            japanese::katakana_to_hiragana(&modernised),
            sound_changed.clone(),
            japanese::katakana_to_hiragana(&sound_changed),
        ] {
            if !variants.contains(&v) {
                variants.push(v);
            }
        }

        let deinflections = deinflector.deinflect(&query_text);
        // The deinflection rules are modern orthography; a legacy surface
        // has to be normalised before they can fire at all, so the modern
        // forms are deinflected too — additive, like the variants above.
        let modernised_deinflections = if modernised != query_text {
            deinflector.deinflect(&modernised)
        } else {
            Vec::new()
        };
        let sound_changed_deinflections = if sound_changed != modernised {
            deinflector.deinflect(&sound_changed)
        } else {
            Vec::new()
        };

        let mut length_candidates: Vec<(String, Option<Vec<String>>, Option<DeinflectionChain>)> =
            Vec::new();

        for v in &variants {
            length_candidates.push((v.clone(), None, None));
            all_terms.insert(v.clone());
        }

        let push_deinflections = |deinflections: &[crate::util::deinflector::DeinflectionResult],
                                      same_as: &str,
                                      all_terms: &mut HashSet<String>,
                                      length_candidates: &mut Vec<(String, Option<Vec<String>>, Option<DeinflectionChain>)>| {
            for d in deinflections {
                // Android requires a reason list too, so a no-op
                // deinflection never adds a duplicate candidate.
                if d.term == same_as || d.reasons.is_empty() {
                    continue;
                }
                let types = if d.rule_types.is_empty() {
                    None
                } else {
                    Some(d.rule_types.clone())
                };
                let chain = Some(DeinflectionChain {
                    surface: query_text_raw.clone(),
                    steps: d.reasons.clone(),
                });
                length_candidates.push((d.term.clone(), types, chain));
                all_terms.insert(d.term.clone());
            }
        };
        push_deinflections(
            &deinflections,
            &query_text,
            &mut all_terms,
            &mut length_candidates,
        );
        push_deinflections(
            &modernised_deinflections,
            &modernised,
            &mut all_terms,
            &mut length_candidates,
        );
        push_deinflections(
            &sound_changed_deinflections,
            &sound_changed,
            &mut all_terms,
            &mut length_candidates,
        );

        candidates_by_length.push((len, length_candidates));
    }

    (all_terms, candidates_by_length)
}


/// Process database results to find matching entries.
/// Preserves database result order (which is by dictionary priority ASC, popularity DESC).
pub fn process_results(
    db_results: &[DictionaryEntry],
    candidates_by_length: &[(usize, Vec<(String, Option<Vec<String>>, Option<DeinflectionChain>)>)],
    _all_terms: &[String],
    following_text: &str,
) -> (Vec<TermMatch>, usize) {
    // Use Vec-based grouping to preserve database result order (priority ASC, popularity DESC)
    let mut results_by_term: Vec<(String, Vec<DictionaryEntry>)> = Vec::new();

    for entry in db_results {
        // Add under kanji key
        if let Some(pos) = results_by_term.iter().position(|(k, _)| k == &entry.kanji) {
            results_by_term[pos].1.push(entry.clone());
        } else {
            results_by_term.push((entry.kanji.clone(), vec![entry.clone()]));
        }
        // Add under reading key (if different from kanji)
        if entry.reading != entry.kanji {
            if let Some(pos) = results_by_term.iter().position(|(k, _)| k == &entry.reading) {
                results_by_term[pos].1.push(entry.clone());
            } else {
                results_by_term.push((entry.reading.clone(), vec![entry.clone()]));
            }
        }
    }

    let mut matches: Vec<TermMatch> = Vec::new();
    let mut max_len = 0;

    for (len, candidates) in candidates_by_length {
        let mut found = false;
        for (term, required_types, chain) in candidates {
            let term_entries = match results_by_term.iter().find(|(k, _)| k == term) {
                Some((_, entries)) => entries,
                None => continue,
            };

            let filtered: Vec<DictionaryEntry> = if let Some(types) = required_types {
                // Deinflected match — filter by rule type
                term_entries
                    .iter()
                    .filter(|entry| {
                        let entry_tags: Vec<&str> = entry.rules.split_whitespace().collect();
                        types.is_empty()
                            || types.iter().any(|t: &String| entry_tags.iter().any(|et| *et == t.as_str()))
                            || (entry_tags.iter().any(|t: &&str| t.starts_with("v"))
                                && types.iter().any(|t: &String| t.starts_with("v")))
                    })
                    .cloned()
                    .collect()
            } else {
                // Direct match — filter out kanji entries that don't match the query text
                let query_text = japanese::normalize(
                    &following_text.chars().take(*len).collect::<String>(),
                );
                term_entries
                    .iter()
                    .filter(|entry| {
                        let is_kanji_entry = entry.onyomi.is_some() || entry.kunyomi.is_some();
                        !is_kanji_entry || entry.kanji == query_text
                    })
                    .cloned()
                    .collect()
            };

            if !filtered.is_empty() {
                matches.push(TermMatch {
                    term: term.clone(),
                    entries: filtered,
                    chain: chain.clone(),
                });
                found = true;
            }
        }
        if found && max_len == 0 {
            max_len = *len;
        }
    }

    // Deduplicate by term
    let mut seen = HashSet::new();
    matches.retain(|m| seen.insert(m.term.clone()));

    (matches, max_len)
}


/// Format dictionary results into displayable entries.
///
/// Mirrors `formatDictionaryResults`: one entry per (term, dictionary)
/// so JMdict and KANJIDIC rows never merge, a reading that repeats an
/// earlier glossary keeps its headword but not its senses, and Jitendex
/// rows — which pack every sense into one glossary — are split into
/// individually numbered senses with their group metadata as a header
/// and their forms/attribution trailing.
pub fn format_dictionary_results(
    matches: &[TermMatch],
    dict_names: &HashMap<i64, String>,
) -> Vec<FormattedEntry> {
    let mut entries = Vec::new();

    for tm in matches {
        // #43: pitch rows are data, not entries. Split them out and key
        // them by reading so a kana form's pitch lands on its group.
        let (pitch_entries, term_entries): (Vec<&DictionaryEntry>, Vec<&DictionaryEntry>) =
            tm.entries.iter().partition(|e| OcrOverlayState::pitch_positions_of(&e.definitions).is_some());
        let mut pitch_by_reading: HashMap<String, Vec<i32>> = HashMap::new();
        for e in &pitch_entries {
            let reading = OcrOverlayState::pitch_reading_of(&e.definitions)
                .unwrap_or_else(|| e.reading.clone());
            let positions = OcrOverlayState::pitch_positions_of(&e.definitions).unwrap_or_default();
            let slot = pitch_by_reading.entry(reading).or_default();
            slot.extend(positions);
        }
        for positions in pitch_by_reading.values_mut() {
            positions.sort();
            positions.dedup();
        }
        if term_entries.is_empty() {
            continue;
        }

        // One entry per (term, dictionary).
        let mut by_dict: Vec<(i64, Vec<&DictionaryEntry>)> = Vec::new();
        for e in &term_entries {
            match by_dict.iter_mut().find(|(id, _)| *id == e.dictionary_id) {
                Some((_, rows)) => rows.push(e),
                None => by_dict.push((e.dictionary_id, vec![e])),
            }
        }

        for (dict_id, dict_entries) in by_dict {
            // Glossaries already rendered for an earlier reading of this word.
            let mut seen_glossaries: HashSet<String> = HashSet::new();
            // Rows whose glossary already rendered for this word: Jitendex
            // (and JMdict) store one row per headword form, and the forms
            // of one word carry identical payloads (e.g. この/此の/斯の).
            // Such a row keeps its headword but must not repeat the senses.
            let mut seen_row_glossaries: HashSet<String> = HashSet::new();
            let mut reading_groups: Vec<FormattedReadingGroup> = Vec::new();

            let mut grouped: Vec<(String, Vec<&DictionaryEntry>)> = Vec::new();
            for entry in &dict_entries {
                if let Some(pos) = grouped.iter().position(|(r, _)| r == &entry.reading) {
                    grouped[pos].1.push(entry);
                } else {
                    grouped.push((entry.reading.clone(), vec![entry]));
                }
            }

            for (reading, reading_entries) in grouped {
                let is_kanji_entry = reading_entries
                    .first()
                    .map(|e| e.onyomi.is_some() || e.kunyomi.is_some())
                    .unwrap_or(false);

                let kanji_variants: Vec<String> = {
                    let mut seen = HashSet::new();
                    reading_entries
                        .iter()
                        .map(|e| e.kanji.clone())
                        .filter(|k| seen.insert(k.clone()))
                        .collect()
                };

                let headwords: Vec<FormattedHeadword> = kanji_variants
                    .iter()
                    .map(|kanji| {
                        let entry = reading_entries
                            .iter()
                            .find(|e| e.kanji == *kanji)
                            .unwrap_or(&reading_entries[0]);
                        FormattedHeadword {
                            kanji: kanji.clone(),
                            onyomi: entry.onyomi.clone(),
                            kunyomi: entry.kunyomi.clone(),
                        }
                    })
                    .collect();

                let mut sense_groups: Vec<FormattedSenseGroup> = Vec::new();
                let mut global_sense_num = 1usize;
                let mut group_seen_tags: HashSet<String> = HashSet::new();
                let mut current_group_tags: Option<Vec<String>> = None;
                let mut current_group_senses: Vec<FormattedSense> = Vec::new();

                fn flush_group(
                    current_group_tags: &mut Option<Vec<String>>,
                    current_group_senses: &mut Vec<FormattedSense>,
                    group_seen_tags: &mut HashSet<String>,
                    sense_groups: &mut Vec<FormattedSenseGroup>,
                ) {
                    let Some(tags) = current_group_tags.take() else {
                        return;
                    };
                    if current_group_senses.is_empty() {
                        return;
                    }
                    let is_forms = tags.iter().any(|t| {
                        t.eq_ignore_ascii_case("Forms") || t.eq_ignore_ascii_case("Other forms")
                    });
                    let filtered_tags: Vec<String> = tags
                        .into_iter()
                        .filter(|t| group_seen_tags.insert(t.clone()))
                        .collect();
                    sense_groups.push(FormattedSenseGroup {
                        tags: filtered_tags,
                        senses: std::mem::take(current_group_senses),
                        is_forms,
                        header: Vec::new(),
                        trailing: Vec::new(),
                    });
                }

                for e in &reading_entries {
                    // A repeated headword form (identical payload already
                    // rendered for this word) keeps its headword in the
                    // list above but adds no senses: mobile's
                    // `seenGlossaries` rule, applied per row.
                    if !seen_row_glossaries.insert(e.definitions.clone()) {
                        continue;
                    }
                    // Fail open on a non-array definition payload: a bare
                    // string is one sense, not zero.
                    let definitions_value: serde_json::Value =
                        serde_json::from_str(&e.definitions).unwrap_or_else(|_| {
                            serde_json::Value::String(e.definitions.clone())
                        });
                    let definitions_list: Vec<serde_json::Value> =
                        match definitions_value {
                            serde_json::Value::Array(arr) => arr,
                            other => vec![other],
                        };

                    let mut meta_tags: Vec<String> = Vec::new();
                    let mut sense_tags_map: HashMap<usize, Vec<String>> = HashMap::new();

                    if let Some(jlpt) = &e.jlpt {
                        if !jlpt.is_empty() {
                            meta_tags.push(format!("jlpt: N{}", jlpt));
                        }
                    }

                    // Parse only the first and third segments (" | "-separated),
                    // matching the Kotlin reference behaviour.
                    let segments: Vec<&str> = e.rules.split(" | ").collect();
                    for segment_idx in [0usize, 2] {
                        if let Some(segment) = segments.get(segment_idx) {
                            let mut current_sense: Option<usize> = None;
                            for tag in segment.split_whitespace() {
                                if let Ok(n) = tag.parse::<usize>() {
                                    current_sense = Some(n);
                                } else if !tag.starts_with("grade:") {
                                    if let Some(sense) = current_sense {
                                        sense_tags_map
                                            .entry(sense)
                                            .or_default()
                                            .push(tag.to_string());
                                    } else {
                                        meta_tags.push(tag.to_string());
                                    }
                                }
                            }
                        }
                    }

                    let tags: Vec<String> = {
                        let mut seen = HashSet::new();
                        meta_tags
                            .clone()
                            .into_iter()
                            .chain(sense_tags_map.get(&1).cloned().unwrap_or_default())
                            .filter(|t| seen.insert(t.clone()))
                            .collect()
                    };

                    let parsed = OcrOverlayState::parse_glossary(&definitions_list);

                    if parsed.structured {
                        // #88: Jitendex packs every sense into one row's
                        // glossary. Each structured sense group is its own
                        // visual group with its shared metadata as a
                        // header; the senses are numbered individually, and
                        // the forms/attribution trailing blocks render
                        // after the last one, unnumbered.
                        flush_group(
                            &mut current_group_tags,
                            &mut current_group_senses,
                            &mut group_seen_tags,
                            &mut sense_groups,
                        );
                        let last = parsed.groups.len().saturating_sub(1);
                        for (i, group) in parsed.groups.into_iter().enumerate() {
                            let senses = group
                                .senses
                                .into_iter()
                                .map(|nodes| {
                                    let sense = FormattedSense {
                                        index: global_sense_num,
                                        nodes,
                                    };
                                    global_sense_num += 1;
                                    sense
                                })
                                .collect();
                            let mut trailing = group.trailing;
                            if i == last {
                                let mut all = parsed.trailing.clone();
                                all.append(&mut trailing);
                                trailing = all;
                            }
                            sense_groups.push(FormattedSenseGroup {
                                tags: Vec::new(),
                                senses,
                                is_forms: false,
                                header: group.header,
                                trailing,
                            });
                        }
                    } else if current_group_tags.is_none()
                        || Some(&tags) == current_group_tags.as_ref()
                    {
                        current_group_tags = Some(tags);
                        current_group_senses.push(FormattedSense {
                            index: global_sense_num,
                            nodes: parsed.plain,
                        });
                        global_sense_num += 1;
                    } else {
                        flush_group(
                            &mut current_group_tags,
                            &mut current_group_senses,
                            &mut group_seen_tags,
                            &mut sense_groups,
                        );
                        current_group_tags = Some(tags);
                        current_group_senses.push(FormattedSense {
                            index: global_sense_num,
                            nodes: parsed.plain,
                        });
                        global_sense_num += 1;
                    }
                }

                flush_group(
                    &mut current_group_tags,
                    &mut current_group_senses,
                    &mut group_seen_tags,
                    &mut sense_groups,
                );

                let render_senses = reading_entries
                    .first()
                    .map(|e| seen_glossaries.insert(e.definitions.clone()))
                    .unwrap_or(true);

                reading_groups.push(FormattedReadingGroup {
                    reading: reading.clone(),
                    headwords,
                    sense_groups,
                    is_kanji_entry,
                    pitch_positions: pitch_by_reading.get(&reading).cloned().unwrap_or_default(),
                    render_senses,
                });
            }

            entries.push(FormattedEntry {
                term: tm.term.clone(),
                reading_groups,
                deinflection: tm.chain.clone(),
                dictionary_name: dict_names.get(&dict_id).cloned(),
            });
        }
    }

    entries
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::nav_graph::NavGraph;

    fn assert_send<T: Send>() {}

    /// Compile-time contract: the binding surface is `Send`. If an `Rc`/`Cell`
    /// ever leaks back into this module, the build (not just the runtime) breaks.
    #[test]
    fn pure_lookup_surface_is_send() {
        assert_send::<LookupOutcome>();
        assert_send::<Vec<FormattedEntry>>();
        assert_send::<Vec<TermMatch>>();
        assert_send::<HashMap<i64, String>>();
        assert_send::<DeinflectionChain>();
        assert_send::<DictionaryDatabase>();
        assert_send::<Deinflector>();
        // The other half of the split: nav-graph data must stay plainly
        // shareable across threads (it rides the same binding surface).
        assert_send::<NavGraph>();
    }

    fn fixture_deinflector() -> Deinflector {
        Deinflector::from_json_file(concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/deinflect.json"))
            .expect("shipped deinflect.json loads")
    }

    /// The three pure stages run on a worker thread and produce output
    /// identical to the caller thread. The `join()` also requires every
    /// captured and returned value to be `Send` — the runtime counterpart of
    /// `pure_lookup_surface_is_send`.
    #[test]
    fn lookup_stages_run_cross_threaded_and_match_inline() {
        let text = "食べる";
        let rows = vec![DictionaryEntry::new(
            "食べる".into(),
            "たべる".into(),
            r#"["to eat"]"#.into(),
            String::new(),
            0,
            1,
        )];
        let dict_names: HashMap<i64, String> =
            [(1i64, "Test".to_string())].into_iter().collect();
        let deinflector = fixture_deinflector();

        let (inline_terms, inline_cands) = prepare_search_candidates(text, &deinflector);
        let inline_all: Vec<String> = inline_terms.into_iter().collect();
        let (inline_matches, inline_max) =
            process_results(&rows, &inline_cands, &inline_all, text);
        let inline_formatted = format_dictionary_results(&inline_matches, &dict_names);

        let worker = std::thread::spawn(move || {
            let (terms, cands) = prepare_search_candidates(text, &deinflector);
            let all: Vec<String> = terms.into_iter().collect();
            let (matches, max) = process_results(&rows, &cands, &all, text);
            (format!("{:?}", format_dictionary_results(&matches, &dict_names)), max)
        });

        // `FormattedEntry` has no `PartialEq`; `Debug` is derived and
        // deterministic for the same inputs.
        let (worker_formatted, worker_max) = worker.join().expect("worker");
        assert_eq!(format!("{inline_formatted:?}"), worker_formatted);
        assert_eq!(inline_max, worker_max);
        assert!(inline_max >= 1, "the fixture term must match");
    }

    /// `lookup_term` crosses a thread boundary with its owned output, database
    /// and deinflector. This is the strongest pin: the thread join requires
    /// every captured and returned value to be `Send`.
    #[test]
    fn lookup_term_runs_on_a_worker_thread() {
        let dir = std::env::temp_dir().join(format!("ijd_lookup_send_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db = DictionaryDatabase::open(dir.join("d.db")).unwrap();
        let did = db.insert_dictionary("JMdict", 0).unwrap();
        db.insert_entries(&[DictionaryEntry::new(
            "食べる".into(),
            "たべる".into(),
            r#"["to eat"]"#.into(),
            String::new(),
            0,
            did,
        )])
        .unwrap();

        let active: Vec<String> = ["食", "べ", "る"]
            .iter()
            .map(|c| c.to_string())
            .collect();
        let following = following_text(&active, 0);
        let outcome = std::thread::spawn(move || {
            lookup_term(&active, 0, &following, &db, &Deinflector::empty())
        })
        .join()
        .expect("worker")
        .expect("outcome");

        assert!(outcome.max_len >= 1);
        assert!(
            format!("{:?}", outcome.matches).contains("食べる"),
            "worker lookup must find the fixture term"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
