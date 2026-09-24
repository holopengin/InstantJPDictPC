//! Pure dictionary-lookup pipeline shared by the desktop and Android.
//!
//! This module is deliberately ungated: it accepts stored rows plus plain text
//! and returns owned data without opening a database or touching UI state. The
//! desktop `lookup` module keeps the SQLite query and redirect traversal, while
//! the Android UniFFI shim maps the records below across the binding boundary.

use std::collections::{HashMap, HashSet};

use crate::data::models::DictionaryEntry;
use crate::definition_format;
use crate::models::{
    DeinflectionChain, FormattedEntry, FormattedHeadword, FormattedReadingGroup, FormattedSense,
    FormattedSenseGroup, TermMatch,
};
use crate::util::deinflector::{DeinflectionResult, Deinflector};
use crate::util::{japanese, pitch};

/// Owned result of a dictionary lookup.
#[derive(Debug)]
pub struct LookupOutcome {
    pub matches: Vec<FormattedEntry>,
    pub max_len: usize,
}

/// One dictionary-form candidate produced for a query prefix.
///
/// `required_types = None` identifies a direct surface variant. Deinflected
/// candidates carry the rule types used to select their rows; an empty vector
/// deliberately means "any rule" and remains distinct from `None`.
#[derive(Clone, Debug)]
pub struct SearchCandidate {
    pub term: String,
    pub required_types: Option<Vec<String>>,
    pub chain: Option<DeinflectionChain>,
}

/// Candidates for one Unicode-character prefix length.
#[derive(Clone, Debug)]
pub struct CandidateGroup {
    pub length: usize,
    pub candidates: Vec<SearchCandidate>,
}

/// Prepared query variants, ready for a database query and result grouping.
#[derive(Clone, Debug)]
pub struct PreparedCandidates {
    pub terms: HashSet<String>,
    pub by_length: Vec<CandidateGroup>,
}

/// Database rows grouped under the matched dictionary terms.
#[derive(Clone, Debug)]
pub struct ProcessedResults {
    pub matches: Vec<TermMatch>,
    pub max_len: usize,
}

/// The search text for a tap: the next 20 global characters with full-width
/// spaces (the blank placeholder) stripped. Shared by the database-bound
/// lookup and the desktop overlay state.
pub fn following_text(active_all_chars: &[String], global_idx: usize) -> String {
    let end_idx = (global_idx + 20).min(active_all_chars.len());
    let text: String = active_all_chars[global_idx..end_idx].join("");
    // Strip full-width spaces (null/void character placeholder) before lookup.
    text.chars().filter(|&c| c != '\u{3000}').collect()
}

/// Count U+3000 chars in the original text that were filtered out, so the
/// highlight spans the correct visual range.
pub fn expand_max_len(active_all_chars: &[String], global_idx: usize, max_len: usize) -> usize {
    let original = &active_all_chars[global_idx..];
    let mut seen_non_space = 0usize;
    let mut expanded = 0usize;
    for ch in original {
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
pub fn prepare_search_candidates(
    following_text: &str,
    deinflector: &Deinflector,
) -> PreparedCandidates {
    prepare_search_candidates_with(following_text, |text| deinflector.deinflect(text))
}

/// Prepare candidates with an injected deinflection operation.
///
/// The PC caller passes its owned [`Deinflector`]. A binding whose host already
/// owns a deinflection facade uses this form to adapt that facade to a
/// callback, while every normalization, variant, grouping, and chain rule
/// still runs here rather than being copied into the host.
pub fn prepare_search_candidates_with<F>(
    following_text: &str,
    mut deinflect: F,
) -> PreparedCandidates
where
    F: FnMut(&str) -> Vec<DeinflectionResult>,
{
    let mut terms = HashSet::new();
    let mut by_length = Vec::new();

    let max_len = following_text.chars().count();
    for len in (1..=max_len).rev() {
        let query_text_raw: String = following_text.chars().take(len).collect();
        let query_text = japanese::normalize(&query_text_raw);

        // Pre-reform orthography is searched additively: the raw prefix stays
        // in the candidate set, so modernisation can only add a reachable form.
        let modernised = japanese::kana_orthography_modernise(&query_text);
        // Historical sound changes cover entry-local variants that the
        // orthography table alone cannot reach (やう→よう, けふ→きょう).
        let sound_changed = japanese::kana_sound_changes_modernise(&modernised);

        let mut variants: Vec<String> = Vec::new();
        for variant in [
            query_text_raw.clone(),
            query_text.clone(),
            japanese::katakana_to_hiragana(&query_text),
            japanese::collapse_emphatic(&query_text),
            modernised.clone(),
            japanese::katakana_to_hiragana(&modernised),
            sound_changed.clone(),
            japanese::katakana_to_hiragana(&sound_changed),
        ] {
            if !variants.contains(&variant) {
                variants.push(variant);
            }
        }

        let deinflections = deinflect(&query_text);
        // The rules are modern orthography, so a legacy surface must be
        // normalised before they can fire. Each stage is additive.
        let modernised_deinflections = if modernised != query_text {
            deinflect(&modernised)
        } else {
            Vec::new()
        };
        let sound_changed_deinflections = if sound_changed != modernised {
            deinflect(&sound_changed)
        } else {
            Vec::new()
        };

        let mut candidates = Vec::new();
        for variant in &variants {
            candidates.push(SearchCandidate {
                term: variant.clone(),
                required_types: None,
                chain: None,
            });
            terms.insert(variant.clone());
        }

        fn push_deinflections(
            deinflections: &[DeinflectionResult],
            same_as: &str,
            surface: &str,
            terms: &mut HashSet<String>,
            candidates: &mut Vec<SearchCandidate>,
        ) {
            for deinflection in deinflections {
                // A no-op result never becomes a duplicate candidate.
                if deinflection.term == same_as || deinflection.reasons.is_empty() {
                    continue;
                }
                let required_types = if deinflection.rule_types.is_empty() {
                    None
                } else {
                    Some(deinflection.rule_types.clone())
                };
                candidates.push(SearchCandidate {
                    term: deinflection.term.clone(),
                    required_types,
                    chain: Some(DeinflectionChain {
                        surface: surface.to_string(),
                        steps: deinflection.reasons.clone(),
                    }),
                });
                terms.insert(deinflection.term.clone());
            }
        }

        push_deinflections(
            &deinflections,
            &query_text,
            &query_text_raw,
            &mut terms,
            &mut candidates,
        );
        push_deinflections(
            &modernised_deinflections,
            &modernised,
            &query_text_raw,
            &mut terms,
            &mut candidates,
        );
        push_deinflections(
            &sound_changed_deinflections,
            &sound_changed,
            &query_text_raw,
            &mut terms,
            &mut candidates,
        );

        by_length.push(CandidateGroup {
            length: len,
            candidates,
        });
    }

    PreparedCandidates { terms, by_length }
}

/// Process database rows into term matches.
///
/// Database result order is preserved (dictionary priority ascending,
/// popularity descending). The first prefix length with any match determines
/// `max_len`; duplicate terms are removed after candidate traversal.
pub fn process_results(
    db_results: &[DictionaryEntry],
    prepared: &PreparedCandidates,
    following_text: &str,
) -> ProcessedResults {
    // Vec-based grouping deliberately preserves the database's ranked order.
    let mut results_by_term: Vec<(String, Vec<DictionaryEntry>)> = Vec::new();
    for entry in db_results {
        if let Some(position) = results_by_term
            .iter()
            .position(|(term, _)| term == &entry.kanji)
        {
            results_by_term[position].1.push(entry.clone());
        } else {
            results_by_term.push((entry.kanji.clone(), vec![entry.clone()]));
        }
        if entry.reading != entry.kanji {
            if let Some(position) = results_by_term
                .iter()
                .position(|(term, _)| term == &entry.reading)
            {
                results_by_term[position].1.push(entry.clone());
            } else {
                results_by_term.push((entry.reading.clone(), vec![entry.clone()]));
            }
        }
    }

    let mut matches = Vec::new();
    let mut max_len = 0usize;
    for group in &prepared.by_length {
        let mut found = false;
        for candidate in &group.candidates {
            let Some((_, term_entries)) = results_by_term
                .iter()
                .find(|(term, _)| term == &candidate.term)
            else {
                continue;
            };

            let filtered: Vec<DictionaryEntry> =
                if let Some(required_types) = &candidate.required_types {
                    term_entries
                        .iter()
                        .filter(|entry| {
                            let entry_tags: Vec<&str> = entry.rules.split_whitespace().collect();
                            required_types.is_empty()
                                || required_types.iter().any(|required| {
                                    entry_tags
                                        .iter()
                                        .any(|entry_tag| *entry_tag == required.as_str())
                                })
                                || (entry_tags.iter().any(|tag| tag.starts_with('v'))
                                    && required_types.iter().any(|tag| tag.starts_with('v')))
                        })
                        .cloned()
                        .collect()
                } else {
                    // A direct reading match must not sweep in an unrelated kanji
                    // entry whose only relationship is a shared reading.
                    let query_text = japanese::normalize(
                        &following_text
                            .chars()
                            .take(group.length)
                            .collect::<String>(),
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
                    term: candidate.term.clone(),
                    entries: filtered,
                    chain: candidate.chain.clone(),
                });
                found = true;
            }
        }
        if found && max_len == 0 {
            max_len = group.length;
        }
    }

    let mut seen = HashSet::new();
    matches.retain(|term_match| seen.insert(term_match.term.clone()));
    ProcessedResults { matches, max_len }
}

/// Format dictionary results into displayable entries.
///
/// One entry is emitted per `(term, dictionary)` so dictionaries never merge.
/// Repeated glossaries retain their headword without repeating senses, pitch
/// rows attach to their reading rather than becoming entries, and Jitendex
/// sense groups are split into individually numbered senses.
pub fn format_dictionary_results(
    matches: &[TermMatch],
    dict_names: &HashMap<i64, String>,
) -> Vec<FormattedEntry> {
    let mut entries = Vec::new();

    for term_match in matches {
        let (pitch_entries, term_entries): (Vec<&DictionaryEntry>, Vec<&DictionaryEntry>) =
            term_match
                .entries
                .iter()
                .partition(|entry| pitch::pitch_positions_of(&entry.definitions).is_some());
        let mut pitch_by_reading: HashMap<String, Vec<i32>> = HashMap::new();
        for entry in &pitch_entries {
            let reading = pitch::pitch_reading_of(&entry.definitions)
                .unwrap_or_else(|| entry.reading.clone());
            pitch_by_reading
                .entry(reading)
                .or_default()
                .extend(pitch::pitch_positions_of(&entry.definitions).unwrap_or_default());
        }
        for positions in pitch_by_reading.values_mut() {
            positions.sort();
            positions.dedup();
        }
        if term_entries.is_empty() {
            continue;
        }

        let mut by_dict: Vec<(i64, Vec<&DictionaryEntry>)> = Vec::new();
        for entry in &term_entries {
            match by_dict
                .iter_mut()
                .find(|(dictionary_id, _)| *dictionary_id == entry.dictionary_id)
            {
                Some((_, rows)) => rows.push(entry),
                None => by_dict.push((entry.dictionary_id, vec![entry])),
            }
        }

        for (dictionary_id, dict_entries) in by_dict {
            let mut seen_glossaries: HashSet<String> = HashSet::new();
            let mut seen_row_glossaries: HashSet<String> = HashSet::new();
            let mut reading_groups = Vec::new();

            let mut grouped: Vec<(String, Vec<&DictionaryEntry>)> = Vec::new();
            for entry in &dict_entries {
                if let Some(position) = grouped
                    .iter()
                    .position(|(reading, _)| reading == &entry.reading)
                {
                    grouped[position].1.push(entry);
                } else {
                    grouped.push((entry.reading.clone(), vec![entry]));
                }
            }

            for (reading, reading_entries) in grouped {
                let is_kanji_entry = reading_entries
                    .first()
                    .map(|entry| entry.onyomi.is_some() || entry.kunyomi.is_some())
                    .unwrap_or(false);
                let kanji_variants: Vec<String> = {
                    let mut seen = HashSet::new();
                    reading_entries
                        .iter()
                        .map(|entry| entry.kanji.clone())
                        .filter(|kanji| seen.insert(kanji.clone()))
                        .collect()
                };
                let headwords: Vec<FormattedHeadword> = kanji_variants
                    .iter()
                    .map(|kanji| {
                        let entry = reading_entries
                            .iter()
                            .find(|entry| entry.kanji == *kanji)
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
                    let is_forms = tags.iter().any(|tag| {
                        tag.eq_ignore_ascii_case("Forms") || tag.eq_ignore_ascii_case("Other forms")
                    });
                    let filtered_tags: Vec<String> = tags
                        .into_iter()
                        .filter(|tag| group_seen_tags.insert(tag.clone()))
                        .collect();
                    sense_groups.push(FormattedSenseGroup {
                        tags: filtered_tags,
                        senses: std::mem::take(current_group_senses),
                        is_forms,
                        header: Vec::new(),
                        trailing: Vec::new(),
                    });
                }

                for entry in &reading_entries {
                    // Headword variants of one payload keep their headword but
                    // do not render a second copy of the senses.
                    if !seen_row_glossaries.insert(entry.definitions.clone()) {
                        continue;
                    }
                    // Fail open: a bare string is one sense, not zero.
                    let definitions_value: serde_json::Value =
                        serde_json::from_str(&entry.definitions).unwrap_or_else(|_| {
                            serde_json::Value::String(entry.definitions.clone())
                        });
                    let definitions_list: Vec<serde_json::Value> = match definitions_value {
                        serde_json::Value::Array(array) => array,
                        other => vec![other],
                    };

                    let mut meta_tags = Vec::new();
                    let mut sense_tags: HashMap<usize, Vec<String>> = HashMap::new();
                    if let Some(jlpt) = &entry.jlpt {
                        if !jlpt.is_empty() {
                            meta_tags.push(format!("jlpt: N{jlpt}"));
                        }
                    }

                    // Only the first and third pipe-delimited rule segments
                    // contain renderable tags.
                    let segments: Vec<&str> = entry.rules.split(" | ").collect();
                    for segment_index in [0usize, 2] {
                        if let Some(segment) = segments.get(segment_index) {
                            let mut current_sense = None;
                            for tag in segment.split_whitespace() {
                                if let Ok(number) = tag.parse::<usize>() {
                                    current_sense = Some(number);
                                } else if !tag.starts_with("grade:") {
                                    if let Some(sense) = current_sense {
                                        sense_tags.entry(sense).or_default().push(tag.to_string());
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
                            .iter()
                            .cloned()
                            .chain(sense_tags.get(&1).cloned().unwrap_or_default())
                            .filter(|tag| seen.insert(tag.clone()))
                            .collect()
                    };
                    let parsed = definition_format::parse_glossary(&definitions_list);

                    if parsed.structured {
                        flush_group(
                            &mut current_group_tags,
                            &mut current_group_senses,
                            &mut group_seen_tags,
                            &mut sense_groups,
                        );
                        let last = parsed.groups.len().saturating_sub(1);
                        for (index, group) in parsed.groups.into_iter().enumerate() {
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
                            if index == last {
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
                    .map(|entry| seen_glossaries.insert(entry.definitions.clone()))
                    .unwrap_or(true);
                let pitch_positions = pitch_by_reading.get(&reading).cloned().unwrap_or_default();

                reading_groups.push(FormattedReadingGroup {
                    reading,
                    headwords,
                    sense_groups,
                    is_kanji_entry,
                    pitch_positions,
                    render_senses,
                });
            }

            entries.push(FormattedEntry {
                term: term_match.term.clone(),
                reading_groups,
                deinflection: term_match.chain.clone(),
                dictionary_name: dict_names.get(&dictionary_id).cloned(),
            });
        }
    }

    entries
}

/// JSON projection of [`format_dictionary_results`] for a UniFFI string
/// boundary. Definition nodes stay represented by their counts: they are
/// recursive presentation values, and Android keeps its existing node mapper
/// rather than crossing a second recursive record model.
pub fn format_dictionary_results_json(
    matches: &[TermMatch],
    dict_names: &HashMap<i64, String>,
) -> String {
    #[derive(serde::Serialize)]
    struct EntrySnapshot<'a> {
        term: &'a str,
        dictionary_name: &'a Option<String>,
        deinflection: Option<ChainSnapshot<'a>>,
        reading_groups: Vec<ReadingSnapshot<'a>>,
    }
    #[derive(serde::Serialize)]
    struct ChainSnapshot<'a> {
        surface: &'a str,
        steps: &'a [String],
    }
    #[derive(serde::Serialize)]
    struct ReadingSnapshot<'a> {
        reading: &'a str,
        headwords: Vec<HeadwordSnapshot<'a>>,
        sense_groups: Vec<SenseGroupSnapshot<'a>>,
        is_kanji_entry: bool,
        pitch_positions: &'a [i32],
        render_senses: bool,
    }
    #[derive(serde::Serialize)]
    struct HeadwordSnapshot<'a> {
        kanji: &'a str,
        onyomi: &'a Option<String>,
        kunyomi: &'a Option<String>,
    }
    #[derive(serde::Serialize)]
    struct SenseGroupSnapshot<'a> {
        tags: &'a [String],
        senses: Vec<SenseSnapshot>,
        is_forms: bool,
        header_node_count: usize,
        trailing_node_count: usize,
    }
    #[derive(serde::Serialize)]
    struct SenseSnapshot {
        index: usize,
        node_count: usize,
    }

    let formatted = format_dictionary_results(matches, dict_names);
    let snapshot: Vec<_> = formatted
        .iter()
        .map(|entry| EntrySnapshot {
            term: &entry.term,
            dictionary_name: &entry.dictionary_name,
            deinflection: entry.deinflection.as_ref().map(|chain| ChainSnapshot {
                surface: &chain.surface,
                steps: &chain.steps,
            }),
            reading_groups: entry
                .reading_groups
                .iter()
                .map(|group| ReadingSnapshot {
                    reading: &group.reading,
                    headwords: group
                        .headwords
                        .iter()
                        .map(|headword| HeadwordSnapshot {
                            kanji: &headword.kanji,
                            onyomi: &headword.onyomi,
                            kunyomi: &headword.kunyomi,
                        })
                        .collect(),
                    sense_groups: group
                        .sense_groups
                        .iter()
                        .map(|sense_group| SenseGroupSnapshot {
                            tags: &sense_group.tags,
                            senses: sense_group
                                .senses
                                .iter()
                                .map(|sense| SenseSnapshot {
                                    index: sense.index,
                                    node_count: sense.nodes.len(),
                                })
                                .collect(),
                            is_forms: sense_group.is_forms,
                            header_node_count: sense_group.header.len(),
                            trailing_node_count: sense_group.trailing.len(),
                        })
                        .collect(),
                    is_kanji_entry: group.is_kanji_entry,
                    pitch_positions: &group.pitch_positions,
                    render_senses: group.render_senses,
                })
                .collect(),
        })
        .collect();
    serde_json::to_string(&snapshot).unwrap_or_else(|_| "[]".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn pure_lookup_surface_is_send() {
        assert_send::<LookupOutcome>();
        assert_send::<PreparedCandidates>();
        assert_send::<ProcessedResults>();
        assert_send::<SearchCandidate>();
        assert_send::<FormattedEntry>();
        assert_send::<TermMatch>();
        assert_send::<HashMap<i64, String>>();
        assert_send::<DeinflectionChain>();
        assert_send::<Deinflector>();
    }

    fn fixture_deinflector() -> Deinflector {
        Deinflector::from_json_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../assets/deinflect.json"
        ))
        .expect("shipped deinflect.json loads")
    }

    /// The pure stages run on a worker thread and produce the same output as
    /// they do inline. The join also checks every captured value is `Send`.
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
        let dict_names: HashMap<i64, String> = [(1i64, "Test".to_string())].into_iter().collect();
        let deinflector = fixture_deinflector();

        let inline_prepared = prepare_search_candidates(text, &deinflector);
        let inline_terms: Vec<String> = inline_prepared.terms.iter().cloned().collect();
        let inline_processed = process_results(&rows, &inline_prepared, text);
        let inline_formatted = format_dictionary_results(&inline_processed.matches, &dict_names);

        let worker = std::thread::spawn(move || {
            let prepared = prepare_search_candidates(text, &deinflector);
            let terms: Vec<String> = prepared.terms.iter().cloned().collect();
            let processed = process_results(&rows, &prepared, text);
            (
                format!(
                    "{:?}",
                    format_dictionary_results(&processed.matches, &dict_names)
                ),
                processed.max_len,
                terms,
            )
        });

        let (worker_formatted, worker_max, worker_terms) = worker.join().expect("worker");
        assert_eq!(format!("{inline_formatted:?}"), worker_formatted);
        assert_eq!(inline_processed.max_len, worker_max);
        assert_eq!(inline_terms.len(), worker_terms.len());
        assert!(inline_processed.max_len >= 1, "the fixture term must match");
    }

    #[test]
    fn following_text_and_highlight_expansion_count_blanks() {
        let active = ["食", "\u{3000}", "べ", "る"]
            .iter()
            .map(|character| character.to_string())
            .collect::<Vec<_>>();
        assert_eq!(following_text(&active, 0), "食べる");
        assert_eq!(expand_max_len(&active, 0, 2), 3);
    }

    #[test]
    fn json_projection_keeps_dictionary_and_chain_shape() {
        let mut row = DictionaryEntry::new(
            "食べる".into(),
            "たべる".into(),
            r#"["to eat"]"#.into(),
            String::new(),
            0,
            1,
        );
        row.id = 7;
        let matches = vec![TermMatch {
            term: "食べる".into(),
            entries: vec![row],
            chain: Some(DeinflectionChain {
                surface: "食べた".into(),
                steps: vec!["past".into()],
            }),
        }];
        let names = [(1i64, "JMdict".to_string())].into_iter().collect();
        let json = format_dictionary_results_json(&matches, &names);
        assert!(json.contains(r#""term":"食べる""#));
        assert!(json.contains(r#""dictionary_name":"JMdict""#));
        assert!(json.contains(r#""surface":"食べた""#));
        assert!(json.contains(r#""index":1"#));
    }
}
