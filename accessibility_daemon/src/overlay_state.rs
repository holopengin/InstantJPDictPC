use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use crate::data::db::DictionaryDatabase;
use crate::nav_graph;
use crate::data::models::DictionaryEntry;
use crate::models::*;
use crate::util::deinflector::Deinflector;
use crate::util::japanese;

/// Result of a dictionary lookup.
pub struct LookupResult {
    pub matches: Vec<FormattedEntry>,
    pub max_len: usize,
    pub tapped_box: BoundingBox,
}

pub struct OcrOverlayState {
    pub current_scale: f32,
    pub current_trans_x: f32,
    pub current_trans_y: f32,
    pub current_word_length: usize,
    pub active_line_boxes: Vec<BoundingBox>,
    pub active_all_chars: Vec<String>,
    pub active_all_alternatives: Vec<Vec<(char, f32)>>,
    pub current_tapped_idx: isize,
    pub current_tapped_line_idx: isize,
    pub current_tapped_char_idx_in_line: isize,
    pub active_line_results: Vec<Option<LineResult>>,
    pub last_highlighted_coords: Vec<(usize, usize)>,
    pub last_landscape_gravity: Gravity,
    pub last_portrait_gravity: Gravity,
    pub is_controller_navigation: bool,
    pub is_dictionary_visible: bool,
    pub is_alternatives_visible: bool,
    /// Cached formatted entries from the last lookup.
    pub cached_entries: Vec<FormattedEntry>,
    /// The term that was looked up (for cache invalidation).
    pub cached_lookup_term: String,
    /// The bounding box of the tapped character, used to compute panel gravity.
    pub tapped_box_for_gravity: Option<BoundingBox>,
    /// Screenshot dimensions, needed to compute the base transform for gravity.
    pub img_w: u32,
    pub img_h: u32,
    /// Current window dimensions, needed for gravity and navigation.
    /// Use Cell so they can be updated from the canvas draw() which only has &self.
    pub window_width: Cell<f32>,
    pub window_height: Cell<f32>,
    /// Navigation graph: for each node (global char index), [north, south, east, west] target indices.
    pub nav_graph: Option<crate::nav_graph::NavGraph>,
    /// Normalized (x, y) positions for each node in [0,1)².
    pub nav_positions: Vec<(f32, f32)>,
}

impl OcrOverlayState {
    pub fn new() -> Self {
        Self {
            current_scale: 1.0,
            current_trans_x: 0.0,
            current_trans_y: 0.0,
            current_word_length: 0,
            active_line_boxes: Vec::new(),
            active_all_chars: Vec::new(),
            active_all_alternatives: Vec::new(),
            current_tapped_idx: -1,
            current_tapped_line_idx: -1,
            current_tapped_char_idx_in_line: -1,
            active_line_results: Vec::new(),
            last_highlighted_coords: Vec::new(),
            last_landscape_gravity: Gravity::Start,
            last_portrait_gravity: Gravity::Top,
            is_controller_navigation: false,
            is_dictionary_visible: false,
            is_alternatives_visible: false,
            cached_entries: Vec::new(),
            cached_lookup_term: String::new(),
            tapped_box_for_gravity: None,
            img_w: 0,
            img_h: 0,
            window_width: Cell::new(800.0),
            window_height: Cell::new(480.0),
            nav_graph: None,
            nav_positions: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn update_global_data(&mut self) {
        self.active_all_chars.clear();
        self.active_all_alternatives.clear();
        for line in self.active_line_results.iter().flatten() {
            self.active_all_chars
                .extend(line.text.chars().map(|c| c.to_string()));
            self.active_all_alternatives
                .extend(line.alternatives.clone());
        }
    }

    pub fn set_line_results(&mut self, lines: Vec<Option<LineResult>>) {
        self.active_line_results = lines;
        self.active_line_boxes.clear();
        self.active_line_boxes.extend(
            self.active_line_results
                .iter()
                .flatten()
                .flat_map(|line| line.chunk_boxes.clone()),
        );
        self.update_global_data();
        self.build_nav_graph();
    }

    /// Replace a single line result at the given index, expanding the vec if needed.
    pub fn set_single_line_result(&mut self, index: usize, line: LineResult) {
        while self.active_line_results.len() <= index {
            self.active_line_results.push(None);
        }
        self.active_line_results[index] = Some(line);
        self.active_line_boxes.clear();
        self.active_line_boxes.extend(
            self.active_line_results
                .iter()
                .flatten()
                .flat_map(|line| line.chunk_boxes.clone()),
        );
        self.update_global_data();
        self.build_nav_graph();
    }

    pub fn get_global_idx(&self, line_idx: usize, char_idx_in_line: usize) -> usize {
        self.active_line_results
            .iter()
            .take(line_idx)
            .flatten()
            .map(|line| line.text.chars().count())
            .sum::<usize>()
            + char_idx_in_line
    }

    pub fn get_coords_from_global_idx(&self, global_idx: usize) -> Option<(usize, usize)> {
        let mut count = 0;
        for (line_idx, line_opt) in self.active_line_results.iter().enumerate() {
            let line = line_opt.as_ref()?;
            let line_len = line.text.chars().count();
            if global_idx < count + line_len {
                return Some((line_idx, global_idx - count));
            }
            count += line_len;
        }
        None
    }

    pub fn ensure_cursor_position(&mut self) {
        if self.current_tapped_line_idx == -1 || self.current_tapped_char_idx_in_line == -1 {
            for (line_idx, line_opt) in self.active_line_results.iter().enumerate() {
                let Some(line) = line_opt else { continue };
                if !line.text.is_empty() {
                    self.current_tapped_line_idx = line_idx as isize;
                    self.current_tapped_char_idx_in_line = 0;
                    self.current_tapped_idx = self.get_global_idx(line_idx, 0) as isize;
                    break;
                }
            }
        }
    }

    pub fn update_character(&mut self, line_idx: usize, char_idx: usize, new_char: char) {
        let Some(line) = self
            .active_line_results
            .get_mut(line_idx)
            .and_then(|line| line.as_mut())
        else {
            return;
        };
        let mut chars: Vec<char> = line.text.chars().collect();
        if let Some(slot) = chars.get_mut(char_idx) {
            *slot = new_char;
            line.text = chars.into_iter().collect();
            self.update_global_data();
self.build_nav_graph();
        }
    }


pub fn navigate(&mut self, action: GamepadAction) -> bool {
        let (line_idx, char_idx) = match self.current_cursor() {
            Some(coords) => coords,
            None => return false,
        };
        let global = self.get_global_idx(line_idx, char_idx);
        if let Some(ref graph) = self.nav_graph {
            let dir = match action {
                GamepadAction::NavigateUp => 0,
                GamepadAction::NavigateDown => 1,
                GamepadAction::NavigateRight => 2,
                GamepadAction::NavigateLeft => 3,
                _ => return false,
            };
            if let Some(next_global) = graph.navigate(global, dir) {
                if let Some((nl, nc)) = self.get_coords_from_global_idx(next_global) {
                    self.current_tapped_line_idx = nl as isize;
                    self.current_tapped_char_idx_in_line = nc as isize;
                    self.current_tapped_idx = next_global as isize;
                    return true;
                }
            }
        }
        false
    }

    pub fn build_nav_graph(&mut self) {
        let mut boxes = Vec::new();
        for line_opt in &self.active_line_results {
            if let Some(line) = line_opt {
                for b in &line.char_boxes {
                    boxes.push(b.clone());
                }
            }
        }
        let graph = nav_graph::NavGraph::build(&boxes);
        self.nav_positions = graph.positions.clone();
        self.nav_graph = Some(graph);
    }

    pub fn current_cursor(&self) -> Option<(usize, usize)> {
        let line_idx = self.current_tapped_line_idx;
        let char_idx = self.current_tapped_char_idx_in_line;
        if line_idx < 0 || char_idx < 0 {
            return None;
        }
        Some((line_idx as usize, char_idx as usize))
    }

    pub fn get_neighbor_ui_state(&self) -> Vec<NeighborLine> {
        self.active_line_results
            .iter()
            .enumerate()
            .map(|(line_idx, line_opt)| {
                let chars = line_opt
                    .as_ref()
                    .map(|line| {
                        line.text
                            .chars()
                            .enumerate()
                            .map(|(char_idx, ch)| NeighborChar {
                                text: ch.to_string(),
                                is_selected: line_idx as isize == self.current_tapped_line_idx
                                    && char_idx as isize == self.current_tapped_char_idx_in_line,
                                line_idx,
                                char_idx,
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                NeighborLine { chars, line_idx }
            })
            .collect()
    }

    pub fn get_alternatives_ui_state(&self) -> Option<AlternativesUiState> {
        let (line_idx, char_idx) = self.current_cursor()?;
        let line = self
            .active_line_results
            .get(line_idx)
            .and_then(|line| line.as_ref())?;
        let alts = line.alternatives.get(char_idx)?;
        let current_char = line.text.chars().nth(char_idx)?;

        Some(AlternativesUiState {
            candidates: alts
                .iter()
                .take(15)
                .map(|(ch, _)| AlternativeChar {
                    char: *ch,
                    is_selected: *ch == current_char,
                })
                .collect(),
            show_manual_input: true,
        })
    }

    pub fn panel_dimensions(&self) -> (f32, f32) {
        let root_width = self.window_width.get();
        let root_height = self.window_height.get();
        let is_landscape = root_width > root_height;
        let panel_width = if is_landscape {
            (root_width * 0.4).min(500.0)
        } else {
            root_width
        };
        let panel_height = if is_landscape {
            root_height
        } else {
            root_height * 0.4
        };
        (panel_width, panel_height)
    }

    /// Determine which side the panel should open on.
    /// Default: left (Start) in landscape, top (Top) in portrait.
    /// Only switch to the other side if the character would be overlapped by the panel.
    /// The panel is on the left in landscape (dict + neighbors + alt = ~386px wide).
    pub fn update_gravity(&mut self, tapped_box: &BoundingBox, panel_width: f32) {
        let root_width = self.window_width.get();
        let root_height = self.window_height.get();
        let is_landscape = root_width > root_height;
        // Compute the base transform (fit image to window) that the canvas uses
        let img_w_f = self.img_w as f32;
        let img_h_f = self.img_h as f32;
        let base_scale = if img_w_f > 0.0 && img_h_f > 0.0 {
            f32::min(root_width / img_w_f, root_height / img_h_f)
        } else {
            1.0
        };
        let base_offset_x = (root_width - img_w_f * base_scale) / 2.0;
        let base_offset_y = (root_height - img_h_f * base_scale) / 2.0;
        // Total transform = base_transform + pan/zoom
        let total_scale = base_scale * self.current_scale;
        let total_offset_x = base_offset_x + self.current_trans_x;
        let total_offset_y = base_offset_y + self.current_trans_y;
        // Character's screen position (center of bounding box)
        let char_center_x = tapped_box.left() as f32 * total_scale
            + total_offset_x
            + (tapped_box.w as f32 / 2.0) * total_scale;
        let char_center_y = tapped_box.top() as f32 * total_scale
            + total_offset_y
            + (tapped_box.h as f32 / 2.0) * total_scale;
        // Character's left edge in screen space (for overlap check)
        let char_left = tapped_box.left() as f32 * total_scale + total_offset_x;
        let char_top = tapped_box.top() as f32 * total_scale + total_offset_y;

        // In portrait, panel takes roughly half the screen height
        let panel_height = root_height * 0.5_f32;

        if is_landscape {
            // Default: panel on the left (Start)
            // Switch to right (End) only if the character would be overlapped by the left panel.
            // The panel occupies [0, panel_width] on the left side of the screen.
            // Check if the character's screen-space bounding box overlaps that region.
            let char_right = char_left + tapped_box.w as f32 * total_scale;
            let overlaps_panel = char_right > 0.0 && char_left < panel_width;
            self.last_landscape_gravity = if overlaps_panel {
                Gravity::End
            } else {
                Gravity::Start
            };
        } else {
            // Default: panel at top (Top)
            // Switch to bottom (Bottom) only if the character would be overlapped by the top panel.
            let char_bottom = char_top + tapped_box.h as f32 * total_scale;
            let overlaps_panel = char_bottom > 0.0 && char_top < panel_height;
            self.last_portrait_gravity = if overlaps_panel {
                Gravity::Bottom
            } else {
                Gravity::Top
            };
        }
    }

    pub fn update_highlight_coords(&mut self, line_idx: usize, char_idx: usize, word_length: usize) {
        self.last_highlighted_coords.clear();
        self.last_highlighted_coords.push((line_idx, char_idx));
        let global_idx = self.get_global_idx(line_idx, char_idx);
        for i in 1..word_length.max(1) {
            if let Some(coords) = self.get_coords_from_global_idx(global_idx + i) {
                self.last_highlighted_coords.push(coords);
            }
        }
    }

    pub fn calculate_display_boxes(&self, line: &LineResult) -> Vec<BoundingBox> {
        let fixed_size = if line.is_vertical {
            line.char_boxes.iter().map(|b| b.w).max().unwrap_or(0)
        } else {
            line.char_boxes.iter().map(|b| b.h).max().unwrap_or(0)
        };

        let mut refined_boxes = Vec::new();
        if let Some(first) = line.char_boxes.first() {
            refined_boxes.push(first.clone());
        }
        for i in 1..line.char_boxes.len() {
            let prev_original = &line.char_boxes[i - 1];
            let cur_original = &line.char_boxes[i];
            let advance = fixed_size;

            if line.is_vertical {
                let new_top = prev_original
                    .top()
                    .saturating_add(advance)
                    .max(cur_original.top());
                refined_boxes.push(BoundingBox::new(
                    cur_original.left(),
                    new_top,
                    cur_original.w,
                    cur_original.h,
                    cur_original.confidence,
                ));
            } else {
                let new_left = prev_original
                    .left()
                    .saturating_add(advance)
                    .max(cur_original.left());
                refined_boxes.push(BoundingBox::new(
                    new_left,
                    cur_original.top(),
                    cur_original.w,
                    cur_original.h,
                    cur_original.confidence,
                ));
            }
        }

        refined_boxes
            .iter()
            .map(|box_item| {
                let center_x = box_item.left() as f32 + (box_item.w as f32 / 2.0);
                let center_y = box_item.top() as f32 + (box_item.h as f32 / 2.0);
                let left = (center_x - fixed_size as f32 / 2.0).round() as i32;
                let top = (center_y - fixed_size as f32 / 2.0).round() as i32;
                BoundingBox::new(left, top, fixed_size, fixed_size, 1.0)
            })
            .collect()
    }

    // -------------------------------------------------------------------------
    // Dictionary lookup
    // -------------------------------------------------------------------------

    /// Perform a dictionary lookup at the given character position.
    /// Returns formatted entries if found, or None if no database/deinflector is available.
    pub fn lookup(
        &mut self,
        line_idx: usize,
        char_idx: usize,
        db: &DictionaryDatabase,
        deinflector: &Deinflector,
    ) -> Option<LookupResult> {
        let global_idx = self.get_global_idx(line_idx, char_idx);
        self.current_tapped_idx = global_idx as isize;
        self.current_tapped_line_idx = line_idx as isize;
        self.current_tapped_char_idx_in_line = char_idx as isize;

        let line = self.active_line_results.get(line_idx)?.as_ref()?;
        let tapped_box = line.char_boxes.get(char_idx)?.clone();

        let end_idx = (global_idx + 20).min(self.active_all_chars.len());
        let following_text: String = self.active_all_chars[global_idx..end_idx].join("");

        if following_text.is_empty() {
            return None;
        }

        // Check cache: if same term was already looked up, return cached result
        if self.cached_lookup_term == following_text {
            if self.cached_entries.is_empty() {
                // Previously returned no results — return None without re-searching
                return None;
            }
            return Some(LookupResult {
                matches: self.cached_entries.clone(),
                max_len: self.current_word_length,
                tapped_box,
            });
        }

        // Build search candidates
        let (all_terms, candidates_by_length) =
            self.prepare_search_candidates(&following_text, deinflector);

        if all_terms.is_empty() {
            return None;
        }

        // Query database
        let all_terms_vec: Vec<String> = all_terms.into_iter().collect();
        let db_results = db.find_by_texts(&all_terms_vec).unwrap_or_default();

        // Process results
        let (matches, max_len) =
            self.process_results(&db_results, &candidates_by_length, &all_terms_vec, &following_text);

        if matches.is_empty() {
            // No results — clear the cached entries so the UI shows "no results"
            self.cached_entries.clear();
            self.current_word_length = 0;
            self.cached_lookup_term = following_text;
            return None;
        }

        // Format results
        let formatted = self.format_dictionary_results(&matches);
        self.current_word_length = max_len;
        self.cached_entries = formatted.clone();
        self.cached_lookup_term = following_text;

        Some(LookupResult {
            matches: formatted,
            max_len,
            tapped_box,
        })
    }

    /// Prepare search candidates from the following text.
    /// Returns (set of all terms to search, candidates grouped by length).
    fn prepare_search_candidates(
        &self,
        following_text: &str,
        deinflector: &Deinflector,
    ) -> (HashSet<String>, Vec<(usize, Vec<(String, Option<Vec<String>>)>)>) {
        let mut all_terms = HashSet::new();
        let mut candidates_by_length = Vec::new();

        let max_len = following_text.chars().count();
        for len in (1..=max_len).rev() {
            let query_text_raw: String = following_text.chars().take(len).collect();
            let query_text = japanese::normalize(&query_text_raw);

            let variants = vec![
                query_text.clone(),
                japanese::katakana_to_hiragana(&query_text),
                japanese::collapse_emphatic(&query_text),
            ];
            let variants: Vec<String> = variants.into_iter().collect();

            let deinflections = deinflector.deinflect(&query_text);
            let mut length_candidates: Vec<(String, Option<Vec<String>>)> = Vec::new();

            for v in &variants {
                length_candidates.push((v.clone(), None));
                all_terms.insert(v.clone());
            }

            for d in &deinflections {
                if d.term != query_text {
                    let types = if d.rule_types.is_empty() {
                        None
                    } else {
                        Some(d.rule_types.clone())
                    };
                    length_candidates.push((d.term.clone(), types));
                    all_terms.insert(d.term.clone());
                }
            }

            candidates_by_length.push((len, length_candidates));
        }

        (all_terms, candidates_by_length)
    }

    /// Process database results to find matching entries.
    /// Preserves database result order (which is by dictionary priority ASC, popularity DESC).
    fn process_results(
        &self,
        db_results: &[DictionaryEntry],
        candidates_by_length: &[(usize, Vec<(String, Option<Vec<String>>)>)],
        _all_terms: &[String],
        following_text: &str,
    ) -> (Vec<(String, Vec<DictionaryEntry>)>, usize) {
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

        let mut matches: Vec<(String, Vec<DictionaryEntry>)> = Vec::new();
        let mut max_len = 0;

        for (len, candidates) in candidates_by_length {
            let mut found = false;
            for (term, required_types) in candidates {
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
                    matches.push((term.clone(), filtered));
                    found = true;
                }
            }
            if found && max_len == 0 {
                max_len = *len;
            }
        }

        // Deduplicate by term
        let mut seen = HashSet::new();
        matches.retain(|(term, _)| seen.insert(term.clone()));

        (matches, max_len)
    }

    /// Format dictionary results into displayable entries.
    fn format_dictionary_results(
        &self,
        matches: &[(String, Vec<DictionaryEntry>)],
    ) -> Vec<FormattedEntry> {
        matches
            .iter()
            .map(|(term, entries)| {
                let mut reading_groups: Vec<FormattedReadingGroup> = Vec::new();
                // Preserve insertion order (matches database priority order)
                let mut grouped: Vec<(String, Vec<&DictionaryEntry>)> = Vec::new();

                for entry in entries {
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
                    let mut global_sense_num = 1;
                    let mut group_seen_tags: HashSet<String> = HashSet::new();
                    let mut current_group_tags: Option<Vec<String>> = None;
                    let mut current_group_senses: Vec<FormattedSense> = Vec::new();

                    for e in &reading_entries {
                        let definitions_list: Vec<serde_json::Value> =
                            serde_json::from_str(&e.definitions).unwrap_or_default();

                        let mut meta_tags = Vec::new();
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

                        let sense_idx = global_sense_num;
                        global_sense_num += 1;
                        let tags: Vec<String> = {
                            let mut seen = HashSet::new();
                            meta_tags.clone().into_iter().chain(
                                sense_tags_map.get(&1).cloned().unwrap_or_default()
                            ).filter(|t| seen.insert(t.clone()))
                                .collect()
                        };

                        let nodes = Self::parse_definition(&definitions_list);

                        if current_group_tags.is_none() || Some(&tags) == current_group_tags.as_ref() {
                            current_group_tags = Some(tags.clone());
                            current_group_senses.push(FormattedSense {
                                index: sense_idx,
                                nodes,
                            });
                        } else {
                            let tags_to_render = current_group_tags.take().unwrap();
                            let is_forms = tags_to_render.iter().any(|t| {
                                t.eq_ignore_ascii_case("Forms") || t.eq_ignore_ascii_case("Other forms")
                            });
                            let filtered_tags: Vec<String> = tags_to_render
                                .into_iter()
                                .filter(|t| group_seen_tags.insert(t.clone()))
                                .collect();
                            sense_groups.push(FormattedSenseGroup {
                                tags: filtered_tags,
                                senses: current_group_senses.clone(),
                                is_forms,
                            });
                            current_group_tags = Some(tags);
                            current_group_senses = vec![FormattedSense {
                                index: sense_idx,
                                nodes,
                            }];
                        }
                    }

                    if let Some(tags_to_render) = current_group_tags.take() {
                        let is_forms = tags_to_render.iter().any(|t| {
                            t.eq_ignore_ascii_case("Forms") || t.eq_ignore_ascii_case("Other forms")
                        });
                        let filtered_tags: Vec<String> = tags_to_render
                            .into_iter()
                            .filter(|t| group_seen_tags.insert(t.clone()))
                            .collect();
                        sense_groups.push(FormattedSenseGroup {
                            tags: filtered_tags,
                            senses: current_group_senses,
                            is_forms,
                        });
                    }

                    reading_groups.push(FormattedReadingGroup {
                        reading,
                        headwords,
                        sense_groups,
                        is_kanji_entry,
                    });
                }

                FormattedEntry {
                    term: term.clone(),
                    reading_groups,
                }
            })
            .collect()
    }

    /// Parse Yomitan definition JSON into displayable nodes.
    fn parse_definition(data: &[serde_json::Value]) -> Vec<DefinitionNode> {
        let mut nodes = Vec::new();
        for item in data {
            Self::parse_definition_item(item, false, &mut nodes);
        }
        nodes
    }

    fn parse_definition_item(
        data: &serde_json::Value,
        in_example: bool,
        nodes: &mut Vec<DefinitionNode>,
    ) {
        match data {
            serde_json::Value::String(s) => {
                let replaced = s
                    .replace("\r\n", " ")
                    .replace('\n', " ")
                    .replace('\r', " ")
                    .replace(';', "; ")
                    .replace(";  ", "; ");
                let trimmed = replaced.trim().split_whitespace().collect::<Vec<_>>().join(" ");
                if !trimmed.is_empty() {
                    nodes.push(DefinitionNode::Text(trimmed));
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    let item_nodes = {
                        let mut sub = Vec::new();
                        Self::parse_definition_item(item, in_example, &mut sub);
                        sub
                    };
                    if !item_nodes.is_empty() {
                        if !nodes.is_empty() && !Self::is_block(item) {
                            let last = nodes.last().unwrap();
                            let first = item_nodes.first().unwrap();
                            if Self::is_inline_node(last) && Self::is_inline_node(first) {
                                let separator = if in_example { "\n" } else { ", " };
                                if let DefinitionNode::Text(ref t) = nodes.last().unwrap() {
                                    let new_text = format!("{}{}", t, separator);
                                    let last_idx = nodes.len() - 1;
                                    nodes[last_idx] = DefinitionNode::Text(new_text);
                                } else {
                                    nodes.push(DefinitionNode::Text(separator.to_string()));
                                }
                            }
                        }
                        nodes.extend(item_nodes);
                    }
                }
            }
            serde_json::Value::Object(map) => {
                let tag = map.get("tag").and_then(|v| v.as_str());
                let content = map.get("content").or_else(|| map.get("list"));
                let sc_content = Self::get_attr(map, "content");
                let sc_class = Self::get_attr(map, "class");

                if Self::is_example(map) {
                    let jp = map
                        .get("japanese")
                        .and_then(|v| v.as_str())
                        .or_else(|| content.and_then(|v| v.as_str()));
                    let en = map.get("english").and_then(|v| v.as_str());
                    if let Some(jp_str) = jp {
                        nodes.push(DefinitionNode::Example {
                            japanese: Some(jp_str.to_string()),
                            english: en.map(|s| s.to_string()),
                            content: None,
                        });
                    } else {
                        let mut sub = Vec::new();
                        if let Some(c) = content {
                            Self::parse_definition_item(c, true, &mut sub);
                        }
                        nodes.push(DefinitionNode::Example {
                            japanese: None,
                            english: en.map(|s| s.to_string()),
                            content: Some(sub),
                        });
                    }
                } else if sc_class.as_deref() == Some("tag")
                    || (tag == Some("span") && sc_content.as_deref().map_or(false, |s| s.ends_with("-info")))
                {
                    let text = content.and_then(|v| v.as_str()).unwrap_or("").to_string();
                    nodes.push(DefinitionNode::Tag {
                        text,
                        category: String::new(),
                    });
                } else if tag == Some("ruby") {
                    if let Some(serde_json::Value::Array(ruby_list)) = content {
                        if ruby_list.len() >= 2 {
                            let term = ruby_list[0].as_str().unwrap_or("").to_string();
                            let reading = ruby_list
                                .get(1)
                                .and_then(|v| v.as_object())
                                .and_then(|m| m.get("content"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            nodes.push(DefinitionNode::Ruby {
                                term,
                                reading,
                                is_mini: true,
                            });
                        }
                    }
                } else if tag == Some("table") {
                    nodes.push(DefinitionNode::Table { rows: Vec::new() });
                } else if tag == Some("ul") || tag == Some("ol") {
                    let is_inline_list = matches!(
                        sc_content.as_deref(),
                        Some("glossary") | Some("infoGlossary") | Some("sourceLanguages") | Some("info-gloss") | Some("sense-note")
                    );
                    if is_inline_list {
                        if let Some(c) = content {
                            Self::parse_definition_item(c, in_example, nodes);
                        }
                    } else {
                        let items: Vec<Vec<DefinitionNode>> = match content {
                            Some(serde_json::Value::Array(arr)) => arr
                                .iter()
                                .map(|item| {
                                    let mut sub = Vec::new();
                                    Self::parse_definition_item(item, in_example, &mut sub);
                                    sub
                                })
                                .collect(),
                            _ => Vec::new(),
                        };
                        nodes.push(DefinitionNode::ListBlock {
                            items,
                            block_type: sc_content.clone(),
                        });
                    }
                } else if let Some(c) = content {
                    Self::parse_definition_item(c, in_example, nodes);
                }
            }
            _ => {}
        }
    }

    fn get_attr(data: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
        data.get("data")
            .and_then(|v| v.as_object())
            .and_then(|m| m.get(key))
            .or_else(|| data.get(&format!("data-{}", key)))
            .or_else(|| data.get(key))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    fn is_example(data: &serde_json::Map<String, serde_json::Value>) -> bool {
        data.get("type")
            .and_then(|v| v.as_str())
            .map_or(false, |t| t == "sentence" || t == "example")
            || data.contains_key("japanese")
            || Self::get_attr(data, "content").map_or(false, |c| {
                c.contains("example") || c == "examples"
            })
            || Self::get_attr(data, "class")
                .map_or(false, |c| c.contains("example"))
    }

    fn is_inline_node(node: &DefinitionNode) -> bool {
        matches!(
            node,
            DefinitionNode::Text(_)
                | DefinitionNode::Ruby { .. }
                | DefinitionNode::Tag { .. }
        )
    }

    /// Check whether a JSON value represents a block-level element (as opposed to
    /// inline).  Mirrors the Kotlin `isBlock` helper: structural elements like
    /// tables and non-glossary lists are blocks, as are example containers.
    fn is_block(data: &serde_json::Value) -> bool {
        match data {
            serde_json::Value::Array(arr) => arr.iter().any(|item| Self::is_block(item)),
            serde_json::Value::Object(map) => {
                if Self::is_example(map) {
                    return true;
                }
                let content = map.get("content").or_else(|| map.get("list"));
                if let Some(c) = content {
                    if Self::is_block(c) {
                        return true;
                    }
                }
                let tag = map.get("tag").and_then(|v| v.as_str());
                let sc_content = Self::get_attr(map, "content");
                tag == Some("table")
                    || (tag == Some("ul") || tag == Some("ol"))
                        && !matches!(
                            sc_content.as_deref(),
                            Some("glossary")
                                | Some("infoGlossary")
                                | Some("sourceLanguages")
                                | Some("info-gloss")
                                | Some("sense-note")
                        )
            }
            _ => false,
        }
    }
}
