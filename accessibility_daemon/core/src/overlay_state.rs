use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use crate::data::db::DictionaryDatabase;
use crate::nav_graph;
use crate::data::models::DictionaryEntry;
use crate::models::*;
use crate::util::char_lm::CharLm;
use crate::util::deinflector::Deinflector;
use crate::util::gap_candidates;
use crate::util::japanese;
use crate::util::kanji_variants::KanjiVariantTable;
use crate::util::oov_candidates::OovCandidates;
use crate::util::oov_suggestions;
use crate::util::oov_suggestions::Source;

/// Result of a dictionary lookup.
pub struct LookupResult {
    pub matches: Rc<Vec<FormattedEntry>>,
    pub max_len: usize,
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
    pub is_dictionary_visible: bool,
    /// Cached formatted entries from the last lookup (Rc for O(1) clone in view()).
    pub cached_entries: Rc<Vec<FormattedEntry>>,
    /// The term that was looked up (for cache invalidation).
    pub cached_lookup_term: String,
    /// Per-session cache of kanji readings to avoid repeated SQLite queries.
    pub kanji_cache: HashMap<String, Vec<DictionaryEntry>>,
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
    /// Mobile `CharLm` (#44): ranks blank candidates by the line context.
    /// `None` when the asset is missing; the list then keeps discovery order.
    pub char_lm: Option<Arc<CharLm>>,
    /// Mobile `OovCandidates` (#44): component neighbours for a tapped
    /// character the dictionary misses. `None` when the component-table asset
    /// is missing; the list is then the head list unchanged.
    pub oov_candidates: Option<Arc<OovCandidates>>,
    /// Mobile `KanjiVariants` (#44): obsolete variant forms offered last in a
    /// tapped character's list. `None` when the variants asset is missing.
    pub kanji_variants: Option<Arc<KanjiVariantTable>>,
}

impl OcrOverlayState {
    pub fn new(window_width: f32, window_height: f32) -> Self {
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
            is_dictionary_visible: false,
            cached_entries: Rc::new(Vec::new()),
            cached_lookup_term: String::new(),
            kanji_cache: HashMap::new(),
            img_w: 0,
            img_h: 0,
            window_width: Cell::new(window_width),
            window_height: Cell::new(window_height),
            nav_graph: None,
            nav_positions: Vec::new(),
            char_lm: None,
            oov_candidates: None,
            kanji_variants: None,
        }
    }

    /// Mobile `installCharLm`: the character n-gram model that ranks blank
    /// candidates (#44). `None` keeps the discovery order.
    pub fn install_char_lm(&mut self, lm: Option<Arc<CharLm>>) {
        self.char_lm = lm;
    }

    /// Mobile `OovCandidates` + `KanjiVariants` (#44): the component table and
    /// variant forms behind a tapped character's extra candidates. `None`
    /// keeps the head-only list.
    pub fn install_oov_candidates(&mut self, oov: Option<Arc<OovCandidates>>) {
        self.oov_candidates = oov;
    }

    pub fn install_kanji_variants(&mut self, variants: Option<Arc<KanjiVariantTable>>) {
        self.kanji_variants = variants;
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


    /// Replace many line results at once (index-keyed, any order). The
    /// derived `active_line_boxes` / global char data are rebuilt once for
    /// the whole set rather than after every line.
    pub fn set_line_results_batch(&mut self, results: Vec<(usize, LineResult)>) {
        for (index, line) in results {
            while self.active_line_results.len() <= index {
                self.active_line_results.push(None);
            }
            self.active_line_results[index] = Some(line);
        }
        self.active_line_boxes.clear();
        self.active_line_boxes.extend(
            self.active_line_results
                .iter()
                .flatten()
                .flat_map(|line| line.chunk_boxes.clone()),
        );
        self.update_global_data();
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
            let line = match line_opt.as_ref() {
                Some(l) => l,
                None => continue,
            };
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
        let char_lm = self.char_lm.clone();
        let Some(line) = self
            .active_line_results
            .get_mut(line_idx)
            .and_then(|line| line.as_mut())
        else {
            return;
        };
        let mut chars: Vec<char> = line.text.chars().collect();
        let Some(slot) = chars.get_mut(char_idx) else {
            return;
        };
        let was_gap = *slot == GAP_CHAR;
        *slot = new_char;
        line.text = chars.into_iter().collect();

        // A filled blank keeps its list (#44): the position's alternatives
        // become the placeholder plus the evidence candidates, so re-opening
        // the panel shows the same list with the new character selected
        // instead of collapsing to the placeholder alone.
        if was_gap {
            let context = gap_candidates::context_before(&line.text, char_idx);
            let lm = char_lm.as_deref();
            let mut ranked =
                gap_candidates::generate(&line.raw_alternatives, gap_candidates::MAX, lm, &context);
            if ranked.is_empty() {
                ranked = gap_candidates::fallback(gap_candidates::MAX, lm, &context);
            }
            if let Some(entry) = line.alternatives.get_mut(char_idx) {
                let mut out = Vec::with_capacity(ranked.len() + 1);
                out.push((GAP_CHAR, 0.0));
                out.extend(ranked.into_iter().map(|c| (c, 0.0)));
                *entry = out;
            }
        }

        self.update_global_data();
        self.build_nav_graph();
    }


pub fn navigate(&mut self, action: GamepadAction) -> bool {
    // Ensure nav graph is fresh before navigating.
    self.rebuild_nav_if_dirty();
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

    /// Mark the nav graph as stale — it will be rebuilt on next navigation or
    /// render.  Called frequently (after each streaming OCR result) instead of
    /// the expensive `build_nav_graph()`.
    pub fn mark_nav_dirty(&mut self) {
        self.nav_graph = None;
    }

    /// Ensure the nav graph is fresh.  Returns true if a rebuild actually happened.
    pub fn rebuild_nav_if_dirty(&mut self) -> bool {
        if self.nav_graph.is_none() {
            self.build_nav_graph();
            true
        } else {
            false
        }
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
        let current_char = line.text.chars().nth(char_idx)?;

        // A blank is not a character to expand from its own table entry (#44):
        // the placeholder's list is the placeholder itself plus the
        // evidence-ranked candidates from the line's per-timestep top-K. This
        // is checked *before* the alternatives table because `with_gap_char`
        // writes a placeholder entry into that table — reading it (as an
        // earlier version did) made the ranked path unreachable and the list
        // came back as the dotted circle alone.
        if current_char == GAP_CHAR {
            let context = gap_candidates::context_before(&line.text, char_idx);
            let lm = self.char_lm.as_deref();
            let mut ranked =
                gap_candidates::generate(&line.raw_alternatives, gap_candidates::MAX, lm, &context);
            if ranked.is_empty() {
                // A blank with nothing to choose from is worse than a guess.
                ranked = gap_candidates::fallback(gap_candidates::MAX, lm, &context);
            }
            let mut candidates = Vec::with_capacity(ranked.len() + 1);
            candidates.push(AlternativeChar {
                char: GAP_CHAR,
                is_selected: true,
                source: Source::Head,
            });
            // Mobile tags the blank path's LM-ranked entries `Lm`: the model
            // chose their order (or, for the fallback, their within-class
            // order), so the tint shows what the LM did. Without a model the
            // pool keeps discovery order and the entries are head evidence.
            let ranked_source = if lm.is_some() {
                Source::Lm
            } else {
                Source::Head
            };
            candidates.extend(ranked.into_iter().map(|ch| AlternativeChar {
                char: ch,
                is_selected: false,
                source: ranked_source,
            }));
            return Some(AlternativesUiState { candidates });
        }

        let alts = line.alternatives.get(char_idx)?;

        // Mobile `OovSuggestions.assemble` (#44): the head's own ranking first
        // and unchanged, then component neighbours by descending IDF mass,
        // then the obsolete variant forms of the current character. The blank
        // path above keeps its own rules (no component expansion).
        let head: Vec<char> = alts.iter().take(15).map(|(ch, _)| *ch).collect();
        let oov = self.oov_candidates.clone();
        let variants = self.kanji_variants.clone();
        let assembled = oov_suggestions::assemble(
            current_char,
            &head,
            oov.as_deref(),
            &|ch| {
                variants
                    .as_deref()
                    .map(|t| t.obsolete_forms_of(ch))
                    .unwrap_or_default()
            },
        );

        Some(AlternativesUiState {
            candidates: assembled
                .into_iter()
                .map(|s| AlternativeChar {
                    char: s.ch,
                    is_selected: s.ch == current_char,
                    source: s.source,
                })
                .collect(),
        })
    }


    /// Determine which side the panel should open on.
    /// Default: left (Start) in landscape, top (Top) in portrait.
    /// Only switch to the other side if the character would be overlapped by the panel.
    /// The panel is on the left in landscape (dict + neighbors + alt = ~386px wide).
    pub fn update_gravity(&mut self, tapped_box: &BoundingBox, _panel_width: f32) {
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

        eprintln!(
            "[GRAVITY] img=({img_w_f},{img_h_f}) window=({root_width},{root_height}) \
             total_scale={total_scale:.2} total_off=({total_offset_x:.0},{total_offset_y:.0}) \
             bbox=({bx},{bw}) char_center=({char_center_x:.0},{char_center_y:.0}) \
             gravity={g:?}",
            img_w_f=img_w_f, img_h_f=img_h_f,
            root_width=root_width, root_height=root_height,
            total_scale=total_scale,
            total_offset_x=total_offset_x, total_offset_y=total_offset_y,
            bx=tapped_box.left(), bw=tapped_box.w,
            g=self.last_landscape_gravity,
        );

        // Android `updateGravity`: the panel takes the opposite half from the
        // tapped character. Landscape → END when the character is in the left
        // half, START otherwise; portrait → BOTTOM in the top half, TOP
        // otherwise.
        if is_landscape {
            self.last_landscape_gravity = if char_center_x < root_width / 2.0 {
                Gravity::End
            } else {
                Gravity::Start
            };
        } else {
            self.last_portrait_gravity = if char_center_y < root_height / 2.0 {
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
        line.char_boxes.get(char_idx)?;

        // Tapping on a full-width space (void/blank placeholder) → no results
        if line.text.chars().nth(char_idx) == Some('\u{3000}') {
            self.cached_entries = Rc::new(Vec::new());
            self.current_word_length = 0;
            self.cached_lookup_term = String::new();
            return None;
        }

        let end_idx = (global_idx + 20).min(self.active_all_chars.len());
        let following_text: String = self.active_all_chars[global_idx..end_idx].join("");
        // Strip full-width spaces (null/void character placeholder) before lookup
        let following_text: String = following_text.chars().filter(|&c| c != '\u{3000}').collect();

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
                matches: Rc::clone(&self.cached_entries),
                max_len: self.current_word_length,
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
        let dict_names = db.dictionary_names().unwrap_or_default();

        // Process results
        let (matches, max_len) =
            self.process_results(&db_results, &candidates_by_length, &all_terms_vec, &following_text);

        // Expand max_len to count U+3000 chars in the original text that were
        // filtered out, so the highlight spans the correct visual range.
        let max_len = {
            let original = &self.active_all_chars[global_idx..];
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
        };

        if matches.is_empty() {
            // No results — clear the cached entries so the UI shows "no results"
            self.cached_entries = Rc::new(Vec::new());
            self.current_word_length = 0;
            self.cached_lookup_term = following_text;
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
            let mut visited: HashSet<String> =
                matches.iter().map(|m| m.term.clone()).collect();
            let mut queue: std::collections::VecDeque<TermMatch> =
                matches.iter().cloned().collect();
            let mut hops = 0;
            while !queue.is_empty() && hops < 3 {
                for _ in 0..queue.len() {
                    let Some(m) = queue.pop_front() else { break };
                    for entry in &m.entries {
                        for target in Self::extract_redirect_targets(&entry.definitions, 3) {
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
        let mut formatted = self.format_dictionary_results(&resolved_matches, &dict_names);
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
        self.current_word_length = max_len;
        self.cached_entries = Rc::new(formatted);
        self.cached_lookup_term = following_text;

        Some(LookupResult {
            matches: Rc::clone(&self.cached_entries),
            max_len,
        })
    }

    /// Prepare search candidates from the following text.
    /// Returns (set of all terms to search, candidates grouped by length).
    fn prepare_search_candidates(
        &self,
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
    fn process_results(
        &self,
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
        &mut self,
        matches: &[TermMatch],
        dict_names: &HashMap<i64, String>,
    ) -> Vec<FormattedEntry> {
        let mut entries = Vec::new();

        for tm in matches {
            // #43: pitch rows are data, not entries. Split them out and key
            // them by reading so a kana form's pitch lands on its group.
            let (pitch_entries, term_entries): (Vec<&DictionaryEntry>, Vec<&DictionaryEntry>) =
                tm.entries.iter().partition(|e| Self::pitch_positions_of(&e.definitions).is_some());
            let mut pitch_by_reading: HashMap<String, Vec<i32>> = HashMap::new();
            for e in &pitch_entries {
                let reading = Self::pitch_reading_of(&e.definitions)
                    .unwrap_or_else(|| e.reading.clone());
                let positions = Self::pitch_positions_of(&e.definitions).unwrap_or_default();
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

                        let parsed = Self::parse_glossary(&definitions_list);

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

    // -------------------------------------------------------------------------
    // Structured-content parsing
    // -------------------------------------------------------------------------

    /// The `data-content` classes that are structural blocks rather than
    /// inline text. A block never gets a comma spliced in front of it, and its
    /// own children keep their own layout.
    const BLOCK_CONTENT_CLASSES: &'static [&'static str] = &[
        "sense-groups", "sense-group", "sense",
        "forms", "extra-info",
        "example-sentence", "xref", "antonym", "related",
        "sense-note", "info-gloss", "lang-source", "attribution", "graphic",
    ];

    /// Jitendex's `extra-info` boxes: each renders as its own line.
    const BOXED_CONTENT_CLASSES: &'static [&'static str] = &[
        "xref", "antonym", "related", "sense-note", "info-gloss", "lang-source",
    ];

    /// `data-content` values that stay inline even on a `ul`/`ol`.
    const INLINE_LIST_CLASSES: &'static [&'static str] = &[
        "glossary", "infoGlossary", "sourceLanguages", "info-gloss", "sense-note",
    ];

    /// Jitendex form-validity cell classes → the glyph upstream draws.
    const FORM_MARKERS: &'static [(&'static str, &'static str)] = &[
        ("form-valid", "◇"),
        ("form-rare", "▽"),
        ("form-pri", "★"),
        ("form-irr", "✕"),
        ("form-out", "古"),
        ("form-old", "旧"),
    ];

    /// Yomitan structured-content class on a node (`data.content`), or None.
    fn content_class(node: &serde_json::Value) -> Option<String> {
        node.as_object()
            .and_then(|m| m.get("data"))
            .and_then(|d| d.as_object())
            .and_then(|d| d.get("content"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Parse a stored definition payload into displayable nodes.
    fn parse_definition(data: &serde_json::Value, separator: &str) -> Vec<DefinitionNode> {
        let mut nodes = Vec::new();
        Self::parse_definition_into(data, separator, &mut nodes);
        nodes
    }

    fn parse_definition_into(
        data: &serde_json::Value,
        separator: &str,
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
                    let mut item_nodes = Vec::new();
                    Self::parse_definition_into(item, separator, &mut item_nodes);
                    if item_nodes.is_empty() {
                        continue;
                    }
                    // Attach the separator to the PREVIOUS text node where
                    // possible, so it cannot wrap onto a line of its own.
                    if !nodes.is_empty() && !separator.is_empty() && !Self::is_block(item) {
                        let last = nodes.last().unwrap();
                        let first = item_nodes.first().unwrap();
                        if Self::is_inline_node(last) && Self::is_inline_node(first) {
                            // #88: a ruby run is one word or sentence, not an
                            // enumeration — never splice a separator between
                            // its pieces (the xref 湾外 must not render
                            // "湾, 外").
                            let glue = if matches!(last, DefinitionNode::Ruby { .. })
                                || matches!(first, DefinitionNode::Ruby { .. })
                            {
                                ""
                            } else {
                                separator
                            };
                            if !glue.is_empty() {
                                let last_idx = nodes.len() - 1;
                                if let DefinitionNode::Text(ref t) = nodes[last_idx] {
                                    nodes[last_idx] = DefinitionNode::Text(format!("{}{}", t, glue));
                                } else {
                                    nodes.push(DefinitionNode::Text(glue.to_string()));
                                }
                            }
                        }
                    }
                    nodes.extend(item_nodes);
                }
            }
            serde_json::Value::Object(map) => {
                let tag = map.get("tag").and_then(|v| v.as_str());
                let content = map.get("content").or_else(|| map.get("list"));
                let sc_content = Self::get_attr(map, "content");
                let sc_class = Self::get_attr(map, "class");

                let citation = Self::content_class(data).as_deref() == Some("attribution");
                if citation {
                    // #88 follow-up: the source line that closes a Jitendex
                    // entry; rendered as a faint citation, not definition text.
                    nodes.push(DefinitionNode::Citation(Self::citation_text(content)));
                } else if Self::is_example(map) {
                    let jp = map
                        .get("japanese")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| content.and_then(|v| v.as_str()).map(|s| s.to_string()));
                    let en = map
                        .get("english")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    if let Some(jp) = jp {
                        nodes.push(DefinitionNode::Example(ExampleNode {
                            japanese: Some(jp),
                            english: en,
                            ..Default::default()
                        }));
                    } else {
                        let parts = Self::example_parts(content);
                        if let Some(parts) = parts {
                            nodes.push(DefinitionNode::Example(ExampleNode {
                                parts,
                                ..Default::default()
                            }));
                        } else {
                            nodes.push(DefinitionNode::Example(ExampleNode {
                                content: content
                                    .map(|c| Self::parse_definition(c, "\n"))
                                    .unwrap_or_default(),
                                ..Default::default()
                            }));
                        }
                    }
                } else if let Some(glyph) = sc_class
                    .as_deref()
                    .and_then(|c| Self::FORM_MARKERS.iter().find(|(k, _)| *k == c))
                    .map(|(_, g)| *g)
                {
                    nodes.push(DefinitionNode::Text(glyph.to_string()));
                } else if sc_class.as_deref() == Some("tag") {
                    let text = content
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    nodes.push(DefinitionNode::Tag { text });
                } else if tag == Some("span")
                    && sc_content.as_deref().map_or(false, |s| s.ends_with("-info"))
                {
                    let text = content
                        .map(|v| {
                            if let Some(s) = v.as_str() {
                                s.to_string()
                            } else {
                                v.to_string()
                            }
                        })
                        .unwrap_or_default();
                    nodes.push(DefinitionNode::Tag { text });
                } else if tag == Some("ruby") {
                    if let Some(serde_json::Value::Array(ruby_list)) = content {
                        if ruby_list.len() >= 2 {
                            let term = ruby_list[0]
                                .as_str()
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| ruby_list[0].to_string());
                            let reading = ruby_list
                                .get(1)
                                .and_then(|v| v.as_object())
                                .and_then(|m| m.get("content"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            nodes.push(DefinitionNode::Ruby { term, reading });
                        } else if let Some(c) = content {
                            // Fail open: a malformed ruby still shows its content.
                            Self::parse_definition_into(c, separator, nodes);
                        }
                    } else if let Some(c) = content {
                        Self::parse_definition_into(c, separator, nodes);
                    }
                } else if tag == Some("table") {
                    // Rows -> cells, real content this time (#88). Each cell is
                    // itself structured content, so a cell can carry ruby.
                    let rows: Vec<Vec<Vec<DefinitionNode>>> = content
                        .and_then(|c| c.as_array())
                        .map(|rows| {
                            rows.iter()
                                .map(|row| {
                                    let cells = row
                                        .as_object()
                                        .and_then(|m| m.get("content"))
                                        .unwrap_or(row);
                                    cells
                                        .as_array()
                                        .map(|cells| {
                                            cells
                                                .iter()
                                                .map(|cell| {
                                                    // Parse the whole cell, not
                                                    // just its content, so a
                                                    // form-validity class on
                                                    // the `td` is seen.
                                                    Self::parse_definition(cell, separator)
                                                })
                                                .collect()
                                        })
                                        .unwrap_or_default()
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    if !rows.is_empty() {
                        nodes.push(DefinitionNode::Table { rows });
                    } else if let Some(c) = content {
                        // Fail open: an unshaped table still shows its content.
                        Self::parse_definition_into(c, separator, nodes);
                    }
                } else if tag == Some("ul") || tag == Some("ol") {
                    if Self::INLINE_LIST_CLASSES.contains(&sc_content.as_deref().unwrap_or("")) {
                        if let Some(c) = content {
                            Self::parse_definition_into(c, separator, nodes);
                        }
                    } else {
                        let items: Vec<Vec<DefinitionNode>> = content
                            .and_then(|c| c.as_array())
                            .map(|items| {
                                items
                                    .iter()
                                    .map(|item| Self::parse_definition(item, separator))
                                    .collect()
                            })
                            .unwrap_or_default();
                        if !items.is_empty() {
                            nodes.push(DefinitionNode::ListBlock {
                                items,
                                list_type: sc_content,
                            });
                        } else if let Some(c) = content {
                            Self::parse_definition_into(c, separator, nodes);
                        }
                    }
                } else if sc_content
                    .as_deref()
                    .map_or(false, |c| Self::BOXED_CONTENT_CLASSES.contains(&c))
                {
                    // #88 follow-up: an extra-info box gets a block of its own,
                    // so it starts a new line instead of being spliced into the
                    // sense's inline text run.
                    let inner = content
                        .map(|c| Self::parse_definition(c, separator))
                        .unwrap_or_default();
                    if !inner.is_empty() {
                        nodes.push(DefinitionNode::Group {
                            nodes: inner,
                            is_inline: false,
                        });
                    }
                } else if let Some(c) = content {
                    // Fail open: any other tag contributes its content.
                    Self::parse_definition_into(c, separator, nodes);
                }
            }
            _ => {}
        }
    }

    /// #88: split a Jitendex example box into its Japanese and English parts.
    /// Returns None when the box does not have that shape.
    fn example_parts(content: Option<&serde_json::Value>) -> Option<Vec<Vec<DefinitionNode>>> {
        let children = content?.as_array()?;
        let parts: Vec<Vec<DefinitionNode>> = children
            .iter()
            .filter_map(|child| {
                let map = child.as_object()?;
                match Self::get_attr(map, "content").as_deref() {
                    Some("example-sentence-a") | Some("example-sentence-b") => {
                        Some(Self::parse_definition(
                            map.get("content").unwrap_or(child),
                            "",
                        ))
                    }
                    _ => None,
                }
            })
            .collect();
        if parts.is_empty() {
            None
        } else {
            Some(parts)
        }
    }

    /// #88: a row's glossary, split for numbering.
    fn parse_glossary(data: &[serde_json::Value]) -> ParsedGlossary {
        let mut groups: Vec<ParsedSenseGroup> = Vec::new();
        let mut trailing: Vec<DefinitionNode> = Vec::new();

        fn walk(
            node: &serde_json::Value,
            groups: &mut Vec<ParsedSenseGroup>,
            trailing: &mut Vec<DefinitionNode>,
        ) {
            match node {
                serde_json::Value::Array(arr) => {
                    for item in arr {
                        walk(item, groups, trailing);
                    }
                }
                serde_json::Value::Object(map) => {
                    match OcrOverlayState::content_class(node).as_deref() {
                        Some("sense-group") => {
                            groups.push(OcrOverlayState::split_sense_group(node));
                        }
                        Some("sense") => {
                            groups.push(ParsedSenseGroup {
                                header: Vec::new(),
                                senses: vec![OcrOverlayState::parse_definition(node, ", ")],
                                trailing: Vec::new(),
                            });
                        }
                        Some("forms") | Some("attribution") => {
                            trailing.extend(OcrOverlayState::parse_definition(node, ", "));
                        }
                        _ => {
                            if let Some(content) = map.get("content") {
                                walk(content, groups, trailing);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        walk(
            &serde_json::Value::Array(data.to_vec()),
            &mut groups,
            &mut trailing,
        );

        if groups.is_empty() {
            ParsedGlossary {
                structured: false,
                groups: Vec::new(),
                plain: Self::parse_definition(&serde_json::Value::Array(data.to_vec()), ", "),
                trailing: Vec::new(),
            }
        } else {
            ParsedGlossary {
                structured: true,
                groups,
                plain: Vec::new(),
                trailing,
            }
        }
    }

    /// Split one Jitendex `sense-group` into its shared header, one entry per
    /// `sense` child, and any forms/attribution block inside it. An unexpected
    /// shape falls back to the whole group as a single sense, so nothing is
    /// dropped.
    fn split_sense_group(sense_group: &serde_json::Value) -> ParsedSenseGroup {
        let mut header: Vec<DefinitionNode> = Vec::new();
        let mut senses: Vec<Vec<DefinitionNode>> = Vec::new();
        let mut trailing: Vec<DefinitionNode> = Vec::new();

        fn walk(
            node: &serde_json::Value,
            header: &mut Vec<DefinitionNode>,
            senses: &mut Vec<Vec<DefinitionNode>>,
            trailing: &mut Vec<DefinitionNode>,
        ) {
            match node {
                serde_json::Value::Array(arr) => {
                    for item in arr {
                        walk(item, header, senses, trailing);
                    }
                }
                serde_json::Value::Object(map) => {
                    match OcrOverlayState::content_class(node).as_deref() {
                        Some("sense") => {
                            senses.push(OcrOverlayState::parse_definition(node, ", "));
                        }
                        Some("forms") | Some("attribution") => {
                            trailing.extend(OcrOverlayState::parse_definition(node, ", "));
                        }
                        Some(_) => {
                            header.extend(OcrOverlayState::parse_definition(node, ", "));
                        }
                        None => {
                            // A wrapper node may carry its single child as an
                            // object rather than a one-element array — Jitendex
                            // emits `ol` with one `li[sense]` this way. Descend
                            // into either shape; treating the object form as
                            // header copy put the sense (and its example) in
                            // the unnumbered header and rendered it again as
                            // the fallback sense.
                            let content = map.get("content");
                            if matches!(
                                content,
                                Some(serde_json::Value::Array(_)) | Some(serde_json::Value::Object(_))
                            ) {
                                walk(content.unwrap(), header, senses, trailing);
                            } else {
                                header.extend(OcrOverlayState::parse_definition(node, ", "));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(content) = sense_group
            .as_object()
            .and_then(|m| m.get("content"))
        {
            walk(content, &mut header, &mut senses, &mut trailing);
        }

        if senses.is_empty() {
            senses.push(Self::parse_definition(sense_group, ", "));
        }
        ParsedSenseGroup {
            header,
            senses,
            trailing,
        }
    }

    /// The plain text of a Jitendex `attribution` block: its link labels joined
    /// in order (`JMdict`, or `JMdict | Tatoeba`).
    fn citation_text(content: Option<&serde_json::Value>) -> String {
        fn walk(node: &serde_json::Value, parts: &mut String) {
            match node {
                serde_json::Value::String(s) => parts.push_str(s),
                serde_json::Value::Array(arr) => {
                    for item in arr {
                        walk(item, parts);
                    }
                }
                serde_json::Value::Object(map) => {
                    if let Some(c) = map.get("content") {
                        walk(c, parts);
                    }
                }
                _ => {}
            }
        }
        let mut parts = String::new();
        if let Some(c) = content {
            walk(c, &mut parts);
        }
        parts
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
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

    /// #88: true only for an actual example container. The old test matched
    /// any `data-content` containing "example", which also caught Jitendex's
    /// `example-sentence-a`/`-b` divisors and `example-keyword` spans and
    /// wrapped each in its own nested box.
    fn is_example(data: &serde_json::Map<String, serde_json::Value>) -> bool {
        let sc_content = Self::get_attr(data, "content");
        data.get("type")
            .and_then(|v| v.as_str())
            .map_or(false, |t| t == "sentence" || t == "example")
            || data.contains_key("japanese")
            || sc_content.as_deref() == Some("example-sentence")
            || sc_content.as_deref() == Some("examples")
            || Self::get_attr(data, "class").map_or(false, |c| c.contains("example"))
    }

    fn is_inline_node(node: &DefinitionNode) -> bool {
        matches!(
            node,
            DefinitionNode::Text(_)
                | DefinitionNode::Ruby { .. }
                | DefinitionNode::Tag { .. }
        )
    }

    /// Check whether a JSON value represents a block-level element (as opposed
    /// to inline). Mirrors the Kotlin `isBlock` helper.
    fn is_block(data: &serde_json::Value) -> bool {
        match data {
            serde_json::Value::Array(arr) => arr.iter().any(|item| Self::is_block(item)),
            serde_json::Value::Object(map) => {
                if Self::is_example(map) {
                    return true;
                }
                let tag = map.get("tag").and_then(|v| v.as_str());
                let sc_content = Self::get_attr(map, "content");

                if tag == Some("table") {
                    return true;
                }
                // Lists are blocks unless they are one of the inline
                // gloss/reference enumerations.
                if tag == Some("ul") || tag == Some("ol") {
                    return !Self::INLINE_LIST_CLASSES
                        .contains(&sc_content.as_deref().unwrap_or(""));
                }
                if let Some(cls) = sc_content.as_deref() {
                    if Self::BLOCK_CONTENT_CLASSES.contains(&cls) {
                        return true;
                    }
                }

                let content = map.get("content").or_else(|| map.get("list"));
                content.map_or(false, |c| Self::is_block(c))
            }
            _ => false,
        }
    }

    /// The matched surface text, taken across the whole OCR stream rather than
    /// only the tapped line, so a word that crosses a line boundary keeps its
    /// trailing kanji lookups (Android slices `activeAllChars`).
    pub fn matched_term_at(&self, global_idx: usize, term_len: usize) -> String {
        self.active_all_chars
            .iter()
            .skip(global_idx)
            .filter(|c| c.as_str() != "\u{3000}")
            .take(term_len)
            .cloned()
            .collect()
    }

    /// #65: JMdict pointer entries (variant spellings) carry only `?query=`
    /// links; [extract_redirect_targets] returns those headwords so lookup can
    /// resolve them. Entries with any real definitional content yield nothing
    /// (their `see also` links are not redirects). Mirrors
    /// `DictionaryRedirects.extractTargets`.
    pub fn extract_redirect_targets(definitions_json: &str, max_targets: usize) -> Vec<String> {
        let Ok(root) = serde_json::from_str::<serde_json::Value>(definitions_json) else {
            return Vec::new();
        };
        let mut targets: Vec<String> = Vec::new();
        let mut has_definitional = false;

        fn visit(
            node: &serde_json::Value,
            targets: &mut Vec<String>,
            has_definitional: &mut bool,
        ) {
            match node {
                serde_json::Value::Object(o) => {
                    if o.get("tag").and_then(|v| v.as_str()) == Some("a") {
                        let href = o.get("href").and_then(|v| v.as_str()).unwrap_or("");
                        let raw = href
                            .split("?query=")
                            .nth(1)
                            .unwrap_or("")
                            .split('&')
                            .next()
                            .unwrap_or("");
                        if !raw.is_empty() {
                            let target = percent_decode(raw);
                            if !target.trim().is_empty() {
                                targets.push(target);
                            }
                        }
                    }
                    if let Some(dc) = o
                        .get("data")
                        .and_then(|d| d.as_object())
                        .and_then(|d| d.get("content"))
                        .and_then(|v| v.as_str())
                    {
                        if dc != "references" && dc != "refGlosses" {
                            *has_definitional = true;
                        }
                    }
                    for value in o.values() {
                        visit(value, targets, has_definitional);
                    }
                }
                serde_json::Value::Array(arr) => {
                    for value in arr {
                        visit(value, targets, has_definitional);
                    }
                }
                _ => {}
            }
        }

        visit(&root, &mut targets, &mut has_definitional);
        if has_definitional {
            return Vec::new();
        }
        targets.dedup();
        targets.truncate(max_targets);
        targets
    }

    /// Downstep positions from a stored pitch payload, or None when the entry
    /// is not pitch data. Detection is by payload shape
    /// (`{"reading":…, "pitches":[{"position":N},…]}`).
    pub fn pitch_positions_of(definitions_json: &str) -> Option<Vec<i32>> {
        let root: serde_json::Value = serde_json::from_str(definitions_json).ok()?;
        let obj = root.as_object()?;
        let pitches = obj.get("pitches")?;
        let pitches = pitches.as_array()?;
        obj.get("reading")?;
        let mut out: Vec<i32> = Vec::new();
        for p in pitches {
            let Some(position) = p
                .as_object()
                .and_then(|m| m.get("position"))
                .and_then(|v| v.as_i64())
            else {
                continue;
            };
            out.push(position as i32);
        }
        out.sort();
        out.dedup();
        Some(out)
    }

    /// Reading of a stored pitch payload (None when absent).
    pub fn pitch_reading_of(definitions_json: &str) -> Option<String> {
        let root: serde_json::Value = serde_json::from_str(definitions_json).ok()?;
        root.as_object()?
            .get("reading")?
            .as_str()
            .map(|s| s.to_string())
    }
}

/// The parsed form of one database row's glossary. Jitendex rows carry
/// every sense in one payload, so `structured` is true and `groups` holds
/// each `sense-group`; JMdict/KANJIDIC rows keep the pre-#88 flat shape.
struct ParsedGlossary {
    structured: bool,
    groups: Vec<ParsedSenseGroup>,
    plain: Vec<DefinitionNode>,
    trailing: Vec<DefinitionNode>,
}

/// One Jitendex `sense-group` split for numbering.
struct ParsedSenseGroup {
    header: Vec<DefinitionNode>,
    senses: Vec<Vec<DefinitionNode>>,
    trailing: Vec<DefinitionNode>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::component_table::ComponentTable;

    // ---------------------------------------------------------------------
    // Pinned Jitendex fixture (the executable spec from Android's
    // `JitendexStructuredContentTest`).
    // ---------------------------------------------------------------------

    fn fixtures() -> serde_json::Value {
        serde_json::from_str(include_str!("../../tests/data/jitendex/entries.json")).unwrap()
    }

    fn fixture_for(term: &str, reading: &str) -> serde_json::Value {
        for e in fixtures()["entries"].as_array().unwrap() {
            if e["term"].as_str() == Some(term) && e["reading"].as_str() == Some(reading) {
                return e["definitions"].clone();
            }
        }
        panic!("no fixture for {term} ({reading})");
    }

    fn fixture(term: &str) -> serde_json::Value {
        for e in fixtures()["entries"].as_array().unwrap() {
            if e["term"].as_str() == Some(term) {
                return e["definitions"].clone();
            }
        }
        panic!("no fixture for {term}");
    }

    fn parse(term: &str) -> Vec<DefinitionNode> {
        OcrOverlayState::parse_definition(&fixture(term), ", ")
    }

    fn children(node: &DefinitionNode) -> Vec<DefinitionNode> {
        match node {
            DefinitionNode::Group { nodes, .. } => nodes.clone(),
            DefinitionNode::ListBlock { items, .. } => {
                items.iter().flatten().cloned().collect()
            }
            DefinitionNode::Example(ex) => ex
                .content
                .iter()
                .chain(ex.parts.iter().flatten())
                .cloned()
                .collect(),
            DefinitionNode::Table { rows } => {
                rows.iter().flatten().flatten().cloned().collect()
            }
            _ => Vec::new(),
        }
    }

    /// All descendant nodes, depth-first.
    fn flatten(nodes: &[DefinitionNode]) -> Vec<DefinitionNode> {
        let mut out: Vec<DefinitionNode> = nodes.to_vec();
        for n in nodes {
            out.extend(flatten(&children(n)));
        }
        out
    }

    /// Readable tokens in render order: text runs and ruby headwords.
    fn tokens(nodes: &[DefinitionNode]) -> Vec<String> {
        flatten(nodes)
            .iter()
            .filter_map(|node| match node {
                DefinitionNode::Text(t) => Some(t.clone()),
                DefinitionNode::Tag { text } => Some(text.clone()),
                DefinitionNode::Ruby { term, .. } => Some(term.clone()),
                _ => None,
            })
            .collect()
    }

    /// A token, ignoring the inline `", "` the walker appends to the run.
    fn has_token(nodes: &[DefinitionNode], text: &str) -> bool {
        tokens(nodes)
            .iter()
            .any(|t| t == text || t.starts_with(&format!("{text},")))
    }

    fn entry(term: &str, reading: &str, definitions: &serde_json::Value) -> DictionaryEntry {
        DictionaryEntry {
            id: 1,
            kanji: term.to_string(),
            reading: reading.to_string(),
            definitions: serde_json::to_string(definitions).unwrap(),
            rules: String::new(),
            popularity: 0,
            dictionary_id: 1,
            onyomi: None,
            kunyomi: None,
            jlpt: None,
        }
    }

    fn dict_names() -> HashMap<i64, String> {
        [(1i64, "Jitendex".to_string())].into_iter().collect()
    }


    fn formatted_groups(matches: Vec<TermMatch>) -> Vec<FormattedEntry> {
        let mut state = OcrOverlayState::new(1024.0, 768.0);
        state.format_dictionary_results(&matches, &dict_names())
    }

    /// Format one real Jitendex row and return its sense groups.
    fn formatted(term: &str) -> Vec<FormattedSenseGroup> {
        let (reading, defs) = {
            let all = fixtures();
            let e = all["entries"]
                .as_array()
                .unwrap()
                .iter()
                .find(|e| e["term"].as_str() == Some(term))
                .expect("fixture");
            (
                e["reading"].as_str().unwrap().to_string(),
                e["definitions"].clone(),
            )
        };
        let matches = vec![TermMatch {
            term: term.to_string(),
            entries: vec![entry(term, &reading, &defs)],
            chain: None,
        }];
        formatted_groups(matches)
            .into_iter()
            .next()
            .unwrap()
            .reading_groups
            .into_iter()
            .next()
            .unwrap()
            .sense_groups
    }

    // ——— the variant-forms table ————————————————————————————————

    #[test]
    fn a_repeated_reading_glossary_is_not_rendered_twice() {
        let defs = fixture("お前");
        let matches = vec![TermMatch {
            term: "お前".to_string(),
            entries: vec![
                entry("お前", "おまえ", &defs),
                entry("お前", "おまい", &defs),
            ],
            chain: None,
        }];
        let out = formatted_groups(matches);
        let entry = out.single_like();
        let groups = &entry.reading_groups;
        assert_eq!(
            groups.iter().map(|g| g.reading.clone()).collect::<Vec<_>>(),
            vec!["おまえ", "おまい"]
        );
        assert_eq!(
            groups.iter().map(|g| g.render_senses).collect::<Vec<_>>(),
            vec![true, false]
        );

        let examples = groups
            .iter()
            .filter(|g| g.render_senses)
            .flat_map(|g| g.sense_groups.iter())
            .flat_map(|sg| sg.senses.iter())
            .flat_map(|s| flatten(&s.nodes))
            .filter(|n| matches!(n, DefinitionNode::Example(_)))
            .count();
        assert_eq!(examples, 1, "the example must render exactly once");
    }

    /// Jitendex (and JMdict) store one row per headword form with identical
    /// payloads (この/此の/斯の, 九/９/玖): every form lists as a headword, the
    /// senses render once.
    #[test]
    fn identical_headword_rows_render_senses_once() {
        let defs = fixture("お前");
        let matches = vec![TermMatch {
            term: "お前".to_string(),
            entries: vec![
                entry("お前", "おまえ", &defs),
                entry("御前", "おまえ", &defs),
            ],
            chain: None,
        }];
        let entry = formatted_groups(matches).single_like();
        let group = entry.reading_groups.single_ref();
        assert_eq!(
            group.headwords.iter().map(|h| h.kanji.clone()).collect::<Vec<_>>(),
            vec!["お前", "御前"],
            "every headword form still lists"
        );
        let examples = group
            .sense_groups
            .iter()
            .flat_map(|sg| sg.senses.iter())
            .flat_map(|s| flatten(&s.nodes))
            .filter(|n| matches!(n, DefinitionNode::Example(_)))
            .count();
        assert_eq!(examples, 1, "the shared payload renders exactly once");
    }

    #[test]
    fn a_lone_sense_is_not_echoed_into_the_header() {
        let defs = fixture_for("分", "ぶ");
        let matches = vec![TermMatch {
            term: "分".to_string(),
            entries: vec![entry("分", "ぶ", &defs)],
            chain: None,
        }];
        let entry = formatted_groups(matches).single_like();
        let groups = &entry.reading_groups.single_ref().sense_groups;
        let mut outside = 0;
        let mut inside = 0;
        for group in groups {
            outside += flatten(&group.header)
                .iter()
                .filter(|n| matches!(n, DefinitionNode::Example(_)))
                .count();
            outside += flatten(&group.trailing)
                .iter()
                .filter(|n| matches!(n, DefinitionNode::Example(_)))
                .count();
            inside += group
                .senses
                .iter()
                .flat_map(|s| flatten(&s.nodes))
                .filter(|n| matches!(n, DefinitionNode::Example(_)))
                .count();
        }
        assert_eq!(outside, 0, "no example may sit outside a numbered sense");
        assert_eq!(inside, 2);
    }

    #[test]
    fn the_forms_table_parses_into_rows_and_cells() {
        let table = parse("支持杭")
            .into_iter()
            .find_map(|n| match n {
                DefinitionNode::Table { rows } => Some(rows),
                _ => None,
            })
            .expect("table");
        assert_eq!(table.len(), 3);
        assert_eq!(table[0].len(), 2);
        assert_eq!(tokens(&table[0][0]), Vec::<String>::new());
        assert_eq!(tokens(&table[0][1]), vec!["支持杭"]);
        assert_eq!(tokens(&table[1][0]), vec!["しじぐい"]);
        assert_eq!(tokens(&table[2][0]), vec!["しじくい"]);
    }

    #[test]
    fn the_forms_table_draws_the_validity_marker_upstream_draws() {
        let table = parse("支持杭")
            .into_iter()
            .find_map(|n| match n {
                DefinitionNode::Table { rows } => Some(rows),
                _ => None,
            })
            .expect("table");
        assert_eq!(tokens(&table[1][1]), vec!["◇"]);
        assert_eq!(tokens(&table[2][1]), vec!["◇"]);
    }

    #[test]
    fn a_forms_section_can_also_be_a_plain_list() {
        let block = flatten(&parse("ダイニングキッチン"))
            .into_iter()
            .filter_map(|n| match n {
                DefinitionNode::ListBlock { items, list_type } => Some((items, list_type)),
                _ => None,
            })
            .find(|(_, t)| t.as_deref().map_or(true, |t| t == "forms"))
            .expect("forms list");
        assert_eq!(
            block
                .0
                .iter()
                .map(|item| tokens(item).first().cloned().unwrap_or_default())
                .collect::<Vec<_>>(),
            vec!["ダイニングキッチン", "ダイニング・キッチン"]
        );
    }

    // ——— examples ————————————————————————————————————————————————

    #[test]
    fn a_jitendex_example_splits_into_japanese_and_english_parts() {
        let example = flatten(&parse("湾内"))
            .into_iter()
            .find_map(|n| match n {
                DefinitionNode::Example(ex) => Some(ex),
                _ => None,
            })
            .expect("example");
        assert_eq!(example.parts.len(), 2);
        assert!(example.parts[0]
            .iter()
            .filter_map(|n| match n {
                DefinitionNode::Ruby { term, .. } => Some(term.clone()),
                _ => None,
            })
            .any(|t| t == "湾"));
        assert!(tokens(&example.parts[0]).join("").contains("えられた。"));
        assert!(tokens(&example.parts[1])
            .join(" ")
            .contains("privilege of fishing in this bay"));
    }

    // ——— cross references / antonyms ——————————————————————————

    #[test]
    fn cross_references_render_as_text_without_comma_noise() {
        let t = tokens(&parse("湾内"));
        assert!(t.contains(&"See also".to_string()));
        assert!(t.contains(&"beyond the bay".to_string()));
        assert!(
            !t.iter().any(|x| x.contains(", ")),
            "ruby runs must not be comma-spliced: {t:?}"
        );
    }

    #[test]
    fn an_extra_info_box_starts_its_own_line() {
        let block = flatten(&parse("湾内"))
            .into_iter()
            .find(|n| match n {
                DefinitionNode::Group { nodes, is_inline } => {
                    !is_inline && has_token(nodes, "See also")
                }
                _ => false,
            });
        assert!(
            block.is_some(),
            "the cross reference must be a block of its own"
        );
    }

    #[test]
    fn antonyms_render_as_text() {
        let nodes = parse("ディフェンシブ");
        assert!(has_token(&nodes, "Antonym"));
        assert!(has_token(&nodes, "オフェンシブ"));
        assert!(has_token(&nodes, "offensive"));
    }

    // ——— notes and source-language info ———————————————————————

    #[test]
    fn a_sense_note_is_a_block_so_it_is_not_comma_joined_to_its_neighbour() {
        let t = tokens(&parse("あかんべえ"));
        assert!(t.contains(&"from 赤目".to_string()));
        let note_index = t.iter().position(|x| x == "from 赤目").unwrap();
        assert_eq!(t[note_index + 1], "See also");
    }

    #[test]
    fn source_language_info_renders_its_label_body_and_tags() {
        let nodes = parse("ダイニングキッチン");
        assert!(has_token(&nodes, "Language of Origin"));
        assert!(has_token(&nodes, "English: \"dining kitchen\""));
        assert!(has_token(&nodes, "wasei"));
    }

    #[test]
    fn a_definition_note_renders_its_label_and_body() {
        let nodes = parse("袋小路文");
        assert!(has_token(&nodes, "Literally"));
        assert!(has_token(&nodes, "cul-de-sac sentence"));
    }

    #[test]
    fn the_source_line_is_a_citation_node_with_its_labels_joined() {
        let citation = flatten(&parse("湾内"))
            .into_iter()
            .find_map(|n| match n {
                DefinitionNode::Citation(text) => Some(text),
                _ => None,
            })
            .expect("citation");
        assert_eq!(citation, "JMdict | Tatoeba");
        let citation = flatten(&parse("支持杭"))
            .into_iter()
            .find_map(|n| match n {
                DefinitionNode::Citation(text) => Some(text),
                _ => None,
            })
            .expect("citation");
        assert_eq!(citation, "JMdict");
    }

    #[test]
    fn the_sense_groups_list_keeps_its_senses() {
        let (items, list_type) = flatten(&parse("あかんべえ"))
            .into_iter()
            .find_map(|n| match n {
                DefinitionNode::ListBlock { items, list_type } => Some((items, list_type)),
                _ => None,
            })
            .expect("sense-groups list");
        assert_eq!(list_type.as_deref(), Some("sense-groups"));
        assert_eq!(items.len(), 3);
        let nodes = parse("あかんべえ");
        assert!(has_token(&nodes, "no way!"));
        assert!(has_token(&nodes, "get lost!"));
        assert!(has_token(&nodes, "あっかんべー"));
    }

    // ——— sense numbering ————————————————————————————————————————

    #[test]
    fn senses_are_numbered_across_a_jitendex_entry_not_all_under_1() {
        assert_eq!(
            formatted("あかんべえ")
                .iter()
                .flat_map(|g| g.senses.iter())
                .map(|s| s.index)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            formatted("いじらしい")
                .single_like()
                .senses
                .iter()
                .map(|s| s.index)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn jitendex_group_metadata_is_a_header_and_forms_trail_the_senses() {
        let group = formatted("支持杭").single_like();
        assert!(!group.header.is_empty());
        assert!(has_token(&group.header, "noun"));
        assert_eq!(group.senses.len(), 1);
        assert!(!group.trailing.is_empty());
        assert!(flatten(&group.trailing)
            .iter()
            .any(|n| matches!(n, DefinitionNode::Table { .. })));
    }

    #[test]
    fn a_plain_row_still_yields_a_single_sense() {
        // A bare-string definition payload is one sense, not zero (fail open).
        let defs = serde_json::json!("other");
        let matches = vec![TermMatch {
            term: "他".to_string(),
            entries: vec![entry("他", "た", &defs)],
            chain: None,
        }];
        let entry = formatted_groups(matches).single_like();
        let senses: Vec<&FormattedSense> = entry
            .reading_groups
            .single_ref()
            .sense_groups
            .iter()
            .flat_map(|g| g.senses.iter())
            .collect();
        assert_eq!(senses.iter().map(|s| s.index).collect::<Vec<_>>(), vec![1]);
        assert!(!senses[0].nodes.is_empty());
    }

    // ——— fail open ————————————————————————————————————————————

    #[test]
    fn an_unknown_tag_renders_its_content_rather_than_dropping_it() {
        let nodes = OcrOverlayState::parse_definition(
            &serde_json::json!({"tag":"mark","content":"kept"}),
            ", ",
        );
        assert_eq!(nodes, vec![DefinitionNode::Text("kept".to_string())]);
    }

    #[test]
    fn an_unknown_node_inside_a_sense_never_blanks_the_sense() {
        let json = serde_json::json!([{"type":"structured-content","content":[
            {"tag":"div","data":{"content":"sense"},"content":[
                {"tag":"ul","data":{"content":"glossary"},"content":{"tag":"li","content":"a gloss"}},
                {"tag":"future-widget","content":"future text"}
            ]}
        ]}]);
        let nodes = OcrOverlayState::parse_definition(&json, ", ");
        assert!(has_token(&nodes, "a gloss"));
        assert!(has_token(&nodes, "future text"));
    }

    #[test]
    fn a_malformed_ruby_still_renders_its_content() {
        let nodes = OcrOverlayState::parse_definition(
            &serde_json::json!({"tag":"ruby","content":"just text"}),
            ", ",
        );
        assert_eq!(nodes, vec![DefinitionNode::Text("just text".to_string())]);
    }

    #[test]
    fn an_unshaped_table_still_renders_its_content() {
        let nodes = OcrOverlayState::parse_definition(
            &serde_json::json!({"tag":"table","content":"not rows"}),
            ", ",
        );
        assert_eq!(nodes, vec![DefinitionNode::Text("not rows".to_string())]);
    }

    // ——— dictionary grouping and lookup plumbing ————————————————

    #[test]
    fn entries_are_split_per_dictionary_with_a_source_label() {
        // "One entry per (term, dictionary): JMdict and KANJIDIC rows must
        // never merge into a single block."
        let defs = serde_json::json!(["gloss"]);
        let mut jm = entry("分", "ぶん", &defs);
        jm.dictionary_id = 1;
        let mut kj = entry("分", "ぶん", &defs);
        kj.dictionary_id = 2;
        kj.onyomi = Some("ブン".to_string());
        let matches = vec![TermMatch {
            term: "分".to_string(),
            entries: vec![jm, kj],
            chain: None,
        }];
        let mut names = HashMap::new();
        names.insert(1i64, "JMdict".to_string());
        names.insert(2i64, "KANJIDIC".to_string());
        let mut state = OcrOverlayState::new(1024.0, 768.0);
        let out = state.format_dictionary_results(&matches, &names);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].dictionary_name.as_deref(), Some("JMdict"));
        assert_eq!(out[1].dictionary_name.as_deref(), Some("KANJIDIC"));
    }

    #[test]
    fn gravity_follows_the_android_halves() {
        let mut s = OcrOverlayState::new(1000.0, 800.0);
        s.img_w = 1000;
        s.img_h = 800;
        // Landscape: left half -> END, right half -> START.
        s.update_gravity(&BoundingBox::new(100, 300, 100, 100, 1.0), 388.0);
        assert_eq!(s.last_landscape_gravity, Gravity::End);
        s.update_gravity(&BoundingBox::new(900, 300, 100, 100, 1.0), 388.0);
        assert_eq!(s.last_landscape_gravity, Gravity::Start);

        let mut p = OcrOverlayState::new(800.0, 1000.0);
        p.img_w = 800;
        p.img_h = 1000;
        // Portrait: top half -> BOTTOM, bottom half -> TOP.
        p.update_gravity(&BoundingBox::new(300, 100, 100, 100, 1.0), 388.0);
        assert_eq!(p.last_portrait_gravity, Gravity::Bottom);
        p.update_gravity(&BoundingBox::new(300, 900, 100, 100, 1.0), 388.0);
        assert_eq!(p.last_portrait_gravity, Gravity::Top);
    }

    // ——— redirects, cross-line lookup and chains ————————————————

    // Verbatim shapes from Android's `DictionaryRedirectsTest`.
    const PURE_SINGLE: &str = r#"[{"content": {"content": ["⟶", {"content": "あかん", "href": "?query=あかん&wildcards=off", "lang": "ja", "tag": "a"}], "style": {"fontSize": "130%"}, "tag": "span"}, "type": "structured-content"}]"#;
    const PURE_DUAL: &str = r#"[{"content": {"content": ["⟶", {"content": "阿呆陀羅", "href": "?query=阿呆陀羅&wildcards=off", "lang": "ja", "tag": "a"}, "（", {"content": "あほんだら", "href": "?query=あほんだら&wildcards=off", "lang": "ja", "tag": "a"}, "）"], "style": {"fontSize": "130%"}, "tag": "span"}, "type": "structured-content"}]"#;
    const GLOSS_WITH_REFS: &str = r#"[{"content": [{"content": {"content": "repetition mark in katakana", "tag": "li"}, "data": {"content": "glossary"}, "lang": "en", "style": {"listStyleType": "circle"}, "tag": "ul"}, {"content": {"content": ["see: ", {"content": "一の字点", "href": "?query=一の字点&wildcards=off", "lang": "ja", "tag": "a"}, {"content": " kana iteration mark", "data": {"content": "refGlosses"}, "style": {"fontSize": "65%", "verticalAlign": "middle"}, "tag": "span"}], "tag": "li"}, "data": {"content": "references"}, "lang": "en", "style": {"listStyleType": "'➡️ '"}, "tag": "ul"}], "type": "structured-content"}]"#;

    #[test]
    fn pure_single_redirect_resolves() {
        assert_eq!(
            OcrOverlayState::extract_redirect_targets(PURE_SINGLE, 3),
            vec!["あかん"]
        );
    }

    #[test]
    fn pure_dual_redirect_resolves_both() {
        assert_eq!(
            OcrOverlayState::extract_redirect_targets(PURE_DUAL, 3),
            vec!["阿呆陀羅", "あほんだら"]
        );
    }

    #[test]
    fn gloss_with_see_also_does_not_redirect() {
        assert!(OcrOverlayState::extract_redirect_targets(GLOSS_WITH_REFS, 3).is_empty());
    }

    #[test]
    fn plain_gloss_does_not_redirect() {
        assert!(OcrOverlayState::extract_redirect_targets("\"ただの定義\"", 3).is_empty());
    }

    #[test]
    fn malformed_and_empty_redirects_yield_nothing() {
        for json in ["not json{[", "", "[]"] {
            assert!(OcrOverlayState::extract_redirect_targets(json, 3).is_empty());
        }
    }

    #[test]
    fn redirect_target_cap_applies() {
        assert_eq!(
            OcrOverlayState::extract_redirect_targets(PURE_DUAL, 1).len(),
            1
        );
    }

    #[test]
    fn lookup_splices_redirect_targets_below_their_source() {
        let dir = std::env::temp_dir().join(format!("ijd_redirect_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db = DictionaryDatabase::open(dir.join("d.db")).unwrap();
        let did = db.insert_dictionary("JMdict", 0).unwrap();
        db.insert_entries(&[
            DictionaryEntry::new(
                "あかーん".into(),
                "あかーん".into(),
                PURE_SINGLE.to_string(),
                String::new(),
                0,
                did,
            ),
            DictionaryEntry::new(
                "あかん".into(),
                "あかん".into(),
                r#"["to not do"]"#.to_string(),
                String::new(),
                0,
                did,
            ),
        ])
        .unwrap();

        let mut state = OcrOverlayState::new(1024.0, 768.0);
        state.active_line_results = vec![Some(LineResult {
            text: "あかーん".into(),
            char_boxes: (0..4)
                .map(|i| BoundingBox::new(i * 20, 0, 20, 20, 1.0))
                .collect(),
            alternatives: vec![],
            raw_alternatives: vec![],
            sample_txt: None,
            is_vertical: false,
            chunk_boxes: vec![],
        })];
        state.update_global_data();

        let result = state
            .lookup(0, 0, &db, &Deinflector::empty())
            .expect("lookup result");
        let terms: Vec<&str> = result.matches.iter().map(|e| e.term.as_str()).collect();
        assert_eq!(terms, vec!["あかーん", "あかん"]);
        let target = result
            .matches
            .iter()
            .find(|e| e.term == "あかん")
            .expect("resolved target");
        let chain = target.deinflection.as_ref().expect("redirect chain");
        assert_eq!(chain.surface, "あかーん");
        assert_eq!(chain.steps, vec!["redirect"]);
    }

    #[test]
    fn deinflected_matches_carry_their_chain() {
        let deinflector = Deinflector::from_json_file(concat!(env!("CARGO_MANIFEST_DIR"), "/../assets/deinflect.json")).unwrap();
        let state = OcrOverlayState::new(1024.0, 768.0);
        let (terms, by_length) = state.prepare_search_candidates("食べた", &deinflector);
        assert!(terms.contains("食べる"), "terms: {terms:?}");
        let chain = by_length
            .iter()
            .flat_map(|(_, candidates)| candidates.iter())
            .find_map(|(term, _, chain)| {
                if term == "食べる" {
                    chain.clone()
                } else {
                    None
                }
            })
            .expect("a deinflected 食べる candidate");
        assert_eq!(chain.surface, "食べた");
        assert_eq!(chain.steps, vec!["past"], "steps carry reason labels, not kana");
    }

    #[test]
    fn matched_term_spans_line_boundaries() {
        let mut s = OcrOverlayState::new(1024.0, 768.0);
        s.active_all_chars = ["日", "本", "語", "を", "学", "ぶ"]
            .iter()
            .map(|c| c.to_string())
            .collect();
        assert_eq!(s.matched_term_at(1, 3), "本語を");
        // Full-width spaces are skipped, matching the filtered search text.
        s.active_all_chars = ["分", "\u{3000}", "野"]
            .iter()
            .map(|c| c.to_string())
            .collect();
        assert_eq!(s.matched_term_at(0, 3), "分野");
    }

    // ——— small helpers to keep the assertions above readable ————

    trait SingleLike<T> {
        fn single_like(self) -> T;
    }
    impl<T> SingleLike<T> for Vec<T> {
        fn single_like(mut self) -> T {
            assert_eq!(self.len(), 1, "expected exactly one element");
            self.remove(0)
        }
    }
    trait SingleRef<T> {
        fn single_ref(&self) -> &T;
    }
    impl<T> SingleRef<T> for Vec<T> {
        fn single_ref(&self) -> &T {
            assert_eq!(self.len(), 1, "expected exactly one element");
            &self[0]
        }
    }

    // ---------------------------------------------------------------------
    // Blank alternatives (mobile `BlankAlternativesTest`, #44).
    // ---------------------------------------------------------------------

    /// A line with a placeholder and the parallel lists a real detection
    /// would carry.
    fn blank_line(
        text: &str,
        alternatives: Vec<Vec<(char, f32)>>,
        raw: Vec<Vec<(char, f32)>>,
    ) -> LineResult {
        LineResult {
            text: text.into(),
            char_boxes: vec![],
            alternatives,
            raw_alternatives: raw,
            sample_txt: None,
            is_vertical: true,
            chunk_boxes: vec![],
        }
    }

    fn state_at(line: LineResult, char_idx: isize) -> OcrOverlayState {
        let mut state = OcrOverlayState::new(1024.0, 768.0);
        state.active_line_results = vec![Some(line)];
        state.current_tapped_line_idx = 0;
        state.current_tapped_char_idx_in_line = char_idx;
        state
    }

    /// Mobile `BlankAlternativesTest.a_blank_offers_more_than_the_placeholder_even_when_the_table_holds_it`:
    /// the alternatives table *does* carry a placeholder entry
    /// (`with_gap_char` writes one), so the ranked path must be taken before
    /// reading that table — the old order made it unreachable and the list
    /// came back as the dotted circle alone.
    #[test]
    fn a_blank_offers_more_than_the_placeholder() {
        let state = state_at(
            blank_line(
                "私\u{25CC}う",
                vec![vec![('私', 1.0)], vec![(GAP_CHAR, 0.0)], vec![('う', 1.0)]],
                vec![vec![('、', 0.7), ('。', 0.5)]],
            ),
            1,
        );
        let ui = state.get_alternatives_ui_state().expect("blank panel");
        let chars: Vec<char> = ui.candidates.iter().map(|c| c.char).collect();
        assert_eq!(chars[0], GAP_CHAR, "the placeholder leads");
        assert!(ui.candidates[0].is_selected);
        assert!(chars.len() > 1, "blank list was {chars:?}");
        assert!(
            chars.contains(&'、'),
            "the timestep evidence is offered: {chars:?}"
        );
    }

    /// The older shape: the table never grew (short `alternatives`), which
    /// used to return no panel at all.
    #[test]
    fn a_blank_offers_more_when_the_table_never_grew() {
        let state = state_at(blank_line("私\u{25CC}う", vec![vec![('私', 1.0)]], vec![]), 1);
        let ui = state.get_alternatives_ui_state().expect("blank panel");
        assert!(ui.candidates.len() > 1, "fallback must not be empty");
    }

    /// With no recogniser evidence the fallback still offers punctuation
    /// first, then kana.
    #[test]
    fn a_blank_without_evidence_falls_back_to_punctuation_then_kana() {
        let state = state_at(blank_line("私\u{25CC}う", vec![], vec![]), 1);
        let ui = state.get_alternatives_ui_state().expect("blank panel");
        let chars: Vec<char> = ui.candidates.iter().map(|c| c.char).collect();
        assert_eq!(chars[0], GAP_CHAR);
        assert_eq!(chars[1], '、', "punctuation first: {chars:?}");
        assert!(chars.contains(&'は'), "then kana: {chars:?}");
    }

    /// An ordinary character still reads its own table entry.
    #[test]
    fn an_ordinary_character_still_gets_its_own_head_list() {
        let state = state_at(
            blank_line(
                "私\u{25CC}う",
                vec![vec![('私', 1.0)], vec![(GAP_CHAR, 0.0)], vec![('う', 1.0)]],
                vec![],
            ),
            0,
        );
        let ui = state.get_alternatives_ui_state().expect("char panel");
        assert_eq!(ui.candidates.len(), 1);
        assert_eq!(ui.candidates[0].char, '私');
        assert!(ui.candidates[0].is_selected);
    }

    /// Filling a blank must not collapse its list: the position keeps the
    /// placeholder plus the evidence candidates, with the chosen character
    /// selected (#44).
    #[test]
    fn filling_a_blank_keeps_its_candidate_list() {
        let mut state = state_at(
            blank_line(
                "私\u{25CC}う",
                vec![vec![('私', 1.0)], vec![(GAP_CHAR, 0.0)], vec![('う', 1.0)]],
                vec![vec![('、', 0.7), ('の', 0.6)]],
            ),
            1,
        );
        state.update_character(0, 1, 'の');
        assert_eq!(
            state.active_line_results[0]
                .as_ref()
                .unwrap()
                .text
                .chars()
                .nth(1),
            Some('の'),
            "the fill lands in the text"
        );
        let ui = state.get_alternatives_ui_state().expect("panel after filling");
        let chars: Vec<char> = ui.candidates.iter().map(|c| c.char).collect();
        assert_eq!(chars[0], GAP_CHAR, "the placeholder stays in the list");
        assert!(
            chars.contains(&'の') && chars.contains(&'、'),
            "the list is kept: {chars:?}"
        );
        let selected: Vec<char> = ui
            .candidates
            .iter()
            .filter(|c| c.is_selected)
            .map(|c| c.char)
            .collect();
        assert_eq!(selected, vec!['の'], "only the filled character is selected");
    }

    /// With a model installed the blank's list is ranked by the line context,
    /// not by the recogniser's own order (the shipped Aozora model; skipped
    /// when the asset is absent).
    #[test]
    fn an_installed_model_ranks_the_blank_list_by_context() {
        let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/..")).join("assets/lm/char_lm.bin");
        let Some(lm) = CharLm::load(&path) else {
            eprintln!("skipping: {} not present", path.display());
            return;
        };
        // The recogniser offered を first; after 今日 the model prefers は.
        let mut state = state_at(
            blank_line(
                "今日\u{25CC}",
                vec![
                    vec![('今', 1.0)],
                    vec![('日', 1.0)],
                    vec![(GAP_CHAR, 0.0)],
                ],
                vec![vec![('を', 0.9)], vec![('は', 0.8)]],
            ),
            2,
        );
        state.install_char_lm(Some(Arc::new(lm)));
        let ui = state.get_alternatives_ui_state().expect("blank panel");
        let chars: Vec<char> = ui.candidates.iter().map(|c| c.char).collect();
        assert_eq!(chars[0], GAP_CHAR);
        assert_eq!(chars[1], 'は', "the context prior leads: {chars:?}");
        assert_eq!(chars[2], 'を');
    }

    // ---------------------------------------------------------------------
    // OOV suggestions (mobile `OovSuggestionsTest`, #44).
    // ---------------------------------------------------------------------

    /// The `OovSuggestionsTest` fixture: 50 kanji, `化` carried by two of
    /// them and `中` by twenty, so a candidate sharing only `化` clears the
    /// 0.7 tier (≈ 0.78) while one sharing only `中` does not (≈ 0.22).
    fn fixture_oov() -> OovCandidates {
        let mut text = String::from("仲:化 中\n伜:化 九 十\n");
        for i in 0..19 {
            text.push(char::from_u32(0x4E00 + i).expect("BMP"));
            text.push_str(":中\n");
        }
        for i in 0..29 {
            text.push(char::from_u32(0x5E00 + i).expect("BMP"));
            text.push_str(":水\n");
        }
        OovCandidates::new(ComponentTable::parse(&text))
    }

    fn panel_chars(ui: &AlternativesUiState) -> Vec<char> {
        ui.candidates.iter().map(|c| c.char).collect()
    }

    /// Mobile `OovSuggestionsTest.component_neighbour_is_appended_after_the_head_list`,
    /// through the panel: a tapped 仲 lists 伜 after the head entry.
    #[test]
    fn a_tapped_character_appends_component_neighbours_after_the_head_list() {
        let mut state = state_at(blank_line("仲", vec![vec![('仲', 1.0)]], vec![]), 0);
        state.install_oov_candidates(Some(Arc::new(fixture_oov())));
        let ui = state.get_alternatives_ui_state().expect("char panel");
        assert_eq!(panel_chars(&ui), vec!['仲', '伜']);
        let selected: Vec<char> =
            ui.candidates.iter().filter(|c| c.is_selected).map(|c| c.char).collect();
        assert_eq!(selected, vec!['仲']);
    }

    /// The variant group is offered last, from the current character:
    /// tapping the canonical 掴 offers its obsolete form 摑, even with no
    /// component table installed.
    #[test]
    fn a_tapped_character_appends_variant_forms_last() {
        let mut state = state_at(blank_line("掴", vec![vec![('掴', 1.0)]], vec![]), 0);
        state.install_kanji_variants(Some(Arc::new(KanjiVariantTable::parse("摑\t掴\n"))));
        let ui = state.get_alternatives_ui_state().expect("char panel");
        assert_eq!(panel_chars(&ui), vec!['掴', '摑']);
    }

    /// An ordinary character with no table entry is unchanged, even with both
    /// tables installed: あ has no components and no variant forms.
    #[test]
    fn a_character_with_no_table_entry_is_unchanged() {
        let mut state = state_at(
            blank_line("あい", vec![vec![('あ', 1.0), ('い', 0.5)]], vec![]),
            0,
        );
        state.install_oov_candidates(Some(Arc::new(fixture_oov())));
        state.install_kanji_variants(Some(Arc::new(KanjiVariantTable::parse("摑\t掴\n"))));
        let ui = state.get_alternatives_ui_state().expect("char panel");
        assert_eq!(panel_chars(&ui), vec!['あ', 'い']);
    }

    /// A blank placeholder still gets the evidence list with no component
    /// expansion, even with both tables installed.
    #[test]
    fn a_blank_gets_no_component_expansion() {
        let mut state = state_at(
            blank_line(
                "私\u{25CC}う",
                vec![vec![('私', 1.0)], vec![(GAP_CHAR, 0.0)], vec![('う', 1.0)]],
                vec![vec![('、', 0.7), ('。', 0.5)]],
            ),
            1,
        );
        state.install_oov_candidates(Some(Arc::new(fixture_oov())));
        state.install_kanji_variants(Some(Arc::new(KanjiVariantTable::parse("摑\t掴\n"))));
        let ui = state.get_alternatives_ui_state().expect("blank panel");
        assert_eq!(panel_chars(&ui), vec![GAP_CHAR, '、', '。']);
    }

    /// The panel carries each entry's source so the tint is truthful: head
    /// evidence reads `Head`, component neighbours and variant forms read
    /// `Components`/`Variant`, and the blank's ranked entries read `Lm` with
    /// a model installed (`Head` on discovery order without one).
    #[test]
    fn panel_entries_carry_their_source_for_the_tint() {
        // Tapped character: head entry, then the component neighbour.
        let mut state = state_at(blank_line("仲", vec![vec![('仲', 1.0)]], vec![]), 0);
        state.install_oov_candidates(Some(Arc::new(fixture_oov())));
        let ui = state.get_alternatives_ui_state().expect("char panel");
        let sources: Vec<(char, Source)> =
            ui.candidates.iter().map(|c| (c.char, c.source)).collect();
        assert_eq!(sources, vec![('仲', Source::Head), ('伜', Source::Components)]);

        // Tapped character: head entry, then the obsolete variant form.
        let mut state = state_at(blank_line("掴", vec![vec![('掴', 1.0)]], vec![]), 0);
        state.install_kanji_variants(Some(Arc::new(KanjiVariantTable::parse("摑\t掴\n"))));
        let ui = state.get_alternatives_ui_state().expect("char panel");
        let sources: Vec<(char, Source)> =
            ui.candidates.iter().map(|c| (c.char, c.source)).collect();
        assert_eq!(sources, vec![('掴', Source::Head), ('摑', Source::Variant)]);

        // Blank without a model: the placeholder and the discovery-order
        // evidence are head entries.
        let mut state = state_at(
            blank_line(
                "私\u{25CC}う",
                vec![vec![('私', 1.0)], vec![(GAP_CHAR, 0.0)], vec![('う', 1.0)]],
                vec![vec![('、', 0.7), ('。', 0.5)]],
            ),
            1,
        );
        let ui = state.get_alternatives_ui_state().expect("blank panel");
        let sources: Vec<(char, Source)> =
            ui.candidates.iter().map(|c| (c.char, c.source)).collect();
        assert_eq!(
            sources,
            vec![(GAP_CHAR, Source::Head), ('、', Source::Head), ('。', Source::Head)]
        );

        // Blank with the shipped model: the ranked entries read `Lm`.
        let path =
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/..")).join("assets/lm/char_lm.bin");
        let Some(lm) = CharLm::load(&path) else {
            eprintln!("skipping LM half: {} not present", path.display());
            return;
        };
        let mut state = state_at(
            blank_line(
                "私\u{25CC}う",
                vec![vec![('私', 1.0)], vec![(GAP_CHAR, 0.0)], vec![('う', 1.0)]],
                vec![vec![('、', 0.7), ('。', 0.5)]],
            ),
            1,
        );
        state.install_char_lm(Some(Arc::new(lm)));
        let ui = state.get_alternatives_ui_state().expect("blank panel");
        assert_eq!(ui.candidates[0].source, Source::Head, "the placeholder");
        assert!(
            ui.candidates[1..].iter().all(|c| c.source == Source::Lm),
            "ranked entries are the model's: {:?}",
            ui.candidates.iter().map(|c| (c.char, c.source)).collect::<Vec<_>>()
        );
    }
}

/// Decode `%XX` escapes (UTF-8 lossily), as `java.net.URLDecoder` does for
/// the redirect hrefs.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 3 <= bytes.len() {
            if let Some(byte) = std::str::from_utf8(&bytes[i + 1..i + 3])
                .ok()
                .and_then(|h| u8::from_str_radix(h, 16).ok())
            {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
