use crate::models::*;

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
            last_landscape_gravity: Gravity::End,
            last_portrait_gravity: Gravity::Bottom,
            is_controller_navigation: false,
            is_dictionary_visible: false,
            is_alternatives_visible: false,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn update_global_data(&mut self) {
        self.active_all_chars.clear();
        self.active_all_alternatives.clear();
        self.active_line_results.iter().flatten().for_each(|line| {
            self.active_all_chars
                .extend(line.text.chars().map(|c| c.to_string()));
            self.active_all_alternatives
                .extend(line.alternatives.clone());
        });
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
        }
    }

    pub fn navigate(&mut self, action: GamepadAction, root_width: f32, root_height: f32) -> bool {
        let (line_idx, char_idx) = match self.current_cursor() {
            Some(coords) => coords,
            None => return false,
        };
        let Some(line) = self
            .active_line_results
            .get(line_idx)
            .and_then(|line| line.as_ref())
        else {
            return false;
        };
        let Some(box_item) = line.char_boxes.get(char_idx) else {
            return false;
        };
        let center_x = box_item.left() as f32 + (box_item.w as f32 / 2.0);
        let center_y = box_item.top() as f32 + (box_item.h as f32 / 2.0);

        let mut best_dist = f32::MAX;
        let mut best_idx = None;
        let mut best_char_idx = None;

        match action {
            GamepadAction::NavigateRight | GamepadAction::NavigateLeft => {
                let dir = if action == GamepadAction::NavigateRight {
                    1
                } else {
                    -1
                };
                let next_char_idx = char_idx as isize + dir;
                if next_char_idx >= 0 && (next_char_idx as usize) < line.char_boxes.len() {
                    self.current_tapped_char_idx_in_line = next_char_idx;
                    self.current_tapped_idx =
                        self.get_global_idx(line_idx, next_char_idx as usize) as isize;
                    return true;
                }

                for (i, other_line_opt) in self.active_line_results.iter().enumerate() {
                    let Some(other_line) = other_line_opt else {
                        continue;
                    };
                    for (c, c_box) in other_line.char_boxes.iter().enumerate() {
                        let mut dx = c_box.left() as f32 + (c_box.w as f32 / 2.0) - center_x;
                        let dy = c_box.top() as f32 + (c_box.h as f32 / 2.0) - center_y;

                        if dir == 1 && dx <= 5.0 {
                            dx += root_width;
                        } else if dir == -1 && dx >= -5.0 {
                            dx -= root_width;
                        }

                        if (dir == 1 && dx <= 5.0) || (dir == -1 && dx >= -5.0) {
                            continue;
                        }

                        let dist = (dx * dx) + (dy * dy * 64.0);
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = Some(i);
                            best_char_idx = Some(c);
                        }
                    }
                }
            }
            GamepadAction::NavigateDown | GamepadAction::NavigateUp => {
                let dir = if action == GamepadAction::NavigateDown {
                    1
                } else {
                    -1
                };
                for (i, other_line_opt) in self.active_line_results.iter().enumerate() {
                    let Some(other_line) = other_line_opt else {
                        continue;
                    };
                    for (c, c_box) in other_line.char_boxes.iter().enumerate() {
                        let dx = c_box.left() as f32 + (c_box.w as f32 / 2.0) - center_x;
                        let mut dy = c_box.top() as f32 + (c_box.h as f32 / 2.0) - center_y;

                        if dir == 1 && dy <= 5.0 {
                            dy += root_height;
                        } else if dir == -1 && dy >= -5.0 {
                            dy -= root_height;
                        }

                        if (dir == 1 && dy <= 5.0) || (dir == -1 && dy >= -5.0) {
                            continue;
                        }

                        let dist = (dx * dx * 64.0) + (dy * dy);
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = Some(i);
                            best_char_idx = Some(c);
                        }
                    }
                }
            }
            _ => {}
        }

        if let (Some(i), Some(c)) = (best_idx, best_char_idx) {
            self.current_tapped_line_idx = i as isize;
            self.current_tapped_char_idx_in_line = c as isize;
            self.current_tapped_idx = self.get_global_idx(i, c) as isize;
            return true;
        }
        false
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

    pub fn panel_dimensions(&self, root_width: f32, root_height: f32) -> (f32, f32) {
        let is_landscape = root_width > root_height;
        let panel_width = if is_landscape {
            root_width * 0.4
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

    pub fn update_gravity(&mut self, root_width: f32, root_height: f32, tapped_box: &BoundingBox) {
        let is_landscape = root_width > root_height;
        let screen_center_x = tapped_box.left() as f32 * self.current_scale
            + self.current_trans_x
            + (tapped_box.w as f32 / 2.0) * self.current_scale;
        let screen_center_y = tapped_box.top() as f32 * self.current_scale
            + self.current_trans_y
            + (tapped_box.h as f32 / 2.0) * self.current_scale;

        if is_landscape {
            self.last_landscape_gravity = if screen_center_x < root_width / 2.0 {
                Gravity::End
            } else {
                Gravity::Start
            };
        } else {
            self.last_portrait_gravity = if screen_center_y < root_height / 2.0 {
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
                let left = (center_x - (fixed_size as f32 / 2.0)).round() as i32;
                let top = (center_y - (fixed_size as f32 / 2.0)).round() as i32;
                BoundingBox::new(left, top, fixed_size, fixed_size, 1.0)
            })
            .collect()
    }
}
