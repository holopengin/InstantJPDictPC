use std::sync::Arc;

use iced::widget::canvas::{
    self, Canvas, Frame, Geometry, Path as CanvasPath, Stroke as CanvasStroke,
    Text as CanvasText,
};
use iced::widget::{
    button, Button, Column, Container, Image as IcedImage, Row, Scrollable, Stack, Text,
};
use iced::widget::container;
use iced::advanced::widget::operation::scrollable::{scroll_to, AbsoluteOffset};
use iced::widget::Id;
use iced::advanced::widget::operate;
use iced::{
    alignment, Color, Element, Font as IcedFont, Length, Pixels, Point, Rectangle, Renderer,
    Size, Theme, mouse,
};

use crate::data::db::DictionaryDatabase;
use crate::models::*;
use crate::overlay_state::OcrOverlayState;
use crate::util::deinflector::Deinflector;

const BOX_FILL_RATIO: f32 = 0.9;

// ---------------------------------------------------------------------------
// OverlayProgram
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct OverlayProgram {
    pub annotations: Vec<DetectedAnnotation>,
    pub img_w: u32,
    pub img_h: u32,
    /// Whether the panel is visible.
    pub panel_visible: bool,
    /// Which side the panel is on.
    pub panel_on_right: bool,
    /// Fixed width of the dictionary panel in pixels.
    pub dict_width: f32,
    /// Whether the alternatives panel is visible.
    pub alternatives_visible: bool,
    /// Current cursor position (line_idx, char_idx) for keyboard navigation.
    pub cursor_pos: Option<(usize, usize)>,
}

impl canvas::Program<Message, Theme, Renderer> for OverlayProgram {
    type State = ();

    fn mouse_interaction(
        &self,
        _state: &Self::State,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        mouse::Interaction::Pointer
    }

    fn update(
        &self,
        _state: &mut Self::State,
        event: &iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<iced::widget::Action<Message>> {
        if let iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) = event {
            if let Some(cursor_position) = cursor.position_in(bounds) {
                // If panel is visible, ignore clicks within the panel area.
                // Panel width = dict_width + neighbors(40) + alternatives(40, if visible)
                if self.panel_visible {
                    let win_w = bounds.width;
                    let win_h = bounds.height;
                    let panel_w = self.dict_width + 40.0 + if self.alternatives_visible { 40.0 } else { 0.0 };
                    let panel_rect = if self.panel_on_right {
                        Rectangle { x: win_w - panel_w, y: 0.0, width: panel_w, height: win_h }
                    } else {
                        Rectangle { x: 0.0, y: 0.0, width: panel_w, height: win_h }
                    };
                    if panel_rect.contains(cursor_position) {
                        return None;
                    }
                }

                // Process click on image characters
                let img_w_f = self.img_w as f32;
                let img_h_f = self.img_h as f32;
                let scale = f32::min(bounds.width / img_w_f, bounds.height / img_h_f);
                let offset_x = (bounds.width - img_w_f * scale) / 2.0;
                let offset_y = (bounds.height - img_h_f * scale) / 2.0;

                for (line_idx, annotation) in self.annotations.iter().enumerate() {
                    if let Some(line) = &annotation.line {
                        let fixed_size = if line.is_vertical {
                            line.char_boxes.iter().map(|b| b.w).max().unwrap_or(0)
                        } else {
                            line.char_boxes.iter().map(|b| b.h).max().unwrap_or(0)
                        };

                        let mut refined: Vec<BoundingBox> = Vec::new();
                        if let Some(first) = line.char_boxes.first() {
                            refined.push(first.clone());
                        }
                        for i in 1..line.char_boxes.len() {
                            let prev = &line.char_boxes[i - 1];
                            let cur = &line.char_boxes[i];
                            if line.is_vertical {
                                let new_top =
                                    prev.top().saturating_add(fixed_size).max(cur.top());
                                refined.push(BoundingBox::new(
                                    cur.left(), new_top, cur.w, cur.h, cur.confidence,
                                ));
                            } else {
                                let new_left =
                                    prev.left().saturating_add(fixed_size).max(cur.left());
                                refined.push(BoundingBox::new(
                                    new_left, cur.top(), cur.w, cur.h, cur.confidence,
                                ));
                            }
                        }

                        let display_boxes: Vec<BoundingBox> = refined
                            .iter()
                            .map(|b| {
                                let cx = b.left() + b.w / 2;
                                let cy = b.top() + b.h / 2;
                                BoundingBox::new(cx - fixed_size / 2, cy - fixed_size / 2, fixed_size, fixed_size, 1.0)
                            })
                            .collect();

                        for (char_idx, db) in display_boxes.iter().enumerate() {
                            let x = db.x as f32 * scale + offset_x;
                            let y = db.y as f32 * scale + offset_y;
                            let w = db.w as f32 * scale;
                            let h = db.h as f32 * scale;
                            let rect = Rectangle::new(Point::new(x, y), Size::new(w, h));

                            if rect.contains(cursor_position) {
                                if let Some(_ch) = line.text.chars().nth(char_idx) {
                                    return Some(iced::widget::Action::publish(
                                        Message::SelectCharacter(line_idx, char_idx),
                                    ));
                                }
                            }
                        }
                    }
                }
                return Some(iced::widget::Action::publish(Message::Back));
            }
        }
        None
    }

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());
        let img_w_f = self.img_w as f32;
        let img_h_f = self.img_h as f32;
        let scale = f32::min(bounds.width / img_w_f, bounds.height / img_h_f);
        let offset_x = (bounds.width - img_w_f * scale) / 2.0;
        let offset_y = (bounds.height - img_h_f * scale) / 2.0;

        let transform = |bbox: &BoundingBox| -> (Point, Size) {
            (Point::new(bbox.x as f32 * scale + offset_x, bbox.y as f32 * scale + offset_y),
             Size::new(bbox.w as f32 * scale, bbox.h as f32 * scale))
        };

        for (line_idx, annotation) in self.annotations.iter().enumerate() {
            let (pt, sz) = transform(&annotation.bbox);
            frame.stroke(&CanvasPath::rectangle(pt, sz),
                CanvasStroke::default().with_color(Color::from_rgb(1.0, 0.0, 0.0)).with_width(2.0));

            if let Some(line) = &annotation.line {
                let fixed_size = if line.is_vertical {
                    line.char_boxes.iter().map(|b| b.w).max().unwrap_or(0)
                } else {
                    line.char_boxes.iter().map(|b| b.h).max().unwrap_or(0)
                };

                let mut refined: Vec<BoundingBox> = Vec::new();
                if let Some(first) = line.char_boxes.first() { refined.push(first.clone()); }
                for i in 1..line.char_boxes.len() {
                    let prev = &line.char_boxes[i - 1];
                    let cur = &line.char_boxes[i];
                    if line.is_vertical {
                        refined.push(BoundingBox::new(cur.left(), prev.top().saturating_add(fixed_size).max(cur.top()), cur.w, cur.h, cur.confidence));
                    } else {
                        refined.push(BoundingBox::new(prev.left().saturating_add(fixed_size).max(cur.left()), cur.top(), cur.w, cur.h, cur.confidence));
                    }
                }

                let display_boxes: Vec<BoundingBox> = refined.iter().map(|b| {
                    let cx = b.left() + b.w / 2;
                    let cy = b.top() + b.h / 2;
                    BoundingBox::new(cx - fixed_size / 2, cy - fixed_size / 2, fixed_size, fixed_size, 1.0)
                }).collect();

                for (i, char_box) in line.char_boxes.iter().enumerate() {
                    let (pt_c, sz_c) = transform(char_box);
                    frame.stroke(&CanvasPath::rectangle(pt_c, sz_c),
                        CanvasStroke::default().with_color(Color::from_rgb(0.0, 1.0, 0.0)).with_width(1.0));

                    // Draw cursor highlight if this is the selected character
                    if self.cursor_pos == Some((line_idx, i)) {
                        let cursor_rect = CanvasPath::rectangle(pt_c, sz_c);
                        frame.fill(&cursor_rect, Color::from_rgba(0.0, 0.8, 1.0, 0.3));
                        frame.stroke(&cursor_rect,
                            CanvasStroke::default().with_color(Color::from_rgb(0.0, 0.8, 1.0)).with_width(2.0));
                    }

                    if let Some(_ch) = line.text.chars().nth(i) {
                        if i < display_boxes.len() {
                            let db = &display_boxes[i];
                            let (pt_db, sz_db) = transform(db);
                            frame.fill_text(CanvasText {
                                content: _ch.to_string(),
                                position: Point::new(pt_db.x + sz_db.width / 2.0, pt_db.y + sz_db.height / 2.0),
                                max_width: 0.0,
                                color: Color::from_rgb(0.0, 1.0, 0.0),
                                size: Pixels(sz_db.height * BOX_FILL_RATIO),
                                line_height: Default::default(),
                                font: IcedFont::default(),
                                align_x: iced::widget::text::Alignment::Center,
                                align_y: alignment::Vertical::Center,
                                shaping: Default::default(),
                            });
                        }
                    }
                }
            }
        }
        vec![frame.into_geometry()]
    }
}

// ---------------------------------------------------------------------------
// OcrViewer
// ---------------------------------------------------------------------------

pub struct OcrViewer {
    pub image_handle: iced::widget::image::Handle,
    /// Raw PNG bytes of the original screenshot, used for cropping character previews.
    image_bytes: Vec<u8>,
    pub img_w: u32,
    pub img_h: u32,
    pub annotations: Vec<DetectedAnnotation>,
    pub state: OcrOverlayState,
    pub selected_word: Option<SelectedWord>,
    pub alternatives_visible: bool,
    pub db: Arc<DictionaryDatabase>,
    pub deinflector: Arc<Deinflector>,
    /// The index of the character that should be scrolled into view in the neighbor panel.
    pub scroll_neighbor_to: Option<usize>,
    /// The index of the character that should be scrolled into view in the alt panel.
    pub scroll_alt_to: Option<usize>,
}

impl OcrViewer {
    pub fn new(
        image_handle: iced::widget::image::Handle,
        image_bytes: Vec<u8>,
        img_w: u32,
        img_h: u32,
        annotations: Vec<DetectedAnnotation>,
        db: Arc<DictionaryDatabase>,
        deinflector: Arc<Deinflector>,
    ) -> Self {
        let mut state = OcrOverlayState::new();
        let line_results = annotations.iter().map(|a| a.line.clone()).collect::<Vec<_>>();
        state.set_line_results(line_results);
        state.ensure_cursor_position();
        Self { image_handle, image_bytes, img_w, img_h, annotations, state, selected_word: None, alternatives_visible: false, db, deinflector, scroll_neighbor_to: None, scroll_alt_to: None }
    }

    /// Crop the screenshot to show the given character with padding.
    /// Returns an image Handle for the cropped region.
    pub fn crop_character_image(&self, line_idx: usize, char_idx: usize) -> Option<iced::widget::image::Handle> {
        let line = self.state.active_line_results.get(line_idx).and_then(|l| l.as_ref())?;
        let box_item = line.char_boxes.get(char_idx)?;

        // Decode the original image
        let img = image::load_from_memory(&self.image_bytes).ok()?;
        let (img_w, img_h) = (img.width() as i32, img.height() as i32);

        // Calculate crop rect with 20% padding
        let pad = (box_item.h as f32 * 0.2) as i32;
        let crop_left = (box_item.left() - pad).max(0);
        let crop_top = (box_item.top() - pad).max(0);
        let crop_right = (box_item.right() + pad).min(img_w);
        let crop_bottom = (box_item.bottom() + pad).min(img_h);
        let crop_w = (crop_right - crop_left).max(1) as u32;
        let crop_h = (crop_bottom - crop_top).max(1) as u32;

        // Crop and encode to PNG
        let cropped = img.crop_imm(crop_left as u32, crop_top as u32, crop_w, crop_h);
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        cropped.write_to(&mut cursor, image::ImageFormat::Png).ok()?;

        Some(iced::widget::image::Handle::from_bytes(buf))
    }

    pub fn select_character(&mut self, line_idx: usize, char_idx: usize) {
        self.state.current_tapped_line_idx = line_idx as isize;
        self.state.current_tapped_char_idx_in_line = char_idx as isize;
        self.state.current_tapped_idx = self.state.get_global_idx(line_idx, char_idx) as isize;
        self.state.update_highlight_coords(line_idx, char_idx, 1);
        self.state.is_dictionary_visible = true;
        self.alternatives_visible = false;

        // Update gravity so panel opens on the opposite side of the character
        let box_item = self.state.active_line_results.get(line_idx)
            .and_then(|l| l.as_ref())
            .and_then(|line| line.char_boxes.get(char_idx))
            .cloned();
        if let Some(b) = box_item {
            self.state.update_gravity(800.0, 480.0, &b);
        }
        self.do_lookup(line_idx, char_idx);

        // Compute scroll targets: center the selected character in each panel
        self.compute_scroll_targets(line_idx, char_idx);
    }

    pub fn select_neighbor(&mut self, line_idx: usize, char_idx: usize) {
        let is_same = self.state.current_tapped_line_idx == line_idx as isize
            && self.state.current_tapped_char_idx_in_line == char_idx as isize;
        self.state.current_tapped_line_idx = line_idx as isize;
        self.state.current_tapped_char_idx_in_line = char_idx as isize;
        self.state.current_tapped_idx = self.state.get_global_idx(line_idx, char_idx) as isize;
        self.state.update_highlight_coords(line_idx, char_idx, 1);
        self.state.is_dictionary_visible = true;

        if is_same && !self.alternatives_visible {
            // Clicking the already-selected character opens alternatives
            // (only if not already open — closing is handled by Back/Escape
            // or by clicking the selected alternative)
            self.alternatives_visible = true;
        }
        // Always do the lookup for the newly selected character.
        // If alternatives were already open (or just toggled open), they stay
        // open and update with the new character's alternatives.
        self.do_lookup(line_idx, char_idx);

        // Don't scroll the neighbor panel when selecting from it — the user
        // already clicked the item they wanted to see. Only image clicks
        // (select_character) should trigger auto-scroll.
        self.scroll_neighbor_to = None;
        self.scroll_alt_to = None;
    }

    fn do_lookup(&mut self, line_idx: usize, char_idx: usize) {
        let Some(line) = self.state.active_line_results.get(line_idx).and_then(|l| l.as_ref()) else { return; };
        let Some(box_item) = line.char_boxes.get(char_idx) else { return; };
        let text = line.text.chars().skip(char_idx).take(3).collect::<String>();
        self.selected_word = Some(SelectedWord { line_idx, char_idx, text, box_item: box_item.clone() });
        if let Some(result) = self.state.lookup(line_idx, char_idx, &self.db, &self.deinflector) {
            self.state.cached_entries = result.matches;
            self.state.current_word_length = result.max_len;
        } else {
            self.state.cached_entries.clear();
            self.state.current_word_length = 1;
        }
    }

    /// Compute the scroll targets for the neighbor and alternatives panels
    /// so the selected character is centered (or as close as possible).
    fn compute_scroll_targets(&mut self, line_idx: usize, char_idx: usize) {
        // Compute the global character index across all lines
        let mut global_idx = 0;
        for (li, line_opt) in self.state.active_line_results.iter().enumerate() {
            if li == line_idx {
                global_idx += char_idx;
                break;
            }
            if let Some(line) = line_opt.as_ref() {
                global_idx += line.text.chars().count();
            }
        }
        self.scroll_neighbor_to = Some(global_idx);
        self.scroll_alt_to = Some(char_idx);
    }

    /// Create a scroll task for the neighbor panel to center the selected character.
    /// Returns None if no scroll is needed.
    #[allow(dead_code)]
    pub fn scroll_neighbor_task(&self) -> Option<iced::Task<Message>> {
        let target = self.scroll_neighbor_to?;
        // Each button is 32px + 2px spacing = 34px per item
        let item_height = 34.0;
        let target_y = target as f32 * item_height;
        // Center in viewport: subtract approximate half viewport height
        let scroll_y = (target_y - 200.0).max(0.0);
        Some(operate(scroll_to(
            Id::new("neighbor_scroll"),
            AbsoluteOffset { x: Some(0.0), y: Some(scroll_y) },
        )))
    }

    /// Create a scroll task for the alternatives panel to center the selected character.
    /// Returns None if no scroll is needed.
    #[allow(dead_code)]
    pub fn scroll_alt_task(&self) -> Option<iced::Task<Message>> {
        let target = self.scroll_alt_to?;
        let item_height = 34.0;
        let target_y = target as f32 * item_height;
        let scroll_y = (target_y - 200.0).max(0.0);
        Some(operate(scroll_to(
            Id::new("alt_scroll"),
            AbsoluteOffset { x: Some(0.0), y: Some(scroll_y) },
        )))
    }

    pub fn view<'a>(&'a self) -> Element<'a, Message> {
        let has_panel = self.selected_word.is_some();

        // Gravity tells us which side the PANEL goes on
        let panel_on_right = self.state.last_landscape_gravity == Gravity::End;
        let panel_on_bottom = self.state.last_portrait_gravity == Gravity::Bottom;
        let overlay = OverlayProgram {
            annotations: self.annotations.clone(),
            img_w: self.img_w,
            img_h: self.img_h,
            panel_visible: has_panel,
            panel_on_right,
            dict_width: 300.0,
            alternatives_visible: self.alternatives_visible,
            cursor_pos: self.state.current_cursor(),
        };

        let image = IcedImage::new(self.image_handle.clone()).width(Length::Fill).height(Length::Fill);
        let canvas = Canvas::new(overlay).width(Length::Fill).height(Length::Fill);
        let image_stack = Stack::new().push(image).push(canvas);

        if !has_panel {
            return Container::new(image_stack).width(Length::Fill).height(Length::Fill).into();
        }

        // Build panel components
        let dict_panel = self.dictionary_panel(if !self.state.cached_entries.is_empty() {
            self.state.cached_entries.clone()
        } else {
            Vec::new()
        });
        let neigh_panel = self.neighbor_panel();

        // Crop the character preview image from the screenshot
        let preview_image = self.selected_word.as_ref().and_then(|sw| {
            self.crop_character_image(sw.line_idx, sw.char_idx)
        });
        let alt_panel = self.alternatives_panel(preview_image.as_ref());

        let dict_width = Pixels(300.0);
        let neigh_width = Pixels(42.0);

        // Stable inner row: neighbors + dictionary. This never changes
        // structure, so the Scrollables inside never reset.
        let mut inner_row = Row::new().spacing(2);
        if panel_on_right {
            inner_row = inner_row.push(neigh_panel.width(neigh_width));
            inner_row = inner_row.push(dict_panel.width(dict_width));
        } else {
            inner_row = inner_row.push(dict_panel.width(dict_width));
            inner_row = inner_row.push(neigh_panel.width(neigh_width));
        }

        // The panel container wraps the stable row.
        let panel = Container::new(inner_row).padding(2).style(container::rounded_box);

        // The stable panel content: neighbors + dictionary in the correct
        // order. This NEVER changes — same children, same structure.
        let panel_content = Container::new(panel)
            .width(Length::Shrink)
            .height(Length::Fill);

        // Position the panel_content at the correct screen edge using a
        // Row with a Spacer. This Row structure also never changes.
        let positioned_panel: Element<'a, Message> = if panel_on_right {
            Row::new()
                .width(Length::Fill)
                .push(iced::widget::Space::new().width(Length::Fill).height(Length::Fill))
                .push(panel_content)
                .into()
        } else {
            Row::new()
                .width(Length::Fill)
                .push(panel_content)
                .push(iced::widget::Space::new().width(Length::Fill).height(Length::Fill))
                .into()
        };

        // The alt panel is in a Stack ON TOP of the positioned panel.
        // When hidden, it's simply not in the Stack — zero layout impact.
        // When visible, it floats on top at the correct edge.
        let mut content_stack = Stack::new().push(positioned_panel);

        if self.alternatives_visible {
            let alt_w = Pixels(42.0);
            let panel_content_w = dict_width + Pixels(2.0) + neigh_width;

            // The alt panel Container.
            let alt_container = Container::new(alt_panel)
                .width(alt_w)
                .height(Length::Fill)
                .padding(2)
                .style(container::rounded_box);

            // Position the alt panel using spacers on both sides.
            // The fixed-width spacer ensures the alt sits adjacent to the
            // panel content (not overlapping it), while the fill spacer
            // takes the remaining space.
            let alt_positioned: Element<'a, Message> = if panel_on_right {
                // Panel content is on the right. Alt should be to its left.
                // [spacer(fill) | alt | spacer(panel_content_w)]
                // The fill spacer pushes alt right, the fixed spacer ensures
                // the alt doesn't overlap the panel content.
                Row::new()
                    .width(Length::Fill)
                    .push(iced::widget::Space::new().width(Length::Fill).height(Length::Fill))
                    .push(alt_container)
                    .push(iced::widget::Space::new().width(panel_content_w).height(Length::Fill))
                    .into()
            } else {
                // Panel content is on the left. Alt should be to its right.
                // [spacer(panel_content_w) | alt | spacer(fill)]
                // The fixed spacer pushes alt left, the fill spacer takes
                // the remaining space on the right.
                Row::new()
                    .width(Length::Fill)
                    .push(iced::widget::Space::new().width(panel_content_w).height(Length::Fill))
                    .push(alt_container)
                    .push(iced::widget::Space::new().width(Length::Fill).height(Length::Fill))
                    .into()
            };

            content_stack = content_stack.push(alt_positioned);
        }

        // Root Stack: image on bottom, content stack on top.
        Container::new(
            Stack::new()
                .push(image_stack)
                .push(content_stack)
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
    }

    fn dictionary_panel<'a>(&'a self, entries: Vec<FormattedEntry>) -> Container<'a, Message> {
        let mut content = Column::new().padding(4).spacing(4).width(Length::Fill);
        if entries.is_empty() {
            content = content.push(Text::new("No dictionary entries found.").size(14));
        }
        for entry in entries {
            let mut entry_col = Column::new().spacing(4).width(Length::Fill);
            for group in entry.reading_groups {
                entry_col = entry_col.push(self.headword_section(group.clone()));
                for sg in group.sense_groups {
                    entry_col = entry_col.push(self.sense_group(sg));
                }
                entry_col = entry_col.push(iced::widget::Space::new().height(Pixels(4.0)));
            }
            content = content.push(entry_col);
        }
        Container::new(Scrollable::new(content))
            .width(Length::Fill).padding(4).style(container::rounded_box)
    }

    fn headword_section<'a>(&'a self, group: FormattedReadingGroup) -> Container<'a, Message> {
        let cyan = Color::from_rgb(0.4, 0.8, 1.0);
        let gray = Color::from_rgb(0.6, 0.6, 0.6);
        let mut content = Column::new().spacing(2);

        if group.is_kanji_entry {
            for hw in &group.headwords {
                let mut row = Row::new().spacing(6).align_y(alignment::Vertical::Center);
                row = row.push(Text::new(hw.kanji.clone()).size(36).color(cyan)
                    .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }));
                if let Some(o) = &hw.onyomi { row = row.push(Text::new(format!("ON: {o}")).size(12).color(gray)); }
                if let Some(k) = &hw.kunyomi { row = row.push(Text::new(format!("KUN: {k}")).size(12).color(gray)); }
                content = content.push(row);
            }
        } else {
            let mut row = Row::new().spacing(4).align_y(alignment::Vertical::Center);
            for (i, hw) in group.headwords.iter().enumerate() {
                if hw.kanji == group.reading {
                    row = row.push(Text::new(hw.kanji.clone()).size(24).color(cyan)
                        .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }));
                } else {
                    let mut rc = Column::new().align_x(alignment::Horizontal::Center).spacing(1);
                    rc = rc.push(Text::new(group.reading.clone()).size(10).color(gray));
                    rc = rc.push(Text::new(hw.kanji.clone()).size(24).color(cyan)
                        .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }));
                    row = row.push(rc);
                }
                if i + 1 < group.headwords.len() {
                    row = row.push(Text::new("、").size(20).color(gray));
                }
            }
            content = content.push(row);
        }
        Container::new(content)
    }

    fn tag_color(tag: &str) -> Color {
        match tag {
            t if t == "pos" || t == "v" || t == "adj" || t == "adj-i" || t == "adj-na" => Color::from_rgb(0.23, 0.35, 0.48),
            t if t == "n" || t == "adv" || t == "pn" => Color::from_rgb(0.23, 0.48, 0.35),
            t if t.starts_with("jlpt") || t.starts_with("grade") || t == "★" => Color::from_rgb(0.48, 0.23, 0.23),
            _ => Color::from_rgb(0.27, 0.27, 0.27),
        }
    }

    fn sense_group<'a>(&'a self, sg: FormattedSenseGroup) -> Container<'a, Message> {
        let cyan = Color::from_rgb(0.4, 0.8, 1.0);
        let gray = Color::from_rgb(0.6, 0.6, 0.6);
        let white = Color::WHITE;
        let mut content = Column::new().spacing(3);

        if !sg.tags.is_empty() {
            let mut tag_row = Row::new().spacing(3).align_y(alignment::Vertical::Center);
            for tag in &sg.tags {
                let bg = Self::tag_color(tag);
                tag_row = tag_row.push(
                    Container::new(Text::new(tag.clone()).size(9).color(white)
                        .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }))
                    .padding([4.0, 1.0])
                    .style(move |_t: &Theme| container::Style {
                        background: Some(iced::Background::Color(bg)),
                        border: iced::Border { radius: 3.0.into(), ..Default::default() },
                        ..Default::default()
                    }),
                );
            }
            content = content.push(tag_row);
        }

        for sense in &sg.senses {
            let mut sense_row = Row::new().spacing(3).align_y(alignment::Vertical::Top);
            sense_row = sense_row.push(Text::new(format!("{}. ", sense.index)).size(13).color(white));
            let mut nodes_col = Column::new().spacing(1).width(Length::Fill);
            for node in &sense.nodes {
                match node {
                    DefinitionNode::Text(t) => {
                        nodes_col = nodes_col.push(
                            Text::new(t.clone()).size(13).color(white).width(Length::Fill)
                                .wrapping(iced::widget::text::Wrapping::Word),
                        );
                    }
                    DefinitionNode::Ruby { term, reading, .. } => {
                        if term == reading {
                            nodes_col = nodes_col.push(Text::new(term.clone()).size(13).color(cyan)
                                .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }));
                        } else {
                            let mut rc = Column::new().align_x(alignment::Horizontal::Center).spacing(0);
                            rc = rc.push(Text::new(reading.clone()).size(8).color(gray));
                            rc = rc.push(Text::new(term.clone()).size(13).color(cyan)
                                .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }));
                            nodes_col = nodes_col.push(rc);
                        }
                    }
                    DefinitionNode::Tag { text, .. } => {
                        nodes_col = nodes_col.push(Text::new(format!("[{text}]")).size(11).color(gray));
                    }
                    _ => {}
                }
            }
            sense_row = sense_row.push(nodes_col);
            content = content.push(sense_row);
        }
        Container::new(content)
    }

    fn neighbor_panel<'a>(&'a self) -> Container<'a, Message> {
        let state = self.state.get_neighbor_ui_state();
        let mut content = Column::new().padding(2).spacing(2);
        for line in state {
            for cs in line.chars {
                content = content.push(
                    Button::new(
                        Text::new(cs.text.clone())
                            .size(20)
                            .align_x(alignment::Horizontal::Center)
                    )
                    .width(Pixels(32.0))
                    .height(Pixels(32.0))
                    .style(if cs.is_selected { button::primary } else { button::secondary })
                    .on_press(Message::SelectNeighbor(line.line_idx, cs.char_idx)),
                );
            }
        }
        // Scrollable with hidden scrollbar — fixed width so buttons always fit
        Container::new(
            Scrollable::new(content)
                .id(Id::new("neighbor_scroll"))
                .direction(iced::widget::scrollable::Direction::Vertical(
                    iced::widget::scrollable::Scrollbar::hidden(),
                )),
        )
        .width(Pixels(42.0))
        .height(Length::Fill)
        .padding(2)
        .style(container::rounded_box)
    }

    fn alternatives_panel<'a>(
        &'a self,
        preview_image: Option<&iced::widget::image::Handle>,
    ) -> Container<'a, Message> {
        let mut content = Column::new().padding(2).spacing(2);

        // Character preview image at the top
        if let Some(img_handle) = preview_image {
            content = content.push(
                IcedImage::new(img_handle.clone())
                    .width(Pixels(32.0))
                    .height(Pixels(32.0)),
            );
        }

        if let Some(alt_state) = self.state.get_alternatives_ui_state() {
            for c in alt_state.candidates {
                content = content.push(
                    Button::new(
                        Text::new(c.char.to_string())
                            .size(20)
                            .align_x(alignment::Horizontal::Center)
                    )
                    .width(Pixels(32.0))
                    .height(Pixels(32.0))
                    .style(if c.is_selected { button::primary } else { button::secondary })
                    .on_press(Message::SelectAlternative(c.char)),
                );
            }
        }
        Container::new(
            Scrollable::new(content)
                .id(Id::new("alt_scroll"))
                .direction(iced::widget::scrollable::Direction::Vertical(
                    iced::widget::scrollable::Scrollbar::hidden(),
                )),
        )
        .height(Length::Fill)
        .padding(2)
        .style(container::rounded_box)
    }
}
