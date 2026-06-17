use iced::widget::canvas::{self, Canvas, Frame, Geometry, Path as CanvasPath, Stroke as CanvasStroke, Text as CanvasText};
use iced::widget::image::Handle as IcedImageHandle;
use iced::widget::{
    button, text, Button, Column, Container, Image as IcedImage, Row, Scrollable, Stack, Text,
};
use iced::widget::container;
use iced::{alignment, Color, Element, Font as IcedFont, Length, mouse, Pixels, Point, Rectangle, Renderer, Size, Theme};

use crate::models::*;
use crate::overlay_state::OcrOverlayState;

const BOX_FILL_RATIO: f32 = 0.9;

// ---------------------------------------------------------------------------
// OverlayProgram
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct OverlayProgram {
    pub annotations: Vec<DetectedAnnotation>,
    pub img_w: u32,
    pub img_h: u32,
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
        match event {
            iced::Event::Keyboard(iced::keyboard::Event::KeyPressed { key, .. }) => {
                println!("[Canvas] KeyPressed event received: {:?}", key);
                if *key == iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape) {
                    println!("[Canvas] Escape detected, publishing Back");
                    return Some(iced::widget::Action::publish(Message::Back));
                }
            }
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Right)) => {
                println!("[Canvas] Right-click detected, publishing Back");
                return Some(iced::widget::Action::publish(Message::Back));
            }
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if let Some(cursor_position) = cursor.position_in(bounds) {
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
                                    let new_top = prev.top().saturating_add(fixed_size).max(cur.top());
                                    refined.push(BoundingBox::new(cur.left(), new_top, cur.w, cur.h, cur.confidence));
                                } else {
                                    let new_left = prev.left().saturating_add(fixed_size).max(cur.left());
                                    refined.push(BoundingBox::new(new_left, cur.top(), cur.w, cur.h, cur.confidence));
                                }
                            }

                            let display_boxes: Vec<BoundingBox> = refined.iter().map(|b| {
                                let center_x = b.left() + b.w / 2;
                                let center_y = b.top() + b.h / 2;
                                let left = center_x - fixed_size / 2;
                                let top = center_y - fixed_size / 2;
                                BoundingBox::new(left, top, fixed_size, fixed_size, 1.0)
                            }).collect();

                            for (char_idx, db) in display_boxes.iter().enumerate() {
                                let x = db.x as f32 * scale + offset_x;
                                let y = db.y as f32 * scale + offset_y;
                                let w = db.w as f32 * scale;
                                let h = db.h as f32 * scale;
                                let rect = Rectangle::new(Point::new(x, y), Size::new(w, h));

                                if rect.contains(cursor_position) {
                                    if let Some(ch) = line.text.chars().nth(char_idx) {
                                        println!("Clicked character: {}", ch);
                                        return Some(iced::widget::Action::publish(Message::SelectCharacter(line_idx, char_idx)));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
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
            let x = bbox.x as f32 * scale + offset_x;
            let y = bbox.y as f32 * scale + offset_y;
            let w = bbox.w as f32 * scale;
            let h = bbox.h as f32 * scale;
            (Point::new(x, y), Size::new(w, h))
        };

        for annotation in &self.annotations {
            let (pt, sz) = transform(&annotation.bbox);
            let path = CanvasPath::rectangle(pt, sz);
            let stroke = CanvasStroke::default()
                .with_color(Color::from_rgb(1.0, 0.0, 0.0))
                .with_width(2.0);
            frame.stroke(&path, stroke);

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
                        let new_top = prev.top().saturating_add(fixed_size).max(cur.top());
                        refined.push(BoundingBox::new(cur.left(), new_top, cur.w, cur.h, cur.confidence));
                    } else {
                        let new_left = prev.left().saturating_add(fixed_size).max(cur.left());
                        refined.push(BoundingBox::new(new_left, cur.top(), cur.w, cur.h, cur.confidence));
                    }
                }

                let display_boxes: Vec<BoundingBox> = refined.iter().map(|b| {
                    let center_x = b.left() + b.w / 2;
                    let center_y = b.top() + b.h / 2;
                    let left = center_x - fixed_size / 2;
                    let top = center_y - fixed_size / 2;
                    BoundingBox::new(left, top, fixed_size, fixed_size, 1.0)
                }).collect();

                for (i, char_box) in line.char_boxes.iter().enumerate() {
                    let (pt_c, sz_c) = transform(char_box);
                    let char_path = CanvasPath::rectangle(pt_c, sz_c);
                    let char_stroke = CanvasStroke::default()
                        .with_color(Color::from_rgb(0.0, 1.0, 0.0))
                        .with_width(1.0);
                    frame.stroke(&char_path, char_stroke);

                    if let Some(ch) = line.text.chars().nth(i) {
                        if i < display_boxes.len() {
                            let db = &display_boxes[i];
                            let (pt_db, sz_db) = transform(db);
                            let font_size = sz_db.height * BOX_FILL_RATIO;
                            let canvas_text = CanvasText {
                                content: ch.to_string(),
                                position: Point::new(
                                    pt_db.x + sz_db.width / 2.0,
                                    pt_db.y + sz_db.height / 2.0,
                                ),
                                max_width: 0.0,
                                color: Color::from_rgb(0.0, 1.0, 0.0),
                                size: Pixels(font_size as f32),
                                line_height: Default::default(),
                                font: IcedFont::default(),
                                align_x: iced::widget::text::Alignment::Center,
                                align_y: iced::alignment::Vertical::Center,
                                shaping: Default::default(),
                            };
                            frame.fill_text(canvas_text);
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
    pub image_handle: IcedImageHandle,
    pub img_w: u32,
    pub img_h: u32,
    pub annotations: Vec<DetectedAnnotation>,
    pub state: OcrOverlayState,
    pub selected_word: Option<SelectedWord>,
    pub alternatives_visible: bool,
}

impl OcrViewer {
    pub fn new(
        image_handle: IcedImageHandle,
        img_w: u32,
        img_h: u32,
        annotations: Vec<DetectedAnnotation>,
    ) -> Self {
        let mut state = OcrOverlayState::new();
        let line_results = annotations
            .iter()
            .map(|annotation| annotation.line.clone())
            .collect::<Vec<_>>();
        state.set_line_results(line_results);
        state.ensure_cursor_position();

        Self {
            image_handle,
            img_w,
            img_h,
            annotations,
            state,
            selected_word: None,
            alternatives_visible: false,
        }
    }

    pub fn select_character(&mut self, line_idx: usize, char_idx: usize) {
        self.state.current_tapped_line_idx = line_idx as isize;
        self.state.current_tapped_char_idx_in_line = char_idx as isize;
        self.state.current_tapped_idx = self.state.get_global_idx(line_idx, char_idx) as isize;
        self.state.update_highlight_coords(line_idx, char_idx, 1);
        self.state.is_dictionary_visible = true;
        self.alternatives_visible = false;

        let Some(line) = self
            .state
            .active_line_results
            .get(line_idx)
            .and_then(|line| line.as_ref())
        else {
            return;
        };
        let Some(box_item) = line.char_boxes.get(char_idx) else {
            return;
        };
        let text = line.text.chars().skip(char_idx).take(3).collect::<String>();
        self.selected_word = Some(SelectedWord {
            line_idx,
            char_idx,
            text,
            box_item: box_item.clone(),
        });
    }

    pub fn dummy_dictionary_entries() -> Vec<FormattedEntry> {
        vec![
            FormattedEntry {
                term: "テスト".to_string(),
                reading_groups: vec![FormattedReadingGroup {
                    reading: "テスト".to_string(),
                    headwords: vec![FormattedHeadword {
                        kanji: "テスト".to_string(),
                        onyomi: None,
                        kunyomi: None,
                    }],
                    sense_groups: vec![FormattedSenseGroup {
                        tags: vec!["n".to_string(), "dummy".to_string()],
                        senses: vec![FormattedSense {
                            index: 1,
                            nodes: vec![
                                DefinitionNode::Text("Dummy dictionary entry for layout testing.".to_string()),
                                DefinitionNode::Text(" This row intentionally contains enough text to exercise wrapping and scrolling.".to_string()),
                                DefinitionNode::Ruby { term: "漢字".to_string(), reading: "かんじ".to_string(), is_mini: false },
                                DefinitionNode::Text(" can appear inline.".to_string()),
                            ],
                        }],
                        is_forms: false,
                    }],
                    is_kanji_entry: false,
                }],
            },
            FormattedEntry {
                term: "確認".to_string(),
                reading_groups: vec![FormattedReadingGroup {
                    reading: "カクニン".to_string(),
                    headwords: vec![FormattedHeadword {
                        kanji: "確認".to_string(),
                        onyomi: Some("カク ニン".to_string()),
                        kunyomi: None,
                    }],
                    sense_groups: vec![FormattedSenseGroup {
                        tags: vec!["suru".to_string(), "vt".to_string()],
                        senses: vec![FormattedSense {
                            index: 1,
                            nodes: vec![DefinitionNode::Text("Dummy sense 1: check, verify, confirm.".to_string())],
                        }],
                        is_forms: false,
                    }],
                    is_kanji_entry: false,
                }],
            },
            FormattedEntry {
                term: "本日".to_string(),
                reading_groups: vec![FormattedReadingGroup {
                    reading: "ホンジツ".to_string(),
                    headwords: vec![FormattedHeadword {
                        kanji: "本日".to_string(),
                        onyomi: Some("ホン ジツ".to_string()),
                        kunyomi: Some("もとじつ".to_string()),
                    }],
                    sense_groups: vec![FormattedSenseGroup {
                        tags: vec!["n".to_string(), "adj-no".to_string()],
                        senses: vec![FormattedSense {
                            index: 1,
                            nodes: vec![DefinitionNode::Text("Dummy sense 2: today; the present day.".to_string())],
                        }],
                        is_forms: false,
                    }],
                    is_kanji_entry: false,
                }],
            },
        ]
    }

    pub fn view<'a>(&'a self) -> Element<'a, Message> {
        let overlay = OverlayProgram {
            annotations: self.annotations.clone(),
            img_w: self.img_w,
            img_h: self.img_h,
        };

        let image = IcedImage::new(self.image_handle.clone())
            .width(Length::Fill)
            .height(Length::Fill);

        let canvas = Canvas::new(overlay.clone())
            .width(Length::Fill)
            .height(Length::Fill);

        let image_stack = Stack::new().push(image).push(canvas);
        let mut root = Container::new(image_stack)
            .width(Length::Fill)
            .height(Length::Fill);

        if let Some(_selected_word) = &self.selected_word {
            let (root_width, root_height) = (800.0_f32, 480.0_f32);
            let is_landscape = true;
            let (panel_width, panel_height) = self.state.panel_dimensions(root_width, root_height);

            let dictionary_entries = Self::dummy_dictionary_entries();
            let dictionary_panel =
                self.dictionary_panel(dictionary_entries, panel_width, panel_height)
                    .width(Length::Fill);
            let neighbor_panel = self.neighbor_panel();
            let alternatives_panel = if self.alternatives_visible {
                self.alternatives_panel()
            } else {
                Container::new(Text::new(""))
            };

            let correction = Row::new().push(neighbor_panel);
            let dictionary = Column::new().push(dictionary_panel);
            let alternatives = Column::new().push(alternatives_panel);

            let content: Container<'a, Message> = if is_landscape {
                match self.state.last_landscape_gravity {
                    Gravity::End => container(
                        Row::new()
                            .push(alternatives)
                            .push(correction)
                            .push(dictionary),
                    ),
                    Gravity::Start => container(
                        Row::new()
                            .push(dictionary)
                            .push(correction)
                            .push(alternatives),
                    ),
                    Gravity::Top | Gravity::Bottom => container(
                        Row::new()
                            .push(dictionary)
                            .push(correction)
                            .push(alternatives),
                    ),
                }
            } else {
                container(
                    Column::new()
                        .push(dictionary)
                        .push(correction)
                        .push(alternatives),
                )
            };

            let panel = Container::new(content)
                .padding(10)
                .style(container::rounded_box)
                .width(Length::Fill)
                .height(Length::Fill);

            let overlay_stack = Stack::new()
                .push(root)
                .push(panel);

            root = Container::new(overlay_stack)
                .width(Length::Fill)
                .height(Length::Fill);
        }

        root.into()
    }

    pub fn dictionary_panel<'a>(
        &self,
        entries: Vec<FormattedEntry>,
        _panel_width: f32,
        _panel_height: f32,
    ) -> Container<'a, Message> {
        let mut content = Column::new().padding(10).spacing(12);

        for entry in entries {
            let mut entry_column = Column::new().spacing(8);
            for group in entry.reading_groups {
                entry_column = entry_column.push(self.headword_section(group.clone()));
                for sense_group in group.sense_groups {
                    entry_column = entry_column.push(self.sense_group(sense_group));
                }
                entry_column = entry_column.push(iced::widget::space::horizontal());
            }
            content = content.push(entry_column);
        }

        Container::new(Scrollable::new(content))
            .padding(12)
            .style(container::rounded_box)
    }

    pub fn headword_section<'a>(&self, group: FormattedReadingGroup) -> Container<'a, Message> {
        let mut content = Column::new().spacing(4);
        if group.is_kanji_entry {
            for headword in group.headwords {
                let mut row = Row::new().spacing(10).align_y(alignment::Vertical::Center);
                row = row.push(
                    Text::new(headword.kanji.clone())
                        .size(48)
                        .style(text::primary),
                );
                if let Some(onyomi) = &headword.onyomi {
                    row = row.push(Text::new(format!("ON: {onyomi}")));
                }
                if let Some(kunyomi) = headword.kunyomi {
                    row = row.push(Text::new(format!("KUN: {kunyomi}")));
                }
                content = content.push(row);
            }
        } else {
            let mut row = Row::new().spacing(8).align_y(alignment::Vertical::Center);
            for (idx, headword) in group.headwords.iter().enumerate() {
                row = row.push(
                    Text::new(headword.kanji.clone())
                        .size(32)
                        .style(text::primary),
                );
                if idx + 1 < group.headwords.len() {
                    row = row.push(Text::new("、"));
                }
            }
            content = content.push(row);
        }
        Container::new(content)
    }

    pub fn sense_group<'a>(&self, sense_group: FormattedSenseGroup) -> Container<'a, Message> {
        let mut content = Column::new().spacing(6);
        if !sense_group.tags.is_empty() {
            for tag in sense_group.tags {
                content = content.push(text(tag));
            }
        }
        for sense in sense_group.senses {
            let text = format!("{}. ", sense.index)
                + &sense
                    .nodes
                    .iter()
                    .map(|node| match node {
                        DefinitionNode::Text(text) => text.clone(),
                        DefinitionNode::Ruby { term, reading, .. } => {
                            format!("{term}【{reading}】")
                        }
                        DefinitionNode::Tag { text, .. } => format!("[{text}]"),
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
            content = content.push(Text::new(text).width(Length::Fill));
        }
        Container::new(content)
    }

    pub fn neighbor_panel(&self) -> Container<'_, Message> {
        let state = self.state.get_neighbor_ui_state();
        let mut content = Column::new().padding(6).spacing(4);
        for line in state {
            let mut row = Row::new().spacing(4);
            for char_state in line.chars {
                let button = Button::new(Text::new(char_state.text.clone()).size(24))
                    .style(if char_state.is_selected {
                        button::primary
                    } else {
                        button::secondary
                    })
                    .on_press(Message::SelectCharacter(line.line_idx, char_state.char_idx));
                row = row.push(button);
            }
            content = content.push(row);
        }
        Container::new(Scrollable::new(content))
            .width(Length::Shrink)
            .height(Length::Fill)
            .padding(6)
            .style(container::rounded_box)
    }

    pub fn alternatives_panel(&self) -> Container<'_, Message> {
        let Some(alt_state) = self.state.get_alternatives_ui_state() else {
            return Container::new(Text::new(""));
        };
        let mut content = Column::new().padding(6).spacing(4);
        for candidate in alt_state.candidates {
            let button = Button::new(Text::new(candidate.char.to_string()).size(28))
                .style(if candidate.is_selected {
                    button::primary
                } else {
                    button::secondary
                })
                .on_press(Message::SelectAlternative(candidate.char));
            content = content.push(button);
        }
        Container::new(Scrollable::new(content))
            .width(Length::Shrink)
            .height(Length::Fill)
            .padding(6)
            .style(container::rounded_box)
    }
}
