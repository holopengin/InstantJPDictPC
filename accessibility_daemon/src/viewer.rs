use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use iced::widget::canvas::{
    self, Canvas, Frame, Geometry, Path as CanvasPath, Stroke as CanvasStroke,
    Text as CanvasText,
};
use iced::widget::{
    Column, Container, Image as IcedImage, Row, Scrollable, Stack, Text,
};
use iced::widget::container;
use iced::advanced::widget::operation::scrollable::{scroll_to, AbsoluteOffset};
use iced::widget::Id;
use iced::advanced::widget::operate;
use iced::{
    alignment, Color, Element, Font as IcedFont, Length, Pixels, Point, Rectangle, Renderer,
    Size, Theme, mouse, touch,
};

use crate::data::db::DictionaryDatabase;
use crate::models::*;
use crate::overlay_state::OcrOverlayState;
use crate::util::deinflector::Deinflector;

/// Font fill ratio for OCR character glyphs drawn on the canvas annotation layer.
const CANVAS_CHAR_RATIO: f32 = 0.9;
/// Font fill ratio for character buttons in the neighbor/alternatives panels.
const BUTTON_CHAR_RATIO: f32 = 0.6;

// ---------------------------------------------------------------------------
// Pan state (used by OverlayProgram::State)
// ---------------------------------------------------------------------------

const TAP_THRESHOLD: f32 = 5.0;
/// Dead zone for drag start in the TapOrDrag widget.
/// The first few pixels of movement don't count as drag, preventing
/// accidental drags from taps.
const DRAG_DEAD_ZONE: f32 = 3.0;

#[derive(Debug, Clone, Copy, Default)]
pub struct PanState {
    pub is_panning: bool,
    pub start_x: f32,
    pub start_y: f32,
    pub trans_x_at_start: f32,
    pub trans_y_at_start: f32,
    /// Whether the current interaction is a tap (short press without movement).
    /// Set to true on press, moved to false once movement exceeds TAP_THRESHOLD.
    pub is_tap: bool,
    /// Last cursor position during pan, used to compute incremental deltas.
    pub last_x: f32,
    pub last_y: f32,
    /// Accumulated movement since press. Used for dead zone:
    /// drag only starts after this exceeds DRAG_DEAD_ZONE.
    pub accumulated_dx: f32,
    pub accumulated_dy: f32,
    /// Whether the drag dead zone has been exceeded.
    pub drag_started: bool,
    /// Pinch zoom: the ID and position of the first finger.
    pub pinch_finger1: Option<(touch::Finger, Point)>,
    /// Pinch zoom: the ID and position of the second finger.
    pub pinch_finger2: Option<(touch::Finger, Point)>,
    /// Pinch zoom: previous frame's distance between fingers.
    pub pinch_prev_dist: f32,
    /// Pinch zoom: previous frame's midpoint X.
    pub pinch_prev_focus_x: f32,
    /// Pinch zoom: previous frame's midpoint Y.
    pub pinch_prev_focus_y: f32,

}

// ---------------------------------------------------------------------------
// OverlayProgram
// ---------------------------------------------------------------------------

use iced::widget::image::Handle as ImageHandle;

#[derive(Clone)]
pub struct OverlayProgram {
    pub annotations: Rc<Vec<DetectedAnnotation>>,
    pub img_w: u32,
    pub img_h: u32,
    /// The screenshot image to draw on the canvas.
    pub image: Option<ImageHandle>,
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
    /// Current zoom scale.
    pub current_scale: f32,
    /// Current translation X.
    pub current_trans_x: f32,
    /// Current translation Y.
    pub current_trans_y: f32,
    /// Whether this canvas draws annotations (true) or just the image (false).
    pub draw_annotations: bool,
    /// Whether the user is actively zooming/panning.
    pub is_zooming: bool,
    /// Whether this canvas instance should handle pan/zoom events.
    /// The image canvas (behind the annotation canvas) must NOT handle them,
    /// or pan/zoom would be applied twice.
    pub handle_pan_zoom: bool,
    /// Coordinates of characters to highlight in yellow (matched word).
    pub highlighted_coords: Vec<(usize, usize)>,
}

impl OverlayProgram {
    /// Compute the base scale (fit image to bounds) and offset (center in bounds).
    fn base_transform(&self, bounds: Rectangle) -> (f32, f32, f32) {
        let img_w_f = self.img_w as f32;
        let img_h_f = self.img_h as f32;
        let base_scale = f32::min(bounds.width / img_w_f, bounds.height / img_h_f);
        let offset_x = (bounds.width - img_w_f * base_scale) / 2.0;
        let offset_y = (bounds.height - img_h_f * base_scale) / 2.0;
        (base_scale, offset_x, offset_y)
    }

    /// Transform a screen-space point back to image-space.
    fn screen_to_image(&self, bounds: Rectangle, screen_x: f32, screen_y: f32) -> (f32, f32) {
        let (base_scale, offset_x, offset_y) = self.base_transform(bounds);
        let total_scale = base_scale * self.current_scale;
        (
            (screen_x - offset_x - self.current_trans_x) / total_scale,
            (screen_y - offset_y - self.current_trans_y) / total_scale,
        )
    }

    /// Check whether a screen-space point is over the panel area.
    /// The panel is positioned at the correct edge of the screen.
    fn is_over_panel(&self, bounds: Rectangle, screen_x: f32, _screen_y: f32) -> bool {
        if !self.panel_visible {
            return false;
        }
        let panel_w = self.dict_width + 42.0 + 42.0 + 6.0; // dict + neighbors + alt + spacing
        if self.panel_on_right {
            screen_x >= bounds.width - panel_w
        } else {
            screen_x <= panel_w
        }
    }

    /// Check whether a screen-space point hits any character bounding box.
    /// Returns `Some((line_idx, char_idx))` if a hit is found.
    fn hit_test(&self, bounds: Rectangle, screen_x: f32, screen_y: f32) -> Option<(usize, usize)> {
        let (img_x, img_y) = self.screen_to_image(bounds, screen_x, screen_y);
        for (line_idx, annotation) in self.annotations.iter().enumerate() {
            if let Some(line) = &annotation.line {
                for (char_idx, char_box) in line.char_boxes.iter().enumerate() {
                    if img_x >= char_box.left() as f32
                        && img_x <= char_box.right() as f32
                        && img_y >= char_box.top() as f32
                        && img_y <= char_box.bottom() as f32
                    {
                        return Some((line_idx, char_idx));
                    }
                }
            }
        }
        None
    }
}

impl canvas::Program<Message, Theme, Renderer> for OverlayProgram {
    type State = PanState;

    fn mouse_interaction(
        &self,
        state: &Self::State,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if state.drag_started {
            mouse::Interaction::Grabbing
        } else {
            mouse::Interaction::Idle
        }
    }

    fn update(
        &self,
        state: &mut Self::State,
        event: &iced::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<iced::widget::Action<Message>> {
        // The image canvas (behind the annotation canvas) must NOT handle pan/zoom,
        // or it would be applied twice (both canvases receive the same events).
        if !self.handle_pan_zoom {
            return None;
        }

        // Check if cursor is over the panel area — if so, don't handle pan/zoom
        let over_panel = self.panel_visible
            && cursor.position().map_or(false, |p| self.is_over_panel(bounds, p.x, p.y));

        match event {
            // --- Scroll wheel zoom (only when NOT over panel) ---
            iced::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                if over_panel { return None; }
                if let Some(cursor_position) = cursor.position_in(bounds) {
                    let (dy, _dx) = match delta {
                        mouse::ScrollDelta::Lines { y, x } => (*y, *x),
                        mouse::ScrollDelta::Pixels { y, x } => (*y / 100.0, *x / 100.0),
                    };
                    if dy != 0.0 {
                        return Some(iced::widget::Action::publish(
                            Message::ZoomOnCursor {
                                delta: dy,
                                cursor_x: cursor_position.x,
                                cursor_y: cursor_position.y,
                            },
                        ));
                    }
                }
            }

            // --- Mouse: left button pressed (pan start or tap) ---
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if over_panel { return None; } // Let panel handle it
                if let Some(cursor_position) = cursor.position_in(bounds) {
                    state.is_panning = true;
                    state.start_x = cursor_position.x;
                    state.start_y = cursor_position.y;
                    state.last_x = cursor_position.x;
                    state.last_y = cursor_position.y;
                    state.trans_x_at_start = self.current_trans_x;
                    state.trans_y_at_start = self.current_trans_y;
                    state.is_tap = true;
                    state.accumulated_dx = 0.0;
                    state.accumulated_dy = 0.0;
                    state.drag_started = false;
                }
                return None;
            }

            // --- Mouse: cursor moved (pan with dead zone) ---
            iced::Event::Mouse(mouse::Event::CursorMoved { position }) => {
                // If cursor moved over panel during pan, cancel the pan
                if over_panel && state.is_panning {
                    state.is_panning = false;
                    state.is_tap = false;
                    state.drag_started = false;
                    state.accumulated_dx = 0.0;
                    state.accumulated_dy = 0.0;
                    return None;
                }
                if state.is_panning {
                    let dx = position.x - state.last_x;
                    let dy = position.y - state.last_y;
                    state.last_x = position.x;
                    state.last_y = position.y;

                    state.accumulated_dx += dx;
                    state.accumulated_dy += dy;
                    let total = (state.accumulated_dx.powi(2) + state.accumulated_dy.powi(2)).sqrt();

                    if !state.drag_started && total > DRAG_DEAD_ZONE {
                        state.drag_started = true;
                        state.is_tap = false;
                    }

                    if state.drag_started && (dx != 0.0 || dy != 0.0) {
                        return Some(iced::widget::Action::publish(Message::PanDelta { dx, dy }));
                    }
                }
            }

            // --- Mouse: left button released (pan end or tap) ---
            iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                // If over panel, don't process — let the panel handle it
                if over_panel {
                    state.is_panning = false;
                    state.is_tap = false;
                    state.drag_started = false;
                    return None;
                }
                let was_tap = state.is_tap;
                state.is_panning = false;
                state.is_tap = false;
                state.drag_started = false;
                state.accumulated_dx = 0.0;
                state.accumulated_dy = 0.0;

                if was_tap {
                    if let Some(cursor_position) = cursor.position_in(bounds) {
                        if let Some((li, ci)) = self.hit_test(bounds, cursor_position.x, cursor_position.y) {
                            return Some(iced::widget::Action::publish(Message::SelectCharacter(li, ci)));
                        }
                        // Left-click on blank area: do nothing (don't close the app)
                    }
                }
            }

            // --- Touch: finger pressed ---
            iced::Event::Touch(touch::Event::FingerPressed { id, position }) => {
                let already_tracking = state.pinch_finger1.is_some() || state.pinch_finger2.is_some();

                if !already_tracking && !over_panel {
                    // First finger — start pan
                    state.is_panning = true;
                    state.pinch_finger1 = Some((*id, *position));
                    state.start_x = position.x;
                    state.start_y = position.y;
                    state.last_x = position.x;
                    state.last_y = position.y;
                    state.trans_x_at_start = self.current_trans_x;
                    state.trans_y_at_start = self.current_trans_y;
                    state.is_tap = true;
                    state.accumulated_dx = 0.0;
                    state.accumulated_dy = 0.0;
                    state.drag_started = false;
                    state.pinch_finger2 = None;
                    state.pinch_prev_dist = 0.0;
                } else if already_tracking && state.pinch_finger1.is_some() && state.pinch_finger2.is_none() {
                    // Second finger — switch to pinch zoom mode
                    state.pinch_finger2 = Some((*id, *position));
                    // Compute initial distance and midpoint
                    if let Some((_, f1_pos)) = state.pinch_finger1 {
                        let dx = position.x - f1_pos.x;
                        let dy = position.y - f1_pos.y;
                        state.pinch_prev_dist = (dx * dx + dy * dy).sqrt().max(10.0);
                        state.pinch_prev_focus_x = (f1_pos.x + position.x) / 2.0;
                        state.pinch_prev_focus_y = (f1_pos.y + position.y) / 2.0;
                    }
                    state.is_tap = false;
                    state.drag_started = true;
                }
                // Third and subsequent fingers — ignore entirely
                return None;
            }

            // --- Touch: finger moved (pan or pinch) ---
            iced::Event::Touch(touch::Event::FingerMoved { id, position }) => {
                if state.is_panning {
                    // Only update positions for our two tracked fingers — ignore any others
                    if let Some((fid, _)) = state.pinch_finger1 {
                        if fid == *id { state.pinch_finger1 = Some((*id, *position)); }
                    }
                    if let Some((fid, _)) = state.pinch_finger2 {
                        if fid == *id { state.pinch_finger2 = Some((*id, *position)); }
                    }

                    if state.pinch_finger2.is_some() {
                        // Pinch zoom: both fingers active
                        if let (Some((_, f1)), Some((_, f2))) = (state.pinch_finger1, state.pinch_finger2) {
                            let dx = f2.x - f1.x;
                            let dy = f2.y - f1.y;
                            let current_dist = (dx * dx + dy * dy).sqrt();
                            let scale_factor = if state.pinch_prev_dist > 0.0 { current_dist / state.pinch_prev_dist } else { 1.0 };
                            let focus_x = (f1.x + f2.x) / 2.0;
                            let focus_y = (f1.y + f2.y) / 2.0;
                            let prev_focus_x = state.pinch_prev_focus_x;
                            let prev_focus_y = state.pinch_prev_focus_y;

                            // Update prev for next frame
                            state.pinch_prev_dist = current_dist;
                            state.pinch_prev_focus_x = focus_x;
                            state.pinch_prev_focus_y = focus_y;

                            // Compute base_offset for the Y axis so the handler can
                            // correctly account for the image centering.
                            let img_h_f = self.img_h as f32;
                            let base_scale = f32::min(bounds.width / self.img_w as f32, bounds.height / img_h_f);
                            let base_offset_y = (bounds.height - img_h_f * base_scale) / 2.0;

                            return Some(iced::widget::Action::publish(
                                Message::PinchZoom {
                                    scale_factor,
                                    focus_x,
                                    focus_y,
                                    prev_focus_x,
                                    prev_focus_y,
                                    base_offset_y,
                                },
                            ));
                        }
                    } else {
                        // Single finger pan with dead zone
                        let dx = position.x - state.last_x;
                        let dy = position.y - state.last_y;
                        state.last_x = position.x;
                        state.last_y = position.y;

                        state.accumulated_dx += dx;
                        state.accumulated_dy += dy;
                        let total = (state.accumulated_dx.powi(2) + state.accumulated_dy.powi(2)).sqrt();

                        if !state.drag_started && total > DRAG_DEAD_ZONE {
                            state.drag_started = true;
                            state.is_tap = false;
                        }

                        if state.drag_started && (dx != 0.0 || dy != 0.0) {
                            return Some(iced::widget::Action::publish(Message::PanDelta { dx, dy }));
                        }
                    }
                }
            }

            // --- Touch: finger lifted (pan end) — detect taps ---
            iced::Event::Touch(touch::Event::FingerLifted { id, position }) => {
                // Only process lifts for our two tracked fingers
                let tracked_f1 = state.pinch_finger1.map(|(fid, _)| fid) == Some(*id);
                let tracked_f2 = state.pinch_finger2.map(|(fid, _)| fid) == Some(*id);

                if !tracked_f1 && !tracked_f2 {
                    // Unknown finger (3rd+) — ignore entirely
                    return None;
                }

                let was_pinch = state.pinch_finger2.is_some();

                // Remove the lifted finger from tracking
                if tracked_f1 { state.pinch_finger1 = None; }
                if tracked_f2 { state.pinch_finger2 = None; }

                // If either tracked finger lifted, end the gesture entirely.
                // Don't try to continue pinch/pan with remaining fingers — this
                // prevents flickering when 3+ fingers are involved.
                let was_tap = state.is_tap;
                state.is_panning = false;
                state.is_tap = false;
                state.drag_started = false;
                state.accumulated_dx = 0.0;
                state.accumulated_dy = 0.0;
                state.pinch_prev_dist = 0.0;
                state.pinch_prev_focus_x = 0.0;
                state.pinch_prev_focus_y = 0.0;

                if was_pinch {
                    return Some(iced::widget::Action::publish(Message::PinchEnd));
                }

                if was_tap {
                    return Some(iced::widget::Action::publish(
                        match self.hit_test(bounds, position.x, position.y) {
                            Some((li, ci)) => Message::SelectCharacter(li, ci),
                            None => Message::Back,
                        },
                    ));
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
        let (base_scale, base_offset_x, base_offset_y) = self.base_transform(bounds);
        let total_scale = base_scale * self.current_scale;
        let total_offset_x = base_offset_x + self.current_trans_x;
        let total_offset_y = base_offset_y + self.current_trans_y;

        // When this is the image-only canvas (draw_annotations == false),
        // draw the screenshot as the bottom layer.
        if !self.draw_annotations {
            if let Some(image) = self.image.as_ref() {
                let img_w = self.img_w as f32;
                let img_h = self.img_h as f32;
                let dest_size = Size::new(img_w * total_scale, img_h * total_scale);
                let dest_pos = Point::new(total_offset_x, total_offset_y);
                frame.draw_image(
                    Rectangle::new(dest_pos, dest_size),
                    image,
                );
            }
            return vec![frame.into_geometry()];
        }

        let transform = |bbox: &BoundingBox| -> (Point, Size) {
            (Point::new(
                bbox.x as f32 * total_scale + total_offset_x,
                bbox.y as f32 * total_scale + total_offset_y,
            ), Size::new(
                bbox.w as f32 * total_scale,
                bbox.h as f32 * total_scale,
            ))
        };

        if self.draw_annotations {
            // Pre-compute visible screen bounds in image coordinates to skip off-screen annotations
            let inv_scale = if total_scale > 0.0 { 1.0 / total_scale } else { 1.0 };
            let vis_left = (-total_offset_x) * inv_scale - 100.0;
            let vis_top = (-total_offset_y) * inv_scale - 100.0;
            let vis_right = (bounds.width - total_offset_x) * inv_scale + 100.0;
            let vis_bottom = (bounds.height - total_offset_y) * inv_scale + 100.0;

            for (line_idx, annotation) in self.annotations.iter().enumerate() {
                let bbox = &annotation.bbox;
                // Skip annotations entirely off-screen
                let bx2 = (bbox.x + bbox.w) as f32;
                let by2 = (bbox.y + bbox.h) as f32;
                let bx1 = bbox.x as f32;
                let by1 = bbox.y as f32;
                if bx2 < vis_left || bx1 > vis_right || by2 < vis_top || by1 > vis_bottom {
                    continue;
                }

                let (pt, sz) = transform(bbox);
                frame.fill_rectangle(pt, sz, Color::from_rgba(0.0, 0.0, 0.0, 0.39));  // argb(100,0,0,0)

                if let Some(line) = &annotation.line {
                    for (i, char_box) in line.char_boxes.iter().enumerate() {
                        let (pt_c, sz_c) = transform(char_box);

                        // Draw cursor highlight if this is the selected character
                        if self.cursor_pos == Some((line_idx, i)) {
                            let cursor_rect = CanvasPath::rectangle(pt_c, sz_c);
                            frame.fill(&cursor_rect, Color::from_rgba(1.0, 1.0, 0.0, 0.3));
                            frame.stroke(&cursor_rect,
                                CanvasStroke::default().with_color(Color::from_rgb(1.0, 1.0, 0.0)).with_width(2.0));
                        }

                        // Render character centered in its detection box
                        if let Some(_ch) = line.text.chars().nth(i) {
                            frame.fill_text(CanvasText {
                                content: _ch.to_string(),
                                position: Point::new(pt_c.x + sz_c.width / 2.0, pt_c.y + sz_c.height / 2.0),
                                max_width: 0.0,
                                color: Color::from_rgb(1.0, 0.467, 0.467),  // #FF7777
                                size: Pixels(sz_c.height * CANVAS_CHAR_RATIO),
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
    /// Decoded image, cached to avoid re-decoding PNG on every crop_character_image call.
    decoded_image: RefCell<Option<image::DynamicImage>>,
    pub img_w: u32,
    pub img_h: u32,
    /// Physical window dimensions. Used for computing font sizes in neighbor/alt panels.
    pub window_width: f32,
    pub window_height: f32,
    pub annotations: Rc<Vec<DetectedAnnotation>>,
    pub state: OcrOverlayState,
    pub selected_word: Option<SelectedWord>,
    pub alternatives_visible: bool,
    pub db: Arc<DictionaryDatabase>,
    pub deinflector: Arc<Deinflector>,
    /// The index of the character that should be scrolled into view in the neighbor panel.
    pub scroll_neighbor_to: Option<usize>,
    /// The index of the character that should be scrolled into view in the alt panel.
    pub scroll_alt_to: Option<usize>,
    /// Requested dictionary scroll delta (px). Set by L1/R1 gamepad, D/F keyboard.
    pub dict_scroll_request: Option<f32>,
    /// When true, the user is actively panning/zooming — annotation drawing is disabled.
    pub is_zooming: bool,
    /// Frames since last zoom/pan event — used to re-enable annotations after zoom ends.
    pub zoom_idle_frames: u32,
    /// Cached character preview image for the alternatives panel.
    /// Stores (line_idx, char_idx, handle) so we only regenerate when the selection changes.
    cached_preview: RefCell<Option<(usize, usize, iced::widget::image::Handle)>>,
}

impl OcrViewer {
    pub fn new(
        image_handle: iced::widget::image::Handle,
        image_bytes: Vec<u8>,
        img_w: u32,
        img_h: u32,
        window_width: f32,
        window_height: f32,
        annotations: Vec<DetectedAnnotation>,
        db: Arc<DictionaryDatabase>,
        deinflector: Arc<Deinflector>,
    ) -> Self {
        let mut state = OcrOverlayState::new();
        let line_results = annotations.iter().map(|a| a.line.clone()).collect::<Vec<_>>();
        state.set_line_results(line_results);
        state.ensure_cursor_position();
        state.img_w = img_w;
        state.img_h = img_h;
        Self {
            image_handle,
            image_bytes,
            decoded_image: RefCell::new(None),
            img_w,
            img_h,
            window_width,
            window_height,
            annotations: Rc::new(annotations),
            state,
            selected_word: None,
            alternatives_visible: false,
            db,
            deinflector,
            scroll_neighbor_to: None,
            scroll_alt_to: None,
            dict_scroll_request: None,
            is_zooming: false,
            zoom_idle_frames: 0,
            cached_preview: RefCell::new(None),
        }
    }

    /// Crop the screenshot to show the given character with padding.
    /// Returns an image Handle for the cropped region.
    pub fn crop_character_image(&self, line_idx: usize, char_idx: usize) -> Option<iced::widget::image::Handle> {
        let line = self.state.active_line_results.get(line_idx).and_then(|l| l.as_ref())?;
        let box_item = line.char_boxes.get(char_idx)?;

        // Lazily decode and cache the original image
        if self.decoded_image.borrow().is_none() {
            if let Ok(img) = image::load_from_memory(&self.image_bytes) {
                *self.decoded_image.borrow_mut() = Some(img);
            }
        }
        let img = self.decoded_image.borrow();
        let img = img.as_ref()?;
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
        self.set_cursor_pos(line_idx, char_idx);
        self.state.is_dictionary_visible = true;
        self.alternatives_visible = false;

        // Update gravity so panel opens on the opposite side of the character.
        let box_item = self.state.active_line_results.get(line_idx)
            .and_then(|l| l.as_ref())
            .and_then(|line| line.char_boxes.get(char_idx))
            .cloned();
        if let Some(b) = box_item {
            self.state.update_gravity(&b, self.panel_width());
        }
        self.do_lookup(line_idx, char_idx);
        self.compute_scroll_targets(line_idx, char_idx);
    }

    /// Move the cursor without opening the dictionary (for gamepad navigation).
    pub fn move_cursor_to(&mut self, line_idx: usize, char_idx: usize) {
        self.set_cursor_pos(line_idx, char_idx);
        // Highlight just the one character at the cursor position.
        self.state.update_highlight_coords(line_idx, char_idx, 1);
        self.compute_scroll_targets(line_idx, char_idx);
    }

    fn set_cursor_pos(&mut self, line_idx: usize, char_idx: usize) {
        self.state.current_tapped_line_idx = line_idx as isize;
        self.state.current_tapped_char_idx_in_line = char_idx as isize;
        self.state.current_tapped_idx = self.state.get_global_idx(line_idx, char_idx) as isize;
    }

    pub fn select_neighbor(&mut self, line_idx: usize, char_idx: usize) {
        let is_same = self.state.current_tapped_line_idx == line_idx as isize
            && self.state.current_tapped_char_idx_in_line == char_idx as isize;
        self.state.current_tapped_line_idx = line_idx as isize;
        self.state.current_tapped_char_idx_in_line = char_idx as isize;
        self.state.current_tapped_idx = self.state.get_global_idx(line_idx, char_idx) as isize;
        self.state.update_highlight_coords(line_idx, char_idx, 1);
        self.state.is_dictionary_visible = true;

        // Don't change gravity when selecting from neighbor/alternative views —
        // the panel is already open and positioned from the initial character click.

        if is_same {
            // Clicking the already-selected character toggles the
            // alternatives panel: open if closed, close if open.
            self.alternatives_visible = !self.alternatives_visible;
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
            // Highlight the full matched word in yellow (like Kotlin)
            self.state.update_highlight_coords(line_idx, char_idx, result.max_len);
        } else {
            self.state.cached_entries.clear();
            self.state.current_word_length = 1;
            self.state.update_highlight_coords(line_idx, char_idx, 1);
        }
    }

    /// Total width of the panel (dict + neighbors + alt + spacing + padding).
    /// Must match the values used in view().
    fn panel_width(&self) -> f32 {
        let dict_width: f32 = 300.0;
        let neigh_width: f32 = 42.0;
        let alt_width: f32 = 42.0;
        let spacing: f32 = 2.0; // Row::new().spacing(2)
        let padding: f32 = 4.0; // Container::padding(2) on each side
        dict_width + neigh_width + alt_width + spacing + padding
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

    /// Scroll the dictionary panel to the top.
    pub fn scroll_dict_to_top_task(&self) -> iced::Task<Message> {
        operate(scroll_to(
            Id::new("dict_scroll"),
            AbsoluteOffset { x: Some(0.0), y: Some(0.0) },
        ))
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

        // Gravity is computed in select_character/update_gravity with the full
        // transform (base + pan/zoom), so the panel opens on the opposite side
        // of the character's actual screen position.
        let panel_on_right = self.state.last_landscape_gravity == Gravity::End;
        let overlay = OverlayProgram {
            annotations: Rc::clone(&self.annotations),
            img_w: self.img_w,
            img_h: self.img_h,
            image: Some(self.image_handle.clone()),
            panel_visible: has_panel,
            panel_on_right,
            dict_width: 300.0,
            alternatives_visible: self.alternatives_visible,
            cursor_pos: self.state.current_cursor(),
            current_scale: self.state.current_scale,
            current_trans_x: self.state.current_trans_x,
            current_trans_y: self.state.current_trans_y,
            draw_annotations: !self.is_zooming,
            is_zooming: self.is_zooming,
            handle_pan_zoom: true,
        };

        // The image is drawn in a SEPARATE canvas underneath, because tiny_skia
        // always composites images after primitives. Two separate Canvas widgets
        // in a Stack gives us correct z-ordering: image (bottom) → annotations (top).
        let image_canvas = Canvas::new(OverlayProgram {
            annotations: Rc::clone(&self.annotations),
            img_w: self.img_w,
            img_h: self.img_h,
            image: Some(self.image_handle.clone()),
            panel_visible: false,
            panel_on_right: false,
            dict_width: 300.0,
            alternatives_visible: false,
            cursor_pos: None,
            current_scale: self.state.current_scale,
            current_trans_x: self.state.current_trans_x,
            current_trans_y: self.state.current_trans_y,
            draw_annotations: false,
            is_zooming: self.is_zooming,
            handle_pan_zoom: false,
        }).width(Length::Fill).height(Length::Fill);

        let annotation_canvas = Canvas::new(overlay).width(Length::Fill).height(Length::Fill);

        if !has_panel {
            return Container::new(Stack::new().push(image_canvas).push(annotation_canvas))
                .width(Length::Fill)
                .height(Length::Fill)
                .into();
        }

        // Build panel components
        let dict_panel = self.dictionary_panel(if !self.state.cached_entries.is_empty() {
            self.state.cached_entries.clone()
        } else {
            Vec::new()
        });
        let neigh_panel = self.neighbor_panel();

        // Crop the character preview image from the screenshot (cached per selection)
        let preview_image = self.selected_word.as_ref().and_then(|sw| {
            let mut cache = self.cached_preview.borrow_mut();
            if let Some((cached_li, cached_ci, handle)) = cache.as_ref() {
                if *cached_li == sw.line_idx && *cached_ci == sw.char_idx {
                    return Some(handle.clone());
                }
            }
            let handle = self.crop_character_image(sw.line_idx, sw.char_idx)?;
            *cache = Some((sw.line_idx, sw.char_idx, handle.clone()));
            Some(handle)
        });
        let alt_panel = self.alternatives_panel(preview_image.as_ref());

        let dict_width = Pixels(300.0);

        // Stable inner row: neighbors + dictionary. This never changes
        // structure, so the Scrollables inside never reset.
        let mut inner_row = Row::new().spacing(2);
        if panel_on_right {
            inner_row = inner_row.push(neigh_panel);
            inner_row = inner_row.push(dict_panel.width(dict_width));
        } else {
            inner_row = inner_row.push(dict_panel.width(dict_width));
            inner_row = inner_row.push(neigh_panel);
        }

        // The panel container wraps the stable row.
        let panel = Container::new(inner_row).padding(2).style(move |_t: &Theme| {
            container::Style {
                background: Some(iced::Background::Color(Color::from_rgba(
                    25.0 / 255.0,
                    25.0 / 255.0,
                    25.0 / 255.0,
                    245.0 / 255.0,
                ))), // argb(245, 25, 25, 25)
                border: iced::Border {
                    radius: 4.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        });

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
            // Estimate neighbor panel width for spacer (box_size = window_width/11 clamped)
            let estimated_neigh = (self.window_width / 11.0).clamp(24.0, 40.0);
            // Add extra margin so the alt panel doesn't overlap with the neighbor panel
            let alt_margin = 4.0;
            let panel_content_w = dict_width + Pixels(2.0) + Pixels(estimated_neigh) + Pixels(alt_margin);

            // The alt panel Container.
            let alt_container = Container::new(alt_panel)
                .height(Length::Fill)
                .style(move |_t: &Theme| container::Style {
                    background: Some(iced::Background::Color(Color::from_rgba(
                        25.0 / 255.0,
                        25.0 / 255.0,
                        25.0 / 255.0,
                        245.0 / 255.0,
                    ))), // argb(245, 25, 25, 25)
                    ..Default::default()
                });

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

        // Root Stack: image (bottom) → annotations (middle) → panel content (top).
        // Two separate canvases because tiny_skia composites images after primitives.
        Container::new(
            Stack::new()
                .push(image_canvas)
                .push(annotation_canvas)
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
        Container::new(
            Scrollable::new(content)
                .id(Id::new("dict_scroll"))
                .direction(iced::widget::scrollable::Direction::Vertical(
                    iced::widget::scrollable::Scrollbar::hidden(),
                )),
        )
        .width(Length::Fill)
        .padding(4)
        .style(move |_t: &Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(
                25.0 / 255.0,
                25.0 / 255.0,
                25.0 / 255.0,
                245.0 / 255.0,
            ))), // argb(245, 25, 25, 25)
            ..Default::default()
        })
    }

    fn headword_section<'a>(&'a self, group: FormattedReadingGroup) -> Container<'a, Message> {
        let cyan = Color::from_rgb(0.0, 1.0, 1.0);      // Android CYAN
        let gray = Color::from_rgb(0.75, 0.75, 0.75);   // Android LTGRAY (#BEBEBE)
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
        let cyan = Color::from_rgb(0.0, 1.0, 1.0); // Android CYAN
        let gray = Color::from_rgb(0.75, 0.75, 0.75); // Android LTGRAY (#BEBEBE)
        let white = Color::WHITE;
        let mut content = Column::new().spacing(3);

        if !sg.tags.is_empty() {
            let mut tag_row = Row::new().spacing(3).align_y(alignment::Vertical::Center);
            for tag in &sg.tags {
                let bg = Self::tag_color(tag);
                tag_row = tag_row.push(
                    Container::new(Text::new(tag.clone()).size(11).color(white)
                        .font(IcedFont { weight: iced::font::Weight::Bold, ..IcedFont::default() }))
                    .padding([1.0, 1.0])
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
        // Compute button size dynamically based on window width
        let item_size = self.window_width / 11.0;
        let box_size = item_size.clamp(24.0, 40.0);
        let mut content = Column::new().spacing(2);
        for line in state {
            for cs in line.chars {
                let msg = Message::SelectNeighbor(line.line_idx, cs.char_idx);
                let is_selected = cs.is_selected;
                let text = cs.text.clone();

                let btn: Element<'a, Message> = Container::new(
                    Text::new(text)
                        .size(Pixels(box_size * BUTTON_CHAR_RATIO))
                        .color(if is_selected { Color::BLACK } else { Color::WHITE })
                        .align_x(alignment::Horizontal::Center)
                        .align_y(alignment::Vertical::Center),
                )
                .width(Pixels(box_size))
                .height(Pixels(box_size))
                .align_x(alignment::Horizontal::Center)
                .align_y(alignment::Vertical::Center)
                .style(move |_t: &Theme| {
                    if is_selected {
                        container::Style {
                            background: Some(iced::Background::Color(
                                Color::from_rgb(1.0, 1.0, 0.0), // YELLOW
                            )),
                            ..Default::default()
                        }
                    } else {
                        container::Style {
                            background: Some(iced::Background::Color(
                                Color::from_rgb(0.255, 0.255, 0.255), // argb(255, 65, 65, 65)
                            )),
                            ..Default::default()
                        }
                    }
                })
                .into();

                content = content.push(
                    TapOrDrag::new(btn, msg),
                );
            }
        }
        // Scrollable with hidden scrollbar — sizes to content so buttons stay square
        Container::new(
            Scrollable::new(content)
                .id(Id::new("neighbor_scroll"))
                .direction(iced::widget::scrollable::Direction::Vertical(
                    iced::widget::scrollable::Scrollbar::hidden(),
                )),
        )
        .height(Length::Fill)
        .style(move |_t: &Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(
                25.0 / 255.0,
                25.0 / 255.0,
                25.0 / 255.0,
                245.0 / 255.0,
            ))), // argb(245, 25, 25, 25)
            ..Default::default()
        })
    }

    fn alternatives_panel<'a>(
        &'a self,
        preview_image: Option<&iced::widget::image::Handle>,
    ) -> Container<'a, Message> {
        let mut content = Column::new().spacing(2);
        // Compute button size dynamically based on window width (same as neighbor_panel)
        let item_size = self.window_width / 11.0;
        let box_size = item_size.clamp(24.0, 40.0);

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
                let msg = Message::SelectAlternative(c.char);
                let is_selected = c.is_selected;
                let ch = c.char;

                let btn: Element<'a, Message> = Container::new(
                    Text::new(ch.to_string())
                        .size(Pixels(box_size * BUTTON_CHAR_RATIO))
                        .color(if is_selected { Color::BLACK } else { Color::WHITE })
                        .align_x(alignment::Horizontal::Center)
                        .align_y(alignment::Vertical::Center),
                )
                .width(Pixels(box_size))
                .height(Pixels(box_size))
                .align_x(alignment::Horizontal::Center)
                .align_y(alignment::Vertical::Center)
                .style(move |_t: &Theme| {
                    if is_selected {
                        container::Style {
                            background: Some(iced::Background::Color(
                                Color::from_rgb(1.0, 1.0, 0.0), // YELLOW
                            )),
                            ..Default::default()
                        }
                    } else {
                        container::Style {
                            background: Some(iced::Background::Color(
                                Color::from_rgb(0.255, 0.255, 0.255), // argb(255, 65, 65, 65)
                            )),
                            ..Default::default()
                        }
                    }
                })
                .into();

                content = content.push(
                    TapOrDrag::new(btn, msg),
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
        .style(move |_t: &Theme| container::Style {
            background: Some(iced::Background::Color(Color::from_rgba(
                25.0 / 255.0,
                25.0 / 255.0,
                25.0 / 255.0,
                245.0 / 255.0,
            ))), // argb(245, 25, 25, 25)
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// TapOrDrag — fires tap only if pointer didn't move between press and release
// ---------------------------------------------------------------------------

use iced::advanced::widget::tree::{self, Tree};
use iced::advanced::{self, Layout, Widget};
use iced::advanced::widget::Operation;
use iced::advanced::Clipboard;

/// A widget that distinguishes taps from drags.
/// Only fires `on_tap` if the pointer moved less than the threshold
/// between press and release. This allows the parent Scrollable to
/// handle drag-to-scroll while still supporting tap-to-activate.
pub struct TapOrDrag<'a, Message> {
    content: Element<'a, Message>,
    on_tap: Message,
    threshold: f32,
}

impl<'a, Message: Clone> TapOrDrag<'a, Message> {
    pub fn new(content: impl Into<Element<'a, Message>>, on_tap: Message) -> Self {
        Self {
            content: content.into(),
            on_tap,
            threshold: 5.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TapState {
    pressed: bool,
    start_x: f32,
    start_y: f32,
    is_tap: bool,
    /// Accumulated movement distance since press.
    accumulated: f32,
}

impl<'a, Message: Clone + 'static> Widget<Message, Theme, Renderer> for TapOrDrag<'a, Message> {
    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
    }

    fn size_hint(&self) -> Size<Length> {
        self.content.as_widget().size_hint()
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &iced::advanced::layout::Limits,
    ) -> iced::advanced::layout::Node {
        self.content.as_widget_mut().layout(&mut tree.children[0], renderer, limits)
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &iced::advanced::renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.content.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            style,
            layout,
            cursor,
            viewport,
        );
    }

    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<TapState>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(TapState::default())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_ref(&self.content));
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &iced::Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut advanced::Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_mut::<TapState>();

        match event {
            iced::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if cursor.position_over(layout.bounds()).is_some() {
                    state.pressed = true;
                    // Use the actual cursor position, not clamped to bounds
                    if let Some(pos) = cursor.position() {
                        state.start_x = pos.x;
                        state.start_y = pos.y;
                    }
                    state.accumulated = 0.0;
                    state.is_tap = true;
                }
            }
            iced::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                if state.pressed {
                    if let Some(pos) = cursor.position() {
                        let dx = pos.x - state.start_x;
                        let dy = pos.y - state.start_y;
                        state.accumulated += (dx * dx + dy * dy).sqrt();
                        if state.accumulated > self.threshold {
                            state.is_tap = false;
                        }
                    }
                }
            }
            iced::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                let was_tap = state.is_tap;
                state.pressed = false;
                state.is_tap = false;
                state.accumulated = 0.0;
                if was_tap {
                    shell.publish(self.on_tap.clone());
                }
            }
            iced::Event::Touch(touch::Event::FingerPressed { .. }) => {
                if cursor.position_over(layout.bounds()).is_some() {
                    state.pressed = true;
                    if let Some(pos) = cursor.position() {
                        state.start_x = pos.x;
                        state.start_y = pos.y;
                    }
                    state.accumulated = 0.0;
                    state.is_tap = true;
                }
            }
            iced::Event::Touch(touch::Event::FingerMoved { .. }) => {
                if state.pressed {
                    if let Some(pos) = cursor.position() {
                        let dx = pos.x - state.start_x;
                        let dy = pos.y - state.start_y;
                        state.accumulated += (dx * dx + dy * dy).sqrt();
                        if state.accumulated > self.threshold {
                            state.is_tap = false;
                        }
                    }
                }
            }
            iced::Event::Touch(touch::Event::FingerLifted { .. }) => {
                let was_tap = state.is_tap;
                state.pressed = false;
                state.is_tap = false;
                state.accumulated = 0.0;
                if was_tap {
                    shell.publish(self.on_tap.clone());
                }
            }
            _ => {}
        }

        self.content.as_widget_mut().update(
            &mut tree.children[0],
            event,
            layout,
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.content.as_widget().mouse_interaction(
            &tree.children[0],
            layout,
            cursor,
            viewport,
            renderer,
        )
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        self.content
            .as_widget_mut()
            .operate(&mut tree.children[0], layout, renderer, operation);
    }
}

impl<'a, Message: Clone + 'static> From<TapOrDrag<'a, Message>> for Element<'a, Message> {
    fn from(widget: TapOrDrag<'a, Message>) -> Self {
        Element::new(widget)
    }
}
