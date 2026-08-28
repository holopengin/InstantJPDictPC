// Shared data model types for the accessibility daemon

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub confidence: f32,
}

impl BoundingBox {
    pub fn new(x: i32, y: i32, w: i32, h: i32, confidence: f32) -> Self {
        BoundingBox { x, y, w, h, confidence }
    }

    pub fn left(&self) -> i32 { self.x }
    pub fn top(&self) -> i32 { self.y }
    pub fn right(&self) -> i32 { self.x + self.w }
    pub fn bottom(&self) -> i32 { self.y + self.h }
    pub fn area(&self) -> i32 { self.w * self.h }
}

/// Minimum-area rotated rectangle around a detected text contour.
/// `angle` is the rotation of the LONG axis from the +x axis, in radians,
/// in y-down image coordinates (positive = toward +y, i.e. clockwise
/// visually). `w` is the extent along the long axis.
#[derive(Debug, Clone, Copy)]
pub struct RotatedBox {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub angle: f32,
    pub confidence: f32,
}

impl RotatedBox {
    pub fn new(cx: f32, cy: f32, w: f32, h: f32, mut angle: f32, confidence: f32) -> Self {
        // Normalize the long-axis angle to [-90°, 90°): 162° ≡ -18°.
        while angle >= std::f32::consts::FRAC_PI_2 {
            angle -= std::f32::consts::PI;
        }
        while angle < -std::f32::consts::FRAC_PI_2 {
            angle += std::f32::consts::PI;
        }
        RotatedBox { cx, cy, w, h, angle, confidence }
    }

    /// Axis-aligned bounding box (x, y, w, h) in image coordinates.
    pub fn aabb(&self) -> (f32, f32, f32, f32) {
        let (ux, uy) = (self.angle.cos(), self.angle.sin());
        let (vx, vy) = (-uy, ux);
        let hw = self.w / 2.0;
        let hh = self.h / 2.0;
        let corners = [
            (self.cx + ux * hw + vx * hh, self.cy + uy * hw + vy * hh),
            (self.cx - ux * hw + vx * hh, self.cy - uy * hw + vy * hh),
            (self.cx + ux * hw - vx * hh, self.cy + uy * hw - vy * hh),
            (self.cx - ux * hw - vx * hh, self.cy - uy * hw - vy * hh),
        ];
        let min_x = corners.iter().map(|c| c.0).fold(f32::MAX, f32::min);
        let min_y = corners.iter().map(|c| c.1).fold(f32::MAX, f32::min);
        let max_x = corners.iter().map(|c| c.0).fold(f32::MIN, f32::max);
        let max_y = corners.iter().map(|c| c.1).fold(f32::MIN, f32::max);
        (min_x, min_y, max_x - min_x, max_y - min_y)
    }

    /// Text runs vertically (long axis nearer the y-axis).
    pub fn is_vertical(&self) -> bool {
        self.angle.abs() > std::f32::consts::FRAC_PI_4
    }

    /// True when the text axis is meaningfully off both horizontal and
    /// vertical — i.e. the crop needs un-rotating before recognition.
    pub fn is_rotated(&self) -> bool {
        const ROT_EPS: f32 = 2.0 * std::f32::consts::PI / 180.0;
        let a = self.angle.abs();
        a > ROT_EPS && (std::f32::consts::FRAC_PI_2 - a).abs() > ROT_EPS
    }

    /// Rotation (radians; positive = content rotates toward +y, i.e.
    /// clockwise visually, about the crop center) that makes the text axis
    /// axis-aligned: horizontal text → horizontal, vertical text → vertical.
    /// Flip-safe: always rotates by the smallest angle toward the nearest
    /// axis (a line at -89° rotates by -1°, never 179°).
    pub fn unrotate_angle(&self) -> f32 {
        if self.is_vertical() {
            if self.angle >= 0.0 {
                std::f32::consts::FRAC_PI_2 - self.angle
            } else {
                -std::f32::consts::FRAC_PI_2 - self.angle
            }
        } else {
            -self.angle
        }
    }

    /// Map a char box given in un-rotated crop coordinates back to image
    /// coordinates. `crop_w/crop_h` = un-rotated crop size (same dims as the
    /// original crop — rotation keeps dimensions), `crop_ox/crop_oy` = crop
    /// origin (this rect's AABB top-left) in image coords.
    pub fn map_char_box(
        &self,
        crop_w: u32, crop_h: u32, crop_ox: u32, crop_oy: u32,
        bx: i32, by: i32, bw: i32, bh: i32,
    ) -> BoundingBox {
        let theta = self.unrotate_angle();
        let (s, c) = theta.sin_cos();
        // Inverse of the un-rotate: crop coords → rect (AABB) coords.
        let cw = crop_w as f32 / 2.0;
        let ch = crop_h as f32 / 2.0;
        let corners = [
            (bx as f32, by as f32),
            (bx as f32 + bw as f32, by as f32),
            (bx as f32 + bw as f32, by as f32 + bh as f32),
            (bx as f32, by as f32 + bh as f32),
        ];
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for (px, py) in corners {
            let dx = px - cw;
            let dy = py - ch;
            let rx = c * dx + s * dy; // R(-theta) = [c, s; -s, c]
            let ry = -s * dx + c * dy;
            let ix = crop_ox as f32 + cw + rx;
            let iy = crop_oy as f32 + ch + ry;
            min_x = min_x.min(ix);
            min_y = min_y.min(iy);
            max_x = max_x.max(ix);
            max_y = max_y.max(iy);
        }
        BoundingBox::new(
            min_x.round() as i32,
            min_y.round() as i32,
            (max_x - min_x).round().max(1.0) as i32,
            (max_y - min_y).round().max(1.0) as i32,
            self.confidence,
        )
    }
}

/// Result of line detection: axis-aligned boxes (for display/navigation) plus
/// the aligned rotated rectangles (for crop un-rotation). Same order.
#[derive(Debug, Default, Clone)]
pub struct DetectionResult {
    pub boxes: Vec<BoundingBox>,
    pub rotated: Vec<RotatedBox>,
}

// ---------------------------------------------------------------------------
// OCR results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CharCandidate {
    pub char: char,
    pub score: f32,
    pub box_coords: [f32; 4],
    pub alternatives: Vec<(char, f32)>,
}

#[derive(Debug, Clone)]
pub struct LineResult {
    pub text: String,
    pub char_boxes: Vec<BoundingBox>,
    pub alternatives: Vec<Vec<(char, f32)>>,
    /// Path to the crop's `.txt` sidecar in /tmp (dataset collection).
    /// The viewer rewrites it when the user picks an alternative, so the
    /// saved label follows the corrected text.
    pub sample_txt: Option<std::path::PathBuf>,
    pub is_vertical: bool,
    pub chunk_boxes: Vec<BoundingBox>,
}

#[derive(Debug, Clone)]
pub struct DetectedAnnotation {
    pub bbox: BoundingBox,
    /// Rotated quad geometry for angled lines — `Some` only when the text
    /// axis is meaningfully off horizontal/vertical. The viewer draws the
    /// quad instead of the axis-aligned AABB when present.
    pub quad: Option<RotatedBox>,
    pub line: Option<LineResult>,
}

// ---------------------------------------------------------------------------
// Selection state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SelectedWord {
    pub line_idx: usize,
    pub char_idx: usize,
    pub text: String,
    pub box_item: BoundingBox,
}

#[derive(Debug, Clone)]
pub struct NeighborChar {
    pub text: String,
    pub is_selected: bool,
    pub line_idx: usize,
    pub char_idx: usize,
}

#[derive(Debug, Clone)]
pub struct NeighborLine {
    pub chars: Vec<NeighborChar>,
    pub line_idx: usize,
}

#[derive(Debug, Clone)]
pub struct AlternativeChar {
    pub char: char,
    pub is_selected: bool,
}

#[derive(Debug, Clone)]
pub struct AlternativesUiState {
    pub candidates: Vec<AlternativeChar>,
    pub show_manual_input: bool,
}

// ---------------------------------------------------------------------------
// UI enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gravity {
    Start,
    End,
    Top,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecognitionMode {
    Horizontal,
    Vertical,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamepadAction {
    None,
    NavigateLeft,
    NavigateRight,
    NavigateUp,
    NavigateDown,
    Confirm,
    Back,
    ScrollUp,
    ScrollDown,
}

// ---------------------------------------------------------------------------
// Dictionary formatting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum DefinitionNode {
    Text(String),
    Ruby { term: String, reading: String, is_mini: bool },
    Tag { text: String, category: String },
    Example {
        japanese: Option<String>,
        english: Option<String>,
        content: Option<Vec<DefinitionNode>>,
    },
    ListBlock {
        items: Vec<Vec<DefinitionNode>>,
        block_type: Option<String>,
    },
    Table {
        rows: Vec<Vec<Vec<DefinitionNode>>>,
    },
    Group {
        nodes: Vec<DefinitionNode>,
        is_inline: bool,
    },
}

#[derive(Debug, Clone)]
pub struct FormattedSense {
    pub index: usize,
    pub nodes: Vec<DefinitionNode>,
}

#[derive(Debug, Clone)]
pub struct FormattedSenseGroup {
    pub tags: Vec<String>,
    pub senses: Vec<FormattedSense>,
    pub is_forms: bool,
}

#[derive(Debug, Clone)]
pub struct FormattedHeadword {
    pub kanji: String,
    pub onyomi: Option<String>,
    pub kunyomi: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FormattedReadingGroup {
    pub reading: String,
    pub headwords: Vec<FormattedHeadword>,
    pub sense_groups: Vec<FormattedSenseGroup>,
    pub is_kanji_entry: bool,
}

#[derive(Debug, Clone)]
pub struct FormattedEntry {
    pub term: String,
    pub reading_groups: Vec<FormattedReadingGroup>,
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Message {
    SelectCharacter(usize, usize),
    SelectNeighbor(usize, usize),
    SelectAlternative(char),
    Back,
    Navigate(GamepadAction),
    /// Zoom the image by a delta (positive = zoom in), centered on the given cursor position.
    ZoomOnCursor { delta: f32, cursor_x: f32, cursor_y: f32 },
    /// Start panning — record the initial cursor position.
    PanStart { start_x: f32, start_y: f32 },
    /// Pan by the given delta.
    PanDelta { dx: f32, dy: f32 },
    /// End panning.
    PanEnd,
    /// Set the zoom scale directly (for pinch zoom).
    SetScale { scale: f32 },
    /// Pinch zoom gesture: new scale + pan delta around focus point.
    PinchZoom { scale_factor: f32, focus_x: f32, focus_y: f32, prev_focus_x: f32, prev_focus_y: f32, base_offset_y: f32 },
    /// Pinch zoom ended (finger lifted).
    PinchEnd,
    /// Periodic tick to detect when zoom/pan has stopped.
    ZoomTick,
    /// The window was resized.
    WindowResized { width: f32, height: f32 },
    /// Line detection complete — show bounding boxes (no text yet).
    OcrDetectionComplete(Vec<DetectedAnnotation>),
    /// One line's character recognition complete.
    OcrRecognitionResult(usize, DetectedAnnotation),
    /// All OCR processing is done.
    OcrAllDone,
    /// Frame tick — drains bootstrap/OCR channels so results stream smoothly.
    Tick,
}
