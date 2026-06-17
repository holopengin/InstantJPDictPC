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
    pub is_vertical: bool,
    pub chunk_boxes: Vec<BoundingBox>,
}

#[derive(Debug, Clone)]
pub struct DetectedAnnotation {
    pub bbox: BoundingBox,
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
    SelectAlternative(char),
    ToggleAlternatives,
    Back,
}
