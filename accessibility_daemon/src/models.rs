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
}

/// Mobile `RotatedGeometry.VERTICAL_MIN_ASPECT` (#28/#53): a frame whose
/// reading axis is at least 1.25x its cross axis is vertical; near-square
/// counts as horizontal so a lone upright character is never fed sideways.
pub const VERTICAL_MIN_ASPECT: f32 = 1.25;

/// Mobile `RotatedGeometry.AXIS_ALIGNED_TOL_DEG`: a fit within this many
/// degrees of the upright axes stays on the axis-aligned path (no un-rotate,
/// no rotated crop).
pub const AXIS_ALIGNED_TOL_RAD: f32 = 1.0 * std::f32::consts::PI / 180.0;

/// Extra straightness allowance for the fit's own quantization. The boundary
/// points are pixel outer corners (±0.5px), so a visually level blob can fit
/// a fraction of a degree off before any real tilt exists; scaled by the
/// frame's long side this is the "indistinguishable from straight" band.
/// (Mobile's constant 1° tolerance is the floor; this only widens it for
/// short noisy frames such as UI text.)
pub const AXIS_ALIGNED_QUANT_TOL_PX: f32 = 1.5;

/// One line's upright-crop frame — the mobile `JpDictQuad`, stored as centre +
/// local extents + the local x axis angle. Corners `c0..c3` are the crop's
/// local `(0,0),(w,0),(w,h),(0,h)` in source pixels: `w` is the crop's x axis
/// (the reading axis for a horizontal line, the cross axis for a vertical
/// one) and `h` its y axis. `angle` is the x axis angle from +x in radians,
/// y-down (positive = clockwise visually).
#[derive(Debug, Clone, Copy, PartialEq)]
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
        // Normalize the x-axis angle to [-90°, 90°): 162° ≡ -18°. The fit
        // always orients local x to the right, so this is a no-op there.
        while angle >= std::f32::consts::FRAC_PI_2 {
            angle -= std::f32::consts::PI;
        }
        while angle < -std::f32::consts::FRAC_PI_2 {
            angle += std::f32::consts::PI;
        }
        RotatedBox { cx, cy, w, h, angle, confidence }
    }

    /// Unit local x axis (left → right in crop space).
    pub fn x_axis(&self) -> (f32, f32) {
        (self.angle.cos(), self.angle.sin())
    }

    /// Unit local y axis (top → bottom in crop space).
    pub fn y_axis(&self) -> (f32, f32) {
        let (x, y) = self.x_axis();
        (-y, x)
    }

    /// Local crop-space origin `c0` in source pixels.
    pub fn origin(&self) -> (f32, f32) {
        let (xa, ya) = (self.x_axis(), self.y_axis());
        (
            self.cx - self.w / 2.0 * xa.0 - self.h / 2.0 * ya.0,
            self.cy - self.w / 2.0 * xa.1 - self.h / 2.0 * ya.1,
        )
    }

    /// Axis-aligned bounding box (x, y, w, h) in image coordinates.
    pub fn aabb(&self) -> (f32, f32, f32, f32) {
        let (ux, uy) = self.x_axis();
        let (vx, vy) = self.y_axis();
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

    /// Mobile `RotatedGeometry.isVertical`: the frame's own local sizes decide
    /// (localHeight >= localWidth * 1.25), like the axis-aligned rule.
    pub fn is_vertical(&self) -> bool {
        self.h >= self.w * VERTICAL_MIN_ASPECT
    }

    /// True when the text axis is meaningfully off both horizontal and
    /// vertical — i.e. the crop needs un-rotating before recognition. The
    /// reading axis' deviation from its upright axis is exactly `angle` for
    /// both orientations (the fit orients vertical frames the same way).
    pub fn is_rotated(&self) -> bool {
        !self.is_axis_aligned()
    }

    /// Mobile `RotatedGeometry.isAxisAligned` (`|tiltDeg| <= 1°`), widened by
    /// the fit's own half-pixel quantization over the frame's long side: a
    /// boundary-corner fit of a visually level blob is indistinguishable from
    /// straight inside that band, so short noisy frames (UI text, tiny
    /// fragments) must not be un-rotated for recognition.
    pub fn is_axis_aligned(&self) -> bool {
        let long = self.w.max(self.h).max(1.0);
        let quantization = (AXIS_ALIGNED_QUANT_TOL_PX / long).atan();
        self.angle.abs() <= AXIS_ALIGNED_TOL_RAD.max(quantization)
    }

    /// Mobile `RotatedGeometry.unclip`: grow both local axes by
    /// `localArea × ratio / localPerimeter` per side.
    pub fn unclip(&self, ratio: f32) -> RotatedBox {
        let (w, h) = (self.w, self.h);
        if w <= 1e-6 || h <= 1e-6 {
            return *self;
        }
        let expand = w * h * ratio / (2.0 * (w + h));
        self.inset(-expand, -expand)
    }

    /// Mobile `RotatedGeometry.inset`: shrink (positive) or grow (negative)
    /// the frame along its local axes. The centre is preserved.
    pub fn inset(&self, x_inset: f32, y_inset: f32) -> RotatedBox {
        RotatedBox {
            w: self.w - 2.0 * x_inset,
            h: self.h - 2.0 * y_inset,
            ..*self
        }
    }

    /// Mobile `RotatedGeometry.containsPoint`, tested on the frame's axes.
    pub fn contains_point(&self, p: (f32, f32)) -> bool {
        let (xa, ya) = (self.x_axis(), self.y_axis());
        let (dx, dy) = (p.0 - self.origin().0, p.1 - self.origin().1);
        let lx = dx * xa.0 + dy * xa.1;
        let ly = dx * ya.0 + dy * ya.1;
        lx >= 0.0 && lx <= self.w && ly >= 0.0 && ly <= self.h
    }

    /// Mobile `RotatedGeometry.mapLocalRect`: a box in the frame's local crop
    /// space as a source-space AABB (char boxes). The default axis-aligned
    /// frame maps one-to-one.
    pub fn map_local_rect(&self, bx: f32, by: f32, bw: f32, bh: f32) -> BoundingBox {
        let (xa, ya) = (self.x_axis(), self.y_axis());
        let (ox, oy) = self.origin();
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        for (lx, ly) in [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)] {
            let ix = ox + lx * xa.0 + ly * ya.0;
            let iy = oy + lx * xa.1 + ly * ya.1;
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

/// Mobile `RotatedGeometry.filterEnclosingBlobs`: drop frames that enclose
/// several smaller, Line-shaped frames (a DB blob that merged neighbouring
/// lines fits one large frame while those lines arrive as their own fits).
/// Enclosed frames must be at most 0.6x the area and at least 2:1 elongated.
pub fn filter_enclosing_blobs(quads: &[RotatedBox]) -> Vec<RotatedBox> {
    const ENCLOSED_AREA_FRACTION: f32 = 0.6;
    const ENCLOSED_MIN_ASPECT: f32 = 2.0;
    const ENCLOSED_MIN_COUNT: usize = 2;
    if quads.len() <= ENCLOSED_MIN_COUNT {
        return quads.to_vec();
    }
    let areas: Vec<f32> = quads.iter().map(|q| q.w * q.h).collect();
    let mut out = Vec::with_capacity(quads.len());
    for (i, quad) in quads.iter().enumerate() {
        let mut enclosed = 0usize;
        for (j, inner) in quads.iter().enumerate() {
            if j == i || areas[j] >= areas[i] * ENCLOSED_AREA_FRACTION {
                continue;
            }
            if inner.w.max(inner.h) < ENCLOSED_MIN_ASPECT * inner.w.min(inner.h) {
                continue;
            }
            if !quad.contains_point((inner.cx, inner.cy)) {
                continue;
            }
            enclosed += 1;
            if enclosed >= ENCLOSED_MIN_COUNT {
                break;
            }
        }
        if enclosed < ENCLOSED_MIN_COUNT {
            out.push(*quad);
        }
    }
    out
}

/// Mobile `RotatedGeometry.convexHull` (monotone chain): hull of interleaved
/// points with collinear points dropped, or None when fewer than three
/// distinct non-collinear points exist.
fn convex_hull(points: &[(f32, f32)]) -> Option<Vec<(f32, f32)>> {
    if points.len() < 3 {
        return None;
    }
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&a, &b| {
        points[a]
            .0
            .total_cmp(&points[b].0)
            .then(points[a].1.total_cmp(&points[b].1))
    });
    let cross = |o: usize, a: usize, b: usize| -> f32 {
        (points[a].0 - points[o].0) * (points[b].1 - points[o].1)
            - (points[a].1 - points[o].1) * (points[b].0 - points[o].0)
    };
    let mut hull: Vec<usize> = Vec::with_capacity(2 * points.len());
    for &p in &order {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    if hull.len() < 3 {
        return None;
    }
    let lower = hull.len() + 1;
    for &p in order.iter().rev().skip(1) {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop(); // the last point repeats the first
    Some(hull.into_iter().map(|i| points[i]).collect())
}

/// Mobile `RotatedGeometry.fitQuad`: fit the minimum-area rectangle to the
/// boundary points (already in source pixels), returning the upright-crop
/// frame. Rotation is recovered by testing every convex-hull edge direction;
/// the reading axis is the long axis when it is vertical and elongated
/// enough, otherwise the fitted axis closer to horizontal (so near-square
/// blobs are never turned sideways).
pub fn fit_quad(points: &[(f32, f32)]) -> Option<RotatedBox> {
    const EPS: f32 = 1e-6;
    if points.len() < 3 {
        return None;
    }
    let hull = convex_hull(points)?;
    let h = hull.len();

    let mut best_area = f32::MAX;
    let (mut cx, mut cy, mut ax, mut ay) = (0.0f32, 0.0f32, 1.0f32, 0.0f32);
    let (mut a_len, mut b_len) = (0.0f32, 0.0f32);
    for i in 0..h {
        let j = (i + 1) % h;
        let (mut dx, mut dy) = (hull[j].0 - hull[i].0, hull[j].1 - hull[i].1);
        let len = (dx * dx + dy * dy).sqrt();
        if len <= EPS {
            continue;
        }
        dx /= len;
        dy /= len;
        let (nx, ny) = (-dy, dx);
        let (mut t_min, mut t_max) = (f32::MAX, f32::MIN);
        let (mut s_min, mut s_max) = (f32::MAX, f32::MIN);
        for p in &hull {
            let t = p.0 * dx + p.1 * dy;
            let s = p.0 * nx + p.1 * ny;
            t_min = t_min.min(t);
            t_max = t_max.max(t);
            s_min = s_min.min(s);
            s_max = s_max.max(s);
        }
        let area = (t_max - t_min) * (s_max - s_min);
        if area < best_area - EPS {
            best_area = area;
            let tc = (t_min + t_max) * 0.5;
            let sc = (s_min + s_max) * 0.5;
            cx = tc * dx + sc * nx;
            cy = tc * dy + sc * ny;
            ax = dx;
            ay = dy;
            a_len = t_max - t_min;
            b_len = s_max - s_min;
        }
    }
    if best_area == f32::MAX {
        return None;
    }

    // Which fitted axis is the long one? The long axis is the reading axis
    // when it lies closer to vertical than to horizontal (and the box is
    // elongated enough); otherwise the axis closer to horizontal is the
    // reading axis — for a near-square box that is not necessarily the
    // longer one. The chosen axes point right/down so the frame is never
    // mirrored.
    let (mut lx, mut ly, long_len, short_len) = if a_len >= b_len {
        (ax, ay, a_len, b_len)
    } else {
        (-ay, ax, b_len, a_len)
    };
    let vertical = long_len >= short_len * VERTICAL_MIN_ASPECT && ly.abs() > lx.abs();
    let ((xa_x, xa_y), _ya, w, hgt) = if vertical {
        if ly < 0.0 {
            lx = -lx;
            ly = -ly;
        }
        // reading = local y; local x = reading rotated 90° CCW = (ry, -rx).
        ((ly, -lx), (lx, ly), short_len, long_len)
    } else {
        // reading = the fitted axis closer to horizontal, which for a
        // near-square box is not necessarily the longer one: a 20x22 box
        // must not use its slightly-longer vertical axis as the reading
        // axis and come out turned 90°.
        let use_d = ax.abs() >= ay.abs();
        lx = if use_d { ax } else { -ay };
        ly = if use_d { ay } else { ax };
        if lx < 0.0 {
            lx = -lx;
            ly = -ly;
        }
        // reading = local x; local y = reading rotated 90° CW = (-ry, rx).
        (
            (lx, ly),
            (-ly, lx),
            if use_d { a_len } else { b_len },
            if use_d { b_len } else { a_len },
        )
    };
    let angle = xa_y.atan2(xa_x);
    Some(RotatedBox::new(cx, cy, w, hgt, angle, 0.0))
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

/// Mobile `OcrEngine.GAP_CHAR`: the placeholder a dropped character becomes
/// (`BlankGaps`, #44). Tapping it opens the alternatives panel — the
/// dictionary lookup deliberately returns null for a placeholder — and the
/// panel offers the line's own per-timestep evidence for what went there.
pub const GAP_CHAR: char = '\u{25CC}';

#[derive(Debug, Clone)]
pub struct LineResult {
    pub text: String,
    pub char_boxes: Vec<BoundingBox>,
    pub alternatives: Vec<Vec<(char, f32)>>,
    /// Top-K alternatives for EVERY CTC timestep, blanks included, descending
    /// by score — mobile `LineResult.rawAlternatives`, the cache a re-decode
    /// walks without re-running the model (see
    /// [`DetectedAnnotation::re_decode_line`]).
    #[allow(dead_code)] // populated at emit; the mobile consumer (gap fallback) is not ported yet
    pub raw_alternatives: Vec<Vec<(char, f32)>>,
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
}

#[derive(Debug, Clone)]
pub struct NeighborChar {
    pub text: String,
    pub is_selected: bool,
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
    /// Where the entry came from (mobile `OovSuggestions.Source`): the
    /// panel tints non-head entries so the source is visible at a glance.
    pub source: crate::util::oov_suggestions::Source,
}

#[derive(Debug, Clone)]
pub struct AlternativesUiState {
    pub candidates: Vec<AlternativeChar>,
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

/// A Jitendex/Yomitan example box: either a JMdict-shaped
/// `{japanese, english}` pair, a set of `example-sentence-a/-b` parts
/// (Japanese then English), or generic structured content.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExampleNode {
    pub japanese: Option<String>,
    pub english: Option<String>,
    pub content: Vec<DefinitionNode>,
    pub parts: Vec<Vec<DefinitionNode>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DefinitionNode {
    Text(String),
    Ruby { term: String, reading: String },
    Tag { text: String },
    /// The closing `attribution` source line (`JMdict | Tatoeba`).
    Citation(String),
    Example(ExampleNode),
    ListBlock { items: Vec<Vec<DefinitionNode>>, list_type: Option<String> },
    Table { rows: Vec<Vec<Vec<DefinitionNode>>> },
    /// A boxed extra-info block (`xref`, `antonym`, `related`, `sense-note`,
    /// `info-gloss`, `lang-source`): `is_inline = false` gives it its own line.
    Group { nodes: Vec<DefinitionNode>, is_inline: bool },
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
    /// JMdict "Forms"/"Other forms" groups render unnumbered.
    pub is_forms: bool,
    /// Jitendex group metadata (POS/field info) rendered before the senses.
    pub header: Vec<DefinitionNode>,
    /// Forms/attribution blocks rendered after the senses, unnumbered.
    pub trailing: Vec<DefinitionNode>,
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
    /// Downstep positions from a pitch dictionary, empty when none.
    pub pitch_positions: Vec<i32>,
    /// False when this reading repeats a glossary already rendered for an
    /// earlier reading: the headword still shows, the senses do not.
    pub render_senses: bool,
}

/// A deinflection (or redirect) hop shown above the senses: `食べた → 食べる`.
#[derive(Debug, Clone)]
pub struct DeinflectionChain {
    pub surface: String,
    /// Rule-type tags for the hop (`past`, `redirect`, …).
    pub steps: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FormattedEntry {
    pub term: String,
    pub reading_groups: Vec<FormattedReadingGroup>,
    pub deinflection: Option<DeinflectionChain>,
    pub dictionary_name: Option<String>,
}

/// One matched term with its database rows and the chain that reached it.
#[derive(Debug, Clone)]
pub struct TermMatch {
    pub term: String,
    pub entries: Vec<crate::data::models::DictionaryEntry>,
    pub chain: Option<DeinflectionChain>,
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
    /// Pan by the given delta.
    PanDelta { dx: f32, dy: f32 },
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
    /// Frame tick — drains bootstrap/OCR channels so results stream smoothly.
    Tick,
    /// Live detection tuning: nudge DET_THRESH / DET_UNCLIP from the viewer.
    TuneDet { param: DetParam, delta: f32 },
    /// Restore both detection tunables to their startup values.
    TuneReset,
    /// Show/hide the detection tuning HUD (F1). Hidden by default.
    ToggleDetHud,
}

/// Which detection tunable the viewer's +/- keys adjust.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetParam {
    Threshold,
    Unclip,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Mobile RotatedGeometryTest.anAxisAlignedComponentFitsItsPixelExtents.
    #[test]
    fn axis_aligned_component_fits_its_pixel_extents() {
        // A 20x10 block of pixels x in [10,30), y in [30,40): the four outer
        // pixel corners, as the detector's boundary walk produces them.
        let pts = [(10.0f32, 30.0f32), (30.0, 30.0), (30.0, 40.0), (10.0, 40.0)];
        let quad = fit_quad(&pts).expect("fit");
        let (x, y, w, h) = quad.aabb();
        assert!(close(x, 10.0, 0.01) && close(y, 30.0, 0.01));
        assert!(close(w, 20.0, 0.01) && close(h, 10.0, 0.01));
        assert!(close(quad.w, 20.0, 0.01));
        assert!(close(quad.h, 10.0, 0.01));
        assert!(close(quad.angle.to_degrees(), 0.0, 0.01));
        assert!(!quad.is_vertical());
        assert!(!quad.is_rotated());
    }

    /// Mobile aRotatedHorizontalRectFitsTheSameRectItWasMadeFrom.
    #[test]
    fn rotated_horizontal_rect_fits_the_rect_it_was_made_from() {
        // A 40x10 rect centred on (100,100), turned 30 degrees clockwise.
        let pts = [
            (85.17949192431122f32, 85.6698729810778f32),
            (119.82050807568876, 105.6698729810778),
            (114.82050807568876, 114.33012701892218),
            (80.17949192431122, 94.33012701892218),
        ];
        let quad = fit_quad(&pts).expect("fit");
        assert!(close(quad.cx, 100.0, 0.01) && close(quad.cy, 100.0, 0.01));
        assert!(close(quad.w, 40.0, 0.01));
        assert!(close(quad.h, 10.0, 0.01));
        assert!(close(quad.angle.to_degrees(), 30.0, 0.01));
        assert!(!quad.is_vertical());
        assert!(quad.is_rotated());
        // Corners: c0 = origin, then x axis, then y axis.
        let (ox, oy) = quad.origin();
        let (xa, ya) = (quad.x_axis(), quad.y_axis());
        let corners = [
            (ox, oy),
            (ox + quad.w * xa.0, oy + quad.w * xa.1),
            (
                ox + quad.w * xa.0 + quad.h * ya.0,
                oy + quad.w * xa.1 + quad.h * ya.1,
            ),
            (ox + quad.h * ya.0, oy + quad.h * ya.1),
        ];
        for (i, (cx, cy)) in corners.iter().enumerate() {
            assert!(close(*cx, pts[i].0, 0.01), "c{i}.x");
            assert!(close(*cy, pts[i].1, 0.01), "c{i}.y");
        }
    }

    /// Mobile aRotatedVerticalColumnReadsTopToBottomAndKeepsTheColumnsOwnRotation.
    #[test]
    fn rotated_vertical_column_reads_top_to_bottom() {
        // A 10x40 column centred on (50,60), turned 15 degrees clockwise.
        let pts = [
            (50.34675177060507f32, 39.38738824870603f32),
            (60.00601003349575, 41.97557869973124),
            (49.65324822939492, 80.61261175129397),
            (39.99398996650424, 78.02442130026876),
        ];
        let quad = fit_quad(&pts).expect("fit");
        assert!(quad.is_vertical());
        assert!(close(quad.w, 10.0, 0.01));
        assert!(close(quad.h, 40.0, 0.01));
        // +15 clockwise, the same sign View.rotation/canvas.rotate take.
        assert!(close(quad.angle.to_degrees(), 15.0, 0.01));
        assert!(close(quad.cx, 50.0, 0.01) && close(quad.cy, 60.0, 0.01));
    }

    /// Mobile nearSquareBoxesStayHorizontalLikeTheAxisAlignedRule.
    #[test]
    fn near_square_boxes_stay_horizontal() {
        let pts = [(0.0f32, 0.0f32), (20.0, 0.0), (20.0, 22.0), (0.0, 22.0)];
        let quad = fit_quad(&pts).expect("fit");
        assert!(!quad.is_vertical());
        assert!(!quad.is_rotated());
        assert!(close(quad.w, 20.0, 0.01) && close(quad.h, 22.0, 0.01));
    }

    /// Mobile interiorAndDuplicatePointsDoNotDisturbTheFit.
    #[test]
    fn interior_and_duplicate_points_do_not_disturb_the_fit() {
        let pts = [
            (0.0f32, 0.0f32),
            (10.0, 0.0),
            (10.0, 20.0),
            (0.0, 20.0),
            (5.0, 10.0),
            (10.0, 0.0),
            (5.0, 0.0),
        ];
        let quad = fit_quad(&pts).expect("fit");
        let (x, y, w, h) = quad.aabb();
        assert!(close(x, 0.0, 0.01) && close(y, 0.0, 0.01));
        assert!(close(w, 10.0, 0.01) && close(h, 20.0, 0.01));
    }

    /// Mobile aDegeneratePointSetHasNoFit.
    #[test]
    fn degenerate_point_sets_have_no_fit() {
        assert!(fit_quad(&[(1.0, 1.0)]).is_none());
        assert!(fit_quad(&[(0.0, 0.0), (10.0, 10.0)]).is_none());
        assert!(fit_quad(&[(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)]).is_none());
    }

    /// Mobile unclipExpandsAlongTheBoxesOwnAxesAndKeepsTheCentre.
    #[test]
    fn unclip_expands_along_the_boxes_own_axes() {
        // 40x10 at the origin: area 400, perimeter 100, expand = 6 per side.
        let quad = RotatedBox::new(20.0, 5.0, 40.0, 10.0, 0.0, 1.0);
        let out = quad.unclip(1.5);
        assert!(close(out.w, 52.0, 0.01));
        assert!(close(out.h, 22.0, 0.01));
        assert!(close(out.cx, quad.cx, 0.001) && close(out.cy, quad.cy, 0.001));
        let (x, y, w, h) = out.aabb();
        assert!(close(x, -6.0, 0.01) && close(y, -6.0, 0.01));
        assert!(close(w, 52.0, 0.01) && close(h, 22.0, 0.01));
    }

    /// The fit's half-pixel boundary quantization must not make visually
    /// level short lines count as rotated (they would be un-rotated for
    /// recognition and back); mobile's 1° floor still governs long frames.
    #[test]
    fn short_noisy_fits_snap_to_straight() {
        // 70px UI text fitted 1.2°: ~1.5px drift, straight in practice.
        let short = RotatedBox::new(0.0, 0.0, 70.0, 6.0, 1.2f32.to_radians(), 1.0);
        assert!(!short.is_rotated());
        // 2° on the same frame is a real tilt.
        let tilted = RotatedBox::new(0.0, 0.0, 70.0, 6.0, 2.0f32.to_radians(), 1.0);
        assert!(tilted.is_rotated());
        // Long frames keep mobile's 1° tolerance.
        let long = RotatedBox::new(0.0, 0.0, 500.0, 30.0, 0.9f32.to_radians(), 1.0);
        assert!(!long.is_rotated());
        let long_tilted = RotatedBox::new(0.0, 0.0, 500.0, 30.0, 3.0f32.to_radians(), 1.0);
        assert!(long_tilted.is_rotated());
    }

    /// Mobile anAxisAlignedFrameMapsLocalBoxesOneToOne.
    #[test]
    fn axis_aligned_frame_maps_local_boxes_one_to_one() {
        let quad = RotatedBox::new(20.0, 35.0, 20.0, 10.0, 0.0, 1.0);
        let mapped = quad.map_local_rect(2.0, 4.0, 8.0, 8.0);
        assert_eq!((mapped.x, mapped.y, mapped.w, mapped.h), (12, 34, 8, 8));
    }

    /// Mobile aRotatedFrameMapsLocalCornersOntoTheFittedCorners.
    #[test]
    fn rotated_frame_maps_local_corners_onto_the_fitted_corners() {
        let pts = [
            (85.17949192431122f32, 85.6698729810778f32),
            (119.82050807568876, 105.6698729810778),
            (114.82050807568876, 114.33012701892218),
            (80.17949192431122, 94.33012701892218),
        ];
        let quad = fit_quad(&pts).expect("fit");
        // The whole local crop is exactly the quad's AABB.
        let whole = quad.map_local_rect(0.0, 0.0, 40.0, 10.0);
        let (x, y, w, h) = quad.aabb();
        assert_eq!(
            (whole.x, whole.y, whole.w, whole.h),
            (x.round() as i32, y.round() as i32, w.round() as i32, h.round() as i32)
        );
        // A local sub-box maps to a rotated rectangle.
        let mapped = quad.map_local_rect(0.0, 0.0, 10.0, 5.0);
        let expected_cx = 85.17949192431122f32 + 5.0 * 0.8660254 + 2.5 * -0.5;
        let expected_cy = 85.6698729810778f32 + 5.0 * 0.5 + 2.5 * 0.8660254;
        assert!(close(
            (mapped.x + mapped.x + mapped.w) as f32 / 2.0,
            expected_cx,
            0.51
        ));
        assert!(close(
            (mapped.y + mapped.y + mapped.h) as f32 / 2.0,
            expected_cy,
            0.51
        ));
        assert!(mapped.w > 10 && mapped.h > 5);
    }

    /// Mobile aFitWithinTheToleranceIsAxisAlignedSoItCanTakeTheRectPath.
    #[test]
    fn a_fit_within_one_degree_is_axis_aligned() {
        let level = fit_quad(&[(90.0, 95.0), (110.0, 95.0), (110.0, 105.0), (90.0, 105.0)])
            .expect("fit");
        assert!(!level.is_rotated());
        let half = fit_quad(&[
            (90.04401344685016, 94.9129250296954),
            (110.04325190813358, 95.08745573966289),
            (109.95598655314984, 105.0870749703046),
            (89.95674809186642, 104.91254426033711),
        ])
        .expect("fit");
        assert!(close(half.angle.to_degrees(), 0.5, 0.01));
        assert!(!half.is_rotated());
        let five = fit_quad(&[
            (90.47383173282083, 94.14746908206469),
            (110.39772569465573, 95.89058393701785),
            (109.52616826717914, 105.85253091793531),
            (89.60227430534424, 104.10941606298215),
        ])
        .expect("fit");
        assert!(close(five.angle.to_degrees(), 5.0, 0.01));
        assert!(five.is_rotated());
    }

    /// Mobile aVerticalRectFitsAsAVerticalQuadWhoseLocalYIsTheReadingAxis.
    #[test]
    fn a_vertical_rect_fits_as_a_vertical_quad() {
        let pts = [(10.0f32, 30.0f32), (20.0, 30.0), (20.0, 70.0), (10.0, 70.0)];
        let quad = fit_quad(&pts).expect("fit");
        assert!(quad.is_vertical());
        assert!(close(quad.w, 10.0, 0.001));
        assert!(close(quad.h, 40.0, 0.001));
        assert!(close(quad.angle.to_degrees(), 0.0, 0.001));
        // Local x runs right, local y runs down.
        let (xa, ya) = (quad.x_axis(), quad.y_axis());
        assert!(close(xa.0, 1.0, 0.001) && close(xa.1, 0.0, 0.001));
        assert!(close(ya.0, 0.0, 0.001) && close(ya.1, 1.0, 0.001));
    }

    /// Mobile RotatedGeometryTest blob-filter cases.
    #[test]
    fn enclosing_blob_filter_matches_android() {
        let rect = |x: f32, y: f32, w: f32, h: f32| RotatedBox::new(x + w / 2.0, y + h / 2.0, w, h, 0.0, 1.0);
        let blob = rect(0.0, 0.0, 300.0, 300.0);
        let line_a = rect(10.0, 10.0, 200.0, 25.0);
        let line_b = rect(10.0, 100.0, 200.0, 25.0);
        assert_eq!(
            filter_enclosing_blobs(&[blob, line_a, line_b]),
            vec![line_a, line_b]
        );
        // Only one enclosed line: kept.
        assert_eq!(
            filter_enclosing_blobs(&[blob, line_a]),
            vec![blob, line_a]
        );
        // Enclosed squares are not Line-shaped: kept.
        let mark_a = rect(10.0, 10.0, 30.0, 30.0);
        let mark_b = rect(100.0, 100.0, 30.0, 30.0);
        assert_eq!(
            filter_enclosing_blobs(&[blob, mark_a, mark_b]),
            vec![blob, mark_a, mark_b]
        );
        // Comparable-size inner frame: not "substantially smaller".
        let outer = rect(0.0, 0.0, 300.0, 300.0);
        let inner_a = rect(10.0, 10.0, 250.0, 250.0);
        let inner_b = rect(20.0, 20.0, 220.0, 20.0);
        assert_eq!(
            filter_enclosing_blobs(&[outer, inner_a, inner_b]),
            vec![outer, inner_a, inner_b]
        );
        // Disjoint lines never dropped.
        let lines = vec![
            rect(0.0, 0.0, 200.0, 30.0),
            rect(0.0, 50.0, 200.0, 30.0),
            rect(0.0, 100.0, 200.0, 30.0),
        ];
        assert_eq!(filter_enclosing_blobs(&lines), lines);
        // A dropped blob leaves the enclosed lines in order.
        let far = rect(500.0, 500.0, 200.0, 30.0);
        assert_eq!(
            filter_enclosing_blobs(&[line_a, blob, line_b, far]),
            vec![line_a, line_b, far]
        );
    }

    /// The frame's AABB stays the rounded corner envelope (mobile toRect).
    #[test]
    fn aabb_rounds_the_frame_corners() {
        let quad = RotatedBox::new(100.0, 100.0, 40.0, 10.0, 30f32.to_radians(), 1.0);
        let (x, y, w, h) = quad.aabb();
        let expected_w = 40.0 * 30f32.to_radians().cos() + 10.0 * 30f32.to_radians().sin();
        let expected_h = 40.0 * 30f32.to_radians().sin() + 10.0 * 30f32.to_radians().cos();
        assert!(close(w, expected_w, 0.01) && close(h, expected_h, 0.01));
        assert!(close(x, 100.0 - expected_w / 2.0, 0.01));
        assert!(close(y, 100.0 - expected_h / 2.0, 0.01));
    }
}
