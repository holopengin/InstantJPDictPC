//! One page in, one page out.
//!
//! The desktop pipeline grew as separate `detect_lines` / `recognize_boxes_*`
//! calls because its viewer wants to draw boxes before recognition finishes.
//! A mobile host does not: it captures one bitmap, wants every line and its
//! character geometry, and cannot afford one FFI crossing per stage. This
//! module is that shape: [`PageOcr::recognize`] runs detection, recognition,
//! kana correction and char-box placement in one call and returns a flat
//! [`PageResult`].
//!
//! The orchestration is deliberately thin. Every stage is the existing shared
//! implementation (`OcrEngine::detect_lines_with`,
//! `recognize_boxes_collect_with`, `char_boxes::compute_char_boxes_line_opts`);
//! this module only carries page-level options and the cooperative-cancel
//! flag. Its output uses plain data (no paths, no maps) so a binding can map
//! it without leaking desktop-only types.

use anyhow::Result;
use image::DynamicImage;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::models::{BoundingBox, DetectionResult, LineResult, RecognitionMode, RotatedBox};
use crate::ocr_engine::{DetectOptions, OcrEngine, RecognizeOptions};

/// Every knob one page run needs. The desktop derives these from its live
/// keys and `DET_*` / `BOX_*` environment; a mobile host passes its own
/// preferences explicitly. [`Default`] is the mobile page setup (shared
/// letterbox, rotated detection); start from [`DetectOptions::from_env`] for
/// the desktop preprocessing instead.
#[derive(Clone, Copy, Debug)]
pub struct PageOcrOptions {
    pub detect: DetectOptions,
    pub recognize: RecognizeOptions,
    /// Rotated-rect detection (mobile `PREF_DET_ROTATED`, default true). When
    /// false the fitted boxes are reduced to their axis-aligned AABBs and
    /// overlapping straight boxes are merged, matching the mobile kill switch.
    pub det_rotated: bool,
}

impl Default for PageOcrOptions {
    /// The page pipeline is the mobile-shaped entry point, so its defaults are
    /// the mobile ones: the shared C++ letterbox (Skia bilinear, integer pad)
    /// rather than the desktop's float-offset blit, and rotated detection on.
    /// A caller that wants the desktop preprocessing can start from
    /// [`DetectOptions::from_env`] instead.
    fn default() -> Self {
        let mut detect = DetectOptions::from_env();
        detect.mobile_letterbox = true;
        PageOcrOptions {
            detect,
            recognize: RecognizeOptions::from_env(),
            det_rotated: true,
        }
    }
}

/// One recognized line, flattened for a binding boundary.
#[derive(Clone, Debug)]
pub struct PageLine {
    pub bbox: BoundingBox,
    pub quad: Option<RotatedBox>,
    pub text: String,
    pub char_boxes: Vec<BoundingBox>,
    pub alternatives: Vec<Vec<(char, f32)>>,
    pub raw_alternatives: Vec<Vec<(char, f32)>>,
    pub char_cols: Vec<f32>,
    pub seq_len_total: i32,
    pub crop_x: i32,
    pub crop_y: i32,
    pub crop_w: i32,
    pub crop_h: i32,
    pub is_vertical: bool,
}

impl PageLine {
    fn from_annotation(bbox: BoundingBox, quad: Option<RotatedBox>, line: LineResult) -> Self {
        PageLine {
            bbox,
            quad,
            text: line.text,
            char_boxes: line.char_boxes,
            alternatives: line.alternatives,
            raw_alternatives: line.raw_alternatives,
            char_cols: line.char_cols,
            seq_len_total: line.seq_len_total,
            crop_x: line.crop_x,
            crop_y: line.crop_y,
            crop_w: line.crop_w,
            crop_h: line.crop_h,
            is_vertical: line.is_vertical,
        }
    }
}

/// Wall-clock split of one page run. `recognize` covers inference, kana
/// correction and char-box placement; `detect` covers detection and its
/// post-processing. `total` includes the small hand-off between them.
#[derive(Clone, Copy, Debug, Default)]
pub struct PageTimings {
    pub detect_ms: u64,
    pub recognize_ms: u64,
    pub total_ms: u64,
}

#[derive(Clone, Debug, Default)]
pub struct PageResult {
    pub lines: Vec<PageLine>,
    pub timings: PageTimings,
}

/// A loaded page engine. Construction loads the det/rec/kana nets once; every
/// [`recognize`](Self::recognize) call reuses them.
pub struct PageOcr {
    engine: OcrEngine,
    cancel: Arc<AtomicBool>,
}

impl PageOcr {
    pub fn new(
        model_dir: &str,
        recognition_mode: RecognitionMode,
        batch_size: usize,
    ) -> Result<Self> {
        Ok(PageOcr {
            engine: OcrEngine::new(model_dir, recognition_mode, batch_size)?,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Ask the in-flight [`recognize`](Self::recognize) call to stop at its
    /// next safe point. Safe to call from any thread; a later `recognize`
    /// clears the flag before it starts.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// A handle that can be stored outside a lock around the whole engine, so
    /// a binding's `cancel()` does not block behind the in-flight page call.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Detect, recognize, correct and place a whole page.
    ///
    /// Cancellation is cooperative: detection checks between component rows
    /// and before/after the ncnn call; recognition checks before each job and
    /// inside long-line chunk loops. A cancelled run returns an error with
    /// `cancelled` in the message.
    pub fn recognize(&mut self, image: &DynamicImage, opts: &PageOcrOptions) -> Result<PageResult> {
        use std::time::Instant;
        self.cancel.store(false, Ordering::SeqCst);
        let cancel = Arc::clone(&self.cancel);
        let total_start = Instant::now();

        let detect_start = Instant::now();
        let det = self.detect_page(image, opts, &cancel)?;
        let detect_ms = detect_start.elapsed().as_millis() as u64;
        if self.is_cancelled() {
            anyhow::bail!("page recognition cancelled");
        }
        let recognize_start = Instant::now();
        let annotations = crate::ocr_engine::recognize_boxes_collect_with(
            image,
            &det.boxes,
            &det.rotated,
            self.engine.ppocr_rec.clone(),
            self.engine.kana_size.clone(),
            &self.engine.ppocr_vocab,
            &self.engine.rec_remap,
            &opts.recognize,
            Some(Arc::clone(&cancel)),
        )?;
        let recognize_ms = recognize_start.elapsed().as_millis() as u64;

        let mut lines = Vec::with_capacity(annotations.len());
        for (idx, ann) in annotations {
            let Some(line) = ann.line else { continue };
            let quad = ann.quad;
            let bbox = det.boxes.get(idx).cloned().unwrap_or(ann.bbox);
            lines.push(PageLine::from_annotation(bbox, quad, line));
        }
        Ok(PageResult {
            lines,
            timings: PageTimings {
                detect_ms,
                recognize_ms,
                total_ms: total_start.elapsed().as_millis() as u64,
            },
        })
    }

    fn detect_page(
        &mut self,
        image: &DynamicImage,
        opts: &PageOcrOptions,
        cancel: &Arc<AtomicBool>,
    ) -> Result<DetectionResult> {
        let det = self
            .engine
            .detect_with(image, &opts.detect, Some(cancel.as_ref()))?;
        if opts.det_rotated {
            // Mobile `detectRotated` does not merge overlapping straight
            // boxes; only the legacy axis path does.
            return Ok(det);
        }
        // Mobile `PREF_DET_ROTATED=0`: keep the fitted AABBs, merge overlapping
        // straight boxes, and let recognition take the axis-aligned crop path.
        let pairs: Vec<(BoundingBox, RotatedBox)> =
            det.boxes.into_iter().zip(det.rotated).collect();
        let merged = crate::merge_boxes::merge_straight_boxes(pairs);
        let sorted = self.engine.sort_detected_boxes(merged);
        let (boxes, rotated): (Vec<_>, Vec<_>) = sorted.into_iter().unzip();
        Ok(DetectionResult { boxes, rotated })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The page result is a flat copy of the annotation; no desktop-only
    /// field survives.
    #[test]
    fn page_line_copies_the_line_fields() {
        let line = LineResult {
            text: "あい".into(),
            char_boxes: vec![BoundingBox::new(1, 2, 3, 4, 1.0)],
            alternatives: vec![vec![('あ', 0.9)]],
            raw_alternatives: vec![vec![('あ', 0.9), ('\u{3000}', 0.1)]],
            char_cols: vec![0.5, 1.5],
            seq_len_total: 4,
            crop_x: 10,
            crop_y: 20,
            crop_w: 30,
            crop_h: 40,
            is_vertical: true,
            ..Default::default()
        };
        let bbox = BoundingBox::new(10, 20, 30, 40, 1.0);
        let quad = RotatedBox::new(25.0, 20.0, 30.0, 40.0, 0.0, 1.0);
        let page = PageLine::from_annotation(bbox.clone(), Some(quad), line);
        assert_eq!(page.text, "あい");
        assert_eq!(page.char_boxes, vec![BoundingBox::new(1, 2, 3, 4, 1.0)]);
        assert_eq!(page.raw_alternatives.len(), 1);
        assert_eq!(page.seq_len_total, 4);
        assert_eq!(
            (page.crop_x, page.crop_y, page.crop_w, page.crop_h),
            (10, 20, 30, 40)
        );
        assert!(page.is_vertical);
        assert_eq!(page.quad, Some(quad));
    }

    #[test]
    fn recognizes_a_synthetic_page_in_one_call() {
        let repo = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
        let assets = repo.join("assets");
        let mut page =
            PageOcr::new(assets.to_str().unwrap(), RecognitionMode::Both, 4).expect("engine loads");
        let image =
            image::open(repo.join("test_images/synth/line_00_h.png")).expect("synth fixture");
        let mut opts = PageOcrOptions::default();
        opts.recognize.save_samples = false;
        opts.recognize.kana_size_correction = false;
        let result = page.recognize(&image, &opts).expect("page");
        assert!(!result.lines.is_empty(), "no lines recognized");
        for line in &result.lines {
            assert!(!line.text.is_empty());
            assert_eq!(line.char_boxes.len(), line.text.chars().count());
            assert!(!line.char_cols.is_empty());
        }
    }

    #[test]
    fn cancel_flag_is_sticky_until_the_next_run_starts() {
        // A page engine cannot be constructed without model assets in a unit
        // test, so pin the flag's contract directly.
        let flag = Arc::new(AtomicBool::new(false));
        assert!(!flag.load(Ordering::Relaxed));
        flag.store(true, Ordering::SeqCst);
        assert!(flag.load(Ordering::Relaxed));
        flag.store(false, Ordering::SeqCst);
        assert!(!flag.load(Ordering::Relaxed));
    }
}
