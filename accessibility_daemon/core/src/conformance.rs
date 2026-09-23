//! Shared cross-platform conformance corpus runner (dataset half of
//! `pipeline-sharing/01`).
//!
//! Cases live in `tests/conformance/cases/*.json` (format: `FORMAT.md`,
//! tolerances: `TOLERANCES.md`). Each `#[test]` below runs every case of one
//! `kind`; `every_case_has_a_runner` fails on orphan or unknown-kind files so
//! a case can never silently stop running.
//!
//! `CONFORMANCE_DUMP=1 cargo test conformance` prints each case's actual
//! pipeline output for seeding new expectations — paste them into the JSON
//! only after checking they are the *correct* semantics.

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::Value;

use crate::furigana::{is_ruby_horizontal, is_ruby_vertical};
use crate::kana_size::{self};
use crate::util::deinflector::Deinflector;
use crate::ruby_style::ruby_style;
use crate::data::models::DictionaryEntry;
use crate::models::{
    BoundingBox, FormattedEntry, LineResult, RotatedBox, TermMatch, GAP_CHAR,
};
use crate::ocr_engine::{
    filter_fitted_quads, fit_components, merge_straight_boxes,
    recognize_boxes_collect, rect_of, OcrEngine,
};
use crate::overlay_state::OcrOverlayState;
use crate::util::char_lm::CharLm;
use crate::util::gap_candidates::{self, MAX as GAP_MAX};
use crate::util::japanese;
use crate::models::RecognitionMode;

fn corpus_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
        .join("tests")
        .join("conformance")
}

fn cases_dir() -> PathBuf {
    corpus_dir().join("cases")
}

fn dump() -> bool {
    std::env::var("CONFORMANCE_DUMP").as_deref() == Ok("1")
}

/// Every case file, sorted by file name.
fn all_cases() -> Vec<(String, Value)> {
    let mut out = Vec::new();
    let mut names: Vec<_> = std::fs::read_dir(cases_dir())
        .expect("tests/conformance/cases exists")
        .map(|e| e.expect("dir entry").file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".json"))
        .collect();
    names.sort();
    for n in names {
        let text = std::fs::read_to_string(cases_dir().join(&n)).expect("case reads");
        let v: Value = serde_json::from_str(&text).expect("case parses");
        out.push((n, v));
    }
    out
}

fn kind_cases(kind: &str) -> Vec<(String, Value)> {
    let all = all_cases();
    let mine: Vec<_> = all
        .into_iter()
        .filter(|(_, v)| v["kind"].as_str() == Some(kind))
        .collect();
    assert!(!mine.is_empty(), "no cases of kind {kind}");
    mine
}

fn case_id(name: &str, v: &Value) -> String {
    let id = v["id"].as_str().expect("case has id");
    assert_eq!(
        format!("{id}.json"),
        *name,
        "file name must match the case id"
    );
    id.to_string()
}

fn box_tol(v: &Value) -> i32 {
    v["tolerances"]["box_px"].as_i64().unwrap_or(2) as i32
}

/// Float-tight box tolerance: `box_px` may be fractional (the
/// `char_placement` kind pins 1.5), which the integer helper would drop to
/// the default and silently widen.
fn box_tol_f(v: &Value) -> f64 {
    v["tolerances"]["box_px"].as_f64().unwrap_or(2.0)
}

fn angle_tol_deg(v: &Value) -> f32 {
    v["tolerances"]["angle_deg"].as_f64().unwrap_or(1.0) as f32
}

fn assert_box_close(id: &str, what: &str, got: &BoundingBox, expect: &[i64], tol: i32) {
    assert_eq!(expect.len(), 4, "{id}: expected box must be [x, y, w, h]");
    let (ex, ey, ew, eh) = (expect[0] as i32, expect[1] as i32, expect[2] as i32, expect[3] as i32);
    for (axis, g, e) in [("x", got.x, ex), ("y", got.y, ey), ("w", got.w, ew), ("h", got.h, eh)] {
        assert!(
            (g - e).abs() <= tol,
            "{id} {what}: box {axis} drifted: got {g}, expected {e} (tol ±{tol}px)"
        );
    }
}

fn rect_of_json(a: &[Value]) -> BoundingBox {
    let v: Vec<i32> = a.iter().map(|x| x.as_i64().expect("rect int") as i32).collect();
    assert_eq!(v.len(), 4, "rects are [left, top, right, bottom]");
    BoundingBox::new(v[0], v[1], v[2] - v[0], v[3] - v[1], 1.0)
}

fn test_engine() -> OcrEngine {
    let dir = format!("{}/assets", concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    OcrEngine::new(&dir, RecognitionMode::Both, 4).expect("engine loads")
}

/// The real post-processing chain in pipeline order over a dark-pixel
/// stand-in prob map (dark pixel < 128 → 0.9, else 0.0).
fn run_detection(
    image: &str,
    det_thresh: f32,
    det_unclip: f32,
    furigana: bool,
) -> Vec<(BoundingBox, RotatedBox)> {
    let path = corpus_dir().join(image);
    let img = image::open(&path).expect("input image loads").to_luma8();
    let (w, h) = (img.width(), img.height());
    let prob: Vec<f32> = img.pixels().map(|p| if p[0] < 128 { 0.9 } else { 0.0 }).collect();
    let (pre, quads) = fit_components(
        &prob,
        w,
        h,
        det_thresh,
        det_unclip,
        f32::INFINITY,
        0.0,
        0.0,
        1.0,
        1.0,
    );
    let kept = filter_fitted_quads(&pre, &quads, w as i32, h as i32, furigana);
    let pairs: Vec<(BoundingBox, RotatedBox)> =
        kept.iter().map(|q| (rect_of(q), *q)).collect();
    let merged = merge_straight_boxes(pairs);
    test_engine().sort_detected_boxes(merged)
}

/// Pack a toy CharLm table exactly like `tools/pack_char_lm.py` writes it.
fn pack_lm(entries: &[(String, u16)], mass: u32) -> Vec<u8> {
    let mut records: Vec<([u16; 4], u16)> = entries
        .iter()
        .map(|(ngram, count)| {
            let mut units = [0u16; 4];
            for (i, ch) in ngram.chars().enumerate().take(4) {
                units[i] = ch as u16;
            }
            (units, *count)
        })
        .collect();
    records.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Vec::new();
    out.extend_from_slice(&0x314D_4C43u32.to_le_bytes());
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&mass.to_le_bytes());
    for (units, count) in records {
        for unit in units {
            out.extend_from_slice(&unit.to_le_bytes());
        }
        out.extend_from_slice(&count.to_le_bytes());
    }
    out
}

fn lm_of(case: &Value) -> Option<CharLm> {
    let lm = &case["lm"];
    if lm.is_null() {
        return None;
    }
    let entries: Vec<(String, u16)> = lm["entries"]
        .as_array()
        .expect("lm.entries")
        .iter()
        .map(|e| {
            (
                e[0].as_str().expect("ngram").to_string(),
                e[1].as_i64().expect("count") as u16,
            )
        })
        .collect();
    CharLm::from_bytes(pack_lm(&entries, lm["mass"].as_i64().expect("mass") as u32))
}

fn kana_line(text: &str) -> LineResult {
    // correct_lines works over the shared LineResult; only the text matters
    // here (no geometry: the scorer is injected).
    LineResult {
        text: text.to_string(),
        char_boxes: Vec::new(),
        alternatives: Vec::new(),
        raw_alternatives: Vec::new(),
        sample_txt: None,
        is_vertical: false,
        chunk_boxes: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// The runners: one #[test] per kind.
// ---------------------------------------------------------------------------

#[test]
fn every_case_has_a_runner() {
    let known = [
        "detection",
        "geometry",
        "reading_order",
        "furigana",
        "gap",
        "char_lm",
        "kana",
        "dictionary",
        "recognition",
        "deinflection",
        "ruby_style",
        "char_placement",
        "normalize",
    ];
    for (name, v) in all_cases() {
        let id = case_id(&name, &v);
        let kind = v["kind"].as_str().expect("case has kind");
        assert!(
            known.contains(&kind),
            "{id}: unknown kind {kind} — add a runner or fix the case"
        );
    }
}

#[test]
fn detection_cases() {
    for (name, v) in kind_cases("detection") {
        let id = case_id(&name, &v);
        let c = &v["case"];
        let boxes = run_detection(
            c["image"].as_str().expect("image"),
            c["det_thresh"].as_f64().expect("det_thresh") as f32,
            c["det_unclip"].as_f64().expect("det_unclip") as f32,
            c["furigana_filter"].as_bool().expect("furigana_filter"),
        );
        let expect = c["expect_lines"].as_array().expect("expect_lines");
        if dump() {
            for (i, (b, q)) in boxes.iter().enumerate() {
                println!(
                    "DUMP {id} line {i}: [{}, {}, {}, {}] vertical={}",
                    b.x,
                    b.y,
                    b.w,
                    b.h,
                    q.is_vertical()
                );
            }
        }
        assert_eq!(
            boxes.len(),
            expect.len(),
            "{id}: line count drifted: got {}, expected {}",
            boxes.len(),
            expect.len()
        );
        let tol = box_tol(&v);
        for (i, ((got, quad), exp)) in boxes.iter().zip(expect.iter()).enumerate() {
            let ebox: Vec<i64> = exp["box"]
                .as_array()
                .expect("box")
                .iter()
                .map(|x| x.as_i64().expect("int"))
                .collect();
            assert_box_close(&id, &format!("line {i}"), got, &ebox, tol);
            if let Some(vertical) = exp.get("vertical").and_then(|x| x.as_bool()) {
                assert_eq!(
                    quad.is_vertical(),
                    vertical,
                    "{id} line {i}: orientation flag drifted"
                );
            }
        }
    }
}

#[test]
fn geometry_cases() {
    for (name, v) in kind_cases("geometry") {
        let id = case_id(&name, &v);
        let tol = angle_tol_deg(&v);
        for f in v["case"]["frames"].as_array().expect("frames") {
            let angle = f["angle_deg"].as_f64().expect("angle_deg") as f32
                * std::f32::consts::PI
                / 180.0;
            let frame = RotatedBox::new(
                f["cx"].as_f64().unwrap_or(0.0) as f32,
                f["cy"].as_f64().unwrap_or(0.0) as f32,
                f["w"].as_f64().expect("w") as f32,
                f["h"].as_f64().expect("h") as f32,
                angle,
                1.0,
            );
            let exp = &f["expect"];
            assert_eq!(
                frame.is_vertical(),
                exp["vertical"].as_bool().expect("vertical"),
                "{id}: is_vertical drifted for {f:?}"
            );
            assert_eq!(
                frame.is_axis_aligned(),
                exp["axis_aligned"].as_bool().expect("axis_aligned"),
                "{id}: is_axis_aligned drifted for {f:?}"
            );
            if let Some(n) = exp["norm_angle_deg"].as_f64() {
                let got_deg = frame.angle * 180.0 / std::f32::consts::PI;
                assert!(
                    (got_deg - n as f32).abs() <= tol,
                    "{id}: angle normalization drifted: got {got_deg:.2}°, expected {n}°"
                );
            }
        }
    }
}

#[test]
fn reading_order_cases() {
    let engine = test_engine();
    for (name, v) in kind_cases("reading_order") {
        let id = case_id(&name, &v);
        let pairs: Vec<(BoundingBox, RotatedBox)> = v["case"]["boxes"]
            .as_array()
            .expect("boxes")
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let e: Vec<i64> = b["box"]
                    .as_array()
                    .expect("box")
                    .iter()
                    .map(|x| x.as_i64().expect("int"))
                    .collect();
                let wl = b["w_local"].as_f64().expect("w_local") as f32;
                let hl = b["h_local"].as_f64().expect("h_local") as f32;
                let bb = BoundingBox::new(e[0] as i32, e[1] as i32, e[2] as i32, e[3] as i32, 1.0);
                // Tag the index in the confidence so the sorted order is
                // directly readable.
                let bb = BoundingBox::new(bb.x, bb.y, bb.w, bb.h, i as f32);
                let q = RotatedBox::new(
                    bb.x as f32 + bb.w as f32 / 2.0,
                    bb.y as f32 + bb.h as f32 / 2.0,
                    wl,
                    hl,
                    0.0,
                    i as f32,
                );
                (bb, q)
            })
            .collect();
        let sorted = engine.sort_detected_boxes(pairs);
        let got: Vec<usize> = sorted.iter().map(|(_, q)| q.confidence as usize).collect();
        let expect: Vec<usize> = v["case"]["expect_order"]
            .as_array()
            .expect("expect_order")
            .iter()
            .map(|x| x.as_i64().expect("int") as usize)
            .collect();
        if dump() {
            println!("DUMP {id} order: {got:?}");
        }
        assert_eq!(got, expect, "{id}: reading order drifted");
    }
}

#[test]
fn furigana_cases() {
    for (name, v) in kind_cases("furigana") {
        let id = case_id(&name, &v);
        let c = &v["case"];
        let img = c["img"].as_array().expect("img");
        let (img_w, img_h) = (img[0].as_i64().expect("int") as i32, img[1].as_i64().expect("int") as i32);
        for p in c["pairs"].as_array().expect("pairs") {
            let get = |k: &str| rect_of_json(p[k].as_array().expect(k));
            let (s_raw, b_raw, s_un, b_un) =
                (get("small_raw"), get("big_raw"), get("small_un"), get("big_un"));
            let orient = p["orientation"].as_str().expect("orientation");
            let got = match orient {
                "horizontal" => is_ruby_horizontal(&s_raw, &b_raw, &s_un, &b_un, img_w, img_h),
                "vertical" => is_ruby_vertical(&s_raw, &b_raw, &s_un, &b_un, img_h),
                o => panic!("{id}: bad orientation {o}"),
            };
            assert_eq!(
                got,
                p["expect_ruby"].as_bool().expect("expect_ruby"),
                "{id} pair '{}': ruby verdict drifted",
                p["name"].as_str().unwrap_or("?")
            );
        }
    }
}

#[test]
fn gap_cases() {
    for (name, v) in kind_cases("gap") {
        let id = case_id(&name, &v);
        let c = &v["case"];
        let alts: Vec<Vec<(char, f32)>> = c["alternatives"]
            .as_array()
            .expect("alternatives")
            .iter()
            .map(|step| {
                step.as_array()
                    .expect("step")
                    .iter()
                    .map(|e| {
                        (
                            e[0].as_str().expect("ch").chars().next().expect("char"),
                            e[1].as_f64().expect("score") as f32,
                        )
                    })
                    .collect()
            })
            .collect();
        let text = c["text"].as_str().expect("text");
        let gap_index = c["gap_index"].as_i64().expect("gap_index") as usize;
        assert_eq!(
            text.chars().nth(gap_index),
            Some(GAP_CHAR),
            "{id}: gap_index must point at the ◌ placeholder"
        );
        let context = gap_candidates::context_before(text, gap_index);
        if let Some(exp_ctx) = c.get("expect_context") {
            let exp: Vec<char> = exp_ctx
                .as_array()
                .expect("expect_context")
                .iter()
                .map(|s| s.as_str().expect("ch").chars().next().expect("char"))
                .collect();
            assert_eq!(context, exp, "{id}: context derivation drifted");
        }
        let lm = lm_of(c);
        let limit = c["limit"].as_i64().unwrap_or(GAP_MAX as i64) as usize;
        let got = gap_candidates::generate(&alts, limit, lm.as_ref(), &context);
        let expect: Vec<char> = c["expect"]
            .as_array()
            .expect("expect")
            .iter()
            .map(|s| s.as_str().expect("ch").chars().next().expect("char"))
            .collect();
        if dump() {
            let s: String = got.iter().collect();
            println!("DUMP {id} candidates: {s}");
        }
        assert_eq!(got, expect, "{id}: candidate list drifted");
        if let Some(first) = c.get("expect_fallback_first") {
            let fb = gap_candidates::fallback(limit, lm.as_ref(), &context);
            let f0 = first.as_str().expect("ch").chars().next().expect("char");
            assert_eq!(fb[0], f0, "{id}: fallback class order drifted: {fb:?}");
        }
    }
}

#[test]
fn char_lm_cases() {
    for (name, v) in kind_cases("char_lm") {
        let id = case_id(&name, &v);
        let c = &v["case"];
        let entries: Vec<(String, u16)> = c["entries"]
            .as_array()
            .expect("entries")
            .iter()
            .map(|e| {
                (
                    e[0].as_str().expect("ngram").to_string(),
                    e[1].as_i64().expect("count") as u16,
                )
            })
            .collect();
        let lm = CharLm::from_bytes(pack_lm(&entries, c["mass"].as_i64().expect("mass") as u32))
            .expect("toy table packs");
        for q in c["counts"].as_array().expect("counts") {
            let ngram: Vec<char> = q["ngram"]
                .as_array()
                .expect("ngram")
                .iter()
                .map(|s| s.as_str().expect("ch").chars().next().expect("char"))
                .collect();
            assert_eq!(
                lm.count(&ngram),
                q["expect"].as_i64().expect("expect") as u32,
                "{id}: count drifted for {ngram:?}"
            );
        }
        for q in c["ranks"].as_array().expect("ranks") {
            let ctx: Vec<char> = q["context"]
                .as_array()
                .expect("context")
                .iter()
                .map(|s| s.as_str().expect("ch").chars().next().expect("char"))
                .collect();
            let pool: Vec<char> = q["pool"]
                .as_array()
                .expect("pool")
                .iter()
                .map(|s| s.as_str().expect("ch").chars().next().expect("char"))
                .collect();
            let expect: Vec<char> = q["expect"]
                .as_array()
                .expect("expect")
                .iter()
                .map(|s| s.as_str().expect("ch").chars().next().expect("char"))
                .collect();
            assert_eq!(lm.rank(&ctx, &pool), expect, "{id}: rank drifted");
        }
    }
}

#[test]
fn kana_cases() {
    for (name, v) in kind_cases("kana") {
        let id = case_id(&name, &v);
        let c = &v["case"];
        if let Some(enc) = c.get("encoder").and_then(|e| e.as_array()) {
        for e in enc {
            let text = e["text"].as_str().expect("text");
            let index = e["index"].as_i64().expect("index") as usize;
            let got = kana_size::window(text, index);
            let head: Vec<i64> = e["expect_window_head"]
                .as_array()
                .expect("expect_window_head")
                .iter()
                .map(|x| x.as_i64().expect("int"))
                .collect();
            assert!(
                got.len() == kana_size::WINDOW_BYTES,
                "{id}: window is {} bytes, expected {}",
                got.len(),
                kana_size::WINDOW_BYTES
            );
            for (i, want) in head.iter().enumerate() {
                assert_eq!(
                    got[i], *want as i32,
                    "{id}: window byte {i} drifted for {text:?}@{index}"
                );
            }
        }
        }
        if let Some(p) = c.get("policy") {
        let lines: Vec<LineResult> = p["lines"]
            .as_array()
            .expect("lines")
            .iter()
            .map(|s| kana_line(s.as_str().expect("line")))
            .collect();
        let logits: Vec<f32> = p["logits"]
            .as_array()
            .expect("logits")
            .iter()
            .map(|x| x.as_f64().expect("logit") as f32)
            .collect();
        let epsilon = p["epsilon"].as_f64().expect("epsilon") as f32;
        let correction =
            kana_size::correct_lines(&lines, |_, _| Some(logits.clone()), epsilon);
        let got_text: Vec<String> = correction.lines.iter().map(|l| l.text.clone()).collect();
        let expect_text: Vec<String> = p["expect_text"]
            .as_array()
            .expect("expect_text")
            .iter()
            .map(|s| s.as_str().expect("line").to_string())
            .collect();
        if dump() {
            println!("DUMP {id} text: {got_text:?} flips: {:?}", correction.flips);
        }
        assert_eq!(got_text, expect_text, "{id}: corrected text drifted");
        let expect_flips = p["expect_flips"].as_array().expect("expect_flips");
        assert_eq!(
            correction.flips.len(),
            expect_flips.len(),
            "{id}: flip count drifted: {:?}",
            correction.flips
        );
        for (got, exp) in correction.flips.iter().zip(expect_flips.iter()) {
            assert_eq!(got.line, exp["line"].as_i64().expect("line") as usize);
            assert_eq!(got.index, exp["index"].as_i64().expect("index") as usize);
            assert_eq!(
                got.from,
                exp["from"].as_str().expect("from").chars().next().expect("char")
            );
            assert_eq!(
                got.to,
                exp["to"].as_str().expect("to").chars().next().expect("char")
            );
        }
        }
    }
}

/// The lookup-normalization pipeline in stage order: width conversion →
/// lookup-variant fold → combining-character normalization. Each entry is one
/// input string and the exact composed output, so the case pins iteration-mark
/// expansion (plain and voiced), the variant fold (Roman numerals, ℃, obsolete
/// kana, Chinese-only forms, the measured Unihan pairs), width/halfwidth
/// widening, and decomposed combining sequences in one place.
#[test]
fn normalize_cases() {
    for (name, v) in kind_cases("normalize") {
        let id = case_id(&name, &v);
        for c in v["case"]["cases"].as_array().expect("cases") {
            let input = c["input"].as_str().expect("input");
            let expect = c["expect"].as_str().expect("expect");
            let got = japanese::normalize(input);
            if dump() {
                println!("DUMP {id} {input:?} -> {got:?}");
            }
            assert_eq!(got, expect, "{id}: normalize drifted for {input:?}");
        }
    }
}

/// Full pixel→text run over a vendored photo/screenshot, in detection
/// (reading) order: `detect_lines` → `recognize_boxes_collect`. Returns one
/// entry per annotation the recognizer emitted, each with its detection box
/// index — boxes the crop stage skips (un-croppable quads) have no entry,
/// and the pinned `i` fails loudly if that set ever changes.
///
/// Deterministic: `recognize_boxes_collect` sorts by box index, normalizing
/// the worker completion order; per-line box/text/conf are byte-identical
/// across runs on the same host (see `TOLERANCES.md`).
fn run_recognition(
    engine: &mut OcrEngine,
    image: &str,
    det_thresh: f32,
    det_unclip: f32,
    furigana: bool,
) -> Vec<(usize, BoundingBox, Option<(bool, String, Vec<BoundingBox>)>)> {
    engine.det_thresh_override = Some(det_thresh);
    engine.det_unclip_override = Some(det_unclip);
    engine.det_furigana = furigana;
    let path = corpus_dir().join(image);
    let img = image::open(&path).expect("input image loads");
    let det = engine.detect_lines(&img).expect("detection runs");
    let vocab = engine.ppocr_vocab.clone();
    let remap = engine.rec_remap.clone();
    // Dataset sidecars go to /tmp (the pipeline's `save_line_sample` always
    // writes there), never next to the vendored fixture.
    let out = recognize_boxes_collect(
        &img,
        &det.boxes,
        &det.rotated,
        engine.ppocr_rec.clone(),
        engine.kana_size.clone(),
        &vocab,
        &remap,
        4,
        RecognitionMode::Both,
        &std::env::temp_dir(),
    )
    .expect("recognition runs");
    out.into_iter()
        .map(|(idx, ann)| {
            let line = ann
                .line
                .map(|l| (l.is_vertical, l.text, l.char_boxes));
            (idx, ann.bbox, line)
        })
        .collect()
}

#[test]
fn recognition_cases() {
    let mut engine = test_engine();
    for (name, v) in kind_cases("recognition") {
        let id = case_id(&name, &v);
        let c = &v["case"];
        let got = run_recognition(
            &mut engine,
            c["image"].as_str().expect("image"),
            c["det_thresh"].as_f64().expect("det_thresh") as f32,
            c["det_unclip"].as_f64().expect("det_unclip") as f32,
            c["furigana_filter"].as_bool().expect("furigana_filter"),
        );
        if dump() {
            for (idx, bbox, line) in &got {
                let (vertical, text, chars) = match line {
                    Some((vert, t, cb)) => (
                        *vert,
                        serde_json::Value::String(t.clone()),
                        cb.iter()
                            .map(|b| serde_json::json!([b.x, b.y, b.w, b.h]))
                            .collect::<Vec<_>>(),
                    ),
                    None => (false, serde_json::Value::Null, Vec::new()),
                };
                let j = serde_json::json!({
                    "i": idx,
                    "box": [bbox.x, bbox.y, bbox.w, bbox.h],
                    "vertical": vertical,
                    "text": text,
                    "char_boxes": chars,
                });
                println!("DUMP-JSON {id} {j}");
            }
            // Regenerate mode: print actuals for every case without failing,
            // so one run refreshes the whole kind. Plain `cargo test`
            // (below) enforces the pins.
            continue;
        }
        let expect = c["expect_lines"].as_array().expect("expect_lines");
        assert_eq!(
            got.len(),
            expect.len(),
            "{id}: recognized line count drifted: got {}, expected {}",
            got.len(),
            expect.len()
        );
        let tol = box_tol(&v);
        for ((idx, bbox, line), exp) in got.iter().zip(expect.iter()) {
            assert_eq!(
                *idx,
                exp["i"].as_i64().expect("i") as usize,
                "{id}: detection-index mapping drifted"
            );
            let ebox: Vec<i64> = exp["box"]
                .as_array()
                .expect("box")
                .iter()
                .map(|x| x.as_i64().expect("int"))
                .collect();
            assert_box_close(&id, &format!("line i={idx}"), bbox, &ebox, tol);
            match (line, exp.get("text")) {
                (Some((vert, text, chars)), Some(t)) if !t.is_null() => {
                    assert_eq!(
                        *vert,
                        exp["vertical"].as_bool().expect("vertical"),
                        "{id} line i={idx}: vertical flag drifted"
                    );
                    assert_eq!(
                        text,
                        t.as_str().expect("text"),
                        "{id} line i={idx}: recognized text drifted"
                    );
                    let expect_chars =
                        exp["char_boxes"].as_array().expect("char_boxes");
                    assert_eq!(
                        chars.len(),
                        expect_chars.len(),
                        "{id} line i={idx}: char box count drifted"
                    );
                    for (j, (cb, ec)) in
                        chars.iter().zip(expect_chars.iter()).enumerate()
                    {
                        let e: Vec<i64> = ec
                            .as_array()
                            .expect("char box")
                            .iter()
                            .map(|x| x.as_i64().expect("int"))
                            .collect();
                        assert_box_close(
                            &id,
                            &format!("line i={idx} char {j}"),
                            cb,
                            &e,
                            tol,
                        );
                    }
                }
                (None, t) if t.is_none_or(|x| x.is_null()) => {}
                (l, t) => panic!(
                    "{id} line i={idx}: recognized/unrecognized flipped: {l:?} vs {t:?}"
                ),
            }
        }
    }
}

fn deinflect_rules_path() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/..")).join("assets/deinflect.json")
}

/// Ticket 07: the shipped `deinflect.json` group keys ride each derivation
/// as human-readable reason labels (mobile `Deinflector` fills `reason` from
/// the map key at load; so does `Deinflector::from_json_file`). Each case
/// pins one surface form reaching one dictionary term with an exact ordered
/// reason list; the no-op case pins the identity candidate carrying no
/// reasons (no chain, no viewer row). Terms are unique per surface (the
/// engine dedupes by term, first derivation wins), so a find-by-term plus an
/// exact reason match is a complete pin — no ranking to freeze.
#[test]
fn deinflection_cases() {
    let rules = Deinflector::from_json_file(deinflect_rules_path())
        .expect("shipped deinflect.json loads");
    for (name, v) in kind_cases("deinflection") {
        let id = case_id(&name, &v);
        let c = &v["case"];
        let surface = c["surface"].as_str().expect("surface");
        let expect_term = c["expect_term"].as_str().expect("expect_term");
        let expect_reasons: Vec<&str> = c["expect_reasons"]
            .as_array()
            .expect("expect_reasons")
            .iter()
            .map(|s| s.as_str().expect("reason str"))
            .collect();
        let got = rules.deinflect(surface);
        if dump() {
            println!("DUMP {id} {surface}: {} candidates", got.len());
            for r in got.iter().take(25) {
                println!("DUMP {id} term={} reasons={:?}", r.term, r.reasons);
            }
        }
        match got.iter().find(|r| r.term == expect_term) {
            Some(hit) => assert_eq!(
                hit.reasons,
                expect_reasons,
                "{id}: reason labels drifted for {surface} → {expect_term}"
            ),
            None => panic!(
                "{id}: no derivation of {surface} reaches {expect_term} ({} candidates)",
                got.len()
            ),
        }
        // Optional negative pin: derivations that must NOT exist. Needed
        // where the requirement is a guard (e.g. single-character terms are
        // never deinflected) that a find-by-term assertion cannot see.
        if let Some(absent) = c["expect_absent"].as_array() {
            for term in absent {
                let term = term.as_str().expect("expect_absent entry is a string");
                assert!(
                    !got.iter().any(|r| r.term == term),
                    "{id}: {surface} must not derive {term} ({} candidates)",
                    got.len()
                );
            }
        }
    }
}

/// Ticket 06: body ruby vs term-display ruby, pinned at the unit-testable
/// style-resolution seam (`ruby_style(is_mini)`), not rendered
/// pixels — painting an iced `Text` needs the UI framework, and sizes plus
/// the gray ruby row stay pinned in the binary's `viewer.rs` unit tests.
/// `mode` selects the renderer input: `body` is `is_mini = true` (everything
/// `inline_line` builds), `term` is `is_mini = false` (headword/term rows).
/// `base` is a color label so a recolor fails with the values attached,
/// never as a silent float drift: `white` is `[1, 1, 1]`, `cyan` is
/// `[0, 1, 1]`.
fn ruby_base_label(c: &[f32; 3]) -> &'static str {
    if *c == [1.0, 1.0, 1.0] {
        "white"
    } else if *c == [0.0, 1.0, 1.0] {
        "cyan"
    } else {
        panic!("unmapped ruby base color {c:?} — extend the label map, never widen a tolerance");
    }
}

#[test]
fn ruby_style_cases() {
    for (name, v) in kind_cases("ruby_style") {
        let id = case_id(&name, &v);
        for m in v["case"]["modes"].as_array().expect("modes") {
            let (mode, is_mini) = match m["mode"].as_str().expect("mode") {
                "body" => ("body", true),
                "term" => ("term", false),
                o => panic!("{id}: bad mode {o} (body = is_mini, term = full-size display)"),
            };
            let style = ruby_style(is_mini);
            if dump() {
                println!(
                    "DUMP {id} {mode}: base={} bold={}",
                    ruby_base_label(&style.base),
                    style.bold
                );
            }
            assert_eq!(
                ruby_base_label(&style.base),
                m["base"].as_str().expect("base"),
                "{id} {mode}: base treatment drifted"
            );
            assert_eq!(
                style.bold,
                m["bold"].as_bool().expect("bold"),
                "{id} {mode}: weight drifted"
            );
        }
    }
}

/// Linear-interpolation percentile over an unsorted sample (numpy's default
/// method — the spec's `mean/p90` metric rows use it).
fn percentile_px(xs: &mut [f64], q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(f64::total_cmp);
    if xs.len() == 1 {
        return xs[0];
    }
    let pos = q / 100.0 * (xs.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    xs[lo] * (1.0 - frac) + xs[hi] * frac
}

/// Tier-1 placement parity (`docs/char-placement-conformance.md` in the
/// Android checkout): recorded decoder evidence + raw luminance bytes in,
/// one full-cross-axis box per character out, compared against the PYTHON
/// reference's `expect_boxes` — box count exact, each of l/t/r/b within the
/// kind's 1.5 px override — plus the tap contract `CharPlacementTest.
/// boxesContainTheirCharactersInkCentre` pins on Android (every gold ink
/// centre lies on its own box's reading axis, 0.5 px slack).
///
/// `gold` also drives the per-case Tier-2-style metrics, printed under
/// `CONFORMANCE_DUMP=1` as *reporting only*. The dump deliberately prints no
/// rects: expectations for this kind are the reference's output, so a dump
/// of the port's geometry would be circular (FORMAT.md, `char_placement`).
#[test]
fn char_placement_cases() {
    for (name, v) in kind_cases("char_placement") {
        let id = case_id(&name, &v);
        let c = &v["case"];
        let crop_w = c["crop_w"].as_i64().expect("crop_w") as u32;
        let crop_h = c["crop_h"].as_i64().expect("crop_h") as u32;
        let gray = std::fs::read(corpus_dir().join(c["gray"].as_str().expect("gray")))
            .expect("gray fixture reads");
        assert_eq!(
            gray.len(),
            crop_w as usize * crop_h as usize,
            "{id}: .gray must hold exactly crop_w * crop_h luminance bytes"
        );
        let lum: Vec<f32> = gray.iter().map(|&b| b as f32).collect();
        let vertical = match c["orientation"].as_str().expect("orientation") {
            "v" => true,
            "h" => false,
            o => panic!("{id}: bad orientation {o:?} (h | v)"),
        };
        let seq_len_total = c["seq_len_total"].as_i64().expect("seq_len_total") as usize;
        let text = c["text"].as_str().expect("text");
        let char_cols: Vec<f32> = c["char_cols"]
            .as_array()
            .expect("char_cols")
            .iter()
            .map(|x| x.as_f64().expect("float") as f32)
            .collect();
        let steps: Vec<Vec<(char, f64)>> = c["steps"]
            .as_array()
            .expect("steps")
            .iter()
            .map(|t| {
                t.as_array()
                    .expect("timestep")
                    .iter()
                    .map(|e| {
                        (
                            e[0].as_str().expect("ch").chars().next().expect("char"),
                            e[1].as_f64().expect("score"),
                        )
                    })
                    .collect()
            })
            .collect();

        // Case values, never platform defaults (same rule as detection's
        // det_thresh): the pin must not move when a default changes.
        let got = crate::char_placement::place(
            text,
            &char_cols,
            seq_len_total,
            crop_w,
            crop_h,
            vertical,
            Some((&lum, crop_w, crop_h)),
            Some(&steps),
        );
        let expect = c["expect_boxes"].as_array().expect("expect_boxes");
        assert_eq!(
            got.len(),
            expect.len(),
            "{id}: box count drifted: got {}, expected {}",
            got.len(),
            expect.len()
        );

        let tol = box_tol_f(&v);
        let mut errs: Vec<f64> = Vec::new();
        let mut worst_delta = 0.0f64;
        for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            let eb: Vec<f64> = e
                .as_array()
                .expect("box [l, t, r, b]")
                .iter()
                .map(|x| x.as_f64().expect("float"))
                .collect();
            assert_eq!(eb.len(), 4, "{id} char {i}: box must be [l, t, r, b]");
            for (axis, gv, ev) in [
                ("left", g[0], eb[0]),
                ("top", g[1], eb[1]),
                ("right", g[2], eb[2]),
                ("bottom", g[3], eb[3]),
            ] {
                assert!(
                    (gv - ev).abs() <= tol,
                    "{id} char {i}: {axis} drifted: got {gv}, expected {ev} \
                     (tol ±{tol}px)\n  got      {g:?}\n  expected {eb:?}"
                );
                worst_delta = worst_delta.max((gv - ev).abs());
            }
            if dump() {
                errs.push((g[2] - g[0]) - (eb[2] - eb[0]));
            }
        }

        // reporting only: how far the PORT sits from the reference (scalar, no
        // rects — the dump must never be paste-able as expectations).
        if dump() {
            println!("DUMP {id} parity: worst edge delta {worst_delta:.4}px (tol ±{tol}px)");
        }

        // The tap contract (Android `boxesContainTheirCharactersInkCentre`):
        // each character's gold ink centre sits on its own box's reading
        // axis, 0.5 px slack. Also the honest gate for the `gold` metrics,
        // which report the Tier-2-style rates per case (report-only: the pin
        // is parity, above).
        if let Some(gold) = c.get("gold") {
            let axis = usize::from(vertical);
            let ink_boxes = gold["ink_boxes"].as_array().expect("ink_boxes");
            let d2t = gold["decoded_to_true"].as_array().expect("decoded_to_true");
            let mut centre_errs: Vec<f64> = Vec::new();
            let mut covered = 0usize;
            let mut tap = 0usize;
            let mut eligible = 0usize;
            let mut first_bad: Option<(usize, f64, f64, f64, [f64; 4])> = None;
            for (i, map) in d2t.iter().enumerate() {
                let Some(true_idx) = map.as_i64() else { continue };
                let Some(ink) = ink_boxes.get(true_idx as usize) else { continue };
                let ink: Vec<f64> = ink
                    .as_array()
                    .expect("ink box")
                    .iter()
                    .map(|x| x.as_f64().expect("float"))
                    .collect();
                if ink[2] <= ink[0] || ink[3] <= ink[1] {
                    continue;
                }
                let Some(got_box) = got.get(i) else { continue };
                eligible += 1;
                let centre = (ink[axis] + ink[axis + 2]) / 2.0;
                let (lo, hi) = (got_box[axis], got_box[axis + 2]);
                if centre >= lo - 0.5 && centre <= hi + 0.5 {
                    covered += 1;
                } else if first_bad.is_none() {
                    first_bad = Some((i, centre, lo, hi, *got_box));
                }
                centre_errs.push((centre - 0.5 * (lo + hi)).abs());
                let box_centre = 0.5 * (lo + hi);
                if box_centre >= ink[axis] - 0.5 && box_centre <= ink[axis + 2] + 0.5 {
                    tap += 1;
                }
            }
            if dump() && eligible > 0 {
                let n = eligible as f64;
                let mean = centre_errs.iter().sum::<f64>() / n;
                let p90 = percentile_px(&mut centre_errs, 90.0);
                println!(
                    "DUMP {id} metrics: n={eligible} centre_err mean={mean:.3}px \
                     p90={p90:.3}px cover={:.3} tap={:.3} width_delta mean={:.3}px",
                    covered as f64 / n,
                    tap as f64 / n,
                    errs.iter().sum::<f64>() / errs.len().max(1) as f64,
                );
            }
            if let Some((i, centre, lo, hi, rect)) = first_bad {
                panic!(
                    "{id} char {i}: ink centre {centre} outside box [{lo}, {hi}] \
                     (±0.5px, {covered}/{eligible} covered)\n  got {rect:?}"
                );
            }
        }
    }
}

fn jitendex_definitions(term: &str, reading: &str) -> Value {
    let path = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
        .join("tests")
        .join("data")
        .join("jitendex")
        .join("entries.json");
    let root: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("entries.json reads"))
            .expect("entries.json parses");
    for e in root["entries"].as_array().expect("entries") {
        if e["term"].as_str() == Some(term) && e["reading"].as_str() == Some(reading) {
            return e["definitions"].clone();
        }
    }
    panic!("no jitendex fixture for {term} ({reading})");
}

fn flatten_nodes(nodes: &[crate::models::DefinitionNode]) -> Vec<crate::models::DefinitionNode> {
    use crate::models::DefinitionNode::*;
    let mut out = nodes.to_vec();
    for n in nodes {
        let kids: Vec<crate::models::DefinitionNode> = match n {
            Group { nodes, .. } => nodes.clone(),
            ListBlock { items, .. } => items.iter().flatten().cloned().collect(),
            Example(ex) => ex.content.iter().chain(ex.parts.iter().flatten()).cloned().collect(),
            Table { rows } => rows.iter().flatten().flatten().cloned().collect(),
            _ => Vec::new(),
        };
        out.extend(flatten_nodes(&kids));
    }
    out
}

#[test]
fn dictionary_cases() {
    for (name, v) in kind_cases("dictionary") {
        let id = case_id(&name, &v);
        let c = &v["case"];
        let term = c["term"].as_str().expect("term");
        let reading = c["reading"].as_str().expect("reading");
        let defs = jitendex_definitions(term, reading);
        let entry = DictionaryEntry {
            id: 1,
            kanji: term.to_string(),
            reading: reading.to_string(),
            definitions: serde_json::to_string(&defs).expect("defs serialize"),
            rules: String::new(),
            popularity: 0,
            dictionary_id: 1,
            onyomi: None,
            kunyomi: None,
            jlpt: None,
        };
        let matches = vec![TermMatch {
            term: term.to_string(),
            entries: vec![entry],
            chain: None,
        }];
        let dict_names: HashMap<i64, String> =
            [(1i64, "Jitendex".to_string())].into_iter().collect();
        let mut state = OcrOverlayState::new(1024.0, 768.0);
        let out: Vec<FormattedEntry> = state.format_dictionary_results(&matches, &dict_names);
        assert_eq!(out.len(), 1, "{id}: expected one formatted entry");
        let exp = &c["expect"];
        let groups = &out[0].reading_groups;
        let readings: Vec<&str> = groups.iter().map(|g| g.reading.as_str()).collect();
        let want_readings: Vec<&str> = exp["readings"]
            .as_array()
            .expect("readings")
            .iter()
            .map(|s| s.as_str().expect("str"))
            .collect();
        assert_eq!(readings, want_readings, "{id}: reading groups drifted");
        let headwords: Vec<&str> = groups
            .iter()
            .flat_map(|g| g.headwords.iter().map(|h| h.kanji.as_str()))
            .collect();
        let want_hw: Vec<&str> = exp["headwords"]
            .as_array()
            .expect("headwords")
            .iter()
            .map(|s| s.as_str().expect("str"))
            .collect();
        assert_eq!(headwords, want_hw, "{id}: headwords drifted");
        let total_groups: usize = groups.iter().map(|g| g.sense_groups.len()).sum();
        assert!(
            total_groups >= exp["min_sense_groups"].as_i64().expect("min_sense_groups") as usize,
            "{id}: sense groups shrank to {total_groups}"
        );
        let texts: Vec<String> = groups
            .iter()
            .flat_map(|g| g.sense_groups.iter())
            .flat_map(|sg| sg.senses.iter())
            .flat_map(|s| flatten_nodes(&s.nodes))
            .filter_map(|n| match n {
                crate::models::DefinitionNode::Text(t) => Some(t),
                _ => None,
            })
            .collect();
        let blob = texts.join("\n");
        for key in ["example_ja", "example_en"] {
            if let Some(want) = exp[key].as_str() {
                assert!(
                    blob.contains(want),
                    "{id}: {key} {want:?} missing from formatted senses:\n{blob}"
                );
            }
        }
        if dump() {
            println!("DUMP {id} readings: {readings:?} headwords: {headwords:?}");
        }
    }
}
