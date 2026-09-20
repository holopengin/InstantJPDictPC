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
use crate::data::models::DictionaryEntry;
use crate::models::{
    BoundingBox, FormattedEntry, LineResult, RotatedBox, TermMatch, GAP_CHAR,
};
use crate::ocr_engine::{
    filter_fitted_quads, fit_components, merge_straight_boxes, rect_of, OcrEngine,
};
use crate::overlay_state::OcrOverlayState;
use crate::util::char_lm::CharLm;
use crate::util::gap_candidates::{self, MAX as GAP_MAX};
use crate::models::RecognitionMode;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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
    let dir = format!("{}/assets", env!("CARGO_MANIFEST_DIR"));
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

fn jitendex_definitions(term: &str, reading: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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
