//! Tier-2 scorer input for `docs/char-placement-conformance.md` §4–5: run CAP
//! over a `synthesize.py` corpus (`cases.jsonl` + crop PNGs) and write the
//! `pc_boxes.jsonl` that `eval.py --boxes-from` scores as the `imported` row.
//!
//! ```text
//! cargo run -p jpdict_core --example cap_dump -- <data_dir> <out.jsonl>
//! /tmp/opencode/cap-venv/bin/python eval.py --data <data_dir> --only clean \
//!     --boxes-from <out.jsonl>
//! ```
//!
//! Inputs are read verbatim from each case record — the same values the
//! `char_placement` conformance kind pins (`decoded_text`, `char_cols`,
//! `seq_len_total`, `crop_w/h`, `orientation`, `steps`) plus the RGB crop
//! (channel-mean f32 luminance, the reference's uint8 path).

use std::io::Write as _;

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: cap_dump <data_dir> <out.jsonl>";
    let data = std::path::PathBuf::from(args.next().unwrap_or_else(|| usage.into()));
    let out_path = args.next().unwrap_or_else(|| usage.into());

    let index = std::fs::read_to_string(data.join("cases.jsonl"))
        .expect("<data_dir>/cases.jsonl reads");
    let mut out = std::fs::File::create(&out_path).expect("out.jsonl creates");
    let mut count = 0usize;
    for line in index.lines().filter(|l| !l.trim().is_empty()) {
        let case: serde_json::Value = serde_json::from_str(line).expect("case is JSON");
        let id = case["id"].as_str().expect("id");
        let crop_w = u32::try_from(case["crop_w"].as_i64().expect("crop_w")).expect("crop_w");
        let crop_h = u32::try_from(case["crop_h"].as_i64().expect("crop_h")).expect("crop_h");
        let img = image::open(data.join(case["image"].as_str().expect("image")))
            .expect("crop PNG loads")
            .to_rgb8();
        assert_eq!(
            (img.width(), img.height()),
            (crop_w, crop_h),
            "{id}: PNG must be crop_w x crop_h"
        );
        let text = case["decoded_text"].as_str().expect("decoded_text");
        let char_cols: Vec<f32> = case["char_cols"]
            .as_array()
            .expect("char_cols")
            .iter()
            .map(|x| x.as_f64().expect("float") as f32)
            .collect();
        let seq_len_total = usize::try_from(case["seq_len_total"].as_i64().expect("seq_len_total"))
            .expect("seq_len_total");
        let vertical = match case["orientation"].as_str().expect("orientation") {
            "v" => true,
            "h" => false,
            o => panic!("{id}: bad orientation {o:?}"),
        };
        let steps: Vec<Vec<(char, f64)>> = case["steps"]
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

        let lum = jpdict_core::char_placement::luminance_from_rgb(img.as_raw(), crop_w, crop_h);
        let boxes = jpdict_core::char_placement::place(
            text,
            &char_cols,
            seq_len_total,
            crop_w,
            crop_h,
            vertical,
            Some((&lum, crop_w, crop_h)),
            Some(&steps),
        );
        assert_eq!(
            boxes.len(),
            char_cols.len(),
            "{id}: one box per decoded character"
        );
        let rec = serde_json::json!({ "id": id, "boxes": boxes });
        writeln!(out, "{rec}").expect("write record");
        count += 1;
    }
    eprintln!("wrote {count} cases -> {out_path}");
}
