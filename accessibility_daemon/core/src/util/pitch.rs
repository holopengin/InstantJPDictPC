//! Pitch-accent payload parsing for stored dictionary definitions.
//!
//! Mobile `PitchAccent` (#43): a pitch dictionary entry is recognised by the
//! shape of its stored payload
//! (`{"reading":…, "pitches":[{"position":N},…]}`), not by which dictionary
//! it came from, so any Yomitan-compatible pitch dictionary works. These are
//! pure functions of the JSON text — no `self`, no database — kept here
//! (ungated) so nav-only builds can reach them; the mora/contour math lives
//! in [`crate::util::japanese`].

/// Downstep positions from a stored pitch payload, or None when the entry
/// is not pitch data. Detection is by payload shape
/// (`{"reading":…, "pitches":[{"position":N},…]}`).
pub fn pitch_positions_of(definitions_json: &str) -> Option<Vec<i32>> {
    let root: serde_json::Value = serde_json::from_str(definitions_json).ok()?;
    let obj = root.as_object()?;
    let pitches = obj.get("pitches")?;
    let pitches = pitches.as_array()?;
    obj.get("reading")?;
    let mut out: Vec<i32> = Vec::new();
    for p in pitches {
        let Some(position) = p
            .as_object()
            .and_then(|m| m.get("position"))
            .and_then(|v| v.as_i64())
        else {
            continue;
        };
        out.push(position as i32);
    }
    out.sort();
    out.dedup();
    Some(out)
}

/// Reading of a stored pitch payload (None when absent).
pub fn pitch_reading_of(definitions_json: &str) -> Option<String> {
    let root: serde_json::Value = serde_json::from_str(definitions_json).ok()?;
    root.as_object()?
        .get("reading")?
        .as_str()
        .map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors Android `PitchAccentTest.positions_parsed_from_kanjium_payload`:
    // a real Kanjium payload yields its downstep positions.
    #[test]
    fn positions_parsed_from_pitch_payload() {
        assert_eq!(
            pitch_positions_of(r#"{"reading":"ひと","pitches":[{"position":0},{"position":2}]}"#),
            Some(vec![0, 2])
        );
    }

    // Duplicates collapse, order normalized (mirrors the Android case).
    #[test]
    fn positions_deduped_and_sorted() {
        assert_eq!(
            pitch_positions_of(
                r#"{"reading":"あ","pitches":[{"position":3},{"position":0},{"position":3}]}"#
            ),
            Some(vec![0, 3])
        );
    }

    // Mirrors Android `PitchAccentTest.non_pitch_payloads_are_rejected`: both
    // halves of the shape are required, and a JSON string is not an object.
    #[test]
    fn non_pitch_payloads_are_rejected() {
        assert_eq!(pitch_positions_of(r#""just a gloss""#), None);
        assert_eq!(pitch_positions_of(r#"{"glossary":"x"}"#), None);
        // pitches present but no reading -> not a pitch payload
        assert_eq!(pitch_positions_of(r#"{"pitches":[{"position":1}]}"#), None);
        // reading present but no pitches -> not pitch data
        assert_eq!(pitch_positions_of(r#"{"reading":"きみ"}"#), None);
        assert_eq!(pitch_positions_of("not json{["), None);
        assert_eq!(pitch_positions_of(""), None);
    }

    // Mirrors Android `PitchAccentTest.reading_extracted`.
    #[test]
    fn reading_extracted() {
        assert_eq!(
            pitch_reading_of(r#"{"reading":"きみ","pitches":[{"position":1}]}"#),
            Some("きみ".to_string())
        );
        assert_eq!(pitch_reading_of(r#"{"pitches":[]}"#), None);
        assert_eq!(pitch_reading_of("nope"), None);
    }
}
