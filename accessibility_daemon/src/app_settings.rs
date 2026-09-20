//! Small JSON settings store in the app data directory.
//!
//! The file sits next to `dictionary.sqlite` in the `ProjectDirs` data dir.
//! `AppSettings` is deliberately a flat, `#[serde(default)]` struct: a missing
//! key reads as its default, an unknown face string normalizes to sans (see
//! [`FontFace`]), and unknown keys written by a newer build are ignored — a
//! partial or stale file never fails the whole load.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::overlay_font::FontFace;

/// File name inside the app data directory.
pub const SETTINGS_FILE: &str = "settings.json";

/// Everything the frontend can persist today. Extend by adding a field with a
/// `Default`; older files keep loading.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// Face the OCR overlay paints in — and, on desktop, the iced dictionary
    /// panel too (see `run_ocr_viewer` for why the mobile overlay/panel split
    /// has no desktop counterpart).
    pub overlay_font: FontFace,
    /// #100: mobile's "Filter furigana in screenshots" switch, **off** by
    /// default — the rule is flaky on camera photos and a wrong drop costs a
    /// whole line. Off keeps every small contour, so ruby survives as its own
    /// line. There is no camera mode on desktop, so this is the only furigana
    /// switch (mobile also has a separate camera key).
    pub furigana_filter: bool,
}

impl AppSettings {
    /// Read the store. A missing file is normal (first run) and yields the
    /// defaults; an unreadable or malformed file warns and does the same, so
    /// a bad settings file can never take the app down.
    pub fn load(data_dir: &Path) -> Self {
        let path = data_dir.join(SETTINGS_FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Self::default(),
            Err(e) => {
                eprintln!(
                    "[settings] cannot read {}: {e}; using defaults",
                    path.display()
                );
                return Self::default();
            }
        };
        match serde_json::from_str(&text) {
            Ok(settings) => settings,
            Err(e) => {
                eprintln!(
                    "[settings] {} is not valid JSON ({e}); using defaults",
                    path.display()
                );
                Self::default()
            }
        }
    }

    /// Write the store. Temp file + rename: a crash mid-write cannot leave a
    /// half-written settings file behind.
    pub fn save(&self, data_dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(data_dir)?;
        let path = data_dir.join(SETTINGS_FILE);
        let tmp = data_dir.join(format!("{SETTINGS_FILE}.tmp"));
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A fresh temp dir; the `--test-threads=1` run makes the name unique.
    fn sandbox(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ijd_settings_test_{}_{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// No file yet: defaults, with sans as the shipped face.
    #[test]
    fn missing_file_reads_defaults() {
        let dir = sandbox("missing");
        let settings = AppSettings::load(&dir);
        assert_eq!(settings, AppSettings::default());
        assert_eq!(settings.overlay_font, FontFace::Sans);
        assert!(!settings.furigana_filter, "the furigana rule ships off");
    }

    /// #100: the furigana switch round-trips through the store.
    #[test]
    fn furigana_filter_round_trips() {
        let dir = sandbox("furigana");
        let on = AppSettings {
            furigana_filter: true,
            ..AppSettings::default()
        };
        on.save(&dir).expect("save on");
        assert!(AppSettings::load(&dir).furigana_filter);

        let off = AppSettings {
            furigana_filter: false,
            ..AppSettings::default()
        };
        off.save(&dir).expect("save off");
        assert!(!AppSettings::load(&dir).furigana_filter);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Selecting serif survives a save/load cycle, and switching back does too.
    #[test]
    fn round_trip_persists_the_face() {
        let dir = sandbox("round_trip");
        let serif = AppSettings {
            overlay_font: FontFace::Serif,
            ..AppSettings::default()
        };
        serif.save(&dir).expect("save serif");
        assert_eq!(AppSettings::load(&dir), serif);
        assert!(std::fs::read_to_string(dir.join(SETTINGS_FILE))
            .unwrap()
            .contains("\"serif\""));

        let sans = AppSettings {
            overlay_font: FontFace::Sans,
            ..AppSettings::default()
        };
        sans.save(&dir).expect("save sans");
        assert_eq!(AppSettings::load(&dir).overlay_font, FontFace::Sans);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A missing key and an unknown face value both read as the default, and
    /// keys from a future build are ignored rather than failing the load.
    #[test]
    fn missing_and_unknown_keys_read_defaults() {
        let dir = sandbox("keys");
        std::fs::create_dir_all(&dir).unwrap();

        std::fs::write(dir.join(SETTINGS_FILE), "{}").unwrap();
        assert_eq!(AppSettings::load(&dir), AppSettings::default());

        std::fs::write(
            dir.join(SETTINGS_FILE),
            r#"{"overlay_font":"comic","future_option":1}"#,
        )
        .unwrap();
        assert_eq!(AppSettings::load(&dir).overlay_font, FontFace::Sans);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A corrupt file degrades to defaults instead of an error.
    #[test]
    fn corrupt_file_reads_defaults() {
        let dir = sandbox("corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(SETTINGS_FILE), "{not json").unwrap();
        assert_eq!(AppSettings::load(&dir), AppSettings::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The temp file must not linger after a successful save.
    #[test]
    fn save_leaves_no_temp_file() {
        let dir = sandbox("temp");
        AppSettings::default().save(&dir).expect("save");
        assert!(!dir.join(format!("{SETTINGS_FILE}.tmp")).exists());
        assert!(dir.join(SETTINGS_FILE).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
