//! The face the zoom overlay paints in — sans/gothic (Noto Sans JP) or
//! serif/mincho (Noto Serif JP) — and the file resolution behind it.
//!
//! Port of mobile `OverlayFont.kt` (#84). Both faces ship next to the
//! executable (`build.rs` copies `fonts/` into `target/{profile}/fonts`), and
//! the sans face keeps the system Noto CJK fallbacks the PC overlay has
//! always had. A serif selection whose bundled file is missing warns and
//! resolves the sans face instead: the overlay degrades, it never panics
//! (mobile falls back to the platform sans face for the same case).

use std::path::{Path, PathBuf};

/// Bundled face files, matching the names mobile ships under `assets/fonts/`.
pub const SANS_ASSET: &str = "NotoSansJP-Regular.ttf";
pub const SERIF_ASSET: &str = "NotoSerifJP-Regular.ttf";
/// Bundled bold companions. Only sans ships one today (the real-bold face for
/// highlights); serif highlights fake-bold like mobile, which bundles Regular
/// instances only.
pub const SANS_BOLD_ASSET: &str = "NotoSansJP-Bold.ttf";
pub const SERIF_BOLD_ASSET: &str = "NotoSerifJP-Bold.ttf";

/// Faces the overlay can paint in. Stored as `"sans"` / `"serif"` — the exact
/// strings mobile's `OverlayFont` reads and writes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FontFace {
    Sans,
    Serif,
}

impl Default for FontFace {
    /// Today's appearance: the sans face the overlay drew before the switch
    /// existed.
    fn default() -> Self {
        Self::Sans
    }
}

impl FontFace {
    /// The value persisted in the settings store.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sans => "sans",
            Self::Serif => "serif",
        }
    }

    /// A stored value this build does not know reads as the default, never as
    /// an error (mobile `OverlayFont.normalize`).
    pub fn parse(raw: &str) -> Self {
        if raw == "serif" {
            Self::Serif
        } else {
            Self::Sans
        }
    }

    /// The bundled Regular file for this face.
    pub fn asset_name(self) -> &'static str {
        match self {
            Self::Sans => SANS_ASSET,
            Self::Serif => SERIF_ASSET,
        }
    }

    /// The bold companion this face would use if bundled. `find_bold_font_path`
    /// returns `None` when the file is absent, and highlights fall back to
    /// synthetic emboldening — no face ever mixes another face's bold.
    pub fn bold_asset_name(self) -> &'static str {
        match self {
            Self::Sans => SANS_BOLD_ASSET,
            Self::Serif => SERIF_BOLD_ASSET,
        }
    }

    /// Family name the font database sees after the file is registered (the
    /// `name` table's family record). Used as the iced application default
    /// font so the dictionary panel renders in the selected face.
    pub fn family_name(self) -> &'static str {
        match self {
            Self::Sans => "Noto Sans JP",
            Self::Serif => "Noto Serif JP",
        }
    }
}

impl serde::Serialize for FontFace {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for FontFace {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::parse(&raw))
    }
}

/// System sans-JP fallbacks, tried when no bundled sans file is present.
/// The `.ttc` collections are readable by ttf-parser but not by fontdue, so
/// the glyph cache refuses them at parse time and the overlay draws boxes
/// without glyphs — the same degradation this path has always had.
const SYSTEM_SANS_FALLBACKS: &[&str] = &[
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc",
];

/// Directories a bundled face may live in, in order: next to the executable
/// (AppImage, installed builds), the AppImage mount, then the working dir
/// (dev runs from the crate root).
fn bundled_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("fonts"));
        }
    }
    if let Ok(appdir) = std::env::var("APPDIR") {
        dirs.push(Path::new(&appdir).join("usr").join("bin").join("fonts"));
    }
    dirs.push(PathBuf::from("fonts"));
    dirs
}

/// First bundled dir holding `file`, per the injectable `exists` check.
fn bundled_candidate(file: &str, exists: &dyn Fn(&Path) -> bool) -> Option<PathBuf> {
    bundled_dirs()
        .into_iter()
        .map(|dir| dir.join(file))
        .find(|path| exists(path))
}

/// Resolve the Regular file for `face`. The selected face's bundled file wins;
/// when a serif file is absent, warn and resolve the sans face instead
/// (mobile #84's fallback to the platform sans face).
pub fn find_font_path(face: FontFace) -> Option<PathBuf> {
    resolve_font_path(face, &|path| path.exists())
}

/// Resolve the bold companion for `face`, if it ships next to the regular
/// face. No system candidates: fontdue cannot read the system `.ttc`
/// collections, so a missing bundled bold simply means synthetic bold.
pub fn find_bold_font_path(face: FontFace) -> Option<PathBuf> {
    bundled_candidate(face.bold_asset_name(), &|path| path.exists())
}

/// Which face a resolved path actually carries — `Serif` only for a bundled
/// Noto Serif file. A serif request that fell back to the sans file therefore
/// pairs with the sans bold, never a mismatched one.
pub fn face_of(path: &Path) -> FontFace {
    match path.file_name().and_then(|name| name.to_str()) {
        Some(name) if name.starts_with("NotoSerifJP") => FontFace::Serif,
        _ => FontFace::Sans,
    }
}

/// Resolution core with an injectable filesystem check, so tests can drive
/// every fallback without touching real files.
fn resolve_font_path(face: FontFace, exists: &dyn Fn(&Path) -> bool) -> Option<PathBuf> {
    if let Some(path) = bundled_candidate(face.asset_name(), exists) {
        return Some(path);
    }
    if face == FontFace::Serif {
        eprintln!("[OverlayFont] bundled serif face {SERIF_ASSET} not found; falling back to sans");
        if let Some(path) = bundled_candidate(FontFace::Sans.asset_name(), exists) {
            return Some(path);
        }
    }
    SYSTEM_SANS_FALLBACKS
        .iter()
        .map(Path::new)
        .find(|path| exists(path))
        .map(Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A filesystem predicate that only "finds" the given file names, so the
    /// fallback chains can be exercised without real files.
    fn only<'a>(names: &'a [&'a str]) -> impl Fn(&Path) -> bool + 'a {
        move |path: &Path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| names.contains(&name))
                .unwrap_or(false)
        }
    }

    /// The stored strings match mobile's `OverlayFont`; anything unknown reads
    /// as the default instead of failing.
    #[test]
    fn stored_values_round_trip_and_normalize() {
        assert_eq!(FontFace::parse("sans"), FontFace::Sans);
        assert_eq!(FontFace::parse("serif"), FontFace::Serif);
        assert_eq!(FontFace::parse("SERIF"), FontFace::Sans, "unknown reads as default");
        assert_eq!(FontFace::parse(""), FontFace::Sans);
        assert_eq!(FontFace::default(), FontFace::Sans);
        assert_eq!(FontFace::Sans.as_str(), "sans");
        assert_eq!(FontFace::Serif.as_str(), "serif");
    }

    #[test]
    fn serde_uses_the_stored_strings_and_never_errors() {
        assert_eq!(serde_json::to_string(&FontFace::Serif).unwrap(), "\"serif\"");
        let serif: FontFace = serde_json::from_str("\"serif\"").unwrap();
        assert_eq!(serif, FontFace::Serif);
        let unknown: FontFace = serde_json::from_str("\"comic\"").unwrap();
        assert_eq!(unknown, FontFace::Sans);
    }

    /// Sans is the default and serif is picked when selected: both resolve the
    /// bundled file of their own face from the dev/test tree.
    #[test]
    fn face_resolution_prefers_the_selected_bundled_file() {
        let sans = find_font_path(FontFace::Sans).expect("bundled sans face");
        let serif = find_font_path(FontFace::Serif).expect("bundled serif face");
        assert_eq!(sans.file_name().unwrap(), SANS_ASSET);
        assert_eq!(serif.file_name().unwrap(), SERIF_ASSET);
        assert_eq!(face_of(&sans), FontFace::Sans);
        assert_eq!(face_of(&serif), FontFace::Serif);
    }

    /// The executable-relative copy is preferred over the working-dir one, so
    /// installed/AppImage runs never pick up a stray dev file.
    #[test]
    fn bundled_search_prefers_the_executable_dir() {
        let sandbox = only(&[SANS_ASSET]);
        let resolved = bundled_candidate(SANS_ASSET, &sandbox).expect("candidate");
        let exe_dir = std::env::current_exe().unwrap();
        assert_eq!(
            resolved,
            exe_dir.parent().unwrap().join("fonts").join(SANS_ASSET)
        );
    }

    /// Mobile #84: a serif selection whose asset is missing resolves the sans
    /// face instead of failing.
    #[test]
    fn missing_serif_falls_back_to_sans() {
        let sandbox = only(&[SANS_ASSET]);
        let resolved = resolve_font_path(FontFace::Serif, &sandbox).expect("sans fallback");
        assert_eq!(resolved.file_name().unwrap(), SANS_ASSET);
        assert_eq!(face_of(&resolved), FontFace::Sans);
        // Sans resolution is unchanged: it picks the same file.
        assert_eq!(resolve_font_path(FontFace::Sans, &sandbox), Some(resolved));
    }

    /// With no bundled files at all, sans still resolves the system CJK
    /// collection the overlay has always fallen back to.
    #[test]
    fn system_sans_is_the_last_resort() {
        let fallback = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc";
        let names = ["NotoSansCJK-Regular.ttc"];
        let sandbox = only(&names);
        assert_eq!(
            resolve_font_path(FontFace::Sans, &sandbox),
            Some(PathBuf::from(fallback))
        );
        // A serif request falls through the same sans chain.
        assert_eq!(
            resolve_font_path(FontFace::Serif, &sandbox),
            Some(PathBuf::from(fallback))
        );
    }

    /// Nothing on disk: resolution reports `None` (the glyph cache warns and
    /// draws no glyphs), and never panics.
    #[test]
    fn no_candidates_resolve_to_none() {
        let nothing = |_: &Path| false;
        assert!(resolve_font_path(FontFace::Sans, &nothing).is_none());
        assert!(resolve_font_path(FontFace::Serif, &nothing).is_none());
    }

    /// The real dev tree ships a sans bold companion; serif ships Regular
    /// only, so its highlights use synthetic bold (mobile parity).
    #[test]
    fn bundled_bold_companion_is_found_for_sans() {
        let bold = find_bold_font_path(FontFace::Sans).expect("bundled sans bold");
        assert_eq!(bold.file_name().unwrap(), SANS_BOLD_ASSET);
        if let Some(serif_bold) = find_bold_font_path(FontFace::Serif) {
            assert_eq!(serif_bold.file_name().unwrap(), SERIF_BOLD_ASSET);
        }
    }
}
