//! Real-file face-resolution tests for `jpdict_core::overlay_font`.
//!
//! These run here (not in the core crate) on purpose: dev-tree resolution
//! ends in the working-directory fallback (`fonts/` next to the crate
//! root), which holds when cargo runs this package's tests — the core
//! crate's own working directory is `core/`, where it does not. The
//! fallback-chain logic itself is pinned by predicate-based unit tests in
//! the core crate; this pins that the dev tree actually ships the files
//! the chain expects.

use jpdict_core::overlay_font::{face_of, find_font_path, FontFace, SANS_ASSET, SERIF_ASSET};

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
