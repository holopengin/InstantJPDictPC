use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    // ------------------------------------------------------------------
    // Shared PP-OCRv6 ncnn backend (native/ppocr_ncnn + the pinned fork)
    // ------------------------------------------------------------------
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let ncnn_dir = env::var("NCNN_PC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("third_party/ncnn-pc/install"));

    let ncnn_lib_dir = ncnn_dir.join("lib");
    let ncnn_include_dir = ncnn_dir.join("include");
    if !ncnn_lib_dir.join("libncnn.a").exists() {
        panic!(
            "pinned ncnn fork not found at {}\n\
             Build it first:  tools/build_ncnn_pc.sh\n\
             (or point NCNN_PC_DIR at an install prefix)",
            ncnn_dir.display()
        );
    }

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include(&ncnn_include_dir)
        .include(manifest_dir.join("native/ppocr_ncnn"))
        .file(manifest_dir.join("native/ppocr_ncnn/ppocr_ncnn_core.cpp"))
        .file(manifest_dir.join("native/ppocr_ncnn/ppocr_ncnn_capi.cpp"))
        .compile("ppocr_ncnn");

    println!("cargo:rustc-link-search=native={}", ncnn_lib_dir.display());
    println!("cargo:rustc-link-lib=static=ncnn");
    // The fork is built with OpenMP on (same as the mobile build); libncnn
    // references GOMP_* symbols, so the final link needs the OpenMP runtime.
    println!("cargo:rustc-link-arg=-fopenmp");
    println!("cargo:rerun-if-changed=native/ppocr_ncnn");
    println!("cargo:rerun-if-env-changed=NCNN_PC_DIR");

    // ------------------------------------------------------------------
    // Kana small/large model (#44) C ABI (native/kana_size)
    // ------------------------------------------------------------------
    // Separate from the shared PP-OCR core on purpose: this model is not part
    // of it (the mobile side has a JNI wrapper, not a core file), but it links
    // the same pinned ncnn fork.
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include(&ncnn_include_dir)
        .include(manifest_dir.join("native/kana_size"))
        .file(manifest_dir.join("native/kana_size/kana_size_capi.cpp"))
        .compile("kana_size");
    println!("cargo:rerun-if-changed=native/kana_size");

    // ------------------------------------------------------------------
    // Assets -> target/{profile}/assets (existing behaviour)
    // ------------------------------------------------------------------
    let out_dir = env::var("OUT_DIR").unwrap();
    let profile = env::var("PROFILE").unwrap();

    // OUT_DIR is target/{profile}/build/{crate}-{hash}/out
    // Walk up to find the target/ directory
    let mut target_dir = Path::new(&out_dir);
    loop {
        let name = target_dir.file_name().and_then(|n| n.to_str());
        if name == Some("target") {
            break;
        }
        target_dir = target_dir
            .parent()
            .expect("Could not find target/ directory in OUT_DIR path");
    }

    let dest_dir = target_dir.join(&profile).join("assets");
    let assets_dir = manifest_dir.join("assets");

    if assets_dir.exists() {
        println!("cargo:rerun-if-changed=assets/");
        // Replace, don't merge: a renamed/removed model dir must not linger
        // in target/{profile}/assets and shadow the current one.
        if dest_dir.exists() {
            if let Err(e) = fs::remove_dir_all(&dest_dir) {
                panic!("Failed to clear stale assets at {:?}: {}", dest_dir, e);
            }
        }
        match copy_dir_all(&assets_dir, &dest_dir) {
            Ok(()) => println!("cargo:warning=Copied assets to {:?}", dest_dir),
            Err(e) => panic!("Failed to copy assets: {}", e),
        }
    }

    // ------------------------------------------------------------------
    // Fonts -> target/{profile}/fonts (next to the executable)
    // ------------------------------------------------------------------
    // overlay_font::find_font_path()/find_bold_font_path() check exe_dir/fonts
    // first, so copying here keeps every bundled face (sans + serif, plus the
    // sans bold companion) discoverable regardless of the working directory
    // (dev builds, systemd app launches, AppImage).
    let fonts_src = manifest_dir.join("fonts");
    let fonts_dest = target_dir.join(&profile).join("fonts");
    if fonts_src.exists() {
        println!("cargo:rerun-if-changed=fonts/");
        if let Err(e) = copy_dir_all(&fonts_src, &fonts_dest) {
            panic!("Failed to copy fonts: {}", e);
        }
        println!("cargo:warning=Copied fonts to {:?}", fonts_dest);
    }
}

fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    if !dst.exists() {
        fs::create_dir_all(dst)?;
    }
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &dst_path)?;
        } else {
            fs::copy(&entry.path(), &dst_path)?;
        }
    }
    Ok(())
}
