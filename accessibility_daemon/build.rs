use std::env;
use std::fs;
use std::path::Path;

fn main() {
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
        target_dir = target_dir.parent()
            .expect("Could not find target/ directory in OUT_DIR path");
    }

    let dest_dir = target_dir.join(&profile).join("assets");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let assets_dir = Path::new(&manifest_dir).join("assets");

    if assets_dir.exists() {
        println!("cargo:rerun-if-changed=assets/");
        match copy_dir_all(&assets_dir, &dest_dir) {
            Ok(()) => println!("cargo:warning=Copied assets to {:?}", dest_dir),
            Err(e) => panic!("Failed to copy assets: {}", e),
        }
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
