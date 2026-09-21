//! Catalog download + integrity check (#71).
//!
//! Nothing reaches the database until the pinned byte size and SHA-256 have
//! both been confirmed, so an intercepted, truncated or corrupted download
//! cannot become a dictionary. Mirrors mobile's `DictionaryDownloader` /
//! `DictionaryDownload.writeVerified`.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;

use crate::data::catalog::CatalogEntry;

/// Download `entry` to `dest`, hashing while it streams, and verify the pins.
/// `on_progress` receives `(written, total)`; the total is the pinned size.
/// `dest` is deleted on any failure.
pub fn download_verified(
    entry: &CatalogEntry,
    dest: &Path,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<()> {
    let response = ureq::get(&entry.url)
        .set("User-Agent", "InstantJPDict/1.0 (dictionary catalog)")
        .call()
        .with_context(|| format!("download failed for {} ({})", entry.name, entry.url))?;
    let mut reader = response.into_reader();
    write_verified(&mut reader, dest, entry.bytes, &entry.sha256, on_progress)
        .with_context(|| format!("verified install failed for {}", entry.name))
}

/// Stream `reader` into `dest`, hashing as it goes, then check the pinned size
/// and SHA-256. Deletes `dest` when anything fails.
pub fn write_verified(
    reader: &mut dyn Read,
    dest: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    on_progress: &mut dyn FnMut(u64, u64),
) -> Result<()> {
    let result = (|| -> Result<()> {
        let mut file =
            std::fs::File::create(dest).with_context(|| format!("cannot create {}", dest.display()))?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 64 * 1024];
        let mut written = 0u64;
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            hasher.update(&buf[..n]);
            written += n as u64;
            on_progress(written, expected_bytes);
        }
        file.flush()?;
        drop(file);
        if written != expected_bytes {
            bail!("size mismatch: got {written} bytes, expected {expected_bytes}");
        }
        let digest = format!("{:x}", hasher.finalize());
        if digest != expected_sha256 {
            bail!("SHA-256 mismatch: got {digest}, expected {expected_sha256}");
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(dest);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::path::PathBuf;

    fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ijd_download_test_{}_{name}.zip", std::process::id()))
    }

    /// A verified stream lands on disk and reports progress up to its size.
    #[test]
    fn a_verified_stream_lands_and_reports_progress() {
        // Larger than the 64 KiB chunk so progress is reported more than once.
        let payload: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        let dest = temp_path("ok");
        let mut seen = Vec::new();
        let mut cursor = Cursor::new(payload.clone());
        write_verified(
            &mut cursor,
            &dest,
            payload.len() as u64,
            &sha256_hex(&payload),
            &mut |written, total| seen.push((written, total)),
        )
        .expect("verified write");
        assert_eq!(std::fs::read(&dest).unwrap(), payload);
        assert!(seen.len() > 1, "chunked progress: {seen:?}");
        assert_eq!(*seen.last().unwrap(), (payload.len() as u64, payload.len() as u64));
        let _ = std::fs::remove_file(&dest);
    }

    /// A size or hash mismatch deletes the partial file: nothing unverified may
    /// survive to be imported.
    #[test]
    fn a_wrong_size_or_hash_leaves_no_file() {
        let payload = b"dictionary bytes".to_vec();
        let dest = temp_path("bad");

        let mut cursor = Cursor::new(payload.clone());
        assert!(write_verified(
            &mut cursor,
            &dest,
            payload.len() as u64 + 1,
            &sha256_hex(&payload),
            &mut |_, _| {},
        )
        .is_err());
        assert!(!dest.exists(), "size mismatch leaves no file");

        let mut cursor = Cursor::new(payload.clone());
        assert!(write_verified(
            &mut cursor,
            &dest,
            payload.len() as u64,
            &"0".repeat(64),
            &mut |_, _| {},
        )
        .is_err());
        assert!(!dest.exists(), "hash mismatch leaves no file");
    }
}
