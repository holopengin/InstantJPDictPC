//! Native file watcher — replaces `screenshot/watch_and_ocr.sh`.
//!
//! Polls the screenshot directories for new images and spawns the OCR viewer
//! (`accessibility_daemon <image>`) for each one. Non-blocking: multiple
//! viewers may run concurrently, and finished children are reaped each poll
//! so no zombies accumulate.
//!
//! Process coordination with the frontend window:
//!   - `watcher.pid`  — written on start, removed on clean stop
//!   - `watcher.stop` — written by the frontend to request a stop
//!
//! Run as: `accessibility_daemon --watcher [daemon args...]`

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime};

/// How often to poll the screenshot directories.
const POLL_INTERVAL: Duration = Duration::from_millis(750);
/// How long to wait for a newly-detected file's size to stabilize.
const WRITE_SETTLE: Duration = Duration::from_millis(700);
/// Image extensions accepted from the screenshot directory.
const IMAGE_EXTS: [&str; 6] = ["png", "jpg", "jpeg", "bmp", "webp", "jxl"];

fn pid_file(data_dir: &Path) -> PathBuf {
    data_dir.join("watcher.pid")
}

fn stop_file(data_dir: &Path) -> PathBuf {
    data_dir.join("watcher.stop")
}

/// True if a watcher process is currently alive.
pub fn is_running(data_dir: &Path) -> bool {
    let Ok(pid_str) = std::fs::read_to_string(pid_file(data_dir)) else {
        return false;
    };
    let Ok(pid) = pid_str.trim().parse::<u32>() else {
        return false;
    };
    let proc_path = format!("/proc/{pid}");
    let proc_root = Path::new(&proc_path);
    if !proc_root.exists() {
        return false;
    }
    // The pid must actually be our watcher, not an unrelated process that
    // reused the pid after the original watcher died (stale pid file).
    let Ok(cmdline) = std::fs::read(proc_root.join("cmdline")) else {
        return false;
    };
    cmdline
        .split(|&b| b == 0)
        .any(|arg| arg == b"--watcher")
}

/// Ask the running watcher to stop: write the stop marker and nudge the
/// process with SIGTERM in case it is mid-scan.
pub fn request_stop(data_dir: &Path) {
    let _ = std::fs::write(stop_file(data_dir), "stop\n");
    if let Ok(pid_str) = std::fs::read_to_string(pid_file(data_dir)) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            let _ = Command::new("kill").arg(pid.to_string()).status();
        }
    }
}

fn screenshot_dir() -> PathBuf {
    std::env::var("SCREENSHOT_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join("Pictures").join("Screenshots")
    })
}

fn has_image_ext(p: &Path) -> bool {
    match p.extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            let ext = ext.to_ascii_lowercase();
            IMAGE_EXTS.contains(&ext.as_str())
        }
        None => false,
    }
}

/// True if this filename is a watcher-owned temp/logged image that must
/// never be re-OCR'd. Covers BOOOCR sidecar crops (`booocr_in_*`) and
/// dataset line samples (`ocr_line_*`) which both land in `/tmp` (and
/// `ocr_line_*` can also appear next to the input in batch mode).
fn is_ignored_temp_output(name: &str) -> bool {
    name.starts_with("ocr_line_") || name.starts_with("booocr_in_")
}

/// Run the watcher loop until a stop is requested. Blocks forever.
pub fn run(data_dir: &Path, extra_args: &[String]) {
    // Single instance: refuse to start if a watcher is already alive.
    // Without this, a frontend relaunch racing the pid-file write would
    // spawn a second watcher that overwrites the pid file, orphaning the
    // first one — which then could never be stopped from the UI.
    if is_running(data_dir) {
        eprintln!("[Watcher] A watcher is already running — refusing to start a second one.");
        return;
    }
    // Fresh start: clear any stale stop marker, then write our PID.
    let _ = std::fs::remove_file(stop_file(data_dir));
    let _ = std::fs::write(pid_file(data_dir), std::process::id().to_string());

    // Only images created/modified after this moment are processed; every
    // pre-existing file in the watch dirs is ignored.
    let started_at = SystemTime::now();

    // Shared BOOOCR pipeline: spawn it once here so every viewer launched
    // from this watcher connects to an already-initialized, fully warm
    // recognizer (DBs mmap'd, norms computed) instead of paying a cold
    // start per screenshot. Viewers discover it via BOOOCR_SOCKET.
    let socket_path = data_dir.join("booocr.sock");
    let _ = std::fs::remove_file(&socket_path); // stale socket
    let mut sidecar: Option<(std::process::Child, std::io::BufReader<std::process::ChildStdout>, PathBuf)> =
        match crate::booocr::spawn_sidecar_for_watcher(&socket_path) {
            Ok((child, reader)) => {
                println!(
                    "[Watcher] BOOOCR pipeline ready on {}",
                    socket_path.display()
                );
                Some((child, reader, socket_path.clone()))
            }
            Err(e) => {
                eprintln!(
                    "[Watcher] BOOOCR shared pipeline failed ({e}); viewers will spawn their own"
                );
                None
            }
        };

    let shot_dir = screenshot_dir();
    let processed_dir = shot_dir.join(".ocr_processed");
    let _ = std::fs::create_dir_all(&shot_dir);
    let _ = std::fs::create_dir_all(&processed_dir);

    println!("[Watcher] PID {}", std::process::id());
    println!(
        "[Watcher] Watching {} and new images in /tmp",
        shot_dir.display()
    );
    println!(
        "[Watcher] Extra daemon args: {}",
        if extra_args.is_empty() {
            "(none)".to_string()
        } else {
            extra_args.join(" ")
        }
    );
    println!(
        "[Watcher] Stop via frontend 'Stop File Watcher', marker {}, or Ctrl+C",
        stop_file(data_dir).display()
    );
    println!("[Watcher] Ready — take a screenshot to start OCR.");

    // Spawned viewer processes; reaped each poll to avoid zombies.
    let mut children: Vec<std::process::Child> = Vec::new();

    loop {
        // Respawn the shared pipeline if it died (crashed/killed). New
        // viewers fall back to their own sidecar until it's back.
        if let Some((child, reader, sock)) = &mut sidecar {
            if let Ok(Some(_)) = child.try_wait() {
                eprintln!("[Watcher] BOOOCR pipeline died — respawning");
                match crate::booocr::spawn_sidecar_for_watcher(sock) {
                    Ok((c, r)) => {
                        *child = c;
                        *reader = r;
                        println!("[Watcher] BOOOCR pipeline respawned on {}", sock.display());
                    }
                    Err(e) => eprintln!("[Watcher] BOOOCR respawn failed ({e})"),
                }
            }
        }

        scan(
            &shot_dir,
            &processed_dir,
            extra_args,
            &mut children,
            started_at,
            sidecar.as_ref().map(|(_, _, s)| s.as_path()),
        );

        // Reap finished viewer processes.
        children.retain_mut(|c| c.try_wait().ok().flatten().is_none());

        if stop_file(data_dir).exists() {
            break;
        }
        std::thread::sleep(POLL_INTERVAL);
    }

    // Stop the shared pipeline and remove its socket.
    if let Some((mut child, _reader, sock)) = sidecar {
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_file(&sock);
    }

    let _ = std::fs::remove_file(pid_file(data_dir));
    let _ = std::fs::remove_file(stop_file(data_dir));
    println!("[Watcher] Stopped.");
}

fn scan(
    shot_dir: &Path,
    processed_dir: &Path,
    extra_args: &[String],
    children: &mut Vec<std::process::Child>,
    started_at: SystemTime,
    socket: Option<&Path>,
) {
    // Screenshots land in /tmp: gamescope_*.png from gamescope, plus any
    // other image the user drops there. The daemon's own dataset crops
    // (ocr_line_*.png) and BOOOCR temp crops (booocr_in_*.png) are
    // excluded so we never re-OCR our own output.
    if let Ok(entries) = std::fs::read_dir("/tmp") {
        for entry in entries.flatten() {
            let p = entry.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !is_ignored_temp_output(name) && has_image_ext(&p) {
                maybe_process(&p, processed_dir, extra_args, children, started_at, socket);
            }
        }
    }

    // Screenshot directory images. Also skip our own output crops — batch
    // mode saves `ocr_line_*` next to the input (file.parent()), which for
    // screenshots IS this directory. Also exclude BOOOCR crops if they
    // ever land there.
    if let Ok(entries) = std::fs::read_dir(shot_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if has_image_ext(&p) && !is_ignored_temp_output(name) {
                maybe_process(&p, processed_dir, extra_args, children, started_at, socket);
            }
        }
    }
}

fn maybe_process(
    file: &Path,
    processed_dir: &Path,
    extra_args: &[String],
    children: &mut Vec<std::process::Child>,
    started_at: SystemTime,
    socket: Option<&Path>,
) {
    if !file.is_file() {
        return;
    }
    // Defense in depth: never re-OCR our own temp outputs even if the
    // scan's prefix filter is bypassed (e.g. file moved/renamed into
    // the watch dir after creation).
    if let Some(name) = file.file_name().and_then(|n| n.to_str()) {
        if is_ignored_temp_output(name) {
            return;
        }
    }
    let Ok(meta) = file.metadata() else {
        return;
    };
    // Ignore anything that predates this watcher run — only images that
    // appeared after startup are new screenshots.
    let Ok(mtime) = meta.modified() else {
        return;
    };
    if mtime < started_at {
        return;
    }
    let base = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let marker = processed_dir.join(base);
    if marker.exists() {
        return;
    }

    // Wait for the file to finish being written: the size must stabilize.
    let s1 = meta.len();
    std::thread::sleep(WRITE_SETTLE);
    let s2 = file.metadata().map(|m| m.len()).unwrap_or(0);
    if s1 == 0 || s1 != s2 {
        return; // still growing (or empty) — try again next poll
    }

    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("[Watcher] cannot resolve current exe");
            return;
        }
    };

    let mut cmd = Command::new(&exe);
    // Extra daemon args first: run_ocr_viewer stops parsing at the first
    // positional (image path), so flags must precede the file.
    cmd.args(extra_args)
        .arg(file)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // Hand the shared BOOOCR pipeline to the viewer so it connects
    // instead of spawning its own sidecar.
    if let Some(sock) = socket {
        cmd.env("BOOOCR_SOCKET", sock);
    }
    match cmd.spawn() {
        Ok(child) => {
            println!("[Watcher] New screenshot: {base} → OCR");
            children.push(child);
            let _ = std::fs::write(marker, "");
        }
        Err(e) => eprintln!("[Watcher] Failed to launch OCR for {base}: {e}"),
    }
}
