//! BOOOCR sidecar client.
//!
//! BOOOCR is a Python feature-vector OCR engine (no AI) living in
//! `~/repos/BOOOCR`. We spawn its stdio server (`scripts/booocr_server.py`)
//! once per daemon run and exchange newline-delimited JSON over pipes:
//! line crops in, per-character candidates out. The server loads all font
//! databases at startup, so per-line latency is just the matcher.
//!
//! The client is not `Sync` (pipes); callers serialize access with a mutex.

use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

/// One recognized character with its alternatives (crop coordinates).
#[derive(Debug, Clone)]
pub struct BooChar {
    pub c: char,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub alts: Vec<(char, f32)>,
}

/// Recognition result for one line crop.
#[derive(Debug, Clone, Default)]
pub struct BooLine {
    pub chars: Vec<BooChar>,
}

/// Parse one `{c, x, y, w, h, alts}` object from a sidecar response.
fn parse_char(c: &serde_json::Value) -> Result<Option<BooChar>> {
    let c = c.as_object().context("bad char object")?;
    let ch = c.get("c").and_then(|x| x.as_str()).unwrap_or("");
    let mut ch_it = ch.chars();
    let Some(ch) = ch_it.next() else { return Ok(None) };
    if ch_it.next().is_some() {
        return Ok(None); // multi-char "c" — skip, shouldn't happen
    }
    let alts = c
        .get("alts")
        .and_then(|a| a.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|pair| {
                    let pair = pair.as_array()?;
                    let ch = pair.first()?.as_str()?;
                    let sc = pair.get(1)?.as_f64()?;
                    ch.chars().next().map(|c| (c, sc as f32))
                })
                .collect::<Vec<(char, f32)>>()
        })
        .unwrap_or_default();
    Ok(Some(BooChar {
        c: ch,
        x: c.get("x").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
        y: c.get("y").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
        w: c.get("w").and_then(|v| v.as_i64()).unwrap_or(1) as i32,
        h: c.get("h").and_then(|v| v.as_i64()).unwrap_or(1) as i32,
        alts,
    }))
}

/// Resolve the BOOOCR checkout dir: `BOOOCR_DIR` env var, else ~/repos/BOOOCR.
pub fn booocr_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BOOOCR_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home).join("repos").join("BOOOCR");
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("BOOOCR")
}

pub struct BooOcrClient {
    child: Option<Child>,
    stdin: BufWriter<Box<dyn Write + Send>>,
    stdout: BufReader<Box<dyn Read + Send>>,
    next_id: u64,
}

/// Default Unix-socket path for the watcher-owned shared pipeline.
pub fn socket_path() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("accessibility_daemon")
            .join("booocr.sock")
    } else {
        PathBuf::from("/tmp/booocr.sock")
    }
}

/// Locate a live shared pipeline socket: `BOOOCR_SOCKET` env var (set by
/// the watcher when spawning viewers), else the default state-dir path.
fn find_socket() -> Option<PathBuf> {
    let p = std::env::var("BOOOCR_SOCKET")
        .map(PathBuf::from)
        .unwrap_or_else(|_| socket_path());
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

impl BooOcrClient {
    /// Connect to the watcher-owned shared pipeline if one is alive,
    /// otherwise spawn our own stdio sidecar. The shared pipeline is
    /// fully warm (DBs mmap'd + norms computed), so per-image cost is a
    /// single socket connect instead of a sidecar spawn + DB load.
    pub fn connect_or_spawn(booocr_dir: &Path) -> Result<Self> {
        if let Some(sock) = find_socket() {
            match std::os::unix::net::UnixStream::connect(&sock) {
                Ok(stream) => {
                    let reader = stream.try_clone()?;
                    println!("[BOOOCR] connected to shared pipeline at {}", sock.display());
                    return Ok(BooOcrClient {
                        child: None,
                        stdin: BufWriter::new(Box::new(stream)),
                        stdout: BufReader::new(Box::new(reader)),
                        next_id: 1,
                    });
                }
                Err(e) => {
                    // Stale socket (watcher died without cleanup) — fall
                    // through to spawning our own sidecar.
                    eprintln!("[BOOOCR] shared pipeline at {} not connectable ({e}); spawning own sidecar", sock.display());
                }
            }
        }
        Self::spawn(booocr_dir)
    }

    /// Spawn the BOOOCR server and wait for its ready line (DB load done).
    pub fn spawn(booocr_dir: &Path) -> Result<Self> {
        let python = booocr_dir.join(".venv").join("bin").join("python");
        let server = booocr_dir.join("scripts").join("booocr_server.py");
        let db_dir = booocr_dir.join("data").join("databases");
        if !python.is_file() {
            anyhow::bail!("BOOOCR python missing at {}", python.display());
        }
        if !server.is_file() {
            anyhow::bail!("BOOOCR server missing at {}", server.display());
        }
        if !db_dir.is_dir() {
            anyhow::bail!("BOOOCR databases missing at {}", db_dir.display());
        }

        let mut child = Command::new(&python)
            .arg(&server)
            .arg("--db-dir")
            .arg(&db_dir)
            .arg("--top-k")
            .arg("15")
            // A stray PYTHONPATH (e.g. another venv) breaks numpy inside the
            // BOOOCR venv — never inherit it.
            .env_remove("PYTHONPATH")
            // Matcher is memory-bandwidth-bound; 4 OpenBLAS threads measured
            // optimal on the 3600X (8-16 slower, uncapped ~3x slower). The
            // sidecar's own batch thread pool uses the same 4.
            .env("OMP_NUM_THREADS", "4")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn BOOOCR sidecar")?;

        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let mut client = BooOcrClient {
            child: Some(child),
            stdin: BufWriter::new(Box::new(stdin)),
            stdout: BufReader::new(Box::new(stdout)),
            next_id: 1,
        };

        let mut line = String::new();
        match client.stdout.read_line(&mut line) {
            Ok(0) => anyhow::bail!("BOOOCR sidecar exited before ready"),
            Ok(_) => {
                if !line.contains("\"ready\"") {
                    anyhow::bail!("unexpected BOOOCR startup line: {line}");
                }
            }
            Err(e) => anyhow::bail!("BOOOCR sidecar read error: {e}"),
        }
        Ok(client)
    }

    /// Recognize many line crops in one batched request. The sidecar
    /// threads the per-line phases and matches every line of a scale class
    /// in a single matrix-matrix product (bit-identical scores). Returns
    /// one BooLine per input path, in order.
    pub fn recognize_crops(&mut self, png_paths: &[PathBuf]) -> Result<Vec<BooLine>> {
        let id = self.next_id;
        self.next_id += 1;
        let mut req = format!("{{\"id\": {id}, \"images\": [");
        for (i, p) in png_paths.iter().enumerate() {
            if i > 0 {
                req.push(',');
            }
            req.push('"');
            req.push_str(&p.display().to_string());
            req.push('"');
        }
        req.push_str("]}\n");
        self.stdin.write_all(req.as_bytes())?;
        self.stdin.flush()?;

        let mut line = String::new();
        self.stdout.read_line(&mut line)?;
        let v: serde_json::Value =
            serde_json::from_str(&line).context("bad JSON from BOOOCR sidecar")?;
        if v.get("id").and_then(|i| i.as_u64()) != Some(id) {
            anyhow::bail!("BOOOCR sidecar response id mismatch");
        }
        if v.get("ok").and_then(|o| o.as_bool()) != Some(true) {
            anyhow::bail!(
                "BOOOCR error: {}",
                v.get("error").and_then(|e| e.as_str()).unwrap_or("unknown")
            );
        }

        let mut out = Vec::with_capacity(png_paths.len());
        if let Some(lines) = v.get("lines").and_then(|l| l.as_array()) {
            for lv in lines {
                let mut chars = Vec::new();
                if let Some(arr) = lv.get("chars").and_then(|c| c.as_array()) {
                    for c in arr {
                        if let Some(ch) = parse_char(c)? {
                            chars.push(ch);
                        }
                    }
                }
                out.push(BooLine { chars });
            }
        }
        // Tolerate a short response (server skipped unreadable lines).
        while out.len() < png_paths.len() {
            out.push(BooLine::default());
        }
        Ok(out)
    }
}

impl Drop for BooOcrClient {
    fn drop(&mut self) {
        // Only kill a sidecar we spawned ourselves. A client connected to
        // the watcher-owned shared pipeline just drops the socket — the
        // watcher owns that server's lifecycle.
        if self.child.is_some() {
            let _ = writeln!(self.stdin, "{{\"id\": -1, \"shutdown\": true}}");
            let _ = self.stdin.flush();
            let _ = self.child.take().unwrap().wait();
        }
    }
}

/// Spawn the BOOOCR server in shared-pipeline mode (Unix socket) for the
/// file watcher. Blocks until the pipeline reports READY; returns the
/// child the watcher must keep alive (and kill on stop) plus its stdout
/// reader — keeping the read end open prevents SIGPIPE/BrokenPipeError in
/// the server if it ever writes to stdout after READY.
pub fn spawn_sidecar_for_watcher(
    socket: &Path,
) -> Result<(Child, BufReader<std::process::ChildStdout>)> {
    let dir = booocr_dir();
    let python = dir.join(".venv").join("bin").join("python");
    let server = dir.join("scripts").join("booocr_server.py");
    let db_dir = dir.join("data").join("databases");
    let mut child = Command::new(&python)
        .arg(&server)
        .arg("--db-dir")
        .arg(&db_dir)
        .arg("--top-k")
        .arg("15")
        .arg("--socket")
        .arg(socket)
        .env_remove("PYTHONPATH")
        .env("OMP_NUM_THREADS", "4")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to spawn BOOOCR shared pipeline")?;

    let stdout = child.stdout.take().expect("piped stdout");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    match reader.read_line(&mut line) {
        Ok(0) => anyhow::bail!("BOOOCR shared pipeline exited before ready"),
        Ok(_) => {
            if !line.contains("\"ready\"") {
                anyhow::bail!("unexpected BOOOCR startup line: {line}");
            }
        }
        Err(e) => anyhow::bail!("BOOOCR shared pipeline read error: {e}"),
    }
    Ok((child, reader))
}

/// Counter for unique temp crop filenames.
static CROP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Save a line crop to a temp png and return its path (for the sidecar).
pub fn save_crop_png(
    crop: &image::DynamicImage,
    job_idx: usize,
) -> Result<PathBuf> {
    let seq = CROP_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "booocr_in_{}_{}_{}.png",
        std::process::id(),
        job_idx,
        seq
    ));
    crop.save(&path).with_context(|| {
        format!("failed to save BOOOCR crop {}", path.display())
    })?;
    Ok(path)
}
