mod nav_graph;
mod data;
mod models;
#[cfg(feature = "ort")]
mod ocr_engine;
#[cfg(feature = "ort")]
mod ocr_parallel;
mod overlay_state;
mod settings_window;
mod util;
mod viewer;

use anyhow::{Context, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

use crate::data::db::DictionaryDatabase;
use crate::models::*;
use crate::settings_window::SettingsWindow;
use crate::util::deinflector::Deinflector;
use crate::viewer::OcrViewer;

use directories;
/// Messages from the bootstrap thread to the Iced UI update function.
/// The bootstrap thread loads everything (image, dict, engine) so the
/// window can appear in << 100 ms.
enum BootstrapMsg {
    ImageReady(iced::widget::image::Handle, Vec<u8>, u32, u32),
    DictReady(Arc<DictionaryDatabase>),
    DeinflectReady(Arc<Deinflector>),
    AnnotationReady(DetectedAnnotation),
}

use iced::window::settings::PlatformSpecific;
use iced_futures::futures;

// ── evdev statics ──────────────────────────────────────────────────────
static GP_BITS: AtomicU32 = AtomicU32::new(0);
static GP_LAST: AtomicU32 = AtomicU32::new(0);
static GP_COUNT: AtomicU32 = AtomicU32::new(0);

// Repeat state (set by Navigate handler, checked by ZoomTick)
static GP_REPEAT_ACTION: std::sync::Mutex<Option<(GamepadAction, Instant)>> =
    std::sync::Mutex::new(None);
/// Keyboard-held navigation action (set on KeyPressed, cleared on KeyReleased).
static KB_ACTION_HELD: std::sync::Mutex<Option<(GamepadAction, Instant)>> =
    std::sync::Mutex::new(None);
/// Number of repeat ticks already fired (used by ZoomTick to avoid over-firing).
static GP_LAST_REPEAT: AtomicU64 = AtomicU64::new(0);
/// Repeat delay before auto-repeat kicks in (ms).
const REPEAT_DELAY_MS: u64 = 250;
/// Interval between repeat ticks (ms) — 20 repeats/s.
const REPEAT_INTERVAL_MS: u64 = 40;

const B_UP: u32 = 1 << 0;
const B_DOWN: u32 = 1 << 1;
const B_LEFT: u32 = 1 << 2;
const B_RIGHT: u32 = 1 << 3;
const B_A: u32 = 1 << 4;
const B_B: u32 = 1 << 5;
const B_L1: u32 = 1 << 6;
const B_R1: u32 = 1 << 7;
const B_L2: u32 = 1 << 8;
const B_R2: u32 = 1 << 9;

/// Resolve a single asset file path, checking binary-relative and AppImage paths.
fn resolve_asset_path(filename: &str) -> String {
    let dir = resolve_asset_dir();
    format!("{}/{}", dir, filename)
}

/// Resolve the assets directory, checking binary-relative and AppImage paths.
fn resolve_asset_dir() -> String {
    // Binary-relative: next to the executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let p = parent.join("assets");
            if p.exists() {
                return p.to_string_lossy().to_string();
            }
        }
    }
    // AppImage mount: $APPDIR/usr/bin/assets/
    if let Ok(appdir) = std::env::var("APPDIR") {
        let p = std::path::Path::new(&appdir)
            .join("usr")
            .join("bin")
            .join("assets");
        if p.exists() {
            return p.to_string_lossy().to_string();
        }
    }
    // Fallback: relative to CWD
    "./assets".to_string()
}

fn start_evdev_thread() {
    std::thread::spawn(|| {
        use evdev::{Device, EventType, AbsoluteAxisCode, KeyCode};
        let mut devices: Vec<Device> = Vec::new();
        if let Ok(entries) = std::fs::read_dir("/dev/input") {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.to_string_lossy().contains("event") { continue; }
                if let Ok(mut d) = Device::open(&path) {
                    let has_btn = d.supported_keys().map_or(false, |caps| {
                        caps.contains(KeyCode::BTN_SOUTH) || caps.contains(KeyCode::new(0x130))
                    });
                    if has_btn {
                        let name = d.name().unwrap_or("?").to_string();
                        println!("[GP] evdev gamepad found: {name} at {p}", p = path.display());
                        // Grab the device so events are captured exclusively and
                        // don't also reach the game running underneath.
                        if let Err(e) = d.grab() {
                            println!("[GP] grab {name} failed (events will pass through): {e}");
                        } else {
                            println!("[GP] grabbed {name} — events blocked from other apps");
                        }
                        devices.push(d);
                    }
                }
            }
        }
        GP_COUNT.store(devices.len() as u32, Ordering::Relaxed);
        println!("[GP] Found {} gamepad devices", devices.len());
        if devices.is_empty() { return; }

        loop {
            std::thread::sleep(std::time::Duration::from_millis(4));
            for dev in &mut devices {
                for ev in dev.fetch_events().into_iter().flatten() {
                    let etype = ev.event_type();
                    let code = ev.code();
                    let val = ev.value();
                    // Determine which button this event is for (regardless of val)
                    let bit = if etype == EventType::ABSOLUTE {
                        0 // handled below
                    } else if etype == EventType::KEY {
                        let kc = KeyCode(code);
                             if kc == KeyCode::BTN_DPAD_UP    || kc == KeyCode::new(0x220) { B_UP }
                        else if kc == KeyCode::BTN_DPAD_DOWN  || kc == KeyCode::new(0x221) { B_DOWN }
                        else if kc == KeyCode::BTN_DPAD_LEFT  || kc == KeyCode::new(0x222) { B_LEFT }
                        else if kc == KeyCode::BTN_DPAD_RIGHT || kc == KeyCode::new(0x223) { B_RIGHT }
                        else if kc == KeyCode::BTN_SOUTH      || kc == KeyCode::new(0x130) { B_A }
                        else if kc == KeyCode::BTN_EAST       || kc == KeyCode::new(0x131) { B_B }
                        else if kc == KeyCode::BTN_TL         || kc == KeyCode::new(0x136) { B_L1 }
                        else if kc == KeyCode::BTN_TR         || kc == KeyCode::new(0x137) { B_R1 }
                        else if kc == KeyCode::BTN_TL2        || kc == KeyCode::new(0x138) { B_L2 }
                        else if kc == KeyCode::BTN_TR2        || kc == KeyCode::new(0x139) { B_R2 }
                        else { 0 }
                    } else { 0 };
                    // HAT absolute axes: modify GP_BITS directly for both directions
                    let hat_handled = if etype == EventType::ABSOLUTE {
                        if code == AbsoluteAxisCode::ABS_HAT0X.0 {
                            Some(if val == -1 { B_LEFT } else if val == 1 { B_RIGHT } else { B_LEFT | B_RIGHT })
                        } else if code == AbsoluteAxisCode::ABS_HAT0Y.0 {
                            Some(if val == -1 { B_UP } else if val == 1 { B_DOWN } else { B_UP | B_DOWN })
                        } else { None }
                    } else { None };
                    if let Some(hat_bits) = hat_handled {
                        if val == -1 {
                            let set = if code == AbsoluteAxisCode::ABS_HAT0X.0 { B_LEFT } else { B_UP };
                            let clear = if code == AbsoluteAxisCode::ABS_HAT0X.0 { B_RIGHT } else { B_DOWN };
                            GP_BITS.fetch_or(set, Ordering::Relaxed);
                            GP_BITS.fetch_and(!clear, Ordering::Relaxed);
                        } else if val == 1 {
                            let set = if code == AbsoluteAxisCode::ABS_HAT0X.0 { B_RIGHT } else { B_DOWN };
                            let clear = if code == AbsoluteAxisCode::ABS_HAT0X.0 { B_LEFT } else { B_UP };
                            GP_BITS.fetch_or(set, Ordering::Relaxed);
                            GP_BITS.fetch_and(!clear, Ordering::Relaxed);
                        } else {
                            GP_BITS.fetch_and(!hat_bits, Ordering::Relaxed); // center: clear both
                        }
                        eprintln!("[GP] evdev HAT: code=0x{code:04x} val={} bits={hat_bits:08b}", val);
                        continue;
                    }
                    if bit == 0 { continue; }

                    eprintln!("[GP] evdev raw: type={} code=0x{code:04x} val={} bit={}",
                        etype.0, val, bit);

                    if val != 0 {
                        GP_BITS.fetch_or(bit, Ordering::Relaxed);
                    } else {
                        GP_BITS.fetch_and(!bit, Ordering::Relaxed);
                    }
                }
            }
        }
    });
}

fn set_val(val: i32, neg_bit: u32, pos_bit: u32) -> u32 {
    match val { -1 => neg_bit, 1 => pos_bit, _ => 0 }
}
fn press_flag(val: i32, bit: u32) -> u32 { if val != 0 { bit } else { 0 } }

fn gp_bits_to_msg(bits: u32) -> Option<Message> {
    match bits {
        p if p & B_UP != 0 => Some(Message::Navigate(GamepadAction::NavigateUp)),
        p if p & B_DOWN != 0 => Some(Message::Navigate(GamepadAction::NavigateDown)),
        p if p & B_LEFT != 0 => Some(Message::Navigate(GamepadAction::NavigateLeft)),
        p if p & B_RIGHT != 0 => Some(Message::Navigate(GamepadAction::NavigateRight)),
        p if p & B_A != 0 => Some(Message::Navigate(GamepadAction::Confirm)),
        p if p & B_B != 0 => Some(Message::Navigate(GamepadAction::Back)),
        p if p & B_L1 != 0 => Some(Message::Navigate(GamepadAction::ScrollUp)),
        p if p & B_R1 != 0 => Some(Message::Navigate(GamepadAction::ScrollDown)),
        p if p & B_L2 != 0 => Some(Message::SetScale { scale: 0.9 }),   // zoom out
        p if p & B_R2 != 0 => Some(Message::SetScale { scale: 1.1 }),   // zoom in
        _ => None,
    }
}

/// Navigate the alternatives selection when the alt panel is visible.
/// In portrait: LEFT/RIGHT cycle through alts. In landscape: UP/DOWN.
fn navigate_alternatives(state: &mut crate::viewer::OcrViewer, dir: GamepadAction) {
    let is_landscape = state.window_width > state.window_height;
    let (line_idx, char_idx) = match state.state.current_cursor() {
        Some(c) => c,
        None => return,
    };
    let line = match state.state.active_line_results.get(line_idx).and_then(|l| l.as_ref()) {
        Some(l) => l,
        None => return,
    };
    let alts = match line.alternatives.get(char_idx) {
        Some(a) => a,
        None => return,
    };
    let current_char = match line.text.chars().nth(char_idx) {
        Some(c) => c,
        None => return,
    };

    let diff = match (dir, is_landscape) {
        (GamepadAction::NavigateDown, true) | (GamepadAction::NavigateRight, false) => 1,
        (GamepadAction::NavigateUp, true) | (GamepadAction::NavigateLeft, false) => -1,
        _ => 0,
    };
    if diff == 0 { return; }

    let candidates: Vec<char> = alts.iter().take(15).map(|(c, _)| *c).collect();
    let current_idx = candidates.iter().position(|c| *c == current_char).unwrap_or(0);
    let new_idx = (current_idx as isize + diff).clamp(0, candidates.len() as isize - 1) as usize;

    if new_idx != current_idx {
        let new_char = candidates[new_idx];
        state.state.update_character(line_idx, char_idx, new_char);
        let _ = state.db.as_ref().and_then(|db| state.deinflector.as_ref().and_then(|deinf| {
            state.state.lookup(line_idx, char_idx, db, deinf)
        }));
        state.state.update_highlight_coords(line_idx, char_idx, state.state.current_word_length);
    }
}

/// Scroll the dictionary panel by N lines (or delta px).
fn handle_dict_scroll(state: &mut crate::viewer::OcrViewer, direction: f32) {
    state.dict_scroll_request = Some(direction * 120.0);
}

fn main() -> Result<()> {
    env_logger::init();
    println!("Accessibility Daemon Starting...");

    let data_dir = directories::ProjectDirs::from("com", "Example", "accessibility_daemon")
        .or(directories::ProjectDirs::from("org", "Example", "accessibility_daemon"))
        .or(directories::ProjectDirs::from("net", "Example", "accessibility_daemon"))
        .map(|dirs| dirs.data_dir().to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    std::fs::create_dir_all(&data_dir)?;
    let _ = data_dir; // data directory created, but dict loaded lazily in bootstrap thread

    let args: Vec<String> = std::env::args().collect();
    // Check for help before any parsing
    if args.len() >= 2 && (args[1] == "--help" || args[1] == "-help" || args[1] == "/?") {
        print_usage();
        return Ok(());
    }
    if args.len() < 2 {
        println!("No image argument provided. Opening settings window...");
        let db_path = data_dir.join("dictionary.sqlite");
        let db = Arc::new(DictionaryDatabase::open(&db_path)?);
        run_settings_window(db)?;
        return Ok(());
    }

    run_ocr_viewer(args)
}

fn print_usage() {
    let name = std::env::args().next().unwrap_or_else(|| "accessibility_daemon".into());
    println!("InstantJPDict accessibility overlay — OCR and dictionary daemon");
    println!();
    println!("USAGE:");
    println!("  {name} [OPTIONS] <IMAGE_PATH>");
    println!("  {name}                     Opens settings window (no arguments)");
    println!("  {name} --help              Show this help message");
    println!();
    println!("OPTIONS:");
    println!("  -h, --headless             Run in headless mode (no GUI window)");
    println!("  -f, --font <PATH>          Path to a custom font file for overlay text");
    println!("      --font=<PATH>          (alternative syntax)");
    println!("  -v, --vert                 Vertical-only recognition (skip horizontal boxes)");
    println!("      --hybrid               Hybrid mode: both horizontal and vertical recognition");
    println!("  -b, --batch-size <N>       Recognition batch size (default: 10)");
    println!("      --batch-size=<N>       (alternative syntax)");
    println!();
    println!("ARGUMENTS:");
    println!("  <IMAGE_PATH>               Path to a screenshot image for OCR analysis");
    println!();
    println!("EXAMPLES:");
    println!("  {name} screenshot.png");
    println!("  {name} --headless --font=~/myfont.ttf image.png");
    println!("  {name}                     (opens interactive settings dialog)");
    println!();
    println!("KEYBOARD SHORTCUTS (when GUI is shown):");
    println!("  D / Shift+J                Scroll dictionary down");
    println!("  F / Shift+K                Scroll dictionary up");
}

fn run_settings_window(db: Arc<DictionaryDatabase>) -> Result<()> {
    let app = iced::application(
        move || SettingsWindow::new(Arc::clone(&db)),
        SettingsWindow::update,
        SettingsWindow::view,
    ).window(iced::window::Settings {
        platform_specific: PlatformSpecific {
            application_id: String::from("accessibility_daemon"),
            ..Default::default()
        },
        ..Default::default()
    });
    app.run().context("Failed to run settings window")?;
    Ok(())
}

/// Try to detect the primary monitor's native resolution.
/// Methods tried in order:
///   1. Linux DRM sysfs (/sys/class/drm/*/modes) — works on any modern Linux
///   2. xrandr (X11)
///   3. wlr‑randr (Wayland wlroots)
///
/// Returns dimensions in landscape orientation: if the detected height exceeds
/// the width (e.g. Steam Deck's 800×1280 native portrait panel), they are
/// swapped so the caller always gets (landscape_w, landscape_h).
fn get_screen_size() -> Option<(f32, f32)> {
    let result = _get_screen_size_inner();
    // Normalise to landscape: if height > width, swap.
    result.map(|(w, h)| if h > w { (h, w) } else { (w, h) })
}

/// Inner implementation — orientation‑agnostic resolution detection.
fn _get_screen_size_inner() -> Option<(f32, f32)> {
    // ── 1. Linux DRM sysfs — no external tool needed ──
    {
        let drm_dir = std::path::Path::new("/sys/class/drm");
        if let Ok(entries) = std::fs::read_dir(drm_dir) {
            for entry in entries.flatten() {
                let status_path = entry.path().join("status");
                let ok = std::fs::read_to_string(&status_path).ok()
                    .map(|s| s.trim() == "connected")
                    .unwrap_or(false);
                if !ok { continue; }
                let modes_path = entry.path().join("modes");
                if let Ok(modes) = std::fs::read_to_string(&modes_path) {
                    for line in modes.lines() {
                        let trimmed = line.trim();
                        if trimmed.is_empty() { continue; }
                        if let Some(x_idx) = trimmed.find('x') {
                            if let Ok(w) = trimmed[..x_idx].parse::<f32>() {
                                let after_x = &trimmed[x_idx + 1..];
                                let h_end = after_x.find(|c: char| !c.is_ascii_digit()).unwrap_or(after_x.len());
                                if let Ok(h) = after_x[..h_end].parse::<f32>() {
                                    return Some((w, h));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── 2. xrandr (X11) ──
    if let Ok(out) = std::process::Command::new("xrandr")
        .arg("--current")
        .output()
    {
        let text = String::from_utf8_lossy(&out.stdout);
        // e.g. "Screen 0: minimum 320 x 200, current 1920 x 1080, maximum …"
        for line in text.lines() {
            if let Some(rest) = line.find("current ") {
                let after = &line[rest + 8..];
                if let Some(x_idx) = after.find('x') {
                    let w_str = after[..x_idx].trim();
                    if let Ok(w) = w_str.parse::<f32>() {
                        let after_x = after[x_idx + 1..].trim();
                        // Find the next non-digit character (space or comma)
                        let h_end = after_x.find(|c: char| !c.is_ascii_digit()).unwrap_or(after_x.len());
                        if let Ok(h) = after_x[..h_end].parse::<f32>() {
                            return Some((w, h));
                        }
                    }
                }
            }
        }
    }

    // ── wlr‑randr (Wayland wlroots) ──
    if let Ok(out) = std::process::Command::new("wlr-randr")
        .output()
    {
        let text = String::from_utf8_lossy(&out.stdout);
        // Look for "Mode: 1920x1080 @" in the first connected output
        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(mode) = trimmed.strip_prefix("Mode: ") {
                if let Some(x_idx) = mode.find('x') {
                    if let Ok(w) = mode[..x_idx].parse::<f32>() {
                        let after_x = &mode[x_idx + 1..];
                        let h_end = after_x.find(|c: char| c == '@' || c == ' ').unwrap_or(after_x.len());
                        if let Ok(h) = after_x[..h_end].parse::<f32>() {
                            return Some((w, h));
                        }
                    }
                }
            }
        }
    }

    None
}

fn run_ocr_viewer(
    args: Vec<String>,
) -> Result<()> {
    let t_start = std::time::Instant::now();
    let mut image_path = None;
    let mut font_path: Option<String> = None;
    let mut headless = false;
    let mut recognition_mode = RecognitionMode::Horizontal;
    let mut batch_size: usize = 10;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--headless" || args[i] == "-h" { headless = true; i += 1; }
        else if args[i].starts_with("--font=") { font_path = Some(args[i].trim_start_matches("--font=").to_string()); i += 1; }
        else if args[i] == "--font" || args[i] == "-f" {
            if i + 1 < args.len() { font_path = Some(args[i + 1].clone()); i += 2; } else { i += 1; }
        } else if args[i] == "--vert" || args[i] == "-v" {
            recognition_mode = RecognitionMode::Vertical; i += 1;
        } else if args[i] == "--hybrid" {
            recognition_mode = RecognitionMode::Both; i += 1;
        } else if args[i] == "-b" || args[i] == "--batch-size" {
            if i + 1 < args.len() {
                batch_size = args[i + 1].parse().unwrap_or(10);
                i += 2;
            } else { i += 1; }
        } else if args[i].starts_with("--batch-size=") {
            batch_size = args[i].trim_start_matches("--batch-size=").parse().unwrap_or(10);
            i += 1;
        } else if args[i].starts_with("-") { i += 1; }
        else { image_path = Some(args[i].clone()); break; }
    }
    let image_path = image_path.context("No image path provided")?;

    // Bootstrap channel — one-shot events (image, dict, deinflector)
    let (bootstrap_tx, bootstrap_rx) = std::sync::mpsc::channel::<BootstrapMsg>();
    // OCR channel — streamed detection boxes and recognition results
    // Tuple is (target_index, DetectedAnnotation). Detection boxes use
    // sequential indices; recognition results use the same index as the
    // detection box they correspond to.
    let (ocr_tx, ocr_rx) = std::sync::mpsc::channel::<(usize, DetectedAnnotation)>();

    let ocr_tx2 = ocr_tx.clone();
    let image_path2 = image_path.clone();
    std::thread::Builder::new()
        .name("bootstrap".into())
        .spawn(move || {
            // Phase 1: load the screenshot image and send it to the viewer
            let t_load = std::time::Instant::now();
            let image = match image::open(&image_path2) {
                Ok(img) => img,
                Err(e) => { eprintln!("[Bootstrap] Failed to open image: {e}"); return; }
            };
            let (w, h) = (image.width(), image.height());
            println!("[Bootstrap] Image loaded: {} ({}x{}) in {:.0} ms",
                image_path2, w, h, t_load.elapsed().as_secs_f64() * 1000.0);

            let display_img = image.to_rgba8();
            let dynimg = image::DynamicImage::ImageRgba8(display_img);
            let mut buf = std::io::Cursor::new(Vec::new());
            if dynimg.write_to(&mut buf, image::ImageFormat::Png).is_err() {
                eprintln!("[Bootstrap] PNG encode failed"); return;
            }
            let png_bytes = buf.into_inner();
            let handle = iced::widget::image::Handle::from_bytes(png_bytes.clone());
            if bootstrap_tx.send(BootstrapMsg::ImageReady(handle, png_bytes, w, h)).is_err() { return; }

            // Phase 2: load dictionary
            let db_path = directories::ProjectDirs::from("com", "Example", "accessibility_daemon")
                .or(directories::ProjectDirs::from("org", "Example", "accessibility_daemon"))
                .or(directories::ProjectDirs::from("net", "Example", "accessibility_daemon"))
                .map(|dirs| dirs.data_dir().to_path_buf())
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("dictionary.sqlite");
            let db = match DictionaryDatabase::open(&db_path) {
                Ok(d) => { println!("[Bootstrap] Dictionary database loaded: {} entries", d.get_entry_count().unwrap_or(0)); d }
                Err(e) => { eprintln!("[Bootstrap] Failed to load dictionary: {e}"); return; }
            };
            if bootstrap_tx.send(BootstrapMsg::DictReady(Arc::new(db))).is_err() { return; }

            // Phase 3: load deinflector
            let deinf_path = resolve_asset_path("deinflect.json");
            let deinf = Deinflector::from_json_file(&deinf_path).unwrap_or_else(|e| {
                eprintln!("[Bootstrap] Warning: deinflector {deinf_path}: {e}");
                Deinflector::empty()
            });
            println!("[Bootstrap] Deinflector loaded: {} rules", deinf.rule_count());
            if bootstrap_tx.send(BootstrapMsg::DeinflectReady(Arc::new(deinf))).is_err() { return; }

            // Phase 4: OCR pipeline (engine creation → detection → recognition)
            #[cfg(feature = "ort")]
            {
                let t_engine = std::time::Instant::now();
                let mut engine = match ocr_engine::OcrEngine::new(&resolve_asset_dir(), recognition_mode, batch_size) {
                    Ok(e) => e,
                    Err(err) => { eprintln!("[Bootstrap] OCR engine error: {err}"); return; }
                };
                println!("[Bootstrap] OCR engine created ({} chars) in {:.0} ms",
                    engine.char_vocab.len(), t_engine.elapsed().as_secs_f64() * 1000.0);

                // Init recognition sessions
                engine.recognize_sessions.get();
                if recognition_mode != RecognitionMode::Horizontal {
                    engine.recognize_sessions_vertical.get();
                }

                // Detection
                let t_detect = std::time::Instant::now();
                let boxes = match engine.detect_lines(&image) {
                    Ok(b) => b,
                    Err(e) => { eprintln!("[OCR] Detection error: {e}"); return; }
                };
                let detect_ms = t_detect.elapsed().as_secs_f64() * 1000.0;
                println!("[OCR timing] Line detection:       {:>8.2} ms ({} boxes)", detect_ms, boxes.len());

                // Send boxes with their original indices
                for (i, b) in boxes.iter().enumerate() {
                    if ocr_tx2.send((
                        i,
                        DetectedAnnotation { bbox: b.clone(), line: None }
                    )).is_err() { return; }
                }

                // Recognition
                let char_vocab = engine.char_vocab.clone();
                let batch_sz = engine.batch_size;
                let rec_mode = engine.recognition_mode;
                let t_recognize = std::time::Instant::now();
                let vert_sessions = if rec_mode != RecognitionMode::Horizontal {
                    engine.recognize_sessions_vertical.get()
                } else {
                    &[]
                };
                if let Err(e) = ocr_engine::recognize_boxes_streaming(
                    &image, &boxes,
                    engine.recognize_sessions.get(),
                    vert_sessions,
                    &char_vocab, batch_sz, rec_mode,
                    ocr_tx2,
                ) {
                    eprintln!("[OCR] Recognition error: {e}");
                }
                let recognize_ms = t_recognize.elapsed().as_secs_f64() * 1000.0;
                println!("[OCR timing] Character recognition: {:>8.2} ms", recognize_ms);
            }
            #[cfg(not(feature = "ort"))]
            { eprintln!("ORT not compiled. Rebuild with --features ort"); }
        })?;

    // Don't go further in headless mode — bootstrap thread handles everything
    if headless { println!("Headless: done (bootstrap running in background)"); return Ok(()); }

    // Gamepad input disabled — Steam Deck game mode prevents exclusive evdev grab.
    // Keyboard controls (arrow keys, Enter, Esc, D/F) are always available.
    // start_evdev_thread();

    let bootstrap_rx = Arc::new(std::sync::Mutex::new(Some(bootstrap_rx)));
    let rx_for_update = Arc::clone(&bootstrap_rx);
    let ocr_rx = Arc::new(std::sync::Mutex::new(Some(ocr_rx)));
    let ocr_rx_update = Arc::clone(&ocr_rx);

    // Detect native screen resolution for UI scaling.
    // Scale factor = screen_w / 1280 so that the logical viewport is always
    // 1280 wide regardless of physical resolution.
    let (screen_w, screen_h) = get_screen_size().unwrap_or((1280.0, 800.0));
    let ui_scale = screen_w / 1280.0;
    println!(
        "[SCALE] detected screen {screen_w:.0}x{screen_h:.0}, ui_scale={ui_scale:.3} — \
         logical viewport {:.0}x{:.0}",
        1280.0,
        screen_h / ui_scale,
    );

    let boot = move || OcrViewer::new_empty(screen_w, screen_h);

    let update = move |state: &mut OcrViewer, msg: Message| -> iced::Task<Message> {
        // Drain bootstrap channel (image, dict, deinflector)
        if let Ok(mut guard) = rx_for_update.lock() {
            if let Some(rx) = guard.as_mut() {
                while let Ok(event) = rx.try_recv() {
                    match event {
                        BootstrapMsg::ImageReady(handle, bytes, w, h) => {
                            state.set_image(handle, bytes, w, h);
                        }
                        BootstrapMsg::DictReady(db) => {
                            state.db = Some(db);
                        }
                        BootstrapMsg::DeinflectReady(d) => {
                            state.deinflector = Some(d);
                        }
                        BootstrapMsg::AnnotationReady(ann) => { /* deprecated */ }
                    }
                }
            }
        }
        // Drain OCR channel (detection boxes + recognition results)
        if let Ok(mut guard) = ocr_rx_update.lock() {
            if let Some(rx) = guard.as_mut() {
                while let Ok((idx, ann)) = rx.try_recv() {
                    let idx = idx;
                    state.handle_ocr_recognition_result(idx, ann);
                }
            }
        }
        match msg {
            Message::SelectCharacter(li, ci) => state.select_character(li, ci),
            Message::SelectNeighbor(li, ci) => state.select_neighbor(li, ci),
            Message::SelectAlternative(c) => {
                if let Some(sel) = state.selected_word.as_ref() {
                    let cur = state.state.active_line_results.get(sel.line_idx)
                        .and_then(|l| l.as_ref()).and_then(|l| l.text.chars().nth(sel.char_idx));
                    if cur == Some(c) { state.alternatives_visible = false; }
                    else {
                        state.state.update_character(sel.line_idx, sel.char_idx, c);
                        let _ = state.db.as_ref().and_then(|db| state.deinflector.as_ref().and_then(|deinf| {
            state.state.lookup(sel.line_idx, sel.char_idx, db, deinf)
        }));
                    }
                }
            }
            Message::Navigate(a) => match a {
                GamepadAction::Confirm => {
                    if state.alternatives_visible {
                        // Alt open → close it
                        state.alternatives_visible = false;
                    } else if state.selected_word.is_some() {
                        // Dict open, no alt → toggle alt open
                        state.alternatives_visible = true;
                    } else if let Some((li, ci)) = state.state.current_cursor() {
                        // Nothing open → open dictionary
                        state.select_character(li, ci);
                    }
                }
                GamepadAction::Back => {
                    if state.alternatives_visible { state.alternatives_visible = false; }
                    else if state.selected_word.is_some() { state.selected_word = None; state.state.is_dictionary_visible = false; }
                    else { std::process::exit(0); }
                    let _ = GP_REPEAT_ACTION.lock().unwrap().take();
                }
                dir @ (GamepadAction::NavigateUp | GamepadAction::NavigateDown
                | GamepadAction::NavigateLeft | GamepadAction::NavigateRight) => {
                    if state.alternatives_visible {
                        navigate_alternatives(state, dir);
                    } else {
                        let cursor_before = state.state.current_cursor();
                        let has_graph = state.state.nav_graph.is_some();
                        let result = state.state.navigate(dir);
                        eprintln!("[NAV] navigate({dir:?}) → {result}, cursor was {cursor_before:?}, graph={has_graph}");
                        // First press: do lookup immediately.
                        // Repeats (handled in ZoomTick) will set defer_lookup = true
                        // so lookups stop. On key-up the deferred final lookup fires.
                        state.defer_lookup = false;
                        if let Some((li, ci)) = state.state.current_cursor() {
                            state.move_cursor_to(li, ci);
                            if state.state.is_dictionary_visible {
                                state.do_lookup(li, ci);
                                state.compute_scroll_targets(li, ci);
                            }
                        } else {
                            eprintln!("[NAV] no cursor after navigate");
                        }
                    }
                    *GP_REPEAT_ACTION.lock().unwrap() = Some((a, Instant::now()));
                }
                GamepadAction::ScrollUp => {
                    handle_dict_scroll(state, -1.0);
                    *GP_REPEAT_ACTION.lock().unwrap() = Some((a, Instant::now()));
                }
                GamepadAction::ScrollDown => {
                    handle_dict_scroll(state, 1.0);
                    *GP_REPEAT_ACTION.lock().unwrap() = Some((a, Instant::now()));
                }
                _ => {}
            },
            Message::Back => {
                if state.alternatives_visible { state.alternatives_visible = false; }
                else if state.selected_word.is_some() { state.selected_word = None; state.state.is_dictionary_visible = false; }
                else { std::process::exit(0); }
                let _ = GP_REPEAT_ACTION.lock().unwrap().take();
            }
            Message::ZoomOnCursor { delta, cursor_x, cursor_y } => {
                let factor = if delta > 0.0 { 1.1 } else { 0.9 };
                let old = state.state.current_scale;
                let new = (old * factor).clamp(0.5, 5.0);
                let actual = if old > 0.0 { new / old } else { 1.0 };
                state.state.current_scale = new;
                state.state.current_trans_x = cursor_x - (cursor_x - state.state.current_trans_x) * actual;
                state.state.current_trans_y = cursor_y - (cursor_y - state.state.current_trans_y) * actual;
                state.is_zooming = true; state.zoom_idle_frames = 0;
            }
            Message::PanDelta { dx, dy } => { state.state.current_trans_x += dx; state.state.current_trans_y += dy; state.zoom_idle_frames = 0; }
            Message::PanStart { .. } => { state.is_zooming = true; state.zoom_idle_frames = 0; }
            Message::PanEnd => { state.is_zooming = false; state.zoom_idle_frames = 0; }
            Message::SetScale { scale } => { state.state.current_scale = scale.clamp(0.5, 5.0); }
            Message::PinchZoom { scale_factor, focus_x, focus_y, prev_focus_x, prev_focus_y, base_offset_y } => {
                let old = state.state.current_scale;
                let new = (old * scale_factor).clamp(0.5, 5.0);
                state.state.current_scale = new;
                let actual = if old > 0.0 { new / old } else { 1.0 };
                state.state.current_trans_x = focus_x - (prev_focus_x - state.state.current_trans_x) * actual;
                state.state.current_trans_y = (focus_y - (prev_focus_y - state.state.current_trans_y - base_offset_y) * actual) - base_offset_y;
                state.is_zooming = true; state.zoom_idle_frames = 0;
            }
            Message::PinchEnd => { state.is_zooming = false; state.zoom_idle_frames = 0; }
            Message::WindowResized { width, height } => {
                state.window_width = width as f32;
                state.window_height = height as f32;
                state.state.window_width.set(width as f32);
                state.state.window_height.set(height as f32);
                // Iced converts the winit physical-size event to logical pixels
                // by dividing by total_scale (native_display * app_scale_factor).
                // We reverse this to recover the physical window width:
                //   logical = physical / (native * app_scale)
                //   physical = logical * native * app_scale
                //   new_app_scale = physical / 1280.0
                // On the first resize we deduce native_scale from the known
                // initial physical screen width.
                let app_scale = state.debounced_ui_scale;
                if state.native_scale == 0.0 && app_scale > 0.0 {
                    state.native_scale = (state.screen_physical_width
                        / (width as f32 * app_scale))
                        .clamp(0.5, 4.0);
                }
                if state.native_scale > 0.0 && app_scale > 0.0 {
                    let physical_w = width as f32 * state.native_scale * app_scale;
                    state.debounced_ui_scale = (physical_w / 1280.0).max(0.5);
                }
            }
            Message::ZoomTick => {
                if state.zoom_idle_frames > 3 { state.is_zooming = false; }
                let gp_held = GP_BITS.load(Ordering::Relaxed) & (B_UP|B_DOWN|B_LEFT|B_RIGHT|B_L1|B_R1);
                let kb_held = KB_ACTION_HELD.lock().unwrap().is_some();
                if gp_held != 0 || kb_held {
                    let ra = if kb_held {
                        KB_ACTION_HELD.lock().unwrap().clone()
                    } else {
                        GP_REPEAT_ACTION.lock().unwrap().clone()
                    };
                    if let Some((action, started)) = ra {
                        let elapsed = started.elapsed();
                        if elapsed >= std::time::Duration::from_millis(REPEAT_DELAY_MS) {
                            let total = ((elapsed.as_millis() - REPEAT_DELAY_MS as u128) / REPEAT_INTERVAL_MS as u128) as u64;
                            let last = GP_LAST_REPEAT.load(Ordering::Relaxed);
                            if total > last {
                                GP_LAST_REPEAT.store(total, Ordering::Relaxed);
                                // Defer lookups during repeat — the final position
                                // will be looked up on key release.
                                state.defer_lookup = true;
                                match action {
                                    GamepadAction::NavigateUp | GamepadAction::NavigateDown
                                    | GamepadAction::NavigateLeft | GamepadAction::NavigateRight => {
                                        if state.alternatives_visible {
                                            navigate_alternatives(state, action);
                                        } else {
                                            state.state.navigate(action);
                                            if let Some((li, ci)) = state.state.current_cursor() {
                                                state.move_cursor_to(li, ci);
                                            }
                                        }
                                    }
                                    GamepadAction::ScrollUp => handle_dict_scroll(state, -1.0),
                                    GamepadAction::ScrollDown => handle_dict_scroll(state, 1.0),
                                    _ => {}
                                }
                            }
                        } else {
                            GP_LAST_REPEAT.store(0, Ordering::Relaxed);
                        }
                    }
                } else {
                    let _ = GP_REPEAT_ACTION.lock().unwrap().take();
                    GP_LAST_REPEAT.store(0, Ordering::Relaxed);
                    // If dict is open and a lookup was deferred, fire it now
                    if state.defer_lookup && state.selected_word.is_some() {
                        state.defer_lookup = false;
                        if let Some((li, ci)) = state.state.current_cursor() {
                            state.do_lookup(li, ci);
                            state.compute_scroll_targets(li, ci);
                        }
                    }
                }
            }
            // OCR streaming messages — receiver drained in update body above
            Message::OcrDetectionComplete(_) | Message::OcrRecognitionResult(_, _) | Message::OcrAllDone => {}
        }
        if matches!(msg, Message::ZoomOnCursor { .. } | Message::PanDelta { .. } | Message::PanStart { .. } | Message::PinchZoom { .. }) {
            state.zoom_idle_frames = 0;
        } else { state.zoom_idle_frames = state.zoom_idle_frames.saturating_add(1); }
        let mut tasks = Vec::new();
        if matches!(msg, Message::SelectCharacter(_, _) | Message::SelectNeighbor(_, _) | Message::SelectAlternative(_)) {
            tasks.push(state.scroll_dict_to_top_task());
        }
        if let Some(t) = state.scroll_neighbor_task() { tasks.push(t); state.scroll_neighbor_to = None; }
        if let Some(t) = state.scroll_alt_task() { tasks.push(t); state.scroll_alt_to = None; }
        if let Some(delta) = state.dict_scroll_request.take() {
            tasks.push(state.scroll_dict_by_delta(delta));
        }
        if tasks.is_empty() { iced::Task::none() } else { iced::Task::batch(tasks) }
    };

    let app = iced::application(boot, update, OcrViewer::view)
        .window(iced::window::Settings {
            size: iced::Size::new(screen_w, screen_h),
            ..Default::default()
        })
        .antialiasing(false)
        .scale_factor(|state: &OcrViewer| state.debounced_ui_scale)
        .subscription(|_state: &OcrViewer| {
            let gp_events = iced::time::every(iced::time::Duration::from_millis(16))
                .map(|_| {
                    let bits = GP_BITS.load(Ordering::Relaxed);
                    let prev = GP_LAST.swap(bits, Ordering::Relaxed);
                    let changed = bits ^ prev;
                    let pressed = changed & bits;
                    let ngp = GP_COUNT.load(Ordering::Relaxed);

                    let held: Vec<&str> = {
                        let mut v = Vec::new();
                        if bits & B_UP != 0 { v.push("UP"); }
                        if bits & B_DOWN != 0 { v.push("DN"); }
                        if bits & B_LEFT != 0 { v.push("LT"); }
                        if bits & B_RIGHT != 0 { v.push("RT"); }
                        if bits & B_A != 0 { v.push("A"); }
                        if bits & B_B != 0 { v.push("B"); }
                        if bits & B_L1 != 0 { v.push("L1"); }
                        if bits & B_R1 != 0 { v.push("R1"); }
                        v
                    };
                    if pressed != 0 || bits != 0 {
                        let held_str = if held.is_empty() { String::new() } else { format!(" held=[{}]", held.join(" ")) };
                        eprintln!("[GP] poll: bits={bits:08b} pressed={pressed:08b}{held_str} gamepads={ngp}");
                    }
                    gp_bits_to_msg(pressed)
                })
                .filter_map(|m| m);

            let keyboard_events = iced_futures::subscription::filter_map(
                "keyboard",
                |event: iced_futures::subscription::Event| {
                    match &event {
                        iced_futures::subscription::Event::Interaction {
                            event: iced::Event::Keyboard(iced::keyboard::Event::KeyPressed { key, .. }),
                            ..
                        } => {
                            // If we already have a held keyboard action, this is an
                            // OS-level key repeat — ignore it. Only our ZoomTick
                            // repeat (300ms/20Hz) handles repeat navigation.
                            if KB_ACTION_HELD.lock().unwrap().is_some() {
                                return None;
                            }
                            let nav_action = match key {
                                k if *k == iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape)
                                    || *k == iced::keyboard::Key::Character("q".into()) =>
                                    return Some(Message::Back),
                                k if *k == iced::keyboard::Key::Named(iced::keyboard::key::Named::Enter) =>
                                    return Some(Message::Navigate(GamepadAction::Confirm)),
                                k if *k == iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowRight)
                                    || *k == iced::keyboard::Key::Character("l".into()) => GamepadAction::NavigateRight,
                                k if *k == iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowLeft)
                                    || *k == iced::keyboard::Key::Character("h".into()) => GamepadAction::NavigateLeft,
                                k if *k == iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowDown)
                                    || *k == iced::keyboard::Key::Character("j".into()) => GamepadAction::NavigateDown,
                                k if *k == iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowUp)
                                    || *k == iced::keyboard::Key::Character("k".into()) => GamepadAction::NavigateUp,
                                k if *k == iced::keyboard::Key::Character("d".into()) => GamepadAction::ScrollDown,
                                k if *k == iced::keyboard::Key::Character("f".into()) => GamepadAction::ScrollUp,
                                _ => return None,
                            };
                            // Set keyboard repeat tracking so ZoomTick can fire repeats.
                            *KB_ACTION_HELD.lock().unwrap() = Some((nav_action, Instant::now()));
                            return Some(Message::Navigate(nav_action));
                        }
                        iced_futures::subscription::Event::Interaction {
                            event: iced::Event::Keyboard(iced::keyboard::Event::KeyReleased { key, .. }),
                            ..
                        } => {
                            // Clear keyboard repeat on any navigation-key release.
                            let is_nav = matches!(key,
                                iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowUp)
                                | iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowDown)
                                | iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowLeft)
                                | iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowRight)
                                | iced::keyboard::Key::Character(_)
                            );
                            if is_nav {
                                *KB_ACTION_HELD.lock().unwrap() = None;
                                GP_LAST_REPEAT.store(0, Ordering::Relaxed);
                            }
                            None
                        }
                        iced_futures::subscription::Event::Interaction {
                            event: iced::Event::Mouse(iced::mouse::Event::ButtonPressed(iced::mouse::Button::Right)),
                            ..
                        } => Some(Message::Back),
                        _ => None,
                    }
                },
            );

            let zoom_timer = iced::time::every(iced::time::Duration::from_millis(30)).map(|_| Message::ZoomTick);
            let resize_events = iced::window::resize_events()
                .map(|(_id, size)| Message::WindowResized { width: size.width, height: size.height });

            iced::Subscription::batch(vec![
                gp_events.into(),
                keyboard_events,
                zoom_timer,
                resize_events,
            ])
        });

    let to_spawn_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    println!("[DEBUG] To window spawn:         {:>8.2} ms", to_spawn_ms);

    if let Err(e) = app.run() { println!("GUI failed: {e:?}"); }
    Ok(())
}
