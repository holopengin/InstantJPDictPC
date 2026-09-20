mod app_settings;
mod nav_graph;
mod capture;
#[cfg(test)]
mod conformance;
mod data;
mod frontend;
mod furigana;
mod kana_size;
mod models;
mod ocr_engine;
mod overlay_font;
mod overlay_state;
mod ppocr;
mod ppocr_ncnn;
mod settings_window;
mod util;
mod viewer;

/// Open an image file, including JPEG XL (`.jxl`) which the `image` crate
/// cannot decode natively — jxl-oxide's `ImageDecoder` integration handles
/// it, producing the same `DynamicImage` the rest of the pipeline uses.
/// No ICC transform is applied: screenshots are treated as opaque RGB for
/// OCR (matching how every other format enters the pipeline).
fn open_image(path: impl AsRef<std::path::Path>) -> Result<image::DynamicImage> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    if ext.as_deref() == Some("jxl") {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let decoder = jxl_oxide::integration::JxlDecoder::new(file)?;
        Ok(image::DynamicImage::from_decoder(decoder)?)
    } else {
        Ok(image::open(path)?)
    }
}
mod watcher;

use anyhow::{Context, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use crate::data::db::DictionaryDatabase;
use crate::frontend::FrontendWindow;
use crate::models::*;
use crate::util::deinflector::Deinflector;
use crate::viewer::OcrViewer;

/// Messages from the bootstrap thread to the Iced UI update function.
/// The bootstrap thread loads everything (image, dict, engine) so the
/// window can appear in << 100 ms.
enum BootstrapMsg {
    ImageReady(iced::widget::image::Handle, Vec<u8>, u32, u32),
    DictReady(Arc<DictionaryDatabase>),
    DeinflectReady(Arc<Deinflector>),
    /// Complete initial result set: `annotations[i]` is detection box `i`,
    /// detection-only (`line: None`) where recognition produced no text.
    Results(Vec<DetectedAnnotation>),
    /// Complete replacement detection+recognition result after a live retune.
    Retuned(Vec<DetectedAnnotation>),
}

use iced::window::settings::PlatformSpecific;

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
    // File watcher mode: `--watcher [daemon args...]` — blocks until stopped.
    if args.len() >= 2 && args[1] == "--watcher" {
        watcher::run(&data_dir, &args[2..]);
        return Ok(());
    }
    // One-time (idempotent) setup for capture mode: install the desktop file
    // KWin requires before it will hand screenshots to this binary.
    if args.len() >= 2 && args[1] == "--setup-capture" {
        let desktop = capture::setup_capture_authorization()?;
        println!(
            "[Capture] Registered {} as an authorized screenshot client.",
            desktop.display()
        );
        println!();
        println!("{}", capture::shortcut_instructions());
        return Ok(());
    }
    // Capture mode: screenshot the focused window, then OCR it. Meant to be
    // bound to a global shortcut; viewer options are forwarded.
    if args.len() >= 2 && args[1] == "--capture" {
        return run_capture(&args[2..], &data_dir);
    }
    if args.len() < 2 {
        println!("No image argument provided. Opening frontend window...");
        let db_path = data_dir.join("dictionary.sqlite");
        let db = Arc::new(DictionaryDatabase::open(&db_path)?);
        // #43: the bundled pitch dictionary installs on a worker thread; the
        // settings window opens (and stays responsive) while a first run
        // imports.
        crate::data::bundled::spawn_bundled_install(
            Arc::clone(&db),
            std::path::PathBuf::from(resolve_asset_dir()),
        );
        run_frontend(db, data_dir)?;
        return Ok(());
    }

    run_ocr_viewer(args, &data_dir)
}

fn run_capture(extra_args: &[String], data_dir: &std::path::Path) -> Result<()> {
    let t_start = std::time::Instant::now();
    let image = capture::capture_active_window()
        .context("failed to take a screenshot of the focused window")?;
    let path = capture::save_capture(&image, data_dir)?;
    println!(
        "[Capture] {}x{} → {} ({:.0} ms)",
        image.width(),
        image.height(),
        path.display(),
        t_start.elapsed().as_secs_f64() * 1000.0,
    );
    // Hand the capture to the normal OCR viewer. Appending the path last is
    // safe: the viewer's option parser accepts positionals anywhere.
    let mut args: Vec<String> = vec!["accessibility_daemon".to_string()];
    args.extend(extra_args.iter().cloned());
    args.push(path.to_string_lossy().into_owned());
    run_ocr_viewer(args, data_dir)
}

fn print_usage() {
    let name = std::env::args().next().unwrap_or_else(|| "accessibility_daemon".into());
    println!("InstantJPDict accessibility overlay — OCR and dictionary daemon");
    println!();
    println!("USAGE:");
    println!("  {name} [OPTIONS] <IMAGE_PATH>");
    println!("  {name} [OPTIONS] --headless <IMAGE_PATH|DIRECTORY> [MORE_IMAGES...]");
    println!("  {name}                     Opens the frontend window (no arguments)");
    println!("  {name} --watcher [ARGS]    Run the file watcher (blocks until stopped)");
    println!("  {name} --capture [OPTIONS] Screenshot the focused window and OCR it");
    println!("  {name} --setup-capture     Authorize KWin screenshots for this binary");
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
    println!("      --det-thresh <F>       Detection threshold 0.01-0.99 (default 0.25)");
    println!("      --det-thresh=<F>       (alternative syntax)");
    println!("      --det-unclip <F>       DB unclip ratio 0.0-5.0 (default 0.7)");
    println!("      --det-unclip=<F>       (alternative syntax)");
    println!();
    println!("CAPTURE MODE (for a global shortcut):");
    println!("  --capture takes the screenshot itself — the focused window, no");
    println!("  picker and no file in ~/Pictures/Screenshots — then opens the OCR");
    println!("  viewer on it. Because nothing lands in a watched folder, the file");
    println!("  watcher cannot open a second viewer for the same image.");
    println!();
    println!("  Run --setup-capture once first: KWin only hands screenshots to");
    println!("  binaries whose .desktop file declares");
    println!("  X-KDE-DBUS-Restricted-Interfaces=org.kde.KWin.ScreenShot2");
    println!("  (the same mechanism Spectacle uses).");
    println!();
    println!("FRONTEND WINDOW:");
    println!("  Launching with no arguments opens a window with two options:");
    println!("    - Launch Settings Window  (dictionary import & configuration)");
    println!("    - Launch File Watcher     (monitors ~/Pictures/Screenshots and new");
    println!("                              images in /tmp; replaced with");
    println!("                              \"Stop File Watcher\" while it is running)");
    println!("  It also carries the overlay font choice: a \"Serif font\" checkbox");
    println!("  (sans is the default) persisted to settings.json next to");
    println!("  dictionary.sqlite, applied to the overlay and dictionary panel on");
    println!("  the next launch.");
    println!();
    println!("ARGUMENTS:");
    println!("  <IMAGE_PATH>               Path to a screenshot image for OCR analysis");
    println!("                             (headless: a directory, or multiple paths,");
    println!("                             are also accepted; line crops + sidecar .txt");
    println!("                             files are written next to each source image)");
    println!();
    println!("EXAMPLES:");
    println!("  {name} screenshot.png");
    println!("  {name} --headless --font=~/myfont.ttf image.png");
    println!("  {name} --headless screenshots_dir/");
    println!("  {name} --headless shot1.png shot2.png shot3.png");
    println!("  {name} --watcher --vert");
    println!("  {name} --setup-capture       (once; then bind a shortcut to --capture)");
    println!("  {name} --capture --vert");
    println!("  {name}                     (opens the frontend window)");
    println!();
    println!("KEYBOARD SHORTCUTS (when GUI is shown):");
    println!("  D / Shift+J                Scroll dictionary down");
    println!("  F / Shift+K                Scroll dictionary up");
    println!("  Esc / Q                    Close viewer / back out of a selection");
    println!("  [ / ]                      Lower / raise DET_UNCLIP by 0.05 (live)");
    println!("  - / =                      Lower / raise DET_THRESH by 0.01 (live)");
    println!("  R                          Reset detection tuning to startup values");
    println!("  F1                         Show/hide the detection tuning HUD");
}

fn run_frontend(db: Arc<DictionaryDatabase>, data_dir: std::path::PathBuf) -> Result<()> {
    let app = iced::application(
        move || FrontendWindow::new(Arc::clone(&db), data_dir.clone()),
        FrontendWindow::update,
        FrontendWindow::view,
    )
    .subscription(FrontendWindow::subscription)
    .window(iced::window::Settings {
        size: iced::Size::new(560.0, 420.0),
        platform_specific: PlatformSpecific {
            application_id: String::from("accessibility_daemon"),
            ..Default::default()
        },
        ..Default::default()
    });
    app.run().context("Failed to run frontend window")?;
    Ok(())
}

/// Detection tunables supplied on the command line.
#[derive(Clone, Copy, Default)]
struct DetTuning {
    thresh: Option<f32>,
    unclip: Option<f32>,
}

/// A live retune request sent from the viewer to the OCR worker.
#[derive(Clone, Copy)]
struct TuneCmd {
    thresh: f32,
    unclip: f32,
}

/// `--det-thresh[=F]` / `--det-unclip[=F]`, both clamped to plausible ranges.
/// Unknown arguments are ignored here; the main parser handles the rest.
fn parse_det_tuning(args: &[String]) -> DetTuning {
    let mut tuning = DetTuning::default();
    let clamp_thresh = |v: f32| v.clamp(0.01, 0.99);
    let clamp_unclip = |v: f32| v.clamp(0.0, 5.0);
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if let Some(v) = arg.strip_prefix("--det-thresh=") {
            tuning.thresh = v.parse::<f32>().ok().map(clamp_thresh);
            i += 1;
        } else if arg == "--det-thresh" {
            if let Some(v) = args.get(i + 1).and_then(|s| s.parse::<f32>().ok()) {
                tuning.thresh = Some(clamp_thresh(v));
            }
            i += 2;
        } else if let Some(v) = arg.strip_prefix("--det-unclip=") {
            tuning.unclip = v.parse::<f32>().ok().map(clamp_unclip);
            i += 1;
        } else if arg == "--det-unclip" {
            if let Some(v) = args.get(i + 1).and_then(|s| s.parse::<f32>().ok()) {
                tuning.unclip = Some(clamp_unclip(v));
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    tuning
}

/// Viewer tuning keys, matched on the **modified** key so layouts that put
/// tuning characters behind Shift still work: on JIS (`jp106`) `=` is
/// Shift+`-` on the same physical key as `-`, and matching the unmodified key
/// made both decrease the threshold.
fn tune_key_message(modified_key: &iced::keyboard::Key) -> Option<Message> {
    use iced::keyboard::Key;

    match modified_key {
        Key::Character(c) if c.as_ref() == "[" => {
            Some(Message::TuneDet { param: DetParam::Unclip, delta: -0.05 })
        }
        Key::Character(c) if c.as_ref() == "]" => {
            Some(Message::TuneDet { param: DetParam::Unclip, delta: 0.05 })
        }
        Key::Character(c) if c.as_ref() == "-" || c.as_ref() == "_" => {
            Some(Message::TuneDet { param: DetParam::Threshold, delta: -0.01 })
        }
        Key::Character(c) if c.as_ref() == "=" || c.as_ref() == "+" => {
            Some(Message::TuneDet { param: DetParam::Threshold, delta: 0.01 })
        }
        Key::Character(c) if c.as_ref() == "r" || c.as_ref() == "R" => Some(Message::TuneReset),
        Key::Named(iced::keyboard::key::Named::F1) => Some(Message::ToggleDetHud),
        _ => None,
    }
}

/// Detection-only annotations for every box, indexed by the box's original
/// index. The recognition pass replaces the entries that produced text, so
/// boxes that recognise nothing still reach the viewer as placeholders.
fn detection_annotations(boxes: &[BoundingBox], rotated: &[RotatedBox]) -> Vec<DetectedAnnotation> {
    boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let quad = rotated.get(i).copied().filter(|r| r.is_rotated());
            DetectedAnnotation { bbox: b.clone(), quad, line: None }
        })
        .collect()
}

fn run_ocr_viewer(
    args: Vec<String>,
    data_dir: &std::path::Path,
) -> Result<()> {
    let t_start = std::time::Instant::now();
    let mut image_paths: Vec<String> = Vec::new();
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
        else { image_paths.push(args[i].clone()); i += 1; }
    }
    let _ = font_path;
    let tuning = parse_det_tuning(&args);

    // Persisted app settings, read once per run: every capture/watcher child
    // is a fresh process, so a change applies to the next run (mobile reads
    // its preferences per detection run).
    let app_settings = crate::app_settings::AppSettings::load(data_dir);

    // Headless batch mode: OCR a directory or a list of images, saving line
    // crops + sidecar text files next to each source image.
    if headless {
        return run_headless_batch(
            image_paths,
            recognition_mode,
            batch_size,
            tuning,
            app_settings.furigana_filter,
        );
    }

    // The face the overlay (and, below, the iced dictionary panel) paints in.
    // Persisted by the frontend's Sans/Serif checkbox; loaded once per run, so
    // a change applies to the next launch.
    let face = app_settings.overlay_font;

    // The character n-gram model behind the blank's candidate ranking (#44).
    // 14 MB, loaded once per process like the fonts; a missing asset only
    // narrows the blank's list, so it is never fatal.
    let char_lm = {
        let path = std::path::Path::new(&resolve_asset_dir()).join("lm/char_lm.bin");
        match crate::util::char_lm::CharLm::load(&path) {
            Some(lm) => {
                println!("[Bootstrap] CharLm loaded: {} entries", lm.entries());
                Some(std::sync::Arc::new(lm))
            }
            None => {
                eprintln!(
                    "[Bootstrap] CharLm unavailable at {}; blank lists keep discovery order",
                    path.display()
                );
                None
            }
        }
    };

    // The kanji component table behind a tapped character's extra candidates
    // (#44). Small text asset, loaded once per process like the LM; a missing
    // asset only narrows the list to the head ranking, so it is never fatal.
    let oov_candidates = {
        let path =
            std::path::Path::new(&resolve_asset_dir()).join("components/krad_components.txt");
        match crate::util::component_table::ComponentTable::load(&path) {
            Some(table) => {
                println!("[Bootstrap] ComponentTable loaded: {} entries", table.entry_count());
                Some(std::sync::Arc::new(crate::util::oov_candidates::OovCandidates::new(table)))
            }
            None => {
                eprintln!(
                    "[Bootstrap] ComponentTable unavailable at {}; tapped lists stay head-only",
                    path.display()
                );
                None
            }
        }
    };

    // The kanji variant forms offered last in a tapped character's list
    // (#44). Same load-once shape as the tables above.
    let kanji_variants = {
        let path =
            std::path::Path::new(&resolve_asset_dir()).join("variants/kanji_variants.txt");
        match crate::util::kanji_variants::KanjiVariantTable::load(&path) {
            Some(table) => {
                println!("[Bootstrap] KanjiVariants loaded: {} variants", table.entry_count());
                Some(std::sync::Arc::new(table))
            }
            None => {
                eprintln!(
                    "[Bootstrap] KanjiVariants unavailable at {}; no variant forms offered",
                    path.display()
                );
                None
            }
        }
    };

    let image_path = image_paths.first().cloned().context("No image path provided")?;

    // Bootstrap channel — one-shot events (image, dict, deinflector,
    // initial results, retunes)
    let (bootstrap_tx, bootstrap_rx) = std::sync::mpsc::channel::<BootstrapMsg>();
    // Live tuning channel — the viewer's tune keys send the latest
    // DET_THRESH / DET_UNCLIP values to the OCR worker.
    let (tune_tx, tune_rx) = std::sync::mpsc::channel::<TuneCmd>();

    let image_path2 = image_path.clone();
    let furigana_filter = app_settings.furigana_filter;
    std::thread::Builder::new()
        .name("bootstrap".into())
        .spawn(move || {
            // Phase 1: load the screenshot image and send it to the viewer
            let t_load = std::time::Instant::now();
            let image = match open_image(&image_path2) {
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
            // #43: install the bundled pitch dictionary if it is missing, on
            // its own worker thread, so the OCR pipeline is never delayed by a
            // first-run import. Subsequent starts skip after one query.
            let db = Arc::new(db);
            crate::data::bundled::spawn_bundled_install(
                Arc::clone(&db),
                std::path::PathBuf::from(resolve_asset_dir()),
            );
            if bootstrap_tx.send(BootstrapMsg::DictReady(db)).is_err() { return; }

            // Phase 3: load deinflector
            let deinf_path = resolve_asset_path("deinflect.json");
            let deinf = Deinflector::from_json_file(&deinf_path).unwrap_or_else(|e| {
                eprintln!("[Bootstrap] Warning: deinflector {deinf_path}: {e}");
                Deinflector::empty()
            });
            println!("[Bootstrap] Deinflector loaded: {} rules", deinf.rule_count());
            if bootstrap_tx.send(BootstrapMsg::DeinflectReady(Arc::new(deinf))).is_err() { return; }

            // Phase 4: OCR pipeline (engine creation → detection → recognition)
            {
                let t_engine = std::time::Instant::now();
                let mut engine = match ocr_engine::OcrEngine::new(&resolve_asset_dir(), recognition_mode, batch_size) {
                    Ok(e) => e,
                    Err(err) => { eprintln!("[Bootstrap] OCR engine error: {err}"); return; }
                };
                // CLI tuning flags apply to the very first detection; the
                // viewer's tune keys adjust the same overrides live.
                engine.det_thresh_override = tuning.thresh;
                engine.det_unclip_override = tuning.unclip;
                // #100: the furigana (ruby) rule switch, off by default.
                engine.det_furigana = furigana_filter;
                println!("[Bootstrap] OCR engine created ({} chars) in {:.0} ms",
                    engine.ppocr_vocab.len(), t_engine.elapsed().as_secs_f64() * 1000.0);

                // Warm up PP-OCR sessions

                // Detection
                let t_detect = std::time::Instant::now();
                let det = match engine.detect_lines(&image) {
                    Ok(d) => d,
                    Err(e) => { eprintln!("[OCR] Detection error: {e}"); return; }
                };
                let boxes = det.boxes;
                let rotated = det.rotated;
                let detect_ms = t_detect.elapsed().as_secs_f64() * 1000.0;
                println!("[OCR timing] Line detection:       {:>8.2} ms ({} boxes)", detect_ms, boxes.len());

                // Detection-only annotations for every box, indexed by the
                // box's original index. Recognition below replaces the
                // entries that produced text; the rest stay as placeholders.
                let mut results = detection_annotations(&boxes, &rotated);

                // Recognition
                let ppocr_vocab = engine.ppocr_vocab.clone();
                let rec_remap = engine.rec_remap.clone();
                let batch_sz = engine.batch_size;
                let rec_mode = engine.recognition_mode;
                let t_recognize = std::time::Instant::now();
                match ocr_engine::recognize_boxes_collect(
                    &image, &boxes, &rotated,
                    engine.ppocr_rec.clone(),
                    engine.kana_size.clone(),
                    &ppocr_vocab, &rec_remap, batch_sz, rec_mode,
                    std::path::Path::new("/tmp"),
                ) {
                    Ok(recognized) => {
                        for (idx, ann) in recognized {
                            if let Some(slot) = results.get_mut(idx) {
                                *slot = ann;
                            }
                        }
                    }
                    Err(e) => eprintln!("[OCR] Recognition error: {e}"),
                }
                let recognize_ms = t_recognize.elapsed().as_secs_f64() * 1000.0;
                println!("[OCR timing] Character recognition: {:>8.2} ms", recognize_ms);
                if bootstrap_tx.send(BootstrapMsg::Results(results)).is_err() { return; }

                // ── Live detection tuning ────────────────────────────────
                // Keep the engine + image alive so the viewer can sweep
                // DET_THRESH / DET_UNCLIP without restarting. Commands that
                // pile up while a sweep is running are coalesced to the latest.
                while let Ok(first) = tune_rx.recv() {
                    let mut cmd = first;
                    while let Ok(next) = tune_rx.try_recv() {
                        cmd = next;
                    }
                    engine.det_thresh_override = Some(cmd.thresh);
                    engine.det_unclip_override = Some(cmd.unclip);
                    let t_tune = std::time::Instant::now();
                    let det = match engine.detect_lines(&image) {
                        Ok(d) => d,
                        Err(e) => {
                            eprintln!("[Tune] detection error: {e}");
                            continue;
                        }
                    };
                    let boxes = det.boxes;
                    let rotated = det.rotated;
                    let mut anns = detection_annotations(&boxes, &rotated);
                    match ocr_engine::recognize_boxes_collect(
                        &image, &boxes, &rotated,
                        engine.ppocr_rec.clone(),
                        engine.kana_size.clone(),
                        &ppocr_vocab, &rec_remap, batch_sz, rec_mode,
                        std::path::Path::new("/tmp"),
                    ) {
                        Ok(recognized) => {
                            for (idx, ann) in recognized {
                                if idx < anns.len() {
                                    anns[idx] = ann;
                                }
                            }
                        }
                        Err(e) => eprintln!("[Tune] recognition error: {e}"),
                    }
                    println!(
                        "[Tune] DET_THRESH={:.2} DET_UNCLIP={:.2} → {} boxes in {:.0} ms",
                        cmd.thresh, cmd.unclip, anns.len(),
                        t_tune.elapsed().as_secs_f64() * 1000.0
                    );
                    if bootstrap_tx.send(BootstrapMsg::Retuned(anns)).is_err() {
                        break;
                    }
                }
            }
        })?;

    // Don't go further in headless mode — bootstrap thread handles everything
    if headless { println!("Headless: done (bootstrap running in background)"); return Ok(()); }

    // Gamepad input is not wired up: Steam Deck game mode prevents exclusive
    // evdev grabs. Keyboard controls (arrow keys, Enter, Esc, D/F) are always
    // available.

    let bootstrap_rx = Arc::new(std::sync::Mutex::new(Some(bootstrap_rx)));
    let rx_for_update = Arc::clone(&bootstrap_rx);

    // Detect native screen resolution for UI scaling.
    // Scale factor = screen_w / 1280 so that the logical viewport is always
    // 1280 wide regardless of physical resolution.
    //
    // Window is sized to the screenshot image dimensions so the image maps 1:1
    // to physical pixels on screen. This is especially important on Steam Deck
    // where the screenshot IS the external display's true resolution, and
    // screen-detection methods may pick up the wrong display.
    let (screen_w, screen_h) = match open_image(&image_path) {
        Ok(img) => {
            let w = img.width() as f32;
            let h = img.height() as f32;
            println!("[SCALE] window sized to image {w:.0}x{h:.0}");
            (w, h)
        }
        Err(e) => {
            eprintln!("[SCALE] failed to open image: {e}; falling back to 1280x800");
            (1280.0, 800.0)
        }
    };
    let ui_scale = screen_w / 1280.0;
    println!(
        "[SCALE] detected screen {screen_w:.0}x{screen_h:.0}, ui_scale={ui_scale:.3} — \
         logical viewport {:.0}x{:.0}",
        1280.0,
        screen_h / ui_scale,
    );

    // The UI keeps the sending end; the worker owns the receiver so it can
    // re-run detection when the viewer's tuning keys change the values.
    let tune_tx = std::sync::Mutex::new(tune_tx);

    // #43/#86: read once per run like the other Behaviour switches, so a
    // change applies to the next launch.
    let show_pitch = app_settings.pitch_accent;
    let boot = move || {
        let mut viewer = OcrViewer::new_empty(screen_w, screen_h, face);
        // #43/#86: the pitch line is opt-in (off by default, like mobile).
        viewer.show_pitch = show_pitch;
        viewer.state.install_char_lm(char_lm.clone());
        viewer.state.install_oov_candidates(oov_candidates.clone());
        viewer.state.install_kanji_variants(kanji_variants.clone());
        if let Some(t) = tuning.thresh {
            viewer.det_thresh = t;
            viewer.det_thresh_default = t;
        }
        if let Some(u) = tuning.unclip {
            viewer.det_unclip = u;
            viewer.det_unclip_default = u;
        }
        viewer
    };

    let update = move |state: &mut OcrViewer, msg: Message| -> iced::Task<Message> {
        // Track whether any channel drained anything — if so, schedule the
        // next frame immediately so we keep draining until they are empty.
        let mut had_ocr_work = false;
        // Drain bootstrap channel (image, dict, deinflector, results, retunes)
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
                        BootstrapMsg::Results(annotations) => {
                            state.apply_ocr_batch(annotations);
                            had_ocr_work = true;
                        }
                        BootstrapMsg::Retuned(annotations) => {
                            state.apply_retune(annotations);
                            state.det_busy = false;
                            had_ocr_work = true;
                        }
                    }
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
                        if let Some(cur_char) = cur {
                            eprintln!(
                                "[ALTERNATIVE] line {} char {}: U+{:04X} '{}' → U+{:04X} '{}'",
                                sel.line_idx, sel.char_idx,
                                cur_char as u32, cur_char,
                                c as u32, c,
                            );
                        } else {
                            eprintln!(
                                "[ALTERNATIVE] line {} char {}: (none) → U+{:04X} '{}'",
                                sel.line_idx, sel.char_idx,
                                c as u32, c,
                            );
                        }
                        state.state.update_character(sel.line_idx, sel.char_idx, c);
                        state.annotations_sync_dirty.set(true);
                        state.edited_lines.insert(sel.line_idx);
                        // Dataset collection: rewrite the crop's .txt sidecar
                        // with the corrected text so /tmp yields curated
                        // (crop, label) pairs for recognizer training.
                        if let Some(line) = state.state.active_line_results.get(sel.line_idx).and_then(|l| l.as_ref()) {
                            if let Some(txt) = line.sample_txt.as_ref() {
                                if let Err(e) = std::fs::write(txt, &line.text) {
                                    eprintln!("[dataset] failed to update {}: {e}", txt.display());
                                } else {
                                    eprintln!("[dataset] updated {}", txt.display());
                                }
                            }
                        }
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
            Message::TuneDet { param, delta } => {
                state.adjust_det_tuning(param, delta);
                state.det_busy = true;
                if let Ok(tx) = tune_tx.lock() {
                    let _ = tx.send(TuneCmd {
                        thresh: state.det_thresh,
                        unclip: state.det_unclip,
                    });
                }
            }
            Message::TuneReset => {
                state.reset_det_tuning();
                state.det_busy = true;
                if let Ok(tx) = tune_tx.lock() {
                    let _ = tx.send(TuneCmd {
                        thresh: state.det_thresh,
                        unclip: state.det_unclip,
                    });
                }
            }
            Message::ToggleDetHud => {
                state.det_hud_visible = !state.det_hud_visible;
            }
            Message::Tick => {
                // Nav graph rebuild is deferred to navigate() — no need
                // to rebuild here every frame during streaming.
            }
        }
        if matches!(msg, Message::ZoomOnCursor { .. } | Message::PanDelta { .. } | Message::PinchZoom { .. }) {
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
        if tasks.is_empty() {
            if had_ocr_work {
                iced::Task::perform(async {}, |_: ()| Message::Tick)
            } else {
                iced::Task::none()
            }
        } else {
            if had_ocr_work {
                tasks.push(iced::Task::perform(async {}, |_: ()| Message::Tick));
            }
            iced::Task::batch(tasks)
        }
    };

    // The bytes registered with iced and the family name must agree with the
    // face actually found: a serif selection whose file is missing falls back
    // to the sans file (with a warning), so ask for the sans family — never
    // for a family nothing registered.
    let font_path = crate::overlay_font::find_font_path(face);
    let loaded_face = font_path
        .as_deref()
        .map(crate::overlay_font::face_of)
        .unwrap_or(face);
    let font_bytes = font_path
        .as_ref()
        .and_then(|path| std::fs::read(path).ok())
        .unwrap_or_default();
    // The dictionary panel's flow layout packs lines against the real face;
    // give it the same bytes iced renders with.
    crate::viewer::init_panel_metrics(&font_bytes);

    let app = iced::application(boot, update, OcrViewer::view)
        // One selected face for the overlay glyph cache AND the iced text
        // (dictionary panel etc.): mixing fonts shows different stroke forms
        // (e.g. JP vs traditional-CN variants) for the same char. Mobile
        // splits the two — its panel goes back to the platform face because
        // the bundled Noto line box is 1.448 em there — but that has no
        // desktop counterpart: PC has no guaranteed system JP face to
        // return to (the bundled files *are* the fallback chain), and the
        // panel layout here was tuned against the same face as the overlay.
        // So the Sans/Serif setting applies to both; only the highlight
        // weight may differ (serif ships no bold, so it fake-bolds).
        .font(font_bytes)
        .default_font(iced::Font::with_name(loaded_face.family_name()))
        .window(iced::window::Settings {
            // iced treats this as LOGICAL pixels and multiplies by the app
            // scale factor (ui_scale = screen_w / 1280) when creating the
            // window. Passing the image size directly shrinks small images
            // quadratically (155x129 -> a 19x16 window). The logical
            // viewport is always 1280 wide, so physical size comes out
            // exactly screen_w x screen_h = the image at 1:1.
            size: iced::Size::new(1280.0, screen_h / ui_scale),
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
                .filter_map(|m| m.or(Some(Message::Tick)));

            let keyboard_events = iced_futures::subscription::filter_map(
                "keyboard",
                |event: iced_futures::subscription::Event| {
                    match &event {
                        iced_futures::subscription::Event::Interaction {
                            event: iced::Event::Keyboard(iced::keyboard::Event::KeyPressed { key, modified_key, .. }),
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
                                // Tuning keys match the MODIFIED key (see
                                // `tune_key_message`): on JIS layouts `=` is
                                // Shift+`-`, so the unmodified key `-` would
                                // otherwise collide with decrease.
                                _ => return tune_key_message(modified_key),
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

/// Headless batch OCR: process a directory or a list of images, saving each
/// detected line crop + sidecar text file into the same folder as the source
/// image. The OCR engine is loaded once and reused across all images.
fn run_headless_batch(
    image_paths: Vec<String>,
    recognition_mode: RecognitionMode,
    batch_size: usize,
    tuning: DetTuning,
    furigana_filter: bool,
) -> Result<()> {
    use std::path::{Path, PathBuf};

    fn is_image_file(p: &Path) -> bool {
        matches!(
            p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
            Some("png") | Some("jpg") | Some("jpeg") | Some("bmp") | Some("webp")
        )
    }
    // Skip previously-saved dataset crops when expanding a directory, so
    // re-running the batch doesn't re-OCR its own output.
    fn is_ocr_line_sample(p: &Path) -> bool {
        p.file_name().and_then(|n| n.to_str()).map_or(false, |n| n.starts_with("ocr_line_"))
    }

    // Expand directories (sorted) into a flat list of image files.
    let mut files: Vec<PathBuf> = Vec::new();
    for p in &image_paths {
        let path = Path::new(p);
        if path.is_dir() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(path)?
                .flatten()
                .map(|e| e.path())
                .filter(|f| is_image_file(f) && !is_ocr_line_sample(f))
                .collect();
            entries.sort();
            println!("[Batch] Directory {}: {} image(s)", p, entries.len());
            files.extend(entries);
        } else if path.is_file() {
            files.push(path.to_path_buf());
        } else {
            eprintln!("[Batch] Skipping (not a file or directory): {p}");
        }
    }
    if files.is_empty() {
        anyhow::bail!("No images found to process");
    }
    println!(
        "[Batch] Processing {} image(s) with {:?} recognition",
        files.len(), recognition_mode
    );

    let t_engine = std::time::Instant::now();
    let mut engine = ocr_engine::OcrEngine::new(&resolve_asset_dir(), recognition_mode, batch_size)?;
    engine.det_thresh_override = tuning.thresh;
    engine.det_unclip_override = tuning.unclip;
    engine.det_furigana = furigana_filter;
    println!("[Batch] OCR engine created in {:.0} ms", t_engine.elapsed().as_secs_f64() * 1000.0);

    let ppocr_vocab = engine.ppocr_vocab.clone();
    let rec_remap = engine.rec_remap.clone();

    for file in &files {
        let image = match open_image(file) {
            Ok(img) => img,
            Err(e) => { eprintln!("[Batch] Failed to open {}: {e}", file.display()); continue; }
        };
        let t_img = std::time::Instant::now();
        let t_det = std::time::Instant::now();
        let det = match engine.detect_lines(&image) {
            Ok(d) => d,
            Err(e) => { eprintln!("[Batch] Detection failed for {}: {e}", file.display()); continue; }
        };
        println!("[OCR timing] Line detection:       {:>8.2} ms ({} boxes)", t_det.elapsed().as_secs_f64() * 1000.0, det.boxes.len());
        let boxes = det.boxes;
        let rotated = det.rotated;
        // Save crops next to the source image.
        let out_dir = file.parent().filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("/tmp"));
        // Keep the receiver alive so worker sends succeed for every line.
        let (tx, rx) = std::sync::mpsc::channel::<(usize, DetectedAnnotation)>();
        if let Err(e) = ocr_engine::recognize_boxes_streaming(
            &image, &boxes, &rotated,
            engine.ppocr_rec.clone(),
            engine.kana_size.clone(),
            &ppocr_vocab, &rec_remap, batch_size, recognition_mode,
            tx, out_dir,
        ) {
            eprintln!("[Batch] Recognition error for {}: {e}", file.display());
        }
        // The channel buffers every finished line; drain it so results are
        // not dropped, and print one machine-readable record per box when
        // `PPOCR_DUMP_LINES` is set (the tuning sweep parses these).
        let dump = std::env::var("PPOCR_DUMP_LINES").is_ok();
        while let Ok((idx, ann)) = rx.recv() {
            if !dump {
                continue;
            }
            let (text, conf, vert) = match ann.line.as_ref() {
                Some(l) => {
                    // `alternatives` stores raw CTC logits; a softmax over the
                    // top-K gives a comparable per-char confidence in (0, 1].
                    let mut sum = 0.0f32;
                    let mut n = 0usize;
                    for alts in &l.alternatives {
                        if alts.is_empty() {
                            continue;
                        }
                        let m = alts.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
                        let z: f32 = alts.iter().map(|(_, s)| (s - m).exp()).sum();
                        sum += if z > 0.0 { 1.0 / z } else { 0.0 };
                        n += 1;
                    }
                    let conf = if n == 0 { 0.0 } else { sum / n as f32 };
                    (l.text.as_str(), conf, l.is_vertical)
                }
                None => ("", 0.0, false),
            };
            println!(
                "[LINE] i={idx} x={} y={} w={} h={} vert={} conf={conf:.4} text={text}",
                ann.bbox.x, ann.bbox.y, ann.bbox.w, ann.bbox.h, u8::from(vert),
            );
        }
        println!(
            "[Batch] {}: {} line(s) in {:.0} ms",
            file.display(), boxes.len(), t_img.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        std::iter::once("accessibility_daemon".to_string())
            .chain(v.iter().map(|s| s.to_string()))
            .collect()
    }

    #[test]
    fn det_tuning_parses_both_syntaxes() {
        let t = parse_det_tuning(&args(&[
            "--det-thresh=0.42",
            "--det-unclip",
            "1.25",
            "img.png",
        ]));
        assert_eq!(t.thresh, Some(0.42));
        assert_eq!(t.unclip, Some(1.25));
    }

    #[test]
    fn det_tuning_clamps_out_of_range_values() {
        let t = parse_det_tuning(&args(&["--det-thresh", "9.0", "--det-unclip=99"]));
        assert_eq!(t.thresh, Some(0.99));
        assert_eq!(t.unclip, Some(5.0));
        let t = parse_det_tuning(&args(&["--det-thresh=-1"]));
        assert_eq!(t.thresh, Some(0.01));
    }

    #[test]
    fn det_tuning_ignores_missing_and_unparsable_values() {
        let t = parse_det_tuning(&args(&["--det-thresh", "--vert", "img.png"]));
        assert_eq!(t.thresh, None);
        assert!(t.unclip.is_none());
        let t = parse_det_tuning(&args(&["--det-unclip=abc"]));
        assert_eq!(t.unclip, None);
    }

    /// Detection placeholders keep the box's index, carry a quad only when
    /// the fitted rect is meaningfully rotated, and survive a short rotated
    /// list.
    #[test]
    fn detection_annotations_keep_indices_and_quads() {
        let boxes = vec![
            BoundingBox::new(0, 0, 10, 10, 0.9),
            BoundingBox::new(20, 20, 30, 10, 0.8),
        ];
        let rotated = vec![
            RotatedBox::new(5.0, 5.0, 10.0, 10.0, 0.0, 0.9),
            RotatedBox::new(35.0, 25.0, 30.0, 10.0, 0.3, 0.8),
        ];
        let anns = detection_annotations(&boxes, &rotated);
        assert_eq!(anns.len(), 2);
        assert_eq!(
            (anns[0].bbox.x, anns[0].bbox.y, anns[0].bbox.w, anns[0].bbox.h),
            (0, 0, 10, 10)
        );
        assert!(anns[0].quad.is_none(), "axis-aligned rect has no quad");
        assert!(anns[0].line.is_none());
        assert!(anns[1].quad.is_some(), "tilted rect keeps its quad");

        let anns = detection_annotations(&boxes, &rotated[..1]);
        assert_eq!(anns.len(), 2, "every box gets a placeholder");
        assert!(anns[1].quad.is_none());
    }

    /// Regression for the JIS layout: `=` is Shift+`-`, so the modified key
    /// must map to increase while the unmodified `-` decreases.
    #[test]
    fn tune_keys_use_the_modified_character() {
        use iced::keyboard::Key;

        assert!(matches!(
            tune_key_message(&Key::Character("=".into())),
            Some(Message::TuneDet { param: DetParam::Threshold, delta }) if delta > 0.0
        ));
        assert!(matches!(
            tune_key_message(&Key::Character("+".into())),
            Some(Message::TuneDet { param: DetParam::Threshold, delta }) if delta > 0.0
        ));
        assert!(matches!(
            tune_key_message(&Key::Character("-".into())),
            Some(Message::TuneDet { param: DetParam::Threshold, delta }) if delta < 0.0
        ));
        assert!(matches!(
            tune_key_message(&Key::Character("_".into())),
            Some(Message::TuneDet { param: DetParam::Threshold, delta }) if delta < 0.0
        ));
        assert!(matches!(
            tune_key_message(&Key::Character("[".into())),
            Some(Message::TuneDet { param: DetParam::Unclip, delta }) if delta < 0.0
        ));
        assert!(matches!(
            tune_key_message(&Key::Character("]".into())),
            Some(Message::TuneDet { param: DetParam::Unclip, delta }) if delta > 0.0
        ));
        assert!(matches!(
            tune_key_message(&Key::Character("R".into())),
            Some(Message::TuneReset)
        ));
        assert!(matches!(
            tune_key_message(&Key::Named(iced::keyboard::key::Named::F1)),
            Some(Message::ToggleDetHud)
        ));
        assert!(tune_key_message(&Key::Character("x".into())).is_none());
    }
}

