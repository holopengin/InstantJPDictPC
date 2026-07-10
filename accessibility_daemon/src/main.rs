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
use image::DynamicImage;
use std::io::Cursor;
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
use iced::window::settings::PlatformSpecific;
use iced_futures::futures;

// ── evdev statics ──────────────────────────────────────────────────────
static GP_BITS: AtomicU32 = AtomicU32::new(0);
static GP_LAST: AtomicU32 = AtomicU32::new(0);
static GP_COUNT: AtomicU32 = AtomicU32::new(0);

// Repeat state (set by Navigate handler, checked by ZoomTick)
static GP_REPEAT_ACTION: std::sync::Mutex<Option<(GamepadAction, Instant)>> =
    std::sync::Mutex::new(None);
/// Number of repeat ticks already fired (used by ZoomTick to avoid over-firing).
static GP_LAST_REPEAT: AtomicU64 = AtomicU64::new(0);

const B_UP: u32 = 1 << 0;
const B_DOWN: u32 = 1 << 1;
const B_LEFT: u32 = 1 << 2;
const B_RIGHT: u32 = 1 << 3;
const B_A: u32 = 1 << 4;
const B_B: u32 = 1 << 5;
const B_L1: u32 = 1 << 6;
const B_R1: u32 = 1 << 7;

fn start_evdev_thread() {
    std::thread::spawn(|| {
        use evdev::{Device, EventType, AbsoluteAxisCode, KeyCode};
        let mut devices: Vec<Device> = Vec::new();
        if let Ok(entries) = std::fs::read_dir("/dev/input") {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.to_string_lossy().contains("event") { continue; }
                if let Ok(d) = Device::open(&path) {
                    let has_btn = d.supported_keys().map_or(false, |caps| {
                        caps.contains(KeyCode::BTN_SOUTH) || caps.contains(KeyCode::new(0x130))
                    });
                    if has_btn {
                        let name = d.name().unwrap_or("?").to_string();
                        println!("[GP] evdev gamepad found: {name} at {p}", p = path.display());
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
        let _ = state.state.lookup(line_idx, char_idx, &state.db, &state.deinflector);
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
    let db_path = data_dir.join("dictionary.sqlite");
    let db = Arc::new(DictionaryDatabase::open(&db_path)?);
    let entry_count = db.get_entry_count()?;
    println!("Dictionary database loaded: {} entries", entry_count);

    let deinflector = Arc::new(
        Deinflector::from_json_file("assets/deinflect.json").unwrap_or_else(|e| {
            println!("Warning: Could not load deinflection rules: {}", e);
            Deinflector::empty()
        }),
    );
    println!("Deinflector loaded: {} rules", deinflector.rule_count());

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("No image argument provided. Opening settings window...");
        run_settings_window(db)?;
        return Ok(());
    }

    // Check for help before any parsing
    if args.iter().any(|a| a == "--help" || a == "-help" || a == "/?") {
        print_usage();
        return Ok(());
    }

    run_ocr_viewer(args, db, deinflector)
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

fn run_ocr_viewer(
    args: Vec<String>,
    db: Arc<DictionaryDatabase>,
    deinflector: Arc<Deinflector>,
) -> Result<()> {
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
    let image = image::open(&image_path).context(format!("Failed to open image: {image_path}"))?;
    println!("Image loaded: {image_path} ({}x{})", image.width(), image.height());

    let (annotations, _annotated_opt) = {
        #[cfg(feature = "ort")]
        {
            println!("Using ORT (ONNX Runtime) backend.");
            let mut engine = ocr_engine::OcrEngine::new("./assets", recognition_mode, batch_size)?;
            println!("Models loaded. {} chars", engine.char_vocab.len());
            engine.run_detection(&image, true, font_path.as_deref())?
        }
        #[cfg(not(feature = "ort"))]
        { eprintln!("ORT not compiled. Rebuild with --features ort"); std::process::exit(1); }
    };

    let display_img = image.to_rgba8();
    let dynimg = DynamicImage::ImageRgba8(display_img.clone());
    let mut buf = Cursor::new(Vec::new());
    dynimg.write_to(&mut buf, image::ImageFormat::Png).context("Failed to encode PNG")?;
    let bytes_arc = Arc::new(buf.into_inner());
    let (w, h) = (display_img.width(), display_img.height());
    if headless { println!("Headless: done"); return Ok(()); }

    start_evdev_thread();

    let boot = {
        let db = Arc::clone(&db);
        let deinflector = Arc::clone(&deinflector);
        move || OcrViewer::new(
            iced::widget::image::Handle::from_bytes(bytes_arc.as_ref().clone()),
            bytes_arc.as_ref().clone(), w, h, 1280.0, 720.0,
            annotations.clone(), Arc::clone(&db), Arc::clone(&deinflector),
        )
    };

    let update = |state: &mut OcrViewer, msg: Message| -> iced::Task<Message> {
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
                        let _ = state.state.lookup(sel.line_idx, sel.char_idx, &state.db, &state.deinflector);
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
                        // Navigate alternatives instead of OCR content
                        navigate_alternatives(state, dir);
                    } else {
                        state.state.navigate(dir);
                        state.defer_lookup = true;
                        if let Some((li, ci)) = state.state.current_cursor() {
                            state.move_cursor_to(li, ci);
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
            Message::WindowResized { width, height } => { state.window_width = width as f32; state.window_height = height as f32; }
            Message::ZoomTick => {
                if state.zoom_idle_frames > 3 { state.is_zooming = false; }
                let bits = GP_BITS.load(Ordering::Relaxed);
                let held = bits & (B_UP|B_DOWN|B_LEFT|B_RIGHT|B_L1|B_R1);
                if held != 0 {
                    let mut ra = GP_REPEAT_ACTION.lock().unwrap();
                    if let Some((action, started)) = *ra {
                        let elapsed = started.elapsed();
                        if elapsed >= std::time::Duration::from_millis(500) {
                            let total = ((elapsed.as_millis() - 500) / 50) as u64;
                            let last = GP_LAST_REPEAT.load(Ordering::Relaxed);
                            if total > last {
                                GP_LAST_REPEAT.store(total, Ordering::Relaxed);
                                match action {
                                    GamepadAction::NavigateUp | GamepadAction::NavigateDown
                                    | GamepadAction::NavigateLeft | GamepadAction::NavigateRight => {
                                        state.state.navigate(action);
                                        if let Some((li, ci)) = state.state.current_cursor() {
                                            state.move_cursor_to(li, ci);
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
                            state.select_character(li, ci);
                        }
                    }
                }
            }
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
            state.dict_scroll_y = (state.dict_scroll_y + delta).max(0.0);
            tasks.push(state.scroll_dict_task(state.dict_scroll_y));
        }
        if tasks.is_empty() { iced::Task::none() } else { iced::Task::batch(tasks) }
    };

    let app = iced::application(boot, update, OcrViewer::view)
        .window(iced::window::Settings {
            size: iced::Size::new(w as f32, h as f32),
            ..Default::default()
        })
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
                            if *key == iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape)
                                || *key == iced::keyboard::Key::Character("q".into()) {
                                return Some(Message::Back);
                            }
                            if *key == iced::keyboard::Key::Named(iced::keyboard::key::Named::Enter) {
                                return Some(Message::Navigate(GamepadAction::Confirm));
                            }
                            let action = match key {
                                k if *k == iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowRight)
                                    || *k == iced::keyboard::Key::Character("l".into()) => Some(GamepadAction::NavigateRight),
                                k if *k == iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowLeft)
                                    || *k == iced::keyboard::Key::Character("h".into()) => Some(GamepadAction::NavigateLeft),
                                k if *k == iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowDown)
                                    || *k == iced::keyboard::Key::Character("j".into()) => Some(GamepadAction::NavigateDown),
                                k if *k == iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowUp)
                                    || *k == iced::keyboard::Key::Character("k".into()) => Some(GamepadAction::NavigateUp),
                                _ => None,
                            };
                            if let Some(a) = action {
                                return Some(Message::Navigate(a));
                            }
                            // Dictionary scroll: D=scroll down, F=scroll up
                            if *key == iced::keyboard::Key::Character("d".into()) {
                                return Some(Message::Navigate(GamepadAction::ScrollDown));
                            }
                            if *key == iced::keyboard::Key::Character("f".into()) {
                                return Some(Message::Navigate(GamepadAction::ScrollUp));
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

    if let Err(e) = app.run() { println!("GUI failed: {e:?}"); }
    Ok(())
}