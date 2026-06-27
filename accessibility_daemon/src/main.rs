mod data;
mod models;
#[cfg(feature = "ort")]
mod ocr_engine;
#[cfg(feature = "ort")]
mod ocr_parallel;
mod overlay_state;
// mod settings_window; // TODO: fix edition 2024 async block issues
mod util;
mod viewer;

use anyhow::{Context, Result};
use image::DynamicImage;
use std::io::Cursor;
use std::sync::Arc;


use crate::data::db::DictionaryDatabase;
use crate::models::*;
// use crate::settings_window::SettingsWindow; // TODO: fix edition 2024 async block issues
use crate::util::deinflector::Deinflector;
use crate::viewer::OcrViewer;

fn main() -> Result<()> {
    env_logger::init();
    println!("Accessibility Daemon Starting...");

    // Initialize dictionary database
    let db = Arc::new(DictionaryDatabase::open("dictionary.sqlite")?);
    let entry_count = db.get_entry_count()?;
    println!("Dictionary database loaded: {} entries", entry_count);

    // Initialize deinflector
    let deinflector = Arc::new(
        Deinflector::from_json_file("assets/deinflect.json").unwrap_or_else(|e| {
            println!("Warning: Could not load deinflection rules: {}", e);
            Deinflector::empty()
        }),
    );
    println!("Deinflector loaded: {} rules", deinflector.rule_count());

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        // No image argument — open the settings / management window
        println!("No image argument provided. Opening settings window...");
        // run_settings_window(db)?; // TODO: fix edition 2024 async block issues
        println!("Settings window disabled (edition 2024 async block issues).");
        return Ok(());
    }

    // Image argument provided — run the OCR viewer
    run_ocr_viewer(args, db, deinflector)
}

// ---------------------------------------------------------------------------
// Settings window (no image argument)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn run_settings_window(_db: Arc<DictionaryDatabase>) -> Result<()> {
    // TODO: fix edition 2024 async block issues
    unimplemented!("settings window requires edition 2024")
}

// ---------------------------------------------------------------------------
// OCR viewer (image argument provided)
// ---------------------------------------------------------------------------

fn run_ocr_viewer(
    args: Vec<String>,
    db: Arc<DictionaryDatabase>,
    deinflector: Arc<Deinflector>,
) -> Result<()> {
    // Parse args: find image path (first non-flag arg) and optional flags
    let mut image_path = None;
    let mut font_path: Option<String> = None;
    let mut headless = false;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--headless" || args[i] == "-h" {
            headless = true;
            i += 1;
        } else if args[i].starts_with("--font=") {
            font_path = Some(args[i].trim_start_matches("--font=").to_string());
            i += 1;
        } else if args[i] == "--font" || args[i] == "-f" {
            if i + 1 < args.len() {
                font_path = Some(args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        } else if args[i].starts_with("-") {
            // Unknown flag, skip
            i += 1;
        } else {
            // First non-flag argument is the image path
            image_path = Some(args[i].clone());
            break;
        }
    }
    let image_path = image_path.context("No image path provided")?;

    let image =
        image::open(&image_path).context(format!("Failed to open image: {}", image_path))?;
    println!(
        "Image loaded successfully: {} ({}x{})",
        image_path,
        image.width(),
        image.height()
    );

    // Run detection + recognition (render=true to draw boxes on the image)
    let (annotations, annotated_opt) = {
        #[cfg(feature = "ort")]
        {
            println!("Using ORT (ONNX Runtime) backend.");
            let mut engine = ocr_engine::OcrEngine::new("./assets")?;
            println!("Models loaded successfully.");
            println!(
                "Character vocabulary loaded: {} chars",
                engine.char_vocab.len()
            );
            engine.run_detection(&image, true, font_path.as_deref())?
        }
        #[cfg(not(feature = "ort"))]
        {
            eprintln!("ORT backend not compiled in. Rebuild with --features ort");
            std::process::exit(1);
        }
    };

    // Use annotated image if available, otherwise original
    let display_img = if let Some(_annotated) = annotated_opt {
        image.to_rgba8()
    } else {
        image.to_rgba8()
    };
    let dynimg = DynamicImage::ImageRgba8(display_img.clone());
    let mut buf = Cursor::new(Vec::new());
    dynimg
        .write_to(&mut buf, image::ImageFormat::Png)
        .context("Failed to encode annotated image to PNG")?;
    let bytes_vec: Vec<u8> = buf.into_inner();
    let bytes_arc = Arc::new(bytes_vec);
    let w = display_img.width();
    let h = display_img.height();

    // If headless mode, skip GUI and exit after detection
    if headless {
        println!("Headless mode: detection complete, skipping GUI");
        return Ok(());
    }

    // Launch Iced application
    let boot = {
        let db = Arc::clone(&db);
        let deinflector = Arc::clone(&deinflector);
        move || {
            OcrViewer::new(
                iced::widget::image::Handle::from_bytes(bytes_arc.as_ref().clone()),
                bytes_arc.as_ref().clone(),
                w,
                h,
                annotations.clone(),
                Arc::clone(&db),
                Arc::clone(&deinflector),
            )
        }
    };

    let update = |state: &mut OcrViewer, message: Message| -> iced::Task<Message> {
        match message {
            Message::SelectCharacter(line_idx, char_idx) => {
                state.select_character(line_idx, char_idx);
            }
            Message::SelectNeighbor(line_idx, char_idx) => {
                state.select_neighbor(line_idx, char_idx);
            }
            Message::SelectAlternative(new_char) => {
                if let Some(selected) = state.selected_word.as_ref() {
                    let current_char = state.state.active_line_results
                        .get(selected.line_idx)
                        .and_then(|l| l.as_ref())
                        .and_then(|line| line.text.chars().nth(selected.char_idx));
                    if current_char == Some(new_char) {
                        state.alternatives_visible = false;
                    } else {
                        state.state.update_character(selected.line_idx, selected.char_idx, new_char);
                        let _ = state.state.lookup(selected.line_idx, selected.char_idx, &state.db, &state.deinflector);
                    }
                }
            }
            Message::Navigate(action) => {
                state.state.navigate(action);
                if let Some((line_idx, char_idx)) = state.state.current_cursor() {
                    state.select_character(line_idx, char_idx);
                }
            }
            Message::Back => {
                if state.alternatives_visible {
                    state.alternatives_visible = false;
                } else if state.selected_word.is_some() {
                    state.selected_word = None;
                    state.state.is_dictionary_visible = false;
                } else {
                    println!("[App] Back with no selection — exiting");
                    std::process::exit(0);
                }
            }
            Message::ZoomOnCursor { delta, cursor_x, cursor_y } => {
                let requested_factor = if delta > 0.0 { 1.1 } else { 0.9 };
                let old_scale = state.state.current_scale;
                let new_scale = (old_scale * requested_factor).clamp(0.5, 5.0);
                // Use the actual factor after clamping so pan matches the real zoom change
                let actual_factor = if old_scale > 0.0 { new_scale / old_scale } else { 1.0 };
                state.state.current_scale = new_scale;
                state.state.current_trans_x = cursor_x - (cursor_x - state.state.current_trans_x) * actual_factor;
                state.state.current_trans_y = cursor_y - (cursor_y - state.state.current_trans_y) * actual_factor;
                state.is_zooming = true;
                state.zoom_idle_frames = 0;
            }
            Message::PanDelta { dx, dy } => {
                state.state.current_trans_x += dx;
                state.state.current_trans_y += dy;
                state.zoom_idle_frames = 0;
            }
            Message::PanStart { .. } => {
                state.is_zooming = true;
                state.zoom_idle_frames = 0;
            }
            Message::PanEnd => {
                state.is_zooming = false;
                state.zoom_idle_frames = 0;
            }
            Message::SetScale { scale } => {
                state.state.current_scale = scale.clamp(0.5, 5.0);
            }
            Message::PinchZoom { scale_factor, focus_x, focus_y, prev_focus_x, prev_focus_y, base_offset_y } => {
                let old_scale = state.state.current_scale;
                let new_scale = (old_scale * scale_factor).clamp(0.5, 5.0);
                state.state.current_scale = new_scale;
                let actual_factor = if old_scale > 0.0 { new_scale / old_scale } else { 1.0 };
                let eff_x = state.state.current_trans_x;
                let eff_y = state.state.current_trans_y + base_offset_y;
                state.state.current_trans_x = focus_x - (prev_focus_x - eff_x) * actual_factor;
                state.state.current_trans_y = (focus_y - (prev_focus_y - eff_y) * actual_factor) - base_offset_y;
                state.is_zooming = true;
                state.zoom_idle_frames = 0;
            }
            Message::PinchEnd => {
                state.is_zooming = false;
                state.zoom_idle_frames = 0;
            }
            Message::WindowResized { width, height } => {
                state.state.window_width.set(width);
                state.state.window_height.set(height);
            }
            Message::ZoomTick => {
                // Re-enable annotations if no zoom/pan activity for a few ticks
                if state.zoom_idle_frames > 3 {
                    state.is_zooming = false;
                }
            }
        }

        // Increment idle counter every update; reset on zoom/pan activity
        if matches!(message, Message::ZoomOnCursor { .. } | Message::PanDelta { .. } | Message::PanStart { .. } | Message::PinchZoom { .. }) {
            state.zoom_idle_frames = 0;
        } else {
            state.zoom_idle_frames = state.zoom_idle_frames.saturating_add(1);
        }

        // Return scroll tasks to auto-scroll panels when selection changes
        let mut tasks = Vec::new();
        // Always scroll dictionary to top when content changes
        if matches!(message, Message::SelectCharacter(_, _) | Message::SelectNeighbor(_, _) | Message::SelectAlternative(_)) {
            tasks.push(state.scroll_dict_to_top_task());
        }
        // Scroll neighbor/alt panels to the selected character.
        // Clear the targets after firing so they don't re-fire on every frame.
        if let Some(task) = state.scroll_neighbor_task() {
            tasks.push(task);
            state.scroll_neighbor_to = None;
        }
        if let Some(task) = state.scroll_alt_task() {
            tasks.push(task);
            state.scroll_alt_to = None;
        }
        if tasks.is_empty() {
            iced::Task::none()
        } else {
            iced::Task::batch(tasks)
        }
    };

    let view = OcrViewer::view;

    let app = iced::application(boot, update, view)
        .subscription(|_state: &OcrViewer| {
            // Merge global keyboard/mouse events with a periodic zoom-check timer
            let global_events = iced_futures::subscription::filter_map(
                "global-events",
                |event: iced_futures::subscription::Event| {
                    match &event {
                        iced_futures::subscription::Event::Interaction {
                            event: iced::Event::Keyboard(iced::keyboard::Event::KeyPressed { key, .. }),
                            ..
                        } => {
                            println!("[Subscription] KeyPressed: {:?}", key);
                            if key == &iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape) {
                                println!("[Subscription] Escape -> Back");
                                return Some(Message::Back);
                            }
                            if key == &iced::keyboard::Key::Named(iced::keyboard::key::Named::Enter) {
                                println!("[Subscription] Enter -> Confirm (SelectCharacter)");
                                return Some(Message::SelectCharacter(0, 0));
                            }
                            let action = if key == &iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowRight)
                                || key == &iced::keyboard::Key::Character("l".into())
                            {
                                Some(GamepadAction::NavigateRight)
                            } else if key == &iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowLeft)
                                || key == &iced::keyboard::Key::Character("h".into())
                            {
                                Some(GamepadAction::NavigateLeft)
                            } else if key == &iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowDown)
                                || key == &iced::keyboard::Key::Character("j".into())
                            {
                                Some(GamepadAction::NavigateDown)
                            } else if key == &iced::keyboard::Key::Named(iced::keyboard::key::Named::ArrowUp)
                                || key == &iced::keyboard::Key::Character("k".into())
                            {
                                Some(GamepadAction::NavigateUp)
                            } else {
                                None
                            };
                            if let Some(nav_action) = action {
                                println!("[Subscription] Navigation {:?}", nav_action);
                                return Some(Message::Navigate(nav_action));
                            }
                            None
                        }
                        iced_futures::subscription::Event::Interaction {
                            event: iced::Event::Mouse(iced::mouse::Event::ButtonPressed(iced::mouse::Button::Right)),
                            ..
                        } => {
                            println!("[Subscription] Right-click -> Back");
                            Some(Message::Back)
                        }
                        _ => None,
                    }
                },
            );

            // Periodic timer to detect when scroll-wheel zoom has stopped
            let zoom_timer = iced::time::every(iced::time::Duration::from_millis(30))
                .map(|_| Message::ZoomTick);

            // Window resize events to track actual window dimensions
            let resize_events = iced::window::resize_events()
                .map(|(_id, size)| Message::WindowResized {
                    width: size.width,
                    height: size.height,
                });

            iced_futures::Subscription::batch(vec![global_events, zoom_timer, resize_events])
        });

    if let Err(e) = app.run() {
        println!("Failed to run GUI: {:?}", e);
    }

    Ok(())
}
