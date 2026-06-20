mod data;
mod models;
mod ocr_engine;
mod overlay_state;
mod settings_window;
mod util;
mod viewer;

use anyhow::{Context, Result};
use image::DynamicImage;
use std::io::Cursor;
use std::sync::Arc;

use crate::data::db::DictionaryDatabase;
use crate::models::*;
use crate::ocr_engine::OcrEngine;
use crate::settings_window::SettingsWindow;
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
        Deinflector::from_json_file("data/deinflection.json").unwrap_or_else(|e| {
            println!("Warning: Could not load deinflection rules: {}", e);
            Deinflector::empty()
        }),
    );
    println!("Deinflector loaded: {} rules", deinflector.rule_count());

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        // No image argument — open the settings / management window
        println!("No image argument provided. Opening settings window...");
        run_settings_window(db)?;
        return Ok(());
    }

    // Image argument provided — run the OCR viewer
    run_ocr_viewer(args, db, deinflector)
}

// ---------------------------------------------------------------------------
// Settings window (no image argument)
// ---------------------------------------------------------------------------

fn run_settings_window(db: Arc<DictionaryDatabase>) -> Result<()> {
    let settings_app = iced::application(
        move || SettingsWindow::new(Arc::clone(&db)),
        SettingsWindow::update,
        SettingsWindow::view,
    );

    settings_app.run().context("Failed to run settings window")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// OCR viewer (image argument provided)
// ---------------------------------------------------------------------------

fn run_ocr_viewer(
    args: Vec<String>,
    db: Arc<DictionaryDatabase>,
    deinflector: Arc<Deinflector>,
) -> Result<()> {
    // Initialize OCR engine
    let mut engine = OcrEngine::new("./models")?;
    println!("Models loaded successfully.");
    println!(
        "Character vocabulary loaded: {} chars",
        engine.char_vocab.len()
    );

    let image_path = args[1].clone();

    // Optional font path parsing
    let mut font_path: Option<String> = None;
    for i in 0..args.len() {
        if args[i].starts_with("--font=") {
            font_path = Some(args[i].trim_start_matches("--font=").to_string());
        } else if args[i] == "--font" || args[i] == "-f" {
            if i + 1 < args.len() {
                font_path = Some(args[i + 1].clone());
            }
        }
    }

    let image =
        image::open(&image_path).context(format!("Failed to open image: {}", image_path))?;
    println!(
        "Image loaded successfully: {} ({}x{})",
        image_path,
        image.width(),
        image.height()
    );

    // Run detection + recognition
    let (annotations, _annotated_opt) =
        engine.run_detection(&image, false, font_path.as_deref())?;

    // Encode original image to PNG bytes for the viewer
    let display_img = image.to_rgba8();
    let dynimg = DynamicImage::ImageRgba8(display_img.clone());
    let mut buf = Cursor::new(Vec::new());
    dynimg
        .write_to(&mut buf, image::ImageFormat::Png)
        .context("Failed to encode annotated image to PNG")?;
    let bytes_vec: Vec<u8> = buf.into_inner();
    let bytes_arc = Arc::new(bytes_vec);
    let w = display_img.width();
    let h = display_img.height();

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
                let (root_width, root_height) = (800.0_f32, 480.0_f32);
                state.state.navigate(action, root_width, root_height);
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
                let zoom_factor = if delta > 0.0 { 1.1 } else { 0.9 };
                state.state.current_scale = (state.state.current_scale * zoom_factor).clamp(0.5, 5.0);
                // Adjust translation to zoom toward cursor position
                state.state.current_trans_x = cursor_x - (cursor_x - state.state.current_trans_x) * zoom_factor;
                state.state.current_trans_y = cursor_y - (cursor_y - state.state.current_trans_y) * zoom_factor;
            }
            Message::PanDelta { dx, dy } => {
                state.state.current_trans_x += dx;
                state.state.current_trans_y += dy;
            }
            Message::PanStart { .. } => {
                // Pan start is handled by the canvas state; nothing to do here
            }
            Message::PanEnd => {
                // Pan end is handled by the canvas state; nothing to do here
            }
            Message::SetScale { scale } => {
                state.state.current_scale = scale.clamp(0.5, 5.0);
            }
            Message::PinchZoom { scale_factor, focus_x, focus_y, prev_focus_x, prev_focus_y, base_offset_y } => {
                let old_scale = state.state.current_scale;
                let new_scale = (old_scale * scale_factor).clamp(0.5, 5.0);
                state.state.current_scale = new_scale;
                let actual_factor = if old_scale > 0.0 { new_scale / old_scale } else { 1.0 };
                // Work in "effective" coordinates: eff = trans + base_offset
                // This centers the zoom around the actual image position, not the canvas origin.
                let eff_x = state.state.current_trans_x + 0.0; // base_offset_x is always 0 (left-aligned)
                let eff_y = state.state.current_trans_y + base_offset_y;
                println!("[Pinch] scale={:.3} factor={:.4} focus=({:.1},{:.1}) prev_focus=({:.1},{:.1}) base_off_y={:.1} eff_before=({:.1},{:.1})",
                    new_scale, actual_factor, focus_x, focus_y, prev_focus_x, prev_focus_y, base_offset_y, eff_x, eff_y);
                // Zoom around previous focus: new_eff = new_focus - (old_focus - old_eff) * factor
                let new_eff_x = focus_x - (prev_focus_x - eff_x) * actual_factor;
                let new_eff_y = focus_y - (prev_focus_y - eff_y) * actual_factor;
                // Convert back to trans coordinates
                state.state.current_trans_x = new_eff_x - 0.0;
                state.state.current_trans_y = new_eff_y - base_offset_y;
                println!("[Pinch] eff_after=({:.1},{:.1}) trans_after=({:.1},{:.1})", new_eff_x, new_eff_y, state.state.current_trans_x, state.state.current_trans_y);
            }
        }

        // Return scroll tasks to auto-scroll neighbor/alt panels to the selected character
        let mut tasks = Vec::new();
        // Auto-scroll to selected character (only on initial select)
        if let Some(task) = state.scroll_neighbor_task() {
            tasks.push(task);
        }
        if let Some(task) = state.scroll_alt_task() {
            tasks.push(task);
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
            iced_futures::subscription::filter_map(
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
                            // Arrow keys and hjkl navigation
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
            )
        });

    if let Err(e) = app.run() {
        println!("Failed to run GUI: {:?}", e);
    }

    Ok(())
}
