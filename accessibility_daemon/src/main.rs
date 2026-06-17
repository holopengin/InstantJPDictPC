mod models;
mod ocr_engine;
mod overlay_state;
mod viewer;

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use std::io::Cursor;
use std::sync::Arc;

use crate::models::*;
use crate::ocr_engine::OcrEngine;
use crate::viewer::OcrViewer;

fn main() -> Result<()> {
    env_logger::init();
    println!("Accessibility Daemon Starting...");

    let mut engine = OcrEngine::new("./models")?;
    println!("Models loaded successfully.");
    println!(
        "Character vocabulary loaded: {} chars",
        engine.char_vocab.len()
    );

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!(
            "Usage: {} <image_path> [--font /path/to.ttf]",
            args.get(0)
                .map(|s| s.as_str())
                .unwrap_or("accessibility_daemon")
        );
        return Ok(());
    }

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
    let boot = move || {
        OcrViewer::new(
            iced::widget::image::Handle::from_bytes(bytes_arc.as_ref().clone()),
            w,
            h,
            annotations.clone(),
        )
    };

    let update = |state: &mut OcrViewer, message: Message| -> iced::Task<Message> {
        match message {
            Message::SelectCharacter(line_idx, char_idx) => {
                state.select_character(line_idx, char_idx);
            }
            Message::SelectAlternative(new_char) => {
                if let Some(selected) = state.selected_word.as_ref() {
                    state
                        .state
                        .update_character(selected.line_idx, selected.char_idx, new_char);
                    state.select_character(selected.line_idx, selected.char_idx);
                }
            }
            Message::ToggleAlternatives => {
                state.alternatives_visible = !state.alternatives_visible;
            }
            Message::Back => {
                state.selected_word = None;
                state.alternatives_visible = false;
                state.state.is_dictionary_visible = false;
            }
        }
        iced::Task::none()
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
                                Some(Message::Back)
                            } else {
                                None
                            }
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
