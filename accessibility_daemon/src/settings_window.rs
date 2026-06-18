//! Settings / management window shown when no image argument is provided.
//! Mirrors `MainActivity` from the Kotlin implementation.

use std::sync::{Arc, Mutex};

use iced::widget::{
    button, checkbox, column, container, row, scrollable, text, Button, Container,
    Space, Text,
};
use iced::{alignment, Element, Length, Pixels, Task};

use crate::data::db::DictionaryDatabase;
use crate::data::importer::{DictionaryImporter, ImportProgress};
use crate::data::models::DictionaryMeta;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum SettingsMessage {
    /// Refresh the dictionary list and status from the database.
    RefreshStatus,
    /// Open a file dialog to pick a Yomitan dictionary ZIP.
    ImportDictionary,
    /// A file was picked from the dialog (or None if cancelled).
    FilePicked(Option<std::path::PathBuf>),
    /// Import progress update from the background task (payload is ignored,
    /// the actual progress is read from the shared state).
    ImportProgress,
    /// Import completed with a result message.
    ImportFinished(String),
    /// Delete a dictionary by its id.
    DeleteDictionary(i64),
    /// A background delete operation finished.
    DeleteFinished(i64, Result<(), String>),
    /// Toggle a dictionary's enabled state.
    ToggleEnabled(i64, bool),
    /// Move a dictionary up in priority.
    MoveUp(i64),
    /// Move a dictionary down in priority.
    MoveDown(i64),
    /// Open the Yomitan dictionary download page in a browser.
    OpenDownloadPage,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub struct SettingsWindow {
    pub db: Arc<DictionaryDatabase>,
    pub dictionaries: Vec<DictionaryMeta>,
    pub entry_count: i64,
    pub status_text: String,
    pub import_running: bool,
    /// Current import progress (None when not importing).
    pub import_progress: Option<ImportProgress>,
    /// Shared progress state updated by the import thread, read by the UI.
    pub import_shared: Arc<Mutex<Option<ImportProgress>>>,
    /// Whether a long-running operation (import or delete) is in progress.
    pub busy: bool,
    /// Name of the dictionary currently being deleted (for status display).
    pub deleting_name: Option<String>,
    /// Whether the native file picker dialog is open.
    pub picker_open: bool,
}

impl SettingsWindow {
    pub fn new(db: Arc<DictionaryDatabase>) -> Self {
        let entry_count = db.get_entry_count().unwrap_or(0);
        let dictionaries = db.get_all_dictionaries().unwrap_or_default();
        Self {
            db,
            dictionaries,
            entry_count,
            status_text: String::new(),
            import_running: false,
            import_progress: None,
            import_shared: Arc::new(Mutex::new(None)),
            busy: false,
            deleting_name: None,
            picker_open: false,
        }
    }

    pub fn update(&mut self, message: SettingsMessage) -> Task<SettingsMessage> {
        match message {
            SettingsMessage::RefreshStatus => {
                self.refresh();
                Task::none()
            }

            SettingsMessage::ImportDictionary => {
                if self.busy || self.picker_open {
                    return Task::none();
                }
                self.picker_open = true;
                self.status_text = "Selecting file...".to_string();
                self.import_progress = None;
                Task::perform(
                    async {
                        rfd::AsyncFileDialog::new()
                            .add_filter("ZIP archive", &["zip"])
                            .set_title("Import Yomitan Dictionary")
                            .pick_file()
                            .await
                            .map(|handle| handle.path().to_path_buf())
                    },
                    SettingsMessage::FilePicked,
                )
            }

            SettingsMessage::FilePicked(Some(path)) => {
                self.picker_open = false;
                self.import_running = true;
                self.busy = true;
                self.status_text = "Importing...".to_string();
                self.import_progress = Some(ImportProgress {
                    entries_imported: 0,
                    banks_done: 0,
                    banks_total: 0,
                    current_file: String::new(),
                });

                let db = Arc::clone(&self.db);
                let path_str = path.display().to_string();
                let shared = Arc::clone(&self.import_shared);

                // Initialize shared state.
                *shared.lock().unwrap() = Some(ImportProgress {
                    entries_imported: 0,
                    banks_done: 0,
                    banks_total: 0,
                    current_file: String::new(),
                });

                // Spawn the import on a dedicated thread.
                let path_clone = path_str.clone();
                std::thread::spawn(move || {
                    let importer = DictionaryImporter::new(&db);
                    let shared_for_cb = Arc::clone(&shared);
                    let cb = Box::new(move |p: ImportProgress| {
                        *shared_for_cb.lock().unwrap() = Some(p);
                    });
                    let result = importer.import_zip(&path_clone, Some(cb));
                    // Write final sentinel to shared state.
                    let final_progress = match result {
                        Ok(count) => ImportProgress {
                            entries_imported: count,
                            banks_done: 0,
                            banks_total: 0,
                            current_file: format!("DONE:{}:{}", count, path_clone),
                        },
                        Err(e) => ImportProgress {
                            entries_imported: 0,
                            banks_done: 0,
                            banks_total: 0,
                            current_file: format!("ERROR:{}", e),
                        },
                    };
                    *shared.lock().unwrap() = Some(final_progress);
                });

                // Return a task that polls the shared state once.
                Task::perform(async { () }, |_| SettingsMessage::ImportProgress)
            }

            SettingsMessage::FilePicked(None) => {
                self.picker_open = false;
                self.status_text = "Import cancelled.".to_string();
                self.import_running = false;
                self.busy = false;
                self.import_progress = None;
                *self.import_shared.lock().unwrap() = None;
                Task::none()
            }

            SettingsMessage::ImportProgress => {
                // Read the latest progress from shared state.
                let progress = self
                    .import_shared
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or(ImportProgress {
                        entries_imported: 0,
                        banks_done: 0,
                        banks_total: 0,
                        current_file: String::new(),
                    });

                let done = progress.current_file.starts_with("DONE:")
                    || progress.current_file.starts_with("ERROR:");

                if done {
                    // Import finished.
                    self.import_running = false;
                    self.busy = false;
                    self.import_progress = None;
                    *self.import_shared.lock().unwrap() = None;

                    let result = if progress.current_file.starts_with("DONE:") {
                        let count_str = progress
                            .current_file
                            .strip_prefix("DONE:")
                            .unwrap_or("")
                            .splitn(2, ':')
                            .next()
                            .unwrap_or("0");
                        let count: usize = count_str.parse().unwrap_or(0);
                        println!("Import complete: Imported {count} entries");
                        format!("Imported {count} entries")
                    } else {
                        let err = progress
                            .current_file
                            .strip_prefix("ERROR:")
                            .unwrap_or("Unknown error");
                        eprintln!("Import error: {err}");
                        format!("Error: {err}")
                    };

                    self.status_text = result;
                    self.refresh();
                    Task::none()
                } else {
                    // Still running — update progress and keep polling.
                    self.import_progress = Some(progress);
                    let shared = Arc::clone(&self.import_shared);
                    Task::perform(
                        async move {
                            futures_timer::Delay::new(std::time::Duration::from_millis(80))
                                .await;
                            shared.lock().unwrap().clone().unwrap_or(ImportProgress {
                                entries_imported: 0,
                                banks_done: 0,
                                banks_total: 0,
                                current_file: String::new(),
                            })
                        },
                        |_| SettingsMessage::ImportProgress,
                    )
                }
            }

            SettingsMessage::ImportFinished(_) => {
                // No longer used; completion handled in ImportProgress.
                self.import_running = false;
                self.busy = false;
                self.import_progress = None;
                *self.import_shared.lock().unwrap() = None;
                Task::none()
            }

            SettingsMessage::DeleteDictionary(id) => {
                if self.busy {
                    return Task::none();
                }

                // Find the dictionary name for the status display.
                let name = self
                    .dictionaries
                    .iter()
                    .find(|d| d.id == id)
                    .map(|d| d.name.clone())
                    .unwrap_or_else(|| format!("#{id}"));

                self.busy = true;
                self.deleting_name = Some(name.clone());
                self.status_text = format!("Deleting \"{name}\"..." );

                let db = Arc::clone(&self.db);

                Task::perform(
                    async move {
                        let (tx, rx) = tokio::sync::oneshot::channel();
                        std::thread::spawn(move || {
                            let result = db
                                .delete_dictionary(id)
                                .map_err(|e| e.to_string());
                            let _ = tx.send((id, result));
                        });
                        match rx.await {
                            Ok((id, result)) => (id, result),
                            Err(_) => (id, Err("Delete thread panicked".to_string())),
                        }
                    },
                    |(id, result)| SettingsMessage::DeleteFinished(id, result),
                )
            }

            SettingsMessage::DeleteFinished(id, result) => {
                self.busy = false;
                self.deleting_name = None;

                // Clear the import_shared state if it was stale.
                *self.import_shared.lock().unwrap() = None;

                match result {
                    Ok(()) => {
                        println!("Dictionary {id} deleted.");
                        self.status_text = "Dictionary deleted.".to_string();
                    }
                    Err(e) => {
                        eprintln!("Error deleting dictionary {id}: {e}");
                        self.status_text = format!("Error deleting: {e}");
                    }
                }
                self.refresh();
                Task::none()
            }

            SettingsMessage::ToggleEnabled(id, enabled) => {
                if self.busy {
                    return Task::none();
                }
                if let Err(e) = self.db.set_dictionary_enabled(id, enabled) {
                    eprintln!("Error toggling dictionary {id}: {e}");
                    self.status_text = format!("Error: {e}");
                }
                self.refresh();
                Task::none()
            }

            SettingsMessage::MoveUp(id) => {
                if self.busy {
                    return Task::none();
                }
                self.swap_priority(id, true);
                self.refresh();
                Task::none()
            }

            SettingsMessage::MoveDown(id) => {
                if self.busy {
                    return Task::none();
                }
                self.swap_priority(id, false);
                self.refresh();
                Task::none()
            }

            SettingsMessage::OpenDownloadPage => {
                let _ = open::that("https://github.com/yomidevs/jmdict-yomitan");
                Task::none()
            }
        }
    }

    fn refresh(&mut self) {
        self.entry_count = self.db.get_entry_count().unwrap_or(0);
        self.dictionaries = self.db.get_all_dictionaries().unwrap_or_default();
    }

    fn swap_priority(&self, id: i64, up: bool) {
        let dicts = match self.db.get_all_dictionaries() {
            Ok(d) => d,
            Err(_) => return,
        };
        let pos = match dicts.iter().position(|d| d.id == id) {
            Some(p) => p,
            None => return,
        };
        let neighbor_idx = if up {
            if pos == 0 {
                return;
            }
            pos - 1
        } else {
            if pos + 1 >= dicts.len() {
                return;
            }
            pos + 1
        };
        let my_priority = dicts[pos].priority;
        let neighbor_priority = dicts[neighbor_idx].priority;
        let _ = self.db.update_dictionary_priority(id, neighbor_priority);
        let _ = self
            .db
            .update_dictionary_priority(dicts[neighbor_idx].id, my_priority);
    }

    pub fn view(&self) -> Element<'_, SettingsMessage> {
        let title = Text::new("Instant JP Dict")
            .size(28)
            .style(text::primary);

        let status = Text::new(format!(
            "DB contains {} entries in {} dictionaries",
            self.entry_count,
            self.dictionaries.len()
        ))
        .size(14);

        let mut content = column![title, status,]
            .spacing(8)
            .padding(24)
            .width(Length::Fill);

        if !self.status_text.is_empty() {
            content = content.push(
                Text::new(&self.status_text)
                    .size(13)
                    .style(text::secondary),
            );
        }

        // --- Import progress display ---
        if self.import_running {
            if let Some(ref progress) = self.import_progress {
                let percentage = if progress.banks_total > 0 {
                    (progress.banks_done as f32 / progress.banks_total as f32) * 100.0
                } else {
                    0.0
                };

                let progress_text = if progress.banks_total > 0 {
                    format!(
                        "Importing: {} entries ({} / {} banks, {:.0}%)",
                        progress.entries_imported,
                        progress.banks_done,
                        progress.banks_total,
                        percentage
                    )
                } else {
                    format!("Importing: {} entries...", progress.entries_imported)
                };

                content = content.push(
                    Text::new(progress_text)
                        .size(13)
                        .style(text::secondary),
                );

                // Show current file being processed
                if !progress.current_file.is_empty() {
                    let file_name = std::path::Path::new(&progress.current_file)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(&progress.current_file);
                    content = content.push(
                        Text::new(format!("  → {file_name}"))
                            .size(11)
                            .style(text::secondary),
                    );
                }
            }
            content = content.push(Space::new().height(Pixels(4.0)));
        }

        content = content.push(Space::new().height(Pixels(8.0)));

        // --- Action buttons (disabled while busy) ---

        let disabled = self.busy;
        content = content.push(action_button(
            "Download Dictionaries",
            SettingsMessage::OpenDownloadPage,
            disabled,
        ));
        content = content.push(action_button(
            "Import Yomitan Dictionary (.zip)",
            SettingsMessage::ImportDictionary,
            disabled,
        ));
        content = content.push(action_button(
            "Refresh Status",
            SettingsMessage::RefreshStatus,
            disabled,
        ));

        content = content.push(Space::new().height(Pixels(16.0)));
        content = content.push(Text::new("Dictionaries").size(18).style(text::primary));
        content = content.push(Space::new().height(Pixels(4.0)));

        // --- Dictionary list (disabled while busy) ---

        if self.dictionaries.is_empty() {
            content = content.push(
                Text::new("No dictionaries imported yet.")
                    .size(13)
                    .style(text::secondary),
            );
        } else {
            for dict in &self.dictionaries {
                content = content.push(self.dict_row(dict, disabled));
            }
        }

        let main_content = container(scrollable(content))
            .width(Length::Fill)
            .height(Length::Fill);

        if self.picker_open {
            // Dim the main window and block all input while the file picker is open.
            // The Button intercepts and swallows all mouse/click events.
            iced::widget::Stack::new()
                .push(main_content)
                .push(
                    iced::widget::Button::new(
                        iced::widget::Text::new("")
                    )
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .on_press(SettingsMessage::RefreshStatus)
                    .style(|_, _| {
                        iced::widget::button::Style::default()
                            .with_background(iced::Color::from_rgba(0.0, 0.0, 0.0, 0.5))
                    }),
                )
                .into()
        } else {
            main_content.into()
        }
    }

    fn dict_row<'a>(
        &'a self,
        dict: &'a DictionaryMeta,
        disabled: bool,
    ) -> Container<'a, SettingsMessage> {
        let name_text = if dict.enabled {
            Text::new(&dict.name).size(15)
        } else {
            Text::new(format!("{} (disabled)", dict.name))
                .size(15)
                .style(text::secondary)
        };

        let enabled_checkbox = checkbox(dict.enabled)
            .on_toggle(move |checked| SettingsMessage::ToggleEnabled(dict.id, checked));

        let up_btn = Button::new(Text::new("▲").size(14))
            .on_press(SettingsMessage::MoveUp(dict.id))
            .style(button::secondary);

        let down_btn = Button::new(Text::new("▼").size(14))
            .on_press(SettingsMessage::MoveDown(dict.id))
            .style(button::secondary);

        let delete_btn = if disabled {
            Button::new(Text::new("Delete").size(13)).style(button::danger)
        } else {
            Button::new(Text::new("Delete").size(13))
                .on_press(SettingsMessage::DeleteDictionary(dict.id))
                .style(button::danger)
        };

        let controls = row![up_btn, down_btn, delete_btn]
            .spacing(6)
            .align_y(alignment::Vertical::Center);

        let row_content = row![
            enabled_checkbox,
            name_text,
            Space::new().width(Length::Fill),
            controls
        ]
        .spacing(10)
        .align_y(alignment::Vertical::Center)
        .width(Length::Fill);

        container(row_content).padding(8).style(container::rounded_box)
    }
}

/// Helper to create a full-width action button.
fn action_button<'a>(
    label: &'static str,
    msg: SettingsMessage,
    disabled: bool,
) -> Button<'a, SettingsMessage> {
    let btn = Button::new(
        Text::new(label)
            .size(15)
            .align_x(iced::alignment::Horizontal::Center),
    )
    .width(Length::Fill)
    .padding(10);

    if disabled {
        btn
    } else {
        btn.on_press(msg)
    }
}
