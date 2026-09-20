//! Settings/management window for dictionary import and configuration.

use iced::{
    alignment::{Horizontal, Vertical},
    widget::{
        button, checkbox, column, container, row, scrollable, text, Button, Column,
        Container, Space, Text,
    },
    Element, Length, Pixels, Task,
};
use std::sync::{Arc, Mutex};

use crate::app_settings::AppSettings;
use crate::data::catalog::{self, CatalogEntry};
use crate::data::db::DictionaryDatabase;
use crate::data::models::DictionaryMeta;
use crate::data::importer::{DictionaryImporter, ImportOptions, ImportProgress};
use crate::overlay_font::FontFace;
use futures_timer::Delay;
use std::time::Duration;

/// Message type for the settings window.
#[derive(Debug, Clone)]
pub enum SettingsMessage {
    ImportDictionary,
    ImportProgress,
    ImportDone(Result<usize, String>),
    /// #71: one-click install of a catalog entry (download, verify, import).
    InstallCatalog(String),
    InstallProgress,
    InstallDone(Result<usize, String>),
    RefreshStatus,
    DeleteDictionary(i64),
    ToggleEnabled(i64, bool),
    MoveUp(i64),
    MoveDown(i64),
    /// #100: the furigana (ruby) rule switch. The frontend owns the settings
    /// file; this window renders the checkbox and reports the change.
    SetFuriganaFilter(bool),
    /// Font face switch, rendered here so all behaviour lives in one place.
    /// Like the furigana switch, the frontend owns the file and saves.
    SetFontFace(FontFace),
    CloseWindow,
}

/// Progress of a catalog install: the download phase, then the import phase
/// (whose `DONE:`/`ERROR:` sentinel marks completion, like the file picker).
#[derive(Debug, Clone)]
enum InstallProgress {
    Download { written: u64, total: u64 },
    Import(ImportProgress),
}

/// Every dictionary the manager lists. Nothing is app-owned any more (the
/// vendored zips are gone): catalog installs and file-picker imports are both
/// ordinary rows the user can reorder or delete.
fn all_dictionaries(db: &DictionaryDatabase) -> Vec<DictionaryMeta> {
    db.get_all_dictionaries().unwrap_or_default()
}

pub struct SettingsWindow {
    db: Arc<DictionaryDatabase>,
    /// Snapshot of the persisted app settings (rendered by the Behaviour
    /// section). The frontend owns the file and saves changes; this copy is
    /// refreshed by [`SettingsMessage::SetFuriganaFilter`].
    app_settings: AppSettings,
    entry_count: usize,
    dictionaries: Vec<DictionaryMeta>,
    status_text: String,
    busy: bool,

    // Import progress
    import_running: bool,
    import_progress: Option<ImportProgress>,
    import_shared: Arc<Mutex<Option<ImportProgress>>>,
    import_join_handle: Option<std::thread::JoinHandle<Result<usize, String>>>,

    // Catalog install progress (#71)
    catalog: &'static [CatalogEntry],
    install_running: bool,
    install_entry: Option<String>,
    install_progress: Option<InstallProgress>,
    install_shared: Arc<Mutex<Option<InstallProgress>>>,
    install_join_handle: Option<std::thread::JoinHandle<Result<usize, String>>>,
}

impl SettingsWindow {
    pub fn new(db: Arc<DictionaryDatabase>, app_settings: AppSettings) -> Self {
        let mut window = Self {
            db: Arc::clone(&db),
            app_settings,
            entry_count: 0,
            dictionaries: Vec::new(),
            status_text: String::new(),
            busy: false,
            import_running: false,
            import_progress: None,
            import_shared: Arc::new(Mutex::new(None)),
            import_join_handle: None,
            catalog: catalog::entries(),
            install_running: false,
            install_entry: None,
            install_progress: None,
            install_shared: Arc::new(Mutex::new(None)),
            install_join_handle: None,
        };
        window.refresh();
        window
    }

    fn refresh(&mut self) {
        self.entry_count = self.db.get_entry_count().unwrap_or(0) as usize;
        self.dictionaries = all_dictionaries(&self.db);
    }

    pub fn update(&mut self, message: SettingsMessage) -> Task<SettingsMessage> {
        match message {
            SettingsMessage::ImportDictionary => {
                if self.busy {
                    return Task::none();
                }
                self.status_text = "Selecting file...".to_string();
                self.import_progress = None;
                *self.import_shared.lock().unwrap() = None;

                // Spawn the file picker and import on a dedicated thread.
                let db = Arc::clone(&self.db);
                let shared = Arc::clone(&self.import_shared);

                let join_handle = std::thread::spawn(move || -> Result<usize, String> {
                    let path = rfd::FileDialog::new()
                        .add_filter("ZIP files", &["zip"])
                        .pick_file();

                    let path = match path {
                        Some(p) => p,
                        None => return Err("No file selected".to_string()),
                    };

                    let importer = DictionaryImporter::new(&db);
                    let shared_for_cb = Arc::clone(&shared);
                    let cb = Box::new(move |p: ImportProgress| {
                        *shared_for_cb.lock().unwrap() = Some(p);
                    });

                    let result = importer.import_zip(path, Some(cb)).map_err(|e| e.to_string());

                    // Write final sentinel to shared state.
                    let final_progress = match &result {
                        Ok(count) => ImportProgress {
                            entries_imported: *count,
                            banks_done: 0,
                            banks_total: 0,
                            current_file: format!("DONE:{count}"),
                        },
                        Err(e) => ImportProgress {
                            entries_imported: 0,
                            banks_done: 0,
                            banks_total: 0,
                            current_file: format!("ERROR:{e}"),
                        },
                    };
                    *shared.lock().unwrap() = Some(final_progress);

                    result
                });

                self.import_running = true;
                self.busy = true;
                self.import_join_handle = Some(join_handle);

                // Return a task that polls the shared state once.
                Task::perform(async {}, |_| SettingsMessage::ImportProgress)
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
                    self.import_join_handle = None;

                    let result = if progress.current_file.starts_with("DONE:") {
                        let count_str = progress
                            .current_file
                            .strip_prefix("DONE:")
                            .unwrap_or("");
                        let count: usize = count_str.parse().unwrap_or(0);
                        Ok(count)
                    } else {
                        let err = progress
                            .current_file
                            .strip_prefix("ERROR:")
                            .unwrap_or("Unknown error");
                        Err(err.to_string())
                    };

                    return Task::perform(
                    async move { result },
                    SettingsMessage::ImportDone,
                );
                } else {
                    // Still running — update progress and keep polling.
                    self.import_progress = Some(progress);
                    let shared = Arc::clone(&self.import_shared);
                    Task::perform(
                        async move {
                            Delay::new(Duration::from_millis(80)).await;
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

            SettingsMessage::ImportDone(result) => {
                match result {
                    Ok(count) => {
                        self.status_text = format!("Imported {count} entries");
                        self.refresh();
                    }
                    Err(e) => {
                        if e != "No file selected" {
                            self.status_text = format!("Error: {e}");
                        } else {
                            self.status_text = "Import cancelled".to_string();
                        }
                    }
                }
                Task::none()
            }

            SettingsMessage::RefreshStatus => {
                self.refresh();
                self.status_text = "Refreshed".to_string();
                Task::none()
            }

            SettingsMessage::DeleteDictionary(id) => {
                if self.busy {
                    return Task::none();
                }
                // The bundled pitch dictionary is app state: never deletable,
                // even if a stale button ever reaches this handler.
                if self.dictionaries.iter().any(|d| d.id == id && d.built_in) {
                    self.status_text = "Bundled dictionaries cannot be deleted".to_string();
                    return Task::none();
                }
                if let Err(e) = self.db.delete_dictionary(id) {
                    self.status_text = format!("Delete failed: {}", e);
                } else {
                    self.status_text = "Dictionary deleted".to_string();
                    self.refresh();
                }
                Task::none()
            }

            SettingsMessage::ToggleEnabled(id, enabled) => {
                if self.busy {
                    return Task::none();
                }
                if let Err(e) = self.db.set_dictionary_enabled(id, enabled) {
                    eprintln!("Error toggling dictionary {}: {}", id, e);
                    self.status_text = format!("Error: {}", e);
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

            SettingsMessage::SetFuriganaFilter(enabled) => {
                // The frontend owns settings.json and saves the change; keep
                // this window's rendered copy in step.
                self.app_settings.furigana_filter = enabled;
                Task::none()
            }

            SettingsMessage::SetFontFace(face) => {
                // Same ownership as the furigana switch: the frontend saves.
                self.app_settings.overlay_font = face;
                Task::none()
            }

            SettingsMessage::InstallCatalog(id) => {
                if self.busy {
                    return Task::none();
                }
                let Some(entry) = self.catalog.iter().find(|e| e.id == id).cloned() else {
                    return Task::none();
                };
                self.status_text = format!("Downloading {}...", entry.name);
                self.install_progress = None;
                *self.install_shared.lock().unwrap() = None;
                println!(
                    "[Catalog] Installing '{}' from {} ({})",
                    entry.name, entry.source, entry.license
                );

                let db = Arc::clone(&self.db);
                let shared = Arc::clone(&self.install_shared);
                let name = entry.name.clone();

                let join_handle = std::thread::spawn(move || -> Result<usize, String> {
                    let dest = std::env::temp_dir().join(format!("ijd-catalog-{}.zip", entry.id));
                    let result = (|| -> Result<usize, String> {
                        // Download and verify first: nothing reaches the
                        // database until the pinned size and SHA-256 match.
                        {
                            let shared = Arc::clone(&shared);
                            let mut on_progress = move |written: u64, total: u64| {
                                *shared.lock().unwrap() =
                                    Some(InstallProgress::Download { written, total });
                            };
                            crate::data::download::download_verified(
                                &entry,
                                &dest,
                                &mut on_progress,
                            )
                            .map_err(|e| format!("{e:#}"))?;
                        }

                        let importer = DictionaryImporter::new(&db);
                        let shared_for_cb = Arc::clone(&shared);
                        let cb = Box::new(move |p: ImportProgress| {
                            *shared_for_cb.lock().unwrap() = Some(InstallProgress::Import(p));
                        });
                        importer
                            .import_zip_with(
                                &dest,
                                Some(cb),
                                ImportOptions {
                                    built_in: false,
                                    catalog_id: Some(entry.id.clone()),
                                },
                            )
                            .map_err(|e| e.to_string())
                    })();
                    let _ = std::fs::remove_file(&dest);

                    // Write the completion sentinel the poll loop watches for,
                    // on success or failure alike: a failed download must not
                    // leave the window polling forever.
                    let final_progress = match &result {
                        Ok(count) => ImportProgress {
                            entries_imported: *count,
                            banks_done: 0,
                            banks_total: 0,
                            current_file: format!("DONE:{count}"),
                        },
                        Err(e) => ImportProgress {
                            entries_imported: 0,
                            banks_done: 0,
                            banks_total: 0,
                            current_file: format!("ERROR:{e}"),
                        },
                    };
                    *shared.lock().unwrap() = Some(InstallProgress::Import(final_progress));
                    result
                });

                self.install_running = true;
                self.busy = true;
                self.install_entry = Some(name);
                self.install_join_handle = Some(join_handle);

                Task::perform(async {}, |_| SettingsMessage::InstallProgress)
            }

            SettingsMessage::InstallProgress => {
                let progress = self.install_shared.lock().unwrap().clone();
                let done = match &progress {
                    Some(InstallProgress::Import(p)) => {
                        p.current_file.starts_with("DONE:") || p.current_file.starts_with("ERROR:")
                    }
                    _ => false,
                };

                if done {
                    self.install_running = false;
                    self.busy = false;
                    self.install_entry = None;
                    self.install_join_handle = None;
                    self.install_progress = None;
                    *self.install_shared.lock().unwrap() = None;

                    let result = match progress {
                        Some(InstallProgress::Import(p)) => {
                            if let Some(count) = p.current_file.strip_prefix("DONE:") {
                                Ok(count.parse().unwrap_or(0))
                            } else {
                                Err(p
                                    .current_file
                                    .strip_prefix("ERROR:")
                                    .unwrap_or("Unknown error")
                                    .to_string())
                            }
                        }
                        _ => Err("Install finished without a result".to_string()),
                    };
                    Task::perform(async move { result }, SettingsMessage::InstallDone)
                } else {
                    self.install_progress = progress;
                    Task::perform(
                        async {
                            Delay::new(Duration::from_millis(80)).await;
                        },
                        |_| SettingsMessage::InstallProgress,
                    )
                }
            }

            SettingsMessage::InstallDone(result) => {
                match result {
                    Ok(count) => {
                        self.status_text = format!("Installed {count} entries");
                        self.refresh();
                    }
                    Err(e) => self.status_text = format!("Install failed: {e}"),
                }
                Task::none()
            }

            SettingsMessage::CloseWindow => {
                std::process::exit(0);
            }
        }
    }

    fn swap_priority(&self, id: i64, up: bool) {
        let dicts = all_dictionaries(&self.db);
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

        // --- Close button at top-right ---
        let close_btn = Button::new(
            Text::new("✕")
                .size(18)
                .align_x(Horizontal::Center),
        )
        .width(Pixels(40.0))
        .height(Pixels(40.0))
        .style(button::danger)
        .on_press(SettingsMessage::CloseWindow);

        let header = row![
            title,
            Space::new().width(Length::Fill),
            close_btn,
        ]
        .spacing(10)
        .align_y(Vertical::Center);

        let mut content = column![header, status,]
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

        // --- Import / catalog install progress display ---
        if self.import_running {
            if let Some(ref progress) = self.import_progress {
                content = import_progress_lines(progress, content);
            }
            content = content.push(Space::new().height(Pixels(4.0)));
        }
        if self.install_running {
            if let Some(InstallProgress::Download { written, total }) = &self.install_progress {
                let mb = |b: u64| b as f64 / 1_048_576.0;
                content = content.push(
                    Text::new(format!(
                        "Downloading {}: {:.1} / {:.1} MB",
                        self.install_entry.as_deref().unwrap_or("dictionary"),
                        mb(*written),
                        mb(*total),
                    ))
                    .size(13)
                    .style(text::secondary),
                );
            }
            if let Some(InstallProgress::Import(progress)) = &self.install_progress {
                content = import_progress_lines(progress, content);
            }
            content = content.push(Space::new().height(Pixels(4.0)));
        }

        content = content.push(Space::new().height(Pixels(8.0)));

        // --- Behaviour: toggles that apply to the next capture / launch. ---

        content = content.push(Text::new("Behaviour").size(18).style(text::primary));
        content = content.push(Space::new().height(Pixels(4.0)));

        // #100: the furigana (ruby) rule, off by default — it is flaky on
        // camera photos and a wrong drop costs a whole line. Off keeps the
        // small ruby contours as their own lines.
        let furigana_check = checkbox(self.app_settings.furigana_filter)
            .on_toggle(SettingsMessage::SetFuriganaFilter);
        content = content.push(
            row![
                furigana_check,
                Text::new("Filter furigana (ruby)").size(14),
            ]
            .spacing(6)
            .align_y(Vertical::Center),
        );
        content = content.push(
            Text::new(
                "Drops small ruby text beside kanji so it does not become its own \
                 result. Applies to the next capture.",
            )
            .size(12)
            .style(text::secondary),
        );

        // Font face, moved here from the frontend menu so all behaviour lives
        // in one place. Applies to the overlay and the dictionary panel alike
        // (both register the same face at viewer startup).
        let serif_check = checkbox(self.app_settings.overlay_font == FontFace::Serif)
            .on_toggle(|serif| {
                SettingsMessage::SetFontFace(if serif {
                    FontFace::Serif
                } else {
                    FontFace::Sans
                })
            });
        content = content.push(
            row![
                serif_check,
                Text::new("Serif font (Noto Serif JP)").size(14),
            ]
            .spacing(6)
            .align_y(Vertical::Center),
        );
        content = content.push(
            Text::new("Overlay & dictionary panel — takes effect next launch.")
                .size(12)
                .style(text::secondary),
        );

        content = content.push(Space::new().height(Pixels(16.0)));

        // --- Action buttons (disabled while busy) ---

        let disabled = self.busy;
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
        content = content.push(Text::new("Dictionary Catalog").size(18).style(text::primary));
        content = content.push(Space::new().height(Pixels(4.0)));

        // --- One-click installs (#71): the dictionaries that used to be
        // bundled. An installed entry keeps its row and shows "Installed"
        // (mobile's chip); the dictionary is also in the list below, where it
        // can be reordered. ---
        for entry in self.catalog {
            let installed = catalog::is_installed(&self.dictionaries, entry);
            content = content.push(self.catalog_row(entry, installed, disabled));
        }

        content = content.push(Space::new().height(Pixels(16.0)));
        content = content.push(Text::new("Dictionary Manager").size(18).style(text::primary));
        content = content.push(Space::new().height(Pixels(4.0)));

        // --- Installed dictionaries (disabled while busy) ---

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

        container(scrollable(content))
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    /// One catalog row: name, description + size + licence, and either an
    /// Install button or the "Installed" marker (mobile's chip).
    fn catalog_row<'a>(
        &'a self,
        entry: &'a CatalogEntry,
        installed: bool,
        disabled: bool,
    ) -> Container<'a, SettingsMessage> {
        let info = column![
            Text::new(&entry.name).size(15),
            Text::new(format!(
                "{} · {:.1} MB · {}",
                entry.description,
                entry.bytes as f64 / 1_048_576.0,
                entry.license
            ))
            .size(12)
            .style(text::secondary),
        ]
        .spacing(2);

        let action: Element<'a, SettingsMessage> = if installed {
            Text::new("Installed").size(13).style(text::secondary).into()
        } else if disabled {
            Button::new(Text::new("Install").size(13)).into()
        } else {
            Button::new(Text::new("Install").size(13))
                .on_press(SettingsMessage::InstallCatalog(entry.id.clone()))
                .into()
        };

        let row_content = row![
            info,
            Space::new().width(Length::Fill),
            action
        ]
        .spacing(10)
        .align_y(Vertical::Center)
        .width(Length::Fill);

        container(row_content).padding(8).style(container::rounded_box)
    }

    fn dict_row<'a>(
        &'a self,
        dict: &'a DictionaryMeta,
        disabled: bool,
    ) -> Container<'a, SettingsMessage> {
        let name_text = if dict.built_in {
            Text::new(format!("{} (bundled)", dict.name)).size(15)
        } else if dict.enabled {
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

        // The bundled dictionary has no Delete: it is app state, not a
        // user import (mobile hides built-ins from the manager entirely).
        let delete_btn = if dict.built_in {
            None
        } else if disabled {
            Some(Button::new(Text::new("Delete").size(13)).style(button::danger))
        } else {
            Some(
                Button::new(Text::new("Delete").size(13))
                    .on_press(SettingsMessage::DeleteDictionary(dict.id))
                    .style(button::danger),
            )
        };

        let mut controls = row![up_btn, down_btn].spacing(6).align_y(Vertical::Center);
        if let Some(btn) = delete_btn {
            controls = controls.push(btn);
        }

        let row_content = row![
            enabled_checkbox,
            name_text,
            Space::new().width(Length::Fill),
            controls
        ]
        .spacing(10)
        .align_y(Vertical::Center)
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
            .align_x(Horizontal::Center),
    )
    .width(Length::Fill)
    .padding(10);

    if disabled {
        btn
    } else {
        btn.on_press(msg)
    }
}

/// The progress lines both import phases share: entry/bank counts and the
/// current file (never the `DONE:`/`ERROR:` sentinel, which the poll loop
/// consumes instead of rendering).
fn import_progress_lines<'a>(
    progress: &ImportProgress,
    mut content: Column<'a, SettingsMessage>,
) -> Column<'a, SettingsMessage> {
    let percentage = if progress.banks_total > 0 {
        (progress.banks_done as f32 / progress.banks_total as f32) * 100.0
    } else {
        0.0
    };
    let progress_text = if progress.banks_total > 0 {
        format!(
            "Importing: {} entries ({} / {} banks, {:.0}%)",
            progress.entries_imported, progress.banks_done, progress.banks_total, percentage
        )
    } else {
        format!("Importing: {} entries...", progress.entries_imported)
    };
    content = content.push(Text::new(progress_text).size(13).style(text::secondary));

    let sentinel = progress.current_file.starts_with("DONE:")
        || progress.current_file.starts_with("ERROR:");
    if !progress.current_file.is_empty() && !sentinel {
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
    content
}