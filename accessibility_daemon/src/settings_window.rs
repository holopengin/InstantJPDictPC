//! Settings/management window for dictionary import and configuration,
//! rendered with Rune.

use std::sync::{Arc, Mutex};

use rune::tok::TextStyle;
use rune::{col, el, row, text, Color, El, ScrollSpec};

use jpdict_core::app_settings::AppSettings;
use jpdict_core::data::catalog::{self, CatalogEntry};
use jpdict_core::data::db::DictionaryDatabase;
use jpdict_core::data::importer::{DictionaryImporter, ImportOptions, ImportProgress};
use jpdict_core::data::models::DictionaryMeta;
use jpdict_core::overlay_font::FontFace;

// Rune's colours are linear-light; these are the sRGB design values decoded.
const BG: Color = Color {
    r: 0.009715,
    g: 0.009715,
    b: 0.009715,
    a: 1.0,
};
const CARD: Color = Color {
    r: 0.018478,
    g: 0.018478,
    b: 0.018478,
    a: 1.0,
};
const CARD_BORDER: Color = Color {
    r: 0.046964,
    g: 0.046964,
    b: 0.046964,
    a: 1.0,
};
const TEXT: Color = Color {
    r: 0.868899,
    g: 0.868899,
    b: 0.868899,
    a: 1.0,
};
const MUTED: Color = Color {
    r: 0.342392,
    g: 0.342392,
    b: 0.342392,
    a: 1.0,
};
const ACCENT: Color = Color {
    r: 0.046964,
    g: 0.223414,
    b: 0.748414,
    a: 1.0,
};
const DANGER: Color = Color {
    r: 0.342392,
    g: 0.039682,
    b: 0.039682,
    a: 1.0,
};

/// Message type for the settings window.
#[derive(Debug, Clone)]
pub enum SettingsMessage {
    ImportDictionary,
    ImportDone(Result<usize, String>),
    /// #71: one-click install of a catalog entry (download, verify, import).
    InstallCatalog(String),
    InstallDone(Result<usize, String>),
    RefreshStatus,
    DeleteDictionary(i64),
    ToggleEnabled(i64, bool),
    MoveUp(i64),
    MoveDown(i64),
    /// #100: the furigana (ruby) rule switch. The frontend owns the settings
    /// file; this window renders the checkbox and reports the change.
    SetFuriganaFilter(bool),
    /// #43/#86: the pitch-accent line switch (mobile's default-off
    /// home-screen toggle). Same ownership as the furigana switch.
    SetPitchAccent(bool),
    /// Font face switch, rendered here so all behaviour lives in one place.
    /// Like the furigana switch, the frontend owns the file and saves.
    SetFontFace(FontFace),
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

/// A view can be hosted by the frontend app or by a standalone settings
/// window, so its handlers ask the host for the window they edit.
pub trait SettingsHost: 'static {
    fn settings_window(&self) -> &SettingsWindow;
    fn settings_window_mut(&mut self) -> &mut SettingsWindow;
}

pub struct SettingsWindow {
    db: Arc<DictionaryDatabase>,
    /// Snapshot of the persisted app settings (rendered by the Behaviour
    /// section). The frontend owns the file and saves changes; this copy is
    /// refreshed by [`SettingsMessage::SetFuriganaFilter`],
    /// [`SettingsMessage::SetPitchAccent`] and [`SettingsMessage::SetFontFace`].
    pub app_settings: AppSettings,
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

    pub fn app_settings(&self) -> &AppSettings {
        &self.app_settings
    }

    fn refresh(&mut self) {
        self.entry_count = self.db.get_entry_count().unwrap_or(0) as usize;
        self.dictionaries = all_dictionaries(&self.db);
    }

    /// Called once per frame by the host. Polls the worker threads and
    /// returns true when something on screen changed.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        if self.import_running {
            let progress = self.import_shared.lock().unwrap().clone();
            if let Some(p) = progress {
                let done =
                    p.current_file.starts_with("DONE:") || p.current_file.starts_with("ERROR:");
                if done {
                    self.import_running = false;
                    self.busy = false;
                    self.import_progress = None;
                    *self.import_shared.lock().unwrap() = None;
                    self.import_join_handle = None;
                    let result = if let Some(count) = p.current_file.strip_prefix("DONE:") {
                        Ok(count.parse().unwrap_or(0))
                    } else {
                        Err(p
                            .current_file
                            .strip_prefix("ERROR:")
                            .unwrap_or("Unknown error")
                            .to_string())
                    };
                    self.handle(SettingsMessage::ImportDone(result));
                } else {
                    self.import_progress = Some(p);
                }
                changed = true;
            }
        }
        if self.install_running {
            let progress = self.install_shared.lock().unwrap().clone();
            if let Some(p) = progress {
                let done = match &p {
                    InstallProgress::Import(ip) => {
                        ip.current_file.starts_with("DONE:")
                            || ip.current_file.starts_with("ERROR:")
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
                    let result = match p {
                        InstallProgress::Import(ip) => {
                            if let Some(count) = ip.current_file.strip_prefix("DONE:") {
                                Ok(count.parse().unwrap_or(0))
                            } else {
                                Err(ip
                                    .current_file
                                    .strip_prefix("ERROR:")
                                    .unwrap_or("Unknown error")
                                    .to_string())
                            }
                        }
                        _ => Err("Install finished without a result".to_string()),
                    };
                    self.handle(SettingsMessage::InstallDone(result));
                } else {
                    self.install_progress = Some(p);
                }
                changed = true;
            }
        }
        changed
    }

    pub fn handle(&mut self, message: SettingsMessage) {
        match message {
            SettingsMessage::ImportDone(result) => match result {
                Ok(count) => {
                    self.status_text = format!("Imported {count} entries");
                    self.refresh();
                }
                Err(e) => {
                    self.status_text = if e == "No file selected" {
                        "Import cancelled".to_string()
                    } else {
                        format!("Error: {e}")
                    };
                }
            },
            SettingsMessage::InstallDone(result) => match result {
                Ok(count) => {
                    self.status_text = format!("Installed {count} entries");
                    self.refresh();
                }
                Err(e) => self.status_text = format!("Install failed: {e}"),
            },
            SettingsMessage::RefreshStatus => {
                self.refresh();
                self.status_text = "Refreshed".to_string();
            }
            SettingsMessage::DeleteDictionary(id) => {
                if self.busy {
                    return;
                }
                // The bundled pitch dictionary is app state: never deletable,
                // even if a stale button ever reaches this handler.
                if self.dictionaries.iter().any(|d| d.id == id && d.built_in) {
                    self.status_text = "Bundled dictionaries cannot be deleted".to_string();
                    return;
                }
                if let Err(e) = self.db.delete_dictionary(id) {
                    self.status_text = format!("Delete failed: {e}");
                } else {
                    self.status_text = "Dictionary deleted".to_string();
                    self.refresh();
                }
            }
            SettingsMessage::ToggleEnabled(id, enabled) => {
                if self.busy {
                    return;
                }
                if let Err(e) = self.db.set_dictionary_enabled(id, enabled) {
                    eprintln!("Error toggling dictionary {id}: {e}");
                    self.status_text = format!("Error: {e}");
                }
                self.refresh();
            }
            SettingsMessage::MoveUp(id) => {
                if !self.busy {
                    self.swap_priority(id, true);
                    self.refresh();
                }
            }
            SettingsMessage::MoveDown(id) => {
                if !self.busy {
                    self.swap_priority(id, false);
                    self.refresh();
                }
            }
            SettingsMessage::SetFuriganaFilter(enabled) => {
                // The frontend owns settings.json and saves the change; keep
                // this window's rendered copy in step.
                self.app_settings.furigana_filter = enabled;
            }
            SettingsMessage::SetPitchAccent(enabled) => {
                self.app_settings.pitch_accent = enabled;
            }
            SettingsMessage::SetFontFace(face) => {
                self.app_settings.overlay_font = face;
            }
            SettingsMessage::ImportDictionary => self.start_import(),
            SettingsMessage::InstallCatalog(id) => self.start_install(id),
        }
    }

    fn start_import(&mut self) {
        if self.busy {
            return;
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

            let result = importer
                .import_zip(path, Some(cb))
                .map_err(|e| e.to_string());

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
    }

    fn start_install(&mut self, id: String) {
        if self.busy {
            return;
        }
        let Some(entry) = self.catalog.iter().find(|e| e.id == id).cloned() else {
            return;
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
                // Download and verify first: nothing reaches the database
                // until the pinned size and SHA-256 match.
                {
                    let shared = Arc::clone(&shared);
                    let mut on_progress = move |written: u64, total: u64| {
                        *shared.lock().unwrap() =
                            Some(InstallProgress::Download { written, total });
                    };
                    jpdict_core::data::download::download_verified(&entry, &dest, &mut on_progress)
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

            // Write the completion sentinel the poll loop watches for, on
            // success or failure alike: a failed download must not leave the
            // window polling forever.
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

    /// The settings screen. `S` is the host app, so the same screen works in
    /// the frontend's morphed state and in a standalone settings window.
    pub fn view<S: SettingsHost>(&self) -> El {
        let title = text("Instant JP Dict")
            .font(TextStyle::new(28.0, 700.0))
            .color(TEXT);

        let status = text(format!(
            "DB contains {} entries in {} dictionaries",
            self.entry_count,
            self.dictionaries.len()
        ))
        .font(TextStyle::new(14.0, 400.0))
        .color(MUTED);

        let close_btn = el()
            .key("close")
            .role("button")
            .label("Close")
            .size(40.0, 40.0)
            .radius(8.0)
            .fill(DANGER)
            .hover(lighten(DANGER, 0.1))
            .cursor(rune::Cursor::Pointer)
            .center()
            .on_press_tx(|_: &mut S, tx| tx.quit())
            .child(
                text("×")
                    .font(TextStyle::new(22.0, 500.0))
                    .color(TEXT)
                    .a11y_hidden(),
            );

        let header = row()
            .w_full()
            .center_y()
            .gap(10.0)
            .child(title)
            .child(el().grow(1.0))
            .child(close_btn);

        let mut content = col()
            .w_full()
            .padding(24.0)
            .gap(8.0)
            .fill(BG)
            .child(header)
            .child(status);

        if !self.status_text.is_empty() {
            content = content.child(
                text(self.status_text.clone())
                    .font(TextStyle::new(13.0, 400.0))
                    .color(MUTED),
            );
        }

        // --- Import / catalog install progress display ---
        if self.import_running {
            if let Some(ref progress) = self.import_progress {
                content = import_progress_lines(progress, content);
            }
        }
        if self.install_running {
            if let Some(InstallProgress::Download { written, total }) = &self.install_progress {
                let mb = |b: u64| b as f64 / 1_048_576.0;
                content = content.child(
                    text(format!(
                        "Downloading {}: {:.1} / {:.1} MB",
                        self.install_entry.as_deref().unwrap_or("dictionary"),
                        mb(*written),
                        mb(*total),
                    ))
                    .font(TextStyle::new(13.0, 400.0))
                    .color(MUTED),
                );
            }
            if let Some(InstallProgress::Import(progress)) = &self.install_progress {
                content = import_progress_lines(progress, content);
            }
        }

        content = content.child(el().h(8.0));

        // --- Behaviour: toggles that apply to the next capture / launch. ---
        content = content.child(
            text("Behaviour")
                .font(TextStyle::new(18.0, 600.0))
                .color(TEXT),
        );

        content = content.child(checkbox_row::<S>(
            "furigana",
            "Filter furigana (ruby)",
            self.app_settings.furigana_filter,
            |s| {
                let on = !s.settings_window().app_settings.furigana_filter;
                s.settings_window_mut()
                    .handle(SettingsMessage::SetFuriganaFilter(on));
            },
        ));
        content = content.child(hint(
            "Drops small ruby text beside kanji so it does not become its own \
             result. Applies to the next capture.",
        ));

        content = content.child(checkbox_row::<S>(
            "pitch",
            "Show pitch accent",
            self.app_settings.pitch_accent,
            |s| {
                let on = !s.settings_window().app_settings.pitch_accent;
                s.settings_window_mut()
                    .handle(SettingsMessage::SetPitchAccent(on));
            },
        ));
        content = content.child(hint(
            "Draws the pitch-accent line under each entry's headwords. \
             Takes effect next launch.",
        ));

        content = content.child(checkbox_row::<S>(
            "serif",
            "Serif font (Noto Serif JP)",
            self.app_settings.overlay_font == FontFace::Serif,
            |s| {
                let on = s.settings_window().app_settings.overlay_font == FontFace::Serif;
                s.settings_window_mut()
                    .handle(SettingsMessage::SetFontFace(if on {
                        FontFace::Sans
                    } else {
                        FontFace::Serif
                    }));
            },
        ));
        content = content.child(hint(
            "Overlay & dictionary panel — takes effect next launch.",
        ));

        content = content.child(el().h(16.0));

        // --- Action buttons (disabled while busy) ---
        let disabled = self.busy;
        content = content.child(action_button::<S>(
            "Import Yomitan Dictionary (.zip)",
            !disabled,
            |s| {
                s.settings_window_mut()
                    .handle(SettingsMessage::ImportDictionary)
            },
        ));
        content = content.child(action_button::<S>("Refresh Status", !disabled, |s| {
            s.settings_window_mut()
                .handle(SettingsMessage::RefreshStatus)
        }));

        content = content.child(el().h(16.0));
        content = content.child(
            text("Dictionary Catalog")
                .font(TextStyle::new(18.0, 600.0))
                .color(TEXT),
        );

        // --- One-click installs (#71): the dictionaries that used to be
        // bundled. An installed entry keeps its row and shows "Installed". ---
        for entry in self.catalog {
            let installed = catalog::is_installed(&self.dictionaries, entry);
            content = content.child(catalog_row::<S>(entry, installed, disabled));
        }

        content = content.child(el().h(16.0));
        content = content.child(
            text("Dictionary Manager")
                .font(TextStyle::new(18.0, 600.0))
                .color(TEXT),
        );

        // --- Installed dictionaries (disabled while busy) ---
        if self.dictionaries.is_empty() {
            content = content.child(
                text("No dictionaries imported yet.")
                    .font(TextStyle::new(13.0, 400.0))
                    .color(MUTED),
            );
        } else {
            for dict in &self.dictionaries {
                content = content.child(dict_row::<S>(dict, disabled));
            }
        }

        content
            .scroll(ScrollSpec {
                stick_bottom: false,
            })
            .w_full()
            .h_full()
            .min_h(0.0)
    }
}

/// Wire a [`SettingsMessage`] to the host's window.
fn send<S: SettingsHost>(msg: SettingsMessage) -> impl Fn(&mut S) + 'static {
    move |s: &mut S| s.settings_window_mut().handle(msg.clone())
}

fn checkbox_row<S: SettingsHost>(
    key: &'static str,
    label: &'static str,
    on: bool,
    flip: fn(&mut S),
) -> El {
    let box_el = el()
        .key(key)
        .role("checkbox")
        .label(label)
        .checked(on)
        .focusable(true)
        .size(18.0, 18.0)
        .radius(4.0)
        .border(1.0, if on { ACCENT } else { CARD_BORDER })
        .fill(if on { ACCENT } else { Color::TRANSPARENT })
        .center()
        .cursor(rune::Cursor::Pointer)
        .on_press(flip)
        .child(
            text(if on { "✓" } else { "" })
                .font(TextStyle::new(12.0, 700.0))
                .color(Color::WHITE)
                .a11y_hidden(),
        );

    row()
        .key(format!("{key}_row"))
        .w_full()
        .gap(6.0)
        .center_y()
        .child(box_el)
        .child(
            text(label)
                .font(TextStyle::new(14.0, 400.0))
                .color(TEXT)
                .a11y_hidden(),
        )
}

fn hint(body: &str) -> El {
    text(body.to_string())
        .font(TextStyle::new(12.0, 400.0))
        .color(MUTED)
        .wrap()
        .w_full()
}

fn action_button<S: SettingsHost>(label: &'static str, enabled: bool, on_press: fn(&mut S)) -> El {
    let mut b = el()
        .key(label)
        .role("button")
        .label(label)
        .w_full()
        .padding(10.0)
        .radius(8.0)
        .fill(if enabled { ACCENT } else { CARD })
        .center();
    if enabled {
        b = b
            .hover(lighten(ACCENT, 0.08))
            .press(lighten(ACCENT, 0.16))
            .cursor(rune::Cursor::Pointer)
            .on_press(on_press);
    }
    b.child(
        text(label)
            .font(TextStyle::new(15.0, 500.0))
            .color(if enabled { Color::WHITE } else { MUTED })
            .a11y_hidden(),
    )
}

/// One catalog row: name, description + size + licence, and either an
/// Install button or the "Installed" marker (mobile's chip).
fn catalog_row<S: SettingsHost>(entry: &CatalogEntry, installed: bool, disabled: bool) -> El {
    let info = col()
        .gap(2.0)
        .grow(1.0)
        .min_w(0.0)
        .child(
            text(entry.name.clone())
                .font(TextStyle::new(15.0, 500.0))
                .color(TEXT)
                .wrap()
                .w_full(),
        )
        .child(
            text(format!(
                "{} · {:.1} MB · {}",
                entry.description,
                entry.bytes as f64 / 1_048_576.0,
                entry.license
            ))
            .font(TextStyle::new(12.0, 400.0))
            .color(MUTED)
            .wrap()
            .w_full(),
        );

    let action = if installed {
        el().child(
            text("Installed")
                .font(TextStyle::new(13.0, 400.0))
                .color(MUTED),
        )
    } else {
        let id = entry.id.clone();
        let mut b = el()
            .role("button")
            .label("Install")
            .px(12.0)
            .py(6.0)
            .radius(6.0)
            .fill(if disabled { CARD } else { ACCENT });
        if !disabled {
            b = b
                .hover(lighten(ACCENT, 0.08))
                .cursor(rune::Cursor::Pointer)
                .on_press(send::<S>(SettingsMessage::InstallCatalog(id)));
        }
        b.child(
            text("Install")
                .font(TextStyle::new(13.0, 500.0))
                .color(if disabled { MUTED } else { Color::WHITE }),
        )
    };

    el().w_full()
        .padding(8.0)
        .radius(8.0)
        .fill(CARD)
        .border(1.0, CARD_BORDER)
        .child(
            row()
                .w_full()
                .gap(10.0)
                .center_y()
                .child(info)
                .child(action),
        )
}

fn dict_row<S: SettingsHost>(dict: &DictionaryMeta, disabled: bool) -> El {
    let name_text = if dict.built_in {
        format!("{} (bundled)", dict.name)
    } else if dict.enabled {
        dict.name.clone()
    } else {
        format!("{} (disabled)", dict.name)
    };

    let id = dict.id;
    let enabled = dict.enabled;
    let enable_box = el()
        .role("checkbox")
        .label(name_text.clone())
        .checked(enabled)
        .focusable(true)
        .size(18.0, 18.0)
        .radius(4.0)
        .border(1.0, if enabled { ACCENT } else { CARD_BORDER })
        .fill(if enabled { ACCENT } else { Color::TRANSPARENT })
        .center()
        .cursor(rune::Cursor::Pointer)
        .on_press(send::<S>(SettingsMessage::ToggleEnabled(id, !enabled)))
        .child(
            text(if enabled { "✓" } else { "" })
                .font(TextStyle::new(12.0, 700.0))
                .color(Color::WHITE),
        );

    let up_btn = small_button("▲", !disabled, send::<S>(SettingsMessage::MoveUp(id)));
    let down_btn = small_button("▼", !disabled, send::<S>(SettingsMessage::MoveDown(id)));

    // The bundled dictionary has no Delete: it is app state, not a user
    // import (mobile hides built-ins from the manager entirely).
    let mut controls = row().gap(6.0).center_y().child(up_btn).child(down_btn);
    if !dict.built_in {
        controls = controls.child(text_button(
            "Delete",
            !disabled,
            DANGER,
            send::<S>(SettingsMessage::DeleteDictionary(id)),
        ));
    }

    el().w_full()
        .padding(8.0)
        .radius(8.0)
        .fill(CARD)
        .border(1.0, CARD_BORDER)
        .child(
            row()
                .w_full()
                .gap(10.0)
                .center_y()
                .child(enable_box)
                .child(
                    el().grow(1.0).min_w(0.0).child(
                        text(name_text)
                            .font(TextStyle::new(15.0, 400.0))
                            .color(if dict.enabled || dict.built_in {
                                TEXT
                            } else {
                                MUTED
                            })
                            .wrap()
                            .w_full(),
                    ),
                )
                .child(controls),
        )
}

fn small_button<S: SettingsHost>(
    label: &'static str,
    enabled: bool,
    on_press: impl Fn(&mut S) + 'static,
) -> El {
    let mut b = el()
        .role("button")
        .label(label)
        .size(30.0, 26.0)
        .radius(6.0)
        .fill(CARD)
        .border(1.0, CARD_BORDER)
        .center();
    if enabled {
        b = b
            .hover(lighten(CARD, 0.08))
            .cursor(rune::Cursor::Pointer)
            .on_press(on_press);
    }
    b.child(text(label).font(TextStyle::new(14.0, 500.0)).color(TEXT))
}

fn text_button<S: SettingsHost>(
    label: &'static str,
    enabled: bool,
    fill: Color,
    on_press: impl Fn(&mut S) + 'static,
) -> El {
    let mut b = el()
        .role("button")
        .label(label)
        .px(10.0)
        .py(5.0)
        .radius(6.0)
        .fill(if enabled { fill } else { CARD });
    if enabled {
        b = b
            .hover(lighten(fill, 0.08))
            .cursor(rune::Cursor::Pointer)
            .on_press(on_press);
    }
    b.child(
        text(label)
            .font(TextStyle::new(13.0, 500.0))
            .color(if enabled { TEXT } else { MUTED }),
    )
}

/// The progress lines both import phases share: entry/bank counts and the
/// current file (never the `DONE:`/`ERROR:` sentinel, which the poll loop
/// consumes instead of rendering).
fn import_progress_lines(progress: &ImportProgress, mut content: El) -> El {
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
    content = content.child(
        text(progress_text)
            .font(TextStyle::new(13.0, 400.0))
            .color(MUTED),
    );

    let sentinel =
        progress.current_file.starts_with("DONE:") || progress.current_file.starts_with("ERROR:");
    if !progress.current_file.is_empty() && !sentinel {
        let file_name = std::path::Path::new(&progress.current_file)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&progress.current_file);
        content = content.child(
            text(format!("  → {file_name}"))
                .font(TextStyle::new(11.0, 400.0))
                .color(MUTED),
        );
    }
    content
}

fn lighten(c: Color, amount: f32) -> Color {
    Color {
        r: (c.r + amount).min(1.0),
        g: (c.g + amount).min(1.0),
        b: (c.b + amount).min(1.0),
        a: c.a,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty throwaway database for window tests (an in-memory SQLite
    /// file: no fixture rows, no disk writes).
    fn empty_db() -> Arc<DictionaryDatabase> {
        Arc::new(DictionaryDatabase::open(":memory:").expect("in-memory db"))
    }

    /// #43/#86: the pitch message updates the window's snapshot copy (the
    /// frontend owns the file and saves; the furigana switch is the model).
    #[test]
    fn pitch_message_updates_snapshot_copy() {
        let mut window = SettingsWindow::new(empty_db(), AppSettings::default());
        assert!(!window.app_settings.pitch_accent);
        window.handle(SettingsMessage::SetPitchAccent(true));
        assert!(window.app_settings.pitch_accent);
        window.handle(SettingsMessage::SetPitchAccent(false));
        assert!(!window.app_settings.pitch_accent);
    }
}
