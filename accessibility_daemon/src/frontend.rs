//! Frontend launcher window.
//!
//! Shown when the app is launched with no arguments. Two options:
//!   - "Launch Settings Window" — the frontend morphs into the settings UI
//!   - "Launch File Watcher"    — spawns `accessibility_daemon --watcher`
//!     as a detached process and closes this window. If a watcher is already
//!     running, the button becomes "Stop File Watcher" instead.

use iced::{
    alignment::Horizontal,
    widget::{button, column, text, Button, Column, Space, Text},
    Element, Length, Pixels, Subscription, Task,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::data::db::DictionaryDatabase;
use crate::settings_window::{SettingsMessage, SettingsWindow};
use crate::watcher;

/// Messages for the frontend window.
#[derive(Debug, Clone)]
pub enum FrontMsg {
    OpenSettings,
    BackToMenu,
    StartWatcher,
    StopWatcher,
    WatcherTick,
    Settings(SettingsMessage),
}

pub struct FrontendWindow {
    db: Arc<DictionaryDatabase>,
    data_dir: PathBuf,
    settings: Option<SettingsWindow>,
    watcher_running: bool,
}

impl FrontendWindow {
    pub fn new(db: Arc<DictionaryDatabase>, data_dir: PathBuf) -> Self {
        let watcher_running = watcher::is_running(&data_dir);
        Self {
            db,
            data_dir,
            settings: None,
            watcher_running,
        }
    }

    /// Poll the watcher's liveness once per second so the button label
    /// stays in sync (e.g. if the watcher dies on its own).
    pub fn subscription(&self) -> Subscription<FrontMsg> {
        iced::time::every(Duration::from_secs(1)).map(|_| FrontMsg::WatcherTick)
    }

    pub fn update(&mut self, msg: FrontMsg) -> Task<FrontMsg> {
        match msg {
            FrontMsg::OpenSettings => {
                if self.settings.is_none() {
                    self.settings = Some(SettingsWindow::new(Arc::clone(&self.db)));
                }
                Task::none()
            }
            FrontMsg::BackToMenu => {
                self.settings = None;
                self.refresh_watcher();
                Task::none()
            }
            FrontMsg::StartWatcher => {
                self.launch_watcher();
                // Give the watcher a moment to write its pid file: if we
                // exit before that, a quick relaunch sees "not running"
                // and starts a second watcher, orphaning this one.
                for _ in 0..25 {
                    if watcher::is_running(&self.data_dir) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                // Close the frontend window; the watcher keeps running detached.
                std::process::exit(0);
            }
            FrontMsg::StopWatcher => {
                watcher::request_stop(&self.data_dir);
                self.watcher_running = false;
                Task::none()
            }
            FrontMsg::WatcherTick => {
                self.refresh_watcher();
                Task::none()
            }
            FrontMsg::Settings(m) => {
                if let Some(sw) = self.settings.as_mut() {
                    sw.update(m).map(FrontMsg::Settings)
                } else {
                    Task::none()
                }
            }
        }
    }

    fn refresh_watcher(&mut self) {
        self.watcher_running = watcher::is_running(&self.data_dir);
    }

    fn launch_watcher(&mut self) {
        let exe = match std::env::current_exe() {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[Frontend] Cannot resolve current exe: {e}");
                return;
            }
        };
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("--watcher")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        // Detach from our process group so closing the terminal doesn't kill it.
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        match cmd.spawn() {
            Ok(_) => println!("[Frontend] File watcher started (detached)."),
            Err(e) => eprintln!("[Frontend] Failed to start watcher: {e}"),
        }
    }

    pub fn view(&self) -> Element<'_, FrontMsg> {
        if let Some(sw) = &self.settings {
            let back = Button::new(Text::new("← Back to Menu").size(15))
                .on_press(FrontMsg::BackToMenu)
                .style(button::secondary);
            return column![back, sw.view().map(FrontMsg::Settings)]
                .spacing(6)
                .padding(8)
                .into();
        }
        self.menu_view()
    }

    fn menu_view(&self) -> Element<'_, FrontMsg> {
        let title = Text::new("Instant JP Dict").size(30).style(text::primary);
        let subtitle = Text::new("OCR overlay & dictionary daemon")
            .size(14)
            .style(text::secondary);

        let settings_btn = Button::new(
            Text::new("Launch Settings Window")
                .size(16)
                .align_x(Horizontal::Center),
        )
        .width(Length::Fill)
        .padding(12)
        .on_press(FrontMsg::OpenSettings);

        let (label, msg) = if self.watcher_running {
            ("Stop File Watcher", FrontMsg::StopWatcher)
        } else {
            ("Launch File Watcher", FrontMsg::StartWatcher)
        };
        let watcher_btn = Button::new(Text::new(label).size(16).align_x(Horizontal::Center))
            .width(Length::Fill)
            .padding(12)
            .on_press(msg);

        let status = Text::new(if self.watcher_running {
            "File watcher: running"
        } else {
            "File watcher: stopped"
        })
        .size(13)
        .style(text::secondary);

        column![
            title,
            subtitle,
            Space::new().height(Pixels(24.0)),
            settings_btn,
            Space::new().height(Pixels(8.0)),
            watcher_btn,
            Space::new().height(Pixels(16.0)),
            status,
        ]
        .spacing(8)
        .padding(24)
        .width(Length::Fill)
        .into()
    }
}
