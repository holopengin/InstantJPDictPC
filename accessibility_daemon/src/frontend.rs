//! Frontend launcher window, rendered with Rune.
//!
//! Shown when the app is launched with no arguments: one entry point into the
//! settings/dictionary-management UI.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::anyhow;
use rune::tok::type_;
use rune::{col, el, row, text, AppSpec, Color, Cx, El};

use crate::settings_window::{SettingsHost, SettingsWindow};
use jpdict_core::app_settings::AppSettings;
use jpdict_core::data::db::DictionaryDatabase;

/// Bundled Japanese face: the frontend also lists dictionary names, so the
/// UI font must cover Japanese. Rune embeds the face in the binary.
pub const SANS_FONT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fonts/NotoSansJP-Regular.ttf"
));
pub const SERIF_FONT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fonts/NotoSerifJP-Regular.ttf"
));

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Screen {
    Menu,
    Settings,
}

pub struct FrontendApp {
    db: Arc<DictionaryDatabase>,
    data_dir: PathBuf,
    app_settings: AppSettings,
    screen: Screen,
    settings: Option<SettingsWindow>,
}

impl FrontendApp {
    pub fn new(db: Arc<DictionaryDatabase>, data_dir: PathBuf) -> Self {
        let app_settings = AppSettings::load(&data_dir);
        Self {
            db,
            data_dir,
            app_settings,
            screen: Screen::Menu,
            settings: None,
        }
    }

    fn open_settings(&mut self) {
        if self.settings.is_none() {
            self.settings = Some(SettingsWindow::new(
                Arc::clone(&self.db),
                self.app_settings.clone(),
            ));
        }
        self.screen = Screen::Settings;
    }

    fn back_to_menu(&mut self) {
        self.screen = Screen::Menu;
    }

    fn poll(&mut self) -> bool {
        let mut changed = false;
        if let Some(sw) = self.settings.as_mut() {
            changed |= sw.poll();
            // Settings own the behaviour switches; keep the frontend's copy
            // in step so a later save cannot write a stale value back.
            if sw.app_settings() != &self.app_settings {
                self.app_settings = sw.app_settings().clone();
                if let Err(e) = self.app_settings.save(&self.data_dir) {
                    eprintln!("[Frontend] Failed to save settings: {e}");
                }
                changed = true;
            }
        }
        changed
    }
}

impl SettingsHost for FrontendApp {
    fn settings_window(&self) -> &SettingsWindow {
        self.settings
            .as_ref()
            .expect("settings screen is only shown with a settings window")
    }
    fn settings_window_mut(&mut self) -> &mut SettingsWindow {
        self.settings
            .as_mut()
            .expect("settings screen is only shown with a settings window")
    }
}

fn app_view(s: &FrontendApp, cx: &mut Cx<FrontendApp>) -> El {
    let _ = cx;
    match s.screen {
        Screen::Menu => menu_view(s),
        Screen::Settings => {
            let body = s
                .settings
                .as_ref()
                .map(|w| w.view::<FrontendApp>())
                .unwrap_or_else(|| text("Settings unavailable").color(MUTED));
            col()
                .w_full()
                .h_full()
                .fill(BG)
                .child(
                    row()
                        .w_full()
                        .px(16.0)
                        .py(10.0)
                        .center_y()
                        .child(back_button("← Back to Menu", |s: &mut FrontendApp| {
                            s.back_to_menu()
                        })),
                )
                .child(el().w_full().grow(1.0).min_h(0.0).child(body))
        }
    }
}

fn menu_view(_s: &FrontendApp) -> El {
    let title = text("Instant JP Dict")
        .font(rune::tok::TextStyle::new(30.0, 700.0))
        .color(TEXT);
    let subtitle = text("OCR overlay & dictionary daemon")
        .font(type_::BODY)
        .color(MUTED);

    let settings_btn = button("Launch Settings Window", ACCENT, |s: &mut FrontendApp| {
        s.open_settings()
    });

    col()
        .w_full()
        .h_full()
        .padding(24.0)
        .gap(8.0)
        .fill(BG)
        .child(title)
        .child(subtitle)
        .child(el().h(16.0))
        .child(settings_btn)
}

fn button(label: &str, fill: Color, on_press: fn(&mut FrontendApp)) -> El {
    el().key(label.to_string())
        .role("button")
        .label(label)
        .w_full()
        .padding(12.0)
        .radius(8.0)
        .fill(fill)
        .hover(lighten(fill, 0.08))
        .press(lighten(fill, 0.16))
        .cursor(rune::Cursor::Pointer)
        .center()
        .on_press(on_press)
        .child(
            text(label.to_string())
                .font(rune::tok::TextStyle::new(16.0, 500.0))
                .color(Color::WHITE)
                .a11y_hidden(),
        )
}

fn back_button(label: &str, on_press: fn(&mut FrontendApp)) -> El {
    el().role("button")
        .label(label)
        .px(12.0)
        .py(6.0)
        .radius(6.0)
        .hover(CARD)
        .cursor(rune::Cursor::Pointer)
        .on_press(on_press)
        .child(
            text(label.to_string())
                .font(type_::BODY)
                .color(TEXT)
                .a11y_hidden(),
        )
}

fn lighten(c: Color, amount: f32) -> Color {
    Color {
        r: (c.r + amount).min(1.0),
        g: (c.g + amount).min(1.0),
        b: (c.b + amount).min(1.0),
        a: c.a,
    }
}

fn app_poll(s: &mut FrontendApp, _t: f64, tx: &mut rune::Tx) -> bool {
    // Keep polling settings import/install progress while the window sits idle.
    tx.wake_after(Duration::from_millis(500));
    s.poll()
}

fn app_key(s: &mut FrontendApp, e: &rune::KeyEvent, _tx: &mut rune::Tx) -> bool {
    if s.screen == Screen::Settings && matches!(e.key, rune::Key::Escape) {
        s.back_to_menu();
        return true;
    }
    false
}

pub fn run(db: Arc<DictionaryDatabase>, data_dir: PathBuf) -> anyhow::Result<()> {
    static FONTS: &[&[u8]] = &[SANS_FONT, SERIF_FONT];
    let spec = AppSpec::<FrontendApp> {
        title: "Instant JP Dict",
        size: (560.0, 300.0),
        fonts: FONTS,
        families: ("Noto Sans JP", "Noto Sans JP"),
        poll: Some(app_poll),
        on_key: Some(app_key),
        ..AppSpec::new(
            "Instant JP Dict",
            (560.0, 300.0),
            || unreachable!("FrontendApp is built by App::init"),
            app_view,
        )
    };
    rune::App::new(spec)
        .init(move || FrontendApp::new(db, data_dir))
        .try_run()
        .map_err(|e| anyhow!(e))
}
