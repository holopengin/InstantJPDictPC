//! Active-window capture — take the screenshot ourselves instead of waiting
//! for one to appear in a folder.
//!
//! On KDE Wayland, KWin exposes `org.kde.KWin.ScreenShot2` on the session bus.
//! The compositor hands the raw pixels back over a pipe we create, so no
//! screenshot ever hits `~/Pictures/Screenshots` and no second viewer can open
//! for the same capture.
//!
//! KWin restricts this interface to processes whose executable is named in an
//! installed `.desktop` file carrying
//! `X-KDE-DBUS-Restricted-Interfaces=org.kde.KWin.ScreenShot2` — the same
//! mechanism Spectacle uses. [`setup_capture_authorization`] writes that
//! desktop file for the running binary; [`capture_active_window`] calls it
//! automatically the first time KWin refuses.
//!
//! The canonical desktop file is also a KDE command shortcut
//! (`X-KDE-GlobalAccel-CommandShortcut`), so [`setup_capture_shortcut`] can
//! launch the app and bind the key in one step. Inside an AppImage, KWin must
//! see the runtime's real executable path, so a hidden compatibility alias
//! carries that path while the visible shortcut keeps the stable `.AppImage`
//! path. KWin reads its allow-list at startup, so a freshly written file may
//! need one logout/login before the compositor accepts captures.
//!
//! Captures are written under `$XDG_RUNTIME_DIR/instantjpdict/captures`
//! (a tmpfs), or the app's data dir as a fallback.

use std::collections::HashMap;
use std::io::Read;
use std::os::fd::AsFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use zbus::zvariant::{Fd, Value};

use image::RgbaImage;

/// Desktop-file id for the KWin authorization / command-shortcut entry.
pub const CAPTURE_DESKTOP_ID: &str = "net.local.instantjpdict.desktop";
/// Shortcut registered by `--setup-capture` when no shortcut exists yet.
pub const DEFAULT_CAPTURE_SHORTCUT: &str = "Meta+Print";
/// Fallback if the default is already taken.
const FALLBACK_CAPTURE_SHORTCUT: &str = "Ctrl+Alt+S";
const CAPTURE_DESKTOP_NAME: &str = "InstantJPDict: Capture Window";
const LEGACY_DESKTOP_ID: &str = "accessibility_daemon.desktop";

const KGLOBALACCEL_BUS_NAME: &str = "org.kde.kglobalaccel";
const KGLOBALACCEL_OBJECT_PATH: &str = "/kglobalaccel";
const KGLOBALACCEL_INTERFACE: &str = "org.kde.KGlobalAccel";
/// `KGlobalAccel::SetPresent`: register the key and persist it.
const KGLOBALACCEL_SET_PRESENT: u32 = 4;

const KWIN_BUS_NAME: &str = "org.kde.KWin";
const KWIN_OBJECT_PATH: &str = "/org/kde/KWin/ScreenShot2";
const KWIN_INTERFACE: &str = "org.kde.KWin.ScreenShot2";
/// Error KWin returns when the caller is not in its restricted-interface list.
const ERROR_NOT_AUTHORIZED: &str = "org.kde.KWin.ScreenShot2.Error.NoAuthorized";
/// Desktop-file key KWin reads to decide who may use ScreenShot2.
const RESTRICTED_INTERFACES_KEY: &str = "X-KDE-DBUS-Restricted-Interfaces";
/// Prefix for files this module writes, so cleanup only removes our own
/// captures.
pub const CAPTURE_PREFIX: &str = "instantjp_capture-";

/// Capture the currently focused window as an RGBA image.
///
/// If KWin refuses because this binary is not registered yet, the desktop-file
/// registration is written automatically and the capture is retried while
/// KWin's service cache catches up.
pub fn capture_active_window() -> Result<RgbaImage> {
    let mut registered = false;
    // First attempt; on a fresh install the service cache can lag behind the
    // desktop file by a moment, hence the retries.
    for attempt in 0..6u32 {
        match capture_once() {
            Ok(image) => return Ok(image),
            // Only the not-authorized reply is retried; everything else is
            // either fatal or actionable as-is.
            Err(e) if attempt < 5 && is_not_authorized(&e) => {
                if !registered {
                    // Best effort: if this fails the error below is still the
                    // useful message.
                    let _ = setup_capture_authorization();
                    registered = true;
                }
                std::thread::sleep(Duration::from_millis(700));
            }
            Err(e) => {
                if let Some(actionable) = classify_capture_error(&e) {
                    return Err(actionable);
                }
                return Err(e).context("KWin ScreenShot2 request failed");
            }
        }
    }
    unreachable!("retry loop always returns by attempt 5");
}

fn is_not_authorized(err: &zbus::Error) -> bool {
    matches!(err, zbus::Error::MethodError(name, _, _) if name.as_str() == ERROR_NOT_AUTHORIZED)
}

/// Turn the compositor's known error replies into messages that say what to do.
fn classify_capture_error(err: &zbus::Error) -> Option<anyhow::Error> {
    let zbus::Error::MethodError(name, _, _) = err else {
        return None;
    };
    match name.as_str() {
        ERROR_NOT_AUTHORIZED => Some(anyhow::anyhow!(
            "KWin still refuses the screenshot after registering this binary \
             and refreshing its service cache.\n\
             Try `{} --setup-capture` again; if it keeps refusing, log out and \
             back in so KWin rebuilds its service cache from scratch.",
            executable_name(),
        )),
        "org.freedesktop.DBus.Error.ServiceUnknown" => Some(anyhow::anyhow!(
            "KWin's ScreenShot2 D-Bus service is not available.\n\
             --capture currently needs a KDE Plasma session; on Steam Deck game\n\
             mode (gamescope), open the screenshot directly."
        )),
        "org.kde.KWin.ScreenShot2.Error.NoActiveWindow" => {
            Some(anyhow::anyhow!("no window is focused, so there is nothing to capture"))
        }
        _ => None,
    }
}

/// One capture attempt, no permission handling: ask KWin for the active window,
/// read the raw pixels from the pipe, convert to RGBA.
fn capture_once() -> Result<RgbaImage, zbus::Error> {
    // A socket pair stands in for a pipe: KWin dup()s and writes to its end,
    // we read ours. The read timeout keeps a compositor bug from hanging the
    // hotkey forever.
    let (mut reader, writer) = UnixStream::pair().map_err(|e| zbus::Error::InputOutput(e.into()))?;
    reader
        .set_read_timeout(Some(Duration::from_secs(20)))
        .map_err(|e| zbus::Error::InputOutput(e.into()))?;

    let conn = zbus::blocking::Connection::session()?;

    let mut options: HashMap<&str, Value<'_>> = HashMap::new();
    // Full physical pixels: OCR does better on native resolution than on the
    // compositor's logical size.
    options.insert("native-resolution", Value::Bool(true));
    // The drop shadow is just background around the window.
    options.insert("include-shadow", Value::Bool(false));

    let reply = conn.call_method(
        Some(KWIN_BUS_NAME),
        KWIN_OBJECT_PATH,
        Some(KWIN_INTERFACE),
        "CaptureActiveWindow",
        &(&options, Fd::from(writer.as_fd())),
    )?;
    // KWin has dup()ed the fd by now; ours can go away.
    drop(writer);

    let body = reply.body();
    let results: HashMap<String, Value<'_>> = body
        .deserialize()
        .map_err(|e| zbus::Error::Failure(format!("malformed ScreenShot2 reply: {e}")))?;

    let width = u32_field(&results, "width")?;
    let height = u32_field(&results, "height")?;
    let stride = u32_field(&results, "stride")?;
    let format = u32_field(&results, "format")?;
    if width == 0 || height == 0 {
        return Err(zbus::Error::Failure(
            "KWin returned an empty screenshot (no active window?)".into(),
        ));
    }

    let row_bytes = width as usize * 4;
    if (stride as usize) < row_bytes {
        return Err(zbus::Error::Failure(format!(
            "KWin returned stride {stride} for width {width} (need at least {row_bytes})"
        )));
    }
    let total = stride as usize * height as usize;
    let mut raw = vec![0u8; total];
    reader
        .read_exact(&mut raw)
        .map_err(|e| zbus::Error::InputOutput(e.into()))?;

    qimage_to_rgba(width, height, stride, format, &raw)
        .map_err(|e| zbus::Error::Failure(e.to_string()))
}

fn u32_field(results: &HashMap<String, Value<'_>>, key: &str) -> Result<u32, zbus::Error> {
    results
        .get(key)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| zbus::Error::Failure(format!("ScreenShot2 reply is missing `{key}`")))
}

/// Convert KWin's raw QImage bytes into RGBA composited onto white.
///
/// KWin has returned `Format_ARGB32_Premultiplied` (BGRA, premultiplied) on
/// older Plasma and `Format_RGBA8888_Premultiplied` on newer; the other raw
/// formats are handled for completeness. Numbers are `QImage::Format` values.
fn qimage_to_rgba(
    width: u32,
    height: u32,
    stride: u32,
    format: u32,
    raw: &[u8],
) -> Result<RgbaImage> {
    const RGB32: u32 = 4;
    const ARGB32: u32 = 5;
    const ARGB32_PREMULTIPLIED: u32 = 6;
    const RGBX8888: u32 = 16;
    const RGBA8888: u32 = 17;
    const RGBA8888_PREMULTIPLIED: u32 = 18;

    if !matches!(
        format,
        RGB32 | ARGB32 | ARGB32_PREMULTIPLIED | RGBX8888 | RGBA8888 | RGBA8888_PREMULTIPLIED
    ) {
        anyhow::bail!(
            "unsupported screenshot pixel format {format} \
             (expected a 32-bit QImage::Format)"
        );
    }

    let row_bytes = width as usize * 4;
    let stride = stride as usize;
    let mut out = RgbaImage::new(width, height);

    // Byte order in memory: ARGB32-family images are BGRA on little-endian,
    // RGBA8888-family are RGBA. Alpha is either straight or premultiplied.
    let (bgra, premultiplied) = match format {
        RGB32 => (true, false),
        ARGB32 => (true, false),
        ARGB32_PREMULTIPLIED => (true, true),
        RGBX8888 => (false, false),
        RGBA8888 => (false, false),
        RGBA8888_PREMULTIPLIED => (false, true),
        _ => unreachable!(),
    };
    let opaque = format == RGB32 || format == RGBX8888;

    for y in 0..height as usize {
        let row = &raw[y * stride..y * stride + row_bytes];
        for x in 0..width as usize {
            let px = &row[x * 4..x * 4 + 4];
            let (r, g, b) = if bgra {
                (px[2], px[1], px[0])
            } else {
                (px[0], px[1], px[2])
            };
            let a = if opaque { 255 } else { px[3] };

            // Composite over white. Premultiplied color is already scaled by
            // alpha, so adding the missing alpha is the whole blend; straight
            // alpha needs the full formula.
            let (r, g, b) = if opaque {
                (r, g, b)
            } else if premultiplied {
                (
                    r.saturating_add(255 - a),
                    g.saturating_add(255 - a),
                    b.saturating_add(255 - a),
                )
            } else {
                let blend = |c: u8| ((c as u32 * a as u32 + 255 * (255 - a as u32)) / 255) as u8;
                (blend(r), blend(g), blend(b))
            };
            out.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, 255]));
        }
    }
    Ok(out)
}

/// Save a capture to the runtime capture directory and return its path.
/// Files older than 24 h are pruned so a long-lived session cannot fill tmpfs.
pub fn save_capture(image: &RgbaImage, data_dir: &Path) -> Result<PathBuf> {
    let dir = capture_dir(data_dir);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create {}", dir.display()))?;
    prune_old_captures(&dir);

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = dir.join(format!("{CAPTURE_PREFIX}{stamp}.png"));
    image
        .save(&path)
        .with_context(|| format!("failed to save capture to {}", path.display()))?;
    Ok(path)
}

fn capture_dir(data_dir: &Path) -> PathBuf {
    match std::env::var("XDG_RUNTIME_DIR") {
        Ok(runtime) if !runtime.is_empty() => PathBuf::from(runtime)
            .join("instantjpdict")
            .join("captures"),
        _ => data_dir.join("captures"),
    }
}

fn prune_old_captures(dir: &Path) {
    const MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with(CAPTURE_PREFIX) {
            continue;
        }
        let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        if now.duration_since(modified).map_or(false, |age| age > MAX_AGE) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Install (or refresh) the desktop file that lets KWin authorize this binary
/// for `org.kde.KWin.ScreenShot2` **and** makes it available to KDE's
/// command-shortcut registry. Returns the desktop file path.
///
/// The canonical file is the command shortcut: its `Exec` is the launcher the
/// user runs (the AppImage path when packaged). KWin authorizes by the
/// caller's real executable path, which inside an AppImage is the transient
/// mount path, so the hidden compatibility alias carries the current runtime
/// path instead. Splitting the two keeps the shortcut stable while KWin still
/// accepts the capture.
pub fn setup_capture_authorization() -> Result<PathBuf> {
    let launch_exe = launch_path();
    let launch_exe = launch_exe.canonicalize().unwrap_or(launch_exe);
    let runtime_exe = runtime_path();
    let runtime_exe = runtime_exe.canonicalize().unwrap_or(runtime_exe);

    let app_dir = applications_dir()?;
    std::fs::create_dir_all(&app_dir)
        .with_context(|| format!("failed to create {}", app_dir.display()))?;

    let desktop_path = app_dir.join(CAPTURE_DESKTOP_ID);
    let contents = format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name={CAPTURE_DESKTOP_NAME}\n\
         Comment=Capture the focused window and look up Japanese text with InstantJPDict\n\
         Exec={} --capture\n\
         Terminal=false\n\
         StartupNotify=false\n\
         NoDisplay=true\n\
         X-KDE-GlobalAccel-CommandShortcut=true\n\
         {RESTRICTED_INTERFACES_KEY}={KWIN_INTERFACE}\n",
        launch_exe.display(),
    );

    // A cookie-cutter rewrite would be harmless, but skipping identical
    // content avoids waking KWin's service cache watcher on every capture.
    let changed = std::fs::read_to_string(&desktop_path).ok().as_deref() != Some(contents.as_str());
    if changed {
        std::fs::write(&desktop_path, contents)
            .with_context(|| format!("failed to write {}", desktop_path.display()))?;
    }

    // Older builds wrote an authorization-only file under the old name. KWin
    // read that file when the session started and re-reads its contents on a
    // request, so keeping a current alias under the old name makes a running
    // session accept the new binary without a logout. The alias deliberately
    // omits the command-shortcut flag, so it cannot appear twice in System
    // Settings.
    sync_legacy_authorization_alias(&app_dir, &runtime_exe)?;

    // Always refresh KService's cache, even when the content was already
    // correct. Rewriting a file in place leaves its *directory* mtime
    // untouched, and KService's staleness check is directory-based, so a
    // cache built before the first registration survives both re-runs of
    // `--setup-capture` and a logout. That left KWin refusing screenshots of
    // a binary whose desktop file was perfectly in order. The same cache is
    // what makes the command shortcut visible to kglobalaccel.
    if !refresh_service_cache() {
        eprintln!(
            "[capture] warning: could not refresh the KDE service cache \
             (kbuildsycoca6 not found). If KWin keeps refusing, run \
             `kbuildsycoca6 --noincremental` once, or log out and back in."
        );
    }
    Ok(desktop_path)
}

/// Result of [`setup_capture_shortcut`].
#[derive(Debug, Clone)]
pub struct CaptureSetup {
    pub desktop_file: PathBuf,
    /// The active key combination, when one is registered.
    pub shortcut: Option<String>,
    /// Whether the shortcut was installed into the running kglobalaccel.
    pub live: bool,
    /// Extra guidance when automatic registration was not possible.
    pub note: Option<String>,
}

/// Install the KWin authorization and register the capture shortcut in one
/// step. An existing shortcut is preserved; otherwise the default is used (or
/// a fallback when the default is already taken).
pub fn setup_capture_shortcut() -> Result<CaptureSetup> {
    let desktop_file = setup_capture_authorization()?;

    if let Some(existing) = capture_shortcut_status() {
        return match register_live_shortcut(&existing) {
            Ok(true) => Ok(CaptureSetup {
                desktop_file,
                shortcut: Some(existing),
                live: true,
                note: None,
            }),
            Ok(false) => Ok(CaptureSetup {
                desktop_file,
                shortcut: Some(existing),
                live: false,
                note: Some(
                    "The shortcut is saved but could not be applied to the running \
                     session; log out and back in once."
                        .to_string(),
                ),
            }),
            Err(e) => Ok(CaptureSetup {
                desktop_file,
                shortcut: Some(existing),
                live: false,
                note: Some(format!(
                    "The shortcut is saved but could not be applied to the running \
                     session ({e}); log out and back in once."
                )),
            }),
        };
    }

    for combo in [DEFAULT_CAPTURE_SHORTCUT, FALLBACK_CAPTURE_SHORTCUT] {
        match register_live_shortcut(combo) {
            Ok(true) => {
                return Ok(CaptureSetup {
                    desktop_file,
                    shortcut: Some(combo.to_string()),
                    live: true,
                    note: None,
                })
            }
            Ok(false) => continue,
            Err(e) => {
                return Ok(CaptureSetup {
                    desktop_file,
                    shortcut: None,
                    live: false,
                    note: Some(format!(
                        "Could not reach KDE's global-shortcut service ({e}). \
                         Add the shortcut manually in System Settings."
                    )),
                })
            }
        }
    }

    Ok(CaptureSetup {
        desktop_file,
        shortcut: None,
        live: false,
        note: Some(format!(
            "{DEFAULT_CAPTURE_SHORTCUT} and {FALLBACK_CAPTURE_SHORTCUT} are both \
             already taken. Add another key in System Settings."
        )),
    })
}

/// The registered capture shortcut, read from `kglobalshortcutsrc`.
pub fn capture_shortcut_status() -> Option<String> {
    let config = config_dir()?.join("kglobalshortcutsrc");
    let text = std::fs::read_to_string(config).ok()?;
    let mut section = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') && line.ends_with(']') {
            section = line.trim_matches(&['[', ']'][..]).to_string();
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != "_launch" {
            continue;
        }
        if section == format!("services][{CAPTURE_DESKTOP_ID}")
            || section == CAPTURE_DESKTOP_ID
        {
            let combo = value.split(',').next().unwrap_or("").trim();
            if !combo.is_empty() && combo != "none" {
                return Some(combo.to_string());
            }
        }
    }
    None
}

/// The exact command a user pastes into System Settings → Shortcuts.
pub fn capture_command() -> String {
    format!("{} --capture", executable_name())
}

/// Ask kglobalaccel to create the command-shortcut component from the freshly
/// installed desktop file, then bind the key. This is the same handshake the
/// Plasma shortcut KCM performs, so the shortcut starts working immediately
/// instead of waiting for the next login.
fn register_live_shortcut(combo: &str) -> Result<bool> {
    let key = key_sequence(combo)
        .with_context(|| format!("unsupported capture shortcut `{combo}`"))?;

    let conn = zbus::blocking::Connection::session()
        .context("no session D-Bus (is this a KDE Plasma session?)")?;

    // A dummy action makes kglobalaccel parse the desktop file and create its
    // KServiceActionComponent. The empty action is then discarded.
    let action_id = vec![
        CAPTURE_DESKTOP_ID.to_string(),
        CAPTURE_DESKTOP_NAME.to_string(),
        String::new(),
        String::new(),
    ];
    conn.call_method(
        Some(KGLOBALACCEL_BUS_NAME),
        KGLOBALACCEL_OBJECT_PATH,
        Some(KGLOBALACCEL_INTERFACE),
        "doRegister",
        &action_id,
    )
    .context("kglobalaccel did not accept the capture command")?;
    let _ = conn.call_method(
        Some(KGLOBALACCEL_BUS_NAME),
        KGLOBALACCEL_OBJECT_PATH,
        Some(KGLOBALACCEL_INTERFACE),
        "unregister",
        &(CAPTURE_DESKTOP_ID, ""),
    );

    let action = vec![
        CAPTURE_DESKTOP_ID.to_string(),
        "_launch".to_string(),
        CAPTURE_DESKTOP_NAME.to_string(),
        CAPTURE_DESKTOP_NAME.to_string(),
    ];
    let reply = conn
        .call_method(
            Some(KGLOBALACCEL_BUS_NAME),
            KGLOBALACCEL_OBJECT_PATH,
            Some(KGLOBALACCEL_INTERFACE),
            "setShortcut",
            &(action, vec![key], KGLOBALACCEL_SET_PRESENT),
        )
        .context("kglobalaccel rejected the shortcut")?;
    let accepted: Vec<i32> = reply.body().deserialize().unwrap_or_default();
    Ok(accepted.first().copied() == Some(key))
}

/// Convert `Meta+Print` / `Ctrl+Alt+S` into Qt's combined key code, which is
/// what kglobalaccel's D-Bus API expects.
fn key_sequence(combo: &str) -> Option<i32> {
    const SHIFT: i32 = 0x0200_0000;
    const CTRL: i32 = 0x0400_0000;
    const ALT: i32 = 0x0800_0000;
    const META: i32 = 0x1000_0000;
    const KEY_PRINT: i32 = 0x0100_0009;

    let mut modifiers = 0;
    let mut key = None;
    for part in combo.split('+') {
        let part = part.trim().to_ascii_lowercase();
        match part.as_str() {
            "shift" => modifiers |= SHIFT,
            "ctrl" | "control" => modifiers |= CTRL,
            "alt" => modifiers |= ALT,
            "meta" | "super" | "win" => modifiers |= META,
            "print" | "printscreen" => key = Some(KEY_PRINT),
            other if other.chars().count() == 1 => {
                key = Some(other.chars().next()?.to_ascii_uppercase() as i32);
            }
            _ => return None,
        }
    }
    key.map(|key| modifiers | key)
}

fn config_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return Some(PathBuf::from(xdg));
        }
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
}

/// Keep the pre-rename authorization file in step with the new one.
///
/// A running KWin has already scanned the desktop files that existed when it
/// started; it re-reads those files for each capture request. Updating the
/// legacy filename therefore authorizes a running session immediately, while
/// a fresh install still needs one logout/login before KWin scans the new
/// canonical file. Only files that identify themselves as InstantJPDict's are
/// rewritten.
fn sync_legacy_authorization_alias(app_dir: &Path, exe: &Path) -> Result<()> {
    let path = app_dir.join(LEGACY_DESKTOP_ID);
    if let Ok(existing) = std::fs::read_to_string(&path) {
        let ours = existing.contains(RESTRICTED_INTERFACES_KEY)
            || existing.contains("Name=InstantJPDict");
        if !ours {
            return Ok(());
        }
    }

    let contents = format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=InstantJPDict Capture (compatibility)\n\
         Comment=Compatibility alias that authorizes KWin screenshots for InstantJPDict\n\
         Exec={} --capture\n\
         Terminal=false\n\
         StartupNotify=false\n\
         NoDisplay=true\n\
         {RESTRICTED_INTERFACES_KEY}={KWIN_INTERFACE}\n",
        exe.display(),
    );
    let changed = std::fs::read_to_string(&path).ok().as_deref() != Some(contents.as_str());
    if changed {
        std::fs::write(&path, contents)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::key_sequence;

    #[test]
    fn default_shortcut_maps_to_the_qt_meta_print_code() {
        assert_eq!(key_sequence("Meta+Print"), Some(0x1100_0009));
    }

    #[test]
    fn fallback_shortcut_maps_to_ctrl_alt_s() {
        assert_eq!(key_sequence("Ctrl+Alt+S"), Some(0x0C00_0053));
    }

    #[test]
    fn unknown_shortcuts_are_rejected() {
        assert_eq!(key_sequence("Hyper+F13"), None);
        assert_eq!(key_sequence("Meta"), None);
    }
}

/// Where user-level .desktop files live (`$XDG_DATA_HOME/applications`, or the
/// conventional `~/.local/share/applications`).
fn applications_dir() -> Result<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
        if !xdg.is_empty() {
            return Ok(PathBuf::from(xdg).join("applications"));
        }
    }
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("applications"))
}

/// Ask ksycoca to rebuild so KWin sees the desktop file immediately.
///
/// This must run even when the desktop file content did not change: KService
/// decides a cache is valid from directory timestamps, and an in-place
/// rewrite does not update them. Returns true when a rebuild command
/// succeeded; missing tools are not fatal (KWin also picks a *new* file up
/// once its watcher fires, since creating one changes the directory mtime).
fn refresh_service_cache() -> bool {
    for tool in [
        "kbuildsycoca6",
        "/usr/bin/kbuildsycoca6",
        "kbuildsycoca5",
        "/usr/bin/kbuildsycoca5",
    ] {
        // AppImage sets LD_LIBRARY_PATH/APPDIR/APPIMAGE for the packaged
        // process; the host's kbuildsycoca must not inherit those or it fails
        // to load the host Qt libraries.
        let spawned = std::process::Command::new(tool)
            .arg("--noincremental")
            .env_remove("LD_LIBRARY_PATH")
            .env_remove("APPDIR")
            .env_remove("APPIMAGE")
            .env_remove("ARGV0")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        match spawned {
            Ok(status) if status.success() => return true,
            Ok(_) | Err(_) => continue,
        }
    }
    false
}

/// The path the user's shortcut should launch.
///
/// Inside an AppImage this is the original `.AppImage` path (the runtime
/// exposes it as `$APPIMAGE`), not the transient mount path.
fn launch_path() -> PathBuf {
    if let Some(appimage) = std::env::var_os("APPIMAGE") {
        if !appimage.is_empty() {
            return PathBuf::from(appimage);
        }
    }
    runtime_path()
}

/// The executable that is actually running: the AppImage runtime uses this
/// for KWin's authorization check, and the packaged AppRun keeps it stable at
/// `~/.local/share/instantjpdict/bin`.
fn runtime_path() -> PathBuf {
    std::env::current_exe().unwrap_or_else(|_| PathBuf::from("instantjpdict"))
}

fn executable_name() -> String {
    launch_path().display().to_string()
}

/// Human-readable instructions for wiring up the global shortcut.
pub fn shortcut_instructions() -> String {
    let command = capture_command();
    format!(
        "To use a different key (KDE Plasma):\n\
         \x20 System Settings → Keyboard → Shortcuts → Add → Command or Script\n\
         \x20 Command: {command}\n\
         \x20 (older Plasma: Custom Shortcuts → Edit → New Global Shortcut → Command/URL)\n\
         \n\
         Suggested keys: {DEFAULT_CAPTURE_SHORTCUT}, or {FALLBACK_CAPTURE_SHORTCUT}.\n\
         The Settings window has a one-click setup button and a “copy command”\n\
         button for this same step.\n\
         \n\
         Pressing the shortcut captures the focused window and opens the OCR\n\
         overlay on it directly — nothing is written to the screenshot folder,\n\
         so no second viewer opens for the same image."
    )
}
