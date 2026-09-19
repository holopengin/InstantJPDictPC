#!/usr/bin/env bash
# capture_and_ocr.sh — Screenshot the focused window and OCR it immediately.
#
# Unlike watch_and_ocr.sh, this does not go through the screenshot folder:
# the daemon (accessibility_daemon --capture) asks KWin for the focused
# window's pixels over D-Bus and hands them straight to the OCR viewer, so
# nothing lands in a watched directory and no second viewer is spawned.
#
# One-time setup:
#   accessibility_daemon/target/release/accessibility_daemon --setup-capture
#
# Then bind this script to a global shortcut in KDE Plasma:
#   System Settings → Keyboard → Shortcuts → Add → Command or Script
#   Command: /path/to/repo/screenshot/capture_and_ocr.sh
#
# All extra arguments are forwarded to the daemon's OCR viewer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DAEMON_BIN="${REPO_ROOT}/accessibility_daemon/target/release/accessibility_daemon"

if [[ ! -x "${DAEMON_BIN}" ]]; then
    echo "ERROR: Daemon binary not found at ${DAEMON_BIN}" >&2
    echo "Build it first with: cd ${REPO_ROOT}/accessibility_daemon && cargo build --release" >&2
    exit 1
fi

exec "${DAEMON_BIN}" --capture "$@"
