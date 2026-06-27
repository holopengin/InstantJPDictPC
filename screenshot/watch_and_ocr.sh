#!/usr/bin/env bash
# watch_and_ocr.sh — Monitor screenshot folder and run OCR on new images
#
# Usage: ./watch_and_ocr.sh [daemon_args...]
#
# Watches ~/Pictures/Screenshots/ (or $SCREENSHOT_DIR) for new image files
# using inotifywait. When a new screenshot appears, passes it to the
# accessibility_daemon for OCR.
#
# All extra arguments are forwarded to the daemon binary.

set -euo pipefail

# Enable nullglob so unmatched globs expand to nothing instead of the literal pattern
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DAEMON_BIN="${REPO_ROOT}/accessibility_daemon/target/release/accessibility_daemon"
SCREENSHOT_DIR="${SCREENSHOT_DIR:-${HOME}/Pictures/Screenshots}"
GAMESCOPE_DIR="/tmp"
PROCESSED_DIR="${SCREENSHOT_DIR}/.ocr_processed"

# Minimum age (seconds) before processing — waits for the file to be fully written
MIN_AGE=1

mkdir -p "${SCREENSHOT_DIR}" "${PROCESSED_DIR}"

if [[ ! -f "${DAEMON_BIN}" ]]; then
    echo "ERROR: Daemon binary not found at ${DAEMON_BIN}" >&2
    echo "Build it first with: cd $(dirname "${DAEMON_BIN}") && cargo build --release" >&2
    exit 1
fi

echo "Watching ${SCREENSHOT_DIR} and ${GAMESCOPE_DIR}/gamescope_*.png for new screenshots..."
echo "Daemon: ${DAEMON_BIN}"
echo "Extra args: $*"
echo ""
echo "Take a screenshot now (e.g. Steam+Screenshot, or Print Screen)"
echo "and it will be processed automatically."
echo "Press Ctrl+C to stop."
echo ""

# Process any existing unprocessed screenshots first
process_file() {
    local file="$1"
    local basename
    basename="$(basename "${file}")"

    # Skip already processed files
    if [[ -f "${PROCESSED_DIR}/${basename}" ]]; then
        return 0
    fi

    # Wait for file to be fully written (size stabilizes)
    local size1 size2
    size1=$(stat -c%s "${file}" 2>/dev/null || echo 0)
    sleep "${MIN_AGE}"
    size2=$(stat -c%s "${file}" 2>/dev/null || echo 0)

    if [[ "${size1}" != "${size2}" ]]; then
        # Still being written, wait more
        sleep 1
    fi

    # Only process image files
    if ! file --mime-type "${file}" 2>/dev/null | grep -q "image/"; then
        touch "${PROCESSED_DIR}/${basename}"
        return 0
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "New screenshot detected: ${basename}"
    echo "  Size: $(du -h "${file}" | cut -f1)"
    echo "  Running OCR..."

    # Run from the daemon's directory so it can find ./assets/
    if (cd "$(dirname "${DAEMON_BIN}")" && "${DAEMON_BIN}" "${file}" "$@") 2>&1; then
        echo "  OCR complete."
    else
        echo "  OCR exited with code $?."
    fi

    # Mark as processed
    touch "${PROCESSED_DIR}/${basename}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Process existing screenshots from both locations
for file in "${SCREENSHOT_DIR}"/*.{png,jpg,jpeg,bmp,webp} "${GAMESCOPE_DIR}"/gamescope_*.png; do
    [[ -f "${file}" ]] || continue
    process_file "${file}"
done

# Watch both directories for new files using inotifywait
if command -v inotifywait &>/dev/null; then
    inotifywait -m -e close_write -e moved_to \
        --format '%w%f' \
        "${SCREENSHOT_DIR}" "${GAMESCOPE_DIR}" 2>/dev/null | \
    while read -r filepath; do
        [[ -f "${filepath}" ]] || continue
        # Only process gamescope_*.png from /tmp, or images from Screenshots dir
        if [[ "${filepath}" == "${GAMESCOPE_DIR}"/gamescope_*.png ]] || \
           [[ "${filepath}" =~ \.(png|jpg|jpeg|bmp|webp)$ ]]; then
            process_file "${filepath}"
        fi
    done || true
else
    # Fallback: poll every 2 seconds
    echo "inotifywait not found, falling back to polling..."
    while true; do
        for file in "${SCREENSHOT_DIR}"/*.{png,jpg,jpeg,bmp,webp} "${GAMESCOPE_DIR}"/gamescope_*.png; do
            [[ -f "${file}" ]] || continue
            basename="$(basename "${file}")"
            [[ -f "${PROCESSED_DIR}/${basename}" ]] && continue
            process_file "${file}"
        done
        sleep 2
    done
fi
