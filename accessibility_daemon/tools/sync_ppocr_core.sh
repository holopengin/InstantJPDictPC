#!/usr/bin/env bash
# Keep the shared PP-OCRv6 ncnn core in lockstep with the Android app.
#
# The Android app and this repo compile the same two files:
#   InstantJPDict:      app/src/main/cpp/ppocr_ncnn_core.{h,cpp}
#   InstantJPDictPC: accessibility_daemon/native/ppocr_ncnn/ppocr_ncnn_core.{h,cpp}
# The JNI wrapper (ncnn_jni.cpp) and the C ABI wrapper (ppocr_ncnn_capi.cpp)
# are platform shells and are allowed to differ.
#
# Usage:
#   tools/sync_ppocr_core.sh --check  /path/to/InstantJPDict   # CI / pre-commit
#   tools/sync_ppocr_core.sh --update /path/to/InstantJPDict   # mirror mobile -> PC
#
# A core fix must land in the Android repo first; PC mirrors it, never the
# other way around.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
PC_DIR="$HERE/native/ppocr_ncnn"
MODE=""
MOBILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE=check; shift;;
    --update) MODE=update; shift;;
    -*) echo "unknown arg: $1" >&2; exit 1;;
    *) MOBILE="$1"; shift;;
  esac
done
[ -n "$MODE" ] && [ -n "$MOBILE" ] || {
  echo "usage: $0 --check|--update /path/to/InstantJPDict" >&2
  exit 1
}
MOBILE_DIR="$MOBILE/app/src/main/cpp"
[ -f "$MOBILE_DIR/ppocr_ncnn_core.cpp" ] || {
  echo "ERROR: $MOBILE_DIR/ppocr_ncnn_core.cpp not found — is that the InstantJPDict checkout?" >&2
  exit 1
}

status=0
for f in ppocr_ncnn_core.h ppocr_ncnn_core.cpp; do
  if [ "$MODE" = update ]; then
    cp "$MOBILE_DIR/$f" "$PC_DIR/$f"
    echo "copied $f"
  else
    if ! diff -q "$MOBILE_DIR/$f" "$PC_DIR/$f" >/dev/null; then
      echo "DIFFERS: $f (mobile $MOBILE_DIR vs PC $PC_DIR)" >&2
      status=1
    else
      echo "ok: $f"
    fi
  fi
done
[ "$status" = 0 ] || {
  echo "PC core is out of sync; run: $0 --update $MOBILE" >&2
  exit 1
}
