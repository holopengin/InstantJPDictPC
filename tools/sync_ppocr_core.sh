#!/usr/bin/env bash
# Keep the shared OCR backend in lockstep with the Android app.
#
# The Android app and this repo compile the same native cores and ship the
# same model/data assets:
#   InstantJPDict:      app/src/main/cpp/{ppocr_ncnn_core,kana_size_core}.{h,cpp}
#   InstantJPDictPC:    native/ppocr_ncnn/ppocr_ncnn_core.{h,cpp}
#                       native/kana_size/kana_size_core.{h,cpp}
# The JNI shells and the C ABI shells are platform shells and are allowed to
# differ.
#
# Model assets and shared pipeline data are mirrored byte-for-byte too:
#   PP-OCRv6_small_ncnn/*, kana_size/*, deinflect.json, catalog,
#   components, lm, and the shared kanji-variant table.
#
# Usage:
#   tools/sync_ppocr_core.sh --check  /path/to/InstantJPDict   # CI / pre-commit
#   tools/sync_ppocr_core.sh --update /path/to/InstantJPDict   # mirror mobile -> PC
#
# A core/asset change must land in the Android repo first; PC mirrors it,
# never the other way around.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
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

# mobile-relative source | PC-relative destination
FILES=(
  "app/src/main/cpp/ppocr_ncnn_core.h|native/ppocr_ncnn/ppocr_ncnn_core.h"
  "app/src/main/cpp/ppocr_ncnn_core.cpp|native/ppocr_ncnn/ppocr_ncnn_core.cpp"
  "app/src/main/cpp/kana_size_core.h|native/kana_size/kana_size_core.h"
  "app/src/main/cpp/kana_size_core.cpp|native/kana_size/kana_size_core.cpp"

  "app/src/main/assets/PP-OCRv6_small_ncnn/PROVENANCE.txt|assets/PP-OCRv6_small_ncnn/PROVENANCE.txt"
  "app/src/main/assets/PP-OCRv6_small_ncnn/det.param|assets/PP-OCRv6_small_ncnn/det.param"
  "app/src/main/assets/PP-OCRv6_small_ncnn/det.bin|assets/PP-OCRv6_small_ncnn/det.bin"
  "app/src/main/assets/PP-OCRv6_small_ncnn/rec_dyn.param|assets/PP-OCRv6_small_ncnn/rec_dyn.param"
  "app/src/main/assets/PP-OCRv6_small_ncnn/rec_dyn.bin|assets/PP-OCRv6_small_ncnn/rec_dyn.bin"
  "app/src/main/assets/PP-OCRv6_small_ncnn/vocab.json|assets/PP-OCRv6_small_ncnn/vocab.json"
  "app/src/main/assets/PP-OCRv6_small_ncnn/rec_remap.txt|assets/PP-OCRv6_small_ncnn/rec_remap.txt"

  "app/src/main/assets/kana_size/PROVENANCE.txt|assets/kana_size/PROVENANCE.txt"
  "app/src/main/assets/kana_size/nb_all.param|assets/kana_size/nb_all.param"
  "app/src/main/assets/kana_size/nb_all.bin|assets/kana_size/nb_all.bin"

  "app/src/main/assets/deinflect.json|assets/deinflect.json"
  "app/src/main/assets/catalog/dictionaries.json|assets/catalog/dictionaries.json"
  "app/src/main/assets/components/krad_components.txt|assets/components/krad_components.txt"
  "app/src/main/assets/components/PROVENANCE.txt|assets/components/PROVENANCE.txt"
  "app/src/main/assets/lm/char_lm.bin|assets/lm/char_lm.bin"
  "app/src/main/assets/lm/PROVENANCE.txt|assets/lm/PROVENANCE.txt"
  "app/src/main/assets/variants/kanji_variants.txt|assets/variants/kanji_variants.txt"
  "app/src/main/assets/variants/PROVENANCE.txt|assets/variants/PROVENANCE.txt"

  # Shared test fixture: the Android conformance runner resolves
  # `definitions_ref` cases through this copy.
  "app/src/test/resources/jitendex/entries.json|tests/data/jitendex/entries.json"
)

status=0
for entry in "${FILES[@]}"; do
  IFS='|' read -r src dst <<< "$entry"
  src_path="$MOBILE/$src"
  dst_path="$HERE/$dst"

  if [ ! -f "$src_path" ]; then
    echo "ERROR: missing mobile file: $src_path" >&2
    status=1
    continue
  fi

  if [ "$MODE" = update ]; then
    mkdir -p "$(dirname "$dst_path")"
    cp "$src_path" "$dst_path"
    echo "copied $dst"
  else
    if [ ! -f "$dst_path" ]; then
      echo "MISSING: $dst (PC copy)" >&2
      status=1
    elif ! cmp -s "$src_path" "$dst_path"; then
      echo "DIFFERS: $dst (mobile $src)" >&2
      status=1
    else
      echo "ok: $dst"
    fi
  fi
done

if [ "$status" != 0 ]; then
  if [ "$MODE" = check ]; then
    echo "Shared OCR backend is out of sync; run: $0 --update $MOBILE" >&2
  fi
  exit 1
fi
