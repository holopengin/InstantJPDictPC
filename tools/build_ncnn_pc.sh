#!/usr/bin/env bash
# Build the vgf89/ncnn fork for the PC (Linux) app.
#
# This is the desktop sibling of the mobile app's tools/build_ncnn.sh: it
# pins the SAME fork commit the Android build ships (FORK_PIN below must be
# bumped in lockstep with InstantJPDict's script), enforces the same
# required-patch markers, and produces a CPU-only static libncnn.a plus
# headers that Cargo's build.rs links.
#
# Output:
#   <out>/build-host/src/libncnn.a   raw archive
#   <out>/install/{lib,include}      installed tree (what build.rs consumes)
#
# Requirements: cmake >= 3.22, a C++17 compiler. First run clones the fork.
#
# Usage:
#   tools/build_ncnn_pc.sh [--out DIR] [--src DIR] [--allow-dirty] [--jobs N]
#
#   --src uses an existing ncnn tree as-is (for testing fork changes); a
#   dirty tree is refused unless --allow-dirty. Markers are verified either
#   way. Without --src the tree at <out>/src is reset to FORK_PIN.
set -euo pipefail

FORK=https://github.com/vgf89/ncnn.git
FORK_PIN=a2b8507f6a3449e80f1c2585a127e3c08e3a9f3f # keep equal to InstantJPDict tools/build_ncnn.sh FORK_PIN
HERE=$(cd "$(dirname "$0")/.." && pwd)

OUT="$HERE/third_party/ncnn-pc"
SRC_OVERRIDE=""
ALLOW_DIRTY=0
JOBS="$(nproc)"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --src) SRC_OVERRIDE="$2"; shift 2;;
    --allow-dirty) ALLOW_DIRTY=1; shift;;
    --jobs) JOBS="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [ -n "$SRC_OVERRIDE" ]; then
  SRC="$SRC_OVERRIDE"
  [ -d "$SRC/src/layer" ] || { echo "ERROR: $SRC is not an ncnn tree" >&2; exit 1; }
  if [ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ]; then
    if [ "$ALLOW_DIRTY" -eq 0 ]; then
      echo "ERROR: $SRC is dirty; refusing (pass --allow-dirty to test local changes)" >&2
      exit 1
    fi
    echo "WARNING: building from dirty tree $SRC" >&2
  fi
  echo "tree: $(git -C "$SRC" log --oneline -1) (override, no checkout)"
else
  SRC="$OUT/src"
  if [ ! -d "$SRC" ]; then
    git clone "$FORK" "$SRC"
  fi
  git -C "$SRC" fetch --quiet origin "$FORK_PIN" 2>/dev/null || true
  git -C "$SRC" checkout --quiet "$FORK_PIN"
  git -C "$SRC" reset --quiet --hard "$FORK_PIN"
  echo "tree: $(git -C "$SRC" log --oneline -1)"
fi

# Marker gate: the same fork patches the shipped models depend on. Without
# these, int8 rec/det inference produces garbage or crashes; fail loudly.
require_marker() {
  local token="$1" desc="$2"
  if grep -rqF "$token" "$SRC/src/layer"; then
    echo "marker ok: $desc"
  else
    echo "ERROR: missing '$token' ($desc) in $SRC" >&2
    exit 1
  fi
}
require_marker "bottom_blob_3d = bottom_blob_unbordered" "int8 1x1 on flattened 1D (SE)"
require_marker "int8_scale_term % 100" "depthwise int8 scale 201/202"
require_marker "activation_ss" "fused GELU type 7 (rec 9=7)"

BUILD="$OUT/build-host"
SRC_ABS=$(cd "$SRC" && pwd -P)
# A build dir carries an absolute source path in its CMake cache; reusing one
# from a different --src tree fails deep inside CMake. Reset it instead.
if [ -f "$BUILD/CMakeCache.txt" ] && \
   ! grep -qF "CMAKE_HOME_DIRECTORY:INTERNAL=$SRC_ABS" "$BUILD/CMakeCache.txt"; then
  echo "build dir $BUILD was configured for another source tree; resetting" >&2
  rm -rf "$BUILD"
fi
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=OFF
cmake --build "$BUILD" -j"$JOBS" --target ncnn

# Artifact gate: on ARM the mobile build greps libncnn.a for `activation_ss`,
# but on x86 that helper is `static NCNN_FORCEINLINE` and is inlined away, so
# no such symbol survives. Require instead that the archive was produced from
# the patched headers (it must be newer than them), so a stale build dir can
# never be linked silently.
if [ "$BUILD/src/libncnn.a" -nt "$SRC/src/layer/fused_activation.h" ] && \
   [ "$BUILD/src/libncnn.a" -nt "$SRC/src/layer/x86/x86_activation.h" ]; then
  echo "artifact ok: libncnn.a newer than fused-activation headers"
else
  echo "ERROR: libncnn.a is older than the fused-activation headers" >&2
  exit 1
fi

cmake --install "$BUILD" --prefix "$OUT/install"
echo "ncnn -> $OUT/install"
echo OK
