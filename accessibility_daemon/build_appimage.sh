#!/bin/sh
# Build the AppImage.
#
# This used to bundle through quick-sharun (Anylinux-AppImages). That path is
# no longer viable here:
#
#   * quick-sharun's `sharun` launcher segfaults on dispatch (its cross-libc
#     re-exec through the bundled ld-linux/libc in AppDir/lib) — reproduced on
#     both Bazzite and Arch, with the repo-pinned sharun 2.2.4 and with
#     current main;
#   * its helper source `anylinux.c` now 404s upstream, so the pinned
#     quick-sharun cannot even assemble an AppDir;
#   * it swept ~1.6 GB of unrelated system data (share/code, share/fonts,
#     share/unicode) into the bundle, for a 470 MB artifact.
#
# None of that is needed: the binary links only system libraries (libc,
# libstdc++, libgomp, libm, libmvec, libgcc_s) and dlopens its graphics
# backends at runtime, so a plain AppRun plus appimagetool produces a working
# AppImage roughly ten times smaller. Verified on Bazzite (Fedora) and Arch.
#
# Because nothing is bundled, the AppImage relies on the host having a
# compatible graphics stack (X11/Wayland/Vulkan/GL). The quick-sharun recipe
# is in git history if a self-contained bundle is ever wanted back.
set -eu

HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$HERE"

ARCH="$(uname -m)"
OUTPATH="${OUTPATH:-$HERE/dist}"
OUTNAME="${OUTNAME:-InstantJPDict-$ARCH.AppImage}"

# Pinned packaging tool: version AND checksum, so an upstream change cannot
# silently alter the artifact the way the unpinned quick-sharun download did.
AT_VERSION=0.5.1
AT_URL="https://github.com/pkgforge-dev/appimagetool/releases/download/$AT_VERSION/appimagetool-full-x86_64-linux"
AT_SHA256=6025afd9d452360ffe84e5cc4e4e7d029a2d500b893f058de1ff396aacee1d79
AT_BIN="${APPIMAGETOOL:-$HERE/target/appimagetool-$AT_VERSION}"

# Shared PP-OCRv6 backend: build the pinned ncnn fork first (static libncnn.a).
if [ ! -f third_party/ncnn-pc/install/lib/libncnn.a ]; then
	./tools/build_ncnn_pc.sh
fi

cargo build --release

if [ ! -x "$AT_BIN" ]; then
	echo "Fetching appimagetool $AT_VERSION..."
	mkdir -p "$(dirname -- "$AT_BIN")"
	if command -v curl >/dev/null 2>&1; then
		curl -sSL -o "$AT_BIN" "$AT_URL"
	else
		wget -qO "$AT_BIN" "$AT_URL"
	fi
	chmod +x "$AT_BIN"
fi
echo "$AT_SHA256  $AT_BIN" | sha256sum -c - >/dev/null

# Assemble the AppDir. `assets/` and `fonts/` must sit next to the binary —
# it resolves both relative to its own location, which is the same layout
# core/build.rs stages under target/{profile}/.
APPDIR="$HERE/target/AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR"
cp target/release/accessibility_daemon "$APPDIR/"
cp -r target/release/assets "$APPDIR/"
cp -r target/release/fonts "$APPDIR/"

# Plain launcher: no sharun, no bundled loader, no cross-libc re-exec.
cat >"$APPDIR/AppRun" <<'APPRUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export APPDIR="$HERE"
exec "$HERE/accessibility_daemon" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

ICON="${ICON:-$HERE/icon.png}"
DESKTOP="${DESKTOP:-$HERE/myapp.desktop}"
cp "$ICON" "$APPDIR/myapp.png"
cp "$ICON" "$APPDIR/.DirIcon"
cp "$DESKTOP" "$APPDIR/myapp.desktop"

mkdir -p "$OUTPATH"
ARCH="$ARCH" "$AT_BIN" "$APPDIR" -o "$OUTPATH" -n "$OUTNAME"

echo "AppImage at: $OUTPATH/$OUTNAME"
