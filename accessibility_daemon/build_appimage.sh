#!/bin/sh
set -eux

ARCH="$(uname -m)"
SHARUN="https://raw.githubusercontent.com/pkgforge-dev/Anylinux-AppImages/refs/heads/main/useful-tools/quick-sharun.sh"

# Configure the AppImage
export ICON=/usr/share/icons/AdwaitaLegacy/48x48/legacy/utilities-terminal.png
export DESKTOP=myapp.desktop
export OUTPATH=./dist
export OUTNAME=InstantJPDict-"$ARCH".AppImage

unset WAYLAND_DISPLAY

# Install your application (example using pacman)
cargo build --release

# Download and run quick-sharun
wget "$SHARUN" -O ./quick-sharun
chmod +x ./quick-sharun

rm -rf AppDir

# Bundle the application
./quick-sharun ./target/release/accessibility_daemon

# Create the AppImage
./quick-sharun --make-appimage
