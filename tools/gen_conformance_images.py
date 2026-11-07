#!/usr/bin/env python3
"""Generate the tiny synthetic input images for the conformance corpus.

Pure stdlib (struct/zlib): no dependencies, deterministic output.
White background (255), near-black bars (0). Each image is a stand-in for a
pipeline input class named in the ticket: plain horizontal text lines,
tategaki columns, and a horizontal line with a ruby strip above it.

Ruby-strip geometry is chosen against the checked-in furigana rule
(src/furigana.rs) on a 640x200 canvas, main bar x 60..580 y 120..150,
ruby x 200..330 y 96..108:
  thinness  12 < 30 * 0.75  (THIN_RATIO)
  width    130 < 520 * 0.65  (HWIDTH_RATIO, on unclipped boxes)
  height     12 < 200 * 0.12 (MAX_FRAC)
  bigness   520 >= 640 * 0.2 (BIG_MIN_FRAC)
  gap        12 <= 30 * 0.5  (GAP_RATIO)
Usage: python3 tools/gen_conformance_images.py  (writes tests/conformance/images/)
"""
import os
import struct
import zlib

ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests", "conformance", "images"
)


def write_gray(path, w, h, rects):
    px = bytearray([255]) * (w * h)
    for (x0, y0, x1, y1) in rects:
        for y in range(max(0, y0), min(h, y1)):
            for x in range(max(0, x0), min(w, x1)):
                px[y * w + x] = 0
    raw = b"".join(b"\x00" + bytes(px[y * w : (y + 1) * w]) for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)
    print("wrote", path, w, "x", h)


os.makedirs(ROOT, exist_ok=True)
# Three horizontal text-line bars.
write_gray(os.path.join(ROOT, "h-bars.png"), 320, 180, [
    (30, 20, 290, 34), (30, 80, 290, 94), (30, 140, 290, 154),
])
# Two tategaki columns (right column must sort first).
write_gray(os.path.join(ROOT, "v-columns.png"), 220, 320, [
    (30, 20, 50, 300), (120, 20, 140, 300),
])
# One horizontal line with a ruby strip above it (ruby must be dropped,
# the main line kept, when the furigana filter is on).
write_gray(os.path.join(ROOT, "ruby-h.png"), 640, 200, [
    (60, 120, 580, 150), (200, 96, 330, 108),
])
