#!/usr/bin/env python3
"""Compare ORT and Burn detection - check tensor data matches."""
import numpy as np
from PIL import Image

img = Image.open('test_image.png')
orig_w, orig_h = img.width, img.height

scale_x = 960.0 / orig_w
scale_y = 544.0 / orig_h
scale = min(scale_x, scale_y)
w_resized = int(orig_w * scale)
h_resized = int(orig_h * scale)

img_resized = img.resize((w_resized, h_resized), Image.BILINEAR)
img_arr = np.array(img_resized).astype(np.float32) / 255.0

padded = np.zeros((544, 960, 3), dtype=np.float32)
padded[:h_resized, :w_resized, :] = img_arr

tensor = np.transpose(padded, (2, 0, 1))
tensor = tensor[np.newaxis, ...]
tensor = np.ascontiguousarray(tensor)

# Check where non-zero data starts in R channel
r_flat = tensor[0, 0, :, :].flatten()
nonzero_indices = np.nonzero(r_flat)[0]
print(f"R channel: {len(nonzero_indices)} non-zero values out of {len(r_flat)}")
if len(nonzero_indices) > 0:
    print(f"First non-zero at index {nonzero_indices[0]}: value={r_flat[nonzero_indices[0]]:.6f}")
    # Convert flat index to (y, x)
    y = nonzero_indices[0] // 960
    x = nonzero_indices[0] % 960
    print(f"  -> y={y}, x={x}")
    print(f"  -> Expected: content starts at row 0 (top-left padding)")

# Check the resized image directly
print(f"\nResized image shape: {img_arr.shape}")
print(f"Resized image first pixel: {img_arr[0, 0, :]}")
print(f"Resized image pixel at [0, 100]: {img_arr[0, 100, :]}")
nonzero_resized = np.nonzero(img_arr[:, :, 0])[0]
print(f"Resized R channel first non-zero flat index: {nonzero_resized[0] if len(nonzero_resized) > 0 else 'none'}")

# Check what the padded image looks like
print(f"\nPadded image shape: {padded.shape}")
print(f"Padded [0,0]: {padded[0, 0, :]}")
print(f"Padded [0,100]: {padded[0, 100, :]}")

# The key question: does the resized image have content at x=0?
# Or are there black bars?
print(f"\nResized image column 0: min={img_arr[:, 0, :].min():.4f} max={img_arr[:, 0, :].max():.4f}")
print(f"Resized image column 100: min={img_arr[:, 100, :].min():.4f} max={img_arr[:, 100, :].max():.4f}")
print(f"Resized image column 200: min={img_arr[:, 200, :].min():.4f} max={img_arr[:, 200, :].max():.4f}")

# Check the original image
orig_arr = np.array(img)
print(f"\nOriginal image shape: {orig_arr.shape}")
print(f"Original image column 0: {orig_arr[0, 0, :]}")
print(f"Original image column 400: {orig_arr[0, 400, :]}")
print(f"Original image column 800: {orig_arr[0, 800, :]}")
