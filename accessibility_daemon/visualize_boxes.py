#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["onnxruntime", "numpy", "Pillow"]
# ///

import json
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

img_path = "/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/test_image.png"
img = Image.open(img_path)
orig_w, orig_h = img.size

# ========== ORT inference ==========
sess = ort.InferenceSession("/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/assets/meiki.text.detect.v0.1.960x544.onnx")
img_resized = img.resize((960, 544), Image.BILINEAR)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_array = np.array(img_resized, dtype=np.float32) / 255.0
img_array = (img_array - mean) / std
img_tensor = img_array.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

# Pass [960, 544] - the format the model expects
ts = np.array([[960, 544]], dtype=np.int64)
ort_outputs = sess.run(None, {"images": img_tensor, "orig_target_sizes": ts})
ort_boxes = ort_outputs[1][0]
ort_scores = ort_outputs[2][0]

# Scale from model input space to original image space
scale_x = orig_w / 960.0
scale_y = orig_h / 544.0

# ========== Load Burn results ==========
with open("/tmp/burn_results.json") as f:
    burn_data = json.load(f)

burn_scores = burn_data["scores"]
burn_boxes_flat = burn_data["boxes"]

# ========== Draw boxes ==========
result_img = img.copy()
draw = ImageDraw.Draw(result_img)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except:
    font = ImageFont.load_default()

# Draw ORT boxes in blue (scaled)
ort_count = 0
for i in range(len(ort_scores)):
    if ort_scores[i] < 0.3:
        continue
    ort_count += 1
    b = ort_boxes[i]
    x1 = max(0, min(orig_w, b[0] * scale_x))
    y1 = max(0, min(orig_h, b[1] * scale_y))
    x2 = max(0, min(orig_w, b[2] * scale_x))
    y2 = max(0, min(orig_h, b[3] * scale_y))
    draw.rectangle([x1, y1, x2, y2], outline=(0, 100, 255), width=3)

# Draw Burn boxes in red (already scaled in Rust)
burn_count = 0
for i in range(len(burn_scores)):
    if burn_scores[i] < 0.3:
        continue
    burn_count += 1
    x1 = max(0, min(orig_w, burn_boxes_flat[i * 4]))
    y1 = max(0, min(orig_h, burn_boxes_flat[i * 4 + 1]))
    x2 = max(0, min(orig_w, burn_boxes_flat[i * 4 + 2]))
    y2 = max(0, min(orig_h, burn_boxes_flat[i * 4 + 3]))
    draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=3)

# Legend
lx, ly = 10, 10
draw.rectangle([lx, ly, lx + 300, ly + 55], fill=(0, 0, 0, 180))
draw.rectangle([lx + 5, ly + 8, lx + 25, ly + 24], outline=(0, 100, 255), width=2)
draw.text((lx + 30, ly + 5), f"ORT  ({ort_count} dets)", fill=(100, 180, 255), font=font)
draw.rectangle([lx + 5, ly + 30, lx + 25, ly + 46], outline=(255, 50, 50), width=2)
draw.text((lx + 30, ly + 28), f"Burn ({burn_count} dets)", fill=(255, 100, 100), font=font)

output_path = "/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/test_image_comparison.png"
result_img.save(output_path)
print(f"Saved to {output_path}")
print(f"ORT: {ort_count} detections")
print(f"Burn: {burn_count} detections")

# Stats
good = ort_scores > 0.3
gb = ort_boxes[good] * [scale_x, scale_y, scale_x, scale_y]
widths = gb[:, 2] - gb[:, 0]
heights = gb[:, 3] - gb[:, 1]
print(f"\nORT: x=[{gb[:,0].min():.0f},{gb[:,2].max():.0f}] y=[{gb[:,1].min():.0f},{gb[:,3].max():.0f}]")
print(f"     w={widths.mean():.1f} h={heights.mean():.1f} aspect={(widths/heights).mean():.2f}")
