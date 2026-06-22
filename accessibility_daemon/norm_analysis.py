#!/usr/bin/env python3
"""Compare normalized box coordinates before scaling to find error source."""
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

def preprocess(img_path):
    img = Image.open(img_path)
    orig_w, orig_h = img.size
    img_resized = img.resize((960, 544), Image.BILINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = (img_array - mean) / std
    img_tensor = img_array.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
    return img_tensor, orig_w, orig_h

img_tensor, orig_w, orig_h = preprocess("/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/test_image.png")
model_path = "/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/assets/meiki.text.detect.v0.1.960x544.onnx"

sess = ort.InferenceSession(model_path)
ts = np.array([[orig_w, orig_h]], dtype=np.int64)
outputs = sess.run(None, {"images": img_tensor, "orig_target_sizes": ts})

ort_boxes = outputs[1][0].astype(np.float64)  # [64, 4] in original image space
ort_scores = outputs[2][0].astype(np.float64)

# Normalize ORT boxes back to model space
ort_boxes_norm = ort_boxes.copy()
ort_boxes_norm[:, 0] /= orig_w  # x1
ort_boxes_norm[:, 1] /= orig_h  # y1
ort_boxes_norm[:, 2] /= orig_w  # x2
ort_boxes_norm[:, 3] /= orig_h  # y2

# Load wgpu results
with open("/tmp/wgpu_results.json") as f:
    wgpu_data = json.load(f)
wgpu_boxes = np.array(wgpu_data["boxes"]).reshape(-1, 4)
wgpu_scores = np.array(wgpu_data["scores"])

# Normalize wgpu boxes
wgpu_boxes_norm = wgpu_boxes.copy()
wgpu_boxes_norm[:, 0] /= orig_w
wgpu_boxes_norm[:, 1] /= orig_h
wgpu_boxes_norm[:, 2] /= orig_w
wgpu_boxes_norm[:, 3] /= orig_h

# Compare normalized coordinates
from scipy.optimize import linear_sum_assignment
cost = np.ones((64, 64))
for i in range(64):
    for j in range(64):
        ix1 = max(ort_boxes_norm[i][0], wgpu_boxes_norm[j][0])
        iy1 = max(ort_boxes_norm[i][1], wgpu_boxes_norm[j][1])
        ix2 = min(ort_boxes_norm[i][2], wgpu_boxes_norm[j][2])
        iy2 = min(ort_boxes_norm[i][3], wgpu_boxes_norm[j][3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_ort = (ort_boxes_norm[i][2] - ort_boxes_norm[i][0]) * (ort_boxes_norm[i][3] - ort_boxes_norm[i][1])
        area_wgpu = (wgpu_boxes_norm[j][2] - wgpu_boxes_norm[j][0]) * (wgpu_boxes_norm[j][3] - wgpu_boxes_norm[j][1])
        union = area_ort + area_wgpu - inter
        iou = inter / union if union > 0 else 0
        cost[i, j] = 1 - iou

row_ind, col_ind = linear_sum_assignment(cost)

print("=== Normalized box coordinate errors (before scaling) ===")
total_err_orig = 0
total_err_norm = 0
for r, c in zip(row_ind, col_ind):
    iou = 1 - cost[r, c]
    if iou < 0.3:
        continue
    err_orig = np.sum(np.abs(ort_boxes[r] - wgpu_boxes[c]))
    err_norm = np.sum(np.abs(ort_boxes_norm[r] - wgpu_boxes_norm[c]))
    total_err_orig += err_orig
    total_err_norm += err_norm

n_matched = sum(1 for r, c in zip(row_ind, col_ind) if 1 - cost[r, c] >= 0.3)
print(f"  Mean L1 error (original space): {total_err_orig/n_matched:.1f} px")
print(f"  Mean L1 error (normalized space): {total_err_norm/n_matched:.6f}")
print(f"  Scale factor (orig/norm): {total_err_orig/total_err_norm:.1f}x")
print(f"  Expected scale factor: ~{orig_w:.0f}x (width), ~{orig_h:.0f}x (height)")

# Check if error is proportional to scale
print("\n=== Error breakdown by coordinate ===")
for coord, name in [(0, 'x1'), (1, 'y1'), (2, 'x2'), (3, 'y2')]:
    orig_err = np.mean([np.abs(ort_boxes[r][coord] - wgpu_boxes[c][coord]) for r, c in zip(row_ind, col_ind) if 1 - cost[r, c] >= 0.3])
    norm_err = np.mean([np.abs(ort_boxes_norm[r][coord] - wgpu_boxes_norm[c][coord]) for r, c in zip(row_ind, col_ind) if 1 - cost[r, c] >= 0.3])
    scale = orig_err / norm_err if norm_err > 0 else 0
    print(f"  {name}: orig_err={orig_err:.2f} norm_err={norm_err:.6f} scale={scale:.1f}x")

# Check if the error is in the detection head or earlier
print("\n=== Score error analysis ===")
score_diffs = []
for r, c in zip(row_ind, col_ind):
    iou = 1 - cost[r, c]
    if iou < 0.3:
        continue
    score_diffs.append(abs(ort_scores[r] - wgpu_scores[c]))

print(f"  Mean score diff: {np.mean(score_diffs):.4f}")
print(f"  Max score diff: {np.max(score_diffs):.4f}")
print(f"  Score diff correlates with box error: check top 10")

# Top 10 worst by score diff
score_pairs = [(r, c, abs(ort_scores[r] - wgpu_scores[c])) for r, c in zip(row_ind, col_ind) if 1 - cost[r, c] >= 0.3]
score_pairs.sort(key=lambda x: -x[2])
print("\nTop 10 worst score diffs:")
for r, c, sd in score_pairs[:10]:
    l1 = np.sum(np.abs(ort_boxes[r] - wgpu_boxes[c]))
    print(f"  ORT_idx={r} wgpu_idx={c} score_diff={sd:.4f} box_L1={l1:.1f} ort_score={ort_scores[r]:.4f} wgpu_score={wgpu_scores[c]:.4f}")
