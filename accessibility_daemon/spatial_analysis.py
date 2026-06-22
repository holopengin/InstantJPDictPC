#!/usr/bin/env python3
"""Quick spatial error analysis using existing comparison data."""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

# Load data
with open("/tmp/wgpu_results.json") as f:
    wgpu_data = json.load(f)
with open("/tmp/burn_results.json") as f:
    burn_data = json.load(f)

# We need ORT data too - run it fresh
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

ort_boxes = outputs[1][0].astype(np.float64)
ort_scores = outputs[2][0].astype(np.float64)
wgpu_boxes = np.array(wgpu_data["boxes"]).reshape(-1, 4)
wgpu_scores = np.array(wgpu_data["scores"])

# Hungarian matching
cost = np.ones((64, 64))
for i in range(64):
    for j in range(64):
        ix1 = max(ort_boxes[i][0], wgpu_boxes[j][0])
        iy1 = max(ort_boxes[i][1], wgpu_boxes[j][1])
        ix2 = min(ort_boxes[i][2], wgpu_boxes[j][2])
        iy2 = min(ort_boxes[i][3], wgpu_boxes[j][3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_ort = (ort_boxes[i][2] - ort_boxes[i][0]) * (ort_boxes[i][3] - ort_boxes[i][1])
        area_wgpu = (wgpu_boxes[j][2] - wgpu_boxes[j][0]) * (wgpu_boxes[j][3] - wgpu_boxes[j][1])
        union = area_ort + area_wgpu - inter
        iou = inter / union if union > 0 else 0
        cost[i, j] = 1 - iou

row_ind, col_ind = linear_sum_assignment(cost)

# Analyze spatial distribution of errors
matches = []
for r, c in zip(row_ind, col_ind):
    iou = 1 - cost[r, c]
    if iou < 0.3:
        continue
    l1 = np.sum(np.abs(ort_boxes[r] - wgpu_boxes[c]))
    score_diff = abs(ort_scores[r] - wgpu_scores[c])
    cx = (ort_boxes[r][0] + ort_boxes[r][2]) / 2
    cy = (ort_boxes[r][1] + ort_boxes[r][3]) / 2
    w = ort_boxes[r][2] - ort_boxes[r][0]
    h = ort_boxes[r][3] - ort_boxes[r][1]
    matches.append({
        'ort_idx': r, 'wgpu_idx': c, 'iou': iou, 'l1': l1,
        'score_diff': score_diff, 'cx': cx, 'cy': cy, 'w': w, 'h': h,
        'ort_box': ort_boxes[r], 'wgpu_box': wgpu_boxes[c]
    })

matches.sort(key=lambda x: -x['l1'])

print(f"=== Spatial error analysis ({len(matches)} matched pairs) ===\n")

# Error by horizontal position (left/middle/right)
regions = {
    'left': {'range': (0, orig_w/3), 'errors': [], 'scores': []},
    'center': {'range': (orig_w/3, 2*orig_w/3), 'errors': [], 'scores': []},
    'right': {'range': (2*orig_w/3, orig_w), 'errors': [], 'scores': []},
    'top': {'range': (0, orig_h/3), 'errors': [], 'scores': []},
    'middle_v': {'range': (orig_h/3, 2*orig_h/3), 'errors': [], 'scores': []},
    'bottom': {'range': (2*orig_h/3, orig_h), 'errors': [], 'scores': []},
}

for m in matches:
    cx, cy = m['cx'], m['cy']
    for name, region in [('left', regions['left']), ('center', regions['center']), ('right', regions['right'])]:
        if region['range'][0] <= cx < region['range'][1]:
            region['errors'].append(m['l1'])
            region['scores'].append(m['score_diff'])
    for name, region in [('top', regions['top']), ('middle_v', regions['middle_v']), ('bottom', regions['bottom'])]:
        if region['range'][0] <= cy < region['range'][1]:
            region['errors'].append(m['l1'])
            region['scores'].append(m['score_diff'])

print("Horizontal position vs error:")
for name in ['left', 'center', 'right']:
    r = regions[name]
    if r['errors']:
        print(f"  {name:>8s}: L1 mean={np.mean(r['errors']):.1f} max={np.max(r['errors']):.1f} n={len(r['errors'])}")

print("\nVertical position vs error:")
for name in ['top', 'middle_v', 'bottom']:
    r = regions[name]
    if r['errors']:
        print(f"  {name:>8s}: L1 mean={np.mean(r['errors']):.1f} max={np.max(r['errors']):.1f} n={len(r['errors'])}")

# Check if edge boxes (near image border) have more error
print("\n=== Edge proximity analysis ===")
edge_threshold = 50  # pixels from edge
edge_errors = []
center_errors = []
for m in matches:
    ob = m['ort_box']
    dist_to_edge = min(ob[0], ob[1], orig_w - ob[2], orig_h - ob[3])
    if dist_to_edge < edge_threshold:
        edge_errors.append(m['l1'])
    else:
        center_errors.append(m['l1'])

if edge_errors:
    print(f"  Edge boxes (within {edge_threshold}px of border): L1 mean={np.mean(edge_errors):.1f} n={len(edge_errors)}")
if center_errors:
    print(f"  Center boxes: L1 mean={np.mean(center_errors):.1f} n={len(center_errors)}")

# Check error vs box size
print("\n=== Error vs box size ===")
small_errors = []
large_errors = []
for m in matches:
    area = m['w'] * m['h']
    if area < 5000:
        small_errors.append(m['l1'])
    elif area > 20000:
        large_errors.append(m['l1'])

if small_errors:
    print(f"  Small boxes (area<5000): L1 mean={np.mean(small_errors):.1f} n={len(small_errors)}")
if large_errors:
    print(f"  Large boxes (area>20000): L1 mean={np.mean(large_errors):.1f} n={len(large_errors)}")

# Worst 10 boxes
print("\n=== Top 10 worst matches by L1 ===")
for m in matches[:10]:
    ob = m['ort_box']
    wb = m['wgpu_box']
    print(f"  IoU={m['iou']:.3f} L1={m['l1']:.1f} score_diff={m['score_diff']:.4f}")
    print(f"    ORT:  [{ob[0]:.1f}, {ob[1]:.1f}, {ob[2]:.1f}, {ob[3]:.1f}] center=({m['cx']:.0f},{m['cy']:.0f})")
    print(f"    wgpu: [{wb[0]:.1f}, {wb[1]:.1f}, {wb[2]:.1f}, {wb[3]:.1f}]")
    print(f"    Delta: [{wb[0]-ob[0]:+.1f}, {wb[1]-ob[1]:+.1f}, {wb[2]-ob[2]:+.1f}, {wb[3]-ob[3]:+.1f}]")
