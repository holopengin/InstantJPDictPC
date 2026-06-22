#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["onnxruntime", "numpy", "Pillow"]
# ///

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

def main():
    img_path = "/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/test_image.png"
    model_path = "/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/assets/meiki.text.detect.v0.1.960x544.onnx"

    img_tensor, orig_w, orig_h = preprocess(img_path)

    # Run ORT with ALL intermediate outputs
    sess = ort.InferenceSession(model_path)
    ts = np.array([[orig_w, orig_h]], dtype=np.int64)

    # Get all node names to extract intermediate outputs
    all_nodes = [node.name for node in sess.get_outputs()]
    print(f"Total output nodes: {len(all_nodes)}")

    # Run with all outputs
    outputs = sess.run(all_nodes, {"images": img_tensor, "orig_target_sizes": ts})
    ort_intermediates = {name: arr for name, arr in zip(all_nodes, outputs)}

    # Load wgpu final results
    with open("/tmp/wgpu_results.json") as f:
        wgpu_data = json.load(f)

    # The final outputs
    ort_boxes = ort_intermediates['boxes'][0].astype(np.float64)  # [64, 4]
    ort_scores = ort_intermediates['scores'][0].astype(np.float64)  # [64]
    wgpu_boxes = np.array(wgpu_data["boxes"]).reshape(-1, 4)
    wgpu_scores = np.array(wgpu_data["scores"])

    # Find the worst-matching boxes (largest L1 error)
    from scipy.optimize import linear_sum_assignment

    # Compute cost matrix
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

    # Find worst matches
    matches = []
    for r, c in zip(row_ind, col_ind):
        iou = 1 - cost[r, c]
        if iou < 0.5:
            continue
        l1 = np.sum(np.abs(ort_boxes[r] - wgpu_boxes[c]))
        score_diff = abs(ort_scores[r] - wgpu_scores[c])
        matches.append((r, c, iou, l1, score_diff, ort_boxes[r], wgpu_boxes[c]))

    # Sort by L1 error descending
    matches.sort(key=lambda x: -x[3])

    print(f"\n=== Top 10 worst box matches (by L1 error) ===")
    print(f"{'ORT_idx':>7} {'wgpu_idx':>7} {'IoU':>6} {'L1':>8} {'Score_diff':>10} | {'ORT_box':>30} | {'wgpu_box':>30}")
    for r, c, iou, l1, sd, ob, wb in matches[:10]:
        print(f"{r:7d} {c:7d} {iou:6.4f} {l1:8.1f} {sd:10.4f} | [{ob[0]:.1f},{ob[1]:.1f},{ob[2]:.1f},{ob[3]:.1f}] | [{wb[0]:.1f},{wb[1]:.1f},{wb[2]:.1f},{wb[3]:.1f}]")

    # Analyze which spatial regions have worst errors
    print(f"\n=== Spatial error analysis ===")
    print("Box positions (center x, center y) colored by L1 error:")
    for r, c, iou, l1, sd, ob, wb in matches[:20]:
        cx_ort = (ob[0] + ob[2]) / 2
        cy_ort = (ob[1] + ob[3]) / 2
        print(f"  center=({cx_ort:.0f},{cy_ort:.0f}) L1={l1:.1f} score_diff={sd:.4f}")

    # Check if errors correlate with position (left/right/top/bottom of image)
    print(f"\n=== Error by image region ===")
    left_errors = []
    right_errors = []
    top_errors = []
    bottom_errors = []
    for r, c, iou, l1, sd, ob, wb in matches:
        cx = (ob[0] + ob[2]) / 2
        cy = (ob[1] + ob[3]) / 2
        if cx < orig_w * 0.33:
            left_errors.append(l1)
        elif cx > orig_w * 0.67:
            right_errors.append(l1)
        if cy < orig_h * 0.33:
            top_errors.append(l1)
        elif cy > orig_h * 0.67:
            bottom_errors.append(l1)

    if left_errors:
        print(f"  Left third:   mean L1 = {np.mean(left_errors):.1f} (n={len(left_errors)})")
    if right_errors:
        print(f"  Right third:  mean L1 = {np.mean(right_errors):.1f} (n={len(right_errors)})")
    if top_errors:
        print(f"  Top third:    mean L1 = {np.mean(top_errors):.1f} (n={len(top_errors)})")
    if bottom_errors:
        print(f"  Bottom third: mean L1 = {np.mean(bottom_errors):.1f} (n={len(bottom_errors)})")

    # Check intermediate outputs that might explain the error
    # The model has submodules - let's look at the detection head outputs
    print(f"\n=== Intermediate output statistics ===")
    for name, arr in sorted(ort_intermediates.items()):
        if arr.size > 0 and arr.ndim >= 2:
            # Check for NaN/Inf
            has_nan = np.any(np.isnan(arr))
            has_inf = np.any(np.isinf(arr))
            if has_nan or has_inf:
                print(f"  {name}: shape={arr.shape} HAS {'NaN' if has_nan else ''} {'INF' if has_inf else ''}")

    # Look at the specific intermediate values that feed into the box regression
    # The detection head outputs are: labels (i64), boxes (f32), scores (f32)
    # The boxes come from the final linear layers
    print(f"\n=== Box coordinate statistics ===")
    print(f"  ORT boxes:  min={ort_boxes.min():.2f} max={ort_boxes.max():.2f}")
    print(f"  wgpu boxes: min={wgpu_boxes.min():.2f} max={wgpu_boxes.max():.2f}")
    print(f"  ORT scores:  min={ort_scores.min():.4f} max={ort_scores.max():.4f}")
    print(f"  wgpu scores: min={wgpu_scores.min():.4f} max={wgpu_scores.max():.4f}")

    # Check if the error is proportional to box size
    print(f"\n=== Error vs box size ===")
    for r, c, iou, l1, sd, ob, wb in matches[:10]:
        w = ob[2] - ob[0]
        h = ob[3] - ob[1]
        area = w * h
        print(f"  Box {r}: area={area:.0f} w={w:.1f} h={h:.1f} L1={l1:.1f} L1/area={l1/area:.4f}")

if __name__ == "__main__":
    main()
