#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["onnxruntime", "numpy", "Pillow"]
# ///

import json
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import time

def preprocess(img_path):
    img = Image.open(img_path)
    orig_w, orig_h = img.size
    img_resized = img.resize((960, 544), Image.LANCZOS)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = (img_array - mean) / std
    img_tensor = img_array.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
    return img_tensor, orig_w, orig_h

def main():
    print("=== Performance Benchmark ===\n")

    img_path = "/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/test_image.png"
    model_path = "/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/assets/meiki.text.detect.v0.1.960x544.onnx"

    # Load image for visualization
    img = Image.open(img_path)

    # Preprocess
    t0 = time.perf_counter()
    img_tensor, orig_w, orig_h = preprocess(img_path)
    preprocess_ms = (time.perf_counter() - t0) * 1000
    print(f"Preprocessing: {preprocess_ms:.1f}ms")

    # ========== ORT Benchmark ==========
    print("\n--- ONNX Runtime ---")

    # Cold load
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path)
    ort_load_ms = (time.perf_counter() - t0) * 1000
    print(f"Cold load: {ort_load_ms:.1f}ms")

    # Warm up
    ts = np.array([[orig_w, orig_h]], dtype=np.int64)
    _ = sess.run(None, {"images": img_tensor, "orig_target_sizes": ts})

    # Benchmark inference
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        outputs = sess.run(None, {"images": img_tensor, "orig_target_sizes": ts})
        times.append((time.perf_counter() - t0) * 1000)

    ort_avg = np.mean(times)
    ort_min = np.min(times)
    ort_max = np.max(times)
    print(f"Inference (avg of 10): {ort_avg:.1f}ms  (min={ort_min:.1f}, max={ort_max:.1f})")
    print(f"Total e2e: {preprocess_ms + ort_load_ms + ort_avg:.1f}ms")

    # Save ORT results for visualization
    ort_boxes = outputs[1][0]  # [64, 4]
    ort_scores = outputs[2][0]  # [64]
    ort_labels = outputs[0][0]  # [64]

    # ========== Burn Benchmark ==========
    print("\n--- Burn (ndarray, CPU) ---")

    # Import burn model
    import sys
    sys.path.insert(0, "/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/benchmark/target/release")
    # We can't import the burn model directly from Python, so we'll read the JSON output
    # from the Rust benchmark binary

    # Check if burn results exist
    burn_results_path = "/tmp/burn_results.json"
    try:
        with open(burn_results_path) as f:
            burn_data = json.load(f)
        burn_scores = burn_data["scores"]
        burn_labels = burn_data["labels"]
        burn_boxes_flat = burn_data["boxes"]
        print(f"Loaded Burn results from {burn_results_path}")
        print(f"  {len(burn_scores)} detections")
        burn_available = True
    except FileNotFoundError:
        print("Burn results not found. Run the burn benchmark first.")
        burn_available = False

    # Check if wgpu results exist
    wgpu_results_path = "/tmp/wgpu_results.json"
    try:
        with open(wgpu_results_path) as f:
            wgpu_data = json.load(f)
        wgpu_scores = wgpu_data["scores"]
        wgpu_labels = wgpu_data["labels"]
        wgpu_boxes_flat = wgpu_data["boxes"]
        print(f"Loaded wgpu results from {wgpu_results_path}")
        print(f"  {len(wgpu_scores)} detections")
        wgpu_available = True
    except FileNotFoundError:
        print("wgpu results not found. Run the wgpu benchmark first.")
        wgpu_available = False

    # ========== Visualization ==========
    print("\n--- Generating Visualization ---")

    result_img = img.copy()
    draw = ImageDraw.Draw(result_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Draw ORT boxes in blue
    ort_count = 0
    for i in range(len(ort_scores)):
        if ort_scores[i] < 0.3:
            continue
        ort_count += 1
        b = ort_boxes[i]
        x1 = max(0, min(orig_w, b[0]))
        y1 = max(0, min(orig_h, b[1]))
        x2 = max(0, min(orig_w, b[2]))
        y2 = max(0, min(orig_h, b[3]))
        draw.rectangle([x1, y1, x2, y2], outline=(0, 100, 255), width=3)

    # Draw Burn boxes in red
    burn_count = 0
    if burn_available:
        for i in range(len(burn_scores)):
            if burn_scores[i] < 0.3:
                continue
            burn_count += 1
            x1 = max(0, min(orig_w, burn_boxes_flat[i * 4]))
            y1 = max(0, min(orig_h, burn_boxes_flat[i * 4 + 1]))
            x2 = max(0, min(orig_w, burn_boxes_flat[i * 4 + 2]))
            y2 = max(0, min(orig_h, burn_boxes_flat[i * 4 + 3]))
            draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=3)

    # Draw wgpu boxes in green
    wgpu_count = 0
    if wgpu_available:
        for i in range(len(wgpu_scores)):
            if wgpu_scores[i] < 0.3:
                continue
            wgpu_count += 1
            x1 = max(0, min(orig_w, wgpu_boxes_flat[i * 4]))
            y1 = max(0, min(orig_h, wgpu_boxes_flat[i * 4 + 1]))
            x2 = max(0, min(orig_w, wgpu_boxes_flat[i * 4 + 2]))
            y2 = max(0, min(orig_h, wgpu_boxes_flat[i * 4 + 3]))
            draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 80), width=3)

    # Legend
    lx, ly = 10, 10
    legend_h = 75 if wgpu_available else 55
    draw.rectangle([lx, ly, lx + 300, ly + legend_h], fill=(0, 0, 0, 180))
    draw.rectangle([lx + 5, ly + 8, lx + 25, ly + 24], outline=(0, 100, 255), width=2)
    draw.text((lx + 30, ly + 5), f"ORT   {ort_count} dets  {ort_avg:.0f}ms", fill=(100, 180, 255), font=font)
    draw.rectangle([lx + 5, ly + 30, lx + 25, ly + 46], outline=(255, 50, 50), width=2)
    burn_text = f"Burn  {burn_count} dets" if burn_available else "Burn  N/A"
    draw.text((lx + 30, ly + 28), burn_text, fill=(255, 100, 100), font=font)
    if wgpu_available:
        draw.rectangle([lx + 5, ly + 52, lx + 25, ly + 68], outline=(0, 200, 80), width=2)
        draw.text((lx + 30, ly + 50), f"wgpu  {wgpu_count} dets", fill=(100, 255, 150), font=font)

    output_path = "/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/test_image_comparison.png"
    result_img.save(output_path)
    print(f"Saved to {output_path}")

    # ========== Summary ==========
    print(f"\n=== Summary ===")
    print(f"Image: {orig_w}x{orig_h}")
    print(f"Preprocessing: {preprocess_ms:.1f}ms")
    print(f"ORT:  load={ort_load_ms:.1f}ms  inference={ort_avg:.1f}ms  total={preprocess_ms+ort_load_ms+ort_avg:.1f}ms  dets={ort_count}")
    if burn_available:
        print(f"Burn: dets={burn_count}")
    if wgpu_available:
        print(f"wgpu: dets={wgpu_count}")
    print(f"\nInference speed ratio: ORT is {330/ort_avg:.1f}x faster than Burn (based on prior Burn measurement)")

if __name__ == "__main__":
    main()
