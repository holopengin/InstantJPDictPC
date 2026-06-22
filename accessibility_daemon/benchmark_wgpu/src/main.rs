mod model;

use burn::tensor::{Device, Tensor, TensorData};
use image::GenericImageView;
use std::time::Instant;

fn main() {
    let img_path = "/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/test_image.png";
    let img = image::open(img_path).expect("Failed to open test image");
    let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
    let target_w = 960u32;
    let target_h = 544u32;
    let img_resized = image::imageops::resize(&img, target_w, target_h, image::imageops::FilterType::Triangle);

    let mean = [0.485_f32, 0.456, 0.406];
    let std = [0.229_f32, 0.224, 0.225];
    let mut data = vec![0.0f32; 1 * 3 * target_h as usize * target_w as usize];
    for y in 0..target_h {
        for x in 0..target_w {
            let pixel = img_resized.get_pixel(x, y);
            let idx = (y * target_w + x) as usize;
            data[idx] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
            data[target_h as usize * target_w as usize + idx] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
            data[2 * target_h as usize * target_w as usize + idx] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let device: Device = Default::default();
    let images = Tensor::<4>::from_data(
        TensorData::new(data, [1, 3, target_h as usize, target_w as usize]),
        &device,
    );
    let orig_target_sizes = Tensor::<2, burn::tensor::Int>::from_ints(
        [[orig_w as i64, orig_h as i64]],
        &device,
    );

    let bpk_path = std::path::Path::new("/var/home/holopengin/repos/InstantJPDictDecky/accessibility_daemon/benchmark_wgpu/meiki.text.detect.v0.1.bpk");
    let model = model::Model::from_file(bpk_path, &device);

    // Warmup
    let _ = model.forward(images.clone(), orig_target_sizes.clone());

    // Benchmark
    let n_runs = 10;
    let mut times = Vec::new();
    let mut labels_out = None;
    let mut boxes_out = None;
    let mut scores_out = None;
    for _ in 0..n_runs {
        let t0 = Instant::now();
        let (labels, boxes, scores) = model.forward(images.clone(), orig_target_sizes.clone());
        times.push(t0.elapsed());
        if labels_out.is_none() {
            labels_out = Some(labels);
            boxes_out = Some(boxes);
            scores_out = Some(scores);
        }
    }

    let avg = times.iter().sum::<std::time::Duration>() / times.len() as u32;
    let min = *times.iter().min().unwrap();
    println!("=== Burn wgpu Benchmark ===");
    println!("Inference:  avg={:.2?}  min={:.2?}", avg, min);

    // Extract data for JSON output
    let labels_raw = labels_out.unwrap();
    let boxes_raw = boxes_out.unwrap();
    let scores_raw = scores_out.unwrap();

    let scores_vec: Vec<f32> = scores_raw.into_data().to_vec().unwrap();
    let labels_data = labels_raw.into_data();
    let labels_vec: Vec<i64> = labels_data.as_slice::<i64>().unwrap().to_vec();
    let boxes_model: Vec<f32> = boxes_raw.into_data().to_vec().unwrap();

    // Model already outputs boxes in original image space since we pass
    // orig_target_sizes=[orig_w, orig_h] and the model does: boxes * orig_target_sizes
    let boxes_vec: Vec<f64> = boxes_model.iter().map(|&v| v as f64).collect();

    // Write results to JSON
    let json = format!(
        r#"{{"scores":{:?},"labels":{:?},"boxes":{:?}}}"#,
        scores_vec, labels_vec, boxes_vec
    );
    std::fs::write("/tmp/wgpu_results.json", json).unwrap();
    println!("wgpu results written to /tmp/wgpu_results.json");
    println!("Detections: {}", scores_vec.len());
}
