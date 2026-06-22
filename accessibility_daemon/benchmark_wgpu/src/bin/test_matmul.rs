/// Test binary to verify the naive matmul kernel produces correct results
/// for a simple known input (no model loading needed).
use burn::tensor::{Device, Tensor, TensorData};
use burn::backend::Wgpu;

fn main() {
    let device: Device = Default::default();

    // Test 1: Simple 2x3 matmul
    // lhs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    // rhs = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    // expected = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //           = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
    //           = [[58, 64], [139, 154]]

    let lhs_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    let lhs = Tensor::<2>::from_data(TensorData::new(lhs_data, [2, 3]), &device);
    let rhs = Tensor::<2>::from_data(TensorData::new(rhs_data, [3, 2]), &device);

    println!("LHS: {:?}", lhs.to_data().as_slice::<f32>().unwrap());
    println!("RHS: {:?}", rhs.to_data().as_slice::<f32>().unwrap());

    // Use the default matmul (which now routes small matmuls through naive kernel)
    let result = lhs.clone().matmul(rhs.clone());
    let data = result.to_data();
    let result_slice = data.as_slice::<f32>().unwrap();

    println!("Result: {:?}", result_slice);
    println!("Expected: [58.0, 64.0, 139.0, 154.0]");

    let expected = [58.0f32, 64.0, 139.0, 154.0];
    let mut all_close = true;
    for (i, (r, e)) in result_slice.iter().zip(expected.iter()).enumerate() {
        let diff = (r - e).abs();
        if diff > 0.01 {
            println!("MISMATCH at {}: got {}, expected {} (diff={})", i, r, e, diff);
            all_close = false;
        }
    }
    if all_close {
        println!("Test 1 PASSED: naive matmul produces correct results for 2x3 * 3x2");
    } else {
        println!("Test 1 FAILED: naive matmul produces incorrect results");
    }

    // Test 2: 1x1 matmul (edge case)
    let lhs2 = Tensor::<2>::from_data(TensorData::new(vec![3.0f32], [1, 1]), &device);
    let rhs2 = Tensor::<2>::from_data(TensorData::new(vec![4.0f32], [1, 1]), &device);
    let result2 = lhs2.clone().matmul(rhs2.clone());
    let data2 = result2.to_data();
    let r2 = data2.as_slice::<f32>().unwrap();
    println!("\nTest 2 (1x1): result = {:?}, expected = [12.0]", r2);
    if (r2[0] - 12.0).abs() < 0.01 {
        println!("Test 2 PASSED");
    } else {
        println!("Test 2 FAILED");
    }

    // Test 3: Larger matmul that should use the tiled kernel (>1M elements)
    // 1024x1024 * 1024x1024 = 1B elements > 1M threshold
    // Actually the threshold is M*N*K, so 1024*1024*1024 = 1B > 1M
    // Let's use 32x32x32 = 32768 < 1M threshold
    let size = 32usize;
    let lhs_data3: Vec<f32> = (0..size*size).map(|i| i as f32).collect();
    let rhs_data3: Vec<f32> = (0..size*size).map(|i| (i + 1) as f32).collect();
    let lhs3 = Tensor::<2>::from_data(TensorData::new(lhs_data3, [size, size]), &device);
    let rhs3 = Tensor::<2>::from_data(TensorData::new(rhs_data3, [size, size]), &device);
    let result3 = lhs3.clone().matmul(rhs3.clone());
    let data3 = result3.to_data();
    let r3 = data3.as_slice::<f32>().unwrap();

    // Check element [0,0] = sum of lhs[0,k] * rhs[k,0] for k in 0..31
    // lhs[0,k] = k*32 + 0 = k (first row of lhs)
    // rhs[k,0] = k*32 + 0 + 1 = k*32 + 1 (first column of rhs)
    // expected[0,0] = sum(k * (k*32 + 1) for k=0..31) = sum(32*k^2 + k)
    // = 32 * sum(k^2 for k=0..31) + sum(k for k=0..31)
    // sum(k for k=0..31) = 31*32/2 = 496
    // sum(k^2 for k=0..31) = 31*32*63/6 = 10416
    // total = 32 * 10416 + 496 = 333312 + 496 = 333808
    let expected_00: f32 = (0..size).map(|k| {
        let kf = k as f32;
        kf * (kf * size as f32 + 1.0)
    }).sum();
    println!("\nTest 3 (32x32): result[0,0] = {}, expected = {}", r3[0], expected_00);
    if (r3[0] - expected_00).abs() < 1.0 {
        println!("Test 3 PASSED");
    } else {
        println!("Test 3 FAILED (diff = {})", (r3[0] - expected_00).abs());
    }

    // Test 4: Transposed tensor (non-contiguous)
    let lhs_t_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let rhs_t_data: Vec<f32> = (0..6).map(|i| (i + 1) as f32).collect();
    let lhs_t = Tensor::<2>::from_data(TensorData::new(lhs_t_data, [3, 2]), &device);
    let rhs_t = Tensor::<2>::from_data(TensorData::new(rhs_t_data, [2, 3]), &device);

    // Transpose to get non-contiguous memory
    let lhs_t2 = lhs_t.transpose(); // now [2, 3] but non-contiguous
    let rhs_t2 = rhs_t.transpose(); // now [3, 2] but non-contiguous

    println!("\nTest 4 (transposed):");
    println!("  lhs_t2 data: {:?}", lhs_t2.to_data().as_slice::<f32>().unwrap());
    println!("  rhs_t2 data: {:?}", rhs_t2.to_data().as_slice::<f32>().unwrap());

    let result4 = lhs_t2.clone().matmul(rhs_t2.clone());
    let data4 = result4.to_data();
    let r4 = data4.as_slice::<f32>().unwrap();
    println!("  Result: {:?}", r4);
    // lhs_t2 (transposed from [3,2] to [2,3]): [[0,2,4],[1,3,5]]
    // rhs_t2 (transposed from [2,3] to [3,2]): [[1,4],[2,5],[3,6]]
    // result = lhs_t2 [2,3] @ rhs_t2 [3,2] = [2,2]
    // result[0,0] = 0*1 + 2*2 + 4*3 = 16
    // result[0,1] = 0*4 + 2*5 + 4*6 = 34
    // result[1,0] = 1*1 + 3*2 + 5*3 = 22
    // result[1,1] = 1*4 + 3*5 + 5*6 = 49
    let expected4 = [16.0f32, 34.0, 22.0, 49.0];
    println!("  Expected: {:?}", expected4);
    let mut all_close4 = true;
    for (i, (r, e)) in r4.iter().zip(expected4.iter()).enumerate() {
        let diff = (r - e).abs();
        if diff > 0.01 {
            println!("  MISMATCH at {}: got {}, expected {} (diff={})", i, r, e, diff);
            all_close4 = false;
        }
    }
    if all_close4 {
        println!("  Test 4 PASSED");
    } else {
        println!("  Test 4 FAILED");
    }
}
