# Burn OPs Fix List — Known Bugs in burn-onnx Affecting Our Model

## Context

We're trying to replace ONNX Runtime (ORT) with Burn for the detection model in `accessibility_daemon`. The model file is `assets/meiki.text.detect.v0.1.960x544.onnx`. Burn produces systematically wrong outputs compared to ORT — both scores and box coordinates are completely different.

The root cause is bugs in `burn-onnx`'s code generation for reduce ops, confirmed by [burn-onnx issue #311](https://github.com/tracel-ai/burn-onnx/issues/311) which reports 181 failing ONNX backend tests.

## Repos

- `burn/` — cloned from https://github.com/tracel-ai/burn (local copy, requires Rust 1.95+)
- `burn-onnx/` — cloned from https://github.com/tracel-ai/burn-onnx

## Problematic Ops in Our Model

### Primary fixes needed (ops used in our model with known bugs):

| Op | Count in model | Known issue | Fix location |
|---|---|---|---|
| `ReduceMean` | 25 | `keepdims` handling, output shape wrong | `burn-onnx/crates/burn-onnx/src/burn/node/reduce.rs` |
| `ReduceSum` | 3 | Output shape wrong (keepdims/rank handling) | Same file |
| `ReduceMax` | 1 | Output shape wrong + bool input support | Same file |
| `GridSample` | 9 | Coordinate transformation / interpolation | `burn-onnx/.../node/grid_sample.rs` + `burn/crates/burn-ndarray/src/ops/grid_sample.rs` |
| `Resize` | 2 | Algorithm differences (cubic, nearest modes) | `burn-onnx/.../node/resize.rs` + `burn/crates/burn-ndarray/src/ops/interpolate.rs` |

### Secondary ops (from issue #311, may affect other models):

| Op | Known issue |
|---|---|
| `ReduceMin` | Output shape wrong (same pattern as ReduceMax) |
| `ReduceProd` | Output shape wrong (keepdims) |
| `ReduceL1` | Output shape wrong |
| `ReduceL2` | Output shape wrong |
| `ReduceLogSum` | Axis handling, empty set |
| `Attention` | Numerical precision on 3D/4D with causal/bias/mask |
| `Cast` | Exotic dtypes (FLOAT4E2M1, FLOAT8E*, INT2/4, UINT2/4) |
| `Triu` | Negative diagonal, edge cases |
| `Tril` | Negative diagonal, edge cases |
| `AveragePool` | `ceil_mode` with same padding |
| `MaxPool` | `ceil_mode` output size |
| `NLLLoss` | Weight handling, reduction modes |
| `Add/Sub/Mul/Div/Max/Min` | uint64 dtype support |

## Already Applied Fix

**File:** `burn-onnx/crates/burn-onnx/src/burn/node/reduce.rs`

The `reduce_by_dims()` function reduces multiple dimensions sequentially but doesn't account for the changing tensor shape after each reduction. When reducing dims [1, 2], after reducing dim 1 the tensor shape changes and dim 2 becomes dim 1, but the code still calls `reduce_dim(2)`.

**Fix applied:** Sort dimensions in descending order before reducing, so higher indices are processed first and lower indices don't shift.

**Status:** Code patched but NOT yet verified against ORT baselines.

## Test Strategy

For each op, write a test that:
1. Creates a small input tensor (e.g., shape [2, 3, 4])
2. Runs it through ORT to generate baseline output
3. Runs it through Burn NdArray backend
4. Compares with `np.allclose(rtol=1e-5, atol=1e-6)`

Test all combinations of:
- `keepdims={true, false}`
- `dims={single, multiple, all, empty}`
- `noop_with_empty_axes` (for empty dims)
- Various input shapes and dtypes

## Key Files

### burn-onnx (code generation):
- `burn-onnx/crates/burn-onnx/src/burn/node/reduce.rs` — Reduce op codegen (already patched)
- `burn-onnx/crates/burn-onnx/src/burn/node/grid_sample.rs` — GridSample codegen
- `burn-onnx/crates/burn-onnx/src/burn/node/resize.rs` — Resize codegen

### burn-ndarray (backend implementation):
- `burn/crates/burn-ndarray/src/ops/grid_sample.rs` — GridSample kernel
- `burn/crates/burn-ndarray/src/ops/interpolate.rs` — Resize/interpolation kernel
- `burn/crates/burn-ndarray/src/ops/base.rs` — Base tensor operations

### Our model:
- `assets/meiki.text.detect.v0.1.960x544.onnx` — Detection model (102 convs, 25 reduce_mean, etc.)
- `accessibility_daemon/src/ocr_engine.rs` — Current ORT-based detection code

## Validation

After fixing the ops, rebuild the detection model and verify:
1. Scores match ORT (within tolerance)
2. Box coordinates match ORT (within tolerance)
3. The full OCR pipeline produces correct results on `test_image.png`

## Rust Version Note

The burn workspace requires Rust 1.95+. The current system has Rust 1.93. You'll need to upgrade Rust before building burn.
