// Platform-independent PP-OCRv6 ncnn inference core.
//
// This file is the single source of truth for the DetNcnn/RecNcnn kernel and
// is compiled by BOTH apps:
//   - Android (InstantJPDict): app/src/main/cpp/ncnn_jni.cpp is a thin JNI
//     wrapper over this core.
//   - PC (InstantJPDictDecky): native/ppocr_ncnn/ppocr_ncnn_core.* is a copy
//     of this file; ppocr_ncnn_capi.cpp exposes it to Rust.
//
// Keep this file and ppocr_ncnn_core.cpp byte-identical between the two
// repositories (PC: tools/sync_ppocr_core.sh can check/update the copy).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ncnn/net.h"

namespace ppocr_ncnn {

struct RecNet {
    ncnn::Net net;
    int targetW;
    int seqLen;
};

struct DetNet {
    ncnn::Net net;
    std::string cachedOutName;
};

/// Load the dynamic-width CTC recognizer. Returns nullptr on failure.
/// seqLen is targetW / 8; per-call sequence length is derived from the
/// actual input width instead (dynamic width, #23).
RecNet* rec_create(const char* paramPath, const char* binPath, int targetW, int numThreads);

void rec_destroy(RecNet* rec);

/// Run one [1,3,48,w] NCHW float input (dataFloats must be 3*48*w).
/// Full-logits path: fills `out` with seqLen * numClasses floats, row-major
/// with classes innermost. Returns false on failure.
bool rec_infer(RecNet* rec, const float* data, size_t dataFloats, int w, int h, std::vector<float>& out);

/// Run one [1,3,48,w] NCHW float input (dataFloats must be 3*48*w).
/// Top-K path (#42): fills `out` with seqLen * K * 2 floats, packed
/// [idx0,val0, idx1,val1, ...] per timestep (indices stored as float, exact
/// for ids < 2^24). K is [rec_top_k]. Returns false on failure.
bool rec_infer_topk(RecNet* rec, const float* data, size_t dataFloats, int w, int h, std::vector<float>& out);

/// Top-K per CTC timestep emitted by rec_infer_topk; must match
/// OcrEngine.TOP_K on the Kotlin side.
int rec_top_k();

/// Load the DB segmentation detector. Returns nullptr on failure.
DetNet* det_create(const char* paramPath, const char* binPath);

void det_destroy(DetNet* det);

/// Run one [1,3,h,w] NCHW float input (dataFloats must be 3*h*w).
/// Fills `out` with the raw probability map (w*h floats). Returns false on
/// failure.
bool det_infer(DetNet* det, const float* data, size_t dataFloats, int w, int h, std::vector<float>& out);

/// det_infer's input half: copy [1,3,h,w] NCHW floats into a Mat. Split out so
/// the cost of that copy (a 2.4 M float memcpy the letterbox path never pays)
/// can be measured without the net's variance swamping it. Returns false on a
/// short buffer. `in` is created or refilled in place.
bool det_fill_input(const float* data, size_t dataFloats, int w, int h, ncnn::Mat& in);

/// Build the [1,3,modelSize,modelSize] NCHW det input straight from pixels:
/// letterbox `resizeW x resizeH` ARGB8888 content into a modelSize² canvas
/// whose padding is pixel-128 gray, then apply the per-channel ImageNet
/// normalisation. `argb` is resizeW*resizeH 0xAARRGGBB pixels, row-major, as
/// Bitmap.getPixels delivers them.
///
/// Bit-identical to the Kotlin DET_NORM_LUT + letterbox it replaces: the
/// normalisation is the same expression in the same order, and the padding goes
/// through it at v=128, so no constant is assumed. The caller supplies the
/// resize, because Android's Skia filter is the reference and must not be
/// second-guessed — and it must also supply `padX`/`padY`, because where Skia
/// actually lands a `(modelSize - content) / 2f` translate is a rounding
/// convention, not a formula (it rounds the half pixel away from zero, so the
/// content starts at `(modelSize - content + 1) / 2`).
///
/// `out` is created, or refilled in place when it already has the right shape —
/// callers keep one per thread, because 9.6 MB is above glibc's mmap threshold
/// and a fresh Mat per detect is ~2,400 first-touch page faults.
/// Returns false on bad geometry.
bool det_build_input(const int* argb, int resizeW, int resizeH, int modelSize, int padX, int padY, ncnn::Mat& out);

/// Run the DB net on a Mat built by det_build_input (or any other prepared
/// [1,3,modelSize,modelSize] input). Split out of det_infer so the letterbox
/// path and the float path share one extract: the cached output name, the
/// name-fallback list and the output guard must not exist twice.
bool det_run(DetNet* det, const ncnn::Mat& in, std::vector<float>& out);

/// det_build_input + det_run. `out` gets the raw probability map.
bool det_infer_letterboxed(DetNet* det, const int* argb, int resizeW, int resizeH, int modelSize, int padX, int padY, std::vector<float>& out);

} // namespace ppocr_ncnn
