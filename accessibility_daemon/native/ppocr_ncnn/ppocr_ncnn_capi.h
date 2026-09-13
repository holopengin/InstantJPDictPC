// C ABI over the shared PP-OCRv6 ncnn core, for Rust.
//
// The core (ppocr_ncnn_core.{h,cpp}) is the same file the Android app
// compiles; this layer only converts std::vector results into malloc'd
// buffers the Rust side can take ownership of, so no C++ type crosses the
// FFI boundary.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ppocr_rec_t ppocr_rec_t;
typedef struct ppocr_det_t ppocr_det_t;

/// Load the dynamic-width CTC recognizer; NULL on failure.
ppocr_rec_t* ppocr_rec_create(const char* param_path, const char* bin_path, int target_w, int num_threads);
void ppocr_rec_destroy(ppocr_rec_t* rec);

/// data must hold 3*h*w floats ([1,3,h,w], h = 48). On success returns a
/// malloc'd buffer of *out_len floats and sets *out_len; NULL on failure
/// with *out_len = -1. Free the buffer with ppocr_floats_free.
float* ppocr_rec_infer(ppocr_rec_t* rec, const float* data, size_t data_floats, int w, int h, int* out_len);
float* ppocr_rec_infer_topk(ppocr_rec_t* rec, const float* data, size_t data_floats, int w, int h, int* out_len);

/// Top-K emitted by ppocr_rec_infer_topk (packed [idx,val] pairs per step).
int ppocr_rec_top_k(void);

/// Load the DB segmentation detector; NULL on failure.
ppocr_det_t* ppocr_det_create(const char* param_path, const char* bin_path);
void ppocr_det_destroy(ppocr_det_t* det);

/// data must hold 3*h*w floats. On success returns a malloc'd prob map of
/// *out_len floats (w*h); NULL on failure with *out_len = -1.
float* ppocr_det_infer(ppocr_det_t* det, const float* data, size_t data_floats, int w, int h, int* out_len);

void ppocr_floats_free(float* data);

#ifdef __cplusplus
}
#endif
