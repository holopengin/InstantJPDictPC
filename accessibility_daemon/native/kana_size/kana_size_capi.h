// C ABI over the kana small/large model (#44), for Rust.
//
// The Android app runs the same graph through app/src/main/cpp/kana_size_ncnn.cpp
// (JNI); this shell only owns the net handle and copies logits into a malloc'd
// buffer, so no C++ type crosses the FFI boundary.
//
// The graph and the pinned net options are duplicated from the mobile wrapper
// deliberately: the fp32 pin is load-bearing (fp16 costs ~1.5e-03 against the
// fp32 reference on this model), and both apps must run identical numerics.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kana_size_t kana_size_t;

/// Load the converted `nb_all` param/bin with the fp32 options pinned
/// (fp16 packed/storage/arithmetic, bf16 and packing all off). NULL on failure.
kana_size_t* kana_size_create(const char* param_path, const char* bin_path);

void kana_size_destroy(kana_size_t* model);

/// Logits for `n` positions. `win` holds n*40 int32 byte values (from the
/// window encoder), `bases` n pair indices; an out-of-range base is clamped to
/// 0 like the mobile native side. On success returns a malloc'd buffer of
/// `n` floats and sets *out_len = n; NULL on failure with *out_len = -1.
float* kana_size_logits(kana_size_t* model, const int* win, const int* bases, int n, int* out_len);

void kana_size_floats_free(float* data);

#ifdef __cplusplus
}
#endif
