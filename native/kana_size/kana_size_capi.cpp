// C ABI over the shared kana small/large ncnn core, for Rust.
//
// The graph (kana_size_core.{h,cpp}) is the same file the Android app
// compiles; this layer only owns the handle and copies logits into a malloc'd
// buffer, so no C++ type crosses the FFI boundary.
#include "kana_size_capi.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#include "kana_size_core.h"

extern "C" {

kana_size_t* kana_size_create(const char* param_path, const char* bin_path) {
    return reinterpret_cast<kana_size_t*>(kana_size::create(param_path, bin_path));
}

void kana_size_destroy(kana_size_t* model) {
    kana_size::destroy(reinterpret_cast<kana_size::Model*>(model));
}

float* kana_size_logits(kana_size_t* model, const int* win, const int* bases, int n, int* out_len) {
    *out_len = -1;
    if (model == nullptr || win == nullptr || bases == nullptr || n <= 0) {
        return nullptr;
    }

    std::vector<float> out;
    if (!kana_size::logits(
            reinterpret_cast<kana_size::Model*>(model), win, bases, n, out)) {
        return nullptr;
    }

    float* p = (float*) std::malloc((size_t) n * sizeof(float));
    if (p == nullptr) {
        return nullptr;
    }
    std::memcpy(p, out.data(), (size_t) n * sizeof(float));
    *out_len = n;
    return p;
}

void kana_size_floats_free(float* data) {
    std::free(data);
}

} // extern "C"
