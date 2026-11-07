#include "ppocr_ncnn_capi.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#include "ppocr_ncnn_core.h"

namespace {

// Copy a core result into a malloc'd buffer for Rust. NULL + *out_len = -1
// means failure; an empty (but successful) result still returns a non-NULL
// buffer with *out_len = 0.
float* dup_floats(const std::vector<float>& v, int* out_len) {
    float* p = (float*) std::malloc(v.size() > 0 ? v.size() * sizeof(float) : sizeof(float));
    if (!p) {
        *out_len = -1;
        return nullptr;
    }
    if (!v.empty()) std::memcpy(p, v.data(), v.size() * sizeof(float));
    *out_len = (int) v.size();
    return p;
}

} // namespace

extern "C" {

ppocr_rec_t* ppocr_rec_create(const char* param_path, const char* bin_path, int target_w, int num_threads) {
    return (ppocr_rec_t*) ppocr_ncnn::rec_create(param_path, bin_path, target_w, num_threads);
}

void ppocr_rec_destroy(ppocr_rec_t* rec) {
    ppocr_ncnn::rec_destroy((ppocr_ncnn::RecNet*) rec);
}

float* ppocr_rec_infer(ppocr_rec_t* rec, const float* data, size_t data_floats, int w, int h, int* out_len) {
    *out_len = -1;
    std::vector<float> out;
    if (!ppocr_ncnn::rec_infer((ppocr_ncnn::RecNet*) rec, data, data_floats, w, h, out)) return nullptr;
    return dup_floats(out, out_len);
}

float* ppocr_rec_infer_topk(ppocr_rec_t* rec, const float* data, size_t data_floats, int w, int h, int* out_len) {
    *out_len = -1;
    std::vector<float> out;
    if (!ppocr_ncnn::rec_infer_topk((ppocr_ncnn::RecNet*) rec, data, data_floats, w, h, out)) return nullptr;
    return dup_floats(out, out_len);
}

int ppocr_rec_top_k(void) {
    return ppocr_ncnn::rec_top_k();
}

ppocr_det_t* ppocr_det_create(const char* param_path, const char* bin_path) {
    return (ppocr_det_t*) ppocr_ncnn::det_create(param_path, bin_path);
}

void ppocr_det_destroy(ppocr_det_t* det) {
    ppocr_ncnn::det_destroy((ppocr_ncnn::DetNet*) det);
}

float* ppocr_det_infer(ppocr_det_t* det, const float* data, size_t data_floats, int w, int h, int* out_len) {
    *out_len = -1;
    std::vector<float> out;
    if (!ppocr_ncnn::det_infer((ppocr_ncnn::DetNet*) det, data, data_floats, w, h, out)) return nullptr;
    return dup_floats(out, out_len);
}

void ppocr_floats_free(float* data) {
    std::free(data);
}

} // extern "C"
