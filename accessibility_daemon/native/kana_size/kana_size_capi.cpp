// The kanadiff kana-size model, run through the same pinned ncnn fork the OCR
// models use.
//
// Ported from the Android app's app/src/main/cpp/kana_size_ncnn.cpp (JNI
// wrapper); the graph walk and the pinned options are kept identical because
// the numerics are the point. Three conversion defects in the mobile history
// are worth remembering, since all three produce *plausible* output rather
// than an error:
//
//   * an int64 initializer once shifted every later weight block by one float;
//   * `Convolution1D` is 2-D and reads a 3-D input as one channel;
//   * fp16 costs ~1.5e-03 against the fp32 reference. The options are pinned
//     below, deliberately.
//
// Graph, as the param expresses it:
//
//   win  : 40 int32 byte values -> byte_emb (256x32)          -> [40,32]
//   pos  : MemoryData 0..39     -> pos_block (40x16)          -> [40,16]
//   cat  : concat -> [40,48] ; permute -> [48,40] channels-first
//   conv1: 48->64 k=5 pad=2 relu ; conv2: 64->64 k=3 pad=1 relu ; max over positions -> [64]
//   base : int32 pair index -> base_emb (20x16) -> [16] ; concat -> [80]
//   fc1  : 80->128 relu ; fc2: 128->1 -> logit          (sigmoid gives p(big))
//
// Positions are run one per extractor call, exactly as mobile does: the param
// declares no batch dimension, and one position per run keeps the int32 Embed
// inputs and the position-axis Reduction unambiguous.
#include "kana_size_capi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ncnn/net.h"

namespace {

constexpr int SEQ = 40;     // window bytes per position
constexpr int PAIRS = 20;   // pair-table size, i.e. valid `base` values
constexpr int THREADS = 2;  // 40 positions of a 47k-param net: threads are not the lever

struct Model {
    ncnn::Net net;
};

}  // namespace

extern "C" {

kana_size_t* kana_size_create(const char* param_path, const char* bin_path) {
    if (param_path == nullptr || bin_path == nullptr || *param_path == '\0' || *bin_path == '\0') {
        std::fprintf(stderr, "[kana_size] create: empty path\n");
        return nullptr;
    }

    auto* m = new Model();
    m->net.opt.use_fp16_packed = false;
    m->net.opt.use_fp16_storage = false;
    m->net.opt.use_fp16_arithmetic = false;
    m->net.opt.use_bf16_storage = false;
    m->net.opt.use_packing_layout = false;
    m->net.opt.num_threads = THREADS;

    if (m->net.load_param(param_path) != 0) {
        std::fprintf(stderr, "[kana_size] load_param failed: %s\n", param_path);
        delete m;
        return nullptr;
    }
    if (m->net.load_model(bin_path) != 0) {
        std::fprintf(stderr, "[kana_size] load_model failed: %s\n", bin_path);
        delete m;
        return nullptr;
    }
    std::fprintf(stderr, "[kana_size] model loaded (ncnn, fp32 pinned): %s\n", param_path);
    return reinterpret_cast<kana_size_t*>(m);
}

void kana_size_destroy(kana_size_t* model) {
    delete reinterpret_cast<Model*>(model);
}

float* kana_size_logits(kana_size_t* model, const int* win, const int* bases, int n, int* out_len) {
    *out_len = -1;
    if (model == nullptr || win == nullptr || bases == nullptr || n <= 0) {
        return nullptr;
    }

    // Copy the byte windows off the caller's buffer: ncnn::Mat wraps memory
    // without owning it, and the mobile side copies through the JNI arrays too.
    std::vector<int> w((size_t) n * SEQ);
    std::memcpy(w.data(), win, sizeof(int) * (size_t) n * SEQ);

    Model* m = reinterpret_cast<Model*>(model);
    std::vector<float> out((size_t) n, 0.0f);
    for (int i = 0; i < n; i++) {
        int base = bases[i];
        if (base < 0 || base >= PAIRS) base = 0;

        // Embed takes int indices, so both inputs are 4-byte elements, not floats.
        ncnn::Mat winMat(SEQ, w.data() + (size_t) i * SEQ, 4u);
        ncnn::Mat baseMat(1, &base, 4u);

        ncnn::Extractor ex = m->net.create_extractor();
        ex.input("win", winMat);
        ex.input("base", baseMat);
        ncnn::Mat logit;
        if (ex.extract("logit", logit) != 0 || logit.w < 1) {
            std::fprintf(stderr, "[kana_size] extract failed at position %d of %d\n", i, n);
            return nullptr;
        }
        out[i] = logit[0];
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

}  // extern "C"
