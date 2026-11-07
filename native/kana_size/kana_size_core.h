// Platform-independent kanadiff kana-size ncnn inference core.
//
// This file is the single source of truth for the small/large kana model
// (#44) and is compiled by BOTH apps:
//   - Android (InstantJPDict): kana_size_ncnn.cpp is a thin JNI wrapper.
//   - PC (InstantJPDictPC): native/kana_size/kana_size_capi.cpp is a thin
//     C ABI wrapper.
//
// Keep this file and kana_size_core.cpp byte-identical between the two
// repositories (PC: tools/sync_ppocr_core.sh can check/update the copy).
//
// The graph and the pinned net options are load-bearing: fp16 costs
// ~1.5e-03 against the fp32 reference on this model, so the options are
// pinned deliberately and must not be relaxed.
#pragma once

#include <vector>

namespace kana_size {

/// Window bytes per position.
constexpr int SEQ = 40;
/// Pair-table size, i.e. valid `base` values.
constexpr int PAIRS = 20;
/// 40 positions of a 47k-param net: threads are not the lever.
constexpr int THREADS = 2;

/// Opaque model handle; the ncnn type stays out of this header.
struct Model;

/// Load the converted `nb_all` param/bin with the fp32 options pinned.
/// Returns nullptr on failure.
Model* create(const char* param_path, const char* bin_path);

void destroy(Model* model);

/// Logits for `n` positions. `win` holds n*SEQ int32 byte values (from the
/// window encoder), `bases` n pair indices; an out-of-range base is clamped
/// to 0 like the historical mobile wrapper. Returns false on failure.
bool logits(Model* model, const int* win, const int* bases, int n, std::vector<float>& out);

} // namespace kana_size
