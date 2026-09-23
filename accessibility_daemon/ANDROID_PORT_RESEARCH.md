# Android Port Assessment — InstantJPDictPC

## Summary

Porting to Android is feasible. None of the core dependencies are Android-incompatible by nature, but several need feature-flag or configuration changes, and the GUI shell needs a different entry point than the current desktop-native main(). Below is a component-by-component analysis.

---

## 1. GUI: Iced (0.14.0) — the hard part

**State**: No official Android support. Maintainer (hecrj) has explicitly declined to maintain mobile targets ([#3099](https://github.com/iced-rs/iced/pull/3099), [#3245](https://github.com/iced-rs/iced/pull/3245)). However, the community has working solutions:

- **`iced_mobile`** crate (Leinnan, [github](https://github.com/Leinnan/iced_mobile)) — thin wrapper that replaces `run()` with `mobile_run()`. Requires calling `iced_mobile::MobileAppRunner` trait method instead of `Sandbox::run()`.

- **`android-iced-example`** (ibaryshnikov, [github](https://github.com/ibaryshnikov/android-iced-example)) — demonstrates Iced 0.14 on Android via both NativeActivity and GameActivity (`android-activity` crate). Already updated to Iced 0.14 + wgpu 27.

**Changes required:**

| File | Change |
|------|--------|
| `Cargo.toml` | Remove `"x11"` from Iced features; add `"wgpu"` only (wgpu already targets Android via Vulkan). |
| `Cargo.toml` | Add `iced_mobile` dependency (or vendor the ~50-line adaptation). |
| `src/main.rs` | Add a second entry point behind `#[cfg(target_os = "android")]` that uses `iced_mobile::mobile_run()` or the `android-activity` integration pattern. The current `main()` stays for desktop. |
| New file: `src/android_main.rs` | Thin wrapper that calls the same `OcrViewer` logic but through the Android activity loop. |

**The `x11` feature** must be removed for Android builds (`iced` enables it by default on Linux). Switch to a conditional: `features = ["wgpu"]` on Android; `features = ["wgpu", "x11"]` on Linux.

**Winit** works on Android via `android-activity` — this is what `iced_mobile` and `android-iced-example` use under the hood.

---

## 2. GPU Backend: wgpu — works

wgpu 0.20+ (used by Iced 0.14) supports Android via Vulkan. Adreno GPUs (Qualcomm), Mali (ARM), and PowerVR are all Vulkan-capable. The Vulkan loader is bundled with Android.

**No change needed** — just remove `x11` from Iced's feature list and let wgpu pick the Vulkan backend.

---

## 3. ONNX Runtime / `ort` crate — works

- ONNX Runtime provides prebuilt `.so` binaries for Android (arm64-v8a, armeabi-v7a, x86_64).
- The `ort` crate has `api-24` through `api-17` feature flags for Android API targeting.
- The `nnapi` execution provider feature exists — NNAPI accelerates on-device ML on Android.
- `xnnpack` execution provider is already enabled and works on Android.
- `download-binaries` flag downloads the correct binary for the target triple.

**Changes required:**

| File | Change |
|------|--------|
| `Cargo.toml` | Replace `"pkg-config"` with conditional — Linux needs it, Android doesn't. Use `cfg(not(target_os = "android"))` gate. |
| `Cargo.toml` | Consider adding `"nnapi"` feature for Android hardware acceleration. |
| `Cargo.toml` | The `"download-binaries"` feature should auto-detect Android triples when cross-compiling with `cargo-ndk`. |

Note: `ort` today uses `"pkg-config"` which won't work during Android cross-compilation. The `ort-sys` crate supports `"disable-linking"` + `"preload-dylibs"` for dynamic loading on Android. This is the same pattern used for iOS.

---

## 4. Image Processing — no change

| Crate | Status |
|-------|--------|
| `image` 0.25 | Pure Rust — compiles on any target. |
| `imageproc` 0.25 | Pure Rust — compiles on any target. |
| `ndarray` 0.16 | Pure Rust — compiles on any target. |

**No change needed.**

---

## 5. Font Rendering — no change

| Crate | Status |
|-------|--------|
| `fontdue` 0.9 | Pure Rust software renderer — compiles on any target. |

**No change needed.** Fontdue is a software rasterizer with no system font library dependency. The bundled `NotoSansJP-Regular.ttf` ships with the APK assets.

---

## 6. Database — no change

| Crate | Status |
|-------|--------|
| `rusqlite` 0.37 + `bundled` | The `bundled` feature compiles SQLite from C source. Works on Android. |

**No change needed.**

---

## 7. Platform-Specific Crate — needs gating

| Crate | Status |
|-------|--------|
| `evdev` 0.13 | **Linux-only**. Uses `/dev/input/` — not available on Android. |
| `rfd` 0.14 | File dialogs. May need `android` feature or gating. |
| `directories` 6.0 | Works on Android (uses `ANDROID_DATA` / `ANDROID_ROOT`). |
| `open` 5.0 | Opens URLs. Works on Android (uses `am start` intent under the hood). |
| `x11` (iced feature) | Must be conditionally disabled. |

**Changes required:**

```toml
[target.'cfg(target_os = "linux")'.dependencies]
evdev = "0.13"

[target.'cfg(not(target_os = "android"))'.dependencies]
rfd = "0.14"
```

---

## 8. Build System

**Desktop**: `cargo build --features ort` (unchanged).

**Android**: Uses `cargo-ndk`:
```
cargo install cargo-ndk
rustup target add aarch64-linux-android
cargo ndk -t arm64-v8a -o app/src/main/jniLibs/ build --features ort
```

Then either:
- **NativeActivity**: Gradle project with `AndroidManifest.xml` + `cargo-ndk` build step.
- **GameActivity** (preferred): Google's modern `GameActivity` (from `android-activity` crate), better lifecycle handling.

The `android-iced-example` repo's `GameActivity/` directory has a working `build.gradle.kts` and `AndroidManifest.xml` to use as a template.

---

## 9. Model Files

PP-OCRv6 ONNX models (~30 MB) need to ship inside the APK:
- Place in `app/src/main/assets/` 
- Load via `android_content_provider` or `AssetManager` at runtime
- OR download on first launch (adds complexity but keeps APK smaller)

---

## 10. Incremental Plan

```
Phase 1 — Build infrastructure (1-2 days)
├── Gate Linux-only deps (evdev → cfg(target_os = "linux"))
├── Remove "x11" from iced features, make conditional
├── Replace "pkg-config" in ort features with cfg gate
├── Add android-activity and iced_mobile deps
├── Set up cross-compile toolchain (cargo-ndk + Android NDK)
└── Verify `cargo ndk build` succeeds (even if the app doesn't display)

Phase 2 — Android entry point (2-3 days)
├── Create android_main.rs as second entry point
├── Wire up iced_mobile::mobile_run or android-activity integration
├── Handle lifecycle (pause/resume — Android destroys GL surfaces)
├── Asset loading from APK (model files, dictionary, font)
└── Verify the app window appears with the screenshot

Phase 3 — Polish (1-2 days)
├── Touch input: verify pinch-zoom and tap-to-select work
├── Soft keyboard: integrate IME for text input alternatives? 
├── NNAPI acceleration for ORT (optional perf boost)
├── File picker via Android Storage Access Framework (if needed)
└── APK packaging and signing

Phase 4 — Desktop regression test
└── Verify `cargo build --features ort` still works unchanged
```

---

## Blockers / Risks

1. **Iced maintainer won't support mobile** — the `iced_mobile` crate is a third-party wrapper that could lag behind Iced releases. Vendor it or fork if needed.

2. **ORT `download-binaries` on Android** — need to verify `ort-sys` correctly fetches `aarch64-linux-android` binaries. The `pkg-config` path definitely won't work; `load-dynamic` is the safe alternative.

3. **Texture atlas corruption** — the existing code has a `wait_for_atlas` hack for desktop wgpu. The same issue may appear on Android's Vulkan driver; might need the same workaround.

4. **Keyboard input** — the PR #3245 notes "keyboard does not show up on Android when text input is pressed." The current app doesn't use text input, so this isn't a blocker.

5. **`directories` crate** — uses environment variables on Android; works but paths differ from Linux. Verify `dictionary.sqlite` location.
