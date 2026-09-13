//! Rust bindings for the shared PP-OCRv6 ncnn core.
//!
//! The core (`native/ppocr_ncnn/ppocr_ncnn_core.{h,cpp}`) is the same file
//! the Android app compiles (`InstantJPDict/app/src/main/cpp`). This module
//! only owns handles and moves float buffers; any change to how the models
//! are loaded or run belongs in the core, so both apps keep one backend.
//!
//! `build.rs` compiles the core + C ABI and links the pinned ncnn fork from
//! `third_party/ncnn-pc/install` (see `tools/build_ncnn_pc.sh`).

use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::path::Path;

use anyhow::{bail, Context, Result};

#[allow(non_camel_case_types)]
mod ffi {
    use super::*;

    pub enum ppocr_rec_t {}
    pub enum ppocr_det_t {}

    extern "C" {
        pub fn ppocr_rec_create(
            param_path: *const c_char,
            bin_path: *const c_char,
            target_w: c_int,
            num_threads: c_int,
        ) -> *mut ppocr_rec_t;
        pub fn ppocr_rec_destroy(rec: *mut ppocr_rec_t);
        pub fn ppocr_rec_infer(
            rec: *mut ppocr_rec_t,
            data: *const f32,
            data_floats: usize,
            w: c_int,
            h: c_int,
            out_len: *mut c_int,
        ) -> *mut f32;
        pub fn ppocr_rec_infer_topk(
            rec: *mut ppocr_rec_t,
            data: *const f32,
            data_floats: usize,
            w: c_int,
            h: c_int,
            out_len: *mut c_int,
        ) -> *mut f32;
        pub fn ppocr_rec_top_k() -> c_int;

        pub fn ppocr_det_create(param_path: *const c_char, bin_path: *const c_char) -> *mut ppocr_det_t;
        pub fn ppocr_det_destroy(det: *mut ppocr_det_t);
        pub fn ppocr_det_infer(
            det: *mut ppocr_det_t,
            data: *const f32,
            data_floats: usize,
            w: c_int,
            h: c_int,
            out_len: *mut c_int,
        ) -> *mut f32;

        pub fn ppocr_floats_free(data: *mut f32);
    }
}

/// Top-K per CTC timestep emitted by [`RecNet::infer_topk`]; matches the
/// mobile app's `OcrEngine.TOP_K`.
pub fn top_k() -> usize {
    unsafe { ffi::ppocr_rec_top_k() as usize }
}

fn cstr(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .with_context(|| format!("model path contains NUL: {}", path.display()))
}

/// Take ownership of a core-allocated float buffer, copying it into a Vec.
unsafe fn take_floats(ptr: *mut f32, len: c_int) -> Result<Vec<f32>> {
    if ptr.is_null() || len < 0 {
        bail!("ncnn inference failed (see the [ppocr_ncnn] log line above)");
    }
    let out = std::slice::from_raw_parts(ptr, len as usize).to_vec();
    ffi::ppocr_floats_free(ptr);
    Ok(out)
}

/// Dynamic-width CTC recognizer (rec_dyn, 48px tall, [1,3,48,w] input).
pub struct RecNet {
    ptr: *mut ffi::ppocr_rec_t,
}

// ncnn `Net::create_extractor()` is const and gives every call its own
// extractor state, so one loaded net can serve several recognition workers.
unsafe impl Send for RecNet {}
unsafe impl Sync for RecNet {}

impl RecNet {
    /// Load `param`/`bin`. `target_w` only seeds the handle's metadata —
    /// inference length comes from the actual input width (#23).
    pub fn create(param: &Path, bin: &Path, target_w: i32, num_threads: i32) -> Result<Self> {
        let param = cstr(param)?;
        let bin = cstr(bin)?;
        let ptr = unsafe {
            ffi::ppocr_rec_create(param.as_ptr(), bin.as_ptr(), target_w, num_threads.max(1))
        };
        if ptr.is_null() {
            bail!(
                "failed to load rec ncnn model {} / {}",
                param.to_string_lossy(),
                bin.to_string_lossy()
            );
        }
        Ok(RecNet { ptr })
    }

    /// Full-logits inference: seq_len * num_classes floats.
    pub fn infer(&self, input: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
        let mut len: c_int = -1;
        let ptr = unsafe {
            ffi::ppocr_rec_infer(
                self.ptr,
                input.as_ptr(),
                input.len(),
                w as c_int,
                h as c_int,
                &mut len,
            )
        };
        unsafe { take_floats(ptr, len) }
    }

    /// Packed top-K inference: seq_len * K * 2 floats ([idx,val] pairs).
    pub fn infer_topk(&self, input: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
        let mut len: c_int = -1;
        let ptr = unsafe {
            ffi::ppocr_rec_infer_topk(
                self.ptr,
                input.as_ptr(),
                input.len(),
                w as c_int,
                h as c_int,
                &mut len,
            )
        };
        unsafe { take_floats(ptr, len) }
    }
}

impl Drop for RecNet {
    fn drop(&mut self) {
        unsafe { ffi::ppocr_rec_destroy(self.ptr) }
    }
}

/// DB segmentation detector ([1,3,h,w] input, probability map out).
pub struct DetNet {
    ptr: *mut ffi::ppocr_det_t,
}

unsafe impl Send for DetNet {}

impl DetNet {
    pub fn create(param: &Path, bin: &Path) -> Result<Self> {
        let param = cstr(param)?;
        let bin = cstr(bin)?;
        let ptr = unsafe { ffi::ppocr_det_create(param.as_ptr(), bin.as_ptr()) };
        if ptr.is_null() {
            bail!(
                "failed to load det ncnn model {} / {}",
                param.to_string_lossy(),
                bin.to_string_lossy()
            );
        }
        Ok(DetNet { ptr })
    }

    /// `&mut self` because the core caches the first working output blob name.
    pub fn infer(&mut self, input: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
        let mut len: c_int = -1;
        let ptr = unsafe {
            ffi::ppocr_det_infer(
                self.ptr,
                input.as_ptr(),
                input.len(),
                w as c_int,
                h as c_int,
                &mut len,
            )
        };
        unsafe { take_floats(ptr, len) }
    }
}

impl Drop for DetNet {
    fn drop(&mut self) {
        unsafe { ffi::ppocr_det_destroy(self.ptr) }
    }
}
