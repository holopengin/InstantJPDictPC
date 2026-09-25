//! Pure ruby-gutter trim shared by the desktop engine and the mobile shim.
//!
//! The desktop used to implement this stage in `ocr_engine.rs`, where the
//! input was an `image::DynamicImage` and the candidate crop was extracted
//! there.  The image-dependent part is now only the conversion to row-major
//! luminance: the algorithm below takes that buffer and its dimensions, so it
//! is available with the nav-only (`--no-default-features`) build as well.
//!
//! `luminance_from_rgb` is the same channel-mean conversion used by the
//! character-placement stage.  The desktop adapter converts a full
//! `DynamicImage` once; the Android shim converts Bitmap ARGB pixels at the
//! UniFFI boundary.  Both then call the identical pure functions here.

use crate::furigana::is_vertical_box;
use crate::models::{BoundingBox, RotatedBox};

// Keep the channel-mean conversion in one place.  `char_placement` is already
// the ungated owner of the reference RGB8 → luminance adapter; re-exporting
// it here lets both image adapters use the exact same conversion without
// making this module depend on an image type.
pub use crate::char_placement::luminance_from_rgb;

/// The clipped, image-space crop bounds used by the old `DynamicImage` code.
///
/// Returning `None` for a malformed/short buffer keeps the pure boundary
/// deterministic and panic-free.  Valid detector boxes follow exactly the
/// old clamp order (`x0`/`y0` use the last valid pixel, while the exclusive
/// edges use the image extent).
fn crop_bounds(
    b: &BoundingBox,
    luminance: &[f32],
    image_width: u32,
    image_height: u32,
) -> Option<(i32, i32, usize, usize)> {
    let iw = i32::try_from(image_width).ok()?;
    let ih = i32::try_from(image_height).ok()?;
    if iw <= 0 || ih <= 0 {
        return None;
    }
    let expected = (image_width as usize).checked_mul(image_height as usize)?;
    if luminance.len() < expected {
        return None;
    }

    let x0 = b.x.clamp(0, iw - 1);
    let x1 = b.x.checked_add(b.w)?.clamp(1, iw);
    let y0 = b.y.clamp(0, ih - 1);
    let y1 = b.y.checked_add(b.h)?.clamp(1, ih);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }

    Some((x0, y0, (x1 - x0) as usize, (y1 - y0) as usize))
}

/// Mobile `findRubyGutterCut` (#48): return the image-space x coordinate at
/// which a ruby-widened vertical box should be cut, or `None` to keep it.
///
/// `luminance` is the full image in row-major order.  The candidate is
/// clipped to the image and converted to a crop-local buffer, then the
/// original per-column ink-fraction rule is applied unchanged: the leftmost
/// clean gutter with ink following it wins, with a thin-spot half-width
/// fallback for touching ruby.
pub fn find_ruby_gutter_cut(
    b: &BoundingBox,
    luminance: &[f32],
    image_width: u32,
    image_height: u32,
) -> Option<i32> {
    let (x0, y0, bw, bh) = crop_bounds(b, luminance, image_width, image_height)?;
    if bw < 24 || bh < 64 {
        return None;
    }

    let iw = image_width as usize;
    let mut crop = Vec::with_capacity(bw * bh);
    for y in 0..bh {
        let row = (y0 as usize + y) * iw + x0 as usize;
        crop.extend_from_slice(&luminance[row..row + bw]);
    }

    // Background polarity from border samples (shared with snapping).
    let mut border: Vec<f32> = Vec::new();
    let mut bi = 0usize;
    while bi < bw {
        border.push(crop[bi]);
        border.push(crop[(bh - 1) * bw + bi]);
        bi += 7;
    }
    bi = 0;
    while bi < bh {
        border.push(crop[bi * bw]);
        border.push(crop[bi * bw + bw - 1]);
        bi += 7;
    }
    if border.is_empty() {
        return None;
    }
    border.sort_by(f32::total_cmp);
    let bg_light = border[border.len() / 2] > 128.0;
    let is_ink = |v: f32| if bg_light { v < 110.0 } else { v > 145.0 };

    // Per-column ink fraction over the full box height.
    let frac: Vec<f32> = (0..bw)
        .map(|x| {
            let mut m = 0usize;
            for y in 0..bh {
                if is_ink(crop[y * bw + x]) {
                    m += 1;
                }
            }
            m as f32 / bh as f32
        })
        .collect();
    let lo = ((bw as f32 * 0.40) as usize).min(bw - 1);
    let hi = (((bw as f32 * 0.80) as usize).max(lo + 1)).min(bw);

    // Leftmost clean gutter with ink following it (not trailing padding).
    let mut x = lo;
    while x + 2 < hi {
        if frac[x] < 0.04 && frac[x + 1] < 0.04 && frac[x + 2] < 0.04 {
            let mut follows = false;
            for k in x + 3..(x + 11).min(bw) {
                if frac[k] >= 0.04 {
                    follows = true;
                    break;
                }
            }
            if follows {
                return Some(x0 + x as i32);
            }
            x += 3;
        } else {
            x += 1;
        }
    }

    // Touching-ruby fallback: thin spot → half width ("remove the right half").
    let mut min_f = f32::MAX;
    for k in lo..hi {
        min_f = min_f.min(frac[k]);
    }
    if min_f < 0.06 {
        return Some(x0 + (bw / 2) as i32);
    }
    None
}

/// Mobile `trimRubyGutterVertical` (#48): trim ruby-widened vertical boxes and
/// keep their axis-aligned quad geometry in sync.  Returns the number of
/// boxes changed.
///
/// This is the same candidate/median/guard walk as the desktop's native
/// implementation.  It deliberately does not read a feature flag: the
/// desktop keeps its `RUBY_TRIM_VERTICAL` environment gate at the call site,
/// while the Android facade keeps its preference gate there as well.  Platform
/// diagnostics stay in the callers so this shared stage remains side-effect
/// free for the mobile binding.
pub fn trim_ruby_gutter_vertical(
    pairs: &mut Vec<(BoundingBox, RotatedBox)>,
    luminance: &[f32],
    image_width: u32,
    image_height: u32,
) -> usize {
    let mut vert_w: Vec<i32> = pairs
        .iter()
        .filter(|(b, _)| is_vertical_box(b))
        .map(|(b, _)| b.w)
        .collect();
    if vert_w.len() < 2 {
        return 0;
    }
    vert_w.sort_unstable();
    let med_w = vert_w[vert_w.len() / 2];
    if med_w <= 0 {
        return 0;
    }

    let mut trimmed = 0usize;
    for (b, r) in pairs.iter_mut() {
        if r.is_rotated() || !is_vertical_box(b) {
            continue;
        }
        let w = b.w;
        if (w as f32) <= med_w as f32 * 1.35 || w - med_w < 12 {
            continue;
        }
        let Some(cut) = find_ruby_gutter_cut(b, luminance, image_width, image_height) else {
            continue;
        };
        if cut <= b.x + 20 || cut >= b.x + b.w - 8 {
            continue;
        }
        if ((cut - b.x) as f32) < (w as f32 * 0.4).round() {
            continue;
        }

        let delta = (b.x + b.w - cut) as f32;
        b.w = cut - b.x;
        // Non-rotated vertical: the cross axis is the frame's local x; keep
        // the quad in sync so crops/char boxes follow the trim.
        r.w -= delta;
        r.cx -= delta / 2.0;
        trimmed += 1;
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bb(x: i32, y: i32, w: i32, h: i32) -> BoundingBox {
        BoundingBox::new(x, y, w, h, 1.0)
    }

    fn frame(b: &BoundingBox) -> RotatedBox {
        RotatedBox::new(
            b.x as f32 + b.w as f32 / 2.0,
            b.y as f32 + b.h as f32 / 2.0,
            b.w as f32,
            b.h as f32,
            0.0,
            1.0,
        )
    }

    /// The PC `ruby_gutter_trim_cuts_the_ruby_side` fixture expressed as
    /// luminance: a main column at x=30..56, ruby at x=66..90, and two
    /// normal columns establish the median width.
    fn cut_fixture() -> Vec<f32> {
        let (w, h) = (400usize, 300usize);
        let mut lum = vec![255.0f32; w * h];
        for y in 10..230 {
            for x in 30..56 {
                lum[y * w + x] = 0.0;
            }
            for x in 66..90 {
                lum[y * w + x] = 0.0;
            }
            for x in 200..260 {
                lum[y * w + x] = 0.0;
            }
            for x in 300..360 {
                lum[y * w + x] = 0.0;
            }
        }
        lum
    }

    /// A white-border / black-interior crop has light background polarity and
    /// ink in every column, so neither the gutter nor the thin-spot fallback
    /// fires.
    fn no_cut_fixture() -> Vec<f32> {
        let (w, h) = (400usize, 300usize);
        let mut lum = vec![0.0f32; w * h];
        let mut x = 0;
        while x < w {
            lum[x] = 255.0;
            lum[(h - 1) * w + x] = 255.0;
            x += 7;
        }
        let mut y = 0;
        while y < h {
            lum[y * w] = 255.0;
            lum[y * w + w - 1] = 255.0;
            y += 7;
        }
        lum
    }

    #[test]
    fn ruby_gutter_trim_cuts_the_ruby_side() {
        let lum = cut_fixture();
        let mut pairs = vec![
            (bb(0, 0, 120, 240), frame(&bb(0, 0, 120, 240))),
            (bb(200, 0, 60, 240), frame(&bb(200, 0, 60, 240))),
            (bb(300, 0, 60, 240), frame(&bb(300, 0, 60, 240))),
        ];
        assert_eq!(find_ruby_gutter_cut(&pairs[0].0, &lum, 400, 300), Some(56));
        assert_eq!(trim_ruby_gutter_vertical(&mut pairs, &lum, 400, 300), 1);
        assert_eq!(pairs[0].0.w, 56);
        assert!((pairs[0].1.w - 56.0).abs() < 0.01);
        assert!((pairs[0].1.cx - 28.0).abs() < 0.01);
        assert_eq!(pairs[1].0.w, 60);
        assert_eq!(pairs[2].0.w, 60);
    }

    #[test]
    fn solid_wide_box_has_no_cut() {
        let lum = no_cut_fixture();
        let b = bb(0, 0, 120, 240);
        assert_eq!(find_ruby_gutter_cut(&b, &lum, 400, 300), None);
        let mut pairs = vec![
            (b.clone(), frame(&b)),
            (bb(200, 0, 60, 240), frame(&bb(200, 0, 60, 240))),
            (bb(300, 0, 60, 240), frame(&bb(300, 0, 60, 240))),
        ];
        assert_eq!(trim_ruby_gutter_vertical(&mut pairs, &lum, 400, 300), 0);
        assert_eq!(pairs[0].0.w, 120);
    }
}
