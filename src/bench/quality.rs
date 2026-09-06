//! Quality metrics: PSNR, SSIM
//!
//! # Everything here is measured on the output grid, not on floats
//!
//! The encoder holds its reconstruction as `f32`, but what a viewer sees is what `pack_u8.wgsl`
//! writes: `u32(clamp(f + 0.5, 0.0, peak))`. Those are not the same signal, and comparing the
//! float one overstates or understates quality depending on the content:
//!
//! - **Clamping** helps. A reconstruction that overshoots 255 in a bright sky is pulled back to
//!   255, which is closer to the source than the float value was.
//! - **Rounding** hurts. Quantising to integers adds up to half a level of error the float
//!   reconstruction does not have.
//!
//! Which one wins depends on how far the reconstruction leaves the valid range versus how much
//! rounding costs, so the error changes *sign* with quality. Measured on blue_sky, 10 frames,
//! last frame of the GOP, encoder-internal PSNR against the real decoded output: **+0.22 dB at
//! q=25, +0.36 at q=50, +0.23 at q=75, −0.53 at q=90.** That was filed as BUG-8, a suspected
//! encoder/decoder drift; it is not drift, and there is no divergence in the reference buffers.
//! It was this metric measuring a reconstruction that never leaves the encoder.
//!
//! It matters because these numbers drive rate control and any RD decision, and at contribution
//! quality — where GNC is aimed — the old metric *overstated* quality by half a dB.
//!
//! So every function here quantises both inputs to the output grid first. For a source loaded
//! from an 8-bit PNG that is a no-op; for a float reconstruction it is the difference between
//! measuring the codec and measuring an intermediate.

/// Quantise a sample to the integer grid the decoder will actually emit.
///
/// Must match `pack_u8.wgsl` / `pack_u16.wgsl` exactly, including the order: add the rounding
/// offset, *then* clamp, then truncate. `peak` is `255.0` for 8-bit and `1023.0` for 10-bit.
#[inline]
pub fn to_output_grid(v: f32, peak: f32) -> f64 {
    ((v + 0.5).clamp(0.0, peak)).floor() as f64
}

/// /// Compute Peak Signal-to-Noise Ratio between original and reconstructed frames.
/// Both inputs are f32 slices in [0, peak_val] range, same length (W*H*3 interleaved).
pub fn psnr(original: &[f32], reconstructed: &[f32], peak_val: f32) -> f64 {
    assert_eq!(original.len(), reconstructed.len());
    let n = original.len() as f64;

    let mse: f64 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| {
            let diff = to_output_grid(*a, peak_val) - to_output_grid(*b, peak_val);
            diff * diff
        })
        .sum::<f64>()
        / n;

    if mse < 1e-10 {
        return f64::INFINITY; // identical
    }

    let peak = peak_val as f64;
    10.0 * (peak * peak / mse).log10()
}

/// Compute per-channel PSNR. Returns (Y_psnr, Co_psnr, Cg_psnr) or (R, G, B).
pub fn psnr_per_channel(original: &[f32], reconstructed: &[f32], peak_val: f32) -> (f64, f64, f64) {
    assert_eq!(original.len(), reconstructed.len());
    assert_eq!(original.len() % 3, 0);
    let pixels = original.len() / 3;

    let mut mse = [0.0f64; 3];
    for i in 0..pixels {
        for c in 0..3 {
            let diff = to_output_grid(original[i * 3 + c], peak_val)
                - to_output_grid(reconstructed[i * 3 + c], peak_val);
            mse[c] += diff * diff;
        }
    }

    let peak = peak_val as f64;
    let peak_sq = peak * peak;
    let n = pixels as f64;

    let to_psnr = |m: f64| -> f64 {
        if m < 1e-10 {
            f64::INFINITY
        } else {
            10.0 * (peak_sq / (m / n)).log10()
        }
    };

    (to_psnr(mse[0]), to_psnr(mse[1]), to_psnr(mse[2]))
}

/// Simplified SSIM computation (per-pixel luminance only, no windowing).
/// This is a rough approximation — for proper SSIM, use windowed computation.
/// Computes SSIM on the luminance channel (first of every 3 values).
pub fn ssim_approx(original: &[f32], reconstructed: &[f32], peak_val: f32) -> f64 {
    assert_eq!(original.len(), reconstructed.len());
    assert_eq!(original.len() % 3, 0);
    let pixels = original.len() / 3;

    let l = peak_val as f64;
    let c1 = (0.01 * l) * (0.01 * l);
    let c2 = (0.03 * l) * (0.03 * l);

    // Compute over 8x8 blocks for a rough windowed SSIM
    // For simplicity, compute global SSIM
    let mut mu_x = 0.0f64;
    let mut mu_y = 0.0f64;
    let mut sigma_x2 = 0.0f64;
    let mut sigma_y2 = 0.0f64;
    let mut sigma_xy = 0.0f64;

    // Use luminance channel (index 0 of each triple)
    for i in 0..pixels {
        let x = to_output_grid(original[i * 3], peak_val);
        let y = to_output_grid(reconstructed[i * 3], peak_val);
        mu_x += x;
        mu_y += y;
    }
    let n = pixels as f64;
    mu_x /= n;
    mu_y /= n;

    for i in 0..pixels {
        let x = to_output_grid(original[i * 3], peak_val);
        let y = to_output_grid(reconstructed[i * 3], peak_val);
        sigma_x2 += (x - mu_x) * (x - mu_x);
        sigma_y2 += (y - mu_y) * (y - mu_y);
        sigma_xy += (x - mu_x) * (y - mu_y);
    }
    sigma_x2 /= n;
    sigma_y2 /= n;
    sigma_xy /= n;

    let numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2);
    let denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x2 + sigma_y2 + c2);

    numerator / denominator
}

/// Compute PSNR separately for pixels near tile boundaries vs interior pixels.
///
/// `halo` is the number of pixels on each side of a tile edge considered "boundary".
/// Returns `(boundary_psnr, interior_psnr)`.  A large gap indicates tile-edge artifacts.
/// Used as a gate experiment for #47 (overlapping tile windows).
pub fn psnr_tile_boundary(
    original: &[f32],
    reconstructed: &[f32],
    peak_val: f32,
    width: usize,
    height: usize,
    tile_size: usize,
    halo: usize,
) -> (f64, f64) {
    assert_eq!(original.len(), reconstructed.len());
    assert_eq!(original.len(), width * height * 3);

    let mut mse_boundary = 0.0f64;
    let mut n_boundary = 0usize;
    let mut mse_interior = 0.0f64;
    let mut n_interior = 0usize;

    for y in 0..height {
        let near_y_edge = (y % tile_size) < halo || (tile_size - y % tile_size) <= halo;
        for x in 0..width {
            let near_x_edge = (x % tile_size) < halo || (tile_size - x % tile_size) <= halo;
            let is_boundary = near_x_edge || near_y_edge;
            let pixel_idx = (y * width + x) * 3;
            let mut sq_err = 0.0f64;
            for c in 0..3 {
                let diff = to_output_grid(original[pixel_idx + c], peak_val)
                    - to_output_grid(reconstructed[pixel_idx + c], peak_val);
                sq_err += diff * diff;
            }
            if is_boundary {
                mse_boundary += sq_err;
                n_boundary += 3;
            } else {
                mse_interior += sq_err;
                n_interior += 3;
            }
        }
    }

    let peak = peak_val as f64;
    let peak_sq = peak * peak;
    let to_psnr = |mse: f64, n: usize| -> f64 {
        if n == 0 { return f64::NAN; }
        let m = mse / n as f64;
        if m < 1e-10 { f64::INFINITY } else { 10.0 * (peak_sq / m).log10() }
    };

    (to_psnr(mse_boundary, n_boundary), to_psnr(mse_interior, n_interior))
}

/// Result of quality measurement
#[derive(Debug, Clone, serde::Serialize)]
pub struct QualityMetrics {
    pub psnr_db: f64,
    pub psnr_r: f64,
    pub psnr_g: f64,
    pub psnr_b: f64,
    pub ssim: f64,
    pub bpp: f64,
    pub compressed_bytes: usize,
}

impl std::fmt::Display for QualityMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PSNR: {:.2} dB (R:{:.2} G:{:.2} B:{:.2}) | SSIM: {:.4} | BPP: {:.2} | Size: {} bytes",
            self.psnr_db,
            self.psnr_r,
            self.psnr_g,
            self.psnr_b,
            self.ssim,
            self.bpp,
            self.compressed_bytes
        )
    }
}
