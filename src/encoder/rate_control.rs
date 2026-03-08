//! Rate controller for CBR and VBR encoding modes.
//!
//! Uses an R-Q model:  bpp ≈ c * qstep^(-alpha)
//! where `c` and `alpha` are estimated online from encoded frame statistics.
//!
//! The controller only adjusts `quantization_step`; it does not touch any
//! other pipeline parameters (wavelet type, entropy coder, etc.).
//!
//! A Video Buffering Verifier (VBV) buffer enforces CBR compliance by
//! clamping the chosen qstep when the buffer fill level gets too high or
//! too low.

use crate::RateMode;

/// A single data point: observed (qstep, bpp) after encoding a frame.
#[derive(Debug, Clone, Copy)]
struct RqSample {
    qstep: f64,
    bpp: f64,
}

/// Rate controller state.
///
/// Create one per sequence; call [`estimate_qstep`] before each frame and
/// [`update`] after encoding.
#[derive(Debug)]
pub struct RateController {
    /// Target bitrate in bits per second.
    target_bitrate: f64,
    /// Frame rate (frames per second) — needed to convert bps -> bits/frame.
    fps: f64,
    /// Frame dimensions (pixels).
    width: u32,
    height: u32,

    /// Rate control mode.
    mode: RateMode,

    // ---- R-Q model: bpp = c * qstep^(-alpha) ----
    /// Model coefficient (log-space intercept).
    model_c: f64,
    /// Model exponent (positive; higher alpha = steeper rate drop per qstep increase).
    model_alpha: f64,

    // ---- VBV (Video Buffering Verifier) ----
    /// VBV buffer capacity in bits.
    vbv_capacity: f64,
    /// Current VBV buffer fill in bits. Starts at 50% capacity.
    vbv_fill: f64,

    /// History of recent (qstep, bpp) samples for model fitting.
    history: Vec<RqSample>,
    /// Maximum history length (sliding window).
    max_history: usize,

    /// Minimum and maximum allowed qstep values.
    qstep_min: f64,
    qstep_max: f64,

    /// Number of frames encoded so far.
    frame_count: u64,
}

impl RateController {
    /// Create a new rate controller.
    ///
    /// - `target_bitrate`: target in bits per second (e.g. 10_000_000 for 10 Mbps).
    /// - `fps`: frame rate of the sequence.
    /// - `width`, `height`: frame dimensions in pixels.
    /// - `mode`: CBR or VBR.
    pub fn new(target_bitrate: f64, fps: f64, width: u32, height: u32, mode: RateMode) -> Self {
        // Initial R-Q model: bpp = c * qstep^(-alpha).
        // Use empirical defaults from typical video content:
        // at qstep=4, typical bpp ~ 0.5 for 1080p content.
        // So c = 0.5 * 4^1.2 ≈ 2.6. These are refined online from actual samples.
        let model_alpha = 1.2;
        let model_c = 2.6;

        // VBV buffer: 1 second of video for CBR, 2 seconds for VBR.
        let vbv_seconds = match mode {
            RateMode::CBR => 1.0,
            RateMode::VBR => 2.0,
        };
        let vbv_capacity = target_bitrate * vbv_seconds;

        Self {
            target_bitrate,
            fps,
            width,
            height,
            mode,
            model_c,
            model_alpha,
            vbv_capacity,
            vbv_fill: vbv_capacity * 0.5, // start half-full
            history: Vec::with_capacity(32),
            max_history: 30,
            qstep_min: 0.5,
            qstep_max: 128.0,
            frame_count: 0,
        }
    }

    /// Target bits per pixel for one frame.
    fn target_bpp(&self) -> f64 {
        let pixels = self.width as f64 * self.height as f64;
        self.target_bitrate / (self.fps * pixels)
    }

    /// Bits budget per frame.
    fn bits_per_frame(&self) -> f64 {
        self.target_bitrate / self.fps
    }

    /// Estimate the quantization step for the next frame.
    ///
    /// Call this before encoding. The returned qstep should be used as
    /// `config.quantization_step`.
    pub fn estimate_qstep(&self) -> f32 {
        let target_bpp = self.target_bpp();

        // Invert the R-Q model: qstep = (c / target_bpp)^(1/alpha)
        let raw_qstep = if target_bpp > 0.0 && self.model_c > 0.0 && self.model_alpha > 0.0 {
            (self.model_c / target_bpp).powf(1.0 / self.model_alpha)
        } else {
            4.0 // safe fallback
        };

        // Clamp to valid range
        let clamped = raw_qstep.clamp(self.qstep_min, self.qstep_max);

        // Apply VBV constraint
        let constrained = self.vbv_constrain(clamped);

        constrained as f32
    }

    /// Apply VBV buffer constraints to the proposed qstep.
    ///
    /// If the buffer is getting too full (risk of overflow), lower qstep (better
    /// quality, more bits used). If too empty (risk of underflow), raise qstep
    /// (lower quality, fewer bits).
    pub fn vbv_constrain(&self, proposed_qstep: f64) -> f64 {
        let fill_ratio = self.vbv_fill / self.vbv_capacity;

        let adjustment = match self.mode {
            RateMode::CBR => {
                // CBR: aggressive VBV enforcement.
                // fill_ratio > 0.8 -> spend more bits (lower qstep)
                // fill_ratio < 0.2 -> spend fewer bits (higher qstep)
                if fill_ratio > 0.8 {
                    // Buffer too full — lower qstep to use more bits
                    let urgency = (fill_ratio - 0.8) / 0.2; // 0..1
                    1.0 / (1.0 + urgency * 0.5) // multiply qstep by 0.67..1.0
                } else if fill_ratio < 0.2 {
                    // Buffer too empty — raise qstep to use fewer bits
                    let urgency = (0.2 - fill_ratio) / 0.2; // 0..1
                    1.0 + urgency * 0.5 // multiply qstep by 1.0..1.5
                } else {
                    1.0
                }
            }
            RateMode::VBR => {
                // VBR: gentler enforcement — only act on extreme fill levels.
                if fill_ratio > 0.9 {
                    let urgency = (fill_ratio - 0.9) / 0.1;
                    1.0 / (1.0 + urgency * 0.3)
                } else if fill_ratio < 0.1 {
                    let urgency = (0.1 - fill_ratio) / 0.1;
                    1.0 + urgency * 0.3
                } else {
                    1.0
                }
            }
        };

        (proposed_qstep * adjustment).clamp(self.qstep_min, self.qstep_max)
    }

    /// Update the rate controller after encoding a frame.
    ///
    /// - `actual_qstep`: the qstep that was actually used.
    /// - `actual_bpp`: the measured bits per pixel of the encoded frame.
    pub fn update(&mut self, actual_qstep: f32, actual_bpp: f64) {
        let qstep = actual_qstep as f64;
        self.frame_count += 1;

        // Record sample
        let sample = RqSample {
            qstep,
            bpp: actual_bpp,
        };
        self.history.push(sample);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        // Update R-Q model via least-squares fit in log-space.
        // ln(bpp) = ln(c) - alpha * ln(qstep)
        self.fit_model();

        // Update VBV buffer: add per-frame budget, subtract actual bits used.
        let frame_bits = actual_bpp * self.width as f64 * self.height as f64;
        self.vbv_fill += self.bits_per_frame(); // incoming budget
        self.vbv_fill -= frame_bits; // consumed by this frame
        self.vbv_fill = self.vbv_fill.clamp(0.0, self.vbv_capacity);
    }

    /// Update rate controller after encoding a GOP of `n_frames` frames.
    ///
    /// `total_bits_bytes` = total encoded bytes for the GOP.
    ///
    /// Unlike calling `update()` n_frames times with identical avg_bpp (which would push
    /// n_frames duplicate samples into history and cause degenerate log-space regression),
    /// this method adds ONE representative sample and advances the VBV budget for all frames.
    pub fn update_gop(&mut self, actual_qstep: f32, total_bits_bytes: usize, n_frames: u32) {
        let pixels_per_frame = self.width as f64 * self.height as f64;
        let total_bits = total_bits_bytes as f64 * 8.0;
        let avg_bpp = total_bits / (pixels_per_frame * n_frames as f64);
        let qstep = actual_qstep as f64;
        self.frame_count += n_frames as u64;

        // Add ONE representative sample to avoid degenerate R-Q regression.
        let sample = RqSample { qstep, bpp: avg_bpp };
        self.history.push(sample);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        self.fit_model();

        // Advance VBV: n_frames of budget in, actual GOP bits out.
        self.vbv_fill += self.bits_per_frame() * n_frames as f64;
        self.vbv_fill -= total_bits;
        self.vbv_fill = self.vbv_fill.clamp(0.0, self.vbv_capacity);
    }

    /// Return current VBV fill ratio (0.0 = empty, 1.0 = full).
    pub fn vbv_fill_ratio(&self) -> f64 {
        self.vbv_fill / self.vbv_capacity
    }

    /// Fit the R-Q model to the sample history using log-space linear regression.
    /// Model: ln(bpp) = ln(c) - alpha * ln(qstep)
    fn fit_model(&mut self) {
        if self.history.len() < 2 {
            // Not enough data to fit; keep initial estimate.
            // But update c from single sample if available.
            if let Some(s) = self.history.last() {
                if s.bpp > 0.0 && s.qstep > 0.0 {
                    self.model_c = s.bpp * s.qstep.powf(self.model_alpha);
                }
            }
            return;
        }

        // Filter out invalid samples
        let valid: Vec<_> = self
            .history
            .iter()
            .filter(|s| s.bpp > 0.0 && s.qstep > 0.0)
            .collect();

        if valid.len() < 2 {
            return;
        }

        // Linear regression in log space: y = a + b*x
        // where y = ln(bpp), x = ln(qstep), a = ln(c), b = -alpha
        let n = valid.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for s in &valid {
            let x = s.qstep.ln();
            let y = s.bpp.ln();
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            // Degenerate case (all same qstep) — just update c from mean bpp.
            let mean_bpp = valid.iter().map(|s| s.bpp).sum::<f64>() / n;
            let mean_qstep = valid.iter().map(|s| s.qstep).sum::<f64>() / n;
            if mean_bpp > 0.0 {
                self.model_c = mean_bpp * mean_qstep.powf(self.model_alpha);
            }
            return;
        }

        let b = (n * sum_xy - sum_x * sum_y) / denom;
        let a = (sum_y - b * sum_x) / n;

        // b = -alpha, a = ln(c)
        let alpha = -b;
        let c = a.exp();

        // Sanity: alpha should be positive (higher qstep -> lower bpp).
        // If regression gives nonsensical results, keep previous model.
        if alpha > 0.1 && alpha < 5.0 && c > 0.0 && c.is_finite() {
            self.model_alpha = alpha;
            self.model_c = c;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_controller_creation() {
        let rc = RateController::new(10_000_000.0, 30.0, 1920, 1080, RateMode::CBR);
        assert_eq!(rc.target_bitrate, 10_000_000.0);
        assert_eq!(rc.fps, 30.0);
        assert_eq!(rc.width, 1920);
        assert_eq!(rc.height, 1080);
        assert_eq!(rc.mode, RateMode::CBR);
        assert_eq!(rc.frame_count, 0);
    }

    #[test]
    fn test_estimate_qstep_returns_positive() {
        let rc = RateController::new(10_000_000.0, 30.0, 1920, 1080, RateMode::VBR);
        let qstep = rc.estimate_qstep();
        assert!(qstep > 0.0, "qstep should be positive, got {}", qstep);
        assert!(qstep <= 128.0, "qstep should be <= max, got {}", qstep);
    }

    #[test]
    fn test_update_adjusts_model() {
        let mut rc = RateController::new(5_000_000.0, 30.0, 1920, 1080, RateMode::VBR);
        let initial_c = rc.model_c;

        // Simulate encoding at qstep=4, getting bpp=0.5
        rc.update(4.0, 0.5);
        // Model should have updated
        assert_ne!(rc.model_c, initial_c);
        assert_eq!(rc.frame_count, 1);
    }

    #[test]
    fn test_higher_bitrate_gives_lower_qstep() {
        let rc_low = RateController::new(1_000_000.0, 30.0, 1920, 1080, RateMode::VBR);
        let rc_high = RateController::new(50_000_000.0, 30.0, 1920, 1080, RateMode::VBR);

        let q_low = rc_low.estimate_qstep();
        let q_high = rc_high.estimate_qstep();

        // Higher bitrate budget -> lower quantization step (better quality)
        assert!(
            q_high < q_low,
            "Higher bitrate should give lower qstep: high={}, low={}",
            q_high,
            q_low
        );
    }

    #[test]
    fn test_vbv_constrains_when_buffer_empty() {
        let mut rc = RateController::new(10_000_000.0, 30.0, 1920, 1080, RateMode::CBR);
        // Drain the buffer: simulate frames that used far more bits than budget
        rc.vbv_fill = rc.vbv_capacity * 0.05; // nearly empty

        let unconstrained = 4.0;
        let constrained = rc.vbv_constrain(unconstrained);
        // When buffer is near empty, qstep should increase (use fewer bits)
        assert!(
            constrained > unconstrained,
            "VBV should raise qstep when buffer is low: constrained={}, unconstrained={}",
            constrained,
            unconstrained
        );
    }

    #[test]
    fn test_vbv_constrains_when_buffer_full() {
        let mut rc = RateController::new(10_000_000.0, 30.0, 1920, 1080, RateMode::CBR);
        // Fill the buffer
        rc.vbv_fill = rc.vbv_capacity * 0.95; // nearly full

        let unconstrained = 4.0;
        let constrained = rc.vbv_constrain(unconstrained);
        // When buffer is near full, qstep should decrease (use more bits)
        assert!(
            constrained < unconstrained,
            "VBV should lower qstep when buffer is full: constrained={}, unconstrained={}",
            constrained,
            unconstrained
        );
    }

    #[test]
    fn test_model_fitting_with_multiple_samples() {
        let mut rc = RateController::new(10_000_000.0, 30.0, 1920, 1080, RateMode::VBR);

        // Feed samples that follow bpp = 10 * qstep^(-1.5)
        let alpha = 1.5;
        let c = 10.0;
        for &qs in &[2.0, 4.0, 8.0, 16.0] {
            let bpp = c * (qs as f64).powf(-alpha);
            rc.update(qs, bpp);
        }

        // Model should have converged close to alpha=1.5, c=10
        assert!(
            (rc.model_alpha - alpha).abs() < 0.2,
            "alpha should be ~{}, got {}",
            alpha,
            rc.model_alpha
        );
        assert!(
            (rc.model_c - c).abs() / c < 0.3,
            "c should be ~{}, got {}",
            c,
            rc.model_c
        );
    }

    #[test]
    fn test_vbr_is_gentler_than_cbr() {
        let mut rc_cbr = RateController::new(10_000_000.0, 30.0, 1920, 1080, RateMode::CBR);
        let mut rc_vbr = RateController::new(10_000_000.0, 30.0, 1920, 1080, RateMode::VBR);

        // Set same buffer fill level — nearly empty
        rc_cbr.vbv_fill = rc_cbr.vbv_capacity * 0.05;
        rc_vbr.vbv_fill = rc_vbr.vbv_capacity * 0.05;

        let base_qstep = 4.0;
        let cbr_q = rc_cbr.vbv_constrain(base_qstep);
        let vbr_q = rc_vbr.vbv_constrain(base_qstep);

        // CBR should be more aggressive (higher qstep adjustment)
        assert!(
            cbr_q >= vbr_q,
            "CBR should be at least as aggressive as VBR: cbr={}, vbr={}",
            cbr_q,
            vbr_q
        );
    }

    #[test]
    fn test_quality_mode_not_affected() {
        // When target_bitrate is None, no rate control is created — the pipeline
        // uses config.quantization_step directly. This test verifies the
        // CodecConfig default has no rate control enabled.
        let config = crate::CodecConfig::default();
        assert!(config.target_bitrate.is_none());
    }
}
