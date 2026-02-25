//! Per-frame quality metrics and temporal consistency tracking for sequences.

use serde::Serialize;

/// Quality metrics for a single frame in a sequence.
#[derive(Debug, Clone, Serialize)]
pub struct FrameMetrics {
    pub frame_idx: usize,
    pub frame_type: String,
    pub psnr: f64,
    pub ssim: f64,
    pub bpp: f64,
    pub encoded_bytes: usize,
}

/// Aggregate statistics for a metric across all frames.
#[derive(Debug, Clone, Serialize)]
pub struct MetricStats {
    pub avg: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

/// Summary statistics for an entire encoded sequence.
#[derive(Debug, Clone, Serialize)]
pub struct SequenceSummary {
    pub psnr: MetricStats,
    pub ssim: MetricStats,
    pub bpp: MetricStats,
    /// Largest PSNR decrease between consecutive frames (positive = drop).
    /// A large value indicates a visually jarring quality discontinuity.
    pub max_psnr_drop: f64,
    /// Standard deviation of inter-frame PSNR differences.
    /// Lower values indicate more temporally consistent quality.
    pub temporal_consistency: f64,
    pub total_frames: usize,
    pub total_bytes: usize,
}

impl std::fmt::Display for SequenceSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Sequence Summary ({} frames, {} bytes):", self.total_frames, self.total_bytes)?;
        writeln!(
            f,
            "  PSNR:  avg {:.2} dB  min {:.2}  max {:.2}  stddev {:.2}",
            self.psnr.avg, self.psnr.min, self.psnr.max, self.psnr.stddev
        )?;
        writeln!(
            f,
            "  SSIM:  avg {:.4}  min {:.4}  max {:.4}  stddev {:.4}",
            self.ssim.avg, self.ssim.min, self.ssim.max, self.ssim.stddev
        )?;
        writeln!(
            f,
            "  BPP:   avg {:.4}  min {:.4}  max {:.4}  stddev {:.4}",
            self.bpp.avg, self.bpp.min, self.bpp.max, self.bpp.stddev
        )?;
        writeln!(
            f,
            "  Temporal: max PSNR drop {:.2} dB, consistency (stddev of dPSNR) {:.2} dB",
            self.max_psnr_drop, self.temporal_consistency
        )?;
        Ok(())
    }
}

/// Compute aggregate statistics from a slice of f64 values.
fn compute_stats(values: &[f64]) -> MetricStats {
    if values.is_empty() {
        return MetricStats {
            avg: 0.0,
            min: 0.0,
            max: 0.0,
            stddev: 0.0,
        };
    }

    let n = values.len() as f64;
    let avg = values.iter().sum::<f64>() / n;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = values.iter().map(|v| (v - avg) * (v - avg)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    MetricStats {
        avg,
        min,
        max,
        stddev,
    }
}

/// Compute sequence-level summary statistics from per-frame metrics.
///
/// Calculates avg/min/max/stddev for PSNR, SSIM, and BPP, plus temporal
/// consistency metrics that measure quality stability across frames.
pub fn compute_sequence_metrics(frames: &[FrameMetrics]) -> SequenceSummary {
    let psnr_values: Vec<f64> = frames.iter().map(|f| f.psnr).collect();
    let ssim_values: Vec<f64> = frames.iter().map(|f| f.ssim).collect();
    let bpp_values: Vec<f64> = frames.iter().map(|f| f.bpp).collect();

    let psnr_stats = compute_stats(&psnr_values);
    let ssim_stats = compute_stats(&ssim_values);
    let bpp_stats = compute_stats(&bpp_values);

    // Temporal consistency: analyze inter-frame PSNR differences
    let (max_psnr_drop, temporal_consistency) = if frames.len() >= 2 {
        let diffs: Vec<f64> = psnr_values
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        // max_psnr_drop: largest decrease (most negative diff, reported as positive)
        let max_drop = diffs
            .iter()
            .cloned()
            .fold(0.0f64, |worst, d| worst.max(-d));

        // temporal_consistency: stddev of inter-frame PSNR differences
        let diff_stats = compute_stats(&diffs);

        (max_drop, diff_stats.stddev)
    } else {
        (0.0, 0.0)
    };

    let total_bytes = frames.iter().map(|f| f.encoded_bytes).sum();

    SequenceSummary {
        psnr: psnr_stats,
        ssim: ssim_stats,
        bpp: bpp_stats,
        max_psnr_drop,
        temporal_consistency,
        total_frames: frames.len(),
        total_bytes,
    }
}

/// Write per-frame metrics and a summary row to a CSV file.
pub fn write_sequence_csv(
    path: &str,
    frames: &[FrameMetrics],
    summary: &SequenceSummary,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;

    wtr.write_record([
        "frame_idx",
        "frame_type",
        "psnr",
        "ssim",
        "bpp",
        "encoded_bytes",
    ])?;

    for fm in frames {
        wtr.write_record([
            &fm.frame_idx.to_string(),
            &fm.frame_type,
            &format!("{:.4}", fm.psnr),
            &format!("{:.6}", fm.ssim),
            &format!("{:.6}", fm.bpp),
            &fm.encoded_bytes.to_string(),
        ])?;
    }

    // Summary row: frame_idx = "summary", frame_type = total_frames count
    wtr.write_record([
        "summary",
        &format!("{} frames", summary.total_frames),
        &format!(
            "avg={:.2} min={:.2} max={:.2} std={:.2}",
            summary.psnr.avg, summary.psnr.min, summary.psnr.max, summary.psnr.stddev
        ),
        &format!(
            "avg={:.4} min={:.4} max={:.4} std={:.4}",
            summary.ssim.avg, summary.ssim.min, summary.ssim.max, summary.ssim.stddev
        ),
        &format!(
            "avg={:.4} min={:.4} max={:.4} std={:.4}",
            summary.bpp.avg, summary.bpp.min, summary.bpp.max, summary.bpp.stddev
        ),
        &summary.total_bytes.to_string(),
    ])?;

    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_sequence_metrics_basic() {
        let frames = vec![
            FrameMetrics {
                frame_idx: 0,
                frame_type: "I".to_string(),
                psnr: 40.0,
                ssim: 0.95,
                bpp: 2.0,
                encoded_bytes: 1000,
            },
            FrameMetrics {
                frame_idx: 1,
                frame_type: "P".to_string(),
                psnr: 38.0,
                ssim: 0.93,
                bpp: 1.0,
                encoded_bytes: 500,
            },
            FrameMetrics {
                frame_idx: 2,
                frame_type: "P".to_string(),
                psnr: 39.0,
                ssim: 0.94,
                bpp: 1.2,
                encoded_bytes: 600,
            },
        ];

        let summary = compute_sequence_metrics(&frames);

        assert_eq!(summary.total_frames, 3);
        assert_eq!(summary.total_bytes, 2100);
        assert!((summary.psnr.avg - 39.0).abs() < 0.01);
        assert!((summary.psnr.min - 38.0).abs() < 0.01);
        assert!((summary.psnr.max - 40.0).abs() < 0.01);
        // max_psnr_drop: frame 0->1 drops 2.0 dB, frame 1->2 gains 1.0 dB
        assert!((summary.max_psnr_drop - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_sequence_metrics_single_frame() {
        let frames = vec![FrameMetrics {
            frame_idx: 0,
            frame_type: "I".to_string(),
            psnr: 42.0,
            ssim: 0.97,
            bpp: 1.5,
            encoded_bytes: 800,
        }];

        let summary = compute_sequence_metrics(&frames);

        assert_eq!(summary.total_frames, 1);
        assert!((summary.psnr.avg - 42.0).abs() < 0.01);
        assert!((summary.max_psnr_drop - 0.0).abs() < 0.01);
        assert!((summary.temporal_consistency - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_sequence_metrics_empty() {
        let frames: Vec<FrameMetrics> = vec![];
        let summary = compute_sequence_metrics(&frames);
        assert_eq!(summary.total_frames, 0);
        assert_eq!(summary.total_bytes, 0);
    }

    #[test]
    fn test_temporal_consistency_constant_quality() {
        // All frames have identical PSNR => temporal_consistency = 0
        let frames: Vec<FrameMetrics> = (0..5)
            .map(|i| FrameMetrics {
                frame_idx: i,
                frame_type: "I".to_string(),
                psnr: 40.0,
                ssim: 0.95,
                bpp: 1.5,
                encoded_bytes: 700,
            })
            .collect();

        let summary = compute_sequence_metrics(&frames);
        assert!((summary.temporal_consistency - 0.0).abs() < 0.001);
        assert!((summary.max_psnr_drop - 0.0).abs() < 0.001);
    }
}
