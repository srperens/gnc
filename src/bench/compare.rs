use super::quality::QualityMetrics;
use super::throughput::ThroughputMetrics;
use serde::Serialize;

/// Combined benchmark result for one configuration
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub quality: QualityMetrics,
    pub throughput: Option<ThroughputMetrics>,
    pub config: BenchmarkConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkConfig {
    pub tile_size: u32,
    pub quantization_step: f32,
    pub dead_zone: f32,
    pub input_file: String,
    pub width: u32,
    pub height: u32,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== {} ===", self.name)?;
        writeln!(
            f,
            "  Config: tile={}  qstep={}  dz={}",
            self.config.tile_size, self.config.quantization_step, self.config.dead_zone
        )?;
        writeln!(f, "  Quality: {}", self.quality)?;
        if let Some(ref tp) = self.throughput {
            writeln!(f, "  Throughput: {}", tp)?;
        }
        Ok(())
    }
}

/// Write benchmark results to CSV
pub fn write_csv(
    results: &[BenchmarkResult],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;

    wtr.write_record(&[
        "name",
        "input",
        "width",
        "height",
        "tile_size",
        "qstep",
        "dead_zone",
        "psnr_db",
        "psnr_r",
        "psnr_g",
        "psnr_b",
        "ssim",
        "bpp",
        "compressed_bytes",
        "encode_ms",
        "decode_ms",
        "encode_fps",
        "decode_fps",
    ])?;

    for r in results {
        let tp = r.throughput.as_ref();
        wtr.write_record(&[
            &r.name,
            &r.config.input_file,
            &r.config.width.to_string(),
            &r.config.height.to_string(),
            &r.config.tile_size.to_string(),
            &r.config.quantization_step.to_string(),
            &r.config.dead_zone.to_string(),
            &format!("{:.2}", r.quality.psnr_db),
            &format!("{:.2}", r.quality.psnr_r),
            &format!("{:.2}", r.quality.psnr_g),
            &format!("{:.2}", r.quality.psnr_b),
            &format!("{:.4}", r.quality.ssim),
            &format!("{:.2}", r.quality.bpp),
            &r.quality.compressed_bytes.to_string(),
            &tp.map_or("".to_string(), |t| format!("{:.2}", t.encode_time_ms)),
            &tp.map_or("".to_string(), |t| format!("{:.2}", t.decode_time_ms)),
            &tp.map_or("".to_string(), |t| format!("{:.1}", t.encode_fps)),
            &tp.map_or("".to_string(), |t| format!("{:.1}", t.decode_fps)),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}
