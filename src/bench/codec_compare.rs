//! Multi-codec comparison: GNC vs JPEG vs JPEG 2000.
//!
//! Uses the `image` crate for JPEG encode/decode (already a dependency),
//! and shells out to OpenJPEG (`opj_compress`/`opj_decompress`) for JPEG 2000.

use super::bdrate::RdPoint;
use super::quality;
use std::io::Cursor;

/// Generate a JPEG rate-distortion curve using the `image` crate.
///
/// For each quality value (1-100), encodes the input PNG as JPEG at that quality,
/// decodes it, and measures PSNR and bpp.
pub fn jpeg_rd_curve(input_png: &str, quality_values: &[u32]) -> Vec<(u32, RdPoint)> {
    let img = image::open(input_png).expect("Failed to open input image");
    let rgb8 = img.to_rgb8();
    let (w, h) = rgb8.dimensions();
    let total_pixels = w as f64 * h as f64;

    // Original as f32 for PSNR computation
    let original_f32: Vec<f32> = rgb8.as_raw().iter().map(|&v| v as f32).collect();

    let mut points = Vec::with_capacity(quality_values.len());

    for &q in quality_values {
        let q_clamped = q.clamp(1, 100) as u8;

        // Encode to JPEG in memory
        let mut jpeg_buf: Vec<u8> = Vec::new();
        {
            let mut cursor = Cursor::new(&mut jpeg_buf);
            let encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, q_clamped);
            rgb8.write_with_encoder(encoder)
                .expect("JPEG encode failed");
        }

        let compressed_bytes = jpeg_buf.len();

        // Decode JPEG back
        let decoded = image::load_from_memory_with_format(&jpeg_buf, image::ImageFormat::Jpeg)
            .expect("JPEG decode failed")
            .to_rgb8();

        let decoded_f32: Vec<f32> = decoded.as_raw().iter().map(|&v| v as f32).collect();

        let psnr = quality::psnr(&original_f32, &decoded_f32, 255.0);
        let bpp = (compressed_bytes as f64 * 8.0) / total_pixels;

        points.push((q, RdPoint { bpp, psnr }));
    }

    points
}

/// Generate a JPEG 2000 rate-distortion curve using OpenJPEG CLI tools.
///
/// Returns `None` if `opj_compress` is not found in PATH.
/// For each compression rate, compresses and decompresses, measuring PSNR and bpp.
///
/// The `rates` parameter contains compression ratios (e.g., 5.0 means 5:1 compression).
/// Higher rate = more compression = lower quality.
pub fn jpeg2000_rd_curve(input_png: &str, rates: &[f32]) -> Option<Vec<(f32, RdPoint)>> {
    // Check if opj_compress is available
    if std::process::Command::new("opj_compress")
        .arg("--help")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_err()
    {
        return None;
    }

    let img = image::open(input_png).expect("Failed to open input image");
    let rgb8 = img.to_rgb8();
    let (w, h) = rgb8.dimensions();
    let total_pixels = w as f64 * h as f64;

    let original_f32: Vec<f32> = rgb8.as_raw().iter().map(|&v| v as f32).collect();

    let tmp_dir = std::env::temp_dir().join("gnc_j2k_compare");
    std::fs::create_dir_all(&tmp_dir).ok();

    // Save input as PPM (raw format OpenJPEG can read)
    let ppm_path = tmp_dir.join("input.ppm");
    // Save as PNG for opj_compress (it reads PNG natively)
    let input_for_opj = tmp_dir.join("input.png");
    rgb8.save(&input_for_opj).expect("Failed to save temp PNG");

    let mut points = Vec::with_capacity(rates.len());

    for &rate in rates {
        let j2k_path = tmp_dir.join(format!("out_r{}.j2k", rate));
        let out_png_path = tmp_dir.join(format!("out_r{}.png", rate));

        // Compress with opj_compress
        let compress_status = std::process::Command::new("opj_compress")
            .args([
                "-i",
                input_for_opj.to_str().unwrap(),
                "-o",
                j2k_path.to_str().unwrap(),
                "-r",
                &format!("{}", rate),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();

        if compress_status.is_err() || !compress_status.unwrap().success() {
            eprintln!("  Warning: opj_compress failed for rate={}, skipping", rate);
            continue;
        }

        // Get compressed file size
        let compressed_bytes = match std::fs::metadata(&j2k_path) {
            Ok(m) => m.len() as usize,
            Err(_) => continue,
        };

        // Decompress with opj_decompress
        let decompress_status = std::process::Command::new("opj_decompress")
            .args([
                "-i",
                j2k_path.to_str().unwrap(),
                "-o",
                out_png_path.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();

        if decompress_status.is_err() || !decompress_status.unwrap().success() {
            eprintln!(
                "  Warning: opj_decompress failed for rate={}, skipping",
                rate
            );
            continue;
        }

        // Load decompressed image and compute PSNR
        let decoded = match image::open(&out_png_path) {
            Ok(img) => img.to_rgb8(),
            Err(_) => continue,
        };
        let decoded_f32: Vec<f32> = decoded.as_raw().iter().map(|&v| v as f32).collect();

        // Verify dimensions match (safety check)
        if decoded.dimensions() != (w, h) {
            eprintln!(
                "  Warning: J2K decoded dimensions mismatch for rate={}, skipping",
                rate
            );
            continue;
        }

        let psnr = quality::psnr(&original_f32, &decoded_f32, 255.0);
        let bpp = (compressed_bytes as f64 * 8.0) / total_pixels;

        points.push((rate, RdPoint { bpp, psnr }));

        // Clean up temp files for this rate
        let _ = std::fs::remove_file(&j2k_path);
        let _ = std::fs::remove_file(&out_png_path);
    }

    // Clean up temp directory
    let _ = std::fs::remove_file(&ppm_path);
    let _ = std::fs::remove_file(&input_for_opj);
    let _ = std::fs::remove_dir(&tmp_dir);

    Some(points)
}

/// Write a unified multi-codec comparison CSV.
///
/// Columns: codec, quality_param, psnr, ssim, bpp
///
/// For GNC and JPEG, `ssim` is computed. For J2K we set ssim to empty since
/// the external tool pipeline does not easily yield it without extra work.
pub fn write_comparison_csv(
    path: &str,
    gnc_points: &[(u32, RdPoint)],
    jpeg_points: &[(u32, RdPoint)],
    j2k_points: &[(f32, RdPoint)],
    gnc_ssim: &[f64],
    jpeg_ssim: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["codec", "quality_param", "psnr", "ssim", "bpp"])?;

    for (i, (q, pt)) in gnc_points.iter().enumerate() {
        let ssim_val = gnc_ssim.get(i).copied().unwrap_or(0.0);
        wtr.write_record(&[
            "GNC".to_string(),
            format!("{}", q),
            format!("{:.4}", pt.psnr),
            format!("{:.6}", ssim_val),
            format!("{:.6}", pt.bpp),
        ])?;
    }

    for (i, (q, pt)) in jpeg_points.iter().enumerate() {
        let ssim_val = jpeg_ssim.get(i).copied().unwrap_or(0.0);
        wtr.write_record(&[
            "JPEG".to_string(),
            format!("{}", q),
            format!("{:.4}", pt.psnr),
            format!("{:.6}", ssim_val),
            format!("{:.6}", pt.bpp),
        ])?;
    }

    for (rate, pt) in j2k_points {
        wtr.write_record(&[
            "JPEG2000".to_string(),
            format!("{:.1}", rate),
            format!("{:.4}", pt.psnr),
            String::new(), // SSIM not computed for J2K
            format!("{:.6}", pt.bpp),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

/// Compute SSIM values for JPEG at given quality levels (for the comparison CSV).
///
/// Returns a Vec of SSIM values, one per quality value, in the same order.
pub fn jpeg_ssim_values(input_png: &str, quality_values: &[u32]) -> Vec<f64> {
    let img = image::open(input_png).expect("Failed to open input image");
    let rgb8 = img.to_rgb8();
    let original_f32: Vec<f32> = rgb8.as_raw().iter().map(|&v| v as f32).collect();

    let mut ssim_values = Vec::with_capacity(quality_values.len());

    for &q in quality_values {
        let q_clamped = q.clamp(1, 100) as u8;

        let mut jpeg_buf: Vec<u8> = Vec::new();
        {
            let mut cursor = Cursor::new(&mut jpeg_buf);
            let encoder =
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, q_clamped);
            rgb8.write_with_encoder(encoder)
                .expect("JPEG encode failed");
        }

        let decoded = image::load_from_memory_with_format(&jpeg_buf, image::ImageFormat::Jpeg)
            .expect("JPEG decode failed")
            .to_rgb8();
        let decoded_f32: Vec<f32> = decoded.as_raw().iter().map(|&v| v as f32).collect();

        ssim_values.push(quality::ssim_approx(&original_f32, &decoded_f32, 255.0));
    }

    ssim_values
}

/// Default JPEG quality values for RD sweep (matches typical JPEG usage patterns).
pub fn default_jpeg_quality_values() -> Vec<u32> {
    vec![5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
}

/// Default JPEG 2000 compression rates for RD sweep.
/// These are compression ratios (e.g., 100 = 100:1, 1 = lossless).
/// Sorted from highest compression (worst quality) to lowest (best).
pub fn default_j2k_rates() -> Vec<f32> {
    vec![100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.5]
}

/// Convenience: extract just the RdPoints from tagged tuples.
pub fn extract_rd_points_u32(tagged: &[(u32, RdPoint)]) -> Vec<RdPoint> {
    tagged.iter().map(|(_, pt)| *pt).collect()
}

/// Convenience: extract just the RdPoints from tagged tuples (f32 keys).
pub fn extract_rd_points_f32(tagged: &[(f32, RdPoint)]) -> Vec<RdPoint> {
    tagged.iter().map(|(_, pt)| *pt).collect()
}

/// Print a summary table of codec comparison results.
pub fn print_comparison_summary(
    gnc_points: &[(u32, RdPoint)],
    jpeg_points: &[(u32, RdPoint)],
    j2k_points: &[(f32, RdPoint)],
) {
    println!("\n=== Multi-Codec Comparison ===\n");

    // GNC summary
    println!("GNC ({} points):", gnc_points.len());
    println!("  {:>8} {:>8} {:>8}", "q", "psnr", "bpp");
    for (q, pt) in gnc_points {
        println!("  {:>8} {:>8.2} {:>8.4}", q, pt.psnr, pt.bpp);
    }

    // JPEG summary
    println!("\nJPEG ({} points):", jpeg_points.len());
    println!("  {:>8} {:>8} {:>8}", "q", "psnr", "bpp");
    for (q, pt) in jpeg_points {
        println!("  {:>8} {:>8.2} {:>8.4}", q, pt.psnr, pt.bpp);
    }

    // J2K summary
    if !j2k_points.is_empty() {
        println!("\nJPEG 2000 ({} points):", j2k_points.len());
        println!("  {:>8} {:>8} {:>8}", "rate", "psnr", "bpp");
        for (rate, pt) in j2k_points {
            println!("  {:>8.1} {:>8.2} {:>8.4}", rate, pt.psnr, pt.bpp);
        }
    } else {
        println!("\nJPEG 2000: skipped (opj_compress not found in PATH)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_quality_values() {
        let vals = default_jpeg_quality_values();
        assert!(vals.len() >= 10);
        assert!(vals.iter().all(|&v| v >= 1 && v <= 100));
    }

    #[test]
    fn test_default_j2k_rates() {
        let rates = default_j2k_rates();
        assert!(rates.len() >= 5);
        assert!(rates.iter().all(|&r| r > 0.0));
    }

    #[test]
    fn test_extract_rd_points() {
        let tagged = vec![
            (
                10u32,
                RdPoint {
                    bpp: 0.5,
                    psnr: 30.0,
                },
            ),
            (
                20,
                RdPoint {
                    bpp: 1.0,
                    psnr: 35.0,
                },
            ),
        ];
        let points = extract_rd_points_u32(&tagged);
        assert_eq!(points.len(), 2);
        assert!((points[0].bpp - 0.5).abs() < 1e-6);
    }
}
