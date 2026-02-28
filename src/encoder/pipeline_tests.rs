use super::*;
use crate::decoder::pipeline::DecoderPipeline;
use wgpu::util::DeviceExt;

fn compute_psnr_single(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse < 1e-10 {
        return 100.0;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

fn compute_psnr(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse < 1e-10 {
        return 100.0;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

/// Generate a synthetic RGB frame: smooth gradient with some detail.
fn make_gradient_frame(w: u32, h: u32, offset: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x as f32 + offset) / w as f32 * 255.0).clamp(0.0, 255.0);
            let g = ((y as f32 + offset) / h as f32 * 255.0).clamp(0.0, 255.0);
            let b = (((x + y) as f32 + offset) / (w + h) as f32 * 255.0).clamp(0.0, 255.0);
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

#[test]
fn test_encode_sequence_all_iframes() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);

    let w = 256;
    let h = 256;
    let f0 = make_gradient_frame(w, h, 0.0);
    let f1 = make_gradient_frame(w, h, 5.0);
    let f2 = make_gradient_frame(w, h, 10.0);

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 1; // all I-frames

    let frames: Vec<&[f32]> = vec![&f0, &f1, &f2];
    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

    assert_eq!(compressed.len(), 3);
    for (i, cf) in compressed.iter().enumerate() {
        assert_eq!(
            cf.frame_type,
            FrameType::Intra,
            "frame {i} should be Intra with ki=1"
        );
        assert!(cf.motion_field.is_none(), "I-frame should have no MVs");
    }
}

#[test]
fn test_encode_sequence_ip_pattern() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);

    let w = 256;
    let h = 256;
    let f0 = make_gradient_frame(w, h, 0.0);
    let f1 = make_gradient_frame(w, h, 3.0);
    let f2 = make_gradient_frame(w, h, 6.0);
    let f3 = make_gradient_frame(w, h, 9.0);

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 4; // I B B P (with B-frames at ki>=4)

    let frames: Vec<&[f32]> = vec![&f0, &f1, &f2, &f3];
    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

    assert_eq!(compressed.len(), 4);
    assert_eq!(compressed[0].frame_type, FrameType::Intra);
    // With B_FRAMES_PER_GROUP=2: display order is I B B P
    assert_eq!(compressed[1].frame_type, FrameType::Bidirectional);
    assert_eq!(compressed[2].frame_type, FrameType::Bidirectional);
    assert_eq!(compressed[3].frame_type, FrameType::Predicted);

    // All inter frames must have motion fields
    for i in 1..4 {
        assert!(
            compressed[i].motion_field.is_some(),
            "Inter frame {i} should have motion field"
        );
    }

    // B-frames should have backward vectors and block modes
    for i in 1..3 {
        let mf = compressed[i].motion_field.as_ref().unwrap();
        assert!(
            mf.backward_vectors.is_some(),
            "B-frame {i} should have backward vectors"
        );
        assert!(
            mf.block_modes.is_some(),
            "B-frame {i} should have block modes"
        );
    }
}

#[test]
fn test_pframe_roundtrip_quality() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256;
    let h = 256;
    let f0 = make_gradient_frame(w, h, 0.0);
    let f1 = make_gradient_frame(w, h, 2.0); // slight shift

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 8;

    let frames: Vec<&[f32]> = vec![&f0, &f1];
    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

    // Decode both frames through the decoder (which maintains reference state)
    let dec0 = dec.decode(&ctx, &compressed[0]);
    let dec1 = dec.decode(&ctx, &compressed[1]);

    // Both frames should reconstruct with good quality
    let psnr0 = compute_psnr(&f0, &dec0);
    let psnr1 = compute_psnr(&f1, &dec1);

    eprintln!("I-frame PSNR: {psnr0:.2} dB, P-frame PSNR: {psnr1:.2} dB");

    assert!(
        psnr0 > 30.0,
        "I-frame PSNR should be > 30 dB, got {psnr0:.2}"
    );
    assert!(
        psnr1 > 25.0,
        "P-frame PSNR should be > 25 dB, got {psnr1:.2}"
    );
}

#[test]
fn test_pframe_identical_frames_correct_decode() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256;
    let h = 256;
    let f0 = make_gradient_frame(w, h, 0.0);
    let f1 = f0.clone(); // identical frame

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 8;

    let frames: Vec<&[f32]> = vec![&f0, &f1];
    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

    // Decode both frames
    let dec0 = dec.decode(&ctx, &compressed[0]);
    let dec1 = dec.decode(&ctx, &compressed[1]);

    // Both should decode with good quality
    let psnr0 = compute_psnr(&f0, &dec0);
    let psnr1 = compute_psnr(&f1, &dec1);
    eprintln!("I-frame PSNR: {psnr0:.2} dB, P-frame (identical input) PSNR: {psnr1:.2} dB");

    assert!(psnr0 > 30.0, "I-frame PSNR too low: {psnr0:.2}");
    assert!(psnr1 > 30.0, "P-frame PSNR too low: {psnr1:.2}");

    // Motion vectors should be near-zero for identical content
    let mf = compressed[1].motion_field.as_ref().unwrap();
    let max_mv: i16 = mf
        .vectors
        .iter()
        .flat_map(|v| v.iter())
        .map(|v| v.abs())
        .max()
        .unwrap_or(0);
    eprintln!("Max MV magnitude for identical frames: {max_mv}");

    // Decoded frame 1 should be very close to decoded frame 0 (identical source)
    let inter_psnr = compute_psnr(&dec0, &dec1);
    eprintln!("Inter-frame PSNR (dec0 vs dec1): {inter_psnr:.2} dB");
    assert!(
        inter_psnr > 30.0,
        "Identical frames should decode similarly: {inter_psnr:.2} dB"
    );
}

#[test]
fn test_sequence_decode_all_frames() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256;
    let h = 256;
    let frames_rgb: Vec<Vec<f32>> = (0..5)
        .map(|i| make_gradient_frame(w, h, i as f32 * 3.0))
        .collect();
    let frame_refs: Vec<&[f32]> = frames_rgb.iter().map(|f| f.as_slice()).collect();

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 3; // I P P I P (no B-frames, ki < 4)

    let compressed = enc.encode_sequence(&ctx, &frame_refs, w, h, &config);

    assert_eq!(compressed[0].frame_type, FrameType::Intra);
    assert_eq!(compressed[1].frame_type, FrameType::Predicted);
    assert_eq!(compressed[2].frame_type, FrameType::Predicted);
    assert_eq!(compressed[3].frame_type, FrameType::Intra);
    assert_eq!(compressed[4].frame_type, FrameType::Predicted);

    // Decode all frames in order (decoder maintains reference state)
    for (i, cf) in compressed.iter().enumerate() {
        let decoded = dec.decode(&ctx, cf);
        let psnr = compute_psnr(&frames_rgb[i], &decoded);
        eprintln!(
            "Frame {i} ({:?}): PSNR={psnr:.2} dB, size={} bytes",
            cf.frame_type,
            cf.byte_size()
        );
        assert!(psnr > 25.0, "Frame {i} PSNR too low: {psnr:.2} dB");
    }
}

#[test]
fn test_bframe_sequence_roundtrip() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256;
    let h = 256;
    // 7 frames with ki=7: I B B P B B P
    let frames_rgb: Vec<Vec<f32>> = (0..7)
        .map(|i| make_gradient_frame(w, h, i as f32 * 2.0))
        .collect();
    let frame_refs: Vec<&[f32]> = frames_rgb.iter().map(|f| f.as_slice()).collect();

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 7; // triggers B-frames (ki >= 4)

    let compressed = enc.encode_sequence(&ctx, &frame_refs, w, h, &config);
    assert_eq!(compressed.len(), 7);

    // Verify frame types: I B B P B B P
    assert_eq!(compressed[0].frame_type, FrameType::Intra);
    assert_eq!(compressed[1].frame_type, FrameType::Bidirectional);
    assert_eq!(compressed[2].frame_type, FrameType::Bidirectional);
    assert_eq!(compressed[3].frame_type, FrameType::Predicted);
    assert_eq!(compressed[4].frame_type, FrameType::Bidirectional);
    assert_eq!(compressed[5].frame_type, FrameType::Bidirectional);
    assert_eq!(compressed[6].frame_type, FrameType::Predicted);

    // Decode using B-frame aware sequence decoder
    let decoded = dec.decode_sequence(&ctx, &compressed);
    assert_eq!(decoded.len(), 7);

    for (i, dec_frame) in decoded.iter().enumerate() {
        let psnr = compute_psnr(&frames_rgb[i], dec_frame);
        eprintln!(
            "Frame {i} ({:?}): PSNR={psnr:.2} dB, bpp={:.3}",
            compressed[i].frame_type,
            compressed[i].bpp()
        );
        assert!(
            psnr > 20.0,
            "Frame {i} ({:?}) PSNR too low: {psnr:.2} dB",
            compressed[i].frame_type
        );
    }
}

/// Generate a test image with integer RGB pixel values spanning a wide range.
fn make_integer_test_image(w: u32, h: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            // Mix of gradients, constants, and patterns to exercise edge cases
            let r = ((x * 255) / (w - 1)).min(255) as f32;
            let g = ((y * 255) / (h - 1)).min(255) as f32;
            let b = (((x + y) * 255) / (w + h - 2)).min(255) as f32;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

#[test]
fn test_lossless_roundtrip_bit_exact() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let config = crate::quality_preset(100);
    assert!(config.is_lossless(), "q=100 should be lossless config");

    // Test on multiple image sizes to cover tile boundary cases
    for &(w, h) in &[(256, 256), (512, 512), (100, 100), (300, 200)] {
        let input = make_integer_test_image(w, h);
        let compressed = enc.encode(&ctx, &input, w, h, &config);
        let decoded = dec.decode(&ctx, &compressed);

        let total_pixels = (w * h * 3) as usize;
        assert_eq!(decoded.len(), total_pixels);

        let mut max_err: f32 = 0.0;
        let mut err_count = 0usize;
        for i in 0..total_pixels {
            let diff = (input[i] - decoded[i]).abs();
            if diff > 0.0 {
                err_count += 1;
                max_err = max_err.max(diff);
            }
        }

        let bpp = compressed.bpp();
        eprintln!(
            "Lossless {w}x{h}: bpp={bpp:.3}, max_err={max_err}, err_pixels={err_count}/{}",
            w * h
        );
        assert_eq!(
            err_count, 0,
            "Lossless round-trip NOT bit-exact for {w}x{h}: {err_count} pixels differ, max_err={max_err}"
        );
    }
}

/// Verify fused quantize+histogram produces identical decoded output to the
/// separate quantize + histogram path. Tests both with and without adaptive QP.
#[test]
fn test_fused_quantize_histogram_matches_separate() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;
    let frame = make_gradient_frame(w, h, 42.0);

    // Baseline: separate quantize + histogram (default path)
    let mut config_separate = CodecConfig::default();
    config_separate.use_fused_quantize_histogram = false;
    config_separate.gpu_entropy_encode = true;
    config_separate.per_subband_entropy = true;
    config_separate.quantization_step = 4.0;
    config_separate.dead_zone = 0.5;
    config_separate.cfl_enabled = false;

    let compressed_sep = enc.encode(&ctx, &frame, w, h, &config_separate);
    let decoded_sep = dec.decode(&ctx, &compressed_sep);

    // Fused path: quantize + histogram in single dispatch
    let mut config_fused = config_separate.clone();
    config_fused.use_fused_quantize_histogram = true;

    let compressed_fused = enc.encode(&ctx, &frame, w, h, &config_fused);
    let decoded_fused = dec.decode(&ctx, &compressed_fused);

    // Compare decoded pixels: must be identical (same quantization, same entropy)
    assert_eq!(decoded_sep.len(), decoded_fused.len(), "decoded length mismatch");
    let mut max_diff: f32 = 0.0;
    let mut diff_count = 0usize;
    for (i, (a, b)) in decoded_sep.iter().zip(decoded_fused.iter()).enumerate() {
        let d = (a - b).abs();
        if d > 0.5 {
            diff_count += 1;
            if diff_count <= 5 {
                eprintln!("pixel {i}: separate={a}, fused={b}, diff={d}");
            }
        }
        max_diff = max_diff.max(d);
    }
    assert!(
        diff_count == 0,
        "Fused vs separate mismatch: {diff_count} pixels differ, max_diff={max_diff}"
    );

    // Also verify bitrate is identical (same entropy coding)
    let bpp_sep = compressed_sep.bpp();
    let bpp_fused = compressed_fused.bpp();
    let bpp_diff = (bpp_sep - bpp_fused).abs();
    eprintln!("bpp: separate={bpp_sep:.4}, fused={bpp_fused:.4}, diff={bpp_diff:.6}");
    assert!(
        bpp_diff < 0.001,
        "Bitrate mismatch: separate={bpp_sep:.4} vs fused={bpp_fused:.4}"
    );
}

/// Verify fused quantize+histogram with adaptive quantization produces
/// identical results to the separate path.
#[test]
fn test_fused_quantize_histogram_with_aq() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;
    let frame = make_gradient_frame(w, h, 17.0);

    // Baseline: separate path with AQ
    let mut config_separate = CodecConfig::default();
    config_separate.use_fused_quantize_histogram = false;
    config_separate.gpu_entropy_encode = true;
    config_separate.per_subband_entropy = true;
    config_separate.quantization_step = 4.0;
    config_separate.dead_zone = 0.5;
    config_separate.cfl_enabled = false;
    config_separate.adaptive_quantization = true;
    config_separate.aq_strength = 0.5;

    let compressed_sep = enc.encode(&ctx, &frame, w, h, &config_separate);
    let decoded_sep = dec.decode(&ctx, &compressed_sep);

    // Fused path with AQ
    let mut config_fused = config_separate.clone();
    config_fused.use_fused_quantize_histogram = true;

    let compressed_fused = enc.encode(&ctx, &frame, w, h, &config_fused);
    let decoded_fused = dec.decode(&ctx, &compressed_fused);

    // Compare decoded pixels
    assert_eq!(decoded_sep.len(), decoded_fused.len(), "decoded length mismatch");
    let mut max_diff: f32 = 0.0;
    let mut diff_count = 0usize;
    for (i, (a, b)) in decoded_sep.iter().zip(decoded_fused.iter()).enumerate() {
        let d = (a - b).abs();
        if d > 0.5 {
            diff_count += 1;
            if diff_count <= 5 {
                eprintln!("pixel {i}: separate={a}, fused={b}, diff={d}");
            }
        }
        max_diff = max_diff.max(d);
    }
    assert!(
        diff_count == 0,
        "Fused+AQ vs separate+AQ mismatch: {diff_count} pixels differ, max_diff={max_diff}"
    );

    let bpp_sep = compressed_sep.bpp();
    let bpp_fused = compressed_fused.bpp();
    let bpp_diff = (bpp_sep - bpp_fused).abs();
    eprintln!("bpp (AQ): separate={bpp_sep:.4}, fused={bpp_fused:.4}, diff={bpp_diff:.6}");
    assert!(
        bpp_diff < 0.001,
        "Bitrate mismatch (AQ): separate={bpp_sep:.4} vs fused={bpp_fused:.4}"
    );
}

#[test]
fn test_block_dct_roundtrip() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256;
    let h = 256;
    let rgb = make_gradient_frame(w, h, 0.0);

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.transform_type = TransformType::BlockDCT8;
    config.cfl_enabled = false;
    config.adaptive_quantization = false;
    config.use_fused_quantize_histogram = false;
    config.quantization_step = 2.0;
    config.dead_zone = 0.0;

    let compressed = enc.encode(&ctx, &rgb, w, h, &config);
    eprintln!(
        "DCT roundtrip: transform_type={:?}, bpp={:.3}, entropy={:?}",
        compressed.config.transform_type,
        compressed.bpp(),
        compressed.config.entropy_coder,
    );

    let decoded = dec.decode(&ctx, &compressed);
    let psnr = compute_psnr(&rgb, &decoded);
    eprintln!("DCT roundtrip PSNR: {psnr:.2} dB");

    // Also check per-channel MSE to see which channel is off
    let n = (w * h) as usize;
    for (ch, name) in [(0, "R"), (1, "G"), (2, "B")] {
        let mse: f64 = (0..n)
            .map(|i| {
                let d = rgb[i * 3 + ch] as f64 - decoded[i * 3 + ch] as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let ch_psnr = if mse < 1e-10 {
            100.0
        } else {
            10.0 * (255.0_f64 * 255.0 / mse).log10()
        };
        eprintln!("  {name}: PSNR={ch_psnr:.2} dB, MSE={mse:.2}");
    }

    // Sample first few decoded pixels to see what's happening
    eprintln!("First 10 pixels (orig vs decoded):");
    for i in 0..10 {
        eprintln!(
            "  px[{i}]: ({:.1}, {:.1}, {:.1}) vs ({:.1}, {:.1}, {:.1})",
            rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2],
            decoded[i * 3], decoded[i * 3 + 1], decoded[i * 3 + 2],
        );
    }

    assert!(
        psnr > 35.0,
        "Block DCT roundtrip PSNR should be > 35 dB at qstep=2.0, got {psnr:.2}"
    );
}

#[test]
fn test_block_dct_multitile() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    // 512×512 = 4 tiles (2×2) with tile_size=256
    let w = 512;
    let h = 512;
    let rgb = make_gradient_frame(w, h, 0.0);

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.transform_type = TransformType::BlockDCT8;
    config.cfl_enabled = false;
    config.adaptive_quantization = false;
    config.use_fused_quantize_histogram = false;
    config.quantization_step = 2.0;
    config.dead_zone = 0.0;

    let compressed = enc.encode(&ctx, &rgb, w, h, &config);
    let decoded = dec.decode(&ctx, &compressed);
    let psnr = compute_psnr(&rgb, &decoded);
    eprintln!("DCT multitile (512x512, 4 tiles) PSNR: {psnr:.2} dB");

    // Check a pixel from tile 0 and tile 1
    for i in [0, 256, 512, 256*512] {
        if i * 3 + 2 < rgb.len() {
            eprintln!(
                "  px[{i}]: ({:.1}, {:.1}, {:.1}) vs ({:.1}, {:.1}, {:.1})",
                rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2],
                decoded[i * 3], decoded[i * 3 + 1], decoded[i * 3 + 2],
            );
        }
    }

    assert!(
        psnr > 35.0,
        "Block DCT multitile PSNR should be > 35 dB, got {psnr:.2}"
    );
}

#[test]
fn test_block_dct_nonaligned() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    // 300×300: not a multiple of 256, gets padded to 512×512
    let w = 300;
    let h = 300;
    let rgb = make_gradient_frame(w, h, 0.0);

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.transform_type = TransformType::BlockDCT8;
    config.cfl_enabled = false;
    config.adaptive_quantization = false;
    config.use_fused_quantize_histogram = false;
    config.quantization_step = 2.0;
    config.dead_zone = 0.0;

    let compressed = enc.encode(&ctx, &rgb, w, h, &config);
    eprintln!(
        "DCT nonaligned: orig {}x{}, padded {}x{}, bpp={:.3}",
        w, h,
        compressed.info.padded_width(), compressed.info.padded_height(),
        compressed.bpp(),
    );

    let decoded = dec.decode(&ctx, &compressed);
    assert_eq!(decoded.len(), (w * h * 3) as usize, "decoded size mismatch");

    let psnr = compute_psnr(&rgb, &decoded);
    eprintln!("DCT nonaligned (300x300) PSNR: {psnr:.2} dB");

    // Check pixels near tile boundary
    for i in [0, 255, 256, 299, 300, 512] {
        if i * 3 + 2 < rgb.len() {
            eprintln!(
                "  px[{i}]: ({:.1}, {:.1}, {:.1}) vs ({:.1}, {:.1}, {:.1})",
                rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2],
                decoded[i * 3], decoded[i * 3 + 1], decoded[i * 3 + 2],
            );
        }
    }

    assert!(
        psnr > 35.0,
        "Block DCT nonaligned PSNR should be > 35 dB, got {psnr:.2}"
    );
}

#[test]
fn test_block_dct_quality_preset() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    // Test with quality_preset settings, like the benchmark uses
    for q in [75, 85, 92, 99] {
        let w = 512;
        let h = 512;
        let rgb = make_gradient_frame(w, h, 0.0);

        let mut config = crate::quality_preset(q);
        config.transform_type = TransformType::BlockDCT8;
        config.cfl_enabled = false;
        config.adaptive_quantization = false;
        config.use_fused_quantize_histogram = false;

        let compressed = enc.encode(&ctx, &rgb, w, h, &config);
        let decoded = dec.decode(&ctx, &compressed);
        let psnr = compute_psnr(&rgb, &decoded);
        eprintln!(
            "q={q}: qstep={:.2}, dz={:.2}, levels={}, psnr={psnr:.2} dB, bpp={:.3}",
            config.quantization_step, config.dead_zone, config.wavelet_levels, compressed.bpp(),
        );

        assert!(
            psnr > 25.0,
            "Block DCT at q={q} should have PSNR > 25 dB, got {psnr:.2}"
        );
    }
}

#[test]
fn test_block_dct_noisy_content() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    // Create image with lots of high-frequency content (checker + noise)
    let w = 512u32;
    let h = 512u32;
    let mut rgb = vec![0.0f32; (w * h * 3) as usize];
    let mut seed = 42u64;
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            // Pseudo-random content with full 0-255 range
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = ((seed >> 33) % 256) as f32;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let g = ((seed >> 33) % 256) as f32;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b = ((seed >> 33) % 256) as f32;
            rgb[idx] = r;
            rgb[idx + 1] = g;
            rgb[idx + 2] = b;
        }
    }

    let mut config = crate::quality_preset(75);
    config.transform_type = TransformType::BlockDCT8;
    config.cfl_enabled = false;
    config.adaptive_quantization = false;
    config.use_fused_quantize_histogram = false;

    let compressed = enc.encode(&ctx, &rgb, w, h, &config);
    let decoded = dec.decode(&ctx, &compressed);
    let psnr = compute_psnr(&rgb, &decoded);
    eprintln!("DCT noisy q=75: psnr={psnr:.2} dB, bpp={:.3}", compressed.bpp());

    // Now with q=99
    let mut config99 = crate::quality_preset(99);
    config99.transform_type = TransformType::BlockDCT8;
    config99.cfl_enabled = false;
    config99.adaptive_quantization = false;
    config99.use_fused_quantize_histogram = false;

    let compressed99 = enc.encode(&ctx, &rgb, w, h, &config99);
    let decoded99 = dec.decode(&ctx, &compressed99);
    let psnr99 = compute_psnr(&rgb, &decoded99);
    eprintln!("DCT noisy q=99: psnr={psnr99:.2} dB, bpp={:.3}", compressed99.bpp());

    // Verify the compressed config is correct
    eprintln!(
        "Compressed config: transform={:?}, wavelet_levels={}, tile_size={}",
        compressed99.config.transform_type,
        compressed99.config.wavelet_levels,
        compressed99.config.tile_size,
    );

    // Test direct quantize→dequantize without entropy coding
    // by encoding and reading back the quantized buffer
    // Use the fused block directly for a single-plane test
    {
        use crate::encoder::fused_block::FusedBlock;
        use crate::encoder::block_transform::{BlockTransform, BlockTransformType};
        use crate::encoder::quantize::Quantizer;

        let fused = FusedBlock::new(&ctx);
        let bt = BlockTransform::new(&ctx);
        let quant = Quantizer::new(&ctx);

        let sz = (w * h) as usize;
        // Create one plane of YCoCg data (just use first channel scaled)
        let plane_data: Vec<f32> = (0..sz).map(|i| rgb[i * 3]).collect();
        let plane_bytes = (sz * 4) as u64;

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let input_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_input"),
            contents: bytemuck::cast_slice(&plane_data),
            usage,
        });
        let quant_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_quant"),
            size: plane_bytes,
            usage,
            mapped_at_creation: false,
        });
        let recon_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_recon"),
            size: plane_bytes,
            usage,
            mapped_at_creation: false,
        });
        let dequant_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_dequant"),
            size: plane_bytes,
            usage,
            mapped_at_creation: false,
        });
        let idct_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_idct"),
            size: plane_bytes,
            usage,
            mapped_at_creation: false,
        });

        let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_dct_roundtrip"),
        });

        // Forward DCT + quantize
        fused.dispatch(&ctx, &mut cmd, &input_buf, &quant_buf, &recon_buf, w, h, 2.0, 0.0);

        // Dequantize (mimicking decoder path)
        let uniform_weights = [1.0f32; 16];
        quant.dispatch_adaptive(
            &ctx, &mut cmd, &quant_buf, &dequant_buf,
            sz as u32, 2.0, 0.0, false, // dequantize
            w, h, 256, 0, &uniform_weights, None,
        );

        // Inverse DCT
        bt.dispatch(&ctx, &mut cmd, &dequant_buf, &idct_buf, w, h, false, BlockTransformType::DCT8);

        ctx.queue.submit(Some(cmd.finish()));

        // Read back results
        let recon = crate::gpu_util::read_buffer_f32(&ctx, &recon_buf, sz);
        let idct_result = crate::gpu_util::read_buffer_f32(&ctx, &idct_buf, sz);
        let quant_data = crate::gpu_util::read_buffer_f32(&ctx, &quant_buf, sz);
        let dequant_data = crate::gpu_util::read_buffer_f32(&ctx, &dequant_buf, sz);

        // Check recon from fused kernel (encode-side local decode)
        let psnr_recon = compute_psnr_single(&plane_data, &recon);
        // Check manual dequant+IDCT (decode-side)
        let psnr_idct = compute_psnr_single(&plane_data, &idct_result);

        eprintln!("Direct roundtrip (no entropy):");
        eprintln!("  Fused recon PSNR: {psnr_recon:.2} dB");
        eprintln!("  Manual dequant+IDCT PSNR: {psnr_idct:.2} dB");

        // Check if quant and dequant match
        let mut dq_match = true;
        for i in 0..10 {
            let q = quant_data[i];
            let dq = dequant_data[i];
            let expected_dq = q * 2.0; // step_size=2.0
            if (dq - expected_dq).abs() > 0.01 {
                dq_match = false;
                eprintln!("  dequant mismatch at {i}: quant={q}, dequant={dq}, expected={expected_dq}");
            }
        }
        if dq_match {
            eprintln!("  First 10 dequant values match expected");
        }

        // Sample first few values
        eprintln!("  Sample quant: {:?}", &quant_data[..5]);
        eprintln!("  Sample dequant: {:?}", &dequant_data[..5]);
        eprintln!("  Sample idct result: {:?}", &idct_result[..5]);
        eprintln!("  Sample original: {:?}", &plane_data[..5]);
        eprintln!("  Sample fused recon: {:?}", &recon[..5]);

        // Direct Rice roundtrip test on the quantized data
        let quant_i32: Vec<i32> = quant_data.iter().map(|&v| v.round() as i32).collect();

        // Split into tiles and Rice encode/decode each tile
        let tile_size = 256usize;
        let tiles_x = w as usize / tile_size;
        let tiles_y = h as usize / tile_size;
        let mut max_err = 0i32;
        let mut err_count = 0u64;

        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let mut tile_coeffs = vec![0i32; tile_size * tile_size];
                for row in 0..tile_size {
                    for col in 0..tile_size {
                        let plane_y = ty * tile_size + row;
                        let plane_x = tx * tile_size + col;
                        tile_coeffs[row * tile_size + col] =
                            quant_i32[plane_y * w as usize + plane_x];
                    }
                }

                let encoded = rice::rice_encode_tile(&tile_coeffs, tile_size as u32, 0);
                let decoded = rice::rice_decode_tile(&encoded);

                for (i, (&orig, &dec)) in tile_coeffs.iter().zip(decoded.iter()).enumerate() {
                    if orig != dec {
                        let err = (orig - dec).abs();
                        if err > max_err {
                            max_err = err;
                        }
                        err_count += 1;
                        if err_count <= 5 {
                            let row = i / tile_size;
                            let col = i % tile_size;
                            eprintln!(
                                "  Rice mismatch tile({tx},{ty}) pos({col},{row}): orig={orig}, decoded={dec}"
                            );
                        }
                    }
                }
            }
        }
        eprintln!("Rice roundtrip: {err_count} errors, max_err={max_err}");

        // Now verify the ACTUAL compressed frame's Rice tiles
        // CPU-decode the Rice tiles from the compressed frame and compare to the direct quant data
        if let EntropyData::Rice(ref tiles) = compressed99.entropy {
            let tiles_per_plane = (tiles_x as usize) * (tiles_y as usize);
            eprintln!("Rice tiles: {} total, {} per plane, num_levels={}", tiles.len(), tiles_per_plane, tiles[0].num_levels);

            // Decode plane 0 (Y) from the compressed frame's Rice tiles
            let mut cpu_decoded_plane = vec![0.0f32; sz];
            for t in 0..tiles_per_plane {
                let tx_i = t % tiles_x as usize;
                let ty_i = t / tiles_x as usize;
                let decoded_tile = rice::rice_decode_tile(&tiles[t]);
                for row in 0..tile_size {
                    for col in 0..tile_size {
                        let py = ty_i * tile_size + row;
                        let px = tx_i * tile_size + col;
                        if py < h as usize && px < w as usize {
                            cpu_decoded_plane[py * w as usize + px] =
                                decoded_tile[row * tile_size + col] as f32;
                        }
                    }
                }
            }

            // Compare CPU-decoded Rice data to direct GPU quantized data
            let mut rice_err_count = 0u64;
            let mut rice_max_err = 0.0f32;
            for i in 0..sz {
                let diff = (cpu_decoded_plane[i] - quant_data[i]).abs();
                if diff > 0.5 {
                    rice_err_count += 1;
                    if diff > rice_max_err {
                        rice_max_err = diff;
                    }
                    if rice_err_count <= 5 {
                        let y = i / w as usize;
                        let x = i % w as usize;
                        eprintln!(
                            "  GPU Rice vs direct: pos({x},{y}): rice_decoded={}, direct_quant={}",
                            cpu_decoded_plane[i], quant_data[i]
                        );
                    }
                }
            }
            eprintln!(
                "GPU Rice vs direct quant: {rice_err_count} mismatches, max_err={rice_max_err:.1}"
            );

            // Check for stream overflow
            let max_stream_bytes = 512u32;
            let mut overflow_count = 0;
            let mut max_stream_len = 0u32;
            for (i, tile) in tiles.iter().enumerate() {
                for (s, &len) in tile.stream_lengths.iter().enumerate() {
                    if len > max_stream_len {
                        max_stream_len = len;
                    }
                    if len > max_stream_bytes {
                        overflow_count += 1;
                        if overflow_count <= 3 {
                            eprintln!("  OVERFLOW: tile {i} stream {s}: {len} bytes (max={max_stream_bytes})");
                        }
                    }
                }
            }
            eprintln!("Max stream length: {max_stream_len} bytes, overflows: {overflow_count}");
        }
    }

    // Test with flat constant image to isolate
    let flat_rgb = vec![128.0f32; (w * h * 3) as usize];
    let compressed_flat = enc.encode(&ctx, &flat_rgb, w, h, &config99);
    let decoded_flat = dec.decode(&ctx, &compressed_flat);
    let psnr_flat = compute_psnr(&flat_rgb, &decoded_flat);
    eprintln!("DCT flat q=99: psnr={psnr_flat:.2} dB, bpp={:.3}", compressed_flat.bpp());

    // Test with checker pattern (sharp edges = high frequency)
    let mut checker_rgb = vec![0.0f32; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let val = if (x / 4 + y / 4) % 2 == 0 { 200.0 } else { 50.0 };
            let idx = ((y * w + x) * 3) as usize;
            checker_rgb[idx] = val;
            checker_rgb[idx + 1] = val;
            checker_rgb[idx + 2] = val;
        }
    }
    let compressed_checker = enc.encode(&ctx, &checker_rgb, w, h, &config99);
    let decoded_checker = dec.decode(&ctx, &compressed_checker);
    let psnr_checker = compute_psnr(&checker_rgb, &decoded_checker);
    eprintln!("DCT checker q=99: psnr={psnr_checker:.2} dB, bpp={:.3}", compressed_checker.bpp());

    // Test with grayscale noisy data (R=G=B) — trivial color conversion
    let mut gray_rgb = vec![0.0f32; (w * h * 3) as usize];
    let mut seed2 = 42u64;
    for i in 0..(w * h) as usize {
        seed2 = seed2.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let val = ((seed2 >> 33) % 256) as f32;
        gray_rgb[i * 3] = val;
        gray_rgb[i * 3 + 1] = val;
        gray_rgb[i * 3 + 2] = val;
    }
    let compressed_gray = enc.encode(&ctx, &gray_rgb, w, h, &config99);
    let decoded_gray = dec.decode(&ctx, &compressed_gray);
    let psnr_gray = compute_psnr(&gray_rgb, &decoded_gray);
    eprintln!("DCT gray-noisy q=99: psnr={psnr_gray:.2} dB, bpp={:.3}", compressed_gray.bpp());

    assert!(psnr_flat > 50.0, "Block DCT flat should be nearly lossless: {psnr_flat:.2}");
    assert!(psnr > 20.0, "Block DCT noisy q=75 PSNR too low: {psnr:.2}");
}

#[test]
fn test_block_dct_color_debug() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;
    let npix = (w * h) as usize;

    // Create random color noise
    let mut rgb = vec![0.0f32; npix * 3];
    let mut seed = 42u64;
    for i in 0..npix {
        for c in 0..3 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rgb[i * 3 + c] = ((seed >> 33) % 256) as f32;
        }
    }

    let mut config = crate::quality_preset(99);
    config.cfl_enabled = false;
    config.adaptive_quantization = false;
    config.use_fused_quantize_histogram = false;

    // --- DCT path ---
    config.transform_type = TransformType::BlockDCT8;
    let comp_dct = enc.encode(&ctx, &rgb, w, h, &config);
    let dec_dct = dec.decode(&ctx, &comp_dct);

    // Immediately read back encoder buffers BEFORE any other encode
    let enc_mc_out = crate::gpu_util::read_buffer_f32(&ctx, &enc.cached.as_ref().unwrap().mc_out, npix);
    let enc_plane_a = crate::gpu_util::read_buffer_f32(&ctx, &enc.cached.as_ref().unwrap().plane_a, npix);
    let enc_ref_upload = crate::gpu_util::read_buffer_f32(&ctx, &enc.cached.as_ref().unwrap().ref_upload, npix);
    let enc_plane_b = crate::gpu_util::read_buffer_f32(&ctx, &enc.cached.as_ref().unwrap().plane_b, npix);

    // Per-channel PSNR for DCT
    let mut r_orig = vec![0.0f32; npix];
    let mut g_orig = vec![0.0f32; npix];
    let mut b_orig = vec![0.0f32; npix];
    let mut r_dec = vec![0.0f32; npix];
    let mut g_dec = vec![0.0f32; npix];
    let mut b_dec = vec![0.0f32; npix];
    for i in 0..npix {
        r_orig[i] = rgb[i * 3];
        g_orig[i] = rgb[i * 3 + 1];
        b_orig[i] = rgb[i * 3 + 2];
        r_dec[i] = dec_dct[i * 3];
        g_dec[i] = dec_dct[i * 3 + 1];
        b_dec[i] = dec_dct[i * 3 + 2];
    }
    let psnr_r = compute_psnr_single(&r_orig, &r_dec);
    let psnr_g = compute_psnr_single(&g_orig, &g_dec);
    let psnr_b = compute_psnr_single(&b_orig, &b_dec);
    let psnr_all = compute_psnr(&rgb, &dec_dct);
    eprintln!("DCT color q=99: R={psnr_r:.2} G={psnr_g:.2} B={psnr_b:.2} all={psnr_all:.2} dB");

    // Print first 8 pixel errors
    eprintln!("First 8 pixels (orig → decoded → error):");
    for i in 0..8 {
        let ro = rgb[i*3]; let go = rgb[i*3+1]; let bo = rgb[i*3+2];
        let rd = dec_dct[i*3]; let gd = dec_dct[i*3+1]; let bd = dec_dct[i*3+2];
        eprintln!("  px{i}: R({ro:.0}→{rd:.1} Δ{:.1}) G({go:.0}→{gd:.1} Δ{:.1}) B({bo:.0}→{bd:.1} Δ{:.1})",
            rd-ro, gd-go, bd-bo);
    }

    // Compute error statistics and find worst pixels
    let mut max_err = 0.0f32;
    let mut sum_abs_err = 0.0f64;
    let mut worst_pixels: Vec<(usize, f32)> = Vec::new();
    for i in 0..npix {
        let mut px_mse = 0.0f32;
        for c in 0..3 {
            let err = (dec_dct[i*3+c] - rgb[i*3+c]).abs();
            if err > max_err { max_err = err; }
            sum_abs_err += err as f64;
            px_mse += err * err;
        }
        if px_mse > 50.0 { // significant per-pixel error
            worst_pixels.push((i, px_mse));
        }
    }
    worst_pixels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let avg_err = sum_abs_err / (npix * 3) as f64;
    eprintln!("DCT error: max={max_err:.2}, avg={avg_err:.2}, large_error_pixels={}", worst_pixels.len());
    for &(idx, mse) in worst_pixels.iter().take(10) {
        let px = idx % w as usize;
        let py = idx / w as usize;
        let bx = px / 8; let by = py / 8;
        eprintln!("  worst px({px},{py}) block({bx},{by}): mse={mse:.1} orig=[{:.0},{:.0},{:.0}] dec=[{:.1},{:.1},{:.1}]",
            rgb[idx*3], rgb[idx*3+1], rgb[idx*3+2],
            dec_dct[idx*3], dec_dct[idx*3+1], dec_dct[idx*3+2]);
    }

    // --- Wavelet path (same data) ---
    config.transform_type = TransformType::Wavelet;
    let comp_wav = enc.encode(&ctx, &rgb, w, h, &config);
    let dec_wav = dec.decode(&ctx, &comp_wav);

    for i in 0..npix {
        r_dec[i] = dec_wav[i * 3];
        g_dec[i] = dec_wav[i * 3 + 1];
        b_dec[i] = dec_wav[i * 3 + 2];
    }
    let wpsnr_r = compute_psnr_single(&r_orig, &r_dec);
    let wpsnr_g = compute_psnr_single(&g_orig, &g_dec);
    let wpsnr_b = compute_psnr_single(&b_orig, &b_dec);
    let wpsnr_all = compute_psnr(&rgb, &dec_wav);
    eprintln!("Wav color q=99: R={wpsnr_r:.2} G={wpsnr_g:.2} B={wpsnr_b:.2} all={wpsnr_all:.2} dB");

    // --- Test with only R varying (G=B=128) ---
    let mut rgb_ronly = vec![128.0f32; npix * 3];
    seed = 42;
    for i in 0..npix {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rgb_ronly[i * 3] = ((seed >> 33) % 256) as f32;
    }
    config.transform_type = TransformType::BlockDCT8;
    let comp_ronly = enc.encode(&ctx, &rgb_ronly, w, h, &config);
    let dec_ronly = dec.decode(&ctx, &comp_ronly);
    let psnr_ronly = compute_psnr(&rgb_ronly, &dec_ronly);
    eprintln!("DCT R-only q=99: psnr={psnr_ronly:.2} dB");

    // --- Test with only G varying ---
    let mut rgb_gonly = vec![128.0f32; npix * 3];
    seed = 42;
    for i in 0..npix {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rgb_gonly[i * 3 + 1] = ((seed >> 33) % 256) as f32;
    }
    let comp_gonly = enc.encode(&ctx, &rgb_gonly, w, h, &config);
    let dec_gonly = dec.decode(&ctx, &comp_gonly);
    let psnr_gonly = compute_psnr(&rgb_gonly, &dec_gonly);
    eprintln!("DCT G-only q=99: psnr={psnr_gonly:.2} dB");

    // --- Test with only B varying ---
    let mut rgb_bonly = vec![128.0f32; npix * 3];
    seed = 42;
    for i in 0..npix {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rgb_bonly[i * 3 + 2] = ((seed >> 33) % 256) as f32;
    }
    let comp_bonly = enc.encode(&ctx, &rgb_bonly, w, h, &config);
    let dec_bonly = dec.decode(&ctx, &comp_bonly);
    let psnr_bonly = compute_psnr(&rgb_bonly, &dec_bonly);
    eprintln!("DCT B-only q=99: psnr={psnr_bonly:.2} dB");

    // Check: is the overall problem in R, G, or B specifically?
    eprintln!("\nSummary:");
    eprintln!("  DCT color:  all={psnr_all:.2} R={psnr_r:.2} G={psnr_g:.2} B={psnr_b:.2}");
    eprintln!("  Wavelet:    all={wpsnr_all:.2} R={wpsnr_r:.2} G={wpsnr_g:.2} B={wpsnr_b:.2}");
    eprintln!("  DCT R-only: {psnr_ronly:.2}");
    eprintln!("  DCT G-only: {psnr_gonly:.2}");
    eprintln!("  DCT B-only: {psnr_bonly:.2}");

    // --- Also test: is the issue with negative-range input? ---
    // Create an image where R=128+noise, G=128, B=128-noise
    // This gives Co = 2*noise (large), Cg ≈ -noise (large negative), Y ≈ 128
    let mut rgb_neg = vec![128.0f32; npix * 3];
    seed = 42;
    for i in 0..npix {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let noise = ((seed >> 33) % 128) as f32;  // [0, 127]
        rgb_neg[i * 3] = 128.0 + noise;      // R = [128, 255]
        rgb_neg[i * 3 + 2] = 128.0 - noise;  // B = [1, 128]
        // G stays 128
    }
    let comp_neg = enc.encode(&ctx, &rgb_neg, w, h, &config);
    let dec_neg = dec.decode(&ctx, &comp_neg);
    let psnr_neg = compute_psnr(&rgb_neg, &dec_neg);
    eprintln!("DCT Co-heavy q=99: psnr={psnr_neg:.2} dB");

    // Check overflow for R-only encoded frame
    if let EntropyData::Rice(ref tiles) = comp_ronly.entropy {
        let mut overflow_count = 0;
        let mut max_stream_len = 0u32;
        for (i, tile) in tiles.iter().enumerate() {
            for (s, &len) in tile.stream_lengths.iter().enumerate() {
                if len > max_stream_len { max_stream_len = len; }
                if len >= 512 {
                    overflow_count += 1;
                    if overflow_count <= 5 {
                        eprintln!("  R-only OVERFLOW: tile {i} stream {s}: {len} bytes");
                    }
                }
            }
        }
        eprintln!("R-only Rice: max_stream={max_stream_len} bytes, overflows={overflow_count}");
    }

    // Test: encode R-only with CPU Rice (no overflow possible)
    {
        let mut cpu_config = config.clone();
        cpu_config.context_adaptive = true; // forces CPU encode/decode
        let comp_cpu = enc.encode(&ctx, &rgb_ronly, w, h, &cpu_config);
        let dec_cpu = dec.decode(&ctx, &comp_cpu);
        let psnr_cpu = compute_psnr(&rgb_ronly, &dec_cpu);
        eprintln!("DCT R-only CPU-Rice q=99: psnr={psnr_cpu:.2} dB");
    }

    // ---- Test GPU color convert + deinterleave vs CPU ----
    {
        use crate::encoder::color::ColorConverter;
        use crate::encoder::interleave::PlaneDeinterleaver;

        let color = ColorConverter::new(&ctx);
        let deinterleaver = PlaneDeinterleaver::new(&ctx);

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let rgb_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_rgb"),
            contents: bytemuck::cast_slice(&rgb),
            usage,
        });
        let ycocg_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_ycocg"), size: (npix * 3 * 4) as u64, usage, mapped_at_creation: false,
        });
        let p0_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("p0"), size: (npix * 4) as u64, usage, mapped_at_creation: false,
        });
        let p1_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("p1"), size: (npix * 4) as u64, usage, mapped_at_creation: false,
        });
        let p2_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("p2"), size: (npix * 4) as u64, usage, mapped_at_creation: false,
        });

        let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_color"),
        });
        color.dispatch(&ctx, &mut cmd, &rgb_buf, &ycocg_buf, w, h, true, false);
        deinterleaver.dispatch(&ctx, &mut cmd, &ycocg_buf, &p0_buf, &p1_buf, &p2_buf, npix as u32);
        ctx.queue.submit(Some(cmd.finish()));

        let gpu_y = crate::gpu_util::read_buffer_f32(&ctx, &p0_buf, npix);
        let gpu_co = crate::gpu_util::read_buffer_f32(&ctx, &p1_buf, npix);
        let gpu_cg = crate::gpu_util::read_buffer_f32(&ctx, &p2_buf, npix);

        // Compare with CPU YCoCg
        let mut cpu_y = vec![0.0f32; npix];
        let mut cpu_co = vec![0.0f32; npix];
        let mut cpu_cg = vec![0.0f32; npix];
        for i in 0..npix {
            let r = rgb[i * 3]; let g = rgb[i * 3 + 1]; let b = rgb[i * 3 + 2];
            let co = r - b;
            let t = b + co * 0.5;
            let cg = g - t;
            let y_val = t + cg * 0.5;
            cpu_y[i] = y_val;
            cpu_co[i] = co;
            cpu_cg[i] = cg;
        }

        let y_psnr = compute_psnr_single(&cpu_y, &gpu_y);
        let co_psnr = compute_psnr_single(&cpu_co, &gpu_co);
        let cg_psnr = compute_psnr_single(&cpu_cg, &gpu_cg);
        eprintln!("GPU vs CPU YCoCg: Y={y_psnr:.2} Co={co_psnr:.2} Cg={cg_psnr:.2} dB");

        // Check first few values
        for i in 0..4 {
            eprintln!("  px{i}: GPU Y={:.4} CPU Y={:.4} | GPU Co={:.4} CPU Co={:.4} | GPU Cg={:.4} CPU Cg={:.4}",
                gpu_y[i], cpu_y[i], gpu_co[i], cpu_co[i], gpu_cg[i], cpu_cg[i]);
        }
    }

    // ---- Critical test: manual YCoCg + per-plane DCT roundtrip + inverse YCoCg ----
    // This bypasses the encoder/decoder pipeline entirely
    {
        use crate::encoder::fused_block::FusedBlock;
        use crate::encoder::block_transform::{BlockTransform, BlockTransformType};
        use crate::encoder::quantize::Quantizer;

        let fused = FusedBlock::new(&ctx);
        let bt = BlockTransform::new(&ctx);
        let quant = Quantizer::new(&ctx);

        let sz = npix;
        let step = 2.0f32;

        // Manual CPU YCoCg forward (matching the shader's lossy mode)
        let mut y_plane = vec![0.0f32; sz];
        let mut co_plane = vec![0.0f32; sz];
        let mut cg_plane = vec![0.0f32; sz];
        for i in 0..sz {
            let r = rgb[i * 3];
            let g = rgb[i * 3 + 1];
            let b = rgb[i * 3 + 2];
            let co = r - b;
            let t = b + co * 0.5;
            let cg = g - t;
            let y = t + cg * 0.5;
            y_plane[i] = y;
            co_plane[i] = co;
            cg_plane[i] = cg;
        }

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let plane_bytes = (sz * 4) as u64;

        // Per-plane DCT roundtrip for all 3 planes
        let mut decoded_planes: Vec<Vec<f32>> = Vec::new();
        for (name, plane_data) in [("Y", &y_plane), ("Co", &co_plane), ("Cg", &cg_plane)] {
            let input_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test_input"),
                contents: bytemuck::cast_slice(plane_data),
                usage,
            });
            let quant_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test_quant"), size: plane_bytes, usage, mapped_at_creation: false,
            });
            let recon_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test_recon"), size: plane_bytes, usage, mapped_at_creation: false,
            });
            let dequant_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test_dequant"), size: plane_bytes, usage, mapped_at_creation: false,
            });
            let idct_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test_idct"), size: plane_bytes, usage, mapped_at_creation: false,
            });

            let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("manual_roundtrip"),
            });
            fused.dispatch(&ctx, &mut cmd, &input_buf, &quant_buf, &recon_buf, w, h, step, 0.0);
            let uniform_weights = [1.0f32; 16];
            quant.dispatch_adaptive(&ctx, &mut cmd, &quant_buf, &dequant_buf,
                sz as u32, step, 0.0, false, w, h, 256, 0, &uniform_weights, None);
            bt.dispatch(&ctx, &mut cmd, &dequant_buf, &idct_buf, w, h, false, BlockTransformType::DCT8);
            ctx.queue.submit(Some(cmd.finish()));

            let result = crate::gpu_util::read_buffer_f32(&ctx, &idct_buf, sz);
            let psnr = compute_psnr_single(plane_data, &result);
            eprintln!("Manual {name} plane DCT roundtrip: {psnr:.2} dB");
            decoded_planes.push(result);
        }

        // Manual inverse YCoCg
        let mut manual_rgb = vec![0.0f32; sz * 3];
        for i in 0..sz {
            let y = decoded_planes[0][i];
            let co = decoded_planes[1][i];
            let cg = decoded_planes[2][i];
            let t = y - cg * 0.5;
            let g = cg + t;
            let b = t - co * 0.5;
            let r = b + co;
            manual_rgb[i * 3] = r;
            manual_rgb[i * 3 + 1] = g;
            manual_rgb[i * 3 + 2] = b;
        }
        let manual_psnr = compute_psnr(&rgb, &manual_rgb);
        eprintln!("Manual YCoCg + DCT roundtrip + inverse YCoCg: {manual_psnr:.2} dB");
        eprintln!("  vs full pipeline DCT: {psnr_all:.2} dB");

        if manual_psnr > 45.0 && psnr_all < 40.0 {
            eprintln!("BUG CONFIRMED: Manual roundtrip works but full pipeline doesn't!");
            eprintln!("  Issue is in encoder/decoder pipeline, not in DCT or color math.");
        }

        // --- Compare early-readback encoder plane_a vs CPU YCoCg ---
        {
            // Compute CPU YCoCg
            let mut cpu_y = vec![0.0f32; sz];
            let mut cpu_co = vec![0.0f32; sz];
            let mut cpu_cg = vec![0.0f32; sz];
            for i in 0..sz {
                let r = rgb[i*3]; let g = rgb[i*3+1]; let b = rgb[i*3+2];
                let co = r - b;
                let t = b + co * 0.5;
                let cg = g - t;
                cpu_y[i] = t + cg * 0.5;
                cpu_co[i] = co;
                cpu_cg[i] = cg;
            }
            let pa_psnr = compute_psnr_single(&cpu_y, &enc_plane_a);
            eprintln!("Early plane_a vs CPU Y: {pa_psnr:.2} dB");
            if pa_psnr < 90.0 {
                for i in 0..4 {
                    eprintln!("  plane_a[{i}]={:.4} cpu_y={:.4} diff={:.4}",
                        enc_plane_a[i], cpu_y[i], enc_plane_a[i]-cpu_y[i]);
                }
            }

            // Do fresh DCT on the early plane_a data and compare with early mc_out
            let fused = crate::encoder::fused_block::FusedBlock::new(&ctx);
            let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
            let chk_in = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chk_in"), contents: bytemuck::cast_slice(&enc_plane_a), usage,
            });
            let chk_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chk_out"), size: (sz*4) as u64, usage, mapped_at_creation: false,
            });
            let chk_recon = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chk_recon"), size: (sz*4) as u64, usage, mapped_at_creation: false,
            });
            let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chk"),
            });
            fused.dispatch(&ctx, &mut cmd, &chk_in, &chk_out, &chk_recon, w, h, 2.0, 0.0);
            ctx.queue.submit(Some(cmd.finish()));
            let chk_quant = crate::gpu_util::read_buffer_f32(&ctx, &chk_out, sz);

            let mut mc_mismatch = 0;
            for i in 0..sz {
                if (enc_mc_out[i] - chk_quant[i]).abs() > 0.5 { mc_mismatch += 1; }
            }
            eprintln!("Early mc_out vs fresh DCT on early plane_a: {mc_mismatch} mismatches");
        }

        // --- Compare encoder's quantized coefficients vs manual ---
        // CPU-decode the Rice tiles from the compressed frame
        if let EntropyData::Rice(ref tiles) = comp_dct.entropy {
            let tile_size = 256usize;
            let tiles_x = w as usize / tile_size;
            let tiles_y = h as usize / tile_size;
            let tiles_per_plane = tiles_x * tiles_y;
            let sz = npix;

            // Also get the manual quantized data for each plane
            let mut manual_quant: Vec<Vec<f32>> = Vec::new();
            for plane_data in [&y_plane, &co_plane, &cg_plane] {
                let input_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mq_input"),
                    contents: bytemuck::cast_slice(plane_data),
                    usage,
                });
                let quant_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mq_quant"), size: plane_bytes, usage, mapped_at_creation: false,
                });
                let recon_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mq_recon"), size: plane_bytes, usage, mapped_at_creation: false,
                });
                let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mq"),
                });
                fused.dispatch(&ctx, &mut cmd, &input_buf, &quant_buf, &recon_buf, w, h, step, 0.0);
                ctx.queue.submit(Some(cmd.finish()));
                manual_quant.push(crate::gpu_util::read_buffer_f32(&ctx, &quant_buf, sz));
            }

            for p in 0..3 {
                let plane_name = ["Y", "Co", "Cg"][p];
                let plane_tiles = &tiles[p * tiles_per_plane..(p + 1) * tiles_per_plane];

                // CPU-decode Rice tiles
                let mut rice_decoded = vec![0.0f32; sz];
                for t in 0..tiles_per_plane {
                    let tx_i = t % tiles_x;
                    let ty_i = t / tiles_x;
                    let decoded = crate::encoder::rice::rice_decode_tile(&plane_tiles[t]);
                    for row in 0..tile_size {
                        for col in 0..tile_size {
                            let py_coord = ty_i * tile_size + row;
                            let px_coord = tx_i * tile_size + col;
                            if py_coord < h as usize && px_coord < w as usize {
                                rice_decoded[py_coord * w as usize + px_coord] =
                                    decoded[row * tile_size + col] as f32;
                            }
                        }
                    }
                }

                // Compare with manual quantized data
                let mut mismatch_count = 0;
                let mut max_diff = 0.0f32;
                for i in 0..sz {
                    let diff = (rice_decoded[i] - manual_quant[p][i]).abs();
                    if diff > 0.5 {
                        mismatch_count += 1;
                        if diff > max_diff { max_diff = diff; }
                        if mismatch_count <= 3 {
                            let px_coord = i % w as usize;
                            let py_coord = i / w as usize;
                            eprintln!("  {plane_name} mismatch at ({px_coord},{py_coord}): pipeline={} manual={}",
                                rice_decoded[i], manual_quant[p][i]);
                        }
                    }
                }
                eprintln!("Encoder {plane_name}: {mismatch_count} mismatches vs manual (max_diff={max_diff:.1})");
            }
        }
    }

    assert!(psnr_all > 45.0, "DCT color q=99 should be >45 dB, got {psnr_all:.2}");
}
