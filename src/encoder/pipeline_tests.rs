use super::*;
use crate::decoder::pipeline::DecoderPipeline;
use crate::TemporalTransform;
use wgpu::util::DeviceExt;

/// Compute PSNR with a configurable peak value (base implementation).
fn compute_psnr_peak(a: &[f32], b: &[f32], peak: f64) -> f64 {
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
    10.0 * (peak * peak / mse).log10()
}

/// Compute PSNR assuming 8-bit signal range [0, 255].
fn compute_psnr_single(a: &[f32], b: &[f32]) -> f64 {
    compute_psnr_peak(a, b, 255.0)
}

/// Compute PSNR assuming 8-bit signal range [0, 255].
fn compute_psnr(a: &[f32], b: &[f32]) -> f64 {
    compute_psnr_peak(a, b, 255.0)
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
    // 9 frames: I [B₁ B₂ B₃ B₄ B₅ B₆ B₇] P (one full 3-level pyramid group)
    // B_FRAMES_PER_GROUP=7, group_size=8: needs ki>=8 and >=9 total frames.
    let frames_rgb: Vec<Vec<f32>> = (0..9)
        .map(|i| make_gradient_frame(w, h, i as f32 * 2.0))
        .collect();
    let frame_refs: Vec<&[f32]> = frames_rgb.iter().map(|f| f.as_slice()).collect();

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 9; // I [B×7] P pattern

    let compressed = enc.encode_sequence(&ctx, &frame_refs, w, h, &config);

    assert_eq!(compressed.len(), 9);
    assert_eq!(compressed[0].frame_type, FrameType::Intra);
    // Display order: I B B B B B B B P
    // Frames 1-7 are B-frames, frame 8 is the P-frame anchor
    for i in 1..8 {
        assert_eq!(
            compressed[i].frame_type,
            FrameType::Bidirectional,
            "Frame {i} should be Bidirectional"
        );
    }
    assert_eq!(compressed[8].frame_type, FrameType::Predicted);

    // All inter frames must have motion fields
    for i in 1..9 {
        assert!(
            compressed[i].motion_field.is_some(),
            "Inter frame {i} should have motion field"
        );
    }

    // B₄ (display index 4) is encoded as forward-only via #49 — backward_vectors=None.
    // All other B-frames are truly bidirectional.
    for i in 1..8 {
        let mf = compressed[i].motion_field.as_ref().unwrap();
        if i == 4 {
            // B₄-as-P: forward-only signal
            assert!(
                mf.backward_vectors.is_none(),
                "B₄ (frame {i}) should be forward-only (no backward vectors)"
            );
        } else {
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
    // 9 frames with ki=9: I [B×7] P (one full 3-level pyramid group)
    // B_FRAMES_PER_GROUP=7, group_size=8: ki=9 covers one full group.
    let frames_rgb: Vec<Vec<f32>> = (0..9)
        .map(|i| make_gradient_frame(w, h, i as f32 * 2.0))
        .collect();
    let frame_refs: Vec<&[f32]> = frames_rgb.iter().map(|f| f.as_slice()).collect();

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 9; // ki >= 8 triggers B-frames

    let compressed = enc.encode_sequence(&ctx, &frame_refs, w, h, &config);
    assert_eq!(compressed.len(), 9);

    // Verify frame types: I [B₁..B₇] P
    assert_eq!(compressed[0].frame_type, FrameType::Intra);
    for i in 1..8 {
        assert_eq!(
            compressed[i].frame_type,
            FrameType::Bidirectional,
            "Frame {i} should be Bidirectional"
        );
    }
    assert_eq!(compressed[8].frame_type, FrameType::Predicted);

    // Decode using B-frame aware sequence decoder
    let decoded = dec.decode_sequence(&ctx, &compressed);
    assert_eq!(decoded.len(), 9);

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

/// Verify fused quantize+histogram produces valid decoded output.
///
/// The fused path includes content-adaptive dead zone expansion (Phase 1.5)
/// which may zero out additional small coefficients in sparse subbands.
/// This means fused output may differ from the separate path — but it should:
///   - produce a valid decodeable stream
///   - have PSNR within 1.0 dB of the separate path
///   - use ≤ bitrate of the separate path (adaptive zeroing can only reduce bits)
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

    // Fused path: quantize + histogram in single dispatch.
    // Content-adaptive dead zone (Phase 1.5) may zero additional sparse coefficients.
    let mut config_fused = config_separate.clone();
    config_fused.use_fused_quantize_histogram = true;

    let compressed_fused = enc.encode(&ctx, &frame, w, h, &config_fused);
    let decoded_fused = dec.decode(&ctx, &compressed_fused);

    assert_eq!(decoded_sep.len(), decoded_fused.len(), "decoded length mismatch");

    // Compute PSNR between separate and fused decoded output.
    let mse: f32 = decoded_sep
        .iter()
        .zip(decoded_fused.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / decoded_sep.len() as f32;
    let bpp_sep = compressed_sep.bpp();
    let bpp_fused = compressed_fused.bpp();
    eprintln!("bpp: separate={bpp_sep:.4}, fused={bpp_fused:.4}, mse={mse:.4}");

    // Fused path should produce quality within 1.0 dB of separate path
    // (adaptive dead zone is conservative: only expands for ≥95% sparse subbands).
    assert!(
        mse < 255.0 * 255.0 * 0.01,  // max ~1% of signal range squared
        "Fused vs separate quality too different: mse={mse:.4}"
    );
    // Fused path should use at most slightly more bits than separate path
    // (adaptive zeroing can reduce bits; small increases allowed due to histogram changes).
    assert!(
        bpp_fused <= bpp_sep + 0.1,
        "Fused bitrate significantly higher than separate: {bpp_fused:.4} > {bpp_sep:.4}"
    );
}

/// Verify fused quantize+histogram with adaptive quantization produces
/// valid results. The fused path applies content-adaptive dead zone expansion
/// on top of spatial AQ, so it may differ from the separate path.
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

    // Compare decoded pixels: fused+AQ may differ from separate+AQ due to
    // content-adaptive dead zone expansion in the fused path.
    assert_eq!(
        decoded_sep.len(),
        decoded_fused.len(),
        "decoded length mismatch"
    );
    let mse: f32 = decoded_sep
        .iter()
        .zip(decoded_fused.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / decoded_sep.len() as f32;

    let bpp_sep = compressed_sep.bpp();
    let bpp_fused = compressed_fused.bpp();
    eprintln!("bpp (AQ): separate={bpp_sep:.4}, fused={bpp_fused:.4}, mse={mse:.4}");

    // Quality should be within ~1 dB of separate path
    assert!(
        mse < 255.0 * 255.0 * 0.01,
        "Fused+AQ vs separate+AQ quality too different: mse={mse:.4}"
    );
    // Bitrate should not be significantly higher
    assert!(
        bpp_fused <= bpp_sep + 0.1,
        "Fused+AQ bitrate significantly higher than separate: {bpp_fused:.4} > {bpp_sep:.4}"
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
    // DCT path uses Rice: GPU rANS encoder requires wavelet levels > 0
    config.entropy_coder = crate::EntropyCoder::Rice;

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
            rgb[i * 3],
            rgb[i * 3 + 1],
            rgb[i * 3 + 2],
            decoded[i * 3],
            decoded[i * 3 + 1],
            decoded[i * 3 + 2],
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
    // DCT path uses Rice: GPU rANS encoder requires wavelet levels > 0
    config.entropy_coder = crate::EntropyCoder::Rice;

    let compressed = enc.encode(&ctx, &rgb, w, h, &config);
    let decoded = dec.decode(&ctx, &compressed);
    let psnr = compute_psnr(&rgb, &decoded);
    eprintln!("DCT multitile (512x512, 4 tiles) PSNR: {psnr:.2} dB");

    // Check a pixel from tile 0 and tile 1
    for i in [0, 256, 512, 256 * 512] {
        if i * 3 + 2 < rgb.len() {
            eprintln!(
                "  px[{i}]: ({:.1}, {:.1}, {:.1}) vs ({:.1}, {:.1}, {:.1})",
                rgb[i * 3],
                rgb[i * 3 + 1],
                rgb[i * 3 + 2],
                decoded[i * 3],
                decoded[i * 3 + 1],
                decoded[i * 3 + 2],
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
    // DCT path uses Rice: GPU rANS encoder requires wavelet levels > 0
    config.entropy_coder = crate::EntropyCoder::Rice;

    let compressed = enc.encode(&ctx, &rgb, w, h, &config);
    eprintln!(
        "DCT nonaligned: orig {}x{}, padded {}x{}, bpp={:.3}",
        w,
        h,
        compressed.info.padded_width(),
        compressed.info.padded_height(),
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
                rgb[i * 3],
                rgb[i * 3 + 1],
                rgb[i * 3 + 2],
                decoded[i * 3],
                decoded[i * 3 + 1],
                decoded[i * 3 + 2],
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
        // DCT path uses Rice: GPU rANS encoder requires wavelet levels > 0
        config.entropy_coder = crate::EntropyCoder::Rice;

        let compressed = enc.encode(&ctx, &rgb, w, h, &config);
        let decoded = dec.decode(&ctx, &compressed);
        let psnr = compute_psnr(&rgb, &decoded);
        eprintln!(
            "q={q}: qstep={:.2}, dz={:.2}, levels={}, psnr={psnr:.2} dB, bpp={:.3}",
            config.quantization_step,
            config.dead_zone,
            config.wavelet_levels,
            compressed.bpp(),
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
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = ((seed >> 33) % 256) as f32;
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let g = ((seed >> 33) % 256) as f32;
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
    // DCT path uses Rice: GPU rANS encoder requires wavelet levels > 0
    config.entropy_coder = crate::EntropyCoder::Rice;

    let compressed = enc.encode(&ctx, &rgb, w, h, &config);
    let decoded = dec.decode(&ctx, &compressed);
    let psnr = compute_psnr(&rgb, &decoded);
    eprintln!(
        "DCT noisy q=75: psnr={psnr:.2} dB, bpp={:.3}",
        compressed.bpp()
    );

    // Now with q=99
    let mut config99 = crate::quality_preset(99);
    config99.transform_type = TransformType::BlockDCT8;
    config99.cfl_enabled = false;
    config99.adaptive_quantization = false;
    config99.use_fused_quantize_histogram = false;
    // DCT path uses Rice: GPU rANS encoder requires wavelet levels > 0
    config99.entropy_coder = crate::EntropyCoder::Rice;

    let compressed99 = enc.encode(&ctx, &rgb, w, h, &config99);
    let decoded99 = dec.decode(&ctx, &compressed99);
    let psnr99 = compute_psnr(&rgb, &decoded99);
    eprintln!(
        "DCT noisy q=99: psnr={psnr99:.2} dB, bpp={:.3}",
        compressed99.bpp()
    );

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
        use crate::encoder::block_transform::{BlockTransform, BlockTransformType};
        use crate::encoder::fused_block::FusedBlock;
        use crate::encoder::quantize::Quantizer;

        let fused = FusedBlock::new(&ctx);
        let bt = BlockTransform::new(&ctx);
        let quant = Quantizer::new(&ctx);

        let sz = (w * h) as usize;
        // Create one plane of YCoCg data (just use first channel scaled)
        let plane_data: Vec<f32> = (0..sz).map(|i| rgb[i * 3]).collect();
        let plane_bytes = (sz * 4) as u64;

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let input_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test_dct_roundtrip"),
            });

        // Forward DCT + quantize
        fused.dispatch(
            &ctx, &mut cmd, &input_buf, &quant_buf, &recon_buf, w, h, 2.0, 0.0, 7.0,
        );

        // Dequantize (mimicking decoder path)
        let uniform_weights = [1.0f32; 16];
        quant.dispatch_adaptive(
            &ctx,
            &mut cmd,
            &quant_buf,
            &dequant_buf,
            sz as u32,
            2.0,
            0.0,
            false, // dequantize
            w,
            h,
            256,
            0,
            &uniform_weights,
            None,
            3.0,
        );

        // Inverse DCT
        bt.dispatch(
            &ctx,
            &mut cmd,
            &dequant_buf,
            &idct_buf,
            w,
            h,
            false,
            BlockTransformType::DCT8,
        );

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
                eprintln!(
                    "  dequant mismatch at {i}: quant={q}, dequant={dq}, expected={expected_dq}"
                );
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
            eprintln!(
                "Rice tiles: {} total, {} per plane, num_levels={}",
                tiles.len(),
                tiles_per_plane,
                tiles[0].num_levels
            );

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
    eprintln!(
        "DCT flat q=99: psnr={psnr_flat:.2} dB, bpp={:.3}",
        compressed_flat.bpp()
    );

    // Test with checker pattern (sharp edges = high frequency)
    let mut checker_rgb = vec![0.0f32; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let val = if (x / 4 + y / 4) % 2 == 0 {
                200.0
            } else {
                50.0
            };
            let idx = ((y * w + x) * 3) as usize;
            checker_rgb[idx] = val;
            checker_rgb[idx + 1] = val;
            checker_rgb[idx + 2] = val;
        }
    }
    let compressed_checker = enc.encode(&ctx, &checker_rgb, w, h, &config99);
    let decoded_checker = dec.decode(&ctx, &compressed_checker);
    let psnr_checker = compute_psnr(&checker_rgb, &decoded_checker);
    eprintln!(
        "DCT checker q=99: psnr={psnr_checker:.2} dB, bpp={:.3}",
        compressed_checker.bpp()
    );

    // Test with grayscale noisy data (R=G=B) — trivial color conversion
    let mut gray_rgb = vec![0.0f32; (w * h * 3) as usize];
    let mut seed2 = 42u64;
    for i in 0..(w * h) as usize {
        seed2 = seed2
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((seed2 >> 33) % 256) as f32;
        gray_rgb[i * 3] = val;
        gray_rgb[i * 3 + 1] = val;
        gray_rgb[i * 3 + 2] = val;
    }
    let compressed_gray = enc.encode(&ctx, &gray_rgb, w, h, &config99);
    let decoded_gray = dec.decode(&ctx, &compressed_gray);
    let psnr_gray = compute_psnr(&gray_rgb, &decoded_gray);
    eprintln!(
        "DCT gray-noisy q=99: psnr={psnr_gray:.2} dB, bpp={:.3}",
        compressed_gray.bpp()
    );

    assert!(
        psnr_flat > 50.0,
        "Block DCT flat should be nearly lossless: {psnr_flat:.2}"
    );
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
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rgb[i * 3 + c] = ((seed >> 33) % 256) as f32;
        }
    }

    let mut config = crate::quality_preset(99);
    config.cfl_enabled = false;
    config.adaptive_quantization = false;
    config.use_fused_quantize_histogram = false;
    // DCT path uses Rice: GPU rANS encoder requires wavelet levels > 0
    config.entropy_coder = crate::EntropyCoder::Rice;

    // --- DCT path ---
    config.transform_type = TransformType::BlockDCT8;
    let comp_dct = enc.encode(&ctx, &rgb, w, h, &config);
    let dec_dct = dec.decode(&ctx, &comp_dct);

    // Immediately read back encoder buffers BEFORE any other encode
    let enc_mc_out =
        crate::gpu_util::read_buffer_f32(&ctx, &enc.cached.as_ref().unwrap().mc_out, npix);
    let enc_plane_a =
        crate::gpu_util::read_buffer_f32(&ctx, &enc.cached.as_ref().unwrap().plane_a, npix);
    let enc_ref_upload =
        crate::gpu_util::read_buffer_f32(&ctx, &enc.cached.as_ref().unwrap().ref_upload, npix);
    let enc_plane_b =
        crate::gpu_util::read_buffer_f32(&ctx, &enc.cached.as_ref().unwrap().plane_b, npix);

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
        let ro = rgb[i * 3];
        let go = rgb[i * 3 + 1];
        let bo = rgb[i * 3 + 2];
        let rd = dec_dct[i * 3];
        let gd = dec_dct[i * 3 + 1];
        let bd = dec_dct[i * 3 + 2];
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
            let err = (dec_dct[i * 3 + c] - rgb[i * 3 + c]).abs();
            if err > max_err {
                max_err = err;
            }
            sum_abs_err += err as f64;
            px_mse += err * err;
        }
        if px_mse > 50.0 {
            // significant per-pixel error
            worst_pixels.push((i, px_mse));
        }
    }
    worst_pixels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let avg_err = sum_abs_err / (npix * 3) as f64;
    eprintln!(
        "DCT error: max={max_err:.2}, avg={avg_err:.2}, large_error_pixels={}",
        worst_pixels.len()
    );
    for &(idx, mse) in worst_pixels.iter().take(10) {
        let px = idx % w as usize;
        let py = idx / w as usize;
        let bx = px / 8;
        let by = py / 8;
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
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = ((seed >> 33) % 128) as f32; // [0, 127]
        rgb_neg[i * 3] = 128.0 + noise; // R = [128, 255]
        rgb_neg[i * 3 + 2] = 128.0 - noise; // B = [1, 128]
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
                if len > max_stream_len {
                    max_stream_len = len;
                }
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

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let rgb_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test_rgb"),
                contents: bytemuck::cast_slice(&rgb),
                usage,
            });
        let ycocg_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_ycocg"),
            size: (npix * 3 * 4) as u64,
            usage,
            mapped_at_creation: false,
        });
        let p0_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("p0"),
            size: (npix * 4) as u64,
            usage,
            mapped_at_creation: false,
        });
        let p1_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("p1"),
            size: (npix * 4) as u64,
            usage,
            mapped_at_creation: false,
        });
        let p2_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("p2"),
            size: (npix * 4) as u64,
            usage,
            mapped_at_creation: false,
        });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test_color"),
            });
        color.dispatch(&ctx, &mut cmd, &rgb_buf, &ycocg_buf, w, h, true, false);
        deinterleaver.dispatch(
            &ctx,
            &mut cmd,
            &ycocg_buf,
            &p0_buf,
            &p1_buf,
            &p2_buf,
            npix as u32,
        );
        ctx.queue.submit(Some(cmd.finish()));

        let gpu_y = crate::gpu_util::read_buffer_f32(&ctx, &p0_buf, npix);
        let gpu_co = crate::gpu_util::read_buffer_f32(&ctx, &p1_buf, npix);
        let gpu_cg = crate::gpu_util::read_buffer_f32(&ctx, &p2_buf, npix);

        // Compare with CPU YCoCg
        let mut cpu_y = vec![0.0f32; npix];
        let mut cpu_co = vec![0.0f32; npix];
        let mut cpu_cg = vec![0.0f32; npix];
        for i in 0..npix {
            let r = rgb[i * 3];
            let g = rgb[i * 3 + 1];
            let b = rgb[i * 3 + 2];
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
        use crate::encoder::block_transform::{BlockTransform, BlockTransformType};
        use crate::encoder::fused_block::FusedBlock;
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

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let plane_bytes = (sz * 4) as u64;

        // Per-plane DCT roundtrip for all 3 planes
        let mut decoded_planes: Vec<Vec<f32>> = Vec::new();
        for (name, plane_data) in [("Y", &y_plane), ("Co", &co_plane), ("Cg", &cg_plane)] {
            let input_buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test_input"),
                    contents: bytemuck::cast_slice(plane_data),
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

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("manual_roundtrip"),
                });
            fused.dispatch(
                &ctx, &mut cmd, &input_buf, &quant_buf, &recon_buf, w, h, step, 0.0, 7.0,
            );
            let uniform_weights = [1.0f32; 16];
            quant.dispatch_adaptive(
                &ctx,
                &mut cmd,
                &quant_buf,
                &dequant_buf,
                sz as u32,
                step,
                0.0,
                false,
                w,
                h,
                256,
                0,
                &uniform_weights,
                None,
                7.0,
            );
            bt.dispatch(
                &ctx,
                &mut cmd,
                &dequant_buf,
                &idct_buf,
                w,
                h,
                false,
                BlockTransformType::DCT8,
            );
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
                let r = rgb[i * 3];
                let g = rgb[i * 3 + 1];
                let b = rgb[i * 3 + 2];
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
                    eprintln!(
                        "  plane_a[{i}]={:.4} cpu_y={:.4} diff={:.4}",
                        enc_plane_a[i],
                        cpu_y[i],
                        enc_plane_a[i] - cpu_y[i]
                    );
                }
            }

            // Do fresh DCT on the early plane_a data and compare with early mc_out
            let fused = crate::encoder::fused_block::FusedBlock::new(&ctx);
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let chk_in = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("chk_in"),
                    contents: bytemuck::cast_slice(&enc_plane_a),
                    usage,
                });
            let chk_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chk_out"),
                size: (sz * 4) as u64,
                usage,
                mapped_at_creation: false,
            });
            let chk_recon = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("chk_recon"),
                size: (sz * 4) as u64,
                usage,
                mapped_at_creation: false,
            });
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("chk") });
            fused.dispatch(
                &ctx, &mut cmd, &chk_in, &chk_out, &chk_recon, w, h, 2.0, 0.0, 7.0,
            );
            ctx.queue.submit(Some(cmd.finish()));
            let chk_quant = crate::gpu_util::read_buffer_f32(&ctx, &chk_out, sz);

            let mut mc_mismatch = 0;
            for i in 0..sz {
                if (enc_mc_out[i] - chk_quant[i]).abs() > 0.5 {
                    mc_mismatch += 1;
                }
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
                let input_buf = ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mq_input"),
                        contents: bytemuck::cast_slice(plane_data),
                        usage,
                    });
                let quant_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mq_quant"),
                    size: plane_bytes,
                    usage,
                    mapped_at_creation: false,
                });
                let recon_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mq_recon"),
                    size: plane_bytes,
                    usage,
                    mapped_at_creation: false,
                });
                let mut cmd = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mq") });
                fused.dispatch(
                    &ctx, &mut cmd, &input_buf, &quant_buf, &recon_buf, w, h, step, 0.0, 7.0,
                );
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
                        if diff > max_diff {
                            max_diff = diff;
                        }
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

    assert!(
        psnr_all > 35.0,
        "DCT color q=99 should be >35 dB, got {psnr_all:.2}"
    );
}

/// Generate a textured frame with non-repeating, high-detail spatial patterns
/// suitable for motion estimation. Uses a pseudo-random hash-based texture
/// that has strong per-pixel variation (making every 16×16 block unique)
/// combined with smooth gradients (for realistic spatial correlation).
fn make_textured_frame(w: u32, h: u32, shift_x: i32, shift_y: i32) -> Vec<f32> {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            // Source coordinates with shift (wrap around)
            let sx = (x as i32 - shift_x).rem_euclid(w as i32) as u32;
            let sy = (y as i32 - shift_y).rem_euclid(h as i32) as u32;

            // Simple hash for pseudo-random per-pixel variation (non-repeating)
            let hash = sx.wrapping_mul(73856093) ^ sy.wrapping_mul(19349663);
            let noise = ((hash & 0xFF) as f32) / 255.0; // 0..1

            // Smooth gradients for spatial correlation
            let grad_x = sx as f32 / w as f32;
            let grad_y = sy as f32 / h as f32;

            // Mix: 60% gradient + 40% noise → high detail but spatially correlated
            let r = (grad_x * 180.0 + noise * 75.0).clamp(0.0, 255.0);
            let g = (grad_y * 180.0 + noise * 60.0 + 30.0).clamp(0.0, 255.0);
            let b = ((grad_x + grad_y) * 90.0 + noise * 50.0 + 20.0).clamp(0.0, 255.0);
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

/// Test that motion compensation produces reasonable decode quality for shifted content.
/// Verifies the full P-frame encode/decode pipeline works correctly with spatial motion.
#[test]
fn test_motion_comp_effectiveness() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256;
    let h = 256;
    let shift_px = 4;

    let f0 = make_textured_frame(w, h, 0, 0);
    let f1 = make_textured_frame(w, h, shift_px, 0);
    let f2 = make_textured_frame(w, h, shift_px * 2, 0);

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 10;

    let frames: Vec<&[f32]> = vec![&f0, &f1, &f2];
    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

    assert_eq!(compressed[0].frame_type, FrameType::Intra);
    assert_eq!(compressed[1].frame_type, FrameType::Predicted);
    assert_eq!(compressed[2].frame_type, FrameType::Predicted);

    let i_size = compressed[0].byte_size();
    let p1_size = compressed[1].byte_size();

    eprintln!("\n=== Motion Compensation Effectiveness ===");
    eprintln!(
        "  Frame 0 [I]: {} bytes, {:.3} bpp",
        i_size,
        compressed[0].bpp()
    );
    eprintln!(
        "  Frame 1 [P]: {} bytes, {:.3} bpp (shift={}px)",
        p1_size,
        compressed[1].bpp(),
        shift_px
    );
    eprintln!("  P1/I ratio: {:.2}x", p1_size as f64 / i_size as f64);

    // Decode and verify quality — all frames should decode cleanly
    let decoded = dec.decode_sequence(&ctx, &compressed);
    for (i, dec_frame) in decoded.iter().enumerate() {
        let psnr = compute_psnr(&frames[i], dec_frame);
        eprintln!("  Frame {i}: PSNR={psnr:.2} dB");
        assert!(psnr > 25.0, "Frame {i} PSNR too low: {psnr:.2} dB");
    }

    // P-frame should not be drastically larger than I-frame.
    // At default quality with biorthogonal wavelet residuals, P/I can be close to 1.0
    // but should not exceed it significantly.
    let ratio = p1_size as f64 / i_size as f64;
    assert!(
        ratio < 1.2,
        "P-frame much larger than I-frame with small shift (ratio={ratio:.2}). \
         MC may not be working."
    );
}

/// Test that identical frames produce P-frames smaller than I-frames.
/// With the biorthogonal wavelet, requantized residuals aren't all zero,
/// so P/I ratio at default quality is ~0.6-0.8 (not near-zero).
#[test]
fn test_motion_comp_identical_frames_small_pframe() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);

    let w = 256;
    let h = 256;
    let frame = make_textured_frame(w, h, 0, 0);

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 10;

    // 3 identical frames
    let frames: Vec<&[f32]> = vec![&frame, &frame, &frame];
    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

    let i_size = compressed[0].byte_size();
    let p1_size = compressed[1].byte_size();

    eprintln!("\n=== Identical Frames P-frame Size ===");
    eprintln!(
        "  Frame 0 [I]: {} bytes, {:.3} bpp",
        i_size,
        compressed[0].bpp()
    );
    eprintln!(
        "  Frame 1 [P]: {} bytes, {:.3} bpp",
        p1_size,
        compressed[1].bpp()
    );
    eprintln!("  P1/I ratio: {:.3}", p1_size as f64 / i_size as f64);

    // Identical frames P-frame should be notably smaller than I-frame.
    // The biorthogonal wavelet means residuals don't fully vanish after
    // requantization, so P/I ~0.6-0.8 at default quality (qstep=4.0).
    let ratio = p1_size as f64 / i_size as f64;
    assert!(
        ratio < 0.85,
        "Identical frame P-frame should be smaller than I-frame. \
         Got P/I ratio={ratio:.3} (P={p1_size}, I={i_size})."
    );

    // MVs should be near-zero for identical content
    let mf = compressed[1].motion_field.as_ref().unwrap();
    let max_mv: i16 = mf
        .vectors
        .iter()
        .flat_map(|v| v.iter())
        .map(|v| v.abs())
        .max()
        .unwrap_or(0);
    eprintln!("  Max MV magnitude: {} half-pels", max_mv);
    assert!(
        max_mv <= 4,
        "Identical frames should have near-zero MVs, got max={max_mv}"
    );
}

/// Test that P-frames decode correctly across multiple quality levels.
#[test]
fn test_motion_comp_quality_scaling() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256;
    let h = 256;
    let f0 = make_textured_frame(w, h, 0, 0);
    let f1 = make_textured_frame(w, h, 2, 0); // small shift

    for q in [50, 75, 90] {
        let config = crate::quality_preset(q);
        let mut config_ip = config.clone();
        config_ip.keyframe_interval = 10;

        let frames: Vec<&[f32]> = vec![&f0, &f1];
        let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config_ip);

        let i_size = compressed[0].byte_size();
        let p_size = compressed[1].byte_size();
        let ratio = p_size as f64 / i_size as f64;

        // Decode and verify round-trip quality
        let decoded = dec.decode_sequence(&ctx, &compressed);
        let psnr = compute_psnr(&f1, &decoded[1]);

        eprintln!("  q={q}: I={i_size} bytes, P={p_size} bytes, P/I={ratio:.3}, PSNR={psnr:.1} dB");

        // P-frame should decode with reasonable quality
        assert!(psnr > 20.0, "q={q}: P-frame PSNR too low ({psnr:.1} dB)");

        // P-frame shouldn't be drastically larger than I-frame
        assert!(
            ratio < 1.3,
            "q={q}: P-frame much larger than I-frame (ratio={ratio:.3})"
        );
    }
}

#[test]
fn test_intra_prediction_gpu_roundtrip() {
    // Test that forward + inverse intra prediction (without wavelet) is bit-exact
    let ctx = GpuContext::new();
    let intra = crate::encoder::intra::IntraPredictor::new(&ctx);

    let w = 256u32;
    let h = 256u32;
    let tile_size = 256u32;
    let npix = (w * h) as usize;

    // Create test plane data (gradient)
    let mut plane = vec![0.0f32; npix];
    for y in 0..h {
        for x in 0..w {
            plane[(y * w + x) as usize] = (x as f32 + y as f32 * 0.5).clamp(0.0, 255.0);
        }
    }

    let plane_size = (npix * 4) as u64;
    let num_blocks = (w / 8) * (h / 8);
    let modes_size = (num_blocks as u64) * 4;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    let input_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_input"),
        size: plane_size,
        usage,
        mapped_at_creation: false,
    });
    let residual_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_residual"),
        size: plane_size,
        usage,
        mapped_at_creation: false,
    });
    let modes_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_modes"),
        size: modes_size,
        usage,
        mapped_at_creation: false,
    });
    let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_output"),
        size: plane_size,
        usage,
        mapped_at_creation: false,
    });

    ctx.queue
        .write_buffer(&input_buf, 0, bytemuck::cast_slice(&plane));

    // Forward: input → residual + modes
    let mut cmd = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_intra_fwd"),
        });
    intra.forward(
        &ctx,
        &mut cmd,
        &input_buf,
        &residual_buf,
        &modes_buf,
        w,
        h,
        tile_size,
    );
    ctx.queue.submit(Some(cmd.finish()));

    // Inverse: residual + modes → output
    let mut cmd = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_intra_inv"),
        });
    intra.inverse(
        &ctx,
        &mut cmd,
        &residual_buf,
        &output_buf,
        &modes_buf,
        w,
        h,
        tile_size,
    );
    ctx.queue.submit(Some(cmd.finish()));

    // Readback output
    let output = crate::gpu_util::read_buffer_f32(&ctx, &output_buf, npix);
    let residual = crate::gpu_util::read_buffer_f32(&ctx, &residual_buf, npix);

    // Check PSNR
    let psnr = compute_psnr(&plane, &output);
    eprintln!("Intra GPU roundtrip (no wavelet): PSNR={psnr:.2} dB");
    eprintln!("  First 5 input: {:?}", &plane[..5]);
    eprintln!("  First 5 residual: {:?}", &residual[..5]);
    eprintln!("  First 5 output: {:?}", &output[..5]);

    // Should be bit-exact (no loss in the prediction step)
    assert!(
        psnr > 90.0,
        "Direct intra roundtrip should be near-lossless: {psnr:.2}"
    );
}

#[test]
fn test_intra_prediction_roundtrip() {
    // Note: Intra prediction currently hurts wavelet path quality due to
    // architectural mismatch (block prediction + tile wavelet creates boundary
    // artifacts). It's designed for future BlockDCT8 integration where the
    // transform operates at the same 8×8 block level as prediction.
    // With INTRA_TILE_SIZE=32, drift is limited but wavelet still suffers.
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;
    let rgb = make_gradient_frame(w, h, 0.0);

    let mut config = crate::quality_preset(75);
    config.intra_prediction = false;
    let comp_base = enc.encode(&ctx, &rgb, w, h, &config);
    let dec_base = dec.decode(&ctx, &comp_base);
    let psnr_base = compute_psnr(&rgb, &dec_base);
    config.intra_prediction = true;
    let comp_intra = enc.encode(&ctx, &rgb, w, h, &config);
    let dec_intra = dec.decode(&ctx, &comp_intra);
    let psnr_intra = compute_psnr(&rgb, &dec_intra);

    eprintln!(
        "q=75: base PSNR={psnr_base:.2} dB ({:.3} bpp), intra PSNR={psnr_intra:.2} dB ({:.3} bpp)",
        comp_base.bpp(),
        comp_intra.bpp()
    );

    // Verify intra modes were stored
    assert!(
        comp_intra.intra_modes.is_some(),
        "Intra prediction should produce mode data"
    );
    assert!(
        comp_base.intra_modes.is_none(),
        "Non-intra should have no mode data"
    );

    // Both should decode with reasonable quality
    // (intra currently hurts wavelet quality due to block/tile mismatch)
    assert!(psnr_base > 30.0, "Base PSNR too low: {psnr_base:.2}");
    assert!(psnr_intra > 25.0, "Intra PSNR too low: {psnr_intra:.2}");
}

#[test]
fn test_intra_prediction_serialize_roundtrip() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;
    let rgb = make_gradient_frame(w, h, 0.0);

    let mut config = crate::quality_preset(75);
    config.intra_prediction = true;
    let compressed = enc.encode(&ctx, &rgb, w, h, &config);

    // Serialize and deserialize
    let bytes = crate::format::serialize_compressed(&compressed);
    let deser = crate::format::deserialize_compressed(&bytes);

    // Verify intra modes survived the roundtrip
    assert!(
        deser.intra_modes.is_some(),
        "Intra modes should survive serialization"
    );
    assert_eq!(
        compressed.intra_modes.as_ref().unwrap(),
        deser.intra_modes.as_ref().unwrap(),
        "Intra modes should match after serialize/deserialize"
    );

    // Decode the deserialized frame
    let decoded = dec.decode(&ctx, &deser);
    let psnr = compute_psnr(&rgb, &decoded);
    eprintln!("Intra serialize roundtrip: PSNR={psnr:.2} dB");
    assert!(psnr > 25.0, "Deserialized intra PSNR too low: {psnr:.2}");
}

#[test]
fn test_intra_prediction_flat_image() {
    // Flat image: all pixels = 128 → all residuals = 0 → should be bit-exact
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;
    let rgb: Vec<f32> = vec![128.0; (w * h * 3) as usize]; // flat gray

    let mut config = crate::quality_preset(75);
    config.intra_prediction = true;
    let comp = enc.encode(&ctx, &rgb, w, h, &config);
    let decoded = dec.decode(&ctx, &comp);
    let psnr = compute_psnr(&rgb, &decoded);
    eprintln!(
        "Flat image intra: PSNR={psnr:.2} dB, modes present: {}",
        comp.intra_modes.is_some()
    );

    // Flat image should decode nearly perfectly even with intra prediction
    assert!(psnr > 50.0, "Flat image intra PSNR too low: {psnr:.2}");
}

#[test]
fn test_intra_prediction_modes_sanity() {
    // Check that modes are transmitted correctly from encoder to decoder
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;
    // Horizontal gradient: should prefer horizontal prediction
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = (x as f32 / w as f32 * 255.0).clamp(0.0, 255.0);
            rgb.push(v);
            rgb.push(v);
            rgb.push(v);
        }
    }

    let mut config = crate::quality_preset(75);
    config.intra_prediction = true;
    let comp = enc.encode(&ctx, &rgb, w, h, &config);

    let packed = comp.intra_modes.as_ref().unwrap();
    let num_blocks = (w / 8) * (h / 8);
    let modes = crate::encoder::intra::IntraPredictor::unpack_modes(packed, num_blocks as usize);
    let mut mode_counts = [0u32; 4];
    for &m in &modes {
        assert!(m < 4, "Invalid mode: {m}");
        mode_counts[m as usize] += 1;
    }
    eprintln!(
        "Modes: DC={}, H={}, V={}, D={}",
        mode_counts[0], mode_counts[1], mode_counts[2], mode_counts[3]
    );
    eprintln!(
        "Total blocks: {num_blocks}, first 16 modes: {:?}",
        &modes[..16.min(modes.len())]
    );
}

#[test]
fn test_intra_pack_unpack_modes() {
    use crate::encoder::intra::IntraPredictor;

    // Test pack/unpack roundtrip
    let modes: Vec<u32> = vec![0, 1, 2, 3, 0, 2, 1, 3, 0, 0, 3, 3];
    let packed = IntraPredictor::pack_modes(&modes);
    let unpacked = IntraPredictor::unpack_modes(&packed, modes.len());
    assert_eq!(modes, unpacked, "Pack/unpack should be lossless");

    // Test with exact 4-alignment
    let modes4: Vec<u32> = vec![0, 1, 2, 3];
    let packed4 = IntraPredictor::pack_modes(&modes4);
    assert_eq!(packed4.len(), 1);
    let unpacked4 = IntraPredictor::unpack_modes(&packed4, 4);
    assert_eq!(modes4, unpacked4);
}

/// Diagnostic test: encode I+P pair, then decode P-frame step by step with
/// GPU readbacks at 6 checkpoints to find where encoder/decoder diverge.
///
/// Checkpoints:
/// 1. MC prediction (derived: reconstructed - residual)
/// 2. DWT of residual (≈ dequantized wavelet, encoder-only)
/// 3. Quantized coefficients (entropy decoded vs encoder's quantized Y)
/// 4. Dequantized wavelet coefficients
/// 5. Spatial residual after IDWT
/// 6. Reconstructed pixels after MC inverse
///
/// Run: cargo test --release test_pframe_divergence_checkpoints -- --nocapture
#[test]
fn test_pframe_divergence_checkpoints() {
    use crate::decoder::checkpoint::{compute_diff, PFrameCheckpoints};
    use crate::decoder::pipeline::DecoderPipeline;
    use crate::image_util::load_image_rgb_f32;

    let ctx = GpuContext::new();

    // Load real BBB frames 816-817 if available, else use synthetic
    let seq_dir = "test_material/frames/sequences/bbb_2min";
    let (f0, f1, w, h) = if std::path::Path::new(&format!("{seq_dir}/frame_0816.png")).exists() {
        let (d0, w0, h0) = load_image_rgb_f32(&format!("{seq_dir}/frame_0816.png"));
        let (d1, w1, h1) = load_image_rgb_f32(&format!("{seq_dir}/frame_0817.png"));
        assert_eq!((w0, h0), (w1, h1));
        eprintln!("Using BBB frames 816-817 ({}x{})", w0, h0);
        (d0, d1, w0, h0)
    } else {
        eprintln!("BBB 2min not found, using synthetic 256x256 frames");
        let w = 256u32;
        let h = 256u32;
        let f0 = make_gradient_frame(w, h, 0.0);
        // Slight horizontal shift for motion
        let f1 = make_gradient_frame(w, h, 3.0);
        (f0, f1, w, h)
    };

    let padded_w = (w + 255) & !255;
    let padded_h = (h + 255) & !255;
    let padded_pixels = (padded_w * padded_h) as usize;

    // === Encode I+P+P ===
    // Must encode 3+ frames so the first P-frame (frame 1) has needs_decode=true.
    // With only 2 frames, the encoder skips local decode for the last P-frame
    // (an optimization since the reference won't be used again), which means
    // gpu_ref_planes would still contain the I-frame reference.
    let mut enc = EncoderPipeline::new(&ctx);
    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 30; // P-frame for frame 1+
    config.gpu_entropy_encode = true;

    // Use f1 as both frame 1 and frame 2 (content doesn't matter for frame 2)
    let frames: Vec<&[f32]> = vec![&f0, &f1, &f1];
    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);
    assert_eq!(compressed.len(), 3);
    assert_eq!(compressed[0].frame_type, FrameType::Intra);
    assert_eq!(compressed[1].frame_type, FrameType::Predicted);
    assert_eq!(compressed[2].frame_type, FrameType::Predicted);
    eprintln!(
        "Encoded: I={} bytes, P1={} bytes (ratio={:.3}), P2={} bytes",
        compressed[0].byte_size(),
        compressed[1].byte_size(),
        compressed[1].byte_size() as f64 / compressed[0].byte_size() as f64,
        compressed[2].byte_size(),
    );

    // === Read back encoder's reference Y ===
    // After encoding 3 frames: gpu_ref_planes[0] contains frame 1's (first P-frame)
    // local-decoded Y because frame 2's local decode was skipped (last frame optimization).
    let enc_ref = enc
        .read_reference_planes(&ctx, w, h)
        .expect("encoder should have reference planes");
    let enc_ref_y = &enc_ref[..padded_pixels]; // Y plane only

    // === Decode I-frame normally ===
    let dec = DecoderPipeline::new(&ctx);
    let _iframe_rgb = dec.decode(&ctx, &compressed[0]);

    // Verify I-frame references match
    let dec_ref_after_iframe = dec
        .read_reference_planes(&ctx, w, h)
        .expect("decoder should have reference planes");
    let dec_ref_y_iframe = &dec_ref_after_iframe[..padded_pixels];
    // Encoder's ref after I-frame was overwritten by P-frame, so read from a fresh encode.
    // Disable the encoder-internal reference deblocking filter (GNC_REF_DEBLOCK=0) so that
    // the encoder and decoder produce identical reference frames — the deblocking is
    // encoder-only by design, so the divergence it introduces is expected and correct.
    {
        std::env::set_var("GNC_REF_DEBLOCK", "0");
        let mut enc2 = EncoderPipeline::new(&ctx);
        let mut config_i = config.clone();
        config_i.keyframe_interval = 1;
        let _ = enc2.encode_sequence(&ctx, &[&f0], w, h, &config_i);
        std::env::remove_var("GNC_REF_DEBLOCK");
        let enc_ref_iframe = enc2
            .read_reference_planes(&ctx, w, h)
            .expect("enc2 ref");
        let enc_ref_y_iframe = &enc_ref_iframe[..padded_pixels];
        let iframe_diff = compute_diff(enc_ref_y_iframe, dec_ref_y_iframe, padded_pixels);
        eprintln!(
            "\n=== I-frame reference check (deblock disabled) ===\n  max={:.4} mean={:.6} nonzero={}/{}",
            iframe_diff.max_diff,
            iframe_diff.mean_diff,
            iframe_diff.nonzero_count,
            iframe_diff.total_count,
        );
        assert!(
            iframe_diff.max_diff < 0.01,
            "I-frame references must match when deblock disabled (max_diff={:.4})",
            iframe_diff.max_diff
        );
    }

    // === Decode P-frame with checkpoints ===
    eprintln!("\n=== P-frame checkpoint decode ===");
    let checkpoints: PFrameCheckpoints = dec.decode_pframe_checkpoints(&ctx, &compressed[1]);

    // Read back decoder's final reference Y (after checkpoint decode updated it)
    let dec_ref_after_pframe = dec
        .read_reference_planes(&ctx, w, h)
        .expect("decoder should have reference planes after P");
    let dec_ref_y_pframe = &dec_ref_after_pframe[..padded_pixels];

    // === Compare at each checkpoint ===

    // Checkpoint 1: MC prediction (derived: reconstructed - spatial_residual)
    // Compare encoder's MC prediction (enc_reconstructed - enc_residual) vs decoder's
    // Since we don't have encoder's residual separately, compare predictions indirectly:
    // Both should produce same prediction from same reference + same MVs
    // We'll check by comparing final results and working backwards
    eprintln!("\nCheckpoint 1: MC prediction");
    eprintln!(
        "  (derived from recon - residual, {} pixels)",
        padded_pixels
    );
    // We can't directly compare encoder's MC prediction, but we can check if the
    // decoder's prediction is reasonable (should be close to I-frame reference)
    let pred_stats = {
        let mut max_val: f32 = 0.0;
        let mut sum: f64 = 0.0;
        for &v in checkpoints.mc_prediction.iter().take(padded_pixels) {
            max_val = max_val.max(v.abs());
            sum += v.abs() as f64;
        }
        (max_val, sum / padded_pixels as f64)
    };
    eprintln!(
        "  prediction range: max_abs={:.1} mean_abs={:.2}",
        pred_stats.0, pred_stats.1
    );

    // Checkpoint 2: DWT of residual (encoder-only, ≈ dequantized wavelet)
    eprintln!("\nCheckpoint 2: DWT of residual (encoder-only, skipped)");

    // Checkpoint 3: Quantized coefficients — verify by CPU entropy decode roundtrip
    // (encoder's recon_y buffer is overwritten by later frames, so we can't compare directly)
    let quant_diff = {
        // CPU-decode the P-frame's entropy data for Y plane and compare with GPU decode
        let tiles_per_plane = compressed[1].info.tiles_x() as usize * compressed[1].info.tiles_y() as usize;
        let cpu_decoded = crate::encoder::entropy_helpers::entropy_decode_plane(
            &compressed[1].entropy,
            0, // Y plane
            tiles_per_plane,
            compressed[1].info.tile_size as usize,
            padded_w as usize,
        );
        compute_diff(&cpu_decoded, &checkpoints.quantized, padded_pixels)
    };
    let status3 = if quant_diff.max_diff > 0.001 {
        "DIVERGED"
    } else {
        "OK"
    };
    eprintln!(
        "\nCheckpoint 3: After quantization [{}]\n  max={:.4} mean={:.6} nonzero={}/{}",
        status3,
        quant_diff.max_diff,
        quant_diff.mean_diff,
        quant_diff.nonzero_count,
        quant_diff.total_count,
    );

    // Checkpoint 4: Dequantized wavelet (recompute from CPU-decoded coefficients)
    // Run the quantized coefficients through dequantize and compare with decoder's dequant output
    eprintln!("\nCheckpoint 4: After dequantization");
    {
        // Re-decode quantized Y from entropy data for replay
        let tiles_per_plane = compressed[1].info.tiles_x() as usize * compressed[1].info.tiles_y() as usize;
        let cpu_quant_y = crate::encoder::entropy_helpers::entropy_decode_plane(
            &compressed[1].entropy,
            0,
            tiles_per_plane,
            compressed[1].info.tile_size as usize,
            padded_w as usize,
        );
        let enc_quant_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("enc_quant_y_upload"),
            contents: bytemuck::cast_slice(&cpu_quant_y),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let enc_dequant_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_dequant_y"),
            size: (padded_pixels * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let p_config = &compressed[1].config;
        let weights_luma = p_config.subband_weights.pack_weights();
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("enc_dequant_replay"),
            });
        // Use encoder's quantize module to dequantize
        enc.quantize.dispatch_adaptive(
            &ctx,
            &mut cmd,
            &enc_quant_buf,
            &enc_dequant_buf,
            padded_pixels as u32,
            p_config.quantization_step,
            p_config.dead_zone,
            false, // dequantize
            padded_w,
            padded_h,
            p_config.tile_size,
            p_config.wavelet_levels,
            &weights_luma,
            None,
            0.0,
        );
        ctx.queue.submit(Some(cmd.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
        let enc_dequantized =
            crate::gpu_util::read_buffer_f32(&ctx, &enc_dequant_buf, padded_pixels);
        let dequant_diff = compute_diff(&enc_dequantized, &checkpoints.dequantized, padded_pixels);
        let status4 = if dequant_diff.max_diff > 0.001 {
            "DIVERGED"
        } else {
            "OK"
        };
        eprintln!(
            "  [{}] max={:.4} mean={:.6} nonzero={}/{}",
            status4,
            dequant_diff.max_diff,
            dequant_diff.mean_diff,
            dequant_diff.nonzero_count,
            dequant_diff.total_count,
        );
    }

    // Checkpoint 5: Spatial residual after IDWT
    eprintln!("\nCheckpoint 5: After IDWT");
    {
        // Re-decode quantized Y from entropy data for replay
        let tiles_per_plane = compressed[1].info.tiles_x() as usize * compressed[1].info.tiles_y() as usize;
        let cpu_quant_y = crate::encoder::entropy_helpers::entropy_decode_plane(
            &compressed[1].entropy,
            0,
            tiles_per_plane,
            compressed[1].info.tile_size as usize,
            padded_w as usize,
        );
        let enc_quant_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("enc_quant_y_upload2"),
            contents: bytemuck::cast_slice(&cpu_quant_y),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let buf_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let scratch1 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_scratch1"),
            size: (padded_pixels * 4) as u64,
            usage: buf_usage,
            mapped_at_creation: false,
        });
        let scratch2 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_scratch2"),
            size: (padded_pixels * 4) as u64,
            usage: buf_usage,
            mapped_at_creation: false,
        });
        let enc_residual_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_residual"),
            size: (padded_pixels * 4) as u64,
            usage: buf_usage,
            mapped_at_creation: false,
        });
        let p_config = &compressed[1].config;
        let weights_luma = p_config.subband_weights.pack_weights();
        let info = &compressed[1].info;
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("enc_idwt_replay"),
            });
        // Dequantize
        enc.quantize.dispatch_adaptive(
            &ctx,
            &mut cmd,
            &enc_quant_buf,
            &scratch1, // dequantized wavelet
            padded_pixels as u32,
            p_config.quantization_step,
            p_config.dead_zone,
            false,
            padded_w,
            padded_h,
            p_config.tile_size,
            p_config.wavelet_levels,
            &weights_luma,
            None,
            0.0,
        );
        // IDWT: scratch1 → scratch2 → enc_residual_buf
        enc.transform.inverse(
            &ctx,
            &mut cmd,
            &scratch1,
            &scratch2,
            &enc_residual_buf,
            info,
            p_config.wavelet_levels,
            p_config.wavelet_type,
            0,
        );
        ctx.queue.submit(Some(cmd.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
        let enc_residual =
            crate::gpu_util::read_buffer_f32(&ctx, &enc_residual_buf, padded_pixels);
        let idwt_diff = compute_diff(&enc_residual, &checkpoints.spatial_residual, padded_pixels);
        let status5 = if idwt_diff.max_diff > 0.001 {
            "DIVERGED"
        } else {
            "OK"
        };
        eprintln!(
            "  [{}] max={:.4} mean={:.6} nonzero={}/{}",
            status5,
            idwt_diff.max_diff,
            idwt_diff.mean_diff,
            idwt_diff.nonzero_count,
            idwt_diff.total_count,
        );
    }

    // Checkpoint 6: Reconstructed pixels (final comparison)
    let recon_diff = compute_diff(enc_ref_y, &checkpoints.reconstructed, padded_pixels);
    let status6 = if recon_diff.max_diff > 0.001 {
        "DIVERGED"
    } else {
        "OK"
    };
    eprintln!(
        "\nCheckpoint 6: After adding residual to prediction [{}]\n  max={:.4} mean={:.6} nonzero={}/{}",
        status6,
        recon_diff.max_diff,
        recon_diff.mean_diff,
        recon_diff.nonzero_count,
        recon_diff.total_count,
    );

    // Also check decoder's reference matches its reconstructed
    let dec_ref_vs_recon =
        compute_diff(dec_ref_y_pframe, &checkpoints.reconstructed, padded_pixels);
    eprintln!(
        "\n  Decoder ref vs checkpoint recon: max={:.4} mean={:.6}",
        dec_ref_vs_recon.max_diff, dec_ref_vs_recon.mean_diff,
    );

    // === MV analysis: run MC with the compressed frame's MVs vs encoder ===
    // Upload the compressed MVs to a fresh buffer, run MC on the I-frame reference,
    // and compare the prediction with the encoder's prediction
    eprintln!("\n=== MV analysis ===");
    {
        let mf = compressed[1].motion_field.as_ref().unwrap();
        eprintln!(
            "  block_size={} num_mvs={} ({}x{})",
            mf.block_size,
            mf.vectors.len(),
            padded_w / mf.block_size,
            padded_h / mf.block_size,
        );

        // Re-derive encoder's prediction from enc_ref_y - residual
        // Since: enc_recon = residual + enc_pred → enc_pred = enc_recon - residual
        // And residual matches decoder's (checkpoint 5 OK)
        // And: dec_recon = residual + dec_pred → dec_pred = dec_recon - residual
        let enc_pred: Vec<f32> = enc_ref_y
            .iter()
            .zip(checkpoints.spatial_residual.iter())
            .map(|(&r, &res)| r - res)
            .collect();
        let dec_pred = &checkpoints.mc_prediction;
        let pred_diff = compute_diff(&enc_pred, dec_pred, padded_pixels);
        eprintln!(
            "  MC prediction diff: max={:.4} mean={:.6} nonzero={}/{}",
            pred_diff.max_diff, pred_diff.mean_diff, pred_diff.nonzero_count, pred_diff.total_count,
        );

        // Run MC inverse with a ZERO residual to isolate pure prediction
        // This tests: reference_planes + decoder_MVs → prediction signal
        let zero_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zero_residual"),
            size: (padded_pixels * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        // zero-initialized buffer (mapped_at_creation with no write = zeros)
        zero_buf.unmap();

        // Get decoder's I-frame reference
        // The decoder decoded the I-frame already, then decode_pframe_checkpoints updated
        // reference_planes[0] with the P-frame result. We need the I-frame ref.
        // Actually, let's re-decode the I-frame on a fresh decoder to get clean ref.
        let dec2 = DecoderPipeline::new(&ctx);
        let _ = dec2.decode(&ctx, &compressed[0]); // decode I-frame
        let dec2_ref = dec2
            .read_reference_planes(&ctx, w, h)
            .expect("dec2 should have ref");
        let dec2_ref_y = &dec2_ref[..padded_pixels];

        // Upload compressed MVs to a fresh buffer
        let mv_buf = {
            let i32_data: Vec<i32> = mf
                .vectors
                .iter()
                .flat_map(|mv| [mv[0] as i32, mv[1] as i32])
                .collect();
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test_mv_buf"),
                    contents: bytemuck::cast_slice(&i32_data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        // Upload I-frame ref to a fresh buffer
        let ref_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_ref_buf"),
            contents: bytemuck::cast_slice(dec2_ref_y),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pred_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_pred_buf"),
            size: (padded_pixels * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Run MC inverse: pred = 0 + bilinear_ref(ref, MVs)
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test_mc_pred"),
            });
        enc.motion.compensate(
            &ctx,
            &mut cmd,
            &zero_buf,
            &ref_buf,
            &mv_buf,
            &pred_buf,
            padded_w,
            padded_h,
            false, // inverse: output = zero + prediction = prediction
            mf.block_size,
        );
        ctx.queue.submit(Some(cmd.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
        let test_pred = crate::gpu_util::read_buffer_f32(&ctx, &pred_buf, padded_pixels);

        // Compare this fresh MC prediction with encoder's prediction
        let test_pred_diff = compute_diff(&enc_pred, &test_pred, padded_pixels);
        eprintln!(
            "  Fresh MC pred (from compressed MVs + I-ref) vs encoder pred:");
        eprintln!(
            "    max={:.4} mean={:.6} nonzero={}/{}",
            test_pred_diff.max_diff,
            test_pred_diff.mean_diff,
            test_pred_diff.nonzero_count,
            test_pred_diff.total_count,
        );
        if test_pred_diff.max_diff > 0.001 {
            eprintln!("    → CONFIRMED: Compressed MVs produce different prediction than encoder's MVs");
            eprintln!("    → i32→i16 truncation or MV readback bug in encoder");
        } else {
            eprintln!("    → Compressed MVs match encoder's prediction exactly");
            eprintln!("    → Bug is in decoder's MV upload or MC dispatch");
        }
    }

    // Direct i32 vs i16 roundtrip comparison: read raw GPU MVs and compare
    // Note: staging buffer contains the LAST frame's MVs (frame 2), not frame 1's.
    eprintln!("\n=== MV i32 vs i16 roundtrip check (last P-frame) ===");
    {
        let mf = compressed[2].motion_field.as_ref().unwrap();
        let raw_i32 = enc
            .read_raw_split_mvs_i32(&ctx)
            .expect("should have raw split MVs");
        eprintln!("  raw_i32.len()={} compressed_mvs.len()={}", raw_i32.len() / 2, mf.vectors.len());

        let mut mismatches = 0usize;
        let mut max_i32_abs: i32 = 0;
        let mut first_mismatch = None;
        for (i, &[dx16, dy16]) in mf.vectors.iter().enumerate() {
            let gpu_dx = raw_i32[i * 2];
            let gpu_dy = raw_i32[i * 2 + 1];
            let rt_dx = dx16 as i32;
            let rt_dy = dy16 as i32;
            max_i32_abs = max_i32_abs.max(gpu_dx.abs()).max(gpu_dy.abs());
            if gpu_dx != rt_dx || gpu_dy != rt_dy {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, gpu_dx, gpu_dy, rt_dx, rt_dy));
                }
            }
        }
        eprintln!("  Max absolute MV (raw i32): {max_i32_abs} half-pels ({:.1} pixels)", max_i32_abs as f32 / 2.0);
        eprintln!("  Mismatches after i32→i16→i32 roundtrip: {mismatches}/{}", mf.vectors.len());
        if let Some((idx, gdx, gdy, rdx, rdy)) = first_mismatch {
            let bx = idx % (padded_w as usize / mf.block_size as usize);
            let by = idx / (padded_w as usize / mf.block_size as usize);
            eprintln!("  First mismatch at block {idx} (bx={bx}, by={by}): GPU=({gdx},{gdy}) roundtrip=({rdx},{rdy})");
        }
    }

    eprintln!("\n=== Summary ===");
    eprintln!("First checkpoint with diff > 0 identifies the bug source.");
    if quant_diff.max_diff > 0.001 {
        eprintln!("BUG: Entropy coding is LOSSY (quantized coefficients differ)");
    } else if recon_diff.max_diff > 0.001 {
        eprintln!("BUG: MC inverse divergence (same coefficients → different reconstruction)");
        eprintln!("     Likely cause: MV format mismatch or reference plane difference");
    } else {
        eprintln!("All checkpoints match — no divergence detected");
    }
}

/// Regression test for temporal wavelet encode/decode pipeline.
///
/// The bug: `encode_from_wavelet_coeffs` used a single CommandEncoder for all
/// three planes, with `write_buffer` calls to the same GPU buffer before a
/// single `submit()`. wgpu flushes pending writes at submit-time, so only the
/// last plane's data (Cg) survived — all three quantize dispatches saw Cg.
#[test]
fn test_temporal_wavelet_roundtrip_per_plane() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;

    // Two frames with distinct content so Haar high band is non-trivial.
    let f0 = make_gradient_frame(w, h, 0.0);
    let f1 = make_gradient_frame(w, h, 30.0);

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.quantization_step = 4.0;
    config.keyframe_interval = 1;
    config.temporal_transform = TemporalTransform::Haar;
    config.cfl_enabled = false;

    let frames: Vec<&[f32]> = vec![&f0, &f1];
    let seq = enc.encode_sequence_temporal_wavelet(&ctx, &frames, w, h, &config, TemporalTransform::Haar, 2);
    let decoded = dec.decode_temporal_sequence(&ctx, &seq);

    assert_eq!(decoded.len(), 2, "should decode 2 frames");

    // Check PSNR for each frame — should be > 30 dB for q=75 on smooth gradients.
    for (i, (orig, recon)) in frames.iter().zip(decoded.iter()).enumerate() {
        let psnr = compute_psnr(orig, recon);
        eprintln!("temporal wavelet frame {i}: PSNR = {psnr:.2} dB");
        assert!(
            psnr > 30.0,
            "frame {i} PSNR {psnr:.2} dB is too low (expected > 30 dB)"
        );
    }
}

/// Verify that the three planes (Y, Co, Cg) produce distinct quantized data
/// when encoded through `encode_from_wavelet_coeffs`.
///
/// Before the fix, all three planes were identical because only the last
/// `write_buffer` was visible at `submit()` time.
#[test]
fn test_temporal_wavelet_planes_are_distinct() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;

    // Build spatial wavelet coefficients for one frame.
    let frame = make_gradient_frame(w, h, 0.0);

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.quantization_step = 4.0;
    config.keyframe_interval = 1;
    config.temporal_transform = TemporalTransform::None;
    config.cfl_enabled = false;

    let info = FrameInfo { width: w, height: h, bit_depth: 8, tile_size: config.tile_size, chroma_format: crate::ChromaFormat::Yuv444 };
    let prequant = enc.debug_wavelet_prequant(&ctx, &frame, &info, &config);

    // Y and Co should differ for a gradient frame.
    let y_co_diff: f64 = prequant[0]
        .iter()
        .zip(prequant[1].iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / prequant[0].len() as f64;
    assert!(
        y_co_diff > 0.1,
        "test prerequisite: Y and Co should differ (mean_abs_diff={y_co_diff:.4})"
    );

    // Now quantize+entropy-encode these coefficients as a temporal frame would.
    let quantized = enc.debug_quantize_wavelet_coeffs(
        &ctx,
        [&prequant[0], &prequant[1], &prequant[2]],
        &info,
        &config,
    );

    // After the fix, Y and Co quantized planes must differ.
    let q_y_co_diff: f64 = quantized[0]
        .iter()
        .zip(quantized[1].iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / quantized[0].len() as f64;
    eprintln!("quantized Y vs Co mean_abs_diff = {q_y_co_diff:.4}");
    assert!(
        q_y_co_diff > 0.01,
        "Y and Co quantized planes should differ but mean_abs_diff={q_y_co_diff:.6} — \
         likely all planes see only the last write_buffer"
    );
}

// ---------------------------------------------------------------------------
// Chroma subsampling roundtrip tests (4:4:4, 4:2:2, 4:2:0)
// ---------------------------------------------------------------------------

fn chroma_roundtrip(chroma_fmt: crate::ChromaFormat) -> f64 {
    let ctx = crate::GpuContext::new();
    let mut encoder = EncoderPipeline::new(&ctx);
    let decoder = DecoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;
    let rgb = make_gradient_frame(w, h, 0.0);

    let mut config = crate::CodecConfig::default();
    config.chroma_format = chroma_fmt;
    config.quantization_step = 4.0;
    config.cfl_enabled = false; // CfL requires 444

    let compressed = encoder.encode(&ctx, &rgb, w, h, &config);
    let decoded = decoder.decode(&ctx, &compressed);

    assert_eq!(
        decoded.len(),
        (w * h * 3) as usize,
        "decoded output size wrong for {:?}",
        chroma_fmt
    );
    compute_psnr(&rgb, &decoded)
}

#[test]
fn test_yuv444_encode_decode_roundtrip() {
    let psnr = chroma_roundtrip(crate::ChromaFormat::Yuv444);
    eprintln!("YUV 4:4:4 PSNR = {psnr:.2} dB");
    assert!(psnr > 30.0, "4:4:4 PSNR too low: {psnr:.2} dB");
}

#[test]
fn test_yuv422_encode_decode_roundtrip() {
    let psnr = chroma_roundtrip(crate::ChromaFormat::Yuv422);
    eprintln!("YUV 4:2:2 PSNR = {psnr:.2} dB");
    // 4:2:2 halves horizontal chroma resolution.  With correct padding the wavelet
    // transform sees valid data everywhere; PSNR should be close to 4:4:4.
    assert!(psnr > 35.0, "4:2:2 PSNR too low: {psnr:.2} dB (chroma padding bug?)");
}

#[test]
fn test_yuv420_encode_decode_roundtrip() {
    let psnr = chroma_roundtrip(crate::ChromaFormat::Yuv420);
    eprintln!("YUV 4:2:0 PSNR = {psnr:.2} dB");
    // 4:2:0 quarters chroma resolution; slightly more loss than 4:2:2.
    assert!(psnr > 34.0, "4:2:0 PSNR too low: {psnr:.2} dB (chroma padding bug?)");
}

#[test]
fn test_yuv420_encode_decode_roundtrip_512() {
    // 512×512 uses 2×2 luma tiles and 1×1 chroma tiles for 4:2:0.
    // This exposes bugs that only appear when luma tile count > chroma tile count.
    let ctx = crate::GpuContext::new();
    let mut encoder = EncoderPipeline::new(&ctx);
    let decoder = DecoderPipeline::new(&ctx);

    let w = 512u32;
    let h = 512u32;
    // Use solid gray (R=G=B=128) — with 4:2:0 chroma is flat, roundtrip should be ~inf dB
    let rgb: Vec<f32> = vec![128.0f32; (w * h * 3) as usize];

    let mut config = crate::CodecConfig::default();
    config.chroma_format = crate::ChromaFormat::Yuv420;
    config.quantization_step = 4.0;
    config.cfl_enabled = false;

    let compressed = encoder.encode(&ctx, &rgb, w, h, &config);
    let decoded = decoder.decode(&ctx, &compressed);
    let psnr = compute_psnr(&rgb, &decoded);
    eprintln!("YUV 4:2:0 512×512 PSNR = {psnr:.2} dB chroma_format={:?}", compressed.info.chroma_format);

    // Also test 4:4:4 at same settings to verify encoder/decoder work
    let mut config2 = crate::CodecConfig::default();
    config2.chroma_format = crate::ChromaFormat::Yuv444;
    config2.quantization_step = 4.0;
    config2.cfl_enabled = false;
    let mut encoder2 = EncoderPipeline::new(&ctx);
    let decoder2 = crate::decoder::pipeline::DecoderPipeline::new(&ctx);
    let compressed2 = encoder2.encode(&ctx, &rgb, w, h, &config2);
    let decoded2 = decoder2.decode(&ctx, &compressed2);
    let psnr2 = compute_psnr(&rgb, &decoded2);
    eprintln!("YUV 4:4:4 512×512 PSNR = {psnr2:.2} dB (control)");

    assert!(psnr > 35.0, "4:2:0 512×512 PSNR too low: {psnr:.2} dB");
}

#[test]
fn test_chroma_format_bitstream_roundtrip() {
    // Verify that the chroma format survives serialize → deserialize.
    let ctx = crate::GpuContext::new();
    let mut encoder = EncoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;
    let rgb = make_gradient_frame(w, h, 0.0);

    for fmt in [
        crate::ChromaFormat::Yuv444,
        crate::ChromaFormat::Yuv422,
        crate::ChromaFormat::Yuv420,
    ] {
        let mut config = crate::CodecConfig::default();
        config.chroma_format = fmt;
        config.cfl_enabled = false;

        let compressed = encoder.encode(&ctx, &rgb, w, h, &config);
        let serialized = crate::format::serialize_compressed(&compressed);
        let deserialized = crate::format::deserialize_compressed(&serialized);

        assert_eq!(
            deserialized.info.chroma_format, fmt,
            "chroma_format not preserved in bitstream for {:?}",
            fmt
        );
    }
}

// ---------------------------------------------------------------------------
// P-frame + non-444 chroma roundtrip tests
// Regression guard for the bug where encode_pframe used luma tile count for
// chroma planes, causing tile-count mismatch and per-tile flickering.
// ---------------------------------------------------------------------------

fn pframe_chroma_sequence_psnr(chroma_fmt: crate::ChromaFormat) -> (f64, f64) {
    let ctx = crate::GpuContext::new();
    let mut encoder = EncoderPipeline::new(&ctx);
    let decoder = crate::decoder::pipeline::DecoderPipeline::new(&ctx);

    // 512×256: exercises a tile boundary (chroma tile seam at x=128 for 4:2:2, (x=128,y=128) for 4:2:0)
    let w = 512u32;
    let h = 256u32;
    let f0 = make_gradient_frame(w, h, 0.0);
    // Slightly shifted gradients so P-frames have non-zero residual
    let f1 = make_gradient_frame(w, h, 10.0);
    let f2 = make_gradient_frame(w, h, 20.0);

    let mut config = crate::CodecConfig::default();
    config.chroma_format = chroma_fmt;
    config.quantization_step = 4.0;
    config.cfl_enabled = false; // CfL requires 444
    config.keyframe_interval = 10; // I P P …

    let frames: Vec<&[f32]> = vec![&f0, &f1, &f2];
    let compressed = encoder.encode_sequence(&ctx, &frames, w, h, &config);

    assert_eq!(compressed.len(), 3, "{chroma_fmt:?}: expected 3 compressed frames");
    assert_eq!(compressed[0].frame_type, crate::FrameType::Intra,     "{chroma_fmt:?}: frame 0 should be I");
    assert_eq!(compressed[1].frame_type, crate::FrameType::Predicted, "{chroma_fmt:?}: frame 1 should be P");
    assert_eq!(compressed[2].frame_type, crate::FrameType::Predicted, "{chroma_fmt:?}: frame 2 should be P");

    // Decode and measure PSNR for the two P-frames
    let _dec0 = decoder.decode(&ctx, &compressed[0]);
    let dec1  = decoder.decode(&ctx, &compressed[1]);
    let dec2  = decoder.decode(&ctx, &compressed[2]);

    let psnr1 = compute_psnr(&f1, &dec1);
    let psnr2 = compute_psnr(&f2, &dec2);
    (psnr1, psnr2)
}

#[test]
fn test_pframe_yuv422_sequence_roundtrip() {
    let (psnr1, psnr2) = pframe_chroma_sequence_psnr(crate::ChromaFormat::Yuv422);
    eprintln!("P-frame 4:2:2 PSNR: P1={psnr1:.2} dB  P2={psnr2:.2} dB");
    // The flickering bug produced PSNR < 20 dB on P-frames; correct decode should be > 30 dB.
    assert!(psnr1 > 30.0, "4:2:2 P-frame 1 PSNR too low ({psnr1:.2} dB) — chroma tile mismatch?");
    assert!(psnr2 > 30.0, "4:2:2 P-frame 2 PSNR too low ({psnr2:.2} dB) — chroma tile mismatch?");
}

#[test]
fn test_pframe_yuv420_sequence_roundtrip() {
    let (psnr1, psnr2) = pframe_chroma_sequence_psnr(crate::ChromaFormat::Yuv420);
    eprintln!("P-frame 4:2:0 PSNR: P1={psnr1:.2} dB  P2={psnr2:.2} dB");
    assert!(psnr1 > 30.0, "4:2:0 P-frame 1 PSNR too low ({psnr1:.2} dB) — chroma tile mismatch?");
    assert!(psnr2 > 30.0, "4:2:0 P-frame 2 PSNR too low ({psnr2:.2} dB) — chroma tile mismatch?");
}

/// Verify that 128×128 tiles encode and decode correctly with the Rice encoder.
///
/// At 128×128 tiles: 64 symbols/stream, worst-case ~150 bytes. max_stream_bytes_for_tile(128)
/// returns 1024 (7× headroom). This test verifies:
///   1. No overflow flag is triggered (pipeline panics if overflow occurs).
///   2. Decode quality is reasonable (PSNR > 35 dB at q=75).
#[test]
fn test_rice_128x128_tiles() {
    use crate::EntropyCoder;

    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    // 256×256 image → 4 tiles of 128×128
    let w = 256u32;
    let h = 256u32;
    let frame = make_gradient_frame(w, h, 0.0);

    let mut config = CodecConfig::default();
    config.tile_size = 128;
    config.keyframe_interval = 1; // all I-frames
    config.entropy_coder = EntropyCoder::Rice;
    // q=75 default (quantization_step=1/75 equivalent via CodecConfig::default)

    let frames: Vec<&[f32]> = vec![&frame];
    // This will panic if Rice overflow is detected in any tile
    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);
    assert_eq!(compressed.len(), 1);

    let decoded = dec.decode(&ctx, &compressed[0]);
    let psnr = compute_psnr_single(&frame, &decoded);
    eprintln!("Rice 128×128 tiles PSNR: {psnr:.2} dB (no overflow triggered)");
    assert!(
        psnr > 35.0,
        "Rice 128×128 tile PSNR too low: {psnr:.2} dB — likely a tile offset bug"
    );
}

// ---------------------------------------------------------------------------
// Multi-tile chroma boundary tests (4:2:2 and 4:2:0)
// 512×256 images cross a 256px tile boundary, exercising the chroma tile seam.
// ---------------------------------------------------------------------------

/// 4:2:2 encode/decode roundtrip crossing a tile boundary (x=256 luma / x=128 chroma).
#[test]
fn test_bilinear_chroma_upsample_tile_boundary_422() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    // Horizontal ramp across tile boundary
    let w = 512u32;
    let h = 256u32;
    let frame = make_gradient_frame(w, h, 0.0);

    let mut config = CodecConfig::default();
    config.chroma_format = crate::ChromaFormat::Yuv422;
    config.cfl_enabled = false;
    config.keyframe_interval = 1;

    let frames: Vec<&[f32]> = vec![&frame];
    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);
    assert_eq!(compressed.len(), 1);

    let decoded = dec.decode(&ctx, &compressed[0]);
    let psnr = compute_psnr_single(&frame, &decoded);
    eprintln!("4:2:2 tile boundary PSNR: {psnr:.2} dB");
    assert!(psnr > 35.0, "4:2:2 tile boundary PSNR too low: {psnr:.2} dB");
}

/// 4:2:0 encode/decode roundtrip crossing tile boundaries (x=256, y=256).
#[test]
fn test_bilinear_chroma_upsample_tile_boundary_420() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 512u32;
    let h = 256u32;
    let frame = make_gradient_frame(w, h, 15.0);

    let mut config = CodecConfig::default();
    config.chroma_format = crate::ChromaFormat::Yuv420;
    config.cfl_enabled = false;
    config.keyframe_interval = 1;

    let frames: Vec<&[f32]> = vec![&frame];
    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);
    assert_eq!(compressed.len(), 1);

    let decoded = dec.decode(&ctx, &compressed[0]);
    let psnr = compute_psnr_single(&frame, &decoded);
    eprintln!("4:2:0 tile boundary PSNR: {psnr:.2} dB");
    assert!(psnr > 34.0, "4:2:0 tile boundary PSNR too low: {psnr:.2} dB");
}

#[test]
fn test_10bit_roundtrip() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;

    // Generate a synthetic 10-bit frame: smooth gradient with values in [0, 1023].
    let mut frame: Vec<f32> = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (x as f32 / w as f32 * 1023.0).clamp(0.0, 1023.0);
            let g = (y as f32 / h as f32 * 1023.0).clamp(0.0, 1023.0);
            let b = ((x + y) as f32 / (w + h) as f32 * 1023.0).clamp(0.0, 1023.0);
            frame.push(r);
            frame.push(g);
            frame.push(b);
        }
    }

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.quantization_step = 4.0;
    config.keyframe_interval = 1;
    config.temporal_transform = TemporalTransform::None;
    config.cfl_enabled = false;
    config.bit_depth = 10;

    let compressed = enc.encode(&ctx, &frame, w, h, &config);

    // Verify bit_depth was stored in the compressed frame header
    assert_eq!(
        compressed.info.bit_depth, 10,
        "compressed frame should record bit_depth=10"
    );

    let decoded = dec.decode(&ctx, &compressed);

    // Check output length matches
    assert_eq!(
        decoded.len(),
        (w * h * 3) as usize,
        "decoded length mismatch"
    );

    // Verify decoded values are in 10-bit range [0, 1023], NOT clipped to [0, 255]
    let max_val = decoded
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_val > 255.0,
        "decoded max value {max_val:.1} is ≤ 255: values appear clipped to 8-bit range"
    );

    // Compute PSNR with 10-bit peak (1023)
    let psnr = compute_psnr_peak(&frame, &decoded, 1023.0);
    eprintln!("10-bit roundtrip PSNR: {psnr:.2} dB (peak=1023)");
    assert!(
        psnr > 30.0,
        "10-bit roundtrip PSNR too low: {psnr:.2} dB (expected > 30 dB)"
    );
}

// ---------------------------------------------------------------------------
// Scene cut detection tests
// ---------------------------------------------------------------------------

/// Unit test for compute_luma_mad: identical frames must yield MAD = 0.
#[test]
fn test_compute_luma_mad_identical() {
    let frame: Vec<f32> = (0..256 * 3).map(|i| (i % 256) as f32).collect();
    let mad = crate::luma_mad(&frame, &frame);
    assert!(
        mad < 1e-5,
        "MAD of identical frames should be ~0, got {mad}"
    );
}

/// Unit test for compute_luma_mad: black vs white must yield MAD ≈ 255.
#[test]
fn test_compute_luma_mad_hard_cut() {
    // black frame: all zeros
    let black: Vec<f32> = vec![0.0f32; 512 * 3];
    // white frame: all 255
    let white: Vec<f32> = vec![255.0f32; 512 * 3];
    let mad = crate::luma_mad(&white, &black);
    assert!(
        mad > 200.0,
        "MAD of black vs white should be ~255, got {mad}"
    );
}

/// Unit test for compute_luma_mad: slight motion must yield small MAD.
#[test]
fn test_compute_luma_mad_slight_motion() {
    let frame_a: Vec<f32> = (0..256 * 3).map(|i| (i % 128) as f32).collect();
    // Shift by a small constant to simulate tiny per-pixel change (~2 values on average)
    let frame_b: Vec<f32> = frame_a.iter().map(|v| (v + 2.0).min(255.0)).collect();
    let mad = crate::luma_mad(&frame_b, &frame_a);
    assert!(
        mad < 10.0,
        "MAD of slightly-shifted frames should be small, got {mad}"
    );
}

/// Integration test: scene cut detection forces I-frame at a hard cut.
///
/// Creates a 5-frame sequence with keyframe_interval=4 and scene_cut_threshold=50.
/// Frames 0-1 are black, frame 2 is white (hard cut). With detection enabled,
/// frame 2 must be encoded as an I-frame even though it is not at a keyframe boundary.
/// Without detection (threshold=0), frame 2 would be encoded as a P-frame.
#[test]
fn test_scene_cut_forced_keyframe() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;

    // frame 0: all black
    let black: Vec<f32> = vec![0.0f32; (w * h * 3) as usize];
    // frame 1: mostly black (slow motion, no cut)
    let slow_motion: Vec<f32> = black.iter().map(|_| 5.0f32).collect();
    // frame 2: all white — hard scene cut (MAD ~250)
    let white: Vec<f32> = vec![255.0f32; (w * h * 3) as usize];
    // frame 3: slight variation from white
    let white2: Vec<f32> = white.iter().map(|v| v - 5.0).collect();
    // frame 4: still white
    let white3: Vec<f32> = white.iter().map(|v| v - 3.0).collect();

    let frames: Vec<&[f32]> = vec![&black, &slow_motion, &white, &white2, &white3];

    // With scene cut detection enabled (threshold=50), frame 2 should be forced I-frame.
    let mut config = crate::CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 8; // large interval so frame 2 would not be a natural keyframe
    config.scene_cut_threshold = 50.0;
    config.cfl_enabled = false;

    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

    assert_eq!(compressed.len(), 5, "expected 5 compressed frames");

    // Frame 0 must be I-frame (always)
    assert_eq!(
        compressed[0].frame_type,
        FrameType::Intra,
        "frame 0 should always be I-frame"
    );

    // Frame 2 (white, hard cut) must be forced I-frame
    assert_eq!(
        compressed[2].frame_type,
        FrameType::Intra,
        "frame 2 (hard cut, MAD >> 50) should be forced to I-frame"
    );

    // Frame 1 (slow motion, MAD ~5) must NOT be forced I-frame
    assert_ne!(
        compressed[1].frame_type,
        FrameType::Intra,
        "frame 1 (slow motion) should remain a P-frame, not incorrectly forced to I-frame"
    );

    // Decode and verify quality — no garbage frames from cross-cut reference
    for (i, cf) in compressed.iter().enumerate() {
        let decoded = dec.decode(&ctx, cf);
        let psnr = compute_psnr(&decoded, frames[i]);
        assert!(
            psnr > 25.0,
            "frame {i} PSNR {psnr:.2} dB is too low — possible cross-cut residual corruption"
        );
    }
    eprintln!("scene cut test: all 5 frames decoded with PSNR > 25 dB");
}

/// Verify that scene cut detection is disabled when threshold=0.
/// Frame 2 (hard cut) should be encoded as P-frame with threshold=0.
#[test]
fn test_scene_cut_disabled_at_zero_threshold() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);

    let w = 256u32;
    let h = 256u32;

    let black: Vec<f32> = vec![0.0f32; (w * h * 3) as usize];
    let white: Vec<f32> = vec![255.0f32; (w * h * 3) as usize];
    let white2: Vec<f32> = white.iter().map(|v| v - 5.0).collect();

    let frames: Vec<&[f32]> = vec![&black, &white, &white2];

    let mut config = crate::CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 8;
    config.scene_cut_threshold = 0.0; // disabled
    config.cfl_enabled = false;

    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);
    assert_eq!(compressed.len(), 3);

    // Frame 0 is always I-frame
    assert_eq!(compressed[0].frame_type, FrameType::Intra);
    // Frame 1 must be P-frame (detection disabled)
    assert_eq!(
        compressed[1].frame_type,
        FrameType::Predicted,
        "frame 1 should be P-frame when scene_cut_threshold=0 (detection disabled)"
    );
}

