use super::*;
use crate::decoder::pipeline::DecoderPipeline;

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
