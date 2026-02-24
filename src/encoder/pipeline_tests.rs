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
    config.keyframe_interval = 4; // I P P P

    let frames: Vec<&[f32]> = vec![&f0, &f1, &f2, &f3];
    let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

    assert_eq!(compressed.len(), 4);
    assert_eq!(compressed[0].frame_type, FrameType::Intra);
    assert_eq!(compressed[1].frame_type, FrameType::Predicted);
    assert_eq!(compressed[2].frame_type, FrameType::Predicted);
    assert_eq!(compressed[3].frame_type, FrameType::Predicted);

    // P-frames must have motion fields
    for i in 1..4 {
        assert!(
            compressed[i].motion_field.is_some(),
            "P-frame {i} should have motion field"
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
    config.keyframe_interval = 3; // I P P I P

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
