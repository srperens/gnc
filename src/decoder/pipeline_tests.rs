use super::*;
use crate::encoder::pipeline::EncoderPipeline;
use crate::CodecConfig;

fn make_gradient_frame(w: u32, h: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (x as f32 / w as f32 * 255.0).clamp(0.0, 255.0);
            let g = (y as f32 / h as f32 * 255.0).clamp(0.0, 255.0);
            let b = ((x + y) as f32 / (w + h) as f32 * 255.0).clamp(0.0, 255.0);
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

#[test]
fn test_decode_to_texture_dimensions() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 256;
    let h = 256;
    let frame_data = make_gradient_frame(w, h);

    let mut config = CodecConfig::default();
    config.tile_size = 256;
    config.keyframe_interval = 1;

    let compressed = enc.encode_sequence(&ctx, &[frame_data.as_slice()], w, h, &config);
    assert_eq!(compressed.len(), 1);

    let handle = dec.decode_to_texture(&ctx, &compressed[0]);
    assert_eq!(handle.width, w);
    assert_eq!(handle.height, h);

    // Verify the texture view is accessible
    let view = dec.output_texture_view();
    assert!(view.is_some());
}

#[test]
fn test_decode_to_texture_non_square() {
    let ctx = GpuContext::new();
    let mut enc = EncoderPipeline::new(&ctx);
    let dec = DecoderPipeline::new(&ctx);

    let w = 320;
    let h = 192;
    let frame_data = make_gradient_frame(w, h);

    let mut config = CodecConfig::default();
    config.tile_size = 64;
    config.keyframe_interval = 1;

    let compressed = enc.encode_sequence(&ctx, &[frame_data.as_slice()], w, h, &config);
    let handle = dec.decode_to_texture(&ctx, &compressed[0]);
    assert_eq!(handle.width, w);
    assert_eq!(handle.height, h);
}
