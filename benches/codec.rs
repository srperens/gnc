use criterion::{criterion_group, criterion_main, Criterion};
use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::image_util::load_image_rgb_f32;
use gnc::{quality_preset, GpuContext};

fn load_test_image(name: &str) -> (Vec<f32>, u32, u32) {
    let path = format!("test_material/frames/{name}");
    load_image_rgb_f32(&path)
}

fn bench_encode(c: &mut Criterion) {
    let ctx = GpuContext::new();
    let config = quality_preset(75);

    let mut group = c.benchmark_group("encode");
    group.sample_size(10);

    for name in &["bbb_1080p.png", "blue_sky_1080p.png"] {
        let (pixels, w, h) = load_test_image(name);
        let mut enc = EncoderPipeline::new(&ctx);

        group.bench_function(*name, |b| {
            b.iter(|| {
                enc.encode_sequence(&ctx, &[pixels.as_slice()], w, h, &config);
            });
        });
    }

    group.finish();
}

fn bench_decode_u8(c: &mut Criterion) {
    let ctx = GpuContext::new();
    let config = quality_preset(75);

    let mut group = c.benchmark_group("decode_u8");
    group.sample_size(10);

    for name in &["bbb_1080p.png", "blue_sky_1080p.png"] {
        let (pixels, w, h) = load_test_image(name);
        let mut enc = EncoderPipeline::new(&ctx);
        let dec = DecoderPipeline::new(&ctx);
        let compressed = enc.encode_sequence(&ctx, &[pixels.as_slice()], w, h, &config);
        let frame = &compressed[0];

        group.bench_function(*name, |b| {
            b.iter(|| {
                dec.decode_u8(&ctx, frame);
            });
        });
    }

    group.finish();
}

fn bench_roundtrip_quality(c: &mut Criterion) {
    let ctx = GpuContext::new();
    let config = quality_preset(75);

    let mut group = c.benchmark_group("roundtrip");
    group.sample_size(10);

    for name in &["bbb_1080p.png", "blue_sky_1080p.png"] {
        let (pixels, w, h) = load_test_image(name);
        let mut enc = EncoderPipeline::new(&ctx);
        let dec = DecoderPipeline::new(&ctx);

        group.bench_function(*name, |b| {
            b.iter(|| {
                let compressed = enc.encode_sequence(&ctx, &[pixels.as_slice()], w, h, &config);
                let decoded = dec.decode(&ctx, &compressed[0]);

                // Compute PSNR (not timed, but useful for tracking)
                let mse: f64 = pixels
                    .iter()
                    .zip(decoded.iter())
                    .map(|(&a, &b)| {
                        let d = a as f64 - b as f64;
                        d * d
                    })
                    .sum::<f64>()
                    / pixels.len() as f64;
                let psnr = if mse > 0.0 {
                    10.0 * (255.0_f64 * 255.0 / mse).log10()
                } else {
                    f64::INFINITY
                };

                // Compute bpp
                let compressed_bytes = gnc::format::serialize_compressed(&compressed[0]).len();
                let bpp = compressed_bytes as f64 * 8.0 / (w * h) as f64;

                eprintln!("{name}: PSNR={psnr:.2} dB, bpp={bpp:.3}");
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode_u8,
    bench_roundtrip_quality
);
criterion_main!(benches);
