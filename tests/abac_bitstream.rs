//! `EntropyCoder::Abac` end to end: encode → GP18 bitstream → GPU decode.
//!
//! The property that makes these tests strong is that entropy coding is **lossless**. Rice and
//! abac code the identical quantised coefficients, so a frame encoded either way must decode to
//! the *same pixels* — not similar ones, the same ones. Anything less means the adaptive coder
//! diverged, and an adaptive coder that diverges does not raise an error: it decodes a plausible
//! wrong image, because each symbol after the divergence is read from a corrupted interval and a
//! corrupted context.
//!
//! Synthesises its own images so these run without test material.

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::{ChromaFormat, EntropyCoder, EntropyData, GpuContext};
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();
fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// Detail at several scales plus noise, so every subband carries something and the deep
/// subbands are not all zero. A flat image would let a broken coder pass.
fn synth_image(w: u32, h: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    let mut rng: u32 = 0x9e37_79b9;
    for y in 0..h {
        for x in 0..w {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let noise = (rng % 24) as f32 - 12.0;
            let fine = if ((x / 2) + (y / 3)) % 2 == 0 { 18.0 } else { 0.0 };
            let coarse = ((x / 40) * 37 % 200) as f32;
            let ramp = y as f32 / h as f32 * 120.0;
            let r = (coarse + ramp + fine + noise).clamp(0.0, 255.0).round();
            let g = (ramp * 1.4 + fine + noise).clamp(0.0, 255.0).round();
            let b = (255.0 - coarse + noise).clamp(0.0, 255.0).round();
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

/// Encode, put the frame through the real container, decode on the GPU. Returns the pixels and
/// the serialised size.
fn roundtrip(
    ctx: &GpuContext,
    img: &[f32],
    w: u32,
    h: u32,
    config: &gnc::CodecConfig,
) -> (Vec<f32>, usize) {
    let mut encoder = EncoderPipeline::new(ctx);
    let decoder = DecoderPipeline::new(ctx);
    let compressed = encoder.encode(ctx, img, w, h, config);
    let bytes = gnc::format::serialize_compressed(&compressed);
    let back = gnc::format::deserialize_compressed(&bytes);
    assert!(
        matches!(back.entropy, EntropyData::Abac(_)) == matches!(compressed.entropy, EntropyData::Abac(_)),
        "the container changed which entropy coder the frame uses"
    );
    (decoder.decode(ctx, &back), bytes.len())
}

/// The headline property: same coefficients, different entropy coder, identical pixels.
#[test]
fn abac_decodes_to_the_same_pixels_as_rice_and_is_smaller() {
    let ctx = gpu();
    let (w, h) = (512u32, 512u32);
    let img = synth_image(w, h);

    for q in [50u32, 75, 90] {
        let mut rice = gnc::quality_preset(q);
        rice.entropy_coder = EntropyCoder::Rice;
        let mut abac = rice.clone();
        abac.entropy_coder = EntropyCoder::Abac;

        let (px_rice, size_rice) = roundtrip(ctx, &img, w, h, &rice);
        let (px_abac, size_abac) = roundtrip(ctx, &img, w, h, &abac);

        assert_eq!(
            px_rice.len(),
            px_abac.len(),
            "q={q}: decoded plane sizes differ"
        );
        let worst = px_rice
            .iter()
            .zip(px_abac.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            worst, 0.0,
            "q={q}: abac and Rice decoded different pixels (max |diff| {worst}). Entropy coding \
             is lossless, so this is a coder divergence, not a quality difference."
        );
        assert!(
            size_abac < size_rice,
            "q={q}: abac produced {size_abac} B against Rice's {size_rice} B — it is supposed to \
             be the smaller coder, so a regression here is a coding bug, not a tuning question"
        );
    }
}

#[test]
fn abac_frames_are_gp18_and_carry_entropy_type_5() {
    let ctx = gpu();
    let (w, h) = (256u32, 256u32);
    let img = synth_image(w, h);
    let mut config = gnc::quality_preset(75);
    config.entropy_coder = EntropyCoder::Abac;

    let mut encoder = EncoderPipeline::new(ctx);
    let compressed = encoder.encode(ctx, &img, w, h, &config);
    let bytes = gnc::format::serialize_compressed(&compressed);
    assert_eq!(&bytes[0..4], b"GP18", "abac frames must declare GP18");

    let back = gnc::format::deserialize_compressed(&bytes);
    assert_eq!(back.config.entropy_coder, EntropyCoder::Abac);
    let EntropyData::Abac(tiles) = &back.entropy else {
        panic!("deserialised frame is not abac");
    };
    // Three planes' worth of tiles, each with the geometry the encoder cut with.
    assert_eq!(tiles.len(), 3, "256x256 at tile 256 is one tile per plane");
    for t in tiles {
        assert_eq!(t.tile_size, config.tile_size);
        assert!(!t.block_lengths.is_empty(), "a tile with no blocks decodes to nothing");
        assert_eq!(
            t.block_data.len(),
            t.block_lengths.iter().sum::<u32>() as usize,
            "block lengths must account for exactly the bytes carried"
        );
    }
}

/// Subsampled chroma reaches abac through the per-plane CPU encode path, where the three planes
/// have different tile counts. `normalize_for_chroma` lets abac through for exactly this reason,
/// so the claim needs a test rather than an argument.
#[test]
fn abac_handles_subsampled_chroma() {
    let ctx = gpu();
    let (w, h) = (512u32, 512u32);
    let img = synth_image(w, h);

    for fmt in [ChromaFormat::Yuv422, ChromaFormat::Yuv420] {
        let mut rice = gnc::quality_preset(75);
        rice.chroma_format = fmt;
        rice.entropy_coder = EntropyCoder::Rice;
        rice.normalize_for_chroma();
        let mut abac = rice.clone();
        abac.entropy_coder = EntropyCoder::Abac;
        abac.normalize_for_chroma();
        assert_eq!(
            abac.entropy_coder,
            EntropyCoder::Abac,
            "{fmt:?}: normalize_for_chroma should not fall abac back to Rice"
        );

        // Rice reaches subsampled chroma through the GPU per-plane encoder; abac is CPU-encoded.
        // Compare against Rice on the same CPU encode path so the assertion is about the entropy
        // coder and not about which quantise shader ran.
        rice.gpu_entropy_encode = false;
        let (px_rice, _) = roundtrip(ctx, &img, w, h, &rice);
        let (px_abac, _) = roundtrip(ctx, &img, w, h, &abac);
        let worst = px_rice
            .iter()
            .zip(px_abac.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0, "{fmt:?}: abac and Rice decoded different pixels");
    }
}

/// Both engines must survive the container. Range is the default for a shipped encode; Interval
/// is still reachable and its streams are not interchangeable with Range's, so the tile header
/// has to carry the choice.
#[test]
fn both_arithmetic_engines_roundtrip_through_the_container() {
    let ctx = gpu();
    let (w, h) = (256u32, 256u32);
    let img = synth_image(w, h);
    let mut config = gnc::quality_preset(75);
    config.entropy_coder = EntropyCoder::Abac;

    let mut sizes = Vec::new();
    for engine in ["range", "interval"] {
        std::env::set_var("GNC_ABAC_CODER", engine);
        let (px, size) = roundtrip(ctx, &img, w, h, &config);
        assert!(px.iter().all(|v| v.is_finite()), "{engine}: decoded NaN or inf");
        sizes.push((engine, px, size));
    }
    std::env::remove_var("GNC_ABAC_CODER");

    let worst = sizes[0]
        .1
        .iter()
        .zip(sizes[1].1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        worst, 0.0,
        "the two engines decoded different pixels: {} vs {}",
        sizes[0].0, sizes[1].0
    );
}

/// Not an abac property, recorded because it was found while testing one and it would otherwise
/// be re-found as an abac bug: at subsampled chroma, Rice's **GPU** encode path and its CPU
/// encode path do not produce identical pixels. Both are Rice, so the entropy coding is not the
/// difference; the quantise stage is (the fused quantize+histogram shader runs only on the GPU
/// encode path). The gap is small and pre-existing, and it is why `abac_handles_subsampled_chroma`
/// pins Rice to the CPU path before comparing.
#[test]
fn rice_gpu_and_cpu_encode_paths_differ_at_subsampled_chroma() {
    let ctx = gpu();
    let (w, h) = (512u32, 512u32);
    let img = synth_image(w, h);

    let mut gpu_cfg = gnc::quality_preset(75);
    gpu_cfg.chroma_format = ChromaFormat::Yuv422;
    gpu_cfg.entropy_coder = EntropyCoder::Rice;
    gpu_cfg.normalize_for_chroma();
    let mut cpu_cfg = gpu_cfg.clone();
    cpu_cfg.gpu_entropy_encode = false;

    let (px_gpu, _) = roundtrip(ctx, &img, w, h, &gpu_cfg);
    let (px_cpu, _) = roundtrip(ctx, &img, w, h, &cpu_cfg);
    let worst = px_gpu
        .iter()
        .zip(px_cpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst > 0.0,
        "the two Rice encode paths now agree at 4:2:2 — if that is deliberate, delete this test \
         and the workaround it documents in abac_handles_subsampled_chroma"
    );
    assert!(
        worst < 8.0,
        "Rice GPU vs CPU encode differ by {worst} at 4:2:2, far more than the ~1.7 observed when \
         this was found — that is a regression in the quantise stage, not the known gap"
    );
}
