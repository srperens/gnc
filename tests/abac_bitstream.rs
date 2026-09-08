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
use gnc::encoder::abac::Coder;
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

    // Per-config, never `set_var`. Cargo runs tests in parallel threads and the environment is
    // process-global, so setting GNC_ABAC_CODER here changed what *other* tests encoded with —
    // which is how `abac_handles_subsampled_chroma` started failing with a max |diff| of 85321
    // after this test was added. That race also found a real decoder bug (one `abac_coder` field
    // for three planes, holding the last plane's engine), so it is worth naming rather than just
    // fixing: shared mutable state in the encoder's configuration is the defect, and the config
    // field is the fix.
    let mut sizes = Vec::new();
    for engine in [Coder::Range, Coder::Interval] {
        config.abac_coder = engine;
        let (px, size) = roundtrip(ctx, &img, w, h, &config);
        assert!(px.iter().all(|v| v.is_finite()), "{engine:?}: decoded NaN or inf");
        sizes.push((engine, px, size));
    }

    let worst = sizes[0]
        .1
        .iter()
        .zip(sizes[1].1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        worst, 0.0,
        "the two engines decoded different pixels: {:?} vs {:?}",
        sizes[0].0, sizes[1].0
    );
}

/// Not an abac property, kept because it was found while testing one and would otherwise be
/// re-found as an abac bug: at subsampled chroma, Rice's **GPU** encode path and its CPU encode
/// path must produce identical pixels.
///
/// **This test asserted the opposite until 2026-09-08.** It was written when the two paths
/// differed by about 1.7 (BUG-16) and its job was to pin that gap down so a regression could be
/// told from the known defect. The cause was the fused quantiser's sparse dead-zone expansion,
/// which existed on that one quantiser and no other; it is now off by default (decision `0038`),
/// so the paths agree and the assertion is inverted. `GNC_SPARSE_DZ=1` brings the divergence back,
/// which is why the failure message below points at it.
///
/// Note for whoever closes **BUG-28**: `abac_handles_subsampled_chroma` pins Rice to the CPU path
/// as a workaround for this gap. That workaround is now *probably* unnecessary — but BUG-28 is a
/// separate open defect about abac and Rice disagreeing at subsampled chroma, so check rather than
/// assume, and unpin it as part of that item rather than in passing.
#[test]
fn rice_gpu_and_cpu_encode_paths_agree_at_subsampled_chroma() {
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
    assert_eq!(
        worst, 0.0,
        "Rice's GPU and CPU encode paths differ by {worst} at 4:2:2. Both are Rice, so this is \
         the quantise stage, not the entropy coder. If GNC_SPARSE_DZ is set, that is the expected \
         cause (BUG-16, decision 0038); if it is not, a second divergence has appeared between \
         the fused quantiser and the separate one."
    );
}

/// Inter frames. The rate result this coder shipped on is intra, and the contexts were tuned on
/// intra coefficients, so nothing here claims abac is *good* on P-frames — only that the two
/// coders agree given the same encode path.
///
/// **Read that scope literally.** Both arms are pinned to the CPU encode path, and BUG-18 says
/// that path encodes every P-frame wrong (the first P after an I diverges from the GPU path and
/// costs 2.1x the bytes). So this test proves "abac == Rice on the CPU path"; it does **not**
/// prove abac's inter works, and the ABAC-SHIP inter rate figure was retracted for exactly that
/// confusion. It stays pinned to one path deliberately: comparing across paths would make the
/// assertion about BUG-18 rather than about the entropy coder. An adaptive coder that diverges reconstructs a plausible wrong image
/// rather than failing, so "it ran and produced a file" is not evidence; the check is again that
/// abac and Rice, coding identical residual coefficients, decode to identical pixels.
#[test]
fn abac_survives_a_p_frame_chain() {
    let ctx = gpu();
    let (w, h) = (256u32, 256u32);
    // Four frames with real motion, so the P-frames carry a residual worth coding.
    let base = synth_image(w, h);
    let shifted: Vec<Vec<f32>> = (0..4)
        .map(|i| {
            let mut f = vec![0.0f32; base.len()];
            let dx = (i * 3) as usize;
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let sx = (x + dx) % w as usize;
                    for c in 0..3 {
                        f[(y * w as usize + x) * 3 + c] = base[(y * w as usize + sx) * 3 + c];
                    }
                }
            }
            f
        })
        .collect();
    let refs: Vec<&[f32]> = shifted.iter().map(|f| f.as_slice()).collect();

    let mut decoded = Vec::new();
    let mut sizes = Vec::new();
    for coder in [EntropyCoder::Rice, EntropyCoder::Abac] {
        let mut config = gnc::quality_preset(75);
        config.entropy_coder = coder;
        config.keyframe_interval = 4;
        // Both arms on the CPU encode path: Rice's GPU encoder does not produce identical
        // coefficients (see rice_gpu_and_cpu_encode_paths_differ_at_subsampled_chroma), which
        // would make this assertion about the quantiser rather than the entropy coder.
        config.gpu_entropy_encode = false;

        let mut encoder = EncoderPipeline::new(ctx);
        let frames = encoder.encode_sequence(ctx, &refs, w, h, &config);
        assert_eq!(frames.len(), 4);
        let round: Vec<gnc::CompressedFrame> = frames
            .iter()
            .map(|f| gnc::format::deserialize_compressed(&gnc::format::serialize_compressed(f)))
            .collect();
        sizes.push(
            frames
                .iter()
                .map(|f| gnc::format::serialize_compressed(f).len())
                .sum::<usize>(),
        );
        let decoder = DecoderPipeline::new(ctx);
        decoded.push(decoder.decode_sequence(ctx, &round));
    }

    for (i, (rice, abac)) in decoded[0].iter().zip(decoded[1].iter()).enumerate() {
        let worst = rice
            .iter()
            .zip(abac.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            worst, 0.0,
            "frame {i} of a 1I+3P chain decoded differently under abac (max |diff| {worst})"
        );
    }
    eprintln!(
        "1I+3P at q=75: Rice {} B, abac {} B ({:+.1}%)",
        sizes[0],
        sizes[1],
        100.0 * (sizes[1] as f64 / sizes[0] as f64 - 1.0)
    );
}

/// GP18's only addition is entropy type 5, so a GP18 frame using any older coder must be a GP17
/// frame with a different label. Asserting that directly is worth more than believing it: relabel
/// the magic, decode, and require the identical picture. It also exercises the other half —
/// that the decoder still reads GP17, which is what every file written before today says.
#[test]
fn gp18_rice_frames_are_gp17_payloads_with_a_new_label() {
    let ctx = gpu();
    let (w, h) = (256u32, 256u32);
    let img = synth_image(w, h);
    let mut config = gnc::quality_preset(75);
    config.entropy_coder = EntropyCoder::Rice;

    let mut encoder = EncoderPipeline::new(ctx);
    let compressed = encoder.encode(ctx, &img, w, h, &config);
    let mut bytes = gnc::format::serialize_compressed(&compressed);
    assert_eq!(&bytes[0..4], b"GP18");

    let decoder = DecoderPipeline::new(ctx);
    let as_gp18 = decoder.decode(ctx, &gnc::format::deserialize_compressed(&bytes));

    bytes[0..4].copy_from_slice(b"GP17");
    let as_gp17 = decoder.decode(ctx, &gnc::format::deserialize_compressed(&bytes));

    assert_eq!(
        as_gp18, as_gp17,
        "relabelling a Rice frame GP18 → GP17 changed the decode, so GP18 moved something other \
         than the magic — either the generation added a field it should not have, or the decoder \
         gates a field on gen >= 18 that older files also carry"
    );
}

// `inter_reconstruction_depends_on_the_encode_path` lived here and asserted that the two encode
// paths *never* produce the same reconstruction on inter. BUG-18 is fixed — ARCH-3 removed the
// second frame encoder the flag was selecting — so the assertion is now the opposite one, and it
// lives in `tests/arch3_entropy_stage.rs`. Two things that test cost and that are worth keeping:
//
// - Two wrong explanations were published before it existed ("the trigger is adaptive
//   quantisation", then "AQ and B-frames are two triggers"). Both were read off **PSNR averages
//   printed to two decimals**, which a max |diff| of 8 on a handful of pixels does not move.
//   Aggregate quality is not evidence of pixel identity.
// - Its intra counterpart, `abac_decodes_to_the_same_pixels_as_rice_and_is_smaller` below, is
//   what said the coder was not the variable. Keep both ends: intra agreeing while inter did not
//   is what located the defect in the frame encoder rather than in abac.
