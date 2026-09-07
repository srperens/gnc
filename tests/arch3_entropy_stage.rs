//! ARCH-3: `gpu_entropy_encode` chooses **where the entropy stage runs**, and nothing else.
//!
//! It used to also choose which of two whole-frame P/B encoders ran, each with its own motion
//! estimation and local decode. A coder that merely lacked a GPU entropy *shader* — abac and
//! bitplane are GPU-decoded but CPU-encoded — was therefore routed onto a second implementation
//! that encoded every P-frame wrong (BUG-18), and that is how a missing shader became broken
//! video and a retracted rate figure.
//!
//! Entropy coding is lossless, so the invariant is exact rather than approximate: moving the
//! entropy stage between CPU and GPU, or swapping one coder for another, must change the
//! **bytes** and leave the **pixels** alone. Both halves are asserted below; a tolerance would
//! have let the original defect through, since its q=90 divergence was under 5/255.
//!
//! Synthesises its own frames, so these run without test material.

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::{EntropyCoder, GpuContext};
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();
fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// Detail at several scales plus noise, so every subband carries something. A pan makes the
/// P-frames carry a real motion-compensated residual rather than nothing.
///
/// `static_right` freezes the right half. That matters more than it looks: `tile_skip_motion`
/// only flags tiles that barely move, so a sequence where *everything* moves never exercises
/// skip mode — and the second half of ARCH-3 (the tile-skip coefficient zeroing being gated on
/// the entropy coder) is invisible on such content. It was found on real 1080p and missed by a
/// full-frame pan at 256x256.
fn panning_sequence_inner(w: u32, h: u32, n: usize, static_right: bool) -> Vec<Vec<f32>> {
    let (wu, hu) = (w as usize, h as usize);
    let mut base = vec![0.0f32; wu * hu * 3];
    let mut rng: u32 = 0x9e37_79b9;
    for y in 0..hu {
        for x in 0..wu {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let noise = (rng % 24) as f32 - 12.0;
            let fine = if ((x / 2) + (y / 3)) % 2 == 0 { 18.0 } else { 0.0 };
            let coarse = ((x / 40) * 37 % 200) as f32;
            let ramp = y as f32 / hu as f32 * 120.0;
            let px = (y * wu + x) * 3;
            base[px] = (coarse + ramp + fine + noise).clamp(0.0, 255.0).round();
            base[px + 1] = (ramp * 1.4 + fine + noise).clamp(0.0, 255.0).round();
            base[px + 2] = (255.0 - coarse + noise).clamp(0.0, 255.0).round();
        }
    }
    (0..n)
        .map(|i| {
            let mut f = vec![0.0f32; base.len()];
            for y in 0..hu {
                for x in 0..wu {
                    let dx = if static_right && x >= wu / 2 { 0 } else { i * 3 };
                    let sx = (x + dx) % wu;
                    for c in 0..3 {
                        f[(y * wu + x) * 3 + c] = base[(y * wu + sx) * 3 + c];
                    }
                }
            }
            f
        })
        .collect()
}

fn panning_sequence(w: u32, h: u32, n: usize) -> Vec<Vec<f32>> {
    panning_sequence_inner(w, h, n, false)
}

/// Half moving, half frozen — the content shape that makes `tile_skip_motion` fire.
fn half_static_sequence(w: u32, h: u32, n: usize) -> Vec<Vec<f32>> {
    panning_sequence_inner(w, h, n, true)
}

/// Encode a GOP through the real container and decode it. Returns the decoded frames and the
/// serialised size of each.
fn roundtrip_sequence(
    ctx: &GpuContext,
    frames: &[Vec<f32>],
    w: u32,
    h: u32,
    config: &gnc::CodecConfig,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let refs: Vec<&[f32]> = frames.iter().map(|f| f.as_slice()).collect();
    let mut encoder = EncoderPipeline::new(ctx);
    let compressed = encoder.encode_sequence(ctx, &refs, w, h, config);
    let sizes = compressed
        .iter()
        .map(|f| gnc::format::serialize_compressed(f).len())
        .collect();
    let decoder = DecoderPipeline::new(ctx);
    (decoder.decode_sequence(ctx, &compressed), sizes)
}

fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(fa, fb)| fa.iter().zip(fb.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0f32, f32::max)
}

/// The ARCH-3 invariant itself. Both arms are the same coder, so `gpu_entropy_encode` is the only
/// variable — and because entropy coding is lossless it must not reach the pixels at all.
///
/// Before the fix this read 28.8 at q=50 and 4.2 at q=90 on the first P-frame, growing down the
/// GOP, with the CPU arm spending 2.1–2.8x the bytes (BUG-18). The ki=2 row matters: with I,P,I,P
/// every P predicts from the I immediately before it, so a divergence there is per-frame and not
/// accumulation.
#[test]
fn entropy_stage_location_does_not_reach_the_pixels() {
    let ctx = gpu();
    let (w, h) = (256u32, 256u32);
    let frames = panning_sequence(w, h, 4);

    for (q, ki) in [(50u32, 9u32), (90, 9), (50, 2), (90, 2)] {
        let run = |gpu_entropy: bool| {
            let mut config = gnc::quality_preset(q);
            config.entropy_coder = EntropyCoder::Rice;
            config.keyframe_interval = ki;
            config.gpu_entropy_encode = gpu_entropy;
            roundtrip_sequence(ctx, &frames, w, h, &config)
        };
        let (gpu_px, gpu_sz) = run(true);
        let (cpu_px, cpu_sz) = run(false);

        let d = max_abs_diff(&gpu_px, &cpu_px);
        eprintln!("q={q} ki={ki}: max |diff| {d}  bytes gpu={gpu_sz:?} cpu={cpu_sz:?}");
        assert_eq!(
            d, 0.0,
            "q={q} ki={ki}: moving the entropy stage between GPU and CPU changed the decoded \
             pixels by {d}. Entropy coding is lossless, so the flag is selecting something else \
             again — that was ARCH-3/BUG-18, and this is the assertion that closed it."
        );
        // The bytes must still move, or the CPU arm is not actually coding anything.
        assert!(
            gpu_sz.iter().all(|&s| s > 0) && cpu_sz.iter().all(|&s| s > 0),
            "q={q} ki={ki}: an empty frame in gpu={gpu_sz:?} cpu={cpu_sz:?}"
        );
    }
}

/// Every entropy coder must produce a working P-frame, whether or not it has a GPU encoder.
///
/// Huffman is why this test exists as well as the one above. It has no GPU encoder wired into the
/// inter path, but it was not on the list of coders that forced the second implementation, so it
/// took the batched pipeline — which pushed nothing into `huffman_tiles`. Every P-frame it wrote
/// carried an empty tile vector. Nothing caught it because no test encoded Huffman video.
#[test]
fn every_coder_codes_a_p_frame() {
    let ctx = gpu();
    let (w, h) = (256u32, 256u32);
    let frames = panning_sequence(w, h, 3);

    for coder in [
        EntropyCoder::Rice,
        EntropyCoder::Rans,
        EntropyCoder::Huffman,
        EntropyCoder::Abac,
        EntropyCoder::Bitplane,
    ] {
        let mut config = gnc::quality_preset(50);
        config.entropy_coder = coder;
        config.keyframe_interval = 9;
        // Assigning the coder is not enough: Huffman's shaders address 8 subband groups, so the
        // decomposition depth has to be re-clamped afterwards. `main.rs` gets this through
        // `normalize_for_chroma`; a caller that only sets the field indexes past the histogram
        // buffer and panics in `huffman_gpu`.
        config.normalize_for_chroma();
        let (decoded, sizes) = roundtrip_sequence(ctx, &frames, w, h, &config);

        // A P-frame whose tiles never got filled serialises to a header and little else. The
        // I-frame is the scale: a P-frame under a hundredth of it is not a coded frame.
        let i_size = sizes[0];
        let floor = i_size / 100;
        for (i, &s) in sizes.iter().enumerate().skip(1) {
            assert!(
                s > floor,
                "{coder:?}: P-frame {i} serialised to {s} bytes against an I-frame of {i_size}. \
                 That is an empty tile vector, not a cheap frame."
            );
        }

        // And it has to decode to something, not just be large.
        let psnr = |a: &[f32], b: &[f32]| -> f32 {
            let mse: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| ((x - y) as f64).powi(2))
                .sum::<f64>()
                / a.len() as f64;
            (10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10()) as f32
        };
        for (i, (dec, src)) in decoded.iter().zip(frames.iter()).enumerate() {
            let p = psnr(dec, src);
            eprintln!("{coder:?} frame {i}: {p:.2} dB, {} bytes", sizes[i]);
            assert!(
                p > 25.0,
                "{coder:?} frame {i} decoded to {p:.2} dB — the frame encoder and the entropy \
                 stage disagree about what was written."
            );
        }
    }
}

/// The payoff, and the thing BUG-18's retraction was waiting for: with the frame encoder no
/// longer chosen by the entropy flag, abac and Rice code the *same* coefficients, so an inter
/// rate comparison between them is at identical pixels rather than across two encoders.
#[test]
fn abac_and_rice_video_decode_to_the_same_pixels() {
    let ctx = gpu();
    let (w, h) = (256u32, 256u32);
    let content = [
        ("pan", panning_sequence(w, h, 4)),
        ("half-static", half_static_sequence(w, h, 4)),
    ];

    for ((label, frames), q) in content.iter().flat_map(|c| [50u32, 90].map(|q| (c, q))) {
        let run = |coder: EntropyCoder| {
            let mut config = gnc::quality_preset(q);
            config.entropy_coder = coder;
            config.keyframe_interval = 9;
            roundtrip_sequence(ctx, frames, w, h, &config)
        };
        let (rice_px, rice_sz) = run(EntropyCoder::Rice);
        let (abac_px, abac_sz) = run(EntropyCoder::Abac);

        let d = max_abs_diff(&rice_px, &abac_px);
        let rice_total: usize = rice_sz.iter().sum();
        let abac_total: usize = abac_sz.iter().sum();
        eprintln!(
            "{label} q={q}: max |diff| {d}  rice={rice_total} abac={abac_total} ({:+.1}%)",
            (abac_total as f64 / rice_total as f64 - 1.0) * 100.0
        );
        assert_eq!(
            d, 0.0,
            "{label} q={q}: abac and Rice video no longer decode to the same pixels ({d}), so an \
             inter rate comparison between them is not at identical pixels and any figure taken \
             from one is not comparable with the other. On `half-static` the usual cause is a \
             coefficient decision gated on the entropy coder — the tile-skip zeroing was."
        );
    }
}
