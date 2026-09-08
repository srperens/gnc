//! Where the two P-frame encode implementations first disagree (BUG-18).
//!
//! `gpu_entropy_encode` does not merely move entropy coding between CPU and GPU: it selects
//! between two independent implementations of the whole P-frame encode, each with its own local
//! decode. This narrows *which frame* and *which plane* first differ, which is the cheap half of
//! the diagnosis.

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::{EntropyCoder, GpuContext};
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();
fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

fn synth(w: u32, h: u32) -> Vec<f32> {
    let mut d = Vec::with_capacity((w * h * 3) as usize);
    let mut rng: u32 = 0x9e37_79b9;
    for y in 0..h {
        for x in 0..w {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let n = (rng % 24) as f32 - 12.0;
            let fine = if ((x / 2) + (y / 3)) % 2 == 0 {
                18.0
            } else {
                0.0
            };
            let coarse = ((x / 40) * 37 % 200) as f32;
            let ramp = y as f32 / h as f32 * 120.0;
            d.push((coarse + ramp + fine + n).clamp(0.0, 255.0).round());
            d.push((ramp * 1.4 + fine + n).clamp(0.0, 255.0).round());
            d.push((255.0 - coarse + n).clamp(0.0, 255.0).round());
        }
    }
    d
}

#[test]
#[ignore = "diagnostic for BUG-18, not a gate"]
fn locate_bug18() {
    let ctx = gpu();
    let (w, h) = (256u32, 256u32);
    let base = synth(w, h);
    let frames: Vec<Vec<f32>> = (0..4)
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
    let refs: Vec<&[f32]> = frames.iter().map(|f| f.as_slice()).collect();

    // ki=2 makes every P predict from the I immediately before it, so a reference that never
    // advances past the I-frame cannot matter. ki=9 chains three Ps. If the paths agree at ki=2
    // and diverge at ki=9, the missing reference update is confirmed rather than inferred.
    for (q, ki) in [(50u32, 9u32), (90, 9), (50, 2), (90, 2)] {
        let mut decoded = Vec::new();
        let mut sizes = Vec::new();
        for gpu_encode in [true, false] {
            let mut config = gnc::quality_preset(q);
            config.entropy_coder = EntropyCoder::Rice;
            config.keyframe_interval = ki;
            config.gpu_entropy_encode = gpu_encode;
            let mut enc = EncoderPipeline::new(ctx);
            let cf = enc.encode_sequence(ctx, &refs, w, h, &config);
            sizes.push(
                cf.iter()
                    .map(|f| gnc::format::serialize_compressed(f).len())
                    .collect::<Vec<_>>(),
            );
            let dec = DecoderPipeline::new(ctx);
            decoded.push(dec.decode_sequence(ctx, &cf));
        }
        eprintln!("--- q={q} ki={ki} ---");
        eprintln!("  bytes gpu={:?}", sizes[0]);
        eprintln!("  bytes cpu={:?}", sizes[1]);
        for (i, (a, b)) in decoded[0].iter().zip(decoded[1].iter()).enumerate() {
            // Interleaved RGB; report per channel so a chroma-only defect is visible.
            let mut per_ch = [0.0f32; 3];
            let mut n_diff = 0usize;
            for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let d = (x - y).abs();
                if d > 0.0 {
                    n_diff += 1;
                }
                let c = k % 3;
                if d > per_ch[c] {
                    per_ch[c] = d;
                }
            }
            eprintln!(
                "  frame {i}: max |diff| R{:.3} G{:.3} B{:.3}   differing samples {n_diff}",
                per_ch[0], per_ch[1], per_ch[2]
            );
        }
    }
}
