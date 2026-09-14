//! The GPU abac *encoder* must produce the CPU encoder's bytes, not merely decodable ones.
//!
//! This is a stronger requirement than the decoder's, and deliberately so. A decoder that
//! diverges reconstructs a wrong image, which at least shows up in PSNR. An *encoder* that
//! diverges produces a perfectly valid stream of a slightly different size — every rate figure
//! abac has (−16.0% over q=60-99, −13.4% lossless) silently becomes a figure for a different
//! coder, and nothing fails. So the assertion is byte-identity per block against `abac.rs`.
//!
//! Both arithmetic engines and both output-sizing modes are covered: they are four combinations
//! of the same binarisation, and a bug in one is invisible to the others.
//!
//! Synthesises its own coefficients so it runs without test material.

use gnc::encoder::abac::Coder;
use gnc::encoder::abac_gpu_encode::{verify_against_cpu_encoder, GpuAbacEncoder, Sizing};
use gnc::GpuContext;
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();
fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// Coefficients shaped like a quantised wavelet plane: a dense LL corner, sparse clustered detail
/// bands, a heavy tail. Uniform noise would leave the significance contexts untested, and the
/// contexts are where every divergence so far has been.
fn synth_plane(w: usize, h: usize, tile: usize, levels: u32, seed: u64) -> Vec<i32> {
    let mut out = vec![0i32; w * h];
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let ll = tile >> levels;
    for y in 0..h {
        for x in 0..w {
            let (ty, tx) = (y % tile, x % tile);
            let r = next();
            // Inside the LL corner: dense and large. Elsewhere: sparse, small, clustered.
            let v = if tx < ll && ty < ll {
                (r % 800) as i32 - 400
            } else if ((ty / 16) + (tx / 16)) % 3 == 0 && r % 3 != 0 {
                match r % 100 {
                    0..=59 => 1,
                    60..=84 => 2,
                    85..=95 => 3 + (r % 7) as i32,
                    _ => 50 + (r % 4000) as i32,
                }
            } else {
                0
            };
            out[y * w + x] = if r % 2 == 0 { v } else { -v };
        }
    }
    out
}

/// (tile_size, tiles_x, tiles_y, num_levels, cb)
///
/// The same shapes the decoder is verified on, expressed in whole tiles because that is the unit
/// the encoder produces: ragged partial code-blocks come from the deep subbands rather than from
/// a ragged plane. `levels = 0` is the MED lossless path, where the whole tile is one region.
const GEOMETRIES: &[(u32, usize, usize, u32, u32)] = &[
    (64, 1, 1, 3, 64),
    (128, 1, 1, 4, 64),
    (256, 1, 1, 5, 64),
    (256, 1, 1, 5, 32),
    (256, 2, 2, 5, 64),
    (64, 3, 2, 0, 64),
    (128, 2, 1, 4, 32),
    // A realistic plane, so a divergence that only appears at scale has somewhere to appear:
    // 2048x1280 is a padded 1080p luma plane, 40 tiles and 1000 code-blocks.
    (256, 8, 5, 5, 64),
];

#[test]
fn gpu_encode_matches_cpu_encoder_byte_for_byte() {
    let ctx = gpu();
    let mut enc = GpuAbacEncoder::new(ctx);
    for (i, &(ts, tx, ty, levels, cb)) in GEOMETRIES.iter().enumerate() {
        let w = tx * ts as usize;
        let h = ty * ts as usize;
        let plane = synth_plane(w, h, ts as usize, levels, i as u64 + 1);
        for coder in [Coder::Interval, Coder::Range] {
            for sizing in [Sizing::CountThenEmit, Sizing::BoundedSlots] {
                let bytes = verify_against_cpu_encoder(
                    ctx, &mut enc, &plane, w, ts, tx, ty, levels, cb, coder, sizing,
                );
                let st = enc.stats();
                assert_eq!(st.bytes, bytes, "stats must report what was produced");
                assert!(
                    st.blocks > 0,
                    "the canary must count the blocks it dispatched"
                );
                eprintln!(
                    "  {w}x{h} ts={ts} levels={levels} cb={cb} {coder:?} {sizing:?}: \
                     {bytes} B, {} blocks, scratch {} B ({:.1}x output), {} coder pass(es)",
                    st.blocks,
                    st.scratch_bytes,
                    st.scratch_bytes as f64 / bytes.max(1) as f64,
                    st.coder_passes,
                );
            }
        }
    }
}

/// Degenerate planes are where an interval coder is most likely to mishandle renormalisation, and
/// where the range coder's carry propagation through a run of 0xFF bytes is most likely to fire.
/// They cost nothing to check and the carry path has no other test.
#[test]
fn gpu_encode_handles_degenerate_planes() {
    let ctx = gpu();
    let mut enc = GpuAbacEncoder::new(ctx);
    let cases: &[(&str, i32)] = &[("zero", 0), ("one", 1), ("minus one", -1), ("big", -9999)];
    for &(label, fill) in cases {
        let plane = vec![fill; 128 * 128];
        for coder in [Coder::Interval, Coder::Range] {
            for sizing in [Sizing::CountThenEmit, Sizing::BoundedSlots] {
                let bytes = verify_against_cpu_encoder(
                    ctx, &mut enc, &plane, 128, 128, 1, 1, 4, 64, coder, sizing,
                );
                eprintln!("  all-{label} 128x128 {coder:?} {sizing:?}: {bytes} B");
            }
        }
    }
}

/// The GPU encoder's output must decode on the GPU decoder, which is the pairing the codec
/// actually ships: encode on one shader, decode on the other, with the CPU in neither path.
#[test]
fn gpu_encode_round_trips_through_gpu_decode() {
    use gnc::encoder::abac_gpu::{BlockInfo, GpuAbacDecoder};
    use gnc::encoder::abac_tile::code_blocks;
    use wgpu::util::DeviceExt;

    let ctx = gpu();
    let mut enc = GpuAbacEncoder::new(ctx);
    let dec = GpuAbacDecoder::new(ctx);

    let (ts, tx, ty, levels, cb) = (256u32, 2usize, 2usize, 5u32, 64u32);
    let (w, h) = (tx * ts as usize, ty * ts as usize);
    let plane = synth_plane(w, h, ts as usize, levels, 99);
    let floats: Vec<f32> = plane.iter().map(|&v| v as f32).collect();
    let input = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rt_input"),
            contents: bytemuck::cast_slice(&floats),
            usage: wgpu::BufferUsages::STORAGE,
        });

    for coder in [Coder::Interval, Coder::Range] {
        for sizing in [Sizing::CountThenEmit, Sizing::BoundedSlots] {
            let tiles =
                enc.encode_plane_to_tiles(ctx, &input, w, tx, ty, ts, levels, cb, coder, sizing);

            // Flatten to the decoder's block table, in plane coordinates.
            let mut packed: Vec<u8> = Vec::new();
            let mut infos: Vec<BlockInfo> = Vec::new();
            for (t, tile) in tiles.iter().enumerate() {
                let origin = (t / tx) * ts as usize * w + (t % tx) * ts as usize;
                let mut off = 0usize;
                for (i, (bx, by, bw, bh)) in code_blocks(ts as usize, levels, cb as usize)
                    .into_iter()
                    .enumerate()
                {
                    let len = tile.block_lengths[i] as usize;
                    infos.push(BlockInfo::new(
                        packed.len() as u32,
                        len as u32,
                        (origin + by * w + bx) as u32,
                        bw as u32,
                        bh as u32,
                        w as u32,
                    ));
                    packed.extend_from_slice(&tile.block_data[off..off + len]);
                    off += len;
                }
            }
            let (got, _) = dec.decode(ctx, &packed, &infos, plane.len(), coder);
            let diff = got
                .iter()
                .zip(plane.iter())
                .map(|(a, b)| (a - b).abs())
                .max()
                .unwrap();
            assert_eq!(
                diff, 0,
                "GPU encode -> GPU decode must be exact ({coder:?}, {sizing:?})"
            );
            eprintln!(
                "  round trip {coder:?} {sizing:?}: {} B, max |diff| 0",
                packed.len()
            );
        }
    }
}
