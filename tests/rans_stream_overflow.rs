//! BUG-9 — rANS at a fine quantiser step must be refused by name, not by wrapping.
//!
//! Two separate limits stop rANS from reaching a fine step, and characterising the first one
//! turned up the second.
//!
//! **The output slot.** `rans_encode.wgsl` gives every stream a fixed 4 KB slot and writes it
//! *backwards* from the end, so `write_ptr` counts down. It used to be decremented with no bound
//! check: a stream needing more than 4 KB ran `write_ptr` past zero, where it wrapped to just
//! under 2^32, and because `stream_base_byte + write_ptr` is u32 arithmetic the wrapped write
//! landed at `stream_base_byte - 1` and downwards — inside the *previous* stream's slot, ORing
//! bits into bytes that were already correct.
//!
//! **The cumfreq table.** Every subband group's table for one tile is loaded into a
//! 4096-entry workgroup array, and it is the *sum* over the tile's groups that has to fit. Past
//! the end the shader read and wrote outside the array. That one does not announce itself: a
//! tile can overrun its tables and still emit streams that fit their slots, and then nothing
//! downstream notices.
//!
//! Which limit binds is not a toss-up. On the per-subband path the **histogram arena** (5120
//! bins, BUG-35) and the **cumfreq table** (4097 entries, BUG-9) both bind before any stream
//! slot: a tile that overruns either still emits streams that fit, and then nothing downstream
//! notices. Histogram is checked first because it is the earlier pass. The window where only
//! cumfreq fires is real — qstep 1.25 on this content needs 4247 cumfreq entries against a
//! histogram that still fits. Measured with `--rans` at the default step on photographic
//! stills: kristensara_720p needs 4020 of 4097 cumfreq entries at q=75 and 4165 at q=76, where
//! it is refused; bbb_1080p still fits at q=76 with 4052 and goes at q=77. The histogram arena
//! overflows later (bbb q=85 at 5322/5120). The slot is reachable on its own only off the
//! subband path, which is what the third test below does.
//!
//! The content here is low-frequency randomness — full-range and unpredictable in the LL band.
//! That matters: neither uniform noise nor a full-contrast checkerboard reaches either limit at
//! any step, because both average out to a flat LL and it is LL magnitude times LL entropy that
//! exhausts the tables and the slots. Real photographic content does reach them.
//!
//! Run with: `cargo test --release --test rans_stream_overflow`

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::GpuContext;
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// 512x512 is 2x2 tiles at the default tile size, so a stream that overran its slot had a
/// neighbour to corrupt. `block` px squares of uniform random keep the LL band both full-range
/// and unpredictable, which is what drives the alphabet and the stream length.
fn make_lowfreq_random(w: u32, h: u32, block: u32) -> Vec<f32> {
    let (bw, bh) = (w.div_ceil(block), h.div_ceil(block));
    let mut rng: u32 = 0x1234_5678;
    let mut blocks = Vec::with_capacity((bw * bh * 3) as usize);
    for _ in 0..(bw * bh * 3) {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        blocks.push((rng >> 24) as f32);
    }

    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let bi = ((y / block * bw) + x / block) * 3;
            data.extend_from_slice(&blocks[bi as usize..bi as usize + 3]);
        }
    }
    data
}

#[test]
fn the_quality_range_that_selects_rans_fits_both_limits() {
    let ctx = gpu();
    let (w, h) = (512u32, 512u32);
    let rgb = make_lowfreq_random(w, h, 16);

    for q in [1u32, 5, 10, 15, 20] {
        let config = gnc::quality_preset(q);
        assert_eq!(
            config.entropy_coder,
            gnc::EntropyCoder::Rans,
            "q={q} is expected to select rANS; if the preset moved, move this test with it"
        );

        let mut encoder = EncoderPipeline::new(ctx);
        let compressed = encoder.encode(ctx, &rgb, w, h, &config);
        let decoded = DecoderPipeline::new(ctx).decode(ctx, &compressed);
        assert_eq!(
            decoded.len(),
            rgb.len(),
            "q={q}: rANS round trip changed the plane size"
        );
    }
}

#[test]
#[should_panic(expected = "histogram bins")]
fn a_histogram_arena_that_does_not_fit_is_refused_by_name() {
    // BUG-35: `shared_hist` holds 5120 bins, the sum of per-group alphabets. At qstep 1.5 this
    // content asks for ~6040, so the histogram check fires *before* the cumfreq one (5120 <
    // 4097+groups). That used to be silent corruption via naga's atomic clamp.
    let mut config = gnc::quality_preset(15);
    config.quantization_step = 1.5;
    encode_with(config);
}

#[test]
#[should_panic(expected = "cumfreq entries but the encode shader")]
fn a_cumfreq_table_that_does_not_fit_is_refused_by_name() {
    // Window where cumfreq overflows 4097 but the histogram still fits 5120: qstep 1.25 on this
    // content needs 4247 cumfreq entries (hist ≈ 4235). Coarser (1.5) overflows the histogram
    // first; finer overshoots both. The slot case below still has to leave the subband path.
    let mut config = gnc::quality_preset(15);
    config.quantization_step = 1.25;
    encode_with(config);
}

#[test]
#[should_panic(expected = "overflowed their 4096-byte output slot")]
fn a_stream_that_does_not_fit_its_slot_is_refused_by_name() {
    // The single-table path caps its own tables at MAX_ALPHABET + 1, so they cannot be what
    // gives out; at qstep 0.5 they sit exactly at 4097 of 4097 and 32 of 128 streams still do
    // not fit their 4 KB slots. That is the only configuration found where the slot goes first,
    // which is the whole reason this test leaves the per-subband path.
    let mut config = gnc::quality_preset(15);
    config.per_subband_entropy = false;
    config.quantization_step = 0.5;
    encode_with(config);
}

fn encode_with(config: gnc::CodecConfig) {
    let ctx = gpu();
    let (w, h) = (512u32, 512u32);
    let rgb = make_lowfreq_random(w, h, 16);
    let mut encoder = EncoderPipeline::new(ctx);
    let _ = encoder.encode(ctx, &rgb, w, h, &config);
}
