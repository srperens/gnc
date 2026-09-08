//! A behavioural fingerprint of the encoder: what it *produces*, not what it is.
//!
//! **COORD-6.** Five of the seven recorded measurement failures in this repository are the same
//! shape: one session measuring correctly while `main` moves underneath, so a table's early rows
//! and late rows come from different codecs. Nothing errors — the numbers are simply from two
//! encoders and are read as one. COORD-4 priced the two obvious mechanisms against six of those
//! instances and refused both: a claim-time `HEAD` stamp would have caught **1 of 6**, printing
//! each claim's commit **0 of 6**, because a claim is taken when an item is picked up and a
//! measurement happens somewhere else entirely.
//!
//! `shasum` of `target/release/gnc` cannot stand in for this. It changes when a doc comment does,
//! so it over-warns hard enough to be ignored, which is the failure mode of every check nobody
//! runs twice.
//!
//! The only honest test of "did the encoder's output move" is running the encoder — so this runs
//! it, on a matrix small enough to be free. Measured on the dev machine: **0.52 s** for ten
//! configurations, against the minutes a real sweep costs. Two numbers carrying the same
//! fingerprint are comparable; two carrying different ones are not.
//!
//! Validated by the knobs whose effect is already known:
//!
//! | knob | output moves? | fingerprint |
//! |---|---|---|
//! | `GNC_REF_FROM_SOURCE=0` | no — 24 of 24 points byte-identical (`0072`) | **unchanged** |
//! | `GNC_PAD_FILL=decay` | yes | changed |
//! | `GNC_DEAD_ZONE=0.3` | yes | changed |
//! | `GNC_REF_DEBLOCK=1` | yes | changed |
//!
//! **The matrix is pinned and versioned.** Changing it changes every fingerprint ever published,
//! so it is a decision record and not a commit.
//!
//! **What it does not do.** It cannot say *why* two fingerprints differ, it says nothing about a
//! path outside the matrix, and it protects only numbers that carry it — the same adoption problem
//! prose has, met with one command instead of a paragraph.

use crate::{format, CodecConfig, EntropyCoder, GpuContext};

/// Bump this when the matrix below changes, and write the decision record.
pub const MATRIX_VERSION: &str = "v1";

/// One configuration's contribution.
pub struct Row {
    pub name: &'static str,
    pub bytes: usize,
    pub crc: u32,
}

/// The compound digest and the rows it was built from.
pub struct Fingerprint {
    pub digest: u32,
    pub rows: Vec<Row>,
}

struct Case {
    name: &'static str,
    q: u32,
    chroma: crate::ChromaFormat,
    coder: EntropyCoder,
    frames: usize,
    ki: u32,
}

/// The pinned matrix. Configurations cross the axes that have actually moved output in this
/// repository's history — entropy coder, chroma format, the lossless boundary, and inter — rather
/// than trying to be exhaustive, which is this tool's stated limit.
fn matrix() -> [Case; 10] {
    use crate::ChromaFormat::{Yuv420, Yuv444};
    use EntropyCoder::{Abac, Rans, Rice};
    [
        Case { name: "still q50 rice 444",  q: 50,  chroma: Yuv444, coder: Rice, frames: 1, ki: 1 },
        Case { name: "still q90 rice 444",  q: 90,  chroma: Yuv444, coder: Rice, frames: 1, ki: 1 },
        Case { name: "still q90 rice 420",  q: 90,  chroma: Yuv420, coder: Rice, frames: 1, ki: 1 },
        Case { name: "still q90 abac 444",  q: 90,  chroma: Yuv444, coder: Abac, frames: 1, ki: 1 },
        Case { name: "still q10 rans 444",  q: 10,  chroma: Yuv444, coder: Rans, frames: 1, ki: 1 },
        Case { name: "still q100 med 444",  q: 100, chroma: Yuv444, coder: Rice, frames: 1, ki: 1 },
        Case { name: "seq   q90 rice 444",  q: 90,  chroma: Yuv444, coder: Rice, frames: 3, ki: 2 },
        Case { name: "seq   q90 rice 420",  q: 90,  chroma: Yuv420, coder: Rice, frames: 3, ki: 2 },
        Case { name: "seq   q99 rice 444",  q: 99,  chroma: Yuv444, coder: Rice, frames: 3, ki: 2 },
        Case { name: "seq  q100 rice 444",  q: 100, chroma: Yuv444, coder: Rice, frames: 3, ki: 2 },
    ]
}

const W: u32 = 384;
const H: u32 = 384;

/// Smooth-plus-texture, generated here rather than loaded.
///
/// Generated because a fingerprint that depends on `test_material/` is not portable between
/// machines. Integer-valued because BUG-45: a fractional source makes a lossless configuration
/// quietly lossy, so a fractional fingerprint would be measuring that instead. Smooth rather than
/// hash noise because on pure noise the bit-exact candidate wins every frame — the q=99 and q=100
/// sequence rows then coded to identical bytes and one of the ten configurations measured nothing.
fn frame(seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            let jitter = |k: u32| {
                let v = (x * 7 + y * 13 + k * 101 + seed * 31).wrapping_mul(2_654_435_761);
                ((v >> 27) & 7) as i32
            };
            let px = [
                (x + y + seed) as i32 / 2 + jitter(0),
                (x * 2 + seed * 3) as i32 / 3 + jitter(1),
                (y * 3) as i32 / 2 + jitter(2),
            ];
            for c in px {
                out.push(f32::from(c.rem_euclid(256) as u8));
            }
        }
    }
    out
}

/// Encode the pinned matrix and digest what came out.
#[must_use]
pub fn compute(ctx: &GpuContext) -> Fingerprint {
    let (f0, f1, f2) = (frame(0), frame(7), frame(19));
    let mut digest_input: Vec<u8> = MATRIX_VERSION.as_bytes().to_vec();
    let mut rows = Vec::new();
    for case in matrix() {
        let mut config = CodecConfig {
            chroma_format: case.chroma,
            keyframe_interval: case.ki,
            entropy_coder: case.coder,
            ..crate::quality_preset(case.q)
        };
        config.set_tile_size(256);
        let mut encoder = crate::encoder::pipeline::EncoderPipeline::new(ctx);
        let bytes = if case.frames == 1 {
            format::serialize_compressed(&encoder.encode(ctx, &f0, W, H, &config))
        } else {
            let refs: Vec<&[f32]> = vec![&f0, &f1, &f2];
            format::serialize_sequence(
                &encoder.encode_sequence(ctx, &refs, W, H, &config),
                (30, 1),
            )
        };
        let crc = format::crc32(&bytes);
        digest_input.extend_from_slice(&crc.to_le_bytes());
        digest_input.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        rows.push(Row { name: case.name, bytes: bytes.len(), crc });
    }
    Fingerprint { digest: format::crc32(&digest_input), rows }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two properties, and the tool is worthless without either.
    ///
    /// **Deterministic**, or a fingerprint printed beside a number means nothing — every
    /// comparison would be a coin flip. **All rows distinct**, or a configuration is measuring
    /// what another one already measured and the matrix is one sample smaller than it claims;
    /// that is exactly what the first version of the input generator did (pure hash noise made
    /// the q=99 and q=100 sequence rows identical).
    #[test]
    fn fingerprint_is_deterministic_and_every_row_is_a_distinct_sample() {
        let ctx = GpuContext::new();
        let a = compute(&ctx);
        let b = compute(&ctx);
        assert_eq!(
            a.digest, b.digest,
            "two runs of the same binary disagree, so the fingerprint cannot certify that two \
             numbers are comparable — which is its only job"
        );
        for (x, y) in a.rows.iter().zip(&b.rows) {
            assert_eq!((x.name, x.crc, x.bytes), (y.name, y.crc, y.bytes));
        }
        let mut seen = std::collections::HashMap::new();
        for row in &a.rows {
            if let Some(prev) = seen.insert(row.crc, row.name) {
                panic!(
                    "configurations {prev} and {} code to identical bytes, so the matrix has \
                     {} distinct samples and not {}. Change the content or the case, do not \
                     delete the assertion",
                    row.name,
                    a.rows.len() - 1,
                    a.rows.len()
                );
            }
        }
    }
}
