//! BUG-12 — the wavelet-level ceiling must follow the tile size that is actually in use.
//!
//! `quality_preset` clamps `wavelet_levels` against the *default* tile size, but `--tile-size`
//! is applied after it returns. Assigning `tile_size` directly therefore left a ceiling derived
//! from a tile size no longer in use: `--tile-size 512` was capped at 5 levels where 512 allows
//! 6, and `GNC_WAVELET_LEVELS=6` was silently discarded along with it. Every past `--tile-size`
//! experiment ran under that hidden cap.
//!
//! `set_tile_size` re-derives the ceiling from the recorded request, in both directions.

use gnc::{quality_preset, CodecConfig};

#[test]
fn ceiling_follows_the_tile_size_in_use() {
    // 8 samples is the floor per level: 512 carries 6 levels, 256 carries 5, 128 carries 4.
    //
    // `(1024, 7)` was in this list until BUG-26. It asserted a level count for a tile size the
    // wavelet shader cannot transform at all — tile 1024 decodes to 7.5 dB with encode and decode
    // both reporting success — so the row was asserting the arithmetic of a configuration that
    // destroys the picture. `set_tile_size` now refuses it; see `refuses_a_tile_the_shader_cannot_transform`.
    for (tile, expect) in [(128u32, 4u32), (256, 5), (512, 6)] {
        let mut cfg = CodecConfig {
            wavelet_levels: 9,
            ..Default::default()
        };
        cfg.set_tile_size(tile);
        assert_eq!(
            cfg.wavelet_levels, expect,
            "tile {tile} should allow {expect} levels, got {}",
            cfg.wavelet_levels
        );
    }
}

#[test]
fn a_larger_tile_restores_levels_the_default_tile_clamped_away() {
    // The regression itself: the preset clamps 6 to 5 against the default 256 px tile, and the
    // CLI's larger tile then has to be able to give the level back.
    let mut cfg = quality_preset(90);
    cfg.requested_wavelet_levels = 6;
    cfg.set_tile_size(512);
    assert_eq!(
        cfg.wavelet_levels, 6,
        "512 px tile must reach the 6 levels it can carry"
    );

    // And a smaller tile must still clamp down.
    cfg.set_tile_size(128);
    assert_eq!(cfg.wavelet_levels, 4, "128 px tile carries only 4 levels");
}

#[test]
fn setting_levels_directly_is_still_honoured_as_a_request() {
    // requested_wavelet_levels defaults to 0 ("nothing recorded"), so a direct assignment to
    // wavelet_levels must not be read as a clamp to preserve.
    let mut cfg = CodecConfig {
        wavelet_levels: 6,
        ..Default::default()
    };
    cfg.set_tile_size(512);
    assert_eq!(cfg.wavelet_levels, 6);
}

#[test]
fn the_shipped_presets_are_unchanged() {
    // 5 levels at q >= 25, 4 below, on the default 256 px tile (BUG-6). This fix must not move
    // them — every measurement in BASELINE.md was taken here.
    //
    // q=100 is excluded: LOSSLESS-1 replaced the wavelet there with MED prediction, so it has no
    // levels to keep. That is asserted separately below rather than dropped.
    for (q, expect) in [(10u32, 4u32), (20, 4), (25, 5), (75, 5), (99, 5)] {
        let cfg = quality_preset(q);
        assert_eq!(cfg.tile_size, 256);
        assert_eq!(
            cfg.wavelet_levels, expect,
            "q={q} should keep {expect} levels"
        );
    }
}

#[test]
fn lossless_uses_med_prediction_not_the_wavelet() {
    // LOSSLESS-1: q=100 codes MED residuals directly. No wavelet means no subbands, so levels
    // must be 0 — the entropy coder groups by subband and would otherwise split flat data.
    let cfg = quality_preset(100);
    assert_eq!(cfg.transform_type, gnc::TransformType::MedPredict);
    assert_eq!(cfg.wavelet_levels, 0);
    assert!(
        cfg.is_lossless(),
        "the MED path must still count as lossless"
    );
    assert!(!cfg.adaptive_quantization);
    assert!(!cfg.cfl_enabled);
}

// ---------------------------------------------------------------------------
// BUG-26 — a tile size the pipeline cannot serve must be refused, not encoded
// ---------------------------------------------------------------------------
//
// Both cases below produced a *valid bitstream carrying a destroyed picture*, with `encode` and
// `decode` each exiting 0. Measured on kristensara_720p at q=90, before the guard:
//
//   tile  504 -> 40.97 dB      tile  520 -> 24.80 dB
//   tile  512 -> 49.66 dB      tile  640 -> 11.24 dB
//   tile  260 -> 20.11 dB      tile 1024 ->  7.50 dB
//
// They are refused rather than clamped on purpose: a clamp encodes something other than what the
// caller asked for, just as quietly.

#[test]
#[should_panic(expected = "outside the supported range")]
fn refuses_a_tile_the_shader_cannot_transform() {
    // transform_97.wgsl stages one tile line in `array<f32, 512>`, and WGSL out-of-bounds
    // workgroup access does not trap — it reads whatever is there.
    CodecConfig::default().set_tile_size(gnc::MAX_TILE_SIZE + 8);
}

#[test]
#[should_panic(expected = "not divisible by")]
fn refuses_a_tile_that_cannot_carry_its_own_levels() {
    // 260 = 4 x 65. `max_wavelet_levels` derives its ceiling from `tile_size / 8` under integer
    // division, so it hands 260 five levels; only two halvings are clean and the rest drop
    // coefficients.
    let mut cfg = CodecConfig {
        wavelet_levels: 9,
        ..Default::default()
    };
    cfg.set_tile_size(260);
}

#[test]
fn every_tile_size_that_encodes_correctly_is_still_accepted() {
    // The guard must not cost a working configuration. These are the sizes measured to round-trip
    // at 49.6-49.7 dB on kristensara_720p at q=90, including the non-power-of-two ones.
    for tile in [64u32, 96, 128, 160, 192, 256, 320, 384, 448, 512] {
        let mut cfg = CodecConfig {
            wavelet_levels: 9,
            ..Default::default()
        };
        cfg.set_tile_size(tile);
        assert_eq!(cfg.tile_size, tile);
        assert!(
            tile.is_multiple_of(1u32 << cfg.wavelet_levels),
            "tile {tile} accepted with {} levels but is not divisible by {}",
            cfg.wavelet_levels,
            1u32 << cfg.wavelet_levels
        );
    }
}
