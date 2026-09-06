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
    for (tile, expect) in [(128u32, 4u32), (256, 5), (512, 6), (1024, 7)] {
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
    for (q, expect) in [(10u32, 4u32), (20, 4), (25, 5), (75, 5), (100, 5)] {
        let cfg = quality_preset(q);
        assert_eq!(cfg.tile_size, 256);
        assert_eq!(
            cfg.wavelet_levels, expect,
            "q={q} should keep {expect} levels"
        );
    }
}
