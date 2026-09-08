//! The dead zone is an intra lever, and a referenced I-frame must not pay for it (INTRA-2).
//!
//! Two facts make this worth a test rather than a comment.
//!
//! **The lever is real and it was dormant.** GNC quantises as `floor(|v|/step + 0.5)` after a
//! `|v| < dead_zone * step` test, so any dead zone at or below 0.5 changes nothing — and the
//! shipped ladder interpolated 0.5 at q=85 down to 0.0 at q>=96, which means GNC had no dead zone
//! at all in its own operating range. Raising it to 0.6 over q=85..95 is worth **BD-rate −3.3% to
//! −7.0% on four stills, mean −5.0%**.
//!
//! **And on a P-chain it is a bad trade by construction.** At ki=9 an I-frame is one frame in
//! sixteen, so its rate saving is diluted to 0.05–0.33% of the sequence, while worst-frame PSNR —
//! the metric a contribution codec is judged on — is *fully* exposed to it, because the worst
//! frame **is** the I-frame. Measured on three sequences: −0.19 dB of worst-frame for −0.33% of
//! rate at q=90. A referenced I-frame has to be better than its own rate suggests, not worse.
//!
//! So `dead_zone` is what an unreferenced intra frame uses and `dead_zone_referenced` is what an
//! I-frame inside a chain and the inter residual path use. Same shape as PAD-1's `pad_fill_decay`
//! (decision `0039`), reached from a different lever.

use gnc::quality_preset;

#[test]
fn the_dead_zone_lever_exists_over_the_range_it_was_measured_in() {
    for q in 85..=95 {
        let cfg = quality_preset(q);
        assert!(
            cfg.dead_zone > 0.5,
            "q={q}: dead_zone {} is at or below 0.5, which the quantiser treats as no dead zone \
             at all — the lever is dormant again",
            cfg.dead_zone
        );
        assert!(
            cfg.dead_zone >= cfg.dead_zone_referenced,
            "q={q}: a referenced frame is being given a *wider* dead zone ({}) than an \
             unreferenced one ({}), which is backwards",
            cfg.dead_zone_referenced,
            cfg.dead_zone
        );
    }
}

#[test]
fn a_referenced_frame_keeps_the_ladders_own_dead_zone() {
    // q=90 is the middle of the measured range and the point the sequence regression was largest.
    let cfg = quality_preset(90);
    assert!(
        cfg.dead_zone_referenced <= 0.5,
        "q=90: dead_zone_referenced is {}, above the 0.5 the quantiser ignores — raising the \
         intra floor has leaked into referenced frames, which is exactly what makes a P-chain \
         lose worst-frame PSNR",
        cfg.dead_zone_referenced
    );
    assert!(
        cfg.dead_zone > cfg.dead_zone_referenced,
        "q=90: the two are equal ({}), so either the floor is gone or the split is not wired",
        cfg.dead_zone
    );
}

#[test]
fn the_floor_is_confined_to_the_measured_range() {
    // Below 85 the ladder is already above 0.5 and active, so the floor must not move it.
    for q in [1u32, 25, 50, 75, 80] {
        let cfg = quality_preset(q);
        assert_eq!(
            cfg.dead_zone, cfg.dead_zone_referenced,
            "q={q}: the floor reached below the range it was measured in"
        );
    }
    // Above 95 RATE-2 codes both ways and keeps the smaller, and three of four stills go
    // bit-exact there, so the lever is mostly inert and there is no measurement behind it.
    for q in [96u32, 97, 99] {
        let cfg = quality_preset(q);
        assert_eq!(
            cfg.dead_zone, cfg.dead_zone_referenced,
            "q={q}: the floor reached above the range it was measured in"
        );
    }
}

/// BUG-30: a dead zone at q=100 silently defeats bit-exact lossless. Both fields must be clear.
#[test]
fn lossless_intent_clears_both_dead_zones() {
    let cfg = quality_preset(100);
    assert_eq!(cfg.dead_zone, 0.0, "q=100 carries an intra dead zone (BUG-30)");
    assert_eq!(
        cfg.dead_zone_referenced, 0.0,
        "q=100 carries a referenced dead zone — the new field reopened BUG-30 through the back door"
    );
}
