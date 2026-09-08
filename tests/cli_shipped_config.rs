//! The CLI must never build a `CodecConfig` that bypasses GNC's shipped policy (BUG-37).
//!
//! There are two legitimate answers to "is the B-pyramid allowed", and that is the whole trap.
//! `CodecConfig::default()` says `true` — "do not veto" — for library callers and for
//! `encoder::pipeline_tests`, which builds a `Default` config precisely to exercise the B-frame
//! path. GNC *as shipped* says `false` since 2026-09-06, on BUG-5's rate measurement and MEAS-6's
//! 160 ms of reordering.
//!
//! The defect was that only one caller knew the shipped answer. The veto lived inline in
//! `quality_preset()`, so `build_ip_config` and four sibling sites in `main.rs` fell through to
//! `CodecConfig::default()` whenever `-q` was absent — and `benchmark-sequence`'s `-q` is an
//! `Option` with no default, where `benchmark`, `encode-sequence` and `benchmark-suite` all
//! default to 75. So `benchmark-sequence -k 9` coded `2I+2P+14B` and the same command with
//! `-q 75` coded `2I+16P+0B`: the flag named *quality* silently selected the *GOP structure*.
//!
//! Patching the one `default_value` would have closed the instance and left the mechanism, with
//! three more sequence sites already diverging and nothing stopping a sixth. So the invariant is
//! structural instead: **`main.rs` constructs no config from `Default::default()`.** Everything
//! goes through `quality_preset` or `manual_config`, and both ask `b_pyramid_enabled()`.
//!
//! The same invariant now covers the **inter dead zone** (INTER-2, `docs/decisions/0041`), which
//! had the identical shape before it was consolidated: `GNC_INTER_DZ_MUL ... unwrap_or(2.0)`
//! written out at three separate sites in `sequence.rs` — the P-frame path and two B-frame
//! paths. Changing the default at two of three would have been a silent, frame-type-dependent
//! quantiser difference, which is BUG-16 all over again.

use std::path::Path;

#[test]
fn the_cli_never_builds_a_config_from_default() {
    let main_rs = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/main.rs");
    let text = std::fs::read_to_string(&main_rs).expect("read src/main.rs");

    let offenders: Vec<String> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("Default::default()"))
        .map(|(i, l)| format!("src/main.rs:{}: {}", i + 1, l.trim()))
        .collect();

    assert!(
        offenders.is_empty(),
        "the CLI must build configs with gnc::quality_preset() or gnc::manual_config(), never \
         from Default::default() — that is the path that bypasses the shipped B-pyramid policy \
         (BUG-37). Offending lines:\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn every_shipped_entry_point_agrees_on_the_pyramid() {
    let shipped = gnc::b_pyramid_enabled();

    // The manual path (no -q) and the preset path (-q) must not disagree. This is the exact
    // comparison that failed before: manual said true, preset said false.
    assert_eq!(
        gnc::manual_config(4.0).b_pyramid,
        shipped,
        "manual_config() disagrees with the shipped policy"
    );

    for q in [1, 20, 50, 75, 85, 90, 100] {
        assert_eq!(
            gnc::quality_preset(q).b_pyramid,
            shipped,
            "quality_preset({q}) disagrees with the shipped policy"
        );
    }
}

#[test]
fn manual_config_keeps_the_qstep_it_was_given() {
    // manual_config is the no-preset path, so the quantiser is the caller's whole input; a
    // regression here would silently re-quantise every `--qstep` run.
    for qstep in [1.0_f32, 4.0, 12.5] {
        assert_eq!(gnc::manual_config(qstep).quantization_step, qstep);
    }
}

#[test]
fn the_library_default_still_permits_b_frames() {
    // Not redundant with the above, and not a duplicate of the shipped policy: this asserts the
    // *other* answer deliberately. `encoder::pipeline_tests` constructs a Default config to
    // exercise the B-frame path, so flipping this would leave those tests passing while silently
    // testing P-only coding. If someone ever does flip it, this failing test is the place that
    // explains why that is the wrong fix for BUG-37.
    assert!(
        gnc::CodecConfig::default().b_pyramid,
        "CodecConfig::default() must keep b_pyramid = true (do not veto); the shipped veto \
         belongs in b_pyramid_enabled(), which quality_preset() and manual_config() apply"
    );
}

/// The inter dead-zone factor must have exactly one source, and it must be 1.0.
///
/// 2.0 was the shipped value until 2026-09-08 and measured as the wrong number: BD-rate on PSNR
/// −4.77% mean for 1.0 across three sequences on a 4-rung ladder, with worst-frame PSNR — the
/// contribution metric — better at 12 of 12 points. 1.0 beats both neighbours (1.5 gives −2.82%,
/// 0.0 gives +2.25% and is *worse* than shipped on animation), so the optimum is bracketed.
#[test]
fn the_inter_dead_zone_has_one_source_and_it_is_one() {
    assert_eq!(
        gnc::inter_dead_zone_mul(),
        1.0,
        "the inter dead zone is the intra dead zone (INTER-2); 2.0 measured worse on every \
         sequence and at every worst-frame point"
    );

    // No site may re-read the variable. Three inline copies is what this replaced.
    let seq = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/encoder/sequence.rs");
    let text = std::fs::read_to_string(&seq).expect("read sequence.rs");
    let offenders: Vec<String> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("GNC_INTER_DZ_MUL"))
        .map(|(i, l)| format!("src/encoder/sequence.rs:{}: {}", i + 1, l.trim()))
        .collect();
    assert!(
        offenders.is_empty(),
        "the inter dead-zone factor must come from gnc::inter_dead_zone_mul(), not from a second \
         read of the environment — it lived at three sites before INTER-2. Offenders:\n  {}",
        offenders.join("\n  ")
    );
}

/// A dead zone of 0.5 or less is arithmetically a no-op, which is why this change stops at q~88.
///
/// GNC quantises as `floor(|v|/step + 0.5)` *after* a `|v| < dz*step` test, so for `dz <= 0.5`
/// the test only zeroes values the rounding would have zeroed anyway. This is the property the
/// whole q-boundary argument rests on, and it is cheap to assert directly rather than trust.
#[test]
fn a_dead_zone_of_half_a_step_changes_nothing() {
    let step = 2.8_f32;
    for i in 0..4000 {
        let v = i as f32 * step / 1000.0;
        let plain = (v / step + 0.5).floor();
        for dz in [0.0_f32, 0.25, 0.5] {
            let gated = if v < dz * step { 0.0 } else { (v / step + 0.5).floor() };
            assert_eq!(
                gated, plain,
                "dz={dz} changed the quantiser at |v|={v}: {gated} vs {plain}"
            );
        }
    }
    // And 0.75 — the anchor value at q<=75 — genuinely is not a no-op, or the item is vacuous.
    let v = 0.6 * step;
    assert_ne!(
        if v < 0.75 * step { 0.0 } else { (v / step + 0.5).floor() },
        (v / step + 0.5).floor(),
        "dz=0.75 should zero a value the plain quantiser keeps"
    );
}
