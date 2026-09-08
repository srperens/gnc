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
