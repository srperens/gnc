//! BUG-32: `benchmark-sequence --throughput` must not score SSIM or run the all-I arm.
//!
//! Default path unchanged: it still prints PSNR/SSIM and the I-only comparison.
//! The flag is the timing path; without it, 86% of wall clock was CPU metrics.

use std::path::Path;
use std::process::Command;

fn gnc() -> &'static str {
    env!("CARGO_BIN_EXE_gnc")
}

fn sequence_pattern() -> Option<&'static str> {
    let first = Path::new("test_material/frames/sequences/bbb_extended/frame_0000.png");
    first
        .exists()
        .then_some("test_material/frames/sequences/bbb_extended/frame_%04d.png")
}

#[test]
fn help_lists_throughput_and_refuses_vmaf() {
    let help = Command::new(gnc())
        .args(["benchmark-sequence", "--help"])
        .output()
        .expect("help");
    let stdout = String::from_utf8_lossy(&help.stdout);
    assert!(
        stdout.contains("--throughput"),
        "benchmark-sequence --help must list --throughput (BUG-32):\n{stdout}"
    );

    let conflict = Command::new(gnc())
        .args([
            "benchmark-sequence",
            "-i",
            "x.y4m",
            "--throughput",
            "--vmaf",
        ])
        .output()
        .expect("conflict");
    assert!(
        !conflict.status.success(),
        "--throughput --vmaf must fail: VMAF needs a decode"
    );
    let err = String::from_utf8_lossy(&conflict.stderr);
    assert!(
        err.contains("cannot be used with") || err.contains("conflicts"),
        "expected clap conflict, got:\n{err}"
    );
}

#[test]
fn throughput_skips_metrics_and_the_all_i_arm() {
    let Some(pattern) = sequence_pattern() else {
        eprintln!("skipping: test_material/frames/sequences/bbb_extended not present");
        return;
    };

    let default = Command::new(gnc())
        .args([
            "benchmark-sequence",
            "-i",
            pattern,
            "-n",
            "3",
            "-k",
            "2",
            "-q",
            "90",
        ])
        .output()
        .expect("default encode");
    assert!(
        default.status.success(),
        "default path failed:\n{}",
        String::from_utf8_lossy(&default.stderr)
    );
    let default_out = String::from_utf8_lossy(&default.stdout);
    assert!(
        default_out.contains("SSIM"),
        "default path must still print SSIM:\n{default_out}"
    );
    assert!(
        default_out.contains("All I-frames"),
        "default path must still run the all-I arm:\n{default_out}"
    );

    let thru = Command::new(gnc())
        .args([
            "benchmark-sequence",
            "-i",
            pattern,
            "-n",
            "3",
            "-k",
            "2",
            "-q",
            "90",
            "--throughput",
        ])
        .output()
        .expect("throughput encode");
    assert!(
        thru.status.success(),
        "throughput path failed:\n{}",
        String::from_utf8_lossy(&thru.stderr)
    );
    let thru_out = String::from_utf8_lossy(&thru.stdout);
    let thru_err = String::from_utf8_lossy(&thru.stderr);
    assert!(
        thru_err.contains("[bug32] throughput=1 metrics=0 i_only=0 decode_retained=0"),
        "canary missing from stderr:\n{thru_err}"
    );
    assert!(
        !thru_out.contains("SSIM"),
        "throughput path must not print SSIM:\n{thru_out}"
    );
    assert!(
        !thru_out.contains("All I-frames"),
        "throughput path must not run the all-I arm:\n{thru_out}"
    );
    assert!(
        thru_out.contains("fps"),
        "throughput path must still print encode fps:\n{thru_out}"
    );
}
