//! No temp path may be a fixed string, because eight sessions share one `TMPDIR` (BUG-36).
//!
//! COORDINATION.md's working mode is eight concurrent Claude sessions on one machine. They share
//! a user, therefore they share `std::env::temp_dir()`. A fixed temp filename is then a
//! cross-session data race rather than an untidiness: two `--vmaf` runs of the same subcommand
//! open the same reference and distorted Y4M, and each scores whichever frames won the race.
//!
//! What makes it worth a test rather than a fix is *how* it fails. There is no error, no warning
//! and no implausible number — just a believable VMAF score that belongs to someone else's clip.
//! Measured 2026-09-08, `benchmark-sequence --vmaf`, 9 frames, q=75, on two sequences whose
//! serial scores are bit-stable across repeated runs at 97.39 and 95.91:
//!
//! | run | old_town_cross | bbb_extended |
//! |---|---|---|
//! | serial, twice | 97.39 | 95.91 |
//! | concurrent, twice | 97.39 | **97.19** (+1.28) |
//! | concurrent, once | **96.37** (-1.02) | 95.91 |
//!
//! Exactly one of the pair is wrong in every concurrent run, in whichever direction the race
//! decided, by 2-2.5x the >0.5-point VMAF move CLAUDE.md calls a BLOCK. VMAF is the lead metric
//! at q<=85, so this is the project's primary number being quietly replaced by another session's.
//!
//! The guard is a source scan and not a runtime assertion on purpose: the defect is *writing* a
//! literal path, and it costs nothing to catch that at the point someone reintroduces it.

use std::path::{Path, PathBuf};

/// Every `.rs` file under `src/`.
fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read_dir src/") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_fixed_temp_paths_outside_the_helper() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);
    assert!(!files.is_empty(), "found no Rust sources under {}", src.display());

    // `session_temp_path` is the one legitimate caller: it is what everything else goes through.
    let helper = src.join("lib.rs");
    let mut offenders = Vec::new();

    for path in &files {
        let text = std::fs::read_to_string(path).expect("read source");
        for (i, line) in text.lines().enumerate() {
            if !line.contains("temp_dir()") {
                continue;
            }
            // The helper's own single use, which appends the process suffix.
            if *path == helper && line.contains("join(stamped)") {
                continue;
            }
            offenders.push(format!(
                "{}:{}: {}",
                path.strip_prefix(&src).unwrap_or(path).display(),
                i + 1,
                line.trim()
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "temp paths must go through gnc::session_temp_path() so concurrent sessions do not \
         share them (BUG-36). Offending lines:\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn session_temp_path_is_unique_per_process_and_keeps_the_extension() {
    let a = gnc::session_temp_path("gnc_probe_ref.y4m");

    // The extension survives, because ffmpeg and vmaf both dispatch on it.
    assert_eq!(
        a.extension().and_then(|e| e.to_str()),
        Some("y4m"),
        "extension must be preserved: {}",
        a.display()
    );

    // The stem carries this process's id, which is what makes it unique: two live processes
    // cannot share one.
    let stem = a.file_stem().and_then(|s| s.to_str()).expect("stem");
    let pid = std::process::id();
    assert!(
        stem.ends_with(&format!("_p{pid}")),
        "stem {stem} should end with _p{pid}"
    );
    assert!(stem.starts_with("gnc_probe_ref"), "stem {stem} lost its name");

    // Distinct names still map to distinct paths.
    let b = gnc::session_temp_path("gnc_probe_dist.y4m");
    assert_ne!(a, b);

    // An extensionless name (the j2k compare directory) is handled too.
    let dir = gnc::session_temp_path("gnc_probe_dir");
    assert!(
        dir.file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s == format!("gnc_probe_dir_p{pid}")),
        "extensionless name mishandled: {}",
        dir.display()
    );
}
