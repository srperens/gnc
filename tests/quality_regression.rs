//! Golden-baseline regression tests for image quality.
//!
//! Encode-decode synthetic test images at q=25/50/75/90 and assert that
//! PSNR and bpp stay within tolerance of stored baselines. If a code change
//! regresses quality, these tests fail.
//!
//! Run with: `cargo test --release --test quality_regression`
//! Update baselines: `cargo test --release --test quality_regression -- --ignored update_golden_baselines`

use gnc::bench::quality;
use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::GpuContext;
use std::collections::HashMap;
use std::sync::OnceLock;

// --- Synthetic test image generators ---

/// Smooth gradient with color variation (easy content).
/// Values are rounded to integers (0-255) to match real image behavior.
fn make_gradient(w: u32, h: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (x as f32 / w as f32 * 255.0).round().clamp(0.0, 255.0);
            let g = (y as f32 / h as f32 * 255.0).round().clamp(0.0, 255.0);
            let b = ((x + y) as f32 / (w + h) as f32 * 255.0).round().clamp(0.0, 255.0);
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

/// Checkerboard pattern with noise (hard content for wavelet codecs).
/// Values are rounded to integers (0-255) to match real image behavior.
fn make_checkerboard(w: u32, h: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    // Simple LCG for deterministic "noise"
    let mut rng: u32 = 42;
    for y in 0..h {
        for x in 0..w {
            let check = ((x / 32) + (y / 32)) % 2 == 0;
            let base = if check { 200.0 } else { 55.0 };
            // Deterministic noise
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((rng >> 16) as f32 / 65536.0 - 0.5) * 20.0;
            let val = (base + noise).clamp(0.0, 255.0).round();
            data.push(val);
            data.push((val * 0.9).round().clamp(0.0, 255.0));
            data.push((val * 0.8).round().clamp(0.0, 255.0));
        }
    }
    data
}

// --- Baseline loading ---

#[derive(Debug)]
struct Baseline {
    psnr_min: f64,
    bpp_max: f64,
}

fn load_baselines() -> HashMap<String, Baseline> {
    let toml_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden_baselines.toml");
    let content = std::fs::read_to_string(toml_path).expect("Failed to read golden_baselines.toml");
    let table: toml::Table = content
        .parse()
        .expect("Failed to parse golden_baselines.toml");

    let mut baselines = HashMap::new();
    for (image_key, value) in &table {
        if let toml::Value::Table(quality_levels) = value {
            for (q_key, qval) in quality_levels {
                if let toml::Value::Table(metrics) = qval {
                    let psnr_min = metrics
                        .get("psnr_min")
                        .and_then(|v| v.as_float())
                        .unwrap_or(0.0);
                    let bpp_max = metrics
                        .get("bpp_max")
                        .and_then(|v| v.as_float())
                        .unwrap_or(100.0);
                    let key = format!("{image_key}.{q_key}");
                    baselines.insert(key, Baseline { psnr_min, bpp_max });
                }
            }
        }
    }
    baselines
}

// --- Shared GPU context (expensive to create) ---

static GPU: OnceLock<GpuContext> = OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

// --- Core test helper ---

struct TestResult {
    psnr: f64,
    ssim: f64,
    bpp: f64,
}

fn encode_decode_measure(
    ctx: &GpuContext,
    rgb_data: &[f32],
    w: u32,
    h: u32,
    quality: u32,
) -> TestResult {
    let mut encoder = EncoderPipeline::new(ctx);
    let decoder = DecoderPipeline::new(ctx);

    let config = gnc::quality_preset(quality);
    let compressed = encoder.encode(ctx, rgb_data, w, h, &config);
    let reconstructed = decoder.decode(ctx, &compressed);

    let psnr = quality::psnr(rgb_data, &reconstructed, 255.0);
    let ssim = quality::ssim_approx(rgb_data, &reconstructed, 255.0);
    let bpp = compressed.bpp();

    TestResult { psnr, ssim, bpp }
}

fn assert_baseline(key: &str, result: &TestResult, baselines: &HashMap<String, Baseline>) {
    let baseline = baselines
        .get(key)
        .unwrap_or_else(|| panic!("Missing baseline for {key}"));

    // Allow 0.5 dB PSNR tolerance below baseline
    assert!(
        result.psnr >= baseline.psnr_min - 0.5,
        "{key}: PSNR {:.2} dB below minimum {:.2} dB (tolerance 0.5 dB)",
        result.psnr,
        baseline.psnr_min
    );

    // Allow 5% bpp tolerance above baseline
    assert!(
        result.bpp <= baseline.bpp_max * 1.05,
        "{key}: bpp {:.4} exceeds maximum {:.4} (+5% tolerance)",
        result.bpp,
        baseline.bpp_max
    );
}

// --- Gradient tests (4 quality levels) ---

#[test]
fn regression_gradient_q25() {
    let ctx = gpu();
    let baselines = load_baselines();
    let img = make_gradient(512, 512);
    let result = encode_decode_measure(ctx, &img, 512, 512, 25);
    eprintln!(
        "gradient q25: PSNR={:.2} SSIM={:.4} bpp={:.4}",
        result.psnr, result.ssim, result.bpp
    );
    assert_baseline("gradient_512x512.q25", &result, &baselines);
}

#[test]
fn regression_gradient_q50() {
    let ctx = gpu();
    let baselines = load_baselines();
    let img = make_gradient(512, 512);
    let result = encode_decode_measure(ctx, &img, 512, 512, 50);
    eprintln!(
        "gradient q50: PSNR={:.2} SSIM={:.4} bpp={:.4}",
        result.psnr, result.ssim, result.bpp
    );
    assert_baseline("gradient_512x512.q50", &result, &baselines);
}

#[test]
fn regression_gradient_q75() {
    let ctx = gpu();
    let baselines = load_baselines();
    let img = make_gradient(512, 512);
    let result = encode_decode_measure(ctx, &img, 512, 512, 75);
    eprintln!(
        "gradient q75: PSNR={:.2} SSIM={:.4} bpp={:.4}",
        result.psnr, result.ssim, result.bpp
    );
    assert_baseline("gradient_512x512.q75", &result, &baselines);
}

#[test]
fn regression_gradient_q90() {
    let ctx = gpu();
    let baselines = load_baselines();
    let img = make_gradient(512, 512);
    let result = encode_decode_measure(ctx, &img, 512, 512, 90);
    eprintln!(
        "gradient q90: PSNR={:.2} SSIM={:.4} bpp={:.4}",
        result.psnr, result.ssim, result.bpp
    );
    assert_baseline("gradient_512x512.q90", &result, &baselines);
}

// --- Checkerboard tests (4 quality levels) ---

#[test]
fn regression_checkerboard_q25() {
    let ctx = gpu();
    let baselines = load_baselines();
    let img = make_checkerboard(512, 512);
    let result = encode_decode_measure(ctx, &img, 512, 512, 25);
    eprintln!(
        "checkerboard q25: PSNR={:.2} SSIM={:.4} bpp={:.4}",
        result.psnr, result.ssim, result.bpp
    );
    assert_baseline("checkerboard_512x512.q25", &result, &baselines);
}

#[test]
fn regression_checkerboard_q50() {
    let ctx = gpu();
    let baselines = load_baselines();
    let img = make_checkerboard(512, 512);
    let result = encode_decode_measure(ctx, &img, 512, 512, 50);
    eprintln!(
        "checkerboard q50: PSNR={:.2} SSIM={:.4} bpp={:.4}",
        result.psnr, result.ssim, result.bpp
    );
    assert_baseline("checkerboard_512x512.q50", &result, &baselines);
}

#[test]
fn regression_checkerboard_q75() {
    let ctx = gpu();
    let baselines = load_baselines();
    let img = make_checkerboard(512, 512);
    let result = encode_decode_measure(ctx, &img, 512, 512, 75);
    eprintln!(
        "checkerboard q75: PSNR={:.2} SSIM={:.4} bpp={:.4}",
        result.psnr, result.ssim, result.bpp
    );
    assert_baseline("checkerboard_512x512.q75", &result, &baselines);
}

#[test]
fn regression_checkerboard_q90() {
    let ctx = gpu();
    let baselines = load_baselines();
    let img = make_checkerboard(512, 512);
    let result = encode_decode_measure(ctx, &img, 512, 512, 90);
    eprintln!(
        "checkerboard q90: PSNR={:.2} SSIM={:.4} bpp={:.4}",
        result.psnr, result.ssim, result.bpp
    );
    assert_baseline("checkerboard_512x512.q90", &result, &baselines);
}

// --- Quality monotonicity test (coarse) ---

#[test]
fn regression_quality_monotonicity() {
    let ctx = gpu();
    let img = make_gradient(512, 512);

    let mut prev_psnr = 0.0;
    let mut prev_bpp = 0.0;

    for q in [10, 25, 50, 75, 90] {
        let result = encode_decode_measure(ctx, &img, 512, 512, q);
        eprintln!(
            "monotonicity q={q}: PSNR={:.2} bpp={:.4}",
            result.psnr, result.bpp
        );

        assert!(
            result.psnr >= prev_psnr - 0.1,
            "PSNR not monotonic: q={q} PSNR={:.2} < prev={:.2}",
            result.psnr,
            prev_psnr
        );
        assert!(
            result.bpp >= prev_bpp - 0.01,
            "bpp not monotonic: q={q} bpp={:.4} < prev={:.4}",
            result.bpp,
            prev_bpp
        );
        prev_psnr = result.psnr;
        prev_bpp = result.bpp;
    }
}

// --- Extended PSNR monotonicity test (step=5, two image types) ---
// Verifies PSNR monotonically increases at q=5,10,...,95,100 on gradient
// and checkerboard. bpp monotonicity is image-dependent (wavelet level changes
// improve efficiency on smooth content) and verified on natural images via
// `gnc rd-curve --q-values 1..100`.

#[test]
fn regression_quality_monotonicity_extended() {
    let ctx = gpu();

    for (name, img) in [("gradient", make_gradient(512, 512)), ("checker", make_checkerboard(512, 512))] {
        let mut prev_psnr = 0.0f64;
        let mut prev_q = 0u32;

        for q in (5..=95).step_by(5).chain(std::iter::once(100)) {
            let result = encode_decode_measure(ctx, &img, 512, 512, q);
            eprintln!(
                "monotonicity {name} q={q}: PSNR={:.2} bpp={:.4}",
                result.psnr, result.bpp
            );

            // inf PSNR (lossless) is always >= any finite value
            if !result.psnr.is_infinite() {
                assert!(
                    result.psnr >= prev_psnr - 0.5,
                    "{name}: PSNR not monotonic: q={q} PSNR={:.2} < prev q={prev_q} PSNR={:.2}",
                    result.psnr,
                    prev_psnr
                );
            }
            prev_psnr = result.psnr;
            prev_q = q;
        }
        let _ = prev_q;
    }
}

// --- Serialization round-trip test ---

#[test]
fn regression_serialize_roundtrip() {
    let ctx = gpu();
    let img = make_gradient(512, 512);

    let mut encoder = EncoderPipeline::new(ctx);
    let decoder = DecoderPipeline::new(ctx);

    let config = gnc::quality_preset(75);
    let compressed = encoder.encode(ctx, &img, 512, 512, &config);

    // Serialize and deserialize
    let serialized = gnc::format::serialize_compressed(&compressed);
    let deserialized = gnc::format::deserialize_compressed(&serialized);

    // Decode the deserialized frame
    let decoded = decoder.decode(ctx, &deserialized);
    let psnr = quality::psnr(&img, &decoded, 255.0);

    eprintln!(
        "serialize roundtrip: PSNR={psnr:.2} dB, serialized size={} bytes",
        serialized.len()
    );
    assert!(psnr > 30.0, "Serialize round-trip PSNR too low: {psnr:.2}");
}

// --- Baseline updater (run with --ignored) ---

#[test]
#[ignore]
fn update_golden_baselines() {
    let ctx = gpu();

    let images: Vec<(&str, Vec<f32>, u32, u32)> = vec![
        ("gradient_512x512", make_gradient(512, 512), 512, 512),
        (
            "checkerboard_512x512",
            make_checkerboard(512, 512),
            512,
            512,
        ),
    ];

    let quality_levels = [25, 50, 75, 90];
    let toml_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden_baselines.toml");

    let mut output = String::from(
        "# Golden baseline values for regression testing.\n\
         # Auto-generated by: cargo test --release --test quality_regression -- --ignored update_golden_baselines\n\
         # Any change that regresses these metrics (beyond tolerance) fails CI.\n\
         #\n\
         # Tolerances: PSNR must not drop more than 0.5 dB; bpp must not increase more than 5%.\n\n",
    );

    for (name, rgb_data, w, h) in &images {
        for q in quality_levels {
            let result = encode_decode_measure(ctx, rgb_data, *w, *h, q);
            eprintln!(
                "{name} q={q}: PSNR={:.2} SSIM={:.4} bpp={:.4}",
                result.psnr, result.ssim, result.bpp
            );
            output.push_str(&format!(
                "[{name}.q{q}]\npsnr_min = {:.2}\nbpp_max = {:.4}\n\n",
                result.psnr, result.bpp
            ));
        }
    }

    std::fs::write(toml_path, &output).expect("Failed to write golden_baselines.toml");
    eprintln!("Updated baselines written to {toml_path}");
}
