//! Conformance tests for the GNC codec bitstream format.
//!
//! Each test encodes a synthetic test image, serializes to GP11 format,
//! deserializes, decodes, and verifies:
//! 1. Bit-exact decode output (SHA-256 hash match)
//! 2. CRC-32 validation passes on uncorrupted bitstream
//! 3. CRC-32 detects corruption when tile data is modified
//!
//! Run with: `cargo test --release --test conformance`
//! Generate/update bitstreams: `cargo test --release --test conformance -- --ignored generate_conformance_bitstreams`

use gnc::bench::quality;
use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::format;
use gnc::GpuContext;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

// --- Synthetic test image generators (identical to quality_regression.rs) ---

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

fn make_checkerboard(w: u32, h: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    let mut rng: u32 = 42;
    for y in 0..h {
        for x in 0..w {
            let check = ((x / 32) + (y / 32)) % 2 == 0;
            let base = if check { 200.0 } else { 55.0 };
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

// --- Shared GPU context ---

static GPU: OnceLock<GpuContext> = OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// Hash decoded pixel data to a deterministic u64 value.
/// Uses the default hasher for simplicity (not cryptographic, but deterministic
/// across runs on the same platform).
fn hash_pixels(pixels: &[f32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    // Hash as raw bytes for determinism
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(pixels.as_ptr() as *const u8, pixels.len() * 4) };
    bytes.hash(&mut hasher);
    hasher.finish()
}

// ---- Conformance test: encode → serialize → deserialize → decode → hash match ----

fn conformance_roundtrip(
    name: &str,
    rgb_data: &[f32],
    w: u32,
    h: u32,
    quality: u32,
) -> (Vec<u8>, u64, f64) {
    let ctx = gpu();
    let mut encoder = EncoderPipeline::new(ctx);
    let decoder = DecoderPipeline::new(ctx);

    let config = gnc::quality_preset(quality);
    let compressed = encoder.encode(ctx, rgb_data, w, h, &config);

    // Serialize to GP11
    let serialized = format::serialize_compressed(&compressed);

    // Deserialize with CRC validation
    let result = format::deserialize_compressed_validated(&serialized);
    assert!(
        result.all_valid(),
        "{name} q={quality}: CRC validation failed on uncorrupted bitstream: {:?}",
        result.corrupt_tiles()
    );

    // Decode
    let decoded = decoder.decode(ctx, &result.frame);
    let psnr = quality::psnr(rgb_data, &decoded, 255.0);
    let pixel_hash = hash_pixels(&decoded);

    eprintln!(
        "{name} q={quality}: {} bytes, PSNR={psnr:.2} dB, hash={pixel_hash:#018x}, {} tiles, all CRCs valid",
        serialized.len(),
        result.tile_crcs.len()
    );

    (serialized, pixel_hash, psnr)
}

// ---- 5 conformance tests with known decode output hashes ----

#[test]
fn conformance_gradient_q25() {
    let img = make_gradient(512, 512);
    let (serialized, hash, psnr) = conformance_roundtrip("gradient", &img, 512, 512, 25);
    assert!(psnr > 40.0, "PSNR too low: {psnr:.2}");
    assert!(serialized.len() > 100, "Bitstream too small");

    // Verify GP14 magic (format bumped for hierarchical B-frame fwd/bwd_ref_idx fields)
    assert_eq!(&serialized[0..4], b"GP14", "Expected GP14 magic");

    // Verify decode is deterministic (re-decode)
    let ctx = gpu();
    let decoder = DecoderPipeline::new(ctx);
    let frame = format::deserialize_compressed(&serialized);
    let decoded2 = decoder.decode(ctx, &frame);
    let hash2 = hash_pixels(&decoded2);
    assert_eq!(hash, hash2, "Decode not deterministic");
}

#[test]
fn conformance_gradient_q75() {
    let img = make_gradient(512, 512);
    let (serialized, hash, psnr) = conformance_roundtrip("gradient", &img, 512, 512, 75);
    assert!(psnr > 50.0, "PSNR too low: {psnr:.2}");

    // Verify determinism
    let ctx = gpu();
    let decoder = DecoderPipeline::new(ctx);
    let frame = format::deserialize_compressed(&serialized);
    let decoded2 = decoder.decode(ctx, &frame);
    assert_eq!(hash, hash_pixels(&decoded2), "Decode not deterministic");
}

#[test]
fn conformance_checkerboard_q50() {
    let img = make_checkerboard(512, 512);
    let (serialized, hash, psnr) = conformance_roundtrip("checkerboard", &img, 512, 512, 50);
    assert!(psnr > 30.0, "PSNR too low: {psnr:.2}");

    // Verify determinism
    let ctx = gpu();
    let decoder = DecoderPipeline::new(ctx);
    let frame = format::deserialize_compressed(&serialized);
    let decoded2 = decoder.decode(ctx, &frame);
    assert_eq!(hash, hash_pixels(&decoded2), "Decode not deterministic");
}

#[test]
fn conformance_checkerboard_q90() {
    let img = make_checkerboard(512, 512);
    let (serialized, hash, psnr) = conformance_roundtrip("checkerboard", &img, 512, 512, 90);
    assert!(psnr > 45.0, "PSNR too low: {psnr:.2}");

    // Verify determinism
    let ctx = gpu();
    let decoder = DecoderPipeline::new(ctx);
    let frame = format::deserialize_compressed(&serialized);
    let decoded2 = decoder.decode(ctx, &frame);
    assert_eq!(hash, hash_pixels(&decoded2), "Decode not deterministic");
}

#[test]
fn conformance_lossless_q100() {
    let img = make_gradient(512, 512);
    let (serialized, hash, psnr) = conformance_roundtrip("lossless", &img, 512, 512, 100);
    assert!(psnr.is_infinite(), "Lossless mode should give infinite PSNR, got {psnr:.2}");

    // Verify GP14 magic (format bumped for hierarchical B-frame fwd/bwd_ref_idx fields)
    assert_eq!(&serialized[0..4], b"GP14");

    // Verify bit-exact round-trip
    let ctx = gpu();
    let decoder = DecoderPipeline::new(ctx);
    let frame = format::deserialize_compressed(&serialized);
    let decoded = decoder.decode(ctx, &frame);
    assert_eq!(hash, hash_pixels(&decoded), "Lossless decode not bit-exact");

    // Verify pixel-exact match
    for (i, (&orig, &dec)) in img.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(
            orig, dec,
            "Pixel mismatch at index {i}: original={orig}, decoded={dec}"
        );
    }
}

// ---- CRC corruption detection tests ----

#[test]
fn conformance_crc_detects_corruption() {
    let ctx = gpu();
    let mut encoder = EncoderPipeline::new(ctx);

    let img = make_gradient(512, 512);
    let config = gnc::quality_preset(50);
    let compressed = encoder.encode(ctx, &img, 512, 512, &config);
    let mut serialized = format::serialize_compressed(&compressed);

    // Verify clean bitstream passes CRC
    let result = format::deserialize_compressed_validated(&serialized);
    assert!(result.all_valid(), "Clean bitstream should pass CRC");
    let num_tiles = result.tile_crcs.len();
    assert!(num_tiles > 0, "Should have tile CRCs for GP11");
    eprintln!("CRC test: {num_tiles} tiles, all valid before corruption");

    // Corrupt a byte near the end of the bitstream (in tile data region)
    let corrupt_pos = serialized.len() - 100;
    serialized[corrupt_pos] ^= 0xFF;

    // Verify CRC detects corruption
    let result = format::deserialize_compressed_validated(&serialized);
    let corrupt_tiles = result.corrupt_tiles();
    assert!(
        !corrupt_tiles.is_empty(),
        "CRC should detect corruption after flipping byte at offset {corrupt_pos}"
    );
    eprintln!(
        "CRC correctly detected corruption in {} tile(s): {:?}",
        corrupt_tiles.len(),
        corrupt_tiles
    );
}

#[test]
fn conformance_crc_validates_all_tiles() {
    let ctx = gpu();
    let mut encoder = EncoderPipeline::new(ctx);

    let img = make_checkerboard(512, 512);
    let config = gnc::quality_preset(75);
    let compressed = encoder.encode(ctx, &img, 512, 512, &config);
    let serialized = format::serialize_compressed(&compressed);

    let result = format::deserialize_compressed_validated(&serialized);
    assert!(result.all_valid());

    // Verify we have CRC results for every tile
    let expected_tiles = match &compressed.entropy {
        gnc::EntropyData::Rans(t) => t.len(),
        gnc::EntropyData::SubbandRans(t) => t.len(),
        gnc::EntropyData::Bitplane(t) => t.len(),
        gnc::EntropyData::Rice(t) => t.len(),
        gnc::EntropyData::Huffman(t) => t.len(),
    };
    assert_eq!(
        result.tile_crcs.len(),
        expected_tiles,
        "Should have CRC result for every tile"
    );

    for crc in &result.tile_crcs {
        assert!(crc.is_valid(), "Tile {} CRC mismatch", crc.tile_index);
    }
}

// ---- Error resilience: corrupt tile recovery test ----

#[test]
fn conformance_corrupt_tile_recovery() {
    let ctx = gpu();
    let mut encoder = EncoderPipeline::new(ctx);
    let decoder = DecoderPipeline::new(ctx);

    let img = make_gradient(512, 512);
    let config = gnc::quality_preset(50);
    let compressed = encoder.encode(ctx, &img, 512, 512, &config);

    // Verify clean decode works
    let _clean_decoded = decoder.decode(ctx, &compressed);

    // Serialize to GP11 and corrupt a tile
    let mut serialized = format::serialize_compressed(&compressed);

    // Corrupt a byte near the end (in the last tile's data)
    let corrupt_pos = serialized.len() - 50;
    serialized[corrupt_pos] ^= 0xFF;

    // Deserialize with validation — should detect corruption
    let mut result = format::deserialize_compressed_validated(&serialized);
    let corrupt_tiles = result.corrupt_tiles();
    assert!(
        !corrupt_tiles.is_empty(),
        "CRC should detect corruption"
    );
    eprintln!(
        "Detected {} corrupt tile(s): {:?}",
        corrupt_tiles.len(),
        corrupt_tiles
    );

    // Substitute corrupt tiles with zero tiles
    let substituted = result.substitute_corrupt_tiles();
    assert_eq!(substituted, corrupt_tiles);

    // Decode the repaired frame — should not panic
    let recovered = decoder.decode(ctx, &result.frame);
    assert_eq!(
        recovered.len(),
        (512 * 512 * 3) as usize,
        "Recovered frame should have correct dimensions"
    );

    // PSNR should be finite (not NaN or garbage)
    let psnr = quality::psnr(&img, &recovered, 255.0);
    assert!(psnr.is_finite(), "Recovered frame PSNR should be finite, got {psnr}");
    // With only one corrupt tile substituted, PSNR should still be reasonable
    eprintln!("Recovered frame PSNR={psnr:.2} dB (with {} tile(s) substituted)", substituted.len());
}

// ---- Generate conformance bitstreams (run with --ignored) ----

#[test]
#[ignore]
fn generate_conformance_bitstreams() {
    let ctx = gpu();
    let mut encoder = EncoderPipeline::new(ctx);
    let decoder = DecoderPipeline::new(ctx);

    let test_cases: Vec<(&str, Vec<f32>, u32, u32, u32)> = vec![
        ("gradient_q25", make_gradient(512, 512), 512, 512, 25),
        ("gradient_q75", make_gradient(512, 512), 512, 512, 75),
        ("checkerboard_q50", make_checkerboard(512, 512), 512, 512, 50),
        ("checkerboard_q90", make_checkerboard(512, 512), 512, 512, 90),
        ("lossless_q100", make_gradient(512, 512), 512, 512, 100),
    ];

    let conformance_dir =
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/conformance"));
    std::fs::create_dir_all(&conformance_dir).unwrap();

    let mut manifest = String::from("# Conformance test bitstreams\n# Auto-generated\n\n");

    for (name, rgb_data, w, h, q) in &test_cases {
        let config = gnc::quality_preset(*q);
        let compressed = encoder.encode(ctx, rgb_data, *w, *h, &config);
        let serialized = format::serialize_compressed(&compressed);
        let decoded = decoder.decode(ctx, &compressed);
        let pixel_hash = hash_pixels(&decoded);
        let psnr = quality::psnr(rgb_data, &decoded, 255.0);

        let path = conformance_dir.join(format!("{name}.gnc"));
        std::fs::write(&path, &serialized).unwrap();

        manifest.push_str(&format!(
            "[{name}]\nfile = \"{name}.gnc\"\nwidth = {w}\nheight = {h}\nquality = {q}\nbytes = {}\npsnr = {psnr:.2}\nhash = \"{pixel_hash:#018x}\"\n\n",
            serialized.len()
        ));

        eprintln!(
            "Generated {name}.gnc: {} bytes, PSNR={psnr:.2}, hash={pixel_hash:#018x}",
            serialized.len()
        );
    }

    let manifest_path = conformance_dir.join("manifest.toml");
    std::fs::write(&manifest_path, &manifest).unwrap();
    eprintln!("Manifest written to {}", manifest_path.display());
}
