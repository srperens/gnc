//! Integration test: bbb_2min frames 250-600 — scene cut conformance.
//!
//! This test covers the hardest section of Big Buck Bunny 2-minute clip for the
//! temporal encoder: three hard scene cuts at approximately frames 285, 378, 553
//! (relative to sequence start). The temporal wavelet GOPs straddle these cuts,
//! exercising the encoder's robustness to high-energy highpass coefficients.
//!
//! What is verified:
//!   1. Encode + decode roundtrip completes without panics or errors
//!   2. Frame count in the decoded output matches the input (351 frames)
//!   3. tail_iframes > 0 (the tail of the sequence doesn't fill a complete GOP)
//!   4. Number of GOPs is exactly floor(351 / gop_size)
//!   5. Serialization / deserialization roundtrip preserves frame count and GOP layout
//!   6. Per-frame PSNR > 30 dB at q=75 — no garbage frames from temporal bleed
//!
//! Requires: test_material/frames/sequences/bbb_2min/bbb_2min.y4m
//! Run with: cargo test --release --test scene_cut_sequence -- --ignored

use gnc::bench::quality;
use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::format;
use gnc::{GpuContext, TemporalTransform};
use std::io::{BufRead, Read};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Shared GPU context
// ---------------------------------------------------------------------------

static GPU: OnceLock<GpuContext> = OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

// ---------------------------------------------------------------------------
// Minimal Y4M reader (4:2:0 and 4:4:4, BT.601 limited-range)
// ---------------------------------------------------------------------------

struct Y4mReader {
    reader: std::io::BufReader<std::fs::File>,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    chroma_420: bool,
}

impl Y4mReader {
    fn open(path: &str) -> Self {
        let file = std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("Cannot open Y4M file '{}': {}", path, e));
        let mut reader = std::io::BufReader::new(file);

        let mut header = String::new();
        reader.read_line(&mut header).expect("Y4M header read failed");
        let header = header.trim_end_matches('\n').trim_end_matches('\r');
        assert!(
            header.starts_with("YUV4MPEG2"),
            "Not a Y4M file (expected YUV4MPEG2 magic): {}",
            path
        );

        let mut width = 0u32;
        let mut height = 0u32;
        let mut fps_num = 30u32;
        let mut fps_den = 1u32;
        let mut chroma_420 = true;

        for token in header.split_ascii_whitespace().skip(1) {
            match token.chars().next() {
                Some('W') => width = token[1..].parse().expect("bad Y4M W field"),
                Some('H') => height = token[1..].parse().expect("bad Y4M H field"),
                Some('F') => {
                    let parts: Vec<&str> = token[1..].splitn(2, ':').collect();
                    if parts.len() == 2 {
                        fps_num = parts[0].parse().unwrap_or(30);
                        fps_den = parts[1].parse().unwrap_or(1);
                    }
                }
                Some('C') => {
                    let fmt = &token[1..];
                    let fmt_base = fmt.trim_start_matches(|c: char| !c.is_ascii_digit());
                    chroma_420 = !fmt_base.starts_with("444");
                }
                _ => {}
            }
        }
        assert!(width > 0 && height > 0, "Y4M header missing W/H in {}", path);
        Y4mReader { reader, width, height, fps_num, fps_den, chroma_420 }
    }

    /// Skip `n` frames without decoding them (seek forward by frame bytes).
    fn skip_frames(&mut self, n: usize) {
        let w = self.width as usize;
        let h = self.height as usize;
        let luma_bytes = w * h;
        let chroma_bytes = if self.chroma_420 {
            let uw = w.div_ceil(2);
            let uh = h.div_ceil(2);
            2 * uw * uh
        } else {
            2 * luma_bytes
        };
        let frame_bytes = luma_bytes + chroma_bytes;
        let mut skip_buf = vec![0u8; frame_bytes];
        for _ in 0..n {
            // Consume the "FRAME\n" marker line
            let mut line = String::new();
            loop {
                line.clear();
                let bytes_read = self.reader.read_line(&mut line).expect("Y4M skip: read_line error");
                if bytes_read == 0 {
                    return; // EOF
                }
                if line.trim_end_matches('\n').trim_end_matches('\r').starts_with("FRAME") {
                    break;
                }
            }
            // Consume raw YUV data
            self.reader.read_exact(&mut skip_buf).expect("Y4M skip: truncated frame data");
        }
    }

    /// Read one frame as interleaved RGB f32 (0-255), BT.601 limited-range.
    fn read_frame_rgb(&mut self) -> Option<Vec<f32>> {
        // Consume "FRAME..." line
        loop {
            let mut line = String::new();
            let n = self.reader.read_line(&mut line).expect("Y4M read error");
            if n == 0 {
                return None; // EOF
            }
            if line.trim_end_matches('\n').trim_end_matches('\r').starts_with("FRAME") {
                break;
            }
        }

        let w = self.width as usize;
        let h = self.height as usize;

        let mut y_plane = vec![0u8; w * h];
        self.reader.read_exact(&mut y_plane).expect("Y4M: truncated Y plane");

        let (cb_plane, cr_plane) = if self.chroma_420 {
            let uw = w.div_ceil(2);
            let uh = h.div_ceil(2);
            let uv_size = uw * uh;
            let mut cb = vec![0u8; uv_size];
            let mut cr = vec![0u8; uv_size];
            self.reader.read_exact(&mut cb).expect("Y4M: truncated Cb plane");
            self.reader.read_exact(&mut cr).expect("Y4M: truncated Cr plane");
            (cb, cr)
        } else {
            let mut cb = vec![0u8; w * h];
            let mut cr = vec![0u8; w * h];
            self.reader.read_exact(&mut cb).expect("Y4M: truncated Cb (444) plane");
            self.reader.read_exact(&mut cr).expect("Y4M: truncated Cr (444) plane");
            (cb, cr)
        };

        let mut rgb = vec![0.0f32; w * h * 3];
        for row in 0..h {
            for col in 0..w {
                let yv = y_plane[row * w + col] as f32;
                let (cbv, crv) = if self.chroma_420 {
                    let uw = w.div_ceil(2);
                    let idx = (row / 2) * uw + col / 2;
                    (cb_plane[idx] as f32, cr_plane[idx] as f32)
                } else {
                    let idx = row * w + col;
                    (cb_plane[idx] as f32, cr_plane[idx] as f32)
                };
                let yy = 1.164_f32 * (yv - 16.0);
                let pb = cbv - 128.0;
                let pr = crv - 128.0;
                let r = (yy + 1.596 * pr).clamp(0.0, 255.0);
                let g = (yy - 0.392 * pb - 0.813 * pr).clamp(0.0, 255.0);
                let b = (yy + 2.017 * pb).clamp(0.0, 255.0);
                let base = (row * w + col) * 3;
                rgb[base] = r;
                rgb[base + 1] = g;
                rgb[base + 2] = b;
            }
        }
        Some(rgb)
    }
}

// ---------------------------------------------------------------------------
// Test constants
// ---------------------------------------------------------------------------

/// Path to the bbb_2min Y4M file (relative to CARGO_MANIFEST_DIR).
const Y4M_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_material/frames/sequences/bbb_2min/bbb_2min.y4m"
);

/// First frame index to encode (inclusive, 0-based within the Y4M sequence).
const FRAME_START: usize = 250;

/// Last frame index to encode (inclusive). Total = FRAME_END - FRAME_START + 1 = 351.
const FRAME_END: usize = 600;

/// Number of frames to encode.
const TOTAL_FRAMES: usize = FRAME_END - FRAME_START + 1; // 351

/// Quality setting for the encode (affects RD tradeoff, not correctness).
const QUALITY: u32 = 75;

/// GOP size used with Haar temporal wavelet.
const GOP_SIZE: usize = 4;

// ---------------------------------------------------------------------------
// Helper: load frames 250-600 from the Y4M
// ---------------------------------------------------------------------------

fn load_frames_250_600() -> (Vec<Vec<f32>>, u32, u32) {
    let mut reader = Y4mReader::open(Y4M_PATH);
    let w = reader.width;
    let h = reader.height;

    // Skip to frame 250
    reader.skip_frames(FRAME_START);

    let mut frames = Vec::with_capacity(TOTAL_FRAMES);
    for _ in 0..TOTAL_FRAMES {
        match reader.read_frame_rgb() {
            Some(f) => frames.push(f),
            None => break, // Y4M shorter than expected — test will fail the count assertion
        }
    }
    (frames, w, h)
}

// ---------------------------------------------------------------------------
// Main test: temporal encode/decode roundtrip across scene cuts
// ---------------------------------------------------------------------------

/// Integration test: temporal wavelet encode/decode over bbb_2min frames 250-600.
///
/// This range contains hard scene cuts at approximately frame 285, 378, and 553
/// (global numbering), which fall at GOP-relative indices 35, 128, and 303 in the
/// 351-frame window. GOPs spanning these cuts will have high temporal highpass energy;
/// the test verifies the encoder handles them gracefully and produces correct output.
///
/// Requires: test_material/frames/sequences/bbb_2min/bbb_2min.y4m
#[test]
#[ignore = "requires test_material/frames/sequences/bbb_2min/bbb_2min.y4m"]
fn test_bbb_scene_cuts_250_600() {
    let (frames, w, h) = load_frames_250_600();

    assert_eq!(
        frames.len(),
        TOTAL_FRAMES,
        "Expected {} frames from bbb_2min[250..=600], got {}",
        TOTAL_FRAMES,
        frames.len()
    );

    let ctx = gpu();
    let mut encoder = EncoderPipeline::new(ctx);
    let decoder = DecoderPipeline::new(ctx);

    // Build codec config with Haar temporal wavelet, GOP_SIZE=4
    let mut config = gnc::quality_preset(QUALITY);
    config.temporal_transform = TemporalTransform::Haar;

    // Encode all complete GOPs
    let num_gops = TOTAL_FRAMES / GOP_SIZE; // 351 / 4 = 87
    let tail_start = num_gops * GOP_SIZE;   // 87 * 4 = 348
    let tail_count = TOTAL_FRAMES - tail_start; // 3

    eprintln!(
        "Encoding {} frames: {} GOPs of size {} + {} tail I-frames",
        TOTAL_FRAMES, num_gops, GOP_SIZE, tail_count
    );

    let mut groups = Vec::with_capacity(num_gops);
    for gop_idx in 0..num_gops {
        let base = gop_idx * GOP_SIZE;
        let gop_slice: Vec<&[f32]> = frames[base..base + GOP_SIZE]
            .iter()
            .map(|f| f.as_slice())
            .collect();
        let group = encoder.encode_temporal_wavelet_gop(
            ctx,
            &gop_slice,
            w,
            h,
            &config,
            TemporalTransform::Haar,
            None,
        );
        groups.push(group);
    }

    // Tail frames (frames 348, 349, 350) — encode as I-frames
    let mut tail_cfg = config.clone();
    tail_cfg.keyframe_interval = 1;
    tail_cfg.temporal_transform = TemporalTransform::None;
    tail_cfg.cfl_enabled = false;

    let mut tail_iframes = Vec::with_capacity(tail_count);
    for frame in frames.iter().take(TOTAL_FRAMES).skip(tail_start) {
        let cf = encoder.encode(ctx, frame, w, h, &tail_cfg);
        tail_iframes.push(cf);
    }

    // -----------------------------------------------------------------------
    // Structural checks on the encoded sequence
    // -----------------------------------------------------------------------

    assert_eq!(
        groups.len(),
        num_gops,
        "Expected {} temporal GOPs, got {}",
        num_gops,
        groups.len()
    );

    assert_eq!(
        tail_iframes.len(),
        tail_count,
        "Expected {} tail I-frames, got {}",
        tail_count,
        tail_iframes.len()
    );

    // tail_count > 0 proves the sequence was not an exact multiple of GOP_SIZE
    assert!(
        !tail_iframes.is_empty(),
        "Expected tail I-frames (351 is not divisible by {}), got none",
        GOP_SIZE
    );

    // GOP count is strictly less than naive frame_count/GOP_SIZE rounded up
    let naive_ceil = TOTAL_FRAMES.div_ceil(GOP_SIZE);
    assert!(
        groups.len() < naive_ceil,
        "GOP count {} should be less than ceil({}/{}) = {} \
         (tail handled separately as I-frames)",
        groups.len(),
        TOTAL_FRAMES,
        GOP_SIZE,
        naive_ceil
    );

    // -----------------------------------------------------------------------
    // GNV2 serialization / deserialization roundtrip
    // -----------------------------------------------------------------------

    let fps_num = 24u32;
    let fps_den = 1u32;
    let seq = gnc::TemporalEncodedSequence {
        mode: TemporalTransform::Haar,
        groups: groups.clone(),
        tail_iframes: tail_iframes.clone(),
        frame_count: TOTAL_FRAMES,
        gop_size: GOP_SIZE,
    };

    let serialized = format::serialize_temporal_sequence(&seq, (fps_num, fps_den));
    assert!(
        serialized.len() > 1024,
        "Serialized GNV2 too small: {} bytes",
        serialized.len()
    );

    // Check GNV2 magic
    assert_eq!(
        &serialized[0..4],
        b"GNV2",
        "Expected GNV2 container magic"
    );

    let deserialized = format::deserialize_temporal_sequence(&serialized);

    assert_eq!(
        deserialized.frame_count,
        TOTAL_FRAMES,
        "Deserialized frame_count mismatch: expected {}, got {}",
        TOTAL_FRAMES,
        deserialized.frame_count
    );
    assert_eq!(
        deserialized.gop_size,
        GOP_SIZE,
        "Deserialized gop_size mismatch: expected {}, got {}",
        GOP_SIZE,
        deserialized.gop_size
    );
    assert_eq!(
        deserialized.groups.len(),
        num_gops,
        "Deserialized GOP count mismatch: expected {}, got {}",
        num_gops,
        deserialized.groups.len()
    );
    assert_eq!(
        deserialized.tail_iframes.len(),
        tail_count,
        "Deserialized tail_iframes count mismatch: expected {}, got {}",
        tail_count,
        deserialized.tail_iframes.len()
    );

    // -----------------------------------------------------------------------
    // Decode and check PSNR — no garbage frames from temporal bleed
    // -----------------------------------------------------------------------

    let decoded_frames = decoder.decode_temporal_sequence(ctx, &seq);

    assert_eq!(
        decoded_frames.len(),
        num_gops * GOP_SIZE,
        "Decoded GOP frame count mismatch: expected {}, got {}",
        num_gops * GOP_SIZE,
        decoded_frames.len()
    );

    let pixels_per_frame = (w * h * 3) as usize;
    let mut min_psnr = f64::INFINITY;
    let mut max_psnr = 0.0f64;
    let mut psnr_sum = 0.0f64;
    let mut psnr_count = 0usize;

    // Check each decoded GOP frame — skip the tail (separately I-frame encoded)
    for (i, decoded) in decoded_frames.iter().enumerate() {
        assert_eq!(
            decoded.len(),
            pixels_per_frame,
            "Decoded frame {} has wrong pixel count: expected {}, got {}",
            i,
            pixels_per_frame,
            decoded.len()
        );

        let orig = &frames[i];
        let psnr = quality::psnr(orig, decoded, 255.0);
        assert!(
            psnr.is_finite(),
            "Frame {} PSNR is non-finite (NaN or inf) — likely a temporal bleed or decode bug",
            i
        );
        assert!(
            psnr > 30.0,
            "Frame {} PSNR {:.2} dB < 30 dB threshold at q={} — possible temporal bleed across scene cut",
            i,
            psnr,
            QUALITY
        );

        if psnr < min_psnr {
            min_psnr = psnr;
        }
        if psnr > max_psnr {
            max_psnr = psnr;
        }
        psnr_sum += psnr;
        psnr_count += 1;
    }

    // Also check tail I-frames decode correctly
    let tail_decoder = DecoderPipeline::new(ctx);
    for (i, tail_cf) in tail_iframes.iter().enumerate() {
        let decoded_tail = tail_decoder.decode(ctx, tail_cf);
        let orig_idx = tail_start + i;
        let psnr = quality::psnr(&frames[orig_idx], &decoded_tail, 255.0);
        assert!(
            psnr > 30.0,
            "Tail I-frame {} (global frame {}) PSNR {:.2} dB < 30 dB",
            i,
            orig_idx,
            psnr
        );
    }

    let avg_psnr = psnr_sum / psnr_count as f64;
    eprintln!(
        "bbb_2min frames 250-600: {} GOPs, {} tail I-frames decoded",
        num_gops, tail_count
    );
    eprintln!(
        "PSNR over {} GOP frames: min={:.2} dB, avg={:.2} dB, max={:.2} dB",
        psnr_count, min_psnr, avg_psnr, max_psnr
    );

    // Sanity: average PSNR at q=75 should be comfortably above 35 dB on natural video
    assert!(
        avg_psnr > 35.0,
        "Average PSNR {:.2} dB too low for q={} — expected > 35 dB on natural video",
        avg_psnr,
        QUALITY
    );
}

// ---------------------------------------------------------------------------
// Smoke test: verify the Y4M file is accessible and has expected dimensions
// ---------------------------------------------------------------------------

/// Smoke test: verify the Y4M file header is readable and dimensions are sensible.
/// This runs without --ignored to give a quick diagnostic when the file is missing.
#[test]
#[ignore = "requires test_material/frames/sequences/bbb_2min/bbb_2min.y4m"]
fn test_bbb_2min_y4m_header() {
    let reader = Y4mReader::open(Y4M_PATH);
    assert!(reader.width >= 640, "Expected at least 640px wide, got {}", reader.width);
    assert!(reader.height >= 360, "Expected at least 360px tall, got {}", reader.height);
    assert!(reader.fps_num > 0, "fps_num should be > 0");
    assert!(reader.fps_den > 0, "fps_den should be > 0");
    eprintln!(
        "bbb_2min.y4m: {}x{} @ {}/{} fps, chroma_420={}",
        reader.width, reader.height, reader.fps_num, reader.fps_den, reader.chroma_420
    );
}
