#![forbid(unsafe_code)]

use clap::{Parser, Subcommand};
use gnc::bench::bdrate;
use gnc::bench::codec_compare;
use gnc::bench::compare::{write_csv, BenchmarkConfig, BenchmarkResult};
use gnc::bench::quality::{self, QualityMetrics};
use gnc::bench::sequence_metrics::{self, FrameMetrics};
use gnc::bench::throughput;
use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::experiments;
use gnc::format::{
    deserialize_compressed, deserialize_sequence_frame, deserialize_sequence_header,
    deserialize_temporal_sequence, seek_to_keyframe, serialize_compressed, serialize_sequence,
    serialize_temporal_sequence,
};
use gnc::image_util::{load_image_rgb_f32, parse_wavelet_type, save_image_rgb_f32};
use gnc::{
    CodecConfig, CompressedFrame, EntropyData, FrameType, GpuContext, RateMode, TemporalTransform,
};

// ---------------------------------------------------------------------------
// Minimal Y4M parser — no external dependency, streaming frame-by-frame.
//
// Y4M spec (YUV4MPEG2):
//   Header:  "YUV4MPEG2 W<w> H<h> F<fps_n>:<fps_d> Ip A0:0 C<chroma>\n"
//   Frame:   "FRAME\n" followed by raw planar YUV data
//
// Supported chroma formats: 420 (default), 444.
// Pixel values are in the range 0-255 (8-bit).
// BT.601 limited-range (studio-swing) Y 16-235, Cb/Cr 16-240 is the Xiph
// convention for 8-bit Y4M sequences.  We convert to full-range RGB f32
// (0.0-255.0 interleaved) as expected by EncoderPipeline::encode*.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Y4mChroma {
    C420,
    C444,
}

struct Y4mReader {
    reader: std::io::BufReader<std::fs::File>,
    pub width: u32,
    pub height: u32,
    /// Frames-per-second numerator / denominator from the header.
    pub fps_num: u32,
    pub fps_den: u32,
    chroma: Y4mChroma,
}

impl Y4mReader {
    fn open(path: &str) -> Self {
        use std::io::BufRead;
        let file = std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("Failed to open Y4M file '{}': {}", path, e));
        let mut reader = std::io::BufReader::new(file);

        // Read header line (terminated by '\n')
        let mut header = String::new();
        reader
            .read_line(&mut header)
            .expect("Failed to read Y4M header");
        let header = header.trim_end_matches('\n').trim_end_matches('\r');
        if !header.starts_with("YUV4MPEG2") {
            panic!("Not a Y4M file (missing YUV4MPEG2 magic): {}", path);
        }

        let mut width = 0u32;
        let mut height = 0u32;
        let mut fps_num = 30u32;
        let mut fps_den = 1u32;
        let mut chroma = Y4mChroma::C420;

        for token in header.split_ascii_whitespace().skip(1) {
            match token.chars().next() {
                Some('W') => width = token[1..].parse().expect("Bad Y4M width"),
                Some('H') => height = token[1..].parse().expect("Bad Y4M height"),
                Some('F') => {
                    let parts: Vec<&str> = token[1..].splitn(2, ':').collect();
                    if parts.len() == 2 {
                        fps_num = parts[0].parse().unwrap_or(30);
                        fps_den = parts[1].parse().unwrap_or(1);
                    }
                }
                Some('C') => {
                    let fmt = &token[1..];
                    // Strip optional bit-depth suffix (e.g. "420p10" → "420")
                    let fmt_base = fmt.trim_start_matches(|c: char| !c.is_ascii_digit());
                    chroma = if fmt_base.starts_with("444") {
                        Y4mChroma::C444
                    } else {
                        Y4mChroma::C420 // default; 420jpeg / 420mpeg2 / plain 420
                    };
                }
                _ => {}
            }
        }
        assert!(width > 0 && height > 0, "Y4M header missing W/H");
        Y4mReader { reader, width, height, fps_num, fps_den, chroma }
    }

    /// Read one frame and return interleaved RGB f32 (0-255), or None at EOF.
    fn read_frame_rgb(&mut self) -> Option<Vec<f32>> {
        use std::io::BufRead;
        use std::io::Read;

        // Read "FRAME..." line
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.reader.read_line(&mut line).expect("Y4M read error");
            if n == 0 {
                return None; // EOF
            }
            let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
            if trimmed.starts_with("FRAME") {
                break;
            }
            // Ignore unexpected lines (e.g. extra headers)
        }

        let w = self.width as usize;
        let h = self.height as usize;

        // Read luma plane (Y): w*h bytes
        let y_size = w * h;
        let mut y_plane = vec![0u8; y_size];
        self.reader
            .read_exact(&mut y_plane)
            .expect("Y4M: truncated Y plane");

        // Read chroma planes (Cb, Cr)
        let (cb_plane, cr_plane) = match self.chroma {
            Y4mChroma::C420 => {
                let uv_w = w.div_ceil(2);
                let uv_h = h.div_ceil(2);
                let uv_size = uv_w * uv_h;
                let mut cb = vec![0u8; uv_size];
                let mut cr = vec![0u8; uv_size];
                self.reader.read_exact(&mut cb).expect("Y4M: truncated Cb plane");
                self.reader.read_exact(&mut cr).expect("Y4M: truncated Cr plane");
                (cb, cr)
            }
            Y4mChroma::C444 => {
                let mut cb = vec![0u8; y_size];
                let mut cr = vec![0u8; y_size];
                self.reader.read_exact(&mut cb).expect("Y4M: truncated Cb plane");
                self.reader.read_exact(&mut cr).expect("Y4M: truncated Cr plane");
                (cb, cr)
            }
        };

        // Convert YCbCr → RGB f32 (0-255), BT.601 limited-range (studio swing).
        //   Y:  16-235 (luma)
        //   Cb/Cr: 16-240 (chroma, centre at 128)
        //
        //   R = clip(1.164*(Y-16)                   + 1.596*(Cr-128), 0, 255)
        //   G = clip(1.164*(Y-16) - 0.392*(Cb-128)  - 0.813*(Cr-128), 0, 255)
        //   B = clip(1.164*(Y-16) + 2.017*(Cb-128),                    0, 255)
        let mut rgb = vec![0.0f32; w * h * 3];
        for row in 0..h {
            for col in 0..w {
                let y_val = y_plane[row * w + col] as f32;
                let (cb_val, cr_val) = match self.chroma {
                    Y4mChroma::C444 => {
                        let idx = row * w + col;
                        (cb_plane[idx] as f32, cr_plane[idx] as f32)
                    }
                    Y4mChroma::C420 => {
                        let uv_w = w.div_ceil(2);
                        let uv_row = row / 2;
                        let uv_col = col / 2;
                        let idx = uv_row * uv_w + uv_col;
                        (cb_plane[idx] as f32, cr_plane[idx] as f32)
                    }
                };
                let yy = 1.164_f32 * (y_val - 16.0);
                let pb = cb_val - 128.0;
                let pr = cr_val - 128.0;
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

#[derive(Parser)]
#[command(name = "gnc", about = "GPU-native broadcast video codec")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Encode an image file
    Encode {
        /// Input image file (PNG, JPEG, etc.)
        #[arg(short, long)]
        input: String,

        /// Output compressed file
        #[arg(short, long)]
        output: String,

        /// Quality preset (1-100). Sets qstep, wavelet, dead zone, etc. from research-tuned presets.
        /// Individual flags (--qstep, --wavelet, etc.) override the preset when specified.
        #[arg(short = 'q', long)]
        quality: Option<u32>,

        /// Quantization step size (overrides quality preset if both specified)
        #[arg(long)]
        qstep: Option<f32>,

        /// Tile size
        #[arg(short, long, default_value = "256")]
        tile_size: u32,

        /// Disable Chroma-from-Luma prediction (enabled by default)
        #[arg(long)]
        no_cfl: bool,

        /// Use bitplane entropy coder instead of rANS (default)
        #[arg(long)]
        bitplane: bool,

        /// Use rANS entropy coder (default; this flag is now a no-op kept for backward compat)
        #[arg(long)]
        rans: bool,

        /// Use Rice+ZRL entropy coder instead of rANS (default) — faster but ~30% worse compression
        #[arg(long)]
        rice: bool,

        /// Use canonical Huffman entropy coder instead of rANS (default)
        #[arg(long)]
        huffman: bool,

        /// Wavelet type: 97 (CDF 9/7) or 53 (LeGall 5/3). Overrides quality preset if specified.
        #[arg(long)]
        wavelet: Option<String>,

        /// Disable per-subband entropy coding (enabled by default)
        #[arg(long)]
        no_per_subband: bool,

        /// Use CPU entropy encoding instead of GPU (for testing/debugging)
        #[arg(long)]
        cpu_encode: bool,
    },

    /// Decode a compressed file back to an image
    Decode {
        /// Input compressed file
        #[arg(short, long)]
        input: String,

        /// Output image file (PNG)
        #[arg(short, long)]
        output: String,
    },

    /// Run encode-decode benchmark on an image
    Benchmark {
        /// Input image file
        #[arg(short, long)]
        input: String,

        /// Run throughput measurement with N iterations
        #[arg(short = 'n', long, default_value = "10")]
        iterations: u32,

        /// Output CSV file for results
        #[arg(long)]
        csv: Option<String>,

        /// Quality preset (1-100, default 75 = good general-purpose quality)
        #[arg(short = 'q', long, default_value = "75")]
        quality: u32,

        /// Use bitplane entropy coder instead of rANS (default)
        #[arg(long)]
        bitplane: bool,

        /// Use rANS entropy coder (default; this flag is now a no-op kept for backward compat)
        #[arg(long)]
        rans: bool,

        /// Use Rice+ZRL entropy coder instead of rANS (default) — faster but ~30% worse compression
        #[arg(long)]
        rice: bool,

        /// Use canonical Huffman entropy coder instead of rANS (default)
        #[arg(long)]
        huffman: bool,

        /// Use CPU entropy encoding instead of GPU
        #[arg(long)]
        cpu_encode: bool,

        /// Use block DCT-8×8 transform instead of wavelet (fewer dispatches, faster)
        #[arg(long)]
        dct: bool,

        /// DCT frequency-dependent quantization strength (default: 3.0, 0=flat)
        #[arg(long)]
        dct_freq_strength: Option<f32>,
    },

    /// Benchmark temporal (I+P frame) encoding on a sequence of frames
    BenchmarkSequence {
        /// Input source: PNG frame pattern (e.g., "frames/frame_%04d.png") or a Y4M file (e.g., "video.y4m").
        /// Y4M input avoids PNG decode overhead and measures actual GNC encoder throughput.
        #[arg(short, long)]
        input: String,

        /// Number of frames to encode
        #[arg(short = 'n', long, default_value = "10")]
        num_frames: usize,

        /// Keyframe interval (1 = all I-frames)
        #[arg(short, long, default_value = "8")]
        keyframe_interval: u32,

        /// Quality preset (1-100). Sets qstep, wavelet, dead zone, etc.
        #[arg(short = 'q', long)]
        quality: Option<u32>,

        /// Quantization step size (overrides quality preset if both specified)
        #[arg(long)]
        qstep: Option<f32>,

        /// Output CSV file for per-frame metrics and sequence summary
        #[arg(long)]
        csv: Option<String>,

        /// Target bitrate for rate control (e.g., "10M" for 10 Mbps, "500K" for 500 Kbps).
        /// When set, the encoder adjusts qstep per frame to hit this target.
        #[arg(long)]
        bitrate: Option<String>,

        /// Rate control mode: "cbr" (constant bitrate) or "vbr" (variable bitrate).
        /// Only used when --bitrate is set. Default: vbr.
        #[arg(long, default_value = "vbr")]
        rate_mode: String,

        /// Frame rate in fps (used for rate control bitrate calculations).
        /// Default: 30.
        #[arg(long, default_value = "30")]
        fps: f64,

        /// Use rANS entropy coder (default; this flag is now a no-op kept for backward compat)
        #[arg(long)]
        rans: bool,

        /// Use Rice+ZRL entropy coder instead of rANS (default) — faster but ~30% worse compression
        #[arg(long)]
        rice: bool,

        /// Enable per-frame encode diagnostics (also via GNC_DIAGNOSTICS=1)
        #[arg(long)]
        diagnostics: bool,

        /// Temporal wavelet mode: none (default, I+P+B motion vectors), auto (Haar with adaptive GOP), haar, 53 (experimental)
        #[arg(long, default_value = "none")]
        temporal_wavelet: String,

        /// Temporal highpass qstep multiplier (e.g. 2.0 = double qstep for highpass)
        #[arg(long)]
        tw_highpass_mul: Option<f32>,

        /// Run A/B comparison against baseline I+P (+ I-only)
        #[arg(long)]
        ab: bool,

        /// Output file path for GNV2 container
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Encode a sequence of image frames into a .gnv container
    EncodeSequence {
        /// Input frame pattern (e.g., "frames/%04d.png")
        #[arg(short, long)]
        input: String,

        /// Output .gnv file
        #[arg(short, long)]
        output: String,

        /// Quality preset (1-100)
        #[arg(short = 'q', long, default_value = "75")]
        quality: u32,

        /// Quantization step size (overrides quality preset if specified)
        #[arg(long)]
        qstep: Option<f32>,

        /// Keyframe interval (1 = all I-frames)
        #[arg(long, default_value = "8")]
        keyframe_interval: u32,

        /// Number of frames to encode (auto-detect if not specified)
        #[arg(short = 'n', long)]
        num_frames: Option<usize>,

        /// Framerate numerator (e.g. 30 for 30fps, 30000 for 29.97fps)
        #[arg(long, default_value = "30")]
        fps_num: u32,

        /// Framerate denominator (e.g. 1 for 30fps, 1001 for 29.97fps)
        #[arg(long, default_value = "1")]
        fps_den: u32,

        /// Target bitrate for rate control (e.g., "10M" for 10 Mbps, "500K" for 500 Kbps).
        /// When set, the encoder adjusts qstep per frame to hit this target.
        #[arg(long)]
        bitrate: Option<String>,

        /// Rate control mode: "cbr" (constant bitrate) or "vbr" (variable bitrate).
        /// Only used when --bitrate is set. Default: vbr.
        #[arg(long, default_value = "vbr")]
        rate_mode: String,

        /// Use rANS entropy coder (default; this flag is now a no-op kept for backward compat)
        #[arg(long)]
        rans: bool,

        /// Use Rice+ZRL entropy coder instead of rANS (default) — faster but ~30% worse compression
        #[arg(long)]
        rice: bool,

        /// Enable per-frame encode diagnostics (also via GNC_DIAGNOSTICS=1)
        #[arg(long)]
        diagnostics: bool,

        /// Temporal wavelet mode: only "none" supported here (I+P+B motion vectors).
        /// For temporal wavelet encoding, use benchmark-sequence instead.
        #[arg(long, default_value = "none")]
        temporal_wavelet: String,
    },

    /// Decode a .gnv or .gnv2 container back to image frames
    DecodeSequence {
        /// Input .gnv file
        #[arg(short, long)]
        input: String,

        /// Output frame pattern (e.g., "frames/%04d.png")
        #[arg(short, long)]
        output: String,

        /// Seek to this time (in seconds) before decoding. Decodes from the
        /// nearest preceding keyframe through the target frame.
        #[arg(long)]
        seek: Option<f64>,
    },

    /// Run experiments on an input image
    Sweep {
        /// Input image file
        #[arg(short, long)]
        input: String,

        /// Output CSV file for results
        #[arg(long, default_value = "results/sweep.csv")]
        csv: String,

        /// Experiment set: all, baseline, deadzone, levels, subband, entropy
        #[arg(short, long, default_value = "all")]
        experiment: String,
    },

    /// Generate rate-distortion curve or compute BD-rate between two curves
    RdCurve {
        /// Input image file (encode sweep mode)
        #[arg(short, long)]
        input: Option<String>,

        /// Output CSV file for RD curve results
        #[arg(short, long, default_value = "rd_curve.csv")]
        output: String,

        /// Second CSV to compute BD-rate against
        #[arg(long)]
        compare: Option<String>,

        /// Comma-separated quality values to sweep
        #[arg(long, default_value = "10,20,30,40,50,60,70,80,90,100")]
        q_values: String,

        /// Also sweep JPEG and JPEG 2000, producing a unified comparison CSV
        #[arg(long)]
        compare_codecs: bool,
    },

    /// Automated benchmark across multiple Xiph sequences. Outputs CSV.
    BenchmarkSuite {
        /// Sequence base directory (default: test_material/frames/sequences)
        #[arg(long, default_value = "test_material/frames/sequences")]
        dir: String,

        /// Comma-separated sequence names (default: rush_hour,crowd_run,stockholm,park_joy,bbb_2min)
        #[arg(long, default_value = "rush_hour,crowd_run,stockholm,park_joy,bbb_2min")]
        sequences: String,

        /// Max frames per sequence (0 = all)
        #[arg(short = 'n', long, default_value = "120")]
        max_frames: usize,

        /// Quality preset (1-100)
        #[arg(short = 'q', long, default_value = "75")]
        quality: u32,

        /// Output CSV file
        #[arg(long, default_value = "results/benchmark_suite.csv")]
        csv: String,

        /// Temporal wavelet mode: none (default), auto, haar
        #[arg(long, default_value = "none")]
        temporal_wavelet: String,

        /// Temporal highpass qstep multiplier (overrides adaptive mul when set)
        #[arg(long)]
        tw_highpass_mul: Option<f32>,
    },

    /// Compare block-based transforms (DCT, Hadamard, Haar) against wavelet baseline
    TransformShootout {
        /// Input image file
        #[arg(short, long)]
        input: String,

        /// Number of timing iterations per transform
        #[arg(short = 'n', long, default_value = "20")]
        iterations: u32,
    },
}

/// Parse a bitrate string like "10M", "500K", "1.5M", or raw number "10000000".
/// Returns bits per second.
fn parse_bitrate(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() {
        panic!("Empty bitrate string");
    }
    let (num_part, multiplier) = if s.ends_with('M') || s.ends_with('m') {
        (&s[..s.len() - 1], 1_000_000.0)
    } else if s.ends_with('K') || s.ends_with('k') {
        (&s[..s.len() - 1], 1_000.0)
    } else if s.ends_with('G') || s.ends_with('g') {
        (&s[..s.len() - 1], 1_000_000_000.0)
    } else {
        (s, 1.0)
    };
    let value: f64 = num_part
        .parse()
        .unwrap_or_else(|_| panic!("Invalid bitrate number: '{}'", num_part));
    value * multiplier
}

/// Parse rate mode string ("cbr" or "vbr").
fn parse_rate_mode(s: &str) -> RateMode {
    match s.to_lowercase().as_str() {
        "cbr" => RateMode::CBR,
        "vbr" => RateMode::VBR,
        other => {
            eprintln!(
                "Unknown rate mode '{}'. Use 'cbr' or 'vbr'. Defaulting to VBR.",
                other
            );
            RateMode::VBR
        }
    }
}

fn parse_temporal_transform(s: &str) -> TemporalTransform {
    match s.to_lowercase().as_str() {
        "none" => TemporalTransform::None,
        "haar" => TemporalTransform::Haar,
        "53" | "5/3" | "legall53" | "legall" => TemporalTransform::LeGall53,
        "auto" => TemporalTransform::Haar, // placeholder — use select_temporal_mode() for proper auto
        other => {
            eprintln!(
                "Unknown temporal wavelet '{}'. Use 'none', 'haar', '53', or 'auto'. Defaulting to none.",
                other
            );
            TemporalTransform::None
        }
    }
}

/// Auto-select temporal transform and GOP size.
/// Always Haar — multilevel dyadic decomposition gives optimal energy compaction.
/// LeGall 5/3 is available via explicit --temporal-wavelet 53 but not auto-selected
/// (4-frame 5/3 produces 2 lowpass + 2 highpass, worse than Haar's 1+3 split).
///
/// Returns (TemporalTransform, gop_size).
fn select_temporal_mode(explicit: &str, fps: f64, quality: u32) -> (TemporalTransform, usize) {
    let lower = explicit.to_lowercase();
    if lower != "auto" {
        let mode = parse_temporal_transform(explicit);
        let gop = match mode {
            TemporalTransform::LeGall53 => 4,
            TemporalTransform::None => 0,
            TemporalTransform::Haar => {
                // Explicit Haar: use keyframe_interval (caller handles)
                0 // sentinel — caller uses keyframe_interval
            }
        };
        eprintln!("Temporal mode: {:?} (explicit)", mode);
        return (mode, gop);
    }
    // Auto: Haar with adaptive GOP size
    //   fps < 1:              None (still image / slideshow)
    //   fps <= 25 or q >= 90: Haar gop=2 (minimal latency, 1 level)
    //   fps > 25:             Haar gop=4 (2 levels, better compression)
    if fps < 1.0 {
        eprintln!("Temporal mode: None (auto, fps={fps} < 1)");
        (TemporalTransform::None, 0)
    } else if fps <= 25.0 || quality >= 90 {
        eprintln!("Temporal mode: Haar gop=2 (auto, fps={fps}, q={quality})");
        (TemporalTransform::Haar, 2)
    } else {
        eprintln!("Temporal mode: Haar gop=4 (auto, fps={fps}, q={quality})");
        (TemporalTransform::Haar, 4)
    }
}

/// Compute mean absolute Y-channel difference between two RGB f32 frames (values in [0,255]).
/// Uses 8× subsampling for speed. Returns a value in [0, 1] (normalized by 255).
/// Scene cuts typically score > 0.15; in-scene motion typically < 0.05.
fn scene_cut_score(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len() / 3;
    if n == 0 {
        return 0.0;
    }
    const STEP: usize = 8;
    let mut sum = 0.0f32;
    let mut count = 0usize;
    let mut i = 0;
    while i < n {
        let y_a = 0.299 * a[i * 3] + 0.587 * a[i * 3 + 1] + 0.114 * a[i * 3 + 2];
        let y_b = 0.299 * b[i * 3] + 0.587 * b[i * 3 + 1] + 0.114 * b[i * 3 + 2];
        sum += (y_a - y_b).abs();
        count += 1;
        i += STEP;
    }
    sum / (count as f32 * 255.0)
}

/// Scene cut threshold for `scene_cut_score` (normalized to [0,1]).
/// Hard cuts score > 0.40 (ffprobe equivalent). 0.15 catches most hard cuts
/// without false positives from fast motion.
const SCENE_CUT_THRESH: f32 = 0.15;

fn csv_with_suffix(path: &str, suffix: &str) -> String {
    if let Some((base, ext)) = path.rsplit_once('.') {
        format!("{}_{}.{}", base, suffix, ext)
    } else {
        format!("{}_{}", path, suffix)
    }
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Command::Encode {
            input,
            output,
            quality,
            qstep,
            tile_size,
            no_cfl,
            bitplane,
            rans,
            rice,
            huffman,
            wavelet,
            no_per_subband,
            cpu_encode,
        } => {
            let (rgb_data, w, h) = load_image_rgb_f32(&input);
            println!("Input: {}x{} ({} pixels)", w, h, w * h);

            let ctx = GpuContext::new();
            let mut encoder = EncoderPipeline::new(&ctx);

            // Build config: quality preset as base, or manual defaults
            let mut config = if let Some(q) = quality {
                println!("Quality preset: q={}", q);
                gnc::quality_preset(q)
            } else {
                CodecConfig {
                    quantization_step: qstep.unwrap_or(4.0),
                    wavelet_type: parse_wavelet_type(wavelet.as_deref().unwrap_or("97")),
                    cfl_enabled: true,
                    per_subband_entropy: true,
                    ..Default::default()
                }
            };

            // Apply explicit CLI overrides
            config.tile_size = tile_size;
            if let Some(qs) = qstep {
                config.quantization_step = qs;
            }
            if let Some(ref wv) = wavelet {
                config.wavelet_type = parse_wavelet_type(wv);
            }
            if no_cfl {
                config.cfl_enabled = false;
            }
            if bitplane {
                config.entropy_coder = gnc::EntropyCoder::Bitplane;
            }
            if rans {
                config.entropy_coder = gnc::EntropyCoder::Rans;
            }
            if rice {
                config.entropy_coder = gnc::EntropyCoder::Rice;
            }
            if huffman {
                config.entropy_coder = gnc::EntropyCoder::Huffman;
            }
            if no_per_subband {
                config.per_subband_entropy = false;
            }
            if cpu_encode {
                config.gpu_entropy_encode = false;
            }

            let compressed = encoder.encode(&ctx, &rgb_data, w, h, &config);
            println!(
                "Compressed: {} bytes ({:.2} bpp)",
                compressed.byte_size(),
                compressed.bpp()
            );

            // Serialize to file
            let serialized = serialize_compressed(&compressed);
            std::fs::write(&output, &serialized).expect("Failed to write output");
            println!("Written to {}", output);
        }

        Command::Decode { input, output } => {
            let data = std::fs::read(&input).expect("Failed to read input");
            let compressed = deserialize_compressed(&data);

            println!(
                "Compressed frame: {}x{} ({} bytes, {:.2} bpp)",
                compressed.info.width,
                compressed.info.height,
                compressed.byte_size(),
                compressed.bpp()
            );

            let ctx = GpuContext::new();
            let decoder = DecoderPipeline::new(&ctx);

            let rgb_data = decoder.decode(&ctx, &compressed);
            save_image_rgb_f32(
                &output,
                &rgb_data,
                compressed.info.width,
                compressed.info.height,
            );
            println!("Decoded to {}", output);
        }

        Command::Benchmark {
            input,
            iterations,
            csv,
            quality,
            bitplane,
            rans,
            rice,
            huffman,
            cpu_encode,
            dct,
            dct_freq_strength,
        } => {
            let (rgb_data, w, h) = load_image_rgb_f32(&input);

            let ctx = GpuContext::new();
            let mut encoder = EncoderPipeline::new(&ctx);
            let decoder = DecoderPipeline::new(&ctx);

            let mut config = gnc::quality_preset(quality);
            if bitplane {
                config.entropy_coder = gnc::EntropyCoder::Bitplane;
            }
            if rans {
                config.entropy_coder = gnc::EntropyCoder::Rans;
            }
            if rice {
                config.entropy_coder = gnc::EntropyCoder::Rice;
            }
            if huffman {
                config.entropy_coder = gnc::EntropyCoder::Huffman;
            }
            if cpu_encode {
                config.gpu_entropy_encode = false;
            }
            if dct {
                config.transform_type = gnc::TransformType::BlockDCT8;
                config.cfl_enabled = false;
                config.adaptive_quantization = false;
                config.use_fused_quantize_histogram = false;
            }
            if let Some(fs) = dct_freq_strength {
                config.dct_freq_strength = fs;
            }

            let coder_name = match config.entropy_coder {
                gnc::EntropyCoder::Bitplane => "bitplane".to_string(),
                gnc::EntropyCoder::Rans if config.per_subband_entropy => "rANS subband".to_string(),
                gnc::EntropyCoder::Rans => "rANS single-table".to_string(),
                gnc::EntropyCoder::Rice => "Rice (sig+Golomb)".to_string(),
                gnc::EntropyCoder::Huffman => "Huffman (sig+canonical)".to_string(),
            };
            let encode_mode = if config.gpu_entropy_encode {
                "GPU"
            } else {
                "CPU"
            };
            println!(
                "Benchmark: {}x{} image, {} iterations, q={}, entropy: {} ({})",
                w, h, iterations, quality, coder_name, encode_mode
            );

            // Quality measurement
            let compressed = encoder.encode(&ctx, &rgb_data, w, h, &config);
            let reconstructed = decoder.decode(&ctx, &compressed);

            let psnr_all = quality::psnr(&rgb_data, &reconstructed, 255.0);
            let (pr, pg, pb) = quality::psnr_per_channel(&rgb_data, &reconstructed, 255.0);
            let ssim = quality::ssim_approx(&rgb_data, &reconstructed, 255.0);

            let qm = QualityMetrics {
                psnr_db: psnr_all,
                psnr_r: pr,
                psnr_g: pg,
                psnr_b: pb,
                ssim,
                bpp: compressed.bpp(),
                compressed_bytes: compressed.byte_size(),
            };

            println!("Quality: {}", qm);

            // Throughput measurement
            // Encode: sequential (each call is self-contained)
            // Decode: pipelined (overlap GPU work with CPU prep for next frame)
            let tp = throughput::measure_throughput(
                || {
                    encoder.encode(&ctx, &rgb_data, w, h, &config);
                },
                || {
                    decoder.decode_u8(&ctx, &compressed);
                },
                w,
                h,
                iterations,
            );
            println!("Throughput (sequential): {}", tp);

            // Pipelined decode: overlap submit/prepare with GPU execution
            {
                // Warmup
                decoder.submit_decode_u8(&ctx, &compressed);
                decoder.finish_decode_u8(&ctx, w, h);

                let start = std::time::Instant::now();
                // Submit first frame
                decoder.submit_decode_u8(&ctx, &compressed);
                for _ in 1..iterations {
                    // Finish previous frame + submit next (overlaps CPU prep with GPU)
                    decoder.finish_decode_u8(&ctx, w, h);
                    decoder.submit_decode_u8(&ctx, &compressed);
                }
                // Finish last frame
                decoder.finish_decode_u8(&ctx, w, h);
                let decode_total = start.elapsed();
                let decode_ms = decode_total.as_secs_f64() * 1000.0 / iterations as f64;
                let decode_fps = 1000.0 / decode_ms;
                println!(
                    "Throughput (pipelined): Decode: {:.2} ms ({:.1} fps)",
                    decode_ms, decode_fps
                );
            }

            if let Some(csv_path) = csv {
                let result = BenchmarkResult {
                    name: format!("baseline_{}", coder_name),
                    quality: qm,
                    throughput: Some(tp),
                    config: BenchmarkConfig {
                        tile_size: config.tile_size,
                        quantization_step: config.quantization_step,
                        dead_zone: config.dead_zone,
                        input_file: input,
                        width: w,
                        height: h,
                    },
                };
                write_csv(&[result], &csv_path).expect("Failed to write CSV");
                println!("Results written to {}", csv_path);
            }
        }

        Command::Sweep {
            input,
            csv,
            experiment,
        } => {
            let (rgb_data, w, h) = load_image_rgb_f32(&input);
            println!("Sweep ({}): {}x{} image", experiment, w, h);

            let ctx = GpuContext::new();
            let mut encoder = EncoderPipeline::new(&ctx);
            let decoder = DecoderPipeline::new(&ctx);

            let all_experiments = match experiment.as_str() {
                "baseline" => experiments::phase1_experiments(),
                "deadzone" => experiments::dead_zone_experiments(),
                "levels" => experiments::wavelet_level_experiments(),
                "subband" => experiments::subband_weight_experiments(),
                "combined" => experiments::combined_dz_subband_experiments(),
                "entropy" => experiments::entropy_experiments(),
                "cfl" => experiments::cfl_experiments(),
                "wavelet" => experiments::wavelet_experiments(),
                "best" => experiments::best_config_experiments(),
                "all" => {
                    let mut e = experiments::phase1_experiments();
                    e.extend(experiments::wavelet_level_experiments());
                    e.extend(experiments::dead_zone_experiments());
                    e.extend(experiments::subband_weight_experiments());
                    e.extend(experiments::combined_dz_subband_experiments());
                    e.extend(experiments::cfl_experiments());
                    e.extend(experiments::wavelet_experiments());
                    e.extend(experiments::entropy_experiments());
                    e.extend(experiments::best_config_experiments());
                    e
                }
                other => {
                    eprintln!("Unknown experiment set: {}. Use: all, baseline, deadzone, levels, subband, combined, cfl, wavelet, entropy, best", other);
                    std::process::exit(1);
                }
            };
            let mut results = Vec::new();

            for exp in &all_experiments {
                println!("\n--- {} ---", exp.name);
                println!("  {}", exp.description);

                let compressed = encoder.encode(&ctx, &rgb_data, w, h, &exp.config);
                let reconstructed = decoder.decode(&ctx, &compressed);

                let psnr_all = quality::psnr(&rgb_data, &reconstructed, 255.0);
                let (pr, pg, pb) = quality::psnr_per_channel(&rgb_data, &reconstructed, 255.0);
                let ssim = quality::ssim_approx(&rgb_data, &reconstructed, 255.0);

                let qm = QualityMetrics {
                    psnr_db: psnr_all,
                    psnr_r: pr,
                    psnr_g: pg,
                    psnr_b: pb,
                    ssim,
                    bpp: compressed.bpp(),
                    compressed_bytes: compressed.byte_size(),
                };

                println!("  {}", qm);

                results.push(BenchmarkResult {
                    name: exp.name.clone(),
                    quality: qm,
                    throughput: None,
                    config: BenchmarkConfig {
                        tile_size: exp.config.tile_size,
                        quantization_step: exp.config.quantization_step,
                        dead_zone: exp.config.dead_zone,
                        input_file: input.clone(),
                        width: w,
                        height: h,
                    },
                });
            }

            write_csv(&results, &csv).expect("Failed to write CSV");
            println!("\nResults written to {}", csv);
        }

        Command::BenchmarkSequence {
            input,
            num_frames,
            keyframe_interval,
            quality,
            qstep,
            csv,
            bitrate,
            rate_mode,
            fps,
            rans,
            rice,
            diagnostics,
            temporal_wavelet,
            tw_highpass_mul,
            ab,
            output,
        } => {
            if diagnostics {
                gnc::encoder::diagnostics::enable();
            }
            let qstep_display = qstep
                .map(|q| format!("{}", q))
                .or_else(|| quality.map(|q| format!("q={}", q)))
                .unwrap_or_else(|| "4.0".to_string());
            let rc_display = if let Some(ref br) = bitrate {
                format!(", rate_control={} {}", br, rate_mode)
            } else {
                String::new()
            };
            println!(
                "Sequence benchmark: pattern={}, frames={}, ki={}, {}{}",
                input, num_frames, keyframe_interval, qstep_display, rc_display
            );

            let ctx = GpuContext::new();
            let mut encoder = EncoderPipeline::new(&ctx);
            let decoder = DecoderPipeline::new(&ctx);

            let (temporal_mode, auto_gop) =
                select_temporal_mode(&temporal_wavelet, fps, quality.unwrap_or(75));
            let run_temporal = temporal_mode != TemporalTransform::None;
            let run_baseline = !run_temporal || ab;

            // Determine GOP size: auto_gop from select_temporal_mode, or keyframe_interval for explicit Haar
            let temporal_gop_size = if temporal_mode == TemporalTransform::LeGall53 {
                4usize
            } else if auto_gop > 0 {
                auto_gop
            } else {
                keyframe_interval as usize
            };

            if run_temporal
                && temporal_mode == TemporalTransform::Haar
                && !gnc::temporal::is_power_of_two(temporal_gop_size)
            {
                eprintln!(
                    "Temporal Haar requires GOP size to be power of two (got {}).",
                    temporal_gop_size
                );
                std::process::exit(1);
            }

            // Streaming path: when using temporal wavelet without A/B comparison,
            // load frames per-GOP to avoid holding all frames in memory.
            // For 1800 frames at 1080p, loading all = ~44 GB; per-GOP (8) = ~192 MB.
            // Supports both PNG frame patterns (%04d) and Y4M files (.y4m).
            if run_temporal && !run_baseline {
                // Detect input format: Y4M if path ends with ".y4m"
                let use_y4m = input.ends_with(".y4m");

                // Obtain dimensions (and, for Y4M, fps from the header).
                // For Y4M we keep the reader open so we can reuse it for warmup frames,
                // avoiding a separate probe-only open.
                let (w, h, y4m_fps, y4m_probe_reader) = if use_y4m {
                    let probe = Y4mReader::open(&input);
                    let fw = probe.width;
                    let fh = probe.height;
                    let y4m_fps_val =
                        probe.fps_num as f64 / probe.fps_den.max(1) as f64;
                    (fw, fh, Some(y4m_fps_val), Some(probe))
                } else {
                    let first_path = input.replace("%04d", &format!("{:04}", 0));
                    let (_, fw, fh) = load_image_rgb_f32(&first_path);
                    (fw, fh, None, None)
                };
                // Use fps from Y4M header when available; otherwise keep CLI --fps value.
                let effective_fps = y4m_fps.unwrap_or(fps);
                let gop_size = temporal_gop_size;

                let mut config_tw = if let Some(q) = quality {
                    gnc::quality_preset(q)
                } else {
                    CodecConfig {
                        quantization_step: qstep.unwrap_or(4.0),
                        ..Default::default()
                    }
                };
                if let Some(qs) = qstep {
                    config_tw.quantization_step = qs;
                }
                config_tw.keyframe_interval = keyframe_interval;
                config_tw.temporal_transform = temporal_mode;
                config_tw.target_bitrate = None;
                if let Some(mul) = tw_highpass_mul {
                    config_tw.temporal_highpass_qstep_mul = mul;
                    config_tw.adaptive_temporal_mul = false;
                }
                if rans {
                    config_tw.entropy_coder = gnc::EntropyCoder::Rans;
                }
                if rice {
                    config_tw.entropy_coder = gnc::EntropyCoder::Rice;
                }
                println!(
                    "\n=== Temporal wavelet ({:?}, streaming, {}) ===",
                    temporal_mode,
                    if use_y4m { "Y4M" } else { "PNG" },
                );
                if let Some(y4m_fps_val) = y4m_fps {
                    println!("  Y4M fps from header: {:.3}", y4m_fps_val);
                }
                println!(
                    "Temporal config: qstep {:.3}, dead_zone {:.3}, entropy {:?}, adaptive_mul {}",
                    config_tw.quantization_step, config_tw.dead_zone, config_tw.entropy_coder,
                    if config_tw.adaptive_temporal_mul { "on" } else { "off" },
                );

                // ---------------------------------------------------------------------------
                // GPU warmup — triggers Metal JIT shader compilation for BOTH pipelines
                // so that GOP 0 timing is not inflated by lazy compilation.
                //
                // Step 1: warm up I-frame path (encoder.encode)
                // Step 2: warm up temporal Haar path (encode_temporal_wavelet_gop)
                // ---------------------------------------------------------------------------
                let warmup_cfg = if let Some(q) = quality {
                    gnc::quality_preset(q)
                } else {
                    CodecConfig {
                        quantization_step: qstep.unwrap_or(4.0),
                        ..Default::default()
                    }
                };

                // Read enough frames for warmup (gop_size frames for temporal warmup)
                // without counting them in the benchmark.
                //
                // For Y4M: reuse the probe reader (already positioned at frame 0) so we
                // avoid a third open of the same file.  The encode stream (y4m_stream below)
                // is a separate, independent open that always restarts from frame 0.
                let warmup_frames: Vec<Vec<f32>> = if let Some(mut y4m_warmup) = y4m_probe_reader {
                    (0..gop_size)
                        .filter_map(|_| y4m_warmup.read_frame_rgb())
                        .collect()
                } else {
                    (0..gop_size)
                        .map(|j| {
                            let path = input.replace("%04d", &format!("{:04}", j));
                            let (rgb, _, _) = load_image_rgb_f32(&path);
                            rgb
                        })
                        .collect()
                };

                // Step 1: I-frame warmup
                let _ = encoder.encode(&ctx, &warmup_frames[0], w, h, &warmup_cfg);

                // Step 2: temporal Haar warmup (warms shaders used in encode_temporal_wavelet_gop)
                if warmup_frames.len() >= gop_size {
                    let wf_refs: Vec<&[f32]> =
                        warmup_frames.iter().map(|f| f.as_slice()).collect();
                    let _ = encoder.encode_temporal_wavelet_gop(
                        &ctx,
                        &wf_refs,
                        w,
                        h,
                        &warmup_cfg,
                        temporal_mode,
                        None,
                    );
                }
                drop(warmup_frames);

                let mut groups: Vec<gnc::TemporalGroup> = Vec::new();
                let mut group_gop_indices: Vec<usize> = Vec::new();
                let mut tail_iframes: Vec<CompressedFrame> = Vec::new();
                let mut tail_iframe_pts: Vec<u32> = Vec::new();
                let mut total_bytes: usize = 0;
                let num_gops = num_frames / gop_size;
                let tail_start = num_gops * gop_size;

                let profile_split = std::env::var("GNC_PROFILE_SPLIT").is_ok();
                let io_label = if use_y4m { "y4m_read" } else { "png_decode" };
                let start = std::time::Instant::now();
                let mut total_io_ms: f64 = 0.0;
                let mut total_enc_ms: f64 = 0.0;

                // For Y4M: open the file once and stream frames sequentially.
                // For PNG: re-derive paths from the pattern per-GOP.
                let mut y4m_stream: Option<Y4mReader> = if use_y4m {
                    Some(Y4mReader::open(&input))
                } else {
                    None
                };

                // Lookahead buffer for async upload pipelining (Haar only).
                // Holds next GOP's frames pre-loaded during the current GOP's encode.
                // The encoder writes these to GPU staging during high_enc (~100ms),
                // hiding the ~22ms write_buffer cost for the subsequent GOP.
                let mut lookahead_frames: Option<Vec<Vec<f32>>> = None;

                // Decoder for per-GOP PSNR in diagnostics mode. Instantiated once and reused.
                // Only active when --diagnostics is set; adds decode overhead per GOP.
                let diag_decoder = if diagnostics {
                    Some(DecoderPipeline::new(&ctx))
                } else {
                    None
                };
                let mut diag_psnr_vals: Vec<f64> = Vec::new();
                let mut diag_steady_fps: Vec<f64> = Vec::new(); // per-GOP fps, excl GOP 0 warmup

                for gop_idx in 0..num_gops {
                    let base = gop_idx * gop_size;
                    // Load this GOP's frames (or use pre-loaded lookahead from previous iteration)
                    let t_io_start = std::time::Instant::now();
                    let gop_frames: Vec<Vec<f32>> = if let Some(preloaded) = lookahead_frames.take() {
                        preloaded // skip I/O; already loaded in previous iteration's lookahead
                    } else {
                        let mut frames = Vec::with_capacity(gop_size);
                        if let Some(ref mut y4m) = y4m_stream {
                            for _ in 0..gop_size {
                                match y4m.read_frame_rgb() {
                                    Some(rgb) => frames.push(rgb),
                                    None => break,
                                }
                            }
                        } else {
                            for j in 0..gop_size {
                                let path = input.replace("%04d", &format!("{:04}", base + j));
                                let (rgb, _, _) = load_image_rgb_f32(&path);
                                frames.push(rgb);
                            }
                        }
                        frames
                    };
                    let gop_io_ms = t_io_start.elapsed().as_secs_f64() * 1000.0;
                    total_io_ms += gop_io_ms;

                    if gop_frames.len() < gop_size {
                        eprintln!(
                            "  Warning: Y4M EOF at GOP {}, only {} frames (expected {}). Stopping.",
                            gop_idx, gop_frames.len(), gop_size
                        );
                        break;
                    }

                    // Scene cut detection: if any adjacent pair in this GOP has a large
                    // luma change, the temporal transform would blend content from two
                    // incompatible scenes. Encode all frames as I-frames instead and let
                    // the next GOP start fresh.
                    if matches!(temporal_mode, TemporalTransform::Haar) {
                        let cut_pos = gop_frames.windows(2).enumerate().find_map(|(i, pair)| {
                            if scene_cut_score(&pair[0], &pair[1]) > SCENE_CUT_THRESH {
                                Some(i + 1)
                            } else {
                                None
                            }
                        });
                        if let Some(cut_at) = cut_pos {
                            let mut i_cfg = config_tw.clone();
                            i_cfg.temporal_transform = gnc::TemporalTransform::None;
                            i_cfg.keyframe_interval = 1;
                            i_cfg.cfl_enabled = false;
                            let t_enc_start = std::time::Instant::now();
                            for (fi, frame) in gop_frames.iter().enumerate() {
                                let pts = (base + fi) as u32;
                                let cf = encoder.encode(&ctx, frame, w, h, &i_cfg);
                                total_bytes += cf.byte_size();
                                tail_iframes.push(cf);
                                tail_iframe_pts.push(pts);
                            }
                            total_enc_ms += t_enc_start.elapsed().as_secs_f64() * 1000.0;
                            // Clear stale lookahead so next GOP loads fresh from y4m.
                            lookahead_frames = None;
                            eprintln!(
                                "  [scene cut] GOP {:3}: cut at frame {} (pos {}/{}), {} frames → I-frames",
                                gop_idx, base + cut_at, cut_at, gop_size, gop_size
                            );
                            continue;
                        }
                    }

                    // Pre-load next GOP for async upload (Haar mode only).
                    // Reads are added to IO time but overlap with encode time via GPU pre-staging.
                    if matches!(temporal_mode, TemporalTransform::Haar) && gop_idx + 1 < num_gops {
                        let t_la_start = std::time::Instant::now();
                        let next_base = (gop_idx + 1) * gop_size;
                        let mut next_frames = Vec::with_capacity(gop_size);
                        if let Some(ref mut y4m) = y4m_stream {
                            for _ in 0..gop_size {
                                match y4m.read_frame_rgb() {
                                    Some(rgb) => next_frames.push(rgb),
                                    None => break,
                                }
                            }
                        } else {
                            for j in 0..gop_size {
                                let path = input.replace("%04d", &format!("{:04}", next_base + j));
                                let (rgb, _, _) = load_image_rgb_f32(&path);
                                next_frames.push(rgb);
                            }
                        }
                        total_io_ms += t_la_start.elapsed().as_secs_f64() * 1000.0;
                        if next_frames.len() == gop_size {
                            lookahead_frames = Some(next_frames);
                        }
                    }

                    let gop_refs: Vec<&[f32]> = gop_frames.iter().map(|f| f.as_slice()).collect();
                    let next_refs: Option<Vec<&[f32]>> = lookahead_frames.as_ref()
                        .map(|v| v.iter().map(|f| f.as_slice()).collect());

                    let t_enc_start = std::time::Instant::now();
                    let group = encoder.encode_temporal_wavelet_gop(
                        &ctx,
                        &gop_refs,
                        w,
                        h,
                        &config_tw,
                        temporal_mode,
                        next_refs.as_deref(),
                    );
                    let gop_enc_ms = t_enc_start.elapsed().as_secs_f64() * 1000.0;
                    total_enc_ms += gop_enc_ms;

                    let low_bytes = group.low_frame.byte_size();
                    let high_bytes: usize = group.high_frames.iter()
                        .flat_map(|lvl| lvl.iter())
                        .map(|f| f.byte_size())
                        .sum();
                    let group_bytes = low_bytes + high_bytes;
                    total_bytes += group_bytes;

                    let elapsed = start.elapsed();
                    let frames_done = (gop_idx + 1) * gop_size;
                    let fps_enc = frames_done as f64 / elapsed.as_secs_f64();

                    if diagnostics || gop_idx % 10 == 0 || gop_idx == num_gops - 1 {
                        eprintln!(
                            "  GOP {:3}/{}: {:6} bytes (low {:5}, high {:5}), {}/{} frames, {:.1} fps  [{:.0}ms enc]",
                            gop_idx + 1,
                            num_gops,
                            group_bytes,
                            low_bytes,
                            high_bytes,
                            frames_done,
                            num_frames,
                            fps_enc,
                            gop_enc_ms,
                        );
                    }
                    if gop_idx > 0 {
                        // Exclude GOP 0 (GPU JIT warmup skews fps high)
                        diag_steady_fps.push(gop_size as f64 / (gop_enc_ms / 1000.0));
                    }

                    if profile_split {
                        let gop_total_ms = gop_io_ms + gop_enc_ms;
                        let io_pct = gop_io_ms / gop_total_ms * 100.0;
                        let enc_pct = gop_enc_ms / gop_total_ms * 100.0;
                        eprintln!(
                            "SPLIT GOP {:3}: {}={:.1}ms ({:.0}%), gnc_encode={:.1}ms ({:.0}%), gop_total={:.1}ms",
                            gop_idx, io_label, gop_io_ms, io_pct, gop_enc_ms, enc_pct, gop_total_ms,
                        );
                    }

                    // Full per-GOP diagnostics (decomposition, coefficients, warnings)
                    if gnc::encoder::diagnostics::enabled() {
                        let q = quality.unwrap_or(75);
                        let mul = tw_highpass_mul.unwrap_or(config_tw.temporal_highpass_qstep_mul);

                        // Decode the GOP to compute per-frame PSNR.
                        // gop_frames is still in scope (owned, not moved).
                        let per_frame_q: Vec<(f64, f64)> = if let Some(ref dec) = diag_decoder {
                            let single_seq = gnc::TemporalEncodedSequence {
                                mode: temporal_mode,
                                groups: vec![group.clone()],
                                group_gop_indices: vec![0],
                                tail_iframes: vec![],
                                tail_iframe_pts: vec![],
                                frame_count: gop_size,
                                gop_size,
                            };
                            let decoded = dec.decode_temporal_sequence(&ctx, &single_seq);
                            gop_frames.iter().zip(decoded.iter())
                                .map(|(orig, dec_frame)| {
                                    let psnr = quality::psnr(orig, dec_frame, 255.0);
                                    let ssim = quality::ssim_approx(orig, dec_frame, 255.0);
                                    (psnr, ssim)
                                })
                                .collect()
                        } else {
                            vec![]
                        };
                        for &(psnr, _) in &per_frame_q {
                            if psnr.is_finite() { diag_psnr_vals.push(psnr); }
                        }

                        gnc::encoder::diagnostics::print_temporal_gop_diagnostics(
                            gop_idx,
                            gop_size,
                            temporal_mode,
                            q,
                            mul,
                            &group,
                            &per_frame_q,
                            None,
                            w,
                            h,
                        );
                    }

                    groups.push(group);
                    group_gop_indices.push(gop_idx);
                    // gop_frames dropped here — memory freed
                }

                // Tail frames (not enough for a full GOP): encode as I-frames.
                // For Y4M the stream is already positioned at tail_start; for PNG derive paths.
                let mut tail_cfg = config_tw.clone();
                tail_cfg.keyframe_interval = 1;
                tail_cfg.temporal_transform = TemporalTransform::None;
                tail_cfg.cfl_enabled = false;
                for i in tail_start..num_frames {
                    let t_io_start = std::time::Instant::now();
                    let rgb = if let Some(ref mut y4m) = y4m_stream {
                        match y4m.read_frame_rgb() {
                            Some(f) => f,
                            None => break,
                        }
                    } else {
                        let path = input.replace("%04d", &format!("{:04}", i));
                        let (f, _, _) = load_image_rgb_f32(&path);
                        f
                    };
                    total_io_ms += t_io_start.elapsed().as_secs_f64() * 1000.0;
                    let t_enc_start = std::time::Instant::now();
                    let cf = encoder.encode(&ctx, &rgb, w, h, &tail_cfg);
                    total_enc_ms += t_enc_start.elapsed().as_secs_f64() * 1000.0;
                    total_bytes += cf.byte_size();
                    tail_iframe_pts.push(i as u32);
                    tail_iframes.push(cf);
                }

                let elapsed = start.elapsed();
                if profile_split {
                    let total_measured_ms = total_io_ms + total_enc_ms;
                    let wall_ms = elapsed.as_secs_f64() * 1000.0;
                    let other_ms = wall_ms - total_measured_ms;
                    let enc_fps = (num_frames as f64) / (total_enc_ms / 1000.0);
                    eprintln!(
                        "SPLIT Total: wall={:.1}ms, {}={:.1}ms ({:.0}%), gnc_encode={:.1}ms ({:.0}%), other={:.1}ms ({:.0}%)",
                        wall_ms,
                        io_label, total_io_ms, total_io_ms / wall_ms * 100.0,
                        total_enc_ms, total_enc_ms / wall_ms * 100.0,
                        other_ms, other_ms / wall_ms * 100.0,
                    );
                    eprintln!(
                        "SPLIT GNC-only fps: {:.1} fps ({} frames / {:.1}ms gnc_encode time)",
                        enc_fps, num_frames, total_enc_ms,
                    );
                }
                let avg_bpp = (total_bytes as f64 * 8.0) / (w as f64 * h as f64) / num_frames as f64;
                println!(
                    "  Total: {} bytes ({:.2} MB), avg {:.2} bpp, {:.1}ms ({:.1} fps)",
                    total_bytes,
                    total_bytes as f64 / (1024.0 * 1024.0),
                    avg_bpp,
                    elapsed.as_secs_f64() * 1000.0,
                    num_frames as f64 / elapsed.as_secs_f64(),
                );
                println!(
                    "  {} GOPs + {} tail I-frames",
                    groups.len(),
                    tail_iframes.len(),
                );

                let encoded_tw = gnc::TemporalEncodedSequence {
                    mode: temporal_mode,
                    groups,
                    group_gop_indices,
                    tail_iframes,
                    tail_iframe_pts,
                    frame_count: num_frames,
                    gop_size,
                };

                // Write GNV2 if output path provided
                if let Some(ref output_path) = output {
                    // Use effective_fps (Y4M header fps when available, else CLI --fps)
                    // so the bitstream reflects the true playback rate of the source.
                    let fps_num = effective_fps.round() as u32;
                    let fps_den = 1u32;
                    let gnv2_data = serialize_temporal_sequence(&encoded_tw, (fps_num, fps_den));
                    std::fs::write(output_path, &gnv2_data).expect("Failed to write GNV2 output");
                    println!(
                        "Wrote GNV2: {} bytes ({:.2} MB) → {}",
                        gnv2_data.len(),
                        gnv2_data.len() as f64 / (1024.0 * 1024.0),
                        output_path,
                    );
                }

                // Diagnostics summary block — printed after all GOPs when --diagnostics is set
                if diagnostics && !diag_psnr_vals.is_empty() {
                    let mean_psnr = diag_psnr_vals.iter().sum::<f64>() / diag_psnr_vals.len() as f64;
                    let min_psnr = diag_psnr_vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_psnr = diag_psnr_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mean_enc_fps = if !diag_steady_fps.is_empty() {
                        diag_steady_fps.iter().sum::<f64>() / diag_steady_fps.len() as f64
                    } else {
                        0.0
                    };
                    let avg_bpp = (total_bytes as f64 * 8.0) / (w as f64 * h as f64) / num_frames as f64;
                    eprintln!(
                        "\n=== Diagnostics Summary ===\n  frames={} GOPs={} avg_bpp={:.2}\n  psnr_avg={:.2} dB  psnr_min={:.2} dB  psnr_max={:.2} dB\n  enc_fps_steady={:.1} fps (excl GOP 0 warmup)",
                        num_frames, num_gops, avg_bpp, mean_psnr, min_psnr, max_psnr, mean_enc_fps,
                    );
                }
                return;
            }

            // Non-streaming path: load all frames into memory
            let mut frames_data: Vec<Vec<f32>> = Vec::new();
            let mut w = 0u32;
            let mut h = 0u32;
            for i in 0..num_frames {
                let path = input.replace("%04d", &format!("{:04}", i));
                let (rgb, fw, fh) = load_image_rgb_f32(&path);
                w = fw;
                h = fh;
                frames_data.push(rgb);
            }

            let frame_refs: Vec<&[f32]> = frames_data.iter().map(|f| f.as_slice()).collect();

            #[allow(unused_assignments)]
            let mut summary_ip: Option<sequence_metrics::SequenceSummary> = None;
            let mut total_bytes_ip: usize = 0;
            let mut avg_bpp_ip: f64 = 0.0;
            let mut frame_metrics_ip: Vec<FrameMetrics> = Vec::new();

            if run_baseline {
            let mut frame_metrics_i: Vec<FrameMetrics> = Vec::new();
            // --- I+P encoding ---
            let mut config_ip = if let Some(q) = quality {
                gnc::quality_preset(q)
            } else {
                CodecConfig {
                    quantization_step: qstep.unwrap_or(4.0),
                    ..Default::default()
                }
            };
            if let Some(qs) = qstep {
                config_ip.quantization_step = qs;
            }
            config_ip.keyframe_interval = keyframe_interval;
            if rans {
                config_ip.entropy_coder = gnc::EntropyCoder::Rans;
            }
            if rice {
                config_ip.entropy_coder = gnc::EntropyCoder::Rice;
            }

            // Apply rate control settings
            if let Some(ref br) = bitrate {
                config_ip.target_bitrate = Some(parse_bitrate(br));
                config_ip.rate_mode = parse_rate_mode(&rate_mode);
            }

            // Warm up GPU shader pipelines (triggers Metal lazy compilation)
            let _ = encoder.encode(&ctx, &frames_data[0], w, h, &config_ip);

            let start = std::time::Instant::now();
            let compressed_ip =
                encoder.encode_sequence_with_fps(&ctx, &frame_refs, w, h, &config_ip, fps);
            let elapsed_ip = start.elapsed();

            println!("\n=== I+P+B (keyframe_interval={}) ===", keyframe_interval);
            // Decode with B-frame reordering support
            let decoded_all = decoder.decode_sequence(&ctx, &compressed_ip);

            // DIVERGENCE TEST: Compare encoder vs decoder decoded output (RGB)
            // Note: cannot compare reference planes directly because the encoder
            // skips local decode for the last P-frame in a sequence (optimization:
            // the reference won't be used again). This means encoder gpu_ref_planes
            // may be stale. Instead, compare decoded RGB output from both sides.
            if diagnostics {
                println!("\n=== ENCODE/DECODE QUALITY CHECK ===");
                for (i, cf) in compressed_ip.iter().enumerate() {
                    let ft = match cf.frame_type {
                        gnc::FrameType::Intra => "I",
                        gnc::FrameType::Predicted => "P",
                        gnc::FrameType::Bidirectional => "B",
                    };
                    let psnr = quality::psnr(&frames_data[i], &decoded_all[i], 255.0);
                    let status = if psnr > 25.0 { "✓ OK" } else { "⚠ LOW" };
                    eprintln!(
                        "  Frame {:2} [{}] {}: PSNR={:.2} dB, {} bytes",
                        i, ft, status, psnr, cf.byte_size()
                    );
                }
            }

            total_bytes_ip = 0;
            frame_metrics_ip.clear();
            for (i, cf) in compressed_ip.iter().enumerate() {
                let ft = match cf.frame_type {
                    gnc::FrameType::Intra => "I",
                    gnc::FrameType::Predicted => "P",
                    gnc::FrameType::Bidirectional => "B",
                };
                let psnr = quality::psnr(&frames_data[i], &decoded_all[i], 255.0);
                let ssim = quality::ssim_approx(&frames_data[i], &decoded_all[i], 255.0);
                total_bytes_ip += cf.byte_size();
                println!(
                    "  Frame {:2} [{}]: {:6} bytes, {:.2} bpp, PSNR {:.2} dB, SSIM {:.4}",
                    i,
                    ft,
                    cf.byte_size(),
                    cf.bpp(),
                    psnr,
                    ssim,
                );
                frame_metrics_ip.push(FrameMetrics {
                    frame_idx: i,
                    frame_type: ft.to_string(),
                    psnr,
                    ssim,
                    bpp: cf.bpp(),
                    encoded_bytes: cf.byte_size(),
                });
            }

            avg_bpp_ip =
                compressed_ip.iter().map(|f| f.bpp()).sum::<f64>() / compressed_ip.len() as f64;
            let i_count = compressed_ip
                .iter()
                .filter(|f| f.frame_type == gnc::FrameType::Intra)
                .count();
            let p_count = compressed_ip
                .iter()
                .filter(|f| f.frame_type == gnc::FrameType::Predicted)
                .count();
            let b_count = compressed_ip
                .iter()
                .filter(|f| f.frame_type == gnc::FrameType::Bidirectional)
                .count();

            println!(
                "  Total: {} bytes, avg {:.2} bpp, {:.1}ms ({:.1} fps), {}I+{}P+{}B",
                total_bytes_ip,
                avg_bpp_ip,
                elapsed_ip.as_secs_f64() * 1000.0,
                compressed_ip.len() as f64 / elapsed_ip.as_secs_f64(),
                i_count,
                p_count,
                b_count,
            );

            // Compute and display I+P sequence summary
            let summary_ip_local = sequence_metrics::compute_sequence_metrics(&frame_metrics_ip);
            summary_ip = Some(summary_ip_local.clone());
            println!("\n{}", summary_ip_local);

            // --- All I-frame baseline ---
            let mut config_i = config_ip.clone();
            config_i.keyframe_interval = 1;

            let start = std::time::Instant::now();
            let compressed_i = encoder.encode_sequence(&ctx, &frame_refs, w, h, &config_i);
            let elapsed_i = start.elapsed();

            println!("=== All I-frames (baseline) ===");
            let mut total_bytes_i: usize = 0;
            frame_metrics_i.clear();
            for (i, cf) in compressed_i.iter().enumerate() {
                let decoded = decoder.decode(&ctx, cf);
                let psnr = quality::psnr(&frames_data[i], &decoded, 255.0);
                let ssim = quality::ssim_approx(&frames_data[i], &decoded, 255.0);
                total_bytes_i += cf.byte_size();
                println!(
                    "  Frame {:2} [I]: {:6} bytes, {:.2} bpp, PSNR {:.2} dB, SSIM {:.4}",
                    i,
                    cf.byte_size(),
                    cf.bpp(),
                    psnr,
                    ssim,
                );
                frame_metrics_i.push(FrameMetrics {
                    frame_idx: i,
                    frame_type: "I".to_string(),
                    psnr,
                    ssim,
                    bpp: cf.bpp(),
                    encoded_bytes: cf.byte_size(),
                });
            }

            let avg_bpp_i =
                compressed_i.iter().map(|f| f.bpp()).sum::<f64>() / compressed_i.len() as f64;

            println!(
                "  Total: {} bytes, avg {:.2} bpp, {:.1}ms ({:.1} fps)",
                total_bytes_i,
                avg_bpp_i,
                elapsed_i.as_secs_f64() * 1000.0,
                compressed_i.len() as f64 / elapsed_i.as_secs_f64(),
            );

            // Compute and display I-only sequence summary
            let summary_i = sequence_metrics::compute_sequence_metrics(&frame_metrics_i);
            println!("\n{}", summary_i);

            // --- Comparison ---
            let saving_pct = (1.0 - total_bytes_ip as f64 / total_bytes_i as f64) * 100.0;
            println!(
                "=== Comparison ===\n  I-only: {} bytes ({:.2} bpp)\n  I+P:    {} bytes ({:.2} bpp)\n  Saving: {:.1}%",
                total_bytes_i, avg_bpp_i, total_bytes_ip, avg_bpp_ip, saving_pct,
            );

            // Temporal consistency comparison
            println!(
                "\n=== Temporal Consistency ===\n  I+P:    max PSNR drop {:.2} dB, consistency {:.2} dB\n  I-only: max PSNR drop {:.2} dB, consistency {:.2} dB",
                summary_ip.as_ref().unwrap().max_psnr_drop,
                summary_ip.as_ref().unwrap().temporal_consistency,
                summary_i.max_psnr_drop,
                summary_i.temporal_consistency,
            );

            // Write CSV if requested (I+P metrics — the primary encoding mode)
            if let Some(ref csv_path) = csv {
                let out_path = if run_temporal {
                    csv_with_suffix(csv_path, "ip")
                } else {
                    csv_path.clone()
                };
                sequence_metrics::write_sequence_csv(
                    &out_path,
                    &frame_metrics_ip,
                    summary_ip.as_ref().unwrap(),
                )
                    .expect("Failed to write sequence CSV");
                println!("\nSequence metrics written to {}", out_path);
            }
            }

            if run_temporal {
                // --- Temporal wavelet encoding (in-memory) ---
                let mut config_tw = if let Some(q) = quality {
                    gnc::quality_preset(q)
                } else {
                    CodecConfig {
                        quantization_step: qstep.unwrap_or(4.0),
                        ..Default::default()
                    }
                };
                if let Some(qs) = qstep {
                    config_tw.quantization_step = qs;
                }
                config_tw.keyframe_interval = keyframe_interval;
                config_tw.temporal_transform = temporal_mode;
                config_tw.target_bitrate = None; // rate control is sequence-based; disable for temporal mode
                if let Some(mul) = tw_highpass_mul {
                    config_tw.temporal_highpass_qstep_mul = mul;
                    config_tw.adaptive_temporal_mul = false;
                }
                if rans {
                    config_tw.entropy_coder = gnc::EntropyCoder::Rans;
                }
                if rice {
                    config_tw.entropy_coder = gnc::EntropyCoder::Rice;
                }
                println!(
                    "Temporal config: qstep {:.3}, dead_zone {:.3}, entropy {:?}, adaptive_mul {}",
                    config_tw.quantization_step, config_tw.dead_zone, config_tw.entropy_coder,
                    if config_tw.adaptive_temporal_mul { "on" } else { "off" },
                );

                let start = std::time::Instant::now();
                let encoded_tw = encoder.encode_sequence_temporal_wavelet(
                    &ctx,
                    &frame_refs,
                    w,
                    h,
                    &config_tw,
                    temporal_mode,
                    temporal_gop_size,
                );
                let elapsed_tw = start.elapsed();

                if let Some(group0) = encoded_tw.groups.first() {
                    let enc_cfg = &group0.low_frame.config;
                    println!(
                        "  Temporal encode cfg (low): qstep {:.3}, dead_zone {:.3}, wavelet_levels {}, subband_weights {:?}",
                        enc_cfg.quantization_step,
                        enc_cfg.dead_zone,
                        enc_cfg.wavelet_levels,
                        enc_cfg.subband_weights
                    );
                    println!(
                        "  Temporal decode cfg (low): qstep {:.3}, dead_zone {:.3}, wavelet_levels {}, subband_weights {:?}",
                        enc_cfg.quantization_step,
                        enc_cfg.dead_zone,
                        enc_cfg.wavelet_levels,
                        enc_cfg.subband_weights
                    );
                    if let Some(first_lvl) = group0.high_frames.first() {
                        if let Some(first_high) = first_lvl.first() {
                            let high_cfg = &first_high.config;
                            println!(
                                "  Temporal encode cfg (high L0[0]): qstep {:.3}, dead_zone {:.3}, wavelet_levels {}, subband_weights {:?}",
                                high_cfg.quantization_step,
                                high_cfg.dead_zone,
                                high_cfg.wavelet_levels,
                                high_cfg.subband_weights
                            );
                            println!(
                                "  Temporal decode cfg (high L0[0]): qstep {:.3}, dead_zone {:.3}, wavelet_levels {}, subband_weights {:?}",
                                high_cfg.quantization_step,
                                high_cfg.dead_zone,
                                high_cfg.wavelet_levels,
                                high_cfg.subband_weights
                            );
                        }
                    }
                }

                // Sanity: compare wavelet coeff roundtrip for first GOP
                let gop = encoded_tw.gop_size.min(frames_data.len());
                if gop >= 2 {
                    let info = gnc::FrameInfo {
                        width: w,
                        height: h,
                        bit_depth: 8,
                        tile_size: config_tw.tile_size,
                    };
                    let mut originals: Vec<[Vec<f32>; 3]> = Vec::new();
                    for fd in &frames_data[..gop] {
                        originals.push(encoder.debug_wavelet_prequant(
                            &ctx,
                            fd,
                            &info,
                            &config_tw,
                        ));
                    }
                    // Pure temporal Haar roundtrip on wavelet coeffs (no quant/entropy)
                    if temporal_mode == TemporalTransform::Haar {
                        let y_in: Vec<&[f32]> = originals.iter().map(|v| v[0].as_slice()).collect();
                        let co_in: Vec<&[f32]> =
                            originals.iter().map(|v| v[1].as_slice()).collect();
                        let cg_in: Vec<&[f32]> =
                            originals.iter().map(|v| v[2].as_slice()).collect();
                        let (low_y, highs_y) = gnc::temporal::haar_multilevel_forward(&y_in);
                        let (low_co, highs_co) = gnc::temporal::haar_multilevel_forward(&co_in);
                        let (low_cg, highs_cg) = gnc::temporal::haar_multilevel_forward(&cg_in);
                        let y_rt = gnc::temporal::haar_multilevel_inverse(&low_y, &highs_y);
                        let co_rt = gnc::temporal::haar_multilevel_inverse(&low_co, &highs_co);
                        let cg_rt = gnc::temporal::haar_multilevel_inverse(&low_cg, &highs_cg);

                        let mut mean_abs = [0.0f64; 3];
                        let mut count = 0usize;
                        for i in 0..gop.min(y_rt.len()) {
                            for (p, (orig, rt)) in [
                                (&originals[i][0], &y_rt[i]),
                                (&originals[i][1], &co_rt[i]),
                                (&originals[i][2], &cg_rt[i]),
                            ]
                            .iter()
                            .enumerate()
                            {
                                let n = orig.len().min(rt.len());
                                let mut sum = 0.0f64;
                                for j in 0..n {
                                    sum += (orig[j] - rt[j]).abs() as f64;
                                }
                                mean_abs[p] += sum / n as f64;
                            }
                            count += 1;
                        }
                        if count > 0 {
                            println!(
                                "  Temporal Haar coeff roundtrip (no quant/entropy): mean_abs_diff Y {:.6}, Co {:.6}, Cg {:.6}",
                                mean_abs[0] / count as f64,
                                mean_abs[1] / count as f64,
                                mean_abs[2] / count as f64
                            );
                        }

                        // Entropy roundtrip check: quantize vs entropy decode
                        if !encoded_tw.groups.is_empty() {
                            let group = &encoded_tw.groups[0];
                            let tiles_x = info.tiles_x() as usize;
                            let tiles_y = info.tiles_y() as usize;
                            let tiles_per_plane = tiles_x * tiles_y;
                            let padded_w = info.padded_width() as usize;
                            let tile_size = config_tw.tile_size as usize;

                            let low_quant = encoder.debug_quantize_wavelet_coeffs(
                                &ctx,
                                [&low_y, &low_co, &low_cg],
                                &info,
                                &config_tw,
                            );

                            let mut mean_abs_q = [0.0f64; 3];
                            let mut max_abs_q = [0.0f64; 3];
                            for p in 0..3 {
                                let dec = encoder.debug_entropy_decode_plane(
                                    &group.low_frame.entropy,
                                    p,
                                    tiles_per_plane,
                                    tile_size,
                                    padded_w,
                                );
                                let a = &low_quant[p];
                                let n = a.len().min(dec.len());
                                let mut sum = 0.0f64;
                                let mut maxv = 0.0f64;
                                for j in 0..n {
                                    let d = (a[j] - dec[j]).abs() as f64;
                                    sum += d;
                                    if d > maxv {
                                        maxv = d;
                                    }
                                }
                                mean_abs_q[p] = sum / n as f64;
                                max_abs_q[p] = maxv;
                            }
                            println!(
                                "  Entropy roundtrip (low frame) mean_abs_diff Y {:.6}, Co {:.6}, Cg {:.6} | max_abs Y {:.3}, Co {:.3}, Cg {:.3}",
                                mean_abs_q[0],
                                mean_abs_q[1],
                                mean_abs_q[2],
                                max_abs_q[0],
                                max_abs_q[1],
                                max_abs_q[2]
                            );

                            if let Some(first_lvl) = group.high_frames.first() {
                                if let Some(first_high) = first_lvl.first() {
                                    let high_y = &highs_y[0][0];
                                    let high_co = &highs_co[0][0];
                                    let high_cg = &highs_cg[0][0];
                                    let high_quant = encoder.debug_quantize_wavelet_coeffs(
                                        &ctx,
                                        [high_y, high_co, high_cg],
                                        &info,
                                        &config_tw,
                                    );
                                    let mut mean_abs_h = [0.0f64; 3];
                                    let mut max_abs_h = [0.0f64; 3];
                                    for p in 0..3 {
                                        let dec = encoder.debug_entropy_decode_plane(
                                            &first_high.entropy,
                                            p,
                                            tiles_per_plane,
                                            tile_size,
                                            padded_w,
                                        );
                                        let a = &high_quant[p];
                                        let n = a.len().min(dec.len());
                                        let mut sum = 0.0f64;
                                        let mut maxv = 0.0f64;
                                        for j in 0..n {
                                            let d = (a[j] - dec[j]).abs() as f64;
                                            sum += d;
                                            if d > maxv {
                                                maxv = d;
                                            }
                                        }
                                        mean_abs_h[p] = sum / n as f64;
                                        max_abs_h[p] = maxv;
                                    }
                                    println!(
                                        "  Entropy roundtrip (high L0[0]) mean_abs_diff Y {:.6}, Co {:.6}, Cg {:.6} | max_abs Y {:.3}, Co {:.3}, Cg {:.3}",
                                        mean_abs_h[0],
                                        mean_abs_h[1],
                                        mean_abs_h[2],
                                        max_abs_h[0],
                                        max_abs_h[1],
                                        max_abs_h[2]
                                    );
                                }
                            }
                        }
                    }
                    let recon_coeffs = decoder.decode_temporal_wavelet_coeffs(&ctx, &encoded_tw);
                    let mut mean_abs = [0.0f64; 3];
                    let mut count = 0usize;
                    for i in 0..gop.min(recon_coeffs.len()) {
                        for p in 0..3 {
                            let a = &originals[i][p];
                            let b = &recon_coeffs[i][p];
                            let n = a.len().min(b.len());
                            let mut sum = 0.0f64;
                            for j in 0..n {
                                sum += (a[j] - b[j]).abs() as f64;
                            }
                            mean_abs[p] += sum / n as f64;
                        }
                        count += 1;
                    }
                    if count > 0 {
                        println!(
                            "  Temporal coeff roundtrip (first GOP): mean_abs_diff Y {:.4}, Co {:.4}, Cg {:.4}",
                            mean_abs[0] / count as f64,
                            mean_abs[1] / count as f64,
                            mean_abs[2] / count as f64
                        );
                    }
                }

                println!(
                    "\n=== Temporal wavelet ({:?}) ===",
                    temporal_mode
                );
                let decoded_tw = decoder.decode_temporal_sequence(&ctx, &encoded_tw);

                // Bytes per output frame: distribute group bytes evenly across frames
                let mut per_frame_bytes: Vec<f64> = vec![0.0; frames_data.len()];
                let mut idx = 0usize;
                let group_size = encoded_tw.gop_size.max(1);
                let mut total_bytes_tw: usize = 0;
                let mut total_low_bytes: usize = 0;
                let mut total_high_bytes: usize = 0;
                let mut high_frame_sizes: Vec<usize> = Vec::new();
                let mut low_frame_sizes: Vec<usize> = Vec::new();
                let mut low_frame_ptrs: Vec<&CompressedFrame> = Vec::new();
                let mut high_level_ptrs: Vec<Vec<&CompressedFrame>> = Vec::new();
                for group in &encoded_tw.groups {
                    let low_bytes = group.low_frame.byte_size();
                    while high_level_ptrs.len() < group.high_frames.len() {
                        high_level_ptrs.push(Vec::new());
                    }
                    let mut high_bytes: usize = 0;
                    for (level_idx, lvl) in group.high_frames.iter().enumerate() {
                        for f in lvl {
                            high_frame_sizes.push(f.byte_size());
                            high_level_ptrs[level_idx].push(f);
                            high_bytes += f.byte_size();
                        }
                    }
                    let group_bytes = low_bytes + high_bytes;
                    total_bytes_tw += group_bytes;
                    total_low_bytes += low_bytes;
                    total_high_bytes += high_bytes;
                    low_frame_sizes.push(low_bytes);
                    low_frame_ptrs.push(&group.low_frame);
                    let per = group_bytes as f64 / group_size as f64;
                    for _ in 0..group_size {
                        if idx < per_frame_bytes.len() {
                            per_frame_bytes[idx] = per;
                            idx += 1;
                        }
                    }
                }
                for cf in &encoded_tw.tail_iframes {
                    if idx < per_frame_bytes.len() {
                        per_frame_bytes[idx] = cf.byte_size() as f64;
                        idx += 1;
                    }
                    total_bytes_tw += cf.byte_size();
                    total_low_bytes += cf.byte_size();
                    low_frame_sizes.push(cf.byte_size());
                    low_frame_ptrs.push(cf);
                }

                let mut frame_metrics_tw: Vec<FrameMetrics> = Vec::new();
                for i in 0..frames_data.len() {
                    let psnr = quality::psnr(&frames_data[i], &decoded_tw[i], 255.0);
                    let ssim = quality::ssim_approx(&frames_data[i], &decoded_tw[i], 255.0);
                    let bytes = per_frame_bytes[i];
                    let bpp = (bytes * 8.0) / (w as f64 * h as f64);
                    let frame_type = if i >= frames_data.len() - encoded_tw.tail_iframes.len() {
                        "I"
                    } else {
                        "T"
                    };
                    println!(
                        "  Frame {:2} [{}]: {:6.0} bytes, {:.2} bpp, PSNR {:.2} dB, SSIM {:.4}",
                        i,
                        frame_type,
                        bytes,
                        bpp,
                        psnr,
                        ssim,
                    );
                    frame_metrics_tw.push(FrameMetrics {
                        frame_idx: i,
                        frame_type: frame_type.to_string(),
                        psnr,
                        ssim,
                        bpp,
                        encoded_bytes: bytes.round() as usize,
                    });
                }

                let avg_bpp_tw =
                    per_frame_bytes.iter().sum::<f64>() * 8.0 / (w as f64 * h as f64) / frames_data.len() as f64;
                println!(
                    "  Total: {} bytes, avg {:.2} bpp, {:.1}ms ({:.1} fps)",
                    total_bytes_tw,
                    avg_bpp_tw,
                    elapsed_tw.as_secs_f64() * 1000.0,
                    frames_data.len() as f64 / elapsed_tw.as_secs_f64(),
                );
                if total_bytes_tw > 0 {
                    let low_pct = (total_low_bytes as f64 / total_bytes_tw as f64) * 100.0;
                    let high_pct = (total_high_bytes as f64 / total_bytes_tw as f64) * 100.0;
                    println!(
                        "  Temporal budget: lowpass {} bytes ({:.1}%), highpass {} bytes ({:.1}%)",
                        total_low_bytes, low_pct, total_high_bytes, high_pct
                    );
                    if !low_frame_sizes.is_empty() {
                        let low_avg =
                            low_frame_sizes.iter().sum::<usize>() as f64 / low_frame_sizes.len() as f64;
                        println!(
                            "  Temporal lowpass frames: {} (avg {:.0} bytes)",
                            low_frame_sizes.len(),
                            low_avg
                        );
                    }
                    if !high_frame_sizes.is_empty() {
                        let high_avg =
                            high_frame_sizes.iter().sum::<usize>() as f64 / high_frame_sizes.len() as f64;
                        println!(
                            "  Temporal highpass frames: {} (avg {:.0} bytes)",
                            high_frame_sizes.len(),
                            high_avg
                        );
                    }
                }

                // --- Temporal stats: mean_abs + percent_zero after quantize per level ---
                if !low_frame_ptrs.is_empty() && !high_level_ptrs.is_empty() {
                    let accumulate_stats = |frames: &[&CompressedFrame]| -> (f64, f64) {
                        let mut sum_abs: i64 = 0;
                        let mut zeros: i64 = 0;
                        let mut total: i64 = 0;
                        for cf in frames {
                            if let EntropyData::Rice(ref tiles) = cf.entropy {
                                for t in tiles {
                                    let coeffs = gnc::encoder::rice::rice_decode_tile(t);
                                    for v in coeffs {
                                        if v == 0 {
                                            zeros += 1;
                                        } else {
                                            sum_abs += v.abs() as i64;
                                        }
                                        total += 1;
                                    }
                                }
                            }
                        }
                        if total == 0 {
                            return (0.0, 0.0);
                        }
                        let mean_abs = sum_abs as f64 / total as f64;
                        let pct_zero = (zeros as f64 / total as f64) * 100.0;
                        (mean_abs, pct_zero)
                    };

                    let (low_mean, low_zero) = accumulate_stats(&low_frame_ptrs);
                    println!(
                        "  Temporal lowpass quantized: mean_abs {:.3}, zero {:.1}%",
                        low_mean, low_zero
                    );

                    for (lvl, frames_lvl) in high_level_ptrs.iter().enumerate() {
                        let (hi_mean, hi_zero) = accumulate_stats(frames_lvl);
                        println!(
                            "  Temporal highpass L{} quantized: mean_abs {:.3}, zero {:.1}%",
                            lvl, hi_mean, hi_zero
                        );
                    }
                }

                // --- Per-GOP diagnostics (to stderr, gated on --diagnostics) ---
                if gnc::encoder::diagnostics::enabled() {
                    let q = quality.unwrap_or(75);
                    let mul = tw_highpass_mul.unwrap_or(config_tw.temporal_highpass_qstep_mul);
                    // Compute all-I baseline bpp from first I-frame if available
                    let all_i_bpp = if !low_frame_ptrs.is_empty() {
                        // Use the lowpass frame bpp as a rough all-I reference
                        // (the lowpass is essentially the I-frame quality signal)
                        Some(low_frame_ptrs[0].bpp())
                    } else {
                        None
                    };

                    let mut frame_offset = 0usize;
                    for (gi, group) in encoded_tw.groups.iter().enumerate() {
                        let gs = encoded_tw.gop_size;
                        let end = (frame_offset + gs).min(frame_metrics_tw.len());
                        let per_frame_q: Vec<(f64, f64)> = frame_metrics_tw[frame_offset..end]
                            .iter()
                            .map(|m| (m.psnr, m.ssim))
                            .collect();
                        gnc::encoder::diagnostics::print_temporal_gop_diagnostics(
                            gi,
                            gs,
                            temporal_mode,
                            q,
                            mul,
                            group,
                            &per_frame_q,
                            all_i_bpp,
                            w,
                            h,
                        );
                        frame_offset += gs;
                    }
                }

                let summary_tw = sequence_metrics::compute_sequence_metrics(&frame_metrics_tw);
                println!("\n{}", summary_tw);

                if run_baseline {
                    println!(
                        "\n=== A/B Comparison ===\n  Baseline I+P: {} bytes ({:.2} bpp)\n  Temporal:     {} bytes ({:.2} bpp)",
                        total_bytes_ip,
                        avg_bpp_ip,
                        total_bytes_tw,
                        avg_bpp_tw,
                    );
                }

                if let Some(ref csv_path) = csv {
                    let out_path = if run_baseline {
                        csv_with_suffix(csv_path, "tw")
                    } else {
                        csv_path.clone()
                    };
                    sequence_metrics::write_sequence_csv(&out_path, &frame_metrics_tw, &summary_tw)
                        .expect("Failed to write sequence CSV");
                    println!("\nSequence metrics written to {}", out_path);
                }

                // Write GNV2 if output path provided
                if let Some(ref output_path) = output {
                    let fps_num = fps.round() as u32;
                    let fps_den = 1u32;
                    let gnv2_data = serialize_temporal_sequence(&encoded_tw, (fps_num, fps_den));
                    std::fs::write(output_path, &gnv2_data).expect("Failed to write GNV2 output");
                    println!(
                        "Wrote GNV2: {} bytes ({:.2} MB)",
                        gnv2_data.len(),
                        gnv2_data.len() as f64 / 1_048_576.0
                    );
                }
            }
        }

        Command::EncodeSequence {
            input,
            output,
            quality,
            qstep,
            keyframe_interval,
            num_frames,
            fps_num,
            fps_den,
            bitrate,
            rate_mode,
            rans,
            rice,
            diagnostics,
            temporal_wavelet,
        } => {
            // encode-sequence is the I+P (motion vector) production encoder → GNV1.
            // For temporal wavelet encoding → GNV2, use benchmark-sequence instead
            // (it has streaming per-GOP encoding that doesn't accumulate all groups in memory).
            if temporal_wavelet.to_lowercase() != "none" {
                eprintln!(
                    "encode-sequence only supports I+P (motion vector) mode (--temporal-wavelet none)."
                );
                eprintln!(
                    "For temporal wavelet encoding, use: benchmark-sequence -o output.gnv2"
                );
                std::process::exit(1);
            }

            if diagnostics {
                gnc::encoder::diagnostics::enable();
            }
            // Auto-detect frame count if not specified
            let frame_count = if let Some(n) = num_frames {
                n
            } else {
                let mut n = 0;
                loop {
                    let path = input.replace("%04d", &format!("{:04}", n));
                    if !std::path::Path::new(&path).exists() {
                        break;
                    }
                    n += 1;
                }
                if n == 0 {
                    eprintln!(
                        "Error: no frames found matching pattern '{}' (tried index 0)",
                        input
                    );
                    std::process::exit(1);
                }
                n
            };

            let actual_fps = fps_num as f64 / fps_den as f64;

            println!(
                "Encoding sequence: {} frames from '{}', ki={}, q={}, {}fps, mode=I+P (motion vectors)",
                frame_count,
                input,
                keyframe_interval,
                quality,
                if fps_den == 1 {
                    format!("{}", fps_num)
                } else {
                    format!("{:.3}", actual_fps)
                },
            );

            // Probe first frame for dimensions
            let first_path = input.replace("%04d", &format!("{:04}", 0));
            let (_, w, h) = load_image_rgb_f32(&first_path);

            let ctx = GpuContext::new();
            let mut encoder = EncoderPipeline::new(&ctx);

            let mut config = gnc::quality_preset(quality);
            if let Some(qs) = qstep {
                config.quantization_step = qs;
            }
            config.keyframe_interval = keyframe_interval;
            if rans {
                config.entropy_coder = gnc::EntropyCoder::Rans;
            }
            if rice {
                config.entropy_coder = gnc::EntropyCoder::Rice;
            }

            // Apply rate control settings
            if let Some(ref br) = bitrate {
                config.target_bitrate = Some(parse_bitrate(br));
                config.rate_mode = parse_rate_mode(&rate_mode);
            }

            let input_pattern = input.clone();
            let start = std::time::Instant::now();
            let compressed = encoder.encode_sequence_streaming(
                &ctx,
                frame_count,
                |i| {
                    let path = input_pattern.replace("%04d", &format!("{:04}", i));
                    let (rgb, fw, fh) = load_image_rgb_f32(&path);
                    assert!(
                        fw == w && fh == h,
                        "Frame {} has different dimensions ({}x{} vs {}x{})",
                        i,
                        fw,
                        fh,
                        w,
                        h
                    );
                    rgb
                },
                w,
                h,
                &config,
                actual_fps,
            );
            let encode_time = start.elapsed();

            for (i, cf) in compressed.iter().enumerate() {
                let ft = if cf.frame_type == gnc::FrameType::Intra {
                    "I"
                } else {
                    "P"
                };
                println!(
                    "  Frame {:4} [{}]: {:6} bytes, {:.2} bpp",
                    i,
                    ft,
                    cf.byte_size(),
                    cf.bpp()
                );
            }

            let gnv_data = serialize_sequence(&compressed, (fps_num, fps_den));
            std::fs::write(&output, &gnv_data).expect("Failed to write GNV1 output");

            let avg_bpp =
                compressed.iter().map(|f| f.bpp()).sum::<f64>() / compressed.len() as f64;
            let i_count = compressed
                .iter()
                .filter(|f| f.frame_type == gnc::FrameType::Intra)
                .count();

            println!(
                "\nEncoded {} frames ({}I + {}P) in {:.1}ms ({:.1} fps)",
                compressed.len(),
                i_count,
                compressed.len() - i_count,
                encode_time.as_secs_f64() * 1000.0,
                compressed.len() as f64 / encode_time.as_secs_f64(),
            );
            println!(
                "Container: {} bytes, avg {:.2} bpp → {}",
                gnv_data.len(),
                avg_bpp,
                output,
            );
        }

        Command::DecodeSequence {
            input,
            output,
            seek,
        } => {
            let gnv_data = std::fs::read(&input).expect("Failed to read input file");

            // Check magic bytes to determine container format
            assert!(gnv_data.len() >= 4, "File too small to be a GNV container");
            let magic = &gnv_data[..4];

            if magic == b"GNV2" {
                // GNV2 temporal wavelet decode
                let seq = deserialize_temporal_sequence(&gnv_data);
                println!(
                    "GNV2 temporal sequence: {}x{}, {} frames, {} GOPs (size {}), {:?}",
                    seq.groups[0].low_frame.info.width,
                    seq.groups[0].low_frame.info.height,
                    seq.frame_count,
                    seq.groups.len(),
                    seq.gop_size,
                    seq.mode,
                );

                let ctx = GpuContext::new();
                let decoder = DecoderPipeline::new(&ctx);
                let start = std::time::Instant::now();
                let decoded_frames = decoder.decode_temporal_sequence(&ctx, &seq);
                let elapsed = start.elapsed();

                println!(
                    "Decoded {} frames in {:.1}ms ({:.1} fps)",
                    decoded_frames.len(),
                    elapsed.as_secs_f64() * 1000.0,
                    decoded_frames.len() as f64 / elapsed.as_secs_f64(),
                );

                // Write output frames
                let width = seq.groups[0].low_frame.info.width;
                let height = seq.groups[0].low_frame.info.height;
                for (i, rgb_f32) in decoded_frames.iter().enumerate() {
                    let out_path = format!("{}_{:04}.png", output, i);
                    save_image_rgb_f32(&out_path, rgb_f32, width, height);
                    println!("  Wrote frame {} → {}", i, out_path);
                }
            } else if magic == b"GNV1" {
                // GNV1 I/P/B frame decode (existing path)
                let header = deserialize_sequence_header(&gnv_data);

                println!(
                    "GNV1 sequence: {}x{}, {} frames, {:.3} fps, {:.2}s duration",
                    header.width,
                    header.height,
                    header.frame_count,
                    header.fps(),
                    header.duration_secs(),
                );

                let ctx = GpuContext::new();
                let decoder = DecoderPipeline::new(&ctx);

                // Determine range of frames to decode
                let (start_frame, end_frame) = if let Some(seek_time) = seek {
                    // Convert seek time to PTS (frame number)
                    let target_pts = if header.framerate_den == 0 {
                        0u64
                    } else {
                        (seek_time * header.framerate_num as f64 / header.framerate_den as f64)
                            as u64
                    };
                    let keyframe_idx = seek_to_keyframe(&header, target_pts);
                    let target_frame = header
                        .index
                        .iter()
                        .rposition(|e| e.pts <= target_pts)
                        .unwrap_or(0);

                    println!(
                        "Seeking to {:.2}s (PTS {}): keyframe at frame {}, target frame {}",
                        seek_time, target_pts, keyframe_idx, target_frame
                    );

                    // Decode from keyframe through the target frame so P-frame
                    // references are valid, but only output the target frame.
                    (keyframe_idx, target_frame + 1)
                } else {
                    (0, header.frame_count as usize)
                };

                let start = std::time::Instant::now();

                // Build decode order from frame types in the GNV1 index.
                // B-frames must be decoded after their anchor P-frame (which comes
                // later in display order but must be decoded first to provide the
                // backward reference).
                let frame_types: Vec<FrameType> = (start_frame..end_frame)
                    .map(|i| match header.index[i].frame_type {
                        0 => FrameType::Intra,
                        1 => FrameType::Predicted,
                        2 => FrameType::Bidirectional,
                        t => panic!("Unknown frame type {} at index {}", t, i),
                    })
                    .collect();

                let decode_order = {
                    let mut order = Vec::with_capacity(frame_types.len());
                    let mut i = 0;
                    while i < frame_types.len() {
                        if frame_types[i] != FrameType::Bidirectional {
                            order.push(i);
                            i += 1;
                        } else {
                            // Collect consecutive B-frames
                            let b_start = i;
                            while i < frame_types.len()
                                && frame_types[i] == FrameType::Bidirectional
                            {
                                i += 1;
                            }
                            let b_end = i;
                            // Anchor (I/P) after B-frames must be decoded first
                            if i < frame_types.len() {
                                order.push(i);
                                i += 1;
                            }
                            // Then the B-frames
                            for b in b_start..b_end {
                                order.push(b);
                            }
                        }
                    }
                    order
                };

                let mut results: Vec<Option<Vec<f32>>> =
                    (0..frame_types.len()).map(|_| None).collect();
                let mut output_count = 0usize;

                let mut di = 0;
                while di < decode_order.len() {
                    let local_idx = decode_order[di];
                    let abs_idx = start_frame + local_idx;
                    let compressed = deserialize_sequence_frame(&gnv_data, &header, abs_idx);

                    // Check if B-frames follow this anchor in decode order
                    let b_frames_follow = di + 1 < decode_order.len()
                        && frame_types[decode_order[di + 1]] == FrameType::Bidirectional;

                    if frame_types[local_idx] == FrameType::Bidirectional {
                        // B-frame: references already set up by preceding anchor
                        results[local_idx] = Some(decoder.decode(&ctx, &compressed));
                        di += 1;
                    } else if b_frames_follow {
                        // Anchor before B-frames: save current ref as backward,
                        // decode the anchor (updates forward ref), then swap
                        decoder.swap_forward_to_backward_ref(&ctx);
                        results[local_idx] = Some(decoder.decode(&ctx, &compressed));
                        decoder.swap_references(); // ref=past, bwd=future
                        di += 1;

                        // Decode all following B-frames
                        while di < decode_order.len()
                            && frame_types[decode_order[di]] == FrameType::Bidirectional
                        {
                            let b_local = decode_order[di];
                            let b_abs = start_frame + b_local;
                            let b_compressed =
                                deserialize_sequence_frame(&gnv_data, &header, b_abs);
                            results[b_local] = Some(decoder.decode(&ctx, &b_compressed));
                            di += 1;
                        }

                        // Swap back: ref=future anchor (for next group's forward ref)
                        decoder.swap_references();
                    } else {
                        // Regular I/P frame with no B-frames following
                        results[local_idx] = Some(decoder.decode(&ctx, &compressed));
                        di += 1;
                    }
                }

                // Output frames in display order
                #[allow(clippy::needless_range_loop)] // local_idx used for both results[] and abs_idx
                for local_idx in 0..frame_types.len() {
                    let abs_idx = start_frame + local_idx;
                    let should_output = if seek.is_some() {
                        abs_idx == end_frame - 1
                    } else {
                        true
                    };

                    if should_output {
                        if let Some(ref rgb_data) = results[local_idx] {
                            let frame_path =
                                output.replace("%04d", &format!("{:04}", abs_idx));
                            save_image_rgb_f32(
                                &frame_path,
                                rgb_data,
                                header.width,
                                header.height,
                            );
                            output_count += 1;
                        }
                    }
                }

                let decode_time = start.elapsed();
                let decoded_count = end_frame - start_frame;
                let fps = decoded_count as f64 / decode_time.as_secs_f64();

                println!(
                    "Decoded {} frames ({} output) in {:.1}ms ({:.1} fps)",
                    decoded_count,
                    output_count,
                    decode_time.as_secs_f64() * 1000.0,
                    fps,
                );
            } else {
                panic!(
                    "Unknown container magic: {:?} (expected GNV1 or GNV2)",
                    std::str::from_utf8(magic).unwrap_or("<invalid utf8>")
                );
            }
        }

        Command::RdCurve {
            input,
            output,
            compare,
            q_values,
            compare_codecs,
        } => {
            // Parse quality values
            let q_vals: Vec<u32> = q_values
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<u32>()
                        .unwrap_or_else(|_| panic!("Invalid quality value: '{}'", s.trim()))
                })
                .collect();

            // Collect GNC RD points and SSIM values for codec comparison
            let mut gnc_rd_points: Vec<(u32, bdrate::RdPoint)> = Vec::new();
            let mut gnc_ssim_values: Vec<f64> = Vec::new();

            // Encode sweep mode: generate RD curve from input image
            if let Some(ref input_path) = input {
                let (rgb_data, w, h) = load_image_rgb_f32(input_path);
                println!(
                    "RD curve sweep: {}x{} image, {} quality points",
                    w,
                    h,
                    q_vals.len()
                );

                let ctx = GpuContext::new();
                let mut encoder = EncoderPipeline::new(&ctx);
                let decoder = DecoderPipeline::new(&ctx);

                // CSV writer
                let mut wtr = csv::Writer::from_path(&output).expect("Failed to create CSV file");
                wtr.write_record([
                    "q",
                    "qstep",
                    "psnr",
                    "ssim",
                    "bpp",
                    "encode_ms",
                    "decode_ms",
                ])
                .expect("Failed to write CSV header");

                // Print table header
                println!(
                    "\n{:>5} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
                    "q", "qstep", "psnr", "ssim", "bpp", "enc_ms", "dec_ms"
                );
                println!("{}", "-".repeat(68));

                for &q in &q_vals {
                    let config = gnc::quality_preset(q);
                    let qstep = config.quantization_step;

                    // Time encode
                    let enc_start = std::time::Instant::now();
                    let compressed = encoder.encode(&ctx, &rgb_data, w, h, &config);
                    let encode_ms = enc_start.elapsed().as_secs_f64() * 1000.0;

                    // Time decode
                    let dec_start = std::time::Instant::now();
                    let reconstructed = decoder.decode(&ctx, &compressed);
                    let decode_ms = dec_start.elapsed().as_secs_f64() * 1000.0;

                    let psnr = quality::psnr(&rgb_data, &reconstructed, 255.0);
                    let ssim = quality::ssim_approx(&rgb_data, &reconstructed, 255.0);
                    let bpp = compressed.bpp();

                    // Collect for codec comparison
                    gnc_rd_points.push((q, bdrate::RdPoint { bpp, psnr }));
                    gnc_ssim_values.push(ssim);

                    // Write CSV row
                    wtr.write_record(&[
                        format!("{}", q),
                        format!("{:.4}", qstep),
                        format!("{:.4}", psnr),
                        format!("{:.6}", ssim),
                        format!("{:.6}", bpp),
                        format!("{:.2}", encode_ms),
                        format!("{:.2}", decode_ms),
                    ])
                    .expect("Failed to write CSV row");

                    // Print table row
                    println!(
                        "{:>5} {:>8.4} {:>8.2} {:>8.4} {:>8.4} {:>10.2} {:>10.2}",
                        q, qstep, psnr, ssim, bpp, encode_ms, decode_ms
                    );
                }

                wtr.flush().expect("Failed to flush CSV");
                println!("\nRD curve written to {}", output);

                // Multi-codec comparison mode
                if compare_codecs {
                    println!("\n=== Multi-Codec Comparison ===");

                    // JPEG sweep
                    let jpeg_q_values = codec_compare::default_jpeg_quality_values();
                    println!(
                        "\nSweeping JPEG ({} quality points)...",
                        jpeg_q_values.len()
                    );
                    let jpeg_points = codec_compare::jpeg_rd_curve(input_path, &jpeg_q_values);
                    let jpeg_ssim = codec_compare::jpeg_ssim_values(input_path, &jpeg_q_values);

                    // JPEG 2000 sweep
                    let j2k_rates = codec_compare::default_j2k_rates();
                    println!("Sweeping JPEG 2000 ({} rate points)...", j2k_rates.len());
                    let j2k_points = codec_compare::jpeg2000_rd_curve(input_path, &j2k_rates)
                        .unwrap_or_else(|| {
                            println!("  opj_compress not found in PATH, skipping JPEG 2000");
                            Vec::new()
                        });

                    // Print summary table
                    codec_compare::print_comparison_summary(
                        &gnc_rd_points,
                        &jpeg_points,
                        &j2k_points,
                    );

                    // Write unified comparison CSV
                    let comparison_csv = output.replace(".csv", "_comparison.csv");
                    codec_compare::write_comparison_csv(
                        &comparison_csv,
                        &gnc_rd_points,
                        &jpeg_points,
                        &j2k_points,
                        &gnc_ssim_values,
                        &jpeg_ssim,
                    )
                    .expect("Failed to write comparison CSV");
                    println!("\nComparison CSV written to {}", comparison_csv);

                    // BD-rate: GNC vs JPEG
                    let gnc_rd: Vec<bdrate::RdPoint> =
                        codec_compare::extract_rd_points_u32(&gnc_rd_points);
                    let jpeg_rd: Vec<bdrate::RdPoint> =
                        codec_compare::extract_rd_points_u32(&jpeg_points);

                    println!("\n--- BD-rate: GNC vs JPEG ---");
                    match bdrate::bd_rate(&jpeg_rd, &gnc_rd) {
                        Some(rate) => {
                            let better_worse = if rate < 0.0 { "better" } else { "worse" };
                            println!(
                                "  BD-rate:  {:.2}% (GNC uses {:.1}% {} bits than JPEG at same quality)",
                                rate,
                                rate.abs(),
                                better_worse
                            );
                        }
                        None => println!("  BD-rate:  N/A (need >= 4 overlapping PSNR points)"),
                    }
                    match bdrate::bd_psnr(&jpeg_rd, &gnc_rd) {
                        Some(psnr) => {
                            let better_worse = if psnr > 0.0 { "better" } else { "worse" };
                            println!(
                                "  BD-PSNR:  {:.2} dB (GNC is {:.2} dB {} than JPEG at same bitrate)",
                                psnr,
                                psnr.abs(),
                                better_worse
                            );
                        }
                        None => println!("  BD-PSNR:  N/A (need >= 4 overlapping points)"),
                    }

                    // BD-rate: GNC vs JPEG 2000
                    if !j2k_points.is_empty() {
                        let j2k_rd: Vec<bdrate::RdPoint> =
                            codec_compare::extract_rd_points_f32(&j2k_points);

                        println!("\n--- BD-rate: GNC vs JPEG 2000 ---");
                        match bdrate::bd_rate(&j2k_rd, &gnc_rd) {
                            Some(rate) => {
                                let better_worse = if rate < 0.0 { "better" } else { "worse" };
                                println!(
                                    "  BD-rate:  {:.2}% (GNC uses {:.1}% {} bits than JPEG 2000 at same quality)",
                                    rate,
                                    rate.abs(),
                                    better_worse
                                );
                            }
                            None => println!("  BD-rate:  N/A (need >= 4 overlapping PSNR points)"),
                        }
                        match bdrate::bd_psnr(&j2k_rd, &gnc_rd) {
                            Some(psnr) => {
                                let better_worse = if psnr > 0.0 { "better" } else { "worse" };
                                println!(
                                    "  BD-PSNR:  {:.2} dB (GNC is {:.2} dB {} than JPEG 2000 at same bitrate)",
                                    psnr,
                                    psnr.abs(),
                                    better_worse
                                );
                            }
                            None => println!("  BD-PSNR:  N/A (need >= 4 overlapping points)"),
                        }
                    }
                }
            }

            // BD-rate comparison mode
            if let Some(ref compare_path) = compare {
                // Determine reference curve path: either the just-generated output or
                // the output path when no input was given
                let ref_path = &output;

                // If no input was provided, ref_path must be an existing CSV
                if input.is_none() && !std::path::Path::new(ref_path).exists() {
                    eprintln!(
                        "Error: no --input given and reference CSV '{}' does not exist.",
                        ref_path
                    );
                    std::process::exit(1);
                }

                println!("\nBD-rate comparison: '{}' vs '{}'", ref_path, compare_path);

                let reference = bdrate::load_rd_curve(ref_path).unwrap_or_else(|e| {
                    eprintln!("Failed to load reference CSV '{}': {}", ref_path, e);
                    std::process::exit(1);
                });
                let test = bdrate::load_rd_curve(compare_path).unwrap_or_else(|e| {
                    eprintln!("Failed to load test CSV '{}': {}", compare_path, e);
                    std::process::exit(1);
                });

                println!(
                    "  Reference: {} points, Test: {} points",
                    reference.len(),
                    test.len()
                );

                match bdrate::bd_rate(&reference, &test) {
                    Some(rate) => {
                        let better_worse = if rate < 0.0 { "better" } else { "worse" };
                        println!(
                            "  BD-rate:  {:.2}% ({} uses {:.1}% {} bits at same quality)",
                            rate,
                            compare_path,
                            rate.abs(),
                            better_worse
                        );
                    }
                    None => println!("  BD-rate:  N/A (need >= 4 overlapping points)"),
                }

                match bdrate::bd_psnr(&reference, &test) {
                    Some(psnr) => {
                        let better_worse = if psnr > 0.0 { "better" } else { "worse" };
                        println!(
                            "  BD-PSNR:  {:.2} dB ({} is {:.2} dB {} at same bitrate)",
                            psnr,
                            compare_path,
                            psnr.abs(),
                            better_worse
                        );
                    }
                    None => println!("  BD-PSNR:  N/A (need >= 4 overlapping points)"),
                }
            }

            // Neither input nor compare given
            if input.is_none() && compare.is_none() {
                eprintln!("Error: provide --input for RD sweep, --compare for BD-rate, or both.");
                std::process::exit(1);
            }
        }

        Command::BenchmarkSuite {
            dir,
            sequences,
            max_frames,
            quality,
            csv,
            temporal_wavelet,
            tw_highpass_mul,
        } => {
            let ctx = GpuContext::new();
            let mut encoder = gnc::encoder::pipeline::EncoderPipeline::new(&ctx);
            let decoder = gnc::decoder::pipeline::DecoderPipeline::new(&ctx);

            let temporal_mode = match temporal_wavelet.as_str() {
                "haar" => TemporalTransform::Haar,
                "53" => TemporalTransform::LeGall53,
                "none" => TemporalTransform::None,
                _ => TemporalTransform::Haar, // auto defaults to Haar
            };

            let seq_names: Vec<&str> = sequences.split(',').map(|s| s.trim()).collect();

            // CSV header
            let mut csv_rows: Vec<String> = vec![
                "sequence,frames,quality,temporal_mode,bpp,psnr_avg,psnr_min,psnr_lowpass_avg,psnr_lowpass_min,fps_enc,fps_dec,total_bytes".to_string(),
            ];

            println!("=== GNC Benchmark Suite ===");
            println!(
                "Sequences: {:?}, quality={}, temporal={:?}",
                seq_names, quality, temporal_mode
            );
            println!();

            for seq_name in &seq_names {
                let seq_dir = format!("{}/{}", dir, seq_name);
                // Count available frames
                let mut frame_count = 0usize;
                loop {
                    let path = format!("{}/frame_{:04}.png", seq_dir, frame_count);
                    if !std::path::Path::new(&path).exists() {
                        break;
                    }
                    frame_count += 1;
                }
                if frame_count == 0 {
                    eprintln!("  {} — no frames found in {}, skipping", seq_name, seq_dir);
                    continue;
                }
                let num_frames = if max_frames > 0 && max_frames < frame_count {
                    max_frames
                } else {
                    frame_count
                };

                let gop_size = match temporal_mode {
                    TemporalTransform::Haar => 4,
                    TemporalTransform::LeGall53 => 4,
                    TemporalTransform::None => 1,
                };

                // Load first frame for dimensions + warmup
                let first_path = format!("{}/frame_0000.png", seq_dir);
                let (first_rgb, w, h) = load_image_rgb_f32(&first_path);
                let mut config_tw = gnc::quality_preset(quality);
                config_tw.temporal_transform = temporal_mode;
                config_tw.target_bitrate = None;
                if let Some(mul) = tw_highpass_mul {
                    config_tw.temporal_highpass_qstep_mul = mul;
                    config_tw.adaptive_temporal_mul = false;
                }
                let _ = encoder.encode(&ctx, &first_rgb, w, h, &config_tw);
                drop(first_rgb);

                let mut total_bytes = 0usize;
                let mut psnr_vals: Vec<f64> = Vec::new();
                let mut psnr_lowpass_vals: Vec<f64> = Vec::new();
                let mut enc_time = std::time::Duration::ZERO;
                let mut dec_time = std::time::Duration::ZERO;

                if temporal_mode == TemporalTransform::None {
                    // All-I mode: encode each frame independently
                    for i in 0..num_frames {
                        let path = format!("{}/frame_{:04}.png", seq_dir, i);
                        let (rgb, _, _) = load_image_rgb_f32(&path);
                        let t0 = std::time::Instant::now();
                        let cf = encoder.encode(&ctx, &rgb, w, h, &config_tw);
                        enc_time += t0.elapsed();
                        total_bytes += cf.byte_size();
                        let t1 = std::time::Instant::now();
                        let dec = decoder.decode(&ctx, &cf);
                        dec_time += t1.elapsed();
                        let psnr = quality::psnr(&rgb, &dec, 255.0);
                        psnr_vals.push(psnr);
                    }
                } else {
                    // Temporal wavelet mode: encode GOPs, then decode
                    let num_gops = num_frames / gop_size;
                    let tail_start = num_gops * gop_size;
                    let mut groups: Vec<gnc::TemporalGroup> = Vec::new();

                    // Encode pass
                    for gop_idx in 0..num_gops {
                        let base = gop_idx * gop_size;
                        let mut gop_frames: Vec<Vec<f32>> = Vec::with_capacity(gop_size);
                        for j in 0..gop_size {
                            let path = format!("{}/frame_{:04}.png", seq_dir, base + j);
                            let (rgb, _, _) = load_image_rgb_f32(&path);
                            gop_frames.push(rgb);
                        }
                        let gop_refs: Vec<&[f32]> =
                            gop_frames.iter().map(|f| f.as_slice()).collect();
                        let t0 = std::time::Instant::now();
                        let group = encoder.encode_temporal_wavelet_gop(
                            &ctx, &gop_refs, w, h, &config_tw, temporal_mode, None,
                        );
                        enc_time += t0.elapsed();
                        let group_bytes: usize = group.low_frame.byte_size()
                            + group
                                .high_frames
                                .iter()
                                .flat_map(|lvl| lvl.iter())
                                .map(|f| f.byte_size())
                                .sum::<usize>();
                        total_bytes += group_bytes;
                        groups.push(group);
                    }

                    // Tail I-frames encode
                    let mut tail_iframes: Vec<CompressedFrame> = Vec::new();
                    for i in tail_start..num_frames {
                        let path = format!("{}/frame_{:04}.png", seq_dir, i);
                        let (rgb, _, _) = load_image_rgb_f32(&path);
                        let t0 = std::time::Instant::now();
                        let cf = encoder.encode(&ctx, &rgb, w, h, &config_tw);
                        enc_time += t0.elapsed();
                        total_bytes += cf.byte_size();
                        tail_iframes.push(cf);
                    }

                    // Decode pass + PSNR (full and lowpass-only)
                    for (gop_idx, group) in groups.iter().enumerate() {
                        let t0 = std::time::Instant::now();
                        let decoded = decoder.decode_temporal_sequence(
                            &ctx,
                            &gnc::TemporalEncodedSequence {
                                mode: temporal_mode,
                                groups: vec![group.clone()],
                                group_gop_indices: vec![0],
                                tail_iframes: vec![],
                                tail_iframe_pts: vec![],
                                frame_count: gop_size,
                                gop_size,
                            },
                        );
                        dec_time += t0.elapsed();

                        // Lowpass-only: decode just the lowpass frame as I-frame
                        let lowpass_rgb = decoder.decode(&ctx, &group.low_frame);

                        let base = gop_idx * gop_size;
                        for (j, dec_frame) in decoded.iter().enumerate() {
                            let path = format!("{}/frame_{:04}.png", seq_dir, base + j);
                            let (orig_rgb, _, _) = load_image_rgb_f32(&path);
                            let psnr = quality::psnr(&orig_rgb, dec_frame, 255.0);
                            psnr_vals.push(psnr);
                            let psnr_lp = quality::psnr(&orig_rgb, &lowpass_rgb, 255.0);
                            psnr_lowpass_vals.push(psnr_lp);
                        }
                    }

                    // Decode tail I-frames
                    for (i, cf) in tail_iframes.iter().enumerate() {
                        let t0 = std::time::Instant::now();
                        let dec = decoder.decode(&ctx, cf);
                        dec_time += t0.elapsed();
                        let path = format!("{}/frame_{:04}.png", seq_dir, tail_start + i);
                        let (orig_rgb, _, _) = load_image_rgb_f32(&path);
                        let psnr = quality::psnr(&orig_rgb, &dec, 255.0);
                        psnr_vals.push(psnr);
                    }
                }

                let fps_enc = num_frames as f64 / enc_time.as_secs_f64();
                let fps_dec = num_frames as f64 / dec_time.as_secs_f64();
                let avg_bpp =
                    (total_bytes as f64 * 8.0) / (w as f64 * h as f64) / num_frames as f64;

                // Cap infinite PSNR (identical frames) at 99 dB for averaging
                let cap = |vals: &[f64]| -> (f64, f64) {
                    if vals.is_empty() {
                        return (0.0, 0.0);
                    }
                    let capped: Vec<f64> = vals.iter().map(|&v| v.min(99.0)).collect();
                    let avg = capped.iter().sum::<f64>() / capped.len() as f64;
                    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    (avg, if min.is_infinite() { 0.0 } else { min })
                };
                let (psnr_avg, psnr_min) = cap(&psnr_vals);
                let (psnr_lp_avg, psnr_lp_min) = cap(&psnr_lowpass_vals);

                if psnr_lowpass_vals.is_empty() {
                    println!(
                        "  {:20} {:4} frames  {:.2} bpp  {:.2}/{:.2} dB  {:.1}/{:.1} fps",
                        seq_name, num_frames, avg_bpp, psnr_avg, psnr_min, fps_enc, fps_dec,
                    );
                } else {
                    println!(
                        "  {:20} {:4} frames  {:.2} bpp  full {:.2}/{:.2} dB  lowpass {:.2}/{:.2} dB  {:.1}/{:.1} fps",
                        seq_name, num_frames, avg_bpp, psnr_avg, psnr_min,
                        psnr_lp_avg, psnr_lp_min, fps_enc, fps_dec,
                    );
                }

                csv_rows.push(format!(
                    "{},{},{},{:?},{:.4},{:.2},{:.2},{:.2},{:.2},{:.1},{:.1},{}",
                    seq_name, num_frames, quality, temporal_mode, avg_bpp, psnr_avg, psnr_min,
                    psnr_lp_avg, psnr_lp_min, fps_enc, fps_dec, total_bytes,
                ));
            }

            // Write CSV
            if let Some(parent) = std::path::Path::new(&csv).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            std::fs::write(&csv, csv_rows.join("\n") + "\n")
                .unwrap_or_else(|e| eprintln!("Failed to write CSV: {}", e));
            println!("\nCSV written to: {}", csv);
        }

        Command::TransformShootout { input, iterations } => {
            let ctx = GpuContext::new();

            // Run Haar diagnostic first
            experiments::transform_shootout::diagnose_haar(&ctx);

            let (rgb_data, w, h) = load_image_rgb_f32(&input);
            println!(
                "\nTransform shootout: {}x{} image, {} iterations",
                w, h, iterations
            );

            let results =
                experiments::transform_shootout::run_shootout(&ctx, &rgb_data, w, h, iterations);
            experiments::transform_shootout::print_results(&results);

            // Fused mega-kernel benchmark
            experiments::transform_shootout::run_fused_benchmark(&ctx, &rgb_data, w, h, iterations);
        }
    }
}
