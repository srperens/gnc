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

        /// Use bitplane entropy coder instead of Rice (default)
        #[arg(long)]
        bitplane: bool,

        /// Use rANS entropy coder instead of Rice (default)
        #[arg(long)]
        rans: bool,

        /// Use canonical Huffman entropy coder instead of Rice (default)
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

        /// Use bitplane entropy coder instead of Rice (default)
        #[arg(long)]
        bitplane: bool,

        /// Use rANS entropy coder instead of Rice (default)
        #[arg(long)]
        rans: bool,

        /// Use canonical Huffman entropy coder instead of Rice (default)
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
        /// Input frame pattern (e.g., "frames/frame_%04d.png")
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

        /// Use rANS entropy coder instead of Rice (default)
        #[arg(long)]
        rans: bool,

        /// Enable per-frame encode diagnostics (also via GNC_DIAGNOSTICS=1)
        #[arg(long)]
        diagnostics: bool,

        /// Temporal wavelet mode: none, haar, 53 (LeGall 5/3)
        #[arg(long, default_value = "none")]
        temporal_wavelet: String,

        /// Temporal highpass qstep multiplier (e.g. 2.0 = double qstep for highpass)
        #[arg(long)]
        tw_highpass_mul: Option<f32>,

        /// Run A/B comparison against baseline I+P (+ I-only)
        #[arg(long)]
        ab: bool,

        /// Output file path for GNV2 container (temporal wavelet mode only)
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

        /// Use rANS entropy coder instead of Rice (default)
        #[arg(long)]
        rans: bool,

        /// Enable per-frame encode diagnostics (also via GNC_DIAGNOSTICS=1)
        #[arg(long)]
        diagnostics: bool,
    },

    /// Decode a .gnv container back to image frames
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
        other => {
            eprintln!(
                "Unknown temporal wavelet '{}'. Use 'none', 'haar', or '53'. Defaulting to none.",
                other
            );
            TemporalTransform::None
        }
    }
}

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

            let temporal_mode = parse_temporal_transform(&temporal_wavelet);
            let run_temporal = temporal_mode != TemporalTransform::None;
            let run_baseline = !run_temporal || ab;
            if run_temporal && temporal_mode == TemporalTransform::Haar {
                if !gnc::temporal::is_power_of_two(keyframe_interval as usize) {
                    eprintln!(
                        "Temporal Haar requires keyframe_interval to be power of two (got {}).",
                        keyframe_interval
                    );
                    std::process::exit(1);
                }
            }

            // Load frames from pattern (e.g., "frames/frame_%04d.png")
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

            let mut summary_ip: Option<sequence_metrics::SequenceSummary> = None;
            let mut summary_i: Option<sequence_metrics::SequenceSummary> = None;
            let mut total_bytes_ip: usize = 0;
            let mut total_bytes_i: usize = 0;
            let mut avg_bpp_ip: f64 = 0.0;
            let mut avg_bpp_i: f64 = 0.0;
            let mut frame_metrics_ip: Vec<FrameMetrics> = Vec::new();
            let mut frame_metrics_i: Vec<FrameMetrics> = Vec::new();

            if run_baseline {
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
            total_bytes_i = 0;
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

            avg_bpp_i =
                compressed_i.iter().map(|f| f.bpp()).sum::<f64>() / compressed_i.len() as f64;

            println!(
                "  Total: {} bytes, avg {:.2} bpp, {:.1}ms ({:.1} fps)",
                total_bytes_i,
                avg_bpp_i,
                elapsed_i.as_secs_f64() * 1000.0,
                compressed_i.len() as f64 / elapsed_i.as_secs_f64(),
            );

            // Compute and display I-only sequence summary
            let summary_i_local = sequence_metrics::compute_sequence_metrics(&frame_metrics_i);
            summary_i = Some(summary_i_local.clone());
            println!("\n{}", summary_i_local);

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
                summary_i.as_ref().unwrap().max_psnr_drop,
                summary_i.as_ref().unwrap().temporal_consistency,
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
                }
                if rans {
                    config_tw.entropy_coder = gnc::EntropyCoder::Rans;
                }
                println!(
                    "Temporal config: qstep {:.3}, dead_zone {:.3}, entropy {:?}",
                    config_tw.quantization_step, config_tw.dead_zone, config_tw.entropy_coder
                );

                let start = std::time::Instant::now();
                let encoded_tw = encoder.encode_sequence_temporal_wavelet(
                    &ctx,
                    &frame_refs,
                    w,
                    h,
                    &config_tw,
                    temporal_mode,
                    keyframe_interval as usize,
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
                    for i in 0..gop {
                        originals.push(encoder.debug_wavelet_prequant(
                            &ctx,
                            &frames_data[i],
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

                            if let Some(first_lvl) = group.high_frames.get(0) {
                                if let Some(first_high) = first_lvl.get(0) {
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
                    let mut accumulate_stats = |frames: &[&CompressedFrame]| -> (f64, f64) {
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
            diagnostics,
        } => {
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

            println!(
                "Encoding sequence: {} frames from '{}', ki={}, q={}, {}fps",
                frame_count,
                input,
                keyframe_interval,
                quality,
                if fps_den == 1 {
                    format!("{}", fps_num)
                } else {
                    format!("{:.3}", fps_num as f64 / fps_den as f64)
                }
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

            // Apply rate control settings
            if let Some(ref br) = bitrate {
                config.target_bitrate = Some(parse_bitrate(br));
                config.rate_mode = parse_rate_mode(&rate_mode);
            }

            let actual_fps = fps_num as f64 / fps_den as f64;
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

            // Print per-frame summary
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

            // Serialize to GNV1 container
            let gnv_data = serialize_sequence(&compressed, (fps_num, fps_den));
            std::fs::write(&output, &gnv_data).expect("Failed to write output file");

            let avg_bpp = compressed.iter().map(|f| f.bpp()).sum::<f64>() / compressed.len() as f64;
            let i_count = compressed
                .iter()
                .filter(|f| f.frame_type == gnc::FrameType::Intra)
                .count();
            let fps = compressed.len() as f64 / encode_time.as_secs_f64();

            println!(
                "\nEncoded {} frames ({}I + {}P) in {:.1}ms ({:.1} fps)",
                compressed.len(),
                i_count,
                compressed.len() - i_count,
                encode_time.as_secs_f64() * 1000.0,
                fps,
            );
            println!(
                "Container: {} bytes, avg {:.2} bpp",
                gnv_data.len(),
                avg_bpp
            );
            println!("Written to {}", output);
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
