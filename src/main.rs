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
    seek_to_keyframe, serialize_compressed, serialize_sequence,
};
use gnc::image_util::{load_image_rgb_f32, parse_wavelet_type, save_image_rgb_f32};
use gnc::{CodecConfig, GpuContext, RateMode};

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
        } => {
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
            let mut total_bytes_ip: usize = 0;
            let mut frame_metrics_ip: Vec<FrameMetrics> = Vec::new();
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

            let avg_bpp_ip =
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
            let summary_ip = sequence_metrics::compute_sequence_metrics(&frame_metrics_ip);
            println!("\n{}", summary_ip);

            // --- All I-frame baseline ---
            let mut config_i = config_ip.clone();
            config_i.keyframe_interval = 1;

            let start = std::time::Instant::now();
            let compressed_i = encoder.encode_sequence(&ctx, &frame_refs, w, h, &config_i);
            let elapsed_i = start.elapsed();

            println!("=== All I-frames (baseline) ===");
            let mut total_bytes_i: usize = 0;
            let mut frame_metrics_i: Vec<FrameMetrics> = Vec::new();
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
                summary_ip.max_psnr_drop,
                summary_ip.temporal_consistency,
                summary_i.max_psnr_drop,
                summary_i.temporal_consistency,
            );

            // Write CSV if requested (I+P metrics — the primary encoding mode)
            if let Some(csv_path) = csv {
                sequence_metrics::write_sequence_csv(&csv_path, &frame_metrics_ip, &summary_ip)
                    .expect("Failed to write sequence CSV");
                println!("\nSequence metrics written to {}", csv_path);
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
        } => {
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

            // Load all frames
            let mut frames_data: Vec<Vec<f32>> = Vec::with_capacity(frame_count);
            let mut w = 0u32;
            let mut h = 0u32;
            for i in 0..frame_count {
                let path = input.replace("%04d", &format!("{:04}", i));
                let (rgb, fw, fh) = load_image_rgb_f32(&path);
                if i == 0 {
                    w = fw;
                    h = fh;
                } else {
                    assert!(
                        fw == w && fh == h,
                        "Frame {} has different dimensions ({}x{} vs {}x{})",
                        i,
                        fw,
                        fh,
                        w,
                        h
                    );
                }
                frames_data.push(rgb);
            }

            let frame_refs: Vec<&[f32]> = frames_data.iter().map(|f| f.as_slice()).collect();

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
            let start = std::time::Instant::now();
            let compressed =
                encoder.encode_sequence_with_fps(&ctx, &frame_refs, w, h, &config, actual_fps);
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
                    (seek_time * header.framerate_num as f64 / header.framerate_den as f64) as u64
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

            // For P-frame decoding, we need the decoder to maintain reference frames.
            // We use the DecoderPipeline's standard decode path for each frame.
            // P-frames need their reference, so we decode sequentially from the keyframe.
            let mut output_count = 0usize;

            for idx in start_frame..end_frame {
                let compressed = deserialize_sequence_frame(&gnv_data, &header, idx);
                let rgb_data = decoder.decode(&ctx, &compressed);

                // Only write output frames (when not seeking, write all; when seeking,
                // only write frames that are at or after the seek target)
                let should_output = if seek.is_some() {
                    // When seeking, output only the target frame (last in range)
                    idx == end_frame - 1
                } else {
                    true
                };

                if should_output {
                    let frame_path = output.replace("%04d", &format!("{:04}", idx));
                    save_image_rgb_f32(&frame_path, &rgb_data, header.width, header.height);
                    output_count += 1;
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

            let results = experiments::transform_shootout::run_shootout(
                &ctx, &rgb_data, w, h, iterations,
            );
            experiments::transform_shootout::print_results(&results);
        }
    }
}
