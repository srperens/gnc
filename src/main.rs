use clap::{Parser, Subcommand};
use gnc::bench::compare::{write_csv, BenchmarkConfig, BenchmarkResult};
use gnc::bench::quality::{self, QualityMetrics};
use gnc::bench::throughput;
use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::experiments;
use gnc::format::{deserialize_compressed, serialize_compressed};
use gnc::image_util::{load_image_rgb_f32, parse_wavelet_type, save_image_rgb_f32};
use gnc::{CodecConfig, GpuContext};

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

        /// Use bitplane entropy coder instead of rANS
        #[arg(long)]
        bitplane: bool,

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

        /// Use bitplane entropy coder instead of rANS
        #[arg(long)]
        bitplane: bool,

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
            if cpu_encode {
                config.gpu_entropy_encode = false;
            }

            let coder_name = match config.entropy_coder {
                gnc::EntropyCoder::Bitplane => "bitplane".to_string(),
                gnc::EntropyCoder::Rans if config.per_subband_entropy => "rANS subband".to_string(),
                gnc::EntropyCoder::Rans => "rANS single-table".to_string(),
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
        } => {
            let qstep_display = qstep
                .map(|q| format!("{}", q))
                .or_else(|| quality.map(|q| format!("q={}", q)))
                .unwrap_or_else(|| "4.0".to_string());
            println!(
                "Sequence benchmark: pattern={}, frames={}, ki={}, {}",
                input, num_frames, keyframe_interval, qstep_display
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

            let start = std::time::Instant::now();
            let compressed_ip = encoder.encode_sequence(&ctx, &frame_refs, w, h, &config_ip);
            let elapsed_ip = start.elapsed();

            println!("\n=== I+P (keyframe_interval={}) ===", keyframe_interval);
            let mut total_bytes_ip: usize = 0;
            for (i, cf) in compressed_ip.iter().enumerate() {
                let ft = if cf.frame_type == gnc::FrameType::Intra {
                    "I"
                } else {
                    "P"
                };
                let decoded = decoder.decode(&ctx, cf);
                let psnr = quality::psnr(&frames_data[i], &decoded, 255.0);
                total_bytes_ip += cf.byte_size();
                println!(
                    "  Frame {:2} [{}]: {:6} bytes, {:.2} bpp, PSNR {:.2} dB",
                    i,
                    ft,
                    cf.byte_size(),
                    cf.bpp(),
                    psnr,
                );
            }

            let avg_bpp_ip =
                compressed_ip.iter().map(|f| f.bpp()).sum::<f64>() / compressed_ip.len() as f64;
            let i_count = compressed_ip
                .iter()
                .filter(|f| f.frame_type == gnc::FrameType::Intra)
                .count();

            println!(
                "  Total: {} bytes, avg {:.2} bpp, {:.1}ms ({:.1} fps), {}I+{}P",
                total_bytes_ip,
                avg_bpp_ip,
                elapsed_ip.as_secs_f64() * 1000.0,
                compressed_ip.len() as f64 / elapsed_ip.as_secs_f64(),
                i_count,
                compressed_ip.len() - i_count,
            );

            // --- All I-frame baseline ---
            let mut config_i = config_ip.clone();
            config_i.keyframe_interval = 1;

            let start = std::time::Instant::now();
            let compressed_i = encoder.encode_sequence(&ctx, &frame_refs, w, h, &config_i);
            let elapsed_i = start.elapsed();

            println!("\n=== All I-frames (baseline) ===");
            let mut total_bytes_i: usize = 0;
            for (i, cf) in compressed_i.iter().enumerate() {
                let decoded = decoder.decode(&ctx, cf);
                let psnr = quality::psnr(&frames_data[i], &decoded, 255.0);
                total_bytes_i += cf.byte_size();
                println!(
                    "  Frame {:2} [I]: {:6} bytes, {:.2} bpp, PSNR {:.2} dB",
                    i,
                    cf.byte_size(),
                    cf.bpp(),
                    psnr,
                );
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

            // --- Comparison ---
            let saving_pct = (1.0 - total_bytes_ip as f64 / total_bytes_i as f64) * 100.0;
            println!(
                "\n=== Comparison ===\n  I-only: {} bytes ({:.2} bpp)\n  I+P:    {} bytes ({:.2} bpp)\n  Saving: {:.1}%",
                total_bytes_i, avg_bpp_i, total_bytes_ip, avg_bpp_ip, saving_pct,
            );
        }
    }
}
