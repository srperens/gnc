use clap::{Parser, Subcommand};
use gnc::bench::compare::{write_csv, BenchmarkConfig, BenchmarkResult};
use gnc::bench::quality::{self, QualityMetrics};
use gnc::bench::throughput;
use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::experiments;
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

        /// Quantization step size
        #[arg(short, long, default_value = "4.0")]
        qstep: f32,

        /// Tile size
        #[arg(short, long, default_value = "256")]
        tile_size: u32,
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
    },

    /// Run all Phase 1 experiments on an input image
    Sweep {
        /// Input image file
        #[arg(short, long)]
        input: String,

        /// Output CSV file for results
        #[arg(long, default_value = "results/phase1_sweep.csv")]
        csv: String,
    },
}

fn load_image_rgb_f32(path: &str) -> (Vec<f32>, u32, u32) {
    let img = image::open(path).expect("Failed to open image").to_rgb8();
    let (w, h) = img.dimensions();
    let pixels: Vec<f32> = img.as_raw().iter().map(|&v| v as f32).collect();
    (pixels, w, h)
}

fn save_image_rgb_f32(path: &str, data: &[f32], width: u32, height: u32) {
    let bytes: Vec<u8> = data
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    let img = image::RgbImage::from_raw(width, height, bytes).expect("Failed to create image");
    img.save(path).expect("Failed to save image");
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Command::Encode {
            input,
            output,
            qstep,
            tile_size,
        } => {
            let (rgb_data, w, h) = load_image_rgb_f32(&input);
            println!("Input: {}x{} ({} pixels)", w, h, w * h);

            let ctx = GpuContext::new();
            let encoder = EncoderPipeline::new(&ctx);

            let config = CodecConfig {
                tile_size,
                quantization_step: qstep,
                dead_zone: 0.0,
                wavelet_levels: 3,
            };

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
        } => {
            let (rgb_data, w, h) = load_image_rgb_f32(&input);
            println!("Benchmark: {}x{} image, {} iterations", w, h, iterations);

            let ctx = GpuContext::new();
            let encoder = EncoderPipeline::new(&ctx);
            let decoder = DecoderPipeline::new(&ctx);

            let config = CodecConfig::default();

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
            let tp = throughput::measure_throughput(
                || {
                    encoder.encode(&ctx, &rgb_data, w, h, &config);
                },
                || {
                    decoder.decode(&ctx, &compressed);
                },
                w,
                h,
                iterations,
            );
            println!("Throughput: {}", tp);

            if let Some(csv_path) = csv {
                let result = BenchmarkResult {
                    name: "baseline".to_string(),
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

        Command::Sweep { input, csv } => {
            let (rgb_data, w, h) = load_image_rgb_f32(&input);
            println!("Phase 1 sweep: {}x{} image", w, h);

            let ctx = GpuContext::new();
            let encoder = EncoderPipeline::new(&ctx);
            let decoder = DecoderPipeline::new(&ctx);

            let mut all_experiments = experiments::phase1_experiments();
            all_experiments.extend(experiments::wavelet_level_experiments());
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
    }
}

// Serialization for compressed frames (rANS per-tile)
fn serialize_compressed(frame: &gnc::CompressedFrame) -> Vec<u8> {
    use gnc::encoder::rans;
    let mut out = Vec::new();
    // Header
    out.extend_from_slice(b"GPC2"); // version 2 = rANS
    out.extend_from_slice(&frame.info.width.to_le_bytes());
    out.extend_from_slice(&frame.info.height.to_le_bytes());
    out.extend_from_slice(&frame.info.bit_depth.to_le_bytes());
    out.extend_from_slice(&frame.info.tile_size.to_le_bytes());
    out.extend_from_slice(&frame.config.quantization_step.to_le_bytes());
    out.extend_from_slice(&frame.config.dead_zone.to_le_bytes());
    out.extend_from_slice(&frame.config.wavelet_levels.to_le_bytes());
    let num_tiles = frame.tiles.len() as u32;
    out.extend_from_slice(&num_tiles.to_le_bytes());
    // Per-tile data
    for tile in &frame.tiles {
        let tile_bytes = rans::serialize_tile(tile);
        out.extend_from_slice(&tile_bytes);
    }
    out
}

fn deserialize_compressed(data: &[u8]) -> gnc::CompressedFrame {
    use gnc::encoder::rans;
    assert!(data.len() >= 36, "File too small");
    assert_eq!(&data[0..4], b"GPC2", "Invalid magic (expected GPC2 for rANS format)");

    let width = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let height = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let bit_depth = u32::from_le_bytes(data[12..16].try_into().unwrap());
    let tile_size = u32::from_le_bytes(data[16..20].try_into().unwrap());
    let qstep = f32::from_le_bytes(data[20..24].try_into().unwrap());
    let dead_zone = f32::from_le_bytes(data[24..28].try_into().unwrap());
    let wavelet_levels = u32::from_le_bytes(data[28..32].try_into().unwrap());
    let num_tiles = u32::from_le_bytes(data[32..36].try_into().unwrap()) as usize;

    let mut pos = 36;
    let mut tiles = Vec::with_capacity(num_tiles);
    for _ in 0..num_tiles {
        let (tile, consumed) = rans::deserialize_tile(&data[pos..]);
        tiles.push(tile);
        pos += consumed;
    }

    gnc::CompressedFrame {
        info: gnc::FrameInfo {
            width,
            height,
            bit_depth,
            tile_size,
        },
        config: gnc::CodecConfig {
            tile_size,
            quantization_step: qstep,
            dead_zone,
            wavelet_levels,
        },
        tiles,
    }
}
