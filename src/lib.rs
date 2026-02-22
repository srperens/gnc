pub mod bench;
pub mod decoder;
pub mod encoder;
pub mod experiments;

use wgpu;

/// Frame dimensions and format info
#[derive(Debug, Clone, Copy)]
pub struct FrameInfo {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u32, // 8 or 10
    pub tile_size: u32, // e.g., 256
}

impl FrameInfo {
    pub fn tiles_x(&self) -> u32 {
        (self.width + self.tile_size - 1) / self.tile_size
    }

    pub fn tiles_y(&self) -> u32 {
        (self.height + self.tile_size - 1) / self.tile_size
    }

    pub fn total_tiles(&self) -> u32 {
        self.tiles_x() * self.tiles_y()
    }

    pub fn pixel_count(&self) -> u32 {
        self.width * self.height
    }

    /// Padded dimensions (rounded up to tile_size)
    pub fn padded_width(&self) -> u32 {
        self.tiles_x() * self.tile_size
    }

    pub fn padded_height(&self) -> u32 {
        self.tiles_y() * self.tile_size
    }
}

/// Per-subband quantization weight multipliers.
///
/// Higher weight = coarser quantization (more compression, less quality).
/// The effective step size for a coefficient is `base_step * weight`.
#[derive(Debug, Clone, PartialEq)]
pub struct SubbandWeights {
    /// Weight for the LL (DC) subband
    pub ll: f32,
    /// Per-level detail weights: \[LH, HL, HH\] for each decomposition level.
    /// Index 0 = outermost (highest spatial frequency), last = innermost.
    pub detail: Vec<[f32; 3]>,
    /// Multiplier applied to all weights for chroma planes (Co, Cg)
    pub chroma_weight: f32,
}

impl SubbandWeights {
    /// All weights = 1.0, reproducing the old uniform quantization behavior.
    pub fn uniform(levels: u32) -> Self {
        Self {
            ll: 1.0,
            detail: vec![[1.0, 1.0, 1.0]; levels as usize],
            chroma_weight: 1.0,
        }
    }

    /// Perceptual weights: quantize HH harder, inner levels harder, chroma harder.
    pub fn perceptual(levels: u32) -> Self {
        let mut detail = Vec::with_capacity(levels as usize);
        for i in 0..levels as usize {
            let lh_hl = 1.0 + 0.5 * i as f32;
            let is_innermost = i == levels as usize - 1;
            let hh = lh_hl + if is_innermost { 1.0 } else { 0.5 };
            detail.push([lh_hl, lh_hl, hh]);
        }
        Self {
            ll: 1.0,
            detail,
            chroma_weight: 1.5,
        }
    }

    /// Pack into 16 f32s for the GPU uniform buffer (4 × vec4).
    /// Layout: [LL, L0_LH, L0_HL, L0_HH, L1_LH, L1_HL, L1_HH, ...]
    pub fn pack_weights(&self) -> [f32; 16] {
        let mut w = [1.0f32; 16];
        w[0] = self.ll;
        for (i, level) in self.detail.iter().enumerate() {
            let base = 1 + i * 3;
            if base + 2 < 16 {
                w[base] = level[0];     // LH
                w[base + 1] = level[1]; // HL
                w[base + 2] = level[2]; // HH
            }
        }
        w
    }

    /// Pack with chroma multiplier applied to all weights.
    pub fn pack_weights_chroma(&self) -> [f32; 16] {
        let mut w = self.pack_weights();
        for v in &mut w {
            *v *= self.chroma_weight;
        }
        w
    }
}

/// Codec configuration
#[derive(Debug, Clone)]
pub struct CodecConfig {
    pub tile_size: u32,
    pub quantization_step: f32,
    pub dead_zone: f32,
    pub wavelet_levels: u32,
    pub subband_weights: SubbandWeights,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            tile_size: 256,
            quantization_step: 4.0,
            dead_zone: 0.0,
            wavelet_levels: 3,
            subband_weights: SubbandWeights::uniform(3),
        }
    }
}

/// Compressed frame data (interleaved rANS entropy coded, per-tile)
#[derive(Debug, Clone)]
pub struct CompressedFrame {
    pub info: FrameInfo,
    pub config: CodecConfig,
    /// Per-tile interleaved rANS compressed data, ordered: plane 0 tiles, plane 1 tiles, plane 2 tiles
    pub tiles: Vec<encoder::rans::InterleavedRansTile>,
}

impl CompressedFrame {
    /// Total compressed size in bytes (all tiles)
    pub fn byte_size(&self) -> usize {
        self.tiles.iter().map(|t| t.byte_size()).sum()
    }

    /// Bits per pixel
    pub fn bpp(&self) -> f64 {
        (self.byte_size() as f64 * 8.0) / (self.info.width as f64 * self.info.height as f64)
    }
}

/// GPU context shared across encoder/decoder
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub fn new() -> Self {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        log::info!("Using GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gpu-codec device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("Failed to create device");

        Self { device, queue }
    }
}
