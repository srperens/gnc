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

/// Codec configuration
#[derive(Debug, Clone)]
pub struct CodecConfig {
    pub tile_size: u32,
    pub quantization_step: f32,
    pub dead_zone: f32,
    pub wavelet_levels: u32,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            tile_size: 256,
            quantization_step: 4.0,
            dead_zone: 0.0,
            wavelet_levels: 3,
        }
    }
}

/// Compressed frame data (rANS entropy coded, per-tile)
#[derive(Debug, Clone)]
pub struct CompressedFrame {
    pub info: FrameInfo,
    pub config: CodecConfig,
    /// Per-tile rANS compressed data, ordered: plane 0 tiles, plane 1 tiles, plane 2 tiles
    pub tiles: Vec<encoder::rans::RansTile>,
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
