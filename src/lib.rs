pub mod bench;
pub mod decoder;
pub mod encoder;
pub mod experiments;
pub mod format;
pub mod gpu_util;
pub mod image_util;

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

    /// Perceptual weights: quantize inner detail harder (less energy), preserve outer detail.
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
            chroma_weight: 1.0,
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
                w[base] = level[0]; // LH
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

/// Quantized CfL (Chroma-from-Luma) alpha coefficients.
///
/// For each tile and each wavelet subband, stores the linear scaling factor
/// `alpha` that predicts chroma from luma: `chroma ≈ alpha * luma`.
/// Encoding the residual `chroma - alpha * luma` instead of raw chroma
/// reduces chroma entropy.
#[derive(Debug, Clone)]
pub struct CflAlphas {
    /// Quantized alpha values (u8), layout: [tile0_co_sb0, tile0_co_sb1, ..., tile0_cg_sb0, ...]
    pub alphas: Vec<u8>,
    /// Number of subbands per tile (1 LL + 3 * num_levels detail = 1 + 3*L)
    pub num_subbands: u32,
}

/// Frame type for temporal coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// I-frame: self-contained, no reference needed
    Intra,
    /// P-frame: encoded as residual from previous frame
    Predicted,
}

/// Per-tile motion vectors: one (dx, dy) per ME_BLOCK_SIZE block.
#[derive(Debug, Clone)]
pub struct MotionField {
    /// (dx, dy) per block, row-major within tile, tiles ordered raster
    pub vectors: Vec<[i16; 2]>,
    /// Block size used for motion estimation (typically 16)
    pub block_size: u32,
}

/// Which wavelet transform to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletType {
    /// LeGall 5/3 — integer-exact, lossless-capable
    LeGall53,
    /// CDF 9/7 — better energy compaction for lossy compression (JPEG 2000 lossy wavelet)
    CDF97,
}

/// Which entropy coder to use for the final coding stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyCoder {
    /// rANS with interleaved streams and zero-run-length pre-coding (CPU encode, GPU decode)
    Rans,
    /// Bitplane coding: sign + magnitude bitplanes per 32x32 block (CPU encode, GPU decode).
    /// Fully parallel on decode, no serial state machines.
    Bitplane,
}

/// Codec configuration
#[derive(Debug, Clone)]
pub struct CodecConfig {
    pub tile_size: u32,
    pub quantization_step: f32,
    pub dead_zone: f32,
    pub wavelet_levels: u32,
    pub subband_weights: SubbandWeights,
    pub cfl_enabled: bool,
    /// Which entropy coder to use (default: Rans for backward compatibility)
    pub entropy_coder: EntropyCoder,
    /// Which wavelet transform to use (default: LeGall53 for backward compatibility)
    pub wavelet_type: WaveletType,
    /// Enable SSIM-guided adaptive quantization.
    /// When enabled, per-block variance analysis drives spatial quantization weighting:
    /// smooth regions get coarser quantization, textured regions get finer.
    pub adaptive_quantization: bool,
    /// Strength of adaptive quantization (0.0 = off, 1.0 = full strength).
    /// Controls how aggressively bits are redistributed from smooth to textured regions.
    pub aq_strength: f32,
    /// Enable per-subband entropy coding (separate frequency tables per wavelet level).
    /// Each subband group (LL + one per detail level) gets its own rANS frequency table,
    /// improving compression by modeling each distribution tightly.
    pub per_subband_entropy: bool,
    /// Keyframe interval for temporal coding.
    /// 1 = all I-frames (default, backward compat). N > 1 = I-frame every N frames.
    pub keyframe_interval: u32,
    /// Use GPU compute shaders for rANS entropy encoding (default: true).
    /// When false, falls back to CPU entropy encoding.
    pub gpu_entropy_encode: bool,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            tile_size: 256,
            quantization_step: 4.0,
            dead_zone: 0.0,
            wavelet_levels: 3,
            subband_weights: SubbandWeights::uniform(3),
            cfl_enabled: false,
            entropy_coder: EntropyCoder::Rans,
            wavelet_type: WaveletType::LeGall53,
            adaptive_quantization: false,
            aq_strength: 0.0,
            per_subband_entropy: false,
            keyframe_interval: 1,
            gpu_entropy_encode: true,
        }
    }
}

/// Map a quality value (1–100) to codec parameters.
///
/// Higher values = better quality, larger files. Lower = more compression.
/// Anchor points based on research log experiments:
/// - 100: near-lossless (qstep=1, LeGall 5/3, uniform weights)
/// - 90: high quality (qstep=2, CDF 9/7, CfL, per-subband)
/// - 75: good quality (qstep=4, perceptual weights, dead zone 0.5)
/// - 50: balanced (qstep=8, dead zone 0.75)
/// - 25: high compression (qstep=16)
/// - 10: maximum compression (qstep=32)
///
/// Intermediate values interpolate between anchor points (log-scale for qstep).
pub fn quality_preset(q: u32) -> CodecConfig {
    let q = q.clamp(1, 100);

    struct Anchor {
        q: u32,
        qstep: f32,
        dead_zone: f32,
        perceptual: bool,
        cdf97: bool,
        cfl: bool,
        per_subband: bool,
    }

    let anchors: &[Anchor] = &[
        Anchor {
            q: 1,
            qstep: 64.0,
            dead_zone: 1.0,
            perceptual: true,
            cdf97: true,
            cfl: true,
            per_subband: true,
        },
        Anchor {
            q: 10,
            qstep: 32.0,
            dead_zone: 0.75,
            perceptual: true,
            cdf97: true,
            cfl: true,
            per_subband: true,
        },
        Anchor {
            q: 25,
            qstep: 16.0,
            dead_zone: 0.75,
            perceptual: true,
            cdf97: true,
            cfl: true,
            per_subband: true,
        },
        Anchor {
            q: 50,
            qstep: 8.0,
            dead_zone: 0.75,
            perceptual: true,
            cdf97: true,
            cfl: true,
            per_subband: true,
        },
        Anchor {
            q: 75,
            qstep: 4.0,
            dead_zone: 0.5,
            perceptual: true,
            cdf97: true,
            cfl: true,
            per_subband: true,
        },
        Anchor {
            q: 90,
            qstep: 2.0,
            dead_zone: 0.0,
            perceptual: false,
            cdf97: true,
            cfl: true,
            per_subband: true,
        },
        Anchor {
            q: 100,
            qstep: 1.0,
            dead_zone: 0.0,
            perceptual: false,
            cdf97: false,
            cfl: false,
            per_subband: false,
        },
    ];

    // Find surrounding anchors and interpolation factor
    let (lo, hi, t) = if q <= anchors[0].q {
        (&anchors[0], &anchors[0], 0.0f32)
    } else if q >= anchors[anchors.len() - 1].q {
        let last = &anchors[anchors.len() - 1];
        (last, last, 0.0f32)
    } else {
        let idx = anchors.iter().position(|a| a.q >= q).unwrap();
        if anchors[idx].q == q {
            (&anchors[idx], &anchors[idx], 0.0f32)
        } else {
            let lo = &anchors[idx - 1];
            let hi = &anchors[idx];
            let t = (q - lo.q) as f32 / (hi.q - lo.q) as f32;
            (lo, hi, t)
        }
    };

    // Log-interpolate qstep for perceptually uniform spacing
    let qstep = (lo.qstep.ln() + t * (hi.qstep.ln() - lo.qstep.ln())).exp();
    // Linear-interpolate dead zone
    let dead_zone = lo.dead_zone + t * (hi.dead_zone - lo.dead_zone);

    // Discrete settings: use lower-quality anchor until midpoint
    let disc = if t < 0.5 { lo } else { hi };

    let wavelet_levels = if q >= 60 { 4 } else { 3 };
    let aq_enabled = false; // AQ broken: spatial weights applied to wavelet-domain positions
    let aq_strength = if q >= 70 { 0.4 } else { 0.3 };
    let mut weights = if disc.perceptual {
        SubbandWeights::perceptual(wavelet_levels)
    } else {
        SubbandWeights::uniform(wavelet_levels)
    };
    if q < 50 {
        weights.chroma_weight = 1.5;
    } else if q < 70 {
        weights.chroma_weight = 1.1;
    }
    CodecConfig {
        quantization_step: qstep,
        dead_zone,
        wavelet_levels,
        subband_weights: weights,
        cfl_enabled: false, // CfL disabled for now — needs per-subband alpha quantization fixes
        wavelet_type: if disc.cdf97 {
            WaveletType::CDF97
        } else {
            WaveletType::LeGall53
        },
        per_subband_entropy: disc.per_subband,
        adaptive_quantization: aq_enabled,
        aq_strength,
        ..Default::default()
    }
}

/// Entropy-coded tile data — rANS, per-subband rANS, or bitplane coded.
/// Tiles are ordered: plane 0 tiles, plane 1 tiles, plane 2 tiles.
#[derive(Debug, Clone)]
pub enum EntropyData {
    Rans(Vec<encoder::rans::InterleavedRansTile>),
    SubbandRans(Vec<encoder::rans::SubbandRansTile>),
    Bitplane(Vec<encoder::bitplane::BitplaneTile>),
}

impl EntropyData {
    pub fn byte_size(&self) -> usize {
        match self {
            EntropyData::Rans(tiles) => tiles.iter().map(|t| t.byte_size()).sum(),
            EntropyData::SubbandRans(tiles) => tiles.iter().map(|t| t.byte_size()).sum(),
            EntropyData::Bitplane(tiles) => tiles.iter().map(|t| t.byte_size()).sum(),
        }
    }
}

/// Compressed frame data
#[derive(Debug, Clone)]
pub struct CompressedFrame {
    pub info: FrameInfo,
    pub config: CodecConfig,
    /// Entropy-coded tile data (rANS or bitplane)
    pub entropy: EntropyData,
    /// CfL alpha coefficients (present when cfl_enabled)
    pub cfl_alphas: Option<CflAlphas>,
    /// Per-block spatial weight map for adaptive quantization.
    /// `None` when adaptive quantization is disabled.
    /// Stored in row-major order, one f32 per AQ_BLOCK_SIZE x AQ_BLOCK_SIZE block.
    pub weight_map: Option<Vec<f32>>,
    /// Frame type for temporal coding (Intra or Predicted)
    pub frame_type: FrameType,
    /// Motion field for P-frames (None for I-frames)
    pub motion_field: Option<MotionField>,
}

impl CompressedFrame {
    /// Total compressed size in bytes (all tiles + CfL alpha + weight map + MV overhead)
    pub fn byte_size(&self) -> usize {
        let tile_bytes = self.entropy.byte_size();
        let cfl_bytes = self.cfl_alphas.as_ref().map_or(0, |a| a.alphas.len());
        let wm_bytes = self
            .weight_map
            .as_ref()
            .map_or(0, |wm| wm.len() * std::mem::size_of::<f32>());
        let mv_bytes = self
            .motion_field
            .as_ref()
            .map_or(0, |mf| mf.vectors.len() * 4); // 2 × i16 per block
        tile_bytes + cfl_bytes + wm_bytes + mv_bytes
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
