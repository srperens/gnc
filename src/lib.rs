#![forbid(unsafe_code)]

pub mod bench;
pub mod decoder;
pub mod encoder;
pub mod experiments;
pub mod format;
pub mod gpu_util;
pub mod image_util;
pub mod temporal;

/// Chroma subsampling format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChromaFormat {
    /// 4:4:4 — Co and Cg at full luma resolution (current default)
    #[default]
    Yuv444,
    /// 4:2:2 — Co and Cg halved horizontally only
    Yuv422,
    /// 4:2:0 — Co and Cg halved in both horizontal and vertical
    Yuv420,
}

impl ChromaFormat {
    /// Horizontal log2 scale factor for chroma (0 = no reduction, 1 = half width)
    pub fn horiz_shift(self) -> u32 {
        match self {
            ChromaFormat::Yuv444 => 0,
            ChromaFormat::Yuv422 | ChromaFormat::Yuv420 => 1,
        }
    }
    /// Vertical log2 scale factor for chroma (0 = no reduction, 1 = half height)
    pub fn vert_shift(self) -> u32 {
        match self {
            ChromaFormat::Yuv444 | ChromaFormat::Yuv422 => 0,
            ChromaFormat::Yuv420 => 1,
        }
    }
    /// Encode to u8 for bitstream
    pub fn to_u8(self) -> u8 {
        match self {
            ChromaFormat::Yuv444 => 0,
            ChromaFormat::Yuv422 => 1,
            ChromaFormat::Yuv420 => 2,
        }
    }
    /// Decode from u8
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(ChromaFormat::Yuv444),
            1 => Some(ChromaFormat::Yuv422),
            2 => Some(ChromaFormat::Yuv420),
            _ => None,
        }
    }
}

/// Frame dimensions and format info
#[derive(Debug, Clone, Copy)]
pub struct FrameInfo {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u32, // 8 or 10
    pub tile_size: u32, // e.g., 256
    pub chroma_format: ChromaFormat,
}

impl FrameInfo {
    pub fn tiles_x(&self) -> u32 {
        self.width.div_ceil(self.tile_size)
    }

    pub fn tiles_y(&self) -> u32 {
        self.height.div_ceil(self.tile_size)
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

    /// Chroma plane width (after subsampling)
    pub fn chroma_width(&self) -> u32 {
        self.width >> self.chroma_format.horiz_shift()
    }

    /// Chroma plane height (after subsampling)
    pub fn chroma_height(&self) -> u32 {
        self.height >> self.chroma_format.vert_shift()
    }

    /// Chroma padded width (rounded up to tile_size)
    pub fn chroma_padded_width(&self) -> u32 {
        self.chroma_width().div_ceil(self.tile_size) * self.tile_size
    }

    /// Chroma padded height (rounded up to tile_size)
    pub fn chroma_padded_height(&self) -> u32 {
        self.chroma_height().div_ceil(self.tile_size) * self.tile_size
    }

    /// Number of chroma tiles in X direction
    pub fn chroma_tiles_x(&self) -> u32 {
        self.chroma_width().div_ceil(self.tile_size)
    }

    /// Number of chroma tiles in Y direction
    pub fn chroma_tiles_y(&self) -> u32 {
        self.chroma_height().div_ceil(self.tile_size)
    }

    /// Total chroma tiles per plane
    pub fn chroma_tiles_per_plane(&self) -> usize {
        (self.chroma_tiles_x() * self.chroma_tiles_y()) as usize
    }

    /// Total luma tiles per plane
    pub fn luma_tiles_per_plane(&self) -> usize {
        (self.tiles_x() * self.tiles_y()) as usize
    }

    /// Build a FrameInfo describing the chroma plane in its own coordinate space.
    ///
    /// For 4:4:4 this is identical to `*self`.
    /// For 4:2:2 / 4:2:0 the width/height are the subsampled chroma dimensions, and
    /// `chroma_format` is set to `Yuv444` so that `tiles_x()`/`tiles_y()` return the
    /// correct tile count for the (smaller) chroma plane rather than luma tile count.
    pub fn make_chroma_info(&self) -> FrameInfo {
        FrameInfo {
            width: self.chroma_width(),
            height: self.chroma_height(),
            chroma_format: ChromaFormat::Yuv444,
            ..*self
        }
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
///
/// Alphas are quantized to i16 with 14-bit precision: [-16384, 16384] maps
/// to [-2.0, 2.0] with step size ~0.000244 (vs ~0.0157 for old u8 encoding).
#[derive(Debug, Clone)]
pub struct CflAlphas {
    /// Quantized alpha values (i16), layout: [tile0_co_sb0, tile0_co_sb1, ..., tile0_cg_sb0, ...]
    pub alphas: Vec<i16>,
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
    /// B-frame: bidirectional prediction from forward + backward references
    Bidirectional,
}

/// Per-tile motion vectors: one (dx, dy) per ME_BLOCK_SIZE block.
#[derive(Debug, Clone)]
pub struct MotionField {
    /// Forward (dx, dy) per block, row-major within tile, tiles ordered raster.
    /// Half-pel units.
    pub vectors: Vec<[i16; 2]>,
    /// Block size used for motion estimation (typically 16)
    pub block_size: u32,
    /// Backward motion vectors (B-frames only). Same layout as `vectors`.
    pub backward_vectors: Option<Vec<[i16; 2]>>,
    /// Per-block prediction mode (B-frames only): 0=fwd, 1=bwd, 2=bidir
    pub block_modes: Option<Vec<u8>>,
}

/// Which wavelet transform to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletType {
    /// LeGall 5/3 — integer-exact, lossless-capable
    LeGall53,
    /// CDF 9/7 — better energy compaction for lossy compression (JPEG 2000 lossy wavelet)
    CDF97,
}

/// Top-level transform selection: wavelet (multi-level, multi-dispatch) or
/// block-based (single-dispatch, fused with quantize).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformType {
    /// Multi-level wavelet (CDF 9/7 or LeGall 5/3). Current default.
    Wavelet,
    /// Block DCT-8×8: single dispatch per plane, fuseable with quantize.
    /// Replaces 24+ wavelet dispatches with 1 per plane.
    BlockDCT8,
}

/// Temporal transform mode (in-memory only for now; not serialized).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalTransform {
    /// No temporal transform; use normal I/P/B coding.
    None,
    /// Haar wavelet across 2 frames (low latency).
    Haar,
    /// LeGall 5/3 wavelet across 4 frames (better energy compaction).
    LeGall53,
}

/// Rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateMode {
    /// Constant bitrate: strictly constrain output to target bitrate via VBV buffer.
    CBR,
    /// Variable bitrate: allow frame-to-frame bitrate variation while targeting average.
    VBR,
}

/// Which entropy coder to use for the final coding stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyCoder {
    /// rANS with interleaved streams and zero-run-length pre-coding (CPU encode, GPU decode)
    Rans,
    /// Bitplane coding: sign + magnitude bitplanes per 32x32 block (CPU encode, GPU decode).
    /// Fully parallel on decode, no serial state machines.
    Bitplane,
    /// Significance map + Golomb-Rice: fully parallel, 256 streams per tile.
    /// Every coefficient encodes independently — no serial state chain.
    Rice,
    /// Significance map + canonical Huffman: 256 parallel streams per tile.
    /// Distribution-adaptive codes (vs Rice's fixed Golomb-Rice), closing the
    /// low-bitrate gap. Same ZRL scheme as Rice for zero runs.
    Huffman,
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
    /// Which entropy coder to use (default: Rans — best compression; Rice is faster but ~30% worse)
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
    /// Enable context-adaptive entropy coding (2 frequency tables per detail subband).
    /// When enabled with per_subband_entropy, each detail subband group gets 2 tables:
    /// one for coefficients whose "above" neighbor (same column, previous row within
    /// the same subband) is zero, and one for coefficients with nonzero above neighbor.
    /// Requires per_subband_entropy = true.
    pub context_adaptive: bool,
    /// Target bitrate in bits per second (e.g. 10_000_000.0 for 10 Mbps).
    /// When `None`, quality-based encoding is used (no rate control).
    pub target_bitrate: Option<f64>,
    /// Rate control mode: CBR (strict) or VBR (relaxed). Only used when target_bitrate is set.
    pub rate_mode: RateMode,
    /// Fuse quantization and histogram building into a single GPU dispatch.
    /// Eliminates one full read+write of the coefficient buffer per plane,
    /// saving ~24MB bandwidth at 1080p. Requires gpu_entropy_encode = true and
    /// non-CfL path (CfL needs separate quantize + dequantize for reconstruction).
    pub use_fused_quantize_histogram: bool,
    /// Transform type: Wavelet (default) or BlockDCT8 (fewer dispatches, faster).
    pub transform_type: TransformType,
    /// DCT frequency-dependent quantization strength.
    /// Controls how aggressively high-frequency DCT coefficients are quantized.
    /// 0.0 = flat (all frequencies equal), 3.0 = typical (highest freq gets 4× coarser step).
    /// Only used when transform_type == BlockDCT8.
    pub dct_freq_strength: f32,
    /// Enable spatial intra prediction before wavelet transform (Y plane only).
    /// Predicts each 8×8 block from left/top neighbors, encodes residual.
    /// 4 modes: DC, Horizontal, Vertical, Diagonal. 2 bits/block overhead.
    pub intra_prediction: bool,
    /// Temporal transform mode (in-memory only; not serialized yet).
    pub temporal_transform: TemporalTransform,
    /// Quantization step multiplier for temporal highpass bands.
    /// Higher values = coarser quantization of temporal detail = smaller files.
    /// 1.0 = same as lowpass, 2.0 = double qstep for highpass (good default).
    /// When `adaptive_temporal_mul` is true, this serves as the max_mul ceiling.
    pub temporal_highpass_qstep_mul: f32,
    /// When true, dynamically compute highpass mul from GOP energy.
    /// When false, use `temporal_highpass_qstep_mul` as a fixed multiplier.
    pub adaptive_temporal_mul: bool,
    /// Chroma subsampling format (default: Yuv444 = no subsampling).
    pub chroma_format: ChromaFormat,
}

impl CodecConfig {
    /// Returns true when the configuration implies bit-exact lossless coding.
    /// This activates integer-exact color conversion and wavelet lifting.
    pub fn is_lossless(&self) -> bool {
        self.quantization_step <= 1.0
            && self.dead_zone == 0.0
            && self.wavelet_type == WaveletType::LeGall53
            && self.transform_type == TransformType::Wavelet
    }
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
            entropy_coder: EntropyCoder::Rice,
            wavelet_type: WaveletType::LeGall53,
            adaptive_quantization: false,
            aq_strength: 0.0,
            per_subband_entropy: false,
            keyframe_interval: 1,
            gpu_entropy_encode: true,
            context_adaptive: false,
            target_bitrate: None,
            rate_mode: RateMode::VBR,
            use_fused_quantize_histogram: false,
            transform_type: TransformType::Wavelet,
            dct_freq_strength: 7.0,
            intra_prediction: false,
            temporal_transform: TemporalTransform::None,
            temporal_highpass_qstep_mul: 2.0,
            adaptive_temporal_mul: true,
            chroma_format: ChromaFormat::Yuv444,
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
        cfl: bool,
        per_subband: bool,
    }

    let anchors: &[Anchor] = &[
        // CfL disabled at extreme compression: alpha precision hurts more than
        // chroma prediction helps at high qstep
        Anchor {
            q: 1,
            qstep: 64.0,
            dead_zone: 1.0,
            perceptual: true,
            cfl: false,
            per_subband: true,
        },
        Anchor {
            q: 10,
            qstep: 32.0,
            dead_zone: 0.75,
            perceptual: true,
            cfl: false, // CfL alpha precision too coarse at high qstep
            per_subband: true,
        },
        Anchor {
            q: 25,
            qstep: 16.0,
            dead_zone: 0.75,
            perceptual: true,
            cfl: false, // CfL alpha too coarse at qstep=16; hurts gradients
            per_subband: true,
        },
        Anchor {
            q: 50,
            qstep: 8.0,
            dead_zone: 0.75,
            perceptual: true,
            cfl: true,
            per_subband: true,
        },
        Anchor {
            q: 75,
            qstep: 4.0,
            dead_zone: 0.75,
            perceptual: true,
            cfl: true,
            per_subband: true,
        },
        Anchor {
            q: 85,
            qstep: 2.8,
            dead_zone: 0.5,
            perceptual: true,
            cfl: true,
            per_subband: true,
        },
        // CDF 9/7 at qstep >= 2.0 keeps rANS alphabet within GPU limits
        Anchor {
            q: 92,
            qstep: 2.05,
            dead_zone: 0.05,
            perceptual: false,
            cfl: false,
            per_subband: true,
        },
        // Slow quality ramp at safe qstep floor — dead_zone→0 squeezes out last dB
        Anchor {
            q: 99,
            qstep: 2.0,
            dead_zone: 0.0,
            perceptual: false,
            cfl: false,
            per_subband: true,
        },
        // Lossless: LeGall 5/3 with integer-exact lifting
        Anchor {
            q: 100,
            qstep: 1.0,
            dead_zone: 0.0,
            perceptual: false,
            cfl: false,
            per_subband: true,
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

    let wavelet_levels = if q >= 50 { 4 } else { 3 };
    let aq_enabled = q <= 80; // AQ helps in lossy range; variance computed on LL subband
    let aq_strength = if q >= 70 { 0.2 } else { 0.15 };
    let mut weights = if disc.perceptual {
        SubbandWeights::perceptual(wavelet_levels)
    } else {
        SubbandWeights::uniform(wavelet_levels)
    };
    // Chroma weighting: HVS is less sensitive to chroma detail.
    // More aggressive at low quality where bitrate savings matter most.
    if q < 40 {
        weights.chroma_weight = 1.5;
    } else if q < 60 {
        weights.chroma_weight = 1.3;
    } else if q < 85 {
        weights.chroma_weight = 1.2;
    }
    CodecConfig {
        quantization_step: qstep,
        dead_zone,
        wavelet_levels,
        subband_weights: weights,
        // CfL: use anchor's setting (disabled at q>=99 to avoid chroma artifacts near lossless)
        cfl_enabled: disc.cfl,
        // LeGall 5/3 only for lossless (q=100); CDF 9/7 for all lossy
        wavelet_type: if q == 100 {
            WaveletType::LeGall53
        } else {
            WaveletType::CDF97
        },
        // Rice: 256 fully independent GPU-parallel streams per tile — matches our architecture.
        // rANS is sequential (2048 ops/thread) and conflicts with GPU-parallel design.
        // rANS available via --rans flag; wins at q≤40 but wrong default for this codec.
        entropy_coder: EntropyCoder::Rice,
        per_subband_entropy: disc.per_subband,
        adaptive_quantization: aq_enabled,
        aq_strength,
        context_adaptive: false, // CPU-only; enable explicitly when GPU implementation exists
        use_fused_quantize_histogram: true, // auto-disabled when CfL is active
        transform_type: TransformType::Wavelet, // block DCT opt-in via config override
        dct_freq_strength: 7.0,
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
    Rice(Vec<encoder::rice::RiceTile>),
    Huffman(Vec<encoder::huffman::HuffmanTile>),
}

impl EntropyData {
    pub fn byte_size(&self) -> usize {
        match self {
            EntropyData::Rans(tiles) => tiles.iter().map(|t| t.byte_size()).sum(),
            EntropyData::SubbandRans(tiles) => tiles.iter().map(|t| t.byte_size()).sum(),
            EntropyData::Bitplane(tiles) => tiles.iter().map(|t| t.byte_size()).sum(),
            EntropyData::Rice(tiles) => tiles.iter().map(|t| t.byte_size()).sum(),
            EntropyData::Huffman(tiles) => tiles.iter().map(|t| t.byte_size()).sum(),
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
    /// Per-LL-block weight map for adaptive quantization.
    /// `None` when adaptive quantization is disabled.
    /// Stored in tile-row-major order, one f32 per LL-subband block.
    pub weight_map: Option<Vec<f32>>,
    /// Frame type for temporal coding (Intra or Predicted)
    pub frame_type: FrameType,
    /// Motion field for P-frames (None for I-frames)
    pub motion_field: Option<MotionField>,
    /// Packed 2-bit intra prediction modes (4 modes per byte, Y plane only).
    /// Present when intra_prediction is enabled.
    pub intra_modes: Option<Vec<u8>>,
    /// Y-plane residual statistics (after MC, before wavelet). Diagnostics only.
    pub residual_stats: Option<ResidualStats>,
    /// Co-plane residual statistics (after MC, before wavelet). Diagnostics only.
    pub residual_stats_co: Option<ResidualStats>,
    /// Cg-plane residual statistics (after MC, before wavelet). Diagnostics only.
    pub residual_stats_cg: Option<ResidualStats>,
}

/// In-memory temporal wavelet encoding result (not yet serializable).
#[derive(Debug, Clone)]
pub struct TemporalEncodedSequence {
    pub mode: TemporalTransform,
    pub groups: Vec<TemporalGroup>,
    /// For each group, which GOP index (0-based) it belongs to.
    /// Normally `group_gop_indices[g] == g`, but when scene cuts cause some GOPs to be
    /// encoded as individual I-frames, those GOP indices are skipped here.
    pub group_gop_indices: Vec<usize>,
    pub tail_iframes: Vec<CompressedFrame>,
    /// Presentation timestamp (frame index) for each entry in `tail_iframes`.
    /// For scene-cut I-frames this may be < last temporal group's PTS.
    pub tail_iframe_pts: Vec<u32>,
    pub frame_count: usize,
    pub gop_size: usize,
}

#[derive(Debug, Clone)]
pub struct TemporalGroup {
    pub low_frame: CompressedFrame,
    /// Highpass frames grouped by level (level 0 = finest / first split).
    pub high_frames: Vec<Vec<CompressedFrame>>,
}

/// Statistics of the spatial-domain residual (current - predicted) for one plane.
/// Only populated when GNC_DIAGNOSTICS is enabled.
#[derive(Debug, Clone, Copy)]
pub struct ResidualStats {
    /// Mean of |residual| across all pixels (in [0,255] scale)
    pub mean_abs: f64,
    /// Standard deviation of residual values
    pub stddev: f64,
    /// Percentage of pixels with |residual| < 1.0
    pub near_zero_pct: f64,
    /// Total pixels measured
    pub pixel_count: usize,
}

impl CompressedFrame {
    /// Total compressed size in bytes (all tiles + CfL alpha + weight map + MV overhead)
    pub fn byte_size(&self) -> usize {
        let tile_bytes = self.entropy.byte_size();
        let cfl_bytes = self
            .cfl_alphas
            .as_ref()
            .map_or(0, |a| a.alphas.len() * std::mem::size_of::<i16>());
        let wm_bytes = self
            .weight_map
            .as_ref()
            .map_or(0, |wm| wm.len() * std::mem::size_of::<f32>());
        let mv_bytes = self.motion_field.as_ref().map_or(0, |mf| {
            let fwd = mf.vectors.len() * 4; // 2 × i16 per block
            let bwd = mf.backward_vectors.as_ref().map_or(0, |v| v.len() * 4);
            let modes = mf.block_modes.as_ref().map_or(0, |m| m.len());
            fwd + bwd + modes
        });
        let intra_bytes = self.intra_modes.as_ref().map_or(0, |m| m.len());
        tile_bytes + cfl_bytes + wm_bytes + mv_bytes + intra_bytes
    }

    /// Bits per pixel
    pub fn bpp(&self) -> f64 {
        (self.byte_size() as f64 * 8.0) / (self.info.width as f64 * self.info.height as f64)
    }
}

/// Compute decode order for a display-ordered sequence with B-frames.
///
/// B-frames require their forward and backward reference frames to be decoded first.
/// This function returns indices into the display-order slice such that decoding
/// in the returned order ensures references are always available.
///
/// For I/P-only sequences (no B-frames), returns identity order `[0, 1, 2, ...]`.
pub fn decode_order(frames: &[CompressedFrame]) -> Vec<usize> {
    let mut order = Vec::with_capacity(frames.len());
    let mut i = 0;
    while i < frames.len() {
        if frames[i].frame_type != FrameType::Bidirectional {
            order.push(i);
            i += 1;
        } else {
            // Collect consecutive B-frames
            let b_start = i;
            while i < frames.len() && frames[i].frame_type == FrameType::Bidirectional {
                i += 1;
            }
            let b_end = i;
            // The next non-B frame (anchor) must be decoded first
            if i < frames.len() {
                order.push(i);
                i += 1;
            }
            // Then decode B-frames (they reference both past and future anchors)
            for b in b_start..b_end {
                order.push(b);
            }
        }
    }
    order
}

/// GPU context shared across encoder/decoder
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for GpuContext {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuContext {
    /// Create a new GPU context (blocking). Not available on WASM — use `new_async()`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Self {
        pollster::block_on(Self::new_async())
    }

    /// Create a new GPU context asynchronously. Required on WASM where blocking is unavailable.
    pub async fn new_async() -> Self {
        Self::try_new_async()
            .await
            .expect("Failed to create GPU context")
    }

    /// Fallible version of `new_async()`. Returns an error string if GPU is unavailable.
    pub async fn try_new_async() -> Result<Self, String> {
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
            .ok_or_else(|| {
                "No suitable GPU adapter found. WebGPU may be unavailable or all adapters \
                 are blocklisted by your browser. Try enabling chrome://flags/#enable-unsafe-webgpu"
                    .to_string()
            })?;

        log::info!("Using GPU: {}", adapter.get_info().name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gpu-codec device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 10,
                        ..wgpu::Limits::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create GPU device: {e}"))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

// ---- WASM/WebGPU entry points ----

#[cfg(target_arch = "wasm32")]
pub mod wasm {
    use wasm_bindgen::prelude::*;

    /// Initialize the codec and decode a .gnc file, returning RGBA pixel data.
    /// Called from JavaScript via wasm-bindgen.
    #[wasm_bindgen]
    pub async fn decode_gnc(data: &[u8]) -> Result<Vec<u8>, JsValue> {
        let ctx = crate::GpuContext::try_new_async()
            .await
            .map_err(|e| JsValue::from_str(&e))?;
        let decoder = crate::decoder::pipeline::DecoderPipeline::new(&ctx);
        let frame = crate::format::deserialize_compressed(data);
        let rgba = decoder.decode_rgba_wasm(&ctx, &frame).await;
        Ok(rgba)
    }

    /// Get the width of a .gnc file without decoding.
    #[wasm_bindgen]
    pub fn gnc_width(data: &[u8]) -> u32 {
        let frame = crate::format::deserialize_compressed(data);
        frame.info.width
    }

    /// Get the height of a .gnc file without decoding.
    #[wasm_bindgen]
    pub fn gnc_height(data: &[u8]) -> u32 {
        let frame = crate::format::deserialize_compressed(data);
        frame.info.height
    }

    // ---- Stateful GNV/GNC Player ----

    /// Stateful player for .gnc (single frame) and .gnv (GNV1 video sequence) files.
    /// Holds GPU context and decoder across frames for zero-reinit playback.
    /// Blit a source texture view to a WebGPU surface using a fullscreen triangle.
    fn blit_to_surface(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &wgpu::Surface<'static>,
        pipeline: &wgpu::RenderPipeline,
        bgl: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        src_view: &wgpu::TextureView,
    ) -> Result<(), JsValue> {
        let frame = surface
            .get_current_texture()
            .map_err(|e| JsValue::from_str(&format!("Surface error: {e}")))?;
        let target_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let mut cmd =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = cmd.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(cmd.finish()));
        frame.present();
        Ok(())
    }

    /// Pre-allocated GPU buffers for one GOP's temporal wavelet decode.
    #[derive(Default)]
    struct TwBufSet {
        frame_bufs: Vec<[wgpu::Buffer; 3]>,
        snapshot_bufs: Vec<wgpu::Buffer>,
    }

    #[wasm_bindgen]
    pub struct GnvPlayer {
        ctx: crate::GpuContext,
        decoder: crate::decoder::pipeline::DecoderPipeline,
        data: Vec<u8>,
        header: Option<crate::format::SequenceHeader>,
        temporal_header: Option<crate::format::TemporalSequenceHeader>,
        width: u32,
        height: u32,
        frame_count: u32,
        fps: f64,
        current_frame: usize,
        /// Pre-decoded RGBA frames for B-frame group buffering (readback path).
        buffered_frames: std::collections::BTreeMap<usize, Vec<u8>>,
        // Zero-copy canvas rendering (set by set_canvas)
        surface: Option<wgpu::Surface<'static>>,
        blit_pipeline: Option<wgpu::RenderPipeline>,
        blit_bgl: Option<wgpu::BindGroupLayout>,
        blit_sampler: Option<wgpu::Sampler>,
        /// Pre-decoded GPU textures for B-frame group buffering (zero-copy path).
        buffered_textures: std::collections::BTreeMap<usize, wgpu::Texture>,
        /// GNV2: base index of the currently-decoded GOP (usize::MAX = none).
        tw_gop_base: usize,
        tw_gop_info: Option<crate::FrameInfo>,
        tw_gop_config: Option<crate::CodecConfig>,
        /// Timing: last seek phase (keyframe→target or GOP Phase 1) in ms.
        last_seek_ms: f64,
        /// Timing: last decode phase (single-frame decode+present) in ms.
        last_decode_ms: f64,
        /// Double-buffered GOP decode: two sets of frame/snapshot buffers.
        /// Set 0 = active (current GOP), Set 1 = prefetch (next GOP).
        tw_buf_sets: [TwBufSet; 2],
        /// Which buffer set (0 or 1) holds the current GOP.
        tw_active: usize,
        /// GOP base of the prefetched set (usize::MAX = none).
        tw_prefetch_base: usize,
        /// Plane size of cached buffers (to detect when re-allocation is needed).
        tw_cached_plane_size: u64,
        /// GOP size of cached buffers.
        tw_cached_gop_size: usize,
    }

    #[wasm_bindgen]
    impl GnvPlayer {
        /// Create a new player from raw file data (.gnc or .gnv).
        /// Auto-detects format by checking for GNV1 magic bytes.
        pub async fn create(data: Vec<u8>) -> Result<GnvPlayer, JsValue> {
            let ctx = crate::GpuContext::try_new_async()
                .await
                .map_err(|e| JsValue::from_str(&e))?;
            let decoder = crate::decoder::pipeline::DecoderPipeline::new(&ctx);

            let is_gnv2 = data.len() >= 4 && &data[0..4] == b"GNV2";
            let is_gnv = data.len() >= 4 && &data[0..4] == b"GNV1";

            if is_gnv2 {
                let header = crate::format::deserialize_temporal_sequence_header(&data);
                let width = header.width;
                let height = header.height;
                let frame_count = header.frame_count;
                let fps = header.fps();
                Ok(GnvPlayer {
                    ctx,
                    decoder,
                    data,
                    header: None,
                    temporal_header: Some(header),
                    width,
                    height,
                    frame_count,
                    fps,
                    current_frame: 0,
                    buffered_frames: std::collections::BTreeMap::new(),
                    surface: None,
                    blit_pipeline: None,
                    blit_bgl: None,
                    blit_sampler: None,
                    buffered_textures: std::collections::BTreeMap::new(),
                    tw_gop_base: usize::MAX,
                    tw_gop_info: None,
                    tw_gop_config: None,
                    last_seek_ms: 0.0,
                    last_decode_ms: 0.0,
                    tw_buf_sets: [TwBufSet::default(), TwBufSet::default()],
                    tw_active: 0,
                    tw_prefetch_base: usize::MAX,
                    tw_cached_plane_size: 0,
                    tw_cached_gop_size: 0,
                })
            } else if is_gnv {
                let header = crate::format::deserialize_sequence_header(&data);
                let width = header.width;
                let height = header.height;
                let frame_count = header.frame_count;
                let fps = header.fps();
                Ok(GnvPlayer {
                    ctx,
                    decoder,
                    data,
                    header: Some(header),
                    temporal_header: None,
                    width,
                    height,
                    frame_count,
                    fps,
                    current_frame: 0,
                    buffered_frames: std::collections::BTreeMap::new(),
                    surface: None,
                    blit_pipeline: None,
                    blit_bgl: None,
                    blit_sampler: None,
                    buffered_textures: std::collections::BTreeMap::new(),
                    tw_gop_base: usize::MAX,
                    tw_gop_info: None,
                    tw_gop_config: None,
                    last_seek_ms: 0.0,
                    last_decode_ms: 0.0,
                    tw_buf_sets: [TwBufSet::default(), TwBufSet::default()],
                    tw_active: 0,
                    tw_prefetch_base: usize::MAX,
                    tw_cached_plane_size: 0,
                    tw_cached_gop_size: 0,
                })
            } else {
                // Single-frame .gnc
                let frame = crate::format::deserialize_compressed(&data);
                let width = frame.info.width;
                let height = frame.info.height;
                Ok(GnvPlayer {
                    ctx,
                    decoder,
                    data,
                    header: None,
                    temporal_header: None,
                    width,
                    height,
                    frame_count: 1,
                    fps: 1.0,
                    current_frame: 0,
                    buffered_frames: std::collections::BTreeMap::new(),
                    surface: None,
                    blit_pipeline: None,
                    blit_bgl: None,
                    blit_sampler: None,
                    buffered_textures: std::collections::BTreeMap::new(),
                    tw_gop_base: usize::MAX,
                    tw_gop_info: None,
                    tw_gop_config: None,
                    last_seek_ms: 0.0,
                    last_decode_ms: 0.0,
                    tw_buf_sets: [TwBufSet::default(), TwBufSet::default()],
                    tw_active: 0,
                    tw_prefetch_base: usize::MAX,
                    tw_cached_plane_size: 0,
                    tw_cached_gop_size: 0,
                })
            }
        }

        /// Image/video width in pixels.
        #[wasm_bindgen(getter)]
        pub fn width(&self) -> u32 {
            self.width
        }

        /// Image/video height in pixels.
        #[wasm_bindgen(getter)]
        pub fn height(&self) -> u32 {
            self.height
        }

        /// Total number of frames (1 for single .gnc).
        #[wasm_bindgen(getter)]
        pub fn frame_count(&self) -> u32 {
            self.frame_count
        }

        /// Frames per second (from GNV1 header, or 1.0 for single .gnc).
        #[wasm_bindgen(getter)]
        pub fn fps(&self) -> f64 {
            self.fps
        }

        /// Current frame index (0-based).
        #[wasm_bindgen(getter)]
        pub fn current_frame(&self) -> u32 {
            self.current_frame as u32
        }

        /// Time spent in seek phase (keyframe→target or GOP Phase 1) in ms.
        #[wasm_bindgen(getter)]
        pub fn last_seek_ms(&self) -> f64 {
            self.last_seek_ms
        }

        /// Time spent in decode phase (single-frame decode+present) in ms.
        #[wasm_bindgen(getter)]
        pub fn last_decode_ms(&self) -> f64 {
            self.last_decode_ms
        }

        /// Decode the next frame and return RGBA u8 data.
        /// Advances the frame counter. Wraps to frame 0 at end.
        /// Handles B-frame reordering: when encountering a B-frame group,
        /// decodes the anchor P first, then B-frames, buffering results.
        pub async fn decode_next_frame(&mut self) -> Result<Vec<u8>, JsValue> {
            if self.current_frame >= self.frame_count as usize {
                self.current_frame = 0;
                self.buffered_frames.clear();
            }

            // Return buffered frame if available (from a previous B-frame group decode)
            if let Some(rgba) = self.buffered_frames.remove(&self.current_frame) {
                self.current_frame += 1;
                return Ok(rgba);
            }

            // GNV2 temporal wavelet
            if let Some(ref th) = self.temporal_header {
                // Check buffer first
                if let Some(rgba) = self.buffered_frames.remove(&self.current_frame) {
                    self.current_frame += 1;
                    return Ok(rgba);
                }

                let current_pts = self.current_frame as u32;
                let gop_size = th.gop_size as usize;

                // Check for an I-frame at this exact PTS (scene-cut or tail I-frame).
                // Extract offset/size as copies so the borrow ends before mutable ops.
                let iframe_offsets: Option<(usize, usize)> = th.index.iter()
                    .find(|e| e.frame_role == 2 && e.pts == current_pts)
                    .map(|e| (e.offset as usize, e.offset as usize + e.size as usize));

                let gop_idx = self.current_frame / gop_size;
                let max_gop_idx = th.index.iter()
                    .filter(|e| e.frame_role != 2)
                    .map(|e| e.gop_index as usize)
                    .max()
                    .unwrap_or(0);

                if let Some((start, end)) = iframe_offsets {
                    let frame = crate::format::deserialize_compressed(&self.data[start..end]);
                    let rgba = self.decoder.decode_rgba_wasm(&self.ctx, &frame).await;
                    self.current_frame += 1;
                    return Ok(rgba);
                }

                if gop_idx <= max_gop_idx {
                    let group = crate::format::deserialize_temporal_group(
                        &self.data, th, gop_idx,
                    );
                    let frames = self.decoder.decode_temporal_group_rgba_wasm(
                        &self.ctx, &group, th.temporal_transform, gop_size,
                    ).await;

                    let base = gop_idx * gop_size;
                    for (i, rgba) in frames.into_iter().enumerate() {
                        self.buffered_frames.insert(base + i, rgba);
                    }
                    let rgba = self.buffered_frames.remove(&self.current_frame).unwrap();
                    self.current_frame += 1;
                    return Ok(rgba);
                } else {
                    // Past end of all frames — wrap to start.
                    self.current_frame = 0;
                    self.buffered_frames.clear();
                    return Err(JsValue::from_str("End of GNV2 sequence, rewound to frame 0"));
                }
            }

            if self.header.is_none() {
                // Single-frame .gnc
                let frame = crate::format::deserialize_compressed(&self.data);
                let rgba = self.decoder.decode_rgba_wasm(&self.ctx, &frame).await;
                self.current_frame += 1;
                return Ok(rgba);
            }

            let header = self.header.as_ref().unwrap();
            let idx = self.current_frame;
            let ft = header.index[idx].frame_type;

            if ft == 2 {
                // B-frame encountered directly. The current reference planes
                // hold the past anchor. Find the future anchor P, decode it
                // first, then decode the B-frame group.
                let b_start = idx;
                let mut b_end = idx;
                while b_end < self.frame_count as usize
                    && header.index[b_end].frame_type == 2
                {
                    b_end += 1;
                }

                if b_end < self.frame_count as usize {
                    // Save past ref, decode future anchor P, swap for B-frames
                    self.decoder.swap_forward_to_backward_ref(&self.ctx);
                    let anchor_frame = crate::format::deserialize_sequence_frame(
                        &self.data, header, b_end,
                    );
                    let anchor_rgba =
                        self.decoder.decode_rgba_wasm(&self.ctx, &anchor_frame).await;
                    self.decoder.swap_references(); // ref=past, bwd=future

                    // Decode all B-frames in this group
                    for b_idx in b_start..b_end {
                        let b_frame = crate::format::deserialize_sequence_frame(
                            &self.data, header, b_idx,
                        );
                        let b_rgba =
                            self.decoder.decode_rgba_wasm(&self.ctx, &b_frame).await;
                        self.buffered_frames.insert(b_idx, b_rgba);
                    }

                    self.decoder.swap_references(); // ref=future anchor P
                    self.buffered_frames.insert(b_end, anchor_rgba);

                    // Return the first B-frame from buffer
                    let rgba = self.buffered_frames.remove(&idx).unwrap();
                    self.current_frame = idx + 1;
                    return Ok(rgba);
                } else {
                    // No anchor available (end of sequence), fallback
                    let frame = crate::format::deserialize_sequence_frame(
                        &self.data, header, idx,
                    );
                    let rgba = self.decoder.decode_rgba_wasm(&self.ctx, &frame).await;
                    self.current_frame += 1;
                    return Ok(rgba);
                }
            }

            // I/P frame: just decode it. If B-frames follow, they'll be
            // handled when current_frame advances to them.
            let frame = crate::format::deserialize_sequence_frame(
                &self.data, header, idx,
            );
            let rgba = self.decoder.decode_rgba_wasm(&self.ctx, &frame).await;
            self.current_frame += 1;
            Ok(rgba)
        }

        /// Seek to a specific frame number. For GNV1, seeks to the nearest
        /// preceding keyframe and decodes forward to the target frame.
        /// Handles B-frame reordering during the decode-forward pass.
        pub async fn seek(&mut self, target_frame: u32) -> Result<Vec<u8>, JsValue> {
            if self.header.is_none() {
                // Single frame — just decode it
                self.current_frame = 0;
                self.buffered_frames.clear();
                return self.decode_next_frame().await;
            }

            let header = self.header.as_ref().unwrap();
            let target = (target_frame as usize).min(header.frame_count as usize - 1);

            // Find nearest preceding keyframe
            let keyframe_idx = crate::format::seek_to_keyframe(header, target as u64);

            // Clear any buffered frames from previous playback position
            self.buffered_frames.clear();

            // Decode from keyframe to target using B-frame-aware ordering.
            // We reset current_frame and use decode_next_frame which handles
            // B-frame groups automatically.
            self.current_frame = keyframe_idx;
            let mut last_rgba = None;

            while self.current_frame <= target {
                let rgba = self.decode_next_frame().await?;
                // decode_next_frame advances current_frame, so the frame we just
                // decoded is at current_frame - 1
                let decoded_idx = self.current_frame - 1;
                if decoded_idx == target {
                    last_rgba = Some(rgba);
                    break;
                }
                // Check if the target is in the buffered frames (from a B-group)
                if let Some(buffered) = self.buffered_frames.get(&target) {
                    last_rgba = Some(buffered.clone());
                    break;
                }
                // If we overshot past target (can happen with B-frame group buffering)
                if self.current_frame > target + 1 {
                    if let Some(buffered) = self.buffered_frames.remove(&target) {
                        last_rgba = Some(buffered);
                    }
                    break;
                }
            }

            if let Some(rgba) = last_rgba {
                self.current_frame = target + 1;
                // Keep only frames after target in buffer (useful for next decode_next_frame)
                self.buffered_frames.retain(|&k, _| k > target);
                return Ok(rgba);
            }

            // Fallback (shouldn't reach here)
            self.current_frame = target + 1;
            Err(JsValue::from_str("Seek failed"))
        }

        /// Returns true if zero-copy canvas rendering is configured.
        #[wasm_bindgen(getter)]
        pub fn has_surface(&self) -> bool {
            self.surface.is_some()
        }

        /// Configure zero-copy rendering to a canvas element.
        /// After this, use `decode_and_present()` instead of `decode_next_frame()`.
        /// The canvas must NOT have a 2D context — WebGPU and 2D contexts are mutually exclusive.
        pub fn set_canvas(
            &mut self,
            canvas: web_sys::HtmlCanvasElement,
        ) -> Result<(), JsValue> {
            let surface = self
                .ctx
                .instance
                .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
                .map_err(|e| JsValue::from_str(&format!("Failed to create surface: {e}")))?;

            let config = surface
                .get_default_config(&self.ctx.adapter, self.width, self.height)
                .ok_or_else(|| {
                    JsValue::from_str("Surface format not compatible with adapter")
                })?;
            surface.configure(&self.ctx.device, &config);

            let shader =
                self.ctx
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("blit"),
                        source: wgpu::ShaderSource::Wgsl(
                            include_str!("shaders/blit.wgsl").into(),
                        ),
                    });

            let bgl =
                self.ctx
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("blit_bgl"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float {
                                        filterable: true,
                                    },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(
                                    wgpu::SamplerBindingType::Filtering,
                                ),
                                count: None,
                            },
                        ],
                    });

            let pl =
                self.ctx
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("blit_pl"),
                        bind_group_layouts: &[&bgl],
                        push_constant_ranges: &[],
                    });

            let pipeline =
                self.ctx
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("blit_pipeline"),
                        layout: Some(&pl),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("vs_main"),
                            buffers: &[],
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some("fs_main"),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: config.format,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                            compilation_options: Default::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            ..Default::default()
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState::default(),
                        multiview: None,
                        cache: None,
                    });

            let sampler = self.ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("blit_sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            self.surface = Some(surface);
            self.blit_pipeline = Some(pipeline);
            self.blit_bgl = Some(bgl);
            self.blit_sampler = Some(sampler);
            Ok(())
        }

        /// Decode the next frame and present it directly to the canvas (zero-copy).
        /// No data is returned to JS. Advances the frame counter.
        /// Handles B-frame reordering with GPU texture buffering.
        /// Updates `last_seek_ms` and `last_decode_ms` timing fields.
        pub fn decode_and_present(&mut self) -> Result<(), JsValue> {
            if self.current_frame >= self.frame_count as usize {
                self.current_frame = 0;
                self.buffered_textures.clear();
                self.buffered_frames.clear();
                self.tw_gop_base = usize::MAX;
            }

            // Return buffered texture if available (from a previous B-frame group decode)
            if let Some(texture) = self.buffered_textures.remove(&self.current_frame) {
                self.last_seek_ms = 0.0;
                let t_dec = Self::now_ms();
                let view =
                    texture.create_view(&wgpu::TextureViewDescriptor::default());
                blit_to_surface(
                    &self.ctx.device,
                    &self.ctx.queue,
                    self.surface.as_ref().unwrap(),
                    self.blit_pipeline.as_ref().unwrap(),
                    self.blit_bgl.as_ref().unwrap(),
                    self.blit_sampler.as_ref().unwrap(),
                    &view,
                )?;
                self.last_decode_ms = Self::now_ms() - t_dec;
                self.current_frame += 1;
                return Ok(());
            }

            // GNV2 temporal wavelet: two-phase decode.
            // Phase 1 (once per GOP): entropy decode + temporal Haar → wavelet-domain GPU bufs.
            // Phase 2 (per frame): inverse spatial wavelet + color → blit.
            if let Some(ref th) = self.temporal_header {
                let current_pts = self.current_frame as u32;
                let gop_size = th.gop_size as usize;
                let temporal_transform = th.temporal_transform;

                // Check for an I-frame at this exact PTS (scene-cut or tail I-frame).
                // Extract offset/size as copies so the borrow ends before mutable ops.
                let iframe_offsets: Option<(usize, usize)> = th.index.iter()
                    .find(|e| e.frame_role == 2 && e.pts == current_pts)
                    .map(|e| (e.offset as usize, e.offset as usize + e.size as usize));

                let gop_idx = self.current_frame / gop_size;
                let max_gop_idx = th.index.iter()
                    .filter(|e| e.frame_role != 2)
                    .map(|e| e.gop_index as usize)
                    .max()
                    .unwrap_or(0);
                // Pre-extract index entries as owned tuples so the th borrow can end before
                // the mutable self.ensure_tw_bufs() call; used for prefetch computation.
                let index_entries: Vec<(u8, u32, u16)> = th.index.iter()
                    .map(|e| (e.frame_role, e.pts, e.gop_index))
                    .collect();

                if let Some((start, end)) = iframe_offsets {
                    self.last_seek_ms = 0.0;
                    let t_dec = Self::now_ms();
                    let frame = crate::format::deserialize_compressed(&self.data[start..end]);
                    self.decoder.decode_to_texture(&self.ctx, &frame);
                    let view = self.decoder.output_texture_view().unwrap();
                    blit_to_surface(
                        &self.ctx.device,
                        &self.ctx.queue,
                        self.surface.as_ref().unwrap(),
                        self.blit_pipeline.as_ref().unwrap(),
                        self.blit_bgl.as_ref().unwrap(),
                        self.blit_sampler.as_ref().unwrap(),
                        &view,
                    )?;
                    self.last_decode_ms = Self::now_ms() - t_dec;
                    self.current_frame += 1;
                    return Ok(());
                }

                if gop_idx <= max_gop_idx {
                    let frame_in_gop = self.current_frame - gop_idx * gop_size;
                    let gop_base = gop_idx * gop_size;

                    // Phase 1: decode GOP wavelet bufs if not cached
                    if self.tw_gop_base != gop_base {
                        // Check if prefetch already decoded this GOP
                        if self.tw_prefetch_base == gop_base {
                            // Swap: prefetched set becomes active
                            self.tw_active = 1 - self.tw_active;
                            self.tw_gop_base = gop_base;
                            self.tw_prefetch_base = usize::MAX;
                            self.last_seek_ms = 0.0;
                        } else {
                            let t_seek = Self::now_ms();
                            let group = crate::format::deserialize_temporal_group(
                                &self.data, th, gop_idx,
                            );
                            let padded_w = group.low_frame.info.padded_width();
                            let padded_h = group.low_frame.info.padded_height();
                            let gop_info = group.low_frame.info;
                            let gop_config = group.low_frame.config.clone();
                            self.ensure_tw_bufs(padded_w, padded_h, gop_size);
                            self.tw_gop_info = Some(gop_info);
                            self.tw_gop_config = Some(gop_config);
                            let set = &self.tw_buf_sets[self.tw_active];
                            self.decoder.decode_temporal_gop_into(
                                &self.ctx, &group, temporal_transform, gop_size,
                                &set.frame_bufs, &set.snapshot_bufs,
                            );
                            self.tw_gop_base = gop_base;
                            self.tw_prefetch_base = usize::MAX;
                            self.last_seek_ms = Self::now_ms() - t_seek;
                        }
                    } else {
                        self.last_seek_ms = 0.0;
                    }

                    // Phase 2: inverse wavelet + color for this frame → cached texture → blit
                    let t_dec = Self::now_ms();
                    {
                        let bufs = &self.tw_buf_sets[self.tw_active].frame_bufs[frame_in_gop];
                        self.decoder.present_wavelet_frame_to_texture(
                            &self.ctx,
                            self.tw_gop_info.as_ref().unwrap(),
                            self.tw_gop_config.as_ref().unwrap(),
                            [&bufs[0], &bufs[1], &bufs[2]],
                        );
                        let view = self.decoder.output_texture_view().unwrap();
                        blit_to_surface(
                            &self.ctx.device,
                            &self.ctx.queue,
                            self.surface.as_ref().unwrap(),
                            self.blit_pipeline.as_ref().unwrap(),
                            self.blit_bgl.as_ref().unwrap(),
                            self.blit_sampler.as_ref().unwrap(),
                            &view,
                        )?;
                    }
                    self.last_decode_ms = Self::now_ms() - t_dec;

                    // Precompute prefetch target using pre-extracted index entries (no th borrow needed).
                    let prefetch_target: Option<usize> = if frame_in_gop == gop_size / 2
                        && gop_idx < max_gop_idx
                        && self.tw_prefetch_base != gop_base + gop_size
                    {
                        let next_gop_base = (gop_idx + 1) * gop_size;
                        index_entries.iter()
                            .filter(|(role, pts, _)| *role == 0 && *pts >= next_gop_base as u32)
                            .map(|(_, _, gop_index)| *gop_index as usize)
                            .min()

                    } else {
                        None
                    };

                    self.current_frame += 1;
                    if let Some(next_g) = prefetch_target {
                        self.prefetch_next_gop(next_g, gop_size);
                    }

                    return Ok(());
                } else {
                    // Past end — wrap
                    self.current_frame = 0;
                    self.tw_gop_base = usize::MAX;
                    return Ok(());
                }
            }

            if self.header.is_none() {
                // Single-frame .gnc
                self.last_seek_ms = 0.0;
                let t_dec = Self::now_ms();
                let frame = crate::format::deserialize_compressed(&self.data);
                self.decoder.decode_to_texture(&self.ctx, &frame);
                let view = self.decoder.output_texture_view().unwrap();
                blit_to_surface(
                    &self.ctx.device,
                    &self.ctx.queue,
                    self.surface.as_ref().unwrap(),
                    self.blit_pipeline.as_ref().unwrap(),
                    self.blit_bgl.as_ref().unwrap(),
                    self.blit_sampler.as_ref().unwrap(),
                    &view,
                )?;
                self.last_decode_ms = Self::now_ms() - t_dec;
                self.current_frame += 1;
                return Ok(());
            }

            let header = self.header.as_ref().unwrap();
            let idx = self.current_frame;
            let ft = header.index[idx].frame_type;

            if ft == 2 {
                // B-frame group: decode anchor + all B-frames to owned textures
                let t_seek = Self::now_ms();
                let b_start = idx;
                let mut b_end = idx;
                while b_end < self.frame_count as usize
                    && header.index[b_end].frame_type == 2
                {
                    b_end += 1;
                }

                if b_end < self.frame_count as usize {
                    self.decoder.swap_forward_to_backward_ref(&self.ctx);
                    let anchor_frame = crate::format::deserialize_sequence_frame(
                        &self.data, header, b_end,
                    );
                    let (anchor_tex, _) = self.decoder.decode_to_owned_texture(
                        &self.ctx,
                        &anchor_frame,
                    );
                    self.decoder.swap_references();

                    for b_idx in b_start..b_end {
                        let b_frame = crate::format::deserialize_sequence_frame(
                            &self.data, header, b_idx,
                        );
                        let (b_tex, _) = self.decoder.decode_to_owned_texture(
                            &self.ctx, &b_frame,
                        );
                        self.buffered_textures.insert(b_idx, b_tex);
                    }

                    self.decoder.swap_references();
                    self.buffered_textures.insert(b_end, anchor_tex);
                    self.last_seek_ms = Self::now_ms() - t_seek;

                    let t_dec = Self::now_ms();
                    let texture =
                        self.buffered_textures.remove(&idx).unwrap();
                    let view = texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    blit_to_surface(
                        &self.ctx.device,
                        &self.ctx.queue,
                        self.surface.as_ref().unwrap(),
                        self.blit_pipeline.as_ref().unwrap(),
                        self.blit_bgl.as_ref().unwrap(),
                        self.blit_sampler.as_ref().unwrap(),
                        &view,
                    )?;
                    self.last_decode_ms = Self::now_ms() - t_dec;
                    self.current_frame = idx + 1;
                    return Ok(());
                } else {
                    // No anchor available — decode as-is
                    self.last_seek_ms = 0.0;
                    let t_dec = Self::now_ms();
                    let frame = crate::format::deserialize_sequence_frame(
                        &self.data, header, idx,
                    );
                    self.decoder.decode_to_texture(&self.ctx, &frame);
                    let view = self.decoder.output_texture_view().unwrap();
                    blit_to_surface(
                        &self.ctx.device,
                        &self.ctx.queue,
                        self.surface.as_ref().unwrap(),
                        self.blit_pipeline.as_ref().unwrap(),
                        self.blit_bgl.as_ref().unwrap(),
                        self.blit_sampler.as_ref().unwrap(),
                        &view,
                    )?;
                    self.last_decode_ms = Self::now_ms() - t_dec;
                    self.current_frame += 1;
                    return Ok(());
                }
            }

            // I/P frame: decode to cached texture and present
            self.last_seek_ms = 0.0;
            let t_dec = Self::now_ms();
            let frame =
                crate::format::deserialize_sequence_frame(&self.data, header, idx);
            self.decoder.decode_to_texture(&self.ctx, &frame);
            let view = self.decoder.output_texture_view().unwrap();
            blit_to_surface(
                &self.ctx.device,
                &self.ctx.queue,
                self.surface.as_ref().unwrap(),
                self.blit_pipeline.as_ref().unwrap(),
                self.blit_bgl.as_ref().unwrap(),
                self.blit_sampler.as_ref().unwrap(),
                &view,
            )?;
            self.last_decode_ms = Self::now_ms() - t_dec;
            self.current_frame += 1;
            Ok(())
        }

        /// Seek to a target frame and present it (zero-copy).
        /// Decodes from nearest keyframe to target without presenting intermediate frames.
        /// Updates `last_seek_ms` and `last_decode_ms` timing fields.
        pub fn seek_and_present(&mut self, target_frame: u32) -> Result<(), JsValue> {
            // GNV2: decode GOP wavelet bufs, present target frame
            if let Some(ref th) = self.temporal_header {
                let target = (target_frame as usize).min(self.frame_count as usize - 1);
                let gop_size = th.gop_size as usize;
                let temporal_transform = th.temporal_transform;
                let gop_idx = target / gop_size;

                // Pre-extract index info to avoid borrow checker conflicts below.
                // (offset, size) for scene-cut I-frame at this PTS, if any.
                let iframe_offsets: Option<(usize, usize)> = th
                    .index
                    .iter()
                    .find(|e| e.frame_role == 2 && e.pts == target as u32)
                    .map(|e| (e.offset as usize, e.offset as usize + e.size as usize));
                // Highest GOP index present in the index (among temporal groups).
                let max_gop_idx: Option<usize> = th
                    .index
                    .iter()
                    .filter(|e| e.frame_role != 2)
                    .map(|e| e.gop_index as usize)
                    .max();

                self.buffered_textures.clear();
                self.buffered_frames.clear();

                if let Some((start, end)) = iframe_offsets {
                    // Scene-cut I-frame: decode inline to avoid async reentrancy.
                    let t_dec = Self::now_ms();
                    let frame = crate::format::deserialize_compressed(&self.data[start..end]);
                    self.decoder.decode_to_texture(&self.ctx, &frame);
                    let view = self.decoder.output_texture_view().unwrap();
                    blit_to_surface(
                        &self.ctx.device,
                        &self.ctx.queue,
                        self.surface.as_ref().unwrap(),
                        self.blit_pipeline.as_ref().unwrap(),
                        self.blit_bgl.as_ref().unwrap(),
                        self.blit_sampler.as_ref().unwrap(),
                        &view,
                    )?;
                    self.last_seek_ms = 0.0;
                    self.last_decode_ms = Self::now_ms() - t_dec;
                    self.current_frame = target + 1;
                    return Ok(());
                }

                // Check whether this GOP index exists in the temporal groups.
                // Use max_gop_idx rather than num_groups to handle gaps from scene cuts:
                // some gop_idx values may be missing because their frames were stored as
                // scene-cut I-frames rather than temporal GOPs.
                let in_temporal = max_gop_idx.is_some_and(|m| gop_idx <= m);
                if in_temporal {
                    let gop_base = gop_idx * gop_size;
                    // Phase 1: decode GOP wavelet bufs
                    let t_seek = Self::now_ms();
                    let th = self.temporal_header.as_ref().unwrap();
                    let group = crate::format::deserialize_temporal_group(
                        &self.data, th, gop_idx,
                    );
                    self.tw_gop_info = Some(group.low_frame.info);
                    self.tw_gop_config = Some(group.low_frame.config.clone());
                    let padded_w = group.low_frame.info.padded_width();
                    let padded_h = group.low_frame.info.padded_height();
                    self.ensure_tw_bufs(padded_w, padded_h, gop_size);
                    let set = &self.tw_buf_sets[self.tw_active];
                    self.decoder.decode_temporal_gop_into(
                        &self.ctx, &group, temporal_transform, gop_size,
                        &set.frame_bufs, &set.snapshot_bufs,
                    );
                    self.tw_gop_base = gop_base;
                    self.tw_prefetch_base = usize::MAX;
                    self.last_seek_ms = Self::now_ms() - t_seek;

                    // Phase 2: present target frame
                    let t_dec = Self::now_ms();
                    let frame_in_gop = target - gop_idx * gop_size;
                    let bufs = &self.tw_buf_sets[self.tw_active].frame_bufs[frame_in_gop];
                    self.decoder.present_wavelet_frame_to_texture(
                        &self.ctx,
                        self.tw_gop_info.as_ref().unwrap(),
                        self.tw_gop_config.as_ref().unwrap(),
                        [&bufs[0], &bufs[1], &bufs[2]],
                    );
                    let view = self.decoder.output_texture_view().unwrap();
                    blit_to_surface(
                        &self.ctx.device,
                        &self.ctx.queue,
                        self.surface.as_ref().unwrap(),
                        self.blit_pipeline.as_ref().unwrap(),
                        self.blit_bgl.as_ref().unwrap(),
                        self.blit_sampler.as_ref().unwrap(),
                        &view,
                    )?;
                    self.last_decode_ms = Self::now_ms() - t_dec;
                } else {
                    // Tail I-frame (end-of-file, not scene cut): decode inline.
                    let tail_entries: Vec<_> = {
                        let th = self.temporal_header.as_ref().unwrap();
                        th.index
                            .iter()
                            .filter(|e| e.frame_role == 2)
                            .map(|e| (e.offset as usize, e.offset as usize + e.size as usize))
                            .collect()
                    };
                    let num_groups = self.temporal_header.as_ref().unwrap().num_groups();
                    let tail_idx = target.saturating_sub(num_groups * gop_size);
                    let t_dec = Self::now_ms();
                    if let Some((start, end)) = tail_entries.get(tail_idx).copied() {
                        let frame = crate::format::deserialize_compressed(&self.data[start..end]);
                        self.decoder.decode_to_texture(&self.ctx, &frame);
                        let view = self.decoder.output_texture_view().unwrap();
                        blit_to_surface(
                            &self.ctx.device,
                            &self.ctx.queue,
                            self.surface.as_ref().unwrap(),
                            self.blit_pipeline.as_ref().unwrap(),
                            self.blit_bgl.as_ref().unwrap(),
                            self.blit_sampler.as_ref().unwrap(),
                            &view,
                        )?;
                    }
                    self.last_seek_ms = 0.0;
                    self.last_decode_ms = Self::now_ms() - t_dec;
                }
                self.current_frame = target + 1;
                return Ok(());
            }

            if self.header.is_none() {
                self.current_frame = 0;
                self.buffered_textures.clear();
                return self.decode_and_present();
            }

            let target =
                (target_frame as usize).min(self.frame_count as usize - 1);
            let keyframe_idx = {
                let header = self.header.as_ref().unwrap();
                crate::format::seek_to_keyframe(header, target as u64)
            };

            self.buffered_frames.clear();
            self.buffered_textures.clear();
            self.current_frame = keyframe_idx;

            // Decode forward from keyframe, discarding intermediate frames
            let t_seek = Self::now_ms();
            while self.current_frame < target
                && !self.buffered_textures.contains_key(&target)
            {
                self.gpu_advance_one();
            }
            self.last_seek_ms = Self::now_ms() - t_seek;

            // Present the target
            let t_dec = Self::now_ms();
            if let Some(texture) = self.buffered_textures.remove(&target) {
                let view =
                    texture.create_view(&wgpu::TextureViewDescriptor::default());
                blit_to_surface(
                    &self.ctx.device,
                    &self.ctx.queue,
                    self.surface.as_ref().unwrap(),
                    self.blit_pipeline.as_ref().unwrap(),
                    self.blit_bgl.as_ref().unwrap(),
                    self.blit_sampler.as_ref().unwrap(),
                    &view,
                )?;
            } else {
                self.current_frame = target;
                self.decode_and_present()?;
            }
            self.last_decode_ms = Self::now_ms() - t_dec;

            self.current_frame = target + 1;
            self.buffered_textures.retain(|&k, _| k > target);
            Ok(())
        }

        /// Reset playback to frame 0.
        pub fn reset(&mut self) {
            self.current_frame = 0;
            self.buffered_frames.clear();
            self.buffered_textures.clear();
            self.tw_gop_base = usize::MAX;
        }
    }

    // Private methods (not exported to JS)
    impl GnvPlayer {
        /// Ensure both double-buffered GOP buffer sets are allocated for the given resolution.
        fn ensure_tw_bufs(&mut self, padded_w: u32, padded_h: u32, gop_size: usize) {
            let padded_pixels = (padded_w as u64) * (padded_h as u64);
            let plane_size = padded_pixels * std::mem::size_of::<f32>() as u64;
            if self.tw_cached_gop_size == gop_size && self.tw_cached_plane_size == plane_size {
                return;
            }
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;
            for (set_idx, set) in self.tw_buf_sets.iter_mut().enumerate() {
                set.frame_bufs = (0..gop_size)
                    .map(|j| {
                        std::array::from_fn(|p| {
                            self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some(&format!("tw_s{}_{}_{}", set_idx, j, p)),
                                size: plane_size,
                                usage,
                                mapped_at_creation: false,
                            })
                        })
                    })
                    .collect();
                set.snapshot_bufs = (0..gop_size)
                    .map(|s| {
                        self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some(&format!("tw_s{}_snap_{}", set_idx, s)),
                            size: plane_size,
                            usage,
                            mapped_at_creation: false,
                        })
                    })
                    .collect();
            }
            self.tw_cached_plane_size = plane_size;
            self.tw_cached_gop_size = gop_size;
        }

        /// Prefetch a GOP into the inactive buffer set.
        fn prefetch_next_gop(&mut self, gop_idx: usize, gop_size: usize) {
            let th = self.temporal_header.as_ref().unwrap();
            let group = crate::format::deserialize_temporal_group(
                &self.data, th, gop_idx,
            );
            self.tw_gop_info = Some(group.low_frame.info);
            self.tw_gop_config = Some(group.low_frame.config.clone());
            let prefetch_set = 1 - self.tw_active;
            let set = &self.tw_buf_sets[prefetch_set];
            self.decoder.decode_temporal_gop_into(
                &self.ctx, &group, th.temporal_transform, gop_size,
                &set.frame_bufs, &set.snapshot_bufs,
            );
            self.tw_prefetch_base = gop_idx * gop_size;
        }

        /// High-resolution timestamp in milliseconds (sub-ms precision via performance.now()).
        fn now_ms() -> f64 {
            web_sys::window()
                .and_then(|w| w.performance())
                .map_or_else(js_sys::Date::now, |p| p.now())
        }

        /// Decode one frame to GPU texture without presenting. Handles B-frame groups.
        /// Used by seek_and_present to skip intermediate frames.
        fn gpu_advance_one(&mut self) {
            // Return buffered texture if available (drop it — we don't present)
            if self.buffered_textures.remove(&self.current_frame).is_some() {
                self.current_frame += 1;
                return;
            }

            let header = self.header.as_ref().unwrap();
            let idx = self.current_frame;
            let ft = header.index[idx].frame_type;

            if ft == 2 {
                let b_start = idx;
                let mut b_end = idx;
                while b_end < self.frame_count as usize
                    && header.index[b_end].frame_type == 2
                {
                    b_end += 1;
                }

                if b_end < self.frame_count as usize {
                    self.decoder.swap_forward_to_backward_ref(&self.ctx);
                    let anchor = crate::format::deserialize_sequence_frame(
                        &self.data, header, b_end,
                    );
                    let (anchor_tex, _) =
                        self.decoder.decode_to_owned_texture(&self.ctx, &anchor);
                    self.decoder.swap_references();

                    for b_idx in b_start..b_end {
                        let b_frame = crate::format::deserialize_sequence_frame(
                            &self.data, header, b_idx,
                        );
                        let (b_tex, _) = self
                            .decoder
                            .decode_to_owned_texture(&self.ctx, &b_frame);
                        self.buffered_textures.insert(b_idx, b_tex);
                    }

                    self.decoder.swap_references();
                    self.buffered_textures.insert(b_end, anchor_tex);

                    // Drop current frame's texture (not presenting)
                    self.buffered_textures.remove(&idx);
                    self.current_frame = idx + 1;
                } else {
                    let frame = crate::format::deserialize_sequence_frame(
                        &self.data, header, idx,
                    );
                    self.decoder.decode_to_texture(&self.ctx, &frame);
                    self.current_frame += 1;
                }
            } else {
                let frame = crate::format::deserialize_sequence_frame(
                    &self.data, header, idx,
                );
                self.decoder.decode_to_texture(&self.ctx, &frame);
                self.current_frame += 1;
            }
        }
    }
}
