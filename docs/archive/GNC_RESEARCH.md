# GNC — GPU-Native Codec: Research Notes on Interesting Techniques

> **Historical document** (originally written 2026-02-22). Many recommendations below have since been implemented. Current architecture and results are in README.md and STATUS_ROADMAP.md.

## GNC Architecture (as of 2026-02-27)
- RGB → YCoCg-R → CDF 9/7 / LeGall 5/3 wavelet → Adaptive quantization (CfL, AQ, perceptual weights) → Rice+ZRL / rANS / Bitplane entropy
- I/P/B frames, half-pel motion estimation, CBR/VBR rate control, GNV1 container
- Best result: 42.1 dB @ 4.09 BPP (q=75, Rice+ZRL), 40 fps encode, 61 fps decode

---

## 1. Entropy Coding: Beyond Basic Per-Tile rANS

### 1.1 Recoil: Parallel rANS Decoding (2023, ICPP)
**Key insight:** A single rANS-encoded bitstream can be decoded from *any arbitrary position* if the intermediate states are known. This means you don't need to partition into independent streams (which wastes overhead).
- Current approach (DietGPU-style): partition input → encode each partition independently → more partitions = more overhead
- Recoil: encode one stream, store periodic state checkpoints → decoder can use as many parallel threads as it has, scaling adaptively
- **Relevance to GNC:** Could reduce per-tile rANS overhead significantly. Instead of N independent rANS streams per tile, use one stream with embedded sync points.

### 1.2 MANS: Multi-byte ANS (SC25, Nov 2025)
Brand new from Chinese Academy of Sciences. Addresses ANS's weakness on multi-byte integer data:
- **ADM (Adaptive Data Mapping)** strategy improves compression ratio for multi-byte integers
- Adaptive kernel fusion for both NVIDIA and AMD GPUs
- **Relevance:** Wavelet coefficients after quantization are multi-byte — this could directly improve GNC's entropy stage.

### 1.3 DietGPU (Meta/Facebook Research)
Reference implementation for GPU rANS. MIT licensed.
- 250–410 GB/s throughput on A100 for rANS encode/decode
- Primitive unit: **4 KiB segments** assigned to individual warps
- Byte-oriented rANS (operates on statistics gathered bytewise)
- **Architecture lesson:** Need (total_bytes / 4KiB) ≥ number of concurrent warps to saturate SMs
- **Relevance:** GNC could adopt the warp-per-segment model. But DietGPU is CUDA-only; GNC needs wgpu/WGSL equivalent.

### 1.4 Interleaved rANS (Fabian Giesen)
Giesen's seminal work shows GPU implementations work best when there are enough independent contexts to fill a warp/wavefront:
- 8-way interleaved streams with SSE4.1: ~550 MB/s decode on Sandy Bridge
- The same principle generalizes to GPU: fill SIMD width with independent streams
- **Idea for GNC:** Within each tile, interleave multiple rANS streams (one per subband, or one per coefficient group) to maximize GPU occupancy.

---

## 2. Wavelet Transform: GPU Optimization Strategies

### 2.1 Non-Separable 2D Lifting (OpenCL research)
Standard approach: separate horizontal + vertical passes. Research shows:
- **Merging separable parts into non-separable units halves the number of kernel passes**
- Non-separable schemes outperform separable ones, especially on pixel shaders
- **Idea:** GNC's LeGall 5/3 could be implemented as a fused non-separable kernel, reducing memory bandwidth.

### 2.2 Mixed-Band Memory Layout
Key GPU DWT optimization from multiple papers:
- Rearranging spatial locations of wavelet coefficients at every level **destroys coalesced memory access**
- **Mixed-band layout:** Keep coefficients interleaved in memory so multi-level transforms run in a single fused kernel
- Uses registers for in-place operations, maximizing on-chip bandwidth
- **Direct applicability:** GNC's tile-independent design is perfect for this — each tile can use a fused multi-level kernel with mixed-band layout.

### 2.3 Unified Buffer (Memory Saving)
Research on unifying input/output buffers for lifting DWT on GPU:
- Chunk-based data rearrangement separates data-independent tasks from data-dependent tasks
- Achieves 2× larger problem sizes with up to 3.9× speedup
- Uses circular permutation interpretation of memory addresses
- **Relevance:** For 4K/8K broadcast frames, memory savings matter hugely on GPU.

### 2.4 Shared Memory Lifting
The CUDA DWT papers consistently show:
- **Shared memory for intermediate lifting steps** is the #1 optimization (conserves global memory bandwidth)
- wgpu equivalent: **workgroup shared memory** in WGSL compute shaders
- Load tile block → shared memory → compute all lifting steps → write back to global memory

---

## 3. Quantization: Low-Hanging Fruit for Better RD

### 3.1 Per-Subband Adaptive Quantization
GNC currently uses uniform scalar quantization with one QStep. Big gains available:
- **Different subbands have different perceptual importance** — LL (approximation) needs finer quantization than HH (diagonal detail)
- Classic approach: multiply QStep by per-subband weights derived from HVS (contrast sensitivity function)
- Typical weighting: `QStep_actual = QStep_base × CSF_weight[subband][level]`
- Higher wavelet levels (lower frequency) → finer quantization; HH detail → coarser quantization
- **Estimated gain: 0.5–1.5 dB PSNR at same BPP**, or 10-20% BPP reduction at same PSNR

### 3.2 Dead Zone Optimization
Current GNC has "optional dead zone." Research shows:
- Optimal dead zone for Laplacian-distributed coefficients (which wavelet coefficients are): `dead_zone = 2 × (1 - δ) × QStep`
- Intra coding: δ = 1/3 (dead zone = 1.33 × QStep)
- The dead zone effectively zeros out small coefficients, giving huge entropy reduction at minimal quality loss
- **GPU-friendly:** No branching needed, just a floor operation with offset

### 3.3 Context-Based Backward Adaptive Quantization
More advanced: classify each coefficient based on surrounding *already-quantized* coefficients:
- Estimate local activity (standard deviation of neighborhood)
- Multiple quantizer classes, adapted on-the-fly
- No side information needed (backward adaptive)
- **GPU challenge:** Sequential dependency. But within GNC's tile-independent design, each tile can run its own backward adaptation.

### 3.4 Perceptual Weighting via VMAF/SSIM-Driven Quantization
Modern trend: use perceptual metrics to drive quantization decisions rather than MSE/PSNR:
- Allocate more bits to smooth regions (where errors are visible) rather than textured regions
- Activity masking: texture hides quantization noise
- **For GNC:** Could compute a simple spatial activity map per tile in a quick GPU pass, then adjust per-tile QStep.

---

## 4. Transform Alternatives & Additions

### 4.1 Learned Lifting Transforms (2024, IEEE TIP)
Replacing fixed lifting operators with small neural networks:
- Keep the lifting *structure* but learn the predict/update operators
- **Key finding:** Retaining fixed steps from base wavelet + adding learned steps is optimal
- Achieves **>25% bit-rate savings** vs JPEG 2000 with compact spatial support
- **For GNC:** Could add 1-2 learned lifting steps as optional enhancement. The fixed LeGall 5/3 base ensures GPU parallelism; learned steps add compression.

### 4.2 Chroma from Luma Prediction (Daala Codec)
The Daala codec (precursor to AV1) pioneered:
- **CfL (Chroma from Luma):** Predict chroma coefficients from luma in frequency domain
- Since YCoCg-R decorrelates well, the Co/Cg channels still have correlation with Y
- Encode luma first, then predict chroma → encode only the residual
- **GPU-friendly:** Prediction is a simple linear operation per tile
- **Estimated gain:** 5-15% BPP reduction on chroma channels

### 4.3 PVQ — Perceptual Vector Quantization (Daala)
Daala's PVQ is highly relevant:
- Separates each frequency band into **gain** (energy/magnitude) and **shape** (unit vector direction)
- Gain is coded as a scalar; shape is coded on a hypersphere using Pyramid VQ
- **Explicit energy preservation** — avoids the "washed out" look of scalar quantization
- Activity masking: perceptual noise is shaped to hide in texture
- **GPU-friendliness:** Band-level operations, no cross-tile dependencies
- **Could replace or augment GNC's scalar quantizer** for significant perceptual quality improvement

---

## 5. Bitstream & Format Ideas

### 5.1 JPEG XS-Inspired Rate Control
JPEG XS (the codec GNC aims to rival) uses:
- Line-based processing for low latency
- Target BPP with per-line budget management
- **For GNC:** Could add a rate control pass that adjusts QStep per-tile to hit a target bitrate. Simple feedback loop on GPU.

### 5.2 Random Access Granularity
GNC already has tile independence — this is a huge feature for:
- Region-of-interest decoding (decode only visible tiles in VR/AR)
- Partial frame updates
- Consider adding a **tile index table** in the bitstream header for O(1) random access

### 5.3 Progressive/Scalable Coding
Wavelet transforms naturally support this:
- Encode lower wavelet levels first → base quality layer
- Higher levels → enhancement layers
- **SNR scalability:** Different quantization per layer
- **Spatial scalability:** Lower levels = lower resolution

---

## 6. Cross-Platform GPU Considerations

### 6.1 wgpu/WGSL Constraints
GNC targets wgpu (Metal, Vulkan, DX12, WebGPU). Key limitations vs CUDA:
- **No warp-level primitives** (shuffle, ballot) — need workgroup-level equivalents
- **Workgroup shared memory** exists but syntax differs
- **No atomic float operations** on all backends
- **Subgroup operations** (WGSL extension) can substitute for warp primitives on supporting hardware
- **Key advantage:** True cross-platform including WebGPU/WASM — no other codec has this

### 6.2 Compute Shader Occupancy
For maximum throughput:
- Tile size should be tuned to workgroup size (typically 64 or 256 threads)
- Each workgroup processes one tile
- Minimize register pressure to maximize concurrent workgroups
- **Profile with different tile sizes** (32×32, 64×64, 128×128) — this dramatically affects both compression ratio and GPU utilization

---

## 7. Prioritized Recommendations

### Quick Wins (high impact, low effort)
1. ~~**Per-subband quantization weights**~~ — **DONE** (perceptual SubbandWeights)
2. ~~**Optimize dead zone**~~ — **DONE** (configurable dead zone in quality_preset)
3. ~~**Tile index table**~~ — **DONE** (GP11 tile index with CRC-32)
4. ~~**Interleaved rANS**~~ — **DONE** (32 streams per tile; also Rice with 256 streams)

### Medium-Term (high impact, moderate effort)
5. **Non-separable fused wavelet kernel** — halve memory passes (not yet done)
6. ~~**Chroma-from-Luma prediction**~~ — **DONE** (14-bit i16 precision, q=50-85)
7. ~~**Rate control**~~ — **DONE** (CBR/VBR with R-Q model)
8. ~~**Shared memory lifting**~~ — **DONE** (workgroup shared memory in wavelet shaders)

### Research/Experimental
9. **PVQ (Perceptual Vector Quantization)** — Daala-style gain/shape separation (not yet done)
10. **Learned lifting operators** — small NN additions to base wavelet (not yet done)
11. **Recoil-style adaptive parallel rANS** — superseded by Rice+ZRL (no state chain)
12. **MANS-style multi-byte ANS** — superseded by Rice+ZRL

---

## Key References
- **Recoil:** Lin et al., "Parallel rANS Decoding with Decoder-Adaptive Scalability" (ICPP 2023)
- **MANS:** Huang et al., "Efficient and Portable ANS Encoding for Multi-Byte Integer Data" (SC25)
- **DietGPU:** github.com/facebookresearch/dietgpu (MIT license)
- **Giesen rANS:** github.com/rygorous/ryg_rans + arxiv:1402.3392
- **Learned Lifting:** Li et al., "Exploration of Learned Lifting-Based Transform Structures" (IEEE TIP 2024)
- **Daala PVQ:** Valin et al., "Daala: A Perceptually-Driven Next Generation Video Codec" (arxiv:1603.03129)
- **GPU DWT:** van der Laan et al., "Accelerating Wavelet Lifting on Graphics Hardware Using CUDA"
- **Mixed-band DWT:** "A fast mixed-band lifting wavelet transform on the GPU" (ResearchGate)
- **hipANS:** github.com/PAA-NCIC/hipANS (HIP port of DietGPU for AMD)
