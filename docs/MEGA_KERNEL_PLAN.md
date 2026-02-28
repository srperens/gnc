# Mega-Kernel Pipeline Plan

## Problem

Current encode pipeline: **71-83 GPU dispatches per sequence frame**, ~25ms at 1080p.
M1 dispatch overhead ~0.1-0.2ms each → **7-14ms spent just launching shaders**.
Multiple `device.poll(Wait)` CPU-GPU syncs add another 3-5ms.

Target: **2-5ms per frame**, scaling linearly to 4K/8K.

## Strategy

**Phase 1**: Build and benchmark candidate algorithms per pipeline stage (standalone shaders).
**Phase 2**: Fuse winning candidates into mega-kernels, minimize dispatch count.
**Phase 3**: Single command encoder per frame, eliminate mid-frame CPU-GPU syncs.
**Phase 4**: Double-buffer frame pipeline for latency hiding.

## Phase 1: Candidate Shootout

### 1a. Transform Candidates (replaces 24-dispatch wavelet)

All candidates: **1 dispatch, 1 block per workgroup, zero global dependencies**.

| Candidate | Block size | Shared mem | Notes |
|-----------|-----------|------------|-------|
| **8×8 DCT** | 8×8 | ~256B | Industry standard (JPEG, H.264 intra) |
| **16×16 DCT** | 16×16 | ~1KB | Better energy compaction, more compute |
| **4×4 Hadamard** | 4×4 | ~64B | No multiplications, extremely fast |
| **Block-local Haar** | 16×16 (2 levels) | ~1KB | Wavelet-like, no cross-block deps |

Test setup:
- Same input: 1080p `bbb_1080p.png`
- Same quantization: uniform scalar (qstep derived from q=25,50,75,90)
- Same entropy: Rice (existing GPU path)
- Measure: dispatch time, total encode time, PSNR vs bpp (RD curve)

### 1b. Entropy Candidates (for later fusion)

| Candidate | Parallelism | Fusion potential | Compression |
|-----------|-------------|-----------------|-------------|
| **Rice (current)** | 256 streams/tile | Fuseable per-block | Baseline |
| **Exp-Golomb** | Per-coefficient | Trivially fuseable | Slightly worse |
| **Fixed-length + bitmask** | Per-block | Trivially fuseable | Worst, fastest |
| **Truncated unary + suffix** | Per-block | Fuseable | Simple, decent |

### 1c. Motion Estimation Candidates (sequence frames)

| Candidate | Dispatches | Quality |
|-----------|-----------|---------|
| **Diamond search** | 1 | ~95% of full search |
| **Block matching (current)** | 2 (coarse+fine) | Best |
| **Zero MV + skip mode** | 0 | Static content only |

### 1d. Quantization (always fused, no standalone test needed)

- Uniform scalar
- Frequency-weighted matrix (JPEG-style perceptual weights per DCT coefficient)
- Dead-zone + matrix

All fuseable into transform kernel — zero extra dispatches.

## Phase 2: Fused Mega-Kernels

Take Phase 1 winners and build fused pipelines:

**Mega-kernel A: Intra encode** (1 dispatch per plane, or 1 total)
```
load pixels → color convert → transform → quantize → entropy encode → write bitstream
                                        ↓
                              dequantize → inverse transform → write to ref buffer
```

One workgroup = one block. Everything in shared memory. Single dispatch replaces:
- 3 preprocess dispatches
- 24 transform dispatches
- 3 quantize dispatches
- 6 entropy dispatches
- 42 local decode dispatches
- **Total: 78 dispatches → 1-3 dispatches**

**Mega-kernel B: Inter encode** (1 dispatch per plane + 1 ME dispatch)
```
ME dispatch: motion vectors per block
Encode dispatch: MC residual → transform → quantize → entropy → local decode
```

**2-4 dispatches per inter frame.**

## Phase 3: Single Submit Per Frame

Current: multiple `queue.submit()` + `device.poll(Wait)` per frame.
Target: one command encoder → one submit → one poll at frame end.

All inter-dispatch dependencies via GPU barriers only.

## Phase 4: Frame Pipeline Overlap

Double-buffer: GPU runs entropy for frame N while CPU prepares frame N+1.
Requires 2× buffer allocation but halves effective per-frame latency.

## Expected Results

| Resolution | Current (71 dispatches) | Target (4-7 dispatches) |
|-----------|------------------------|------------------------|
| 1080p | ~25ms | ~2-4ms |
| 4K | ~30ms | ~3-5ms |
| 8K | ~50ms | ~5-8ms |

Dispatch overhead drops from dominant cost to negligible.
GPU compute and memory bandwidth become the only limiting factors — exactly what GPUs are good at.

## Execution Order

1. **Transform shootout** — build all 4 candidates, benchmark, pick winner
2. **Fused intra mega-kernel** — preprocess+transform+quant+entropy+local-decode
3. **ME shootout** — diamond vs current block match
4. **Fused inter mega-kernel** — ME + encode + local-decode
5. **Single submit + frame pipeline**
