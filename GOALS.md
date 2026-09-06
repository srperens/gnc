# GNC — Goals, Rules & Priorities

## 1. What GNC Is

GNC is a patent-free **video codec** designed from scratch for GPU parallelism. Everything runs as wgpu compute shaders (WGSL) — cross-platform on Metal, Vulkan, DX12, and WebGPU/WASM. The core idea: tile-independent processing with thousands of parallel threads instead of sequential CPU-era algorithms.

### GNC is a contribution codec, not a distribution codec

This is the decision that sets every target below, so it comes first. The full reasoning, the
market requirements it rests on, and the measurements behind it are in
**[docs/POSITIONING.md](docs/POSITIONING.md)** — read that before changing any target here.

A distribution codec (H.264, HEVC, AV1) is encoded once and decoded a billion times. Spending
enormous encoder effort to shave a percent off the bitrate is rational there, because the bitrate
is paid a billion times over. **That is not what GNC is for.** GNC encodes and decodes roughly as
often as each other: contribution links, mezzanine storage, low-latency preview, browser playback.

So GNC does **not** try to beat H.264 or AV1 on compression. The target is to be *about as good as
H.264* while winning on the axes that matter for contribution:

| | fixed-function (NVENC/QSV/VideoToolbox) | GNC |
|---|---|---|
| where it runs | one vendor, specific silicon generations | any GPU with WebGPU/Vulkan/Metal/DX12 |
| concurrent streams | a fixed number of encoder blocks per chip, plus driver session limits | limited by general compute, so it scales with the card |
| 10-bit 4:2:2 | only on recent hardware (NVENC: Blackwell and later) | a design target from the start |
| patents | licensed formats | patent-free |

The structural argument is that the number of hardware encoder blocks in a chip is roughly
constant no matter how large and expensive the GPU is, while shader throughput scales with the
card. A bigger GPU should therefore buy more GNC instances; it does not buy more NVENC blocks.
**That claim is currently unproven and is the single most important thing to measure.**

Consequences that follow from this positioning:

- The interesting operating point is **contribution quality** (high bitrate, visually lossless to
  near-lossless), not streaming bitrates. Historical BD-rate numbers measured at distribution
  bitrates describe an operating point GNC is not built for.
- The headline throughput metric is **concurrent streams per GPU** and **latency per frame**, not
  fps on a single stream.
- Low latency, frame-accurate seeking and per-tile error resilience are features, not overhead —
  they are what a contribution link needs, and tile independence is what buys them.

**GNC is still both intra and inter.** Most established contribution formats are all-intra, and
going all-intra would be the easy answer here — it is explicitly rejected. Strong intra *and*
strong inter is the goal. Inter matters at contribution quality too: a static studio shot should
cost almost nothing, and today it does not.

## 2. Design Rules

1. **Patent-free** — No patented techniques, period. If it's patented, we don't use it.
2. **GPU-first** — Everything runs in compute shaders. No CPU fallback paths. CPU reference implementations only for validation/testing.
3. **Massive parallelism via tile independence** — No cross-tile dependencies at any stage. Each tile encodes/decodes in isolation. This is what enables thousands of parallel GPU threads.
4. **Cross-platform** — Must work on Metal, Vulkan, DX12, and WebGPU (WASM). No backend-specific features. WGSL shaders are the single source.
5. **No f64 in shaders** — Target hardware (M1, mobile GPUs) has no hardware double precision.
6. **Open source only** — All dependencies must be open source.
7. **English only** — All code, comments, docs, and commit messages in English.
8. **Measure everything** — Every change benchmarked: PSNR, SSIM, bpp, encode/decode FPS. Compare against baseline and previous best. Optionally compare against relevant codecs (H.264, H.265, AV1, MJPEG, JPEG XS, ProRes) for context.
9. **No code duplication** — Extract shared logic. Code must pass `cargo fmt` and `cargo clippy` with zero warnings.
10. **No legacy** — Nobody runs GNC in production. We can break the bitstream format, change the container, rename fields, restructure anything. No backward compatibility constraints.
11. **Video codec first** — GNC is a video codec, not an image codec. Sequence encode/decode performance is the primary metric. Single-frame performance only matters as a component of video throughput.

## 3. Current State

**Entropy coder: Rice+ZRL** (256 independent streams/tile, fully GPU-parallel, patent-free).
rANS and Huffman exist in the codebase but are parked — Rice is the default for all paths.

| Quality | PSNR | BPP | Encode | Decode |
|---------|------|-----|--------|--------|
| q=25 | 33.2 dB | 1.71 | 39 fps | 72 fps |
| q=50 | 37.7 dB | 2.37 | 40 fps | 60 fps |
| q=75 | 42.8 dB | 4.01 | 40 fps | 59 fps |
| q=90 | 50.5 dB | 8.90 | 40 fps | 63 fps |

*(Single-frame, 1080p bbb reference, M1 GPU, 2026-02-27)*

**Sequence encode: see [BASELINE.md](BASELINE.md) — three different quantities have been called
"encode fps" and they differ by 2.4x.** The previously quoted 31.7 fps is not reproducible and its
stated parameters are internally inconsistent (ki=8 cannot produce B-frames). Measured 2026-09-06
on a non-idle machine: GPU encode phase 12.2 fps, end to end 5.0 fps.

**What works:**
- Full I/P/B frame video pipeline with motion estimation, rate control, GNV1 container
- Rice+ZRL entropy (GPU encode + decode), rANS and Huffman available but parked
- Fused quantize+histogram shader
- 128+ tests, golden-baseline regression, 5 conformance bitstreams
- 33 WGSL compute shaders
- WASM/WebGPU decoder builds (263 KB)

**Key GPU architecture insight:** On M1, shared memory occupancy dominates performance. 16KB shared memory = 2 workgroups/core (full occupancy). Rice uses < 1KB shared → excellent occupancy.

**Known gaps:**
- Sequence encode 31.7 fps → target 60 fps
- Single-frame encode 40 fps → target 60 fps
- 8-bit only (10-bit not implemented) — the main format gap for broadcast contribution
- 4:4:4 / 4:2:2 / 4:2:0 all implemented (`--chroma-format`)
- ~~No true lossless with Rice~~ — **wrong, corrected 2026-09-06.** `q=100` is bit-exact lossless
  on every entropy coder (Rice, rANS, default), verified on two 1080p images: max error 0, zero
  wrong pixels. At 1.99:1 it beats JPEG 2000 lossless by 10.8% and PNG by 7.8%, and loses to FFV1
  by 27% and x264 `-qp 0` by 43% — both of which use spatial prediction

## 4. Where We Stand & Goals

**Status (2026-09-05): intra is competitive; the inter gap is now located.**

The 2026-03 experiment sweep (~40 gated experiments, archived in
[docs/archive/BACKLOG_CLOSED.md](docs/archive/BACKLOG_CLOSED.md)) exhausted the cheap and
medium-cost incremental inter ideas. What it measured: the spatial layer is already strong —
BD-rate +13.9% vs H.264 all-I, and *better* than H.264 all-I above ~36 dB — while the current
I/P/B inter path saves only 17–27% vs all-I where H.264 saves 60–70%.

**MEAS-1 (2026-09-05) measured the gap properly for the first time — at the wrong operating
point. QUAL-1 (2026-09-06) re-measured it at the right one.** MEAS-1 found GNC needing **5-7x**
the bitrate of H.264 (BD-rate +457% / +494% / +672%) at *distribution* bitrates, with the quality
ladder above q=92 dead. Re-run at contribution quality with that ladder working, the same harness
and the same parameters gives **+90.5% BD-rate on PSNR — about 1.9x** (+129.0% bbb, +71.9%
old_town, +70.6% crowd_run). Nothing in the coder changed between the two; the 5-7x figure was
measured somewhere GNC is not built to operate. **Use +90.5%, and do not quote a VMAF BD-rate at
this end** — widening the quality ladder moved the VMAF figure by 47.5 points on average and the
PSNR figure by 1.0. The +13.9% still-image figure is PSNR on stills, a third quantity again.
See [RESEARCH_LOG.md](RESEARCH_LOG.md), 2026-09-06.

**And luma alone misleads here.** At rate matched to 1%, GNC beats x264 on CIEDE2000 (0.611 vs
0.684 on bbb, 0.911 vs 0.949 on old_town, better 95th percentile on all three) while losing
7.4-8.8 dB of luma PSNR. The two codecs allocate rate differently between luma and chroma, so a
single luma number overstates the gap for a colour-critical use case and understates the luma
deficit. Quote both.

**MEAS-4 (2026-09-05) located that gap.** It is *prediction quality*, not the coding model.
Simulating both models on GNC's own motion-compensated residuals at matched distortion, an
idealised per-block DCT with oracle block skip beats GNC's wavelet by only 4–23% at broadcast
quality and *loses* by 3–18% at low bitrate; context-adaptive entropy coding is worth ≤3.4%; and
only 0–2% of blocks are skippable at q=75, meaning the prediction leaves error nearly everywhere.
An x264 ablation on the same content agrees from the other side: H.264's largest inter lever is
multi-reference and B-frame prediction (+29–32%), three times CABAC and thirty times sub-block
partitioning. Full writeup:
[docs/decisions/0005-meas4-inter-gap-decomposition.md](docs/decisions/0005-meas4-inter-gap-decomposition.md).

So meaningful temporal compression stays a goal, and the form is now much clearer than "rebuild
the inter pipeline": GNC uses **single-reference P-frames**, and the lever the measurement says
matters most is the one it does not have. Multi-reference prediction is ordinary and
GPU-parallel. That is where the inter work goes next ([BACKLOG.md](BACKLOG.md) #25); per-block
inter transforms, block skip and context entropy are ruled out by measurement.

GNC's distinguishing properties hold regardless — **patent-free + GPU-native + tile-independent +
low-latency + WebGPU/WASM browser decode** (JPEG XS is patented, VC-2 is CPU-era, JPEG 2000 is
slow) — and they serve broadcast contribution, mezzanine storage, low-latency preview and
browser playback.

GNC should become a **good, robust codec** — not optimized along a single axis. We iterate across multiple dimensions simultaneously, looking for combinations of techniques that work well together. No single property is a hard blocker for the others.

**Target properties (all of these, no strict order):**

| Property | Current | Target |
|----------|---------|--------|
| **Concurrent streams per GPU** | **never measured** | beat NVENC's session/block ceiling on the same machine |
| **Latency per frame** | never measured | sub-frame, end to end |
| Encode speed | 31.7 fps (seq, 1080p q=75) | 60 fps |
| Bit depth | 8-bit | 10-bit, in the format from the start |
| Chroma formats | 4:4:4, 4:2:2, 4:2:0 | keep all three working at 10-bit |
| Compression (intra) | +46–55% vs H.264 all-I on video (VMAF); +13.9% on stills (PSNR) | ≤ H.264 all-I, measured at contribution quality |
| Compression (video) | **+90.5% BD-rate on PSNR vs H.264 at contribution quality** (QUAL-1, 2026-09-06; +457% to +672% was distribution bitrates and is superseded) | ≤ +25%, and the remaining gap is intra |
| Colour accuracy | **ahead of x264 on dE00 at matched rate** (0.611 vs 0.684 mean) while 7.4–8.8 dB behind on luma | keep the colour lead, close the luma gap |
| Quality range | q=1–100 functional | smooth, predictable quality curve |
| Robustness | basic test coverage | no artifacts, stable across q and content |
| Bitstream | GNV1/GNV2 defined | well-specified, documented |

The two metrics at the top of that table have never been measured, and they are the ones the
whole positioning rests on. They come before further compression work.

**On the compression numbers:** both objections to the +457% to +672% figures have now been
settled rather than merely noted. B-frames were defective (BUG-5) and are off by default; the
operating point was wrong, and QUAL-1 re-measured it at the contribution end, where the gap is
**+90.5%**. Treat +457% to +672% as a historical distribution-bitrate figure only. The remaining
gap is **intra** — inter breaks even for both codecs at this quality, and does so for x264 too.

VC-2 (Dirac) demonstrates that a patent-free wavelet codec can do real temporal work (MCTF) and
reach H.264-class compression. That remains the reference point for where the inter path could go.

**How we iterate:**

- Pick the next backlog item based on what provides the most overall value right now — not what happens to be listed as P1
- Rotate between compression, speed, and robustness — progress in one area does not unlock another
- Always measure on ≥3 sequences and multiple q levels — a codec that is only good over a narrow quality band is not a good codec
- Technology choices are driven by: patent freedom, GPU parallelism, measurable improvement

## 5. Design Philosophy

**Correctness over speed.** A codec with subtle bugs is worthless. Verify every change end-to-end. A fast encoder that produces subtly wrong output is not a working encoder.

**Measure before assuming.** Numbers that look too good probably are. Numbers that look unchanged might mean the code isn't running. Run twice. Test on diverse content. Compare against baseline.

**Simplicity has value.** A complex change for 0.3 dB gain is probably not worth the maintenance cost. When two approaches produce similar results, prefer the simpler one. Clever code that nobody understands will break.

**Low-latency by design.** Tile independence is not just about parallelism — it also enables low-latency decode and random seek without full GOP decode. Preserve this property in every pipeline stage.

**Broad content coverage.** A codec that is only good on one type of content is not a good codec. Always validate on high-motion (crowd_run), low-motion (rush_hour), and mixed (stockholm) sequences. Synthetic tests are for correctness, not quality measurement.

**Challenge your own work.** After implementing something, actively try to prove it is wrong before calling it done. Reproduce results before celebrating. If the same bug resurfaces twice — stop and diagnose the root cause properly.

## 5b. Hard Architecture Rules (agents must not violate these)

These rules encode mistakes that were made and reverted. They are non-negotiable.

**Entropy coder: Rice is the default, always.**
rANS is disqualified as default entropy. rANS is sequential (2048 ops/thread, 32 interleaved streams) — it conflicts with GPU-parallel tile-independent design. Rice has 256 fully independent streams per tile and scales with GPU threads. rANS may be kept in the codebase for experimentation at q≤40, but must never be the default.

**No spatial block prediction on the wavelet path.**
Intra prediction (DC, planar, angular) operates at block scale. Wavelet operates at tile scale. Combining them creates block-boundary discontinuities that the wavelet handles poorly. This was implemented, measured, and found to always hurt quality. It is disabled and must stay disabled on the wavelet path.

**bpp reduction is not proof of correctness.**
If a change reduces bpp but quality (PSNR/VMAF) is not validated, the change is not done. Reduced bpp from throwing away coefficients (zeroing tiles, aggressive dead zone) looks identical to reduced bpp from better coding — until you decode and see ghosting or blocking. Validate quality before committing.

**Temporal highpass coefficients must not be zeroed based on motion energy.**
High-motion tiles need highpass coefficients most — they capture the temporal difference that distinguishes frames. Zeroing highpass for high-energy tiles forces the decoder to use the temporal average (LL only), producing ghosting. This was implemented (TILE_ENERGY_ZERO_THRESH), calibrated across 3 commits, and then removed entirely for +4.22 dB.

**WASM must be tested after every change to lib.rs or the decoder.**
WASM crashes are silent until a user opens a browser. Scene cut handling and WASM borrow checker issues were found by the user, not by tests. After any change touching `src/lib.rs`, run `wasm-pack build --target web` and do a smoke test.

**Read diagnostics directly; don't delegate diagnostic interpretation.**
When a diagnostic output exists (--diagnostics, per-frame PSNR, tile energy logs), read it directly and reason about it. Delegating diagnostic interpretation to a new agent loses context and adds latency. If the output is too large, grep for the key numbers.

## 6. Non-Goals

- **Beating AV1/H.265 on compression ratio** — We occupy a different design point: parallel, low-latency, patent-free. We compete on speed and simplicity, not maximum compression.
- **CPU decode path** — GPU-only by design. No software fallback.
- **Backward compatibility** — No legacy bitstreams to support (rule 10).
- **Neural/ML compression** — Extreme complexity for marginal gains. Not worth it for GPU-native design.
- **Maximum single-thread performance** — We scale with parallelism, not clock speed.
