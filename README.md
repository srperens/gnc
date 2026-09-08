# GNC — GPU-Native Codec

Research project exploring what video compression looks like when designed from scratch for GPU parallelism, rather than adapting CPU-era algorithms.

**Rust + wgpu compute shaders (WGSL). Written against the WebGPU feature set — Metal, Vulkan, DX12, WebGPU/WASM. Patent-free.** Metal is the backend every figure in this file is measured on. **Vulkan runs intra *and* inter end to end on two independent implementations** as of 2026-09-08; see [Portability, as measured](#portability-as-measured) before relying on any backend but those two.

## Why

Traditional codecs (H.264, HEVC, AV1) are shaped by decades of CPU constraints — sequential processing, complex prediction modes, intricate entropy coding with state chains. GPUs offer thousands of parallel threads, but these codecs can't exploit them.

GNC asks: if you start from zero with a GPU-first mindset, what do you end up with?

The answer so far: tile-independent processing, fully parallel entropy coding (256 independent streams per tile), and wavelet transforms that map naturally to GPU workgroups. It runs a full I/P/B video pipeline at 1080p on an integrated GPU. *(This line claimed "in real time … on an eight-core integrated GPU" until 2026-09-08. The core count was the wrong-hardware label BUG-29 retired, and "real time" is not what this project's own figures say — 1080p end to end is 5.0 fps. See [Video sequence](#video-sequence).)*

GNC is deliberately **broad**: intra and inter, 4:2:0 / 4:2:2 / 4:4:4 at 8 and 10 bits, and a quality range that runs from heavy compression through visually lossless to bit-exact lossless. The uses it is built for — contribution links, mezzanine and archival storage, low-latency preview, browser playback — encode about as often as they decode, which bounds how much encoder *search* is worth buying but not the compression target: that is roughly H.264-class across the whole range. Every figure below therefore names the operating point it was measured at, because several of this project's retracted results came from measuring one end and quoting it as if it described the codec.

## Status

**Working end to end:** I/P/B video pipeline with motion estimation, 8- and 10-bit, 4:4:4 / 4:2:2 / 4:2:0, five entropy coders of which three are selectable (Rice, `--rans`, `--abac`; Huffman and Bitplane are parked), and bit-exact lossless at `q=100`. **On Metal**, and — for correctness, not for throughput — **on Vulkan** since 2026-09-08. What runs on DX12 and in a browser is measured below and is less than this sentence used to claim.

**Where it stands against H.264** (measured 2026-09-06, `scripts/meas1_vs_h264.py`, 1080p, ki=9, x264 at defaults):

- **Contribution quality: +90.5% BD-rate on PSNR** — about 1.9x the bitrate of x264 for the same luma quality, across three sequences.
- **Colour: no advantage over x264, and the row that claimed one is withdrawn (CHROMA-2, 2026-09-07).** The control this README asked for has been run — give x264 the same allocation via `--chroma-qp-offset` and re-measure CIEDE2000 at the same total rate — and **x264 comes out ahead on all six runs** (three sequences x 4:2:0 and 4:4:4). On five of the six it does not need the offset at all: it leads on colour at offset 0 *while also leading luma by 4.1-7.4 dB*. The earlier row, which had GNC ahead on dE00, rested on a table measured an hour before CHROMA-1 changed q>=85 output and does not reproduce. GNC's colour is still good in absolute terms (dE00 0.54-0.92 mean, at or below the nominal JND) — it is just not better than x264's.
- **Lossless: the best wavelet result in the field.** 1.99:1 at `q=100`, beating JPEG 2000 lossless by 10.8% and PNG by 7.8%; behind FFV1 by 27% and x264 `-qp 0` by 43%, both of which predict against the neighbouring pixel rather than across scales.
- **Latency: ~80 ms round trip** intra or P-only, ~240 ms with the B-pyramid, 1080p on an Apple M5 Pro. On an NVIDIA RTX 4000 Ada over Vulkan the single-frame loop is **13.95 ms encode / 7.29 ms decode** (CANARY-1, 2026-09-07)
  (MEAS-6). That is the low-latency-HEVC band, **not** the JPEG XS band — JPEG XS codes 1–32
  lines and EBU measures it under one frame. The 256-line tile floor is not reachable today: the
  pipeline processes whole frames, so the practical floor is one full frame whatever the tile
  size. See [`docs/POSITIONING.md`](docs/POSITIONING.md) for where that leaves GNC against the
  incumbents in this segment.
- **Against the intra codecs it actually competes with** (MEAS-9, 2026-09-07,
  `scripts/meas9_contribution.py`, four images, one metric path). BD-rate, positive = GNC needs
  more bits at matched quality; both columns are given because a single one reverses the ranking:

  | | RGB PSNR | RGB, `--abac` | Y-PSNR (YCoCg-R) | Y, `--abac` |
  |---|---|---|---|---|
  | JPEG XS 4:4:4 | −10.2% | **−25.8%** | +29.4% | +7.7% |
  | ProRes 4444 | +20.2% | **+1.3%** | +29.3% | +9.1% |
  | JPEG 2000 9/7 | +54.2% | **+27.1%** | +79.7% | +48.3% |

  **JPEG 2000 uses the same transform as GNC — 9/7 wavelet, five levels — and still needs 54%
  fewer bits**, winning on CIEDE2000 at matched rate too. When the transform is the same, the gap
  is the entropy coder: J2K's is EBCOT, and `--abac` closes **exactly half** of it (ENT-4, −16.0%
  of rate at bit-identical pixels on 24 of 24 rungs). With `--abac` GNC matches ProRes 4444 and is
  ahead of JPEG XS 4:4:4 on RGB PSNR, and stays behind both on luma — an entropy coder does not
  move bits between planes. **Where the remaining 27% lives is now mostly accounted for** (INTRA-1, 2026-09-08, decisions `0026`–`0028`): entropy coding ≤7.5 points, chroma rate allocation 8.5 and *not* a coding deficiency, tiling 0.6% realisable, cross-tile allocation 0.95%, tile-boundary handling 0, and a dead zone worth ~3 — intra-only, because on video it costs up to 1.93 dB of worst-frame PSNR. **~6 points remain unexplained**, with no candidate left on the list.
  The 4:2:2 arms — JPEG XS 4:2:2, ProRes 422 — cannot be BD-rate compared at all: chroma
  subsampling caps them at 39–45 dB RGB PSNR, below GNC's range, and at matched rate GNC beats
  both on luma and colour, which is what full chroma resolution buys rather than a coding result.
- At *distribution* bitrates the gap is much larger. GNC is not built for that operating point.

**Off by default, and why:** the B-frame pyramid (costs 7–31% in rate on camera content and 160 ms in latency), temporal wavelet mode (loses 2–5 dB on high motion), and motion-compensated temporal filtering (measured 1.04–1.14x *worse* than a P-frame chain on every sequence tested).

See [`RESEARCH_LOG.md`](RESEARCH_LOG.md) for every measurement, including the ones that failed — roughly two dozen ideas have been tested and rejected, and they are written up as carefully as the wins.

## Portability, as measured

GNC targets the WebGPU feature set and asks wgpu for its *default* limits rather than the
adapter's, so the same WGSL is meant to run everywhere. That is the design. This is the evidence,
as of 2026-09-08:

| backend | status | evidence |
|---|---|---|
| **Metal** | measured end to end | every figure in this README |
| **Vulkan** | **intra and inter both run**, on two independent implementations | Intra throughput measured on three real GPUs: RTX 4000 Ada 13.95 ms encode / 7.29 ms decode, and on Windows an Intel Arc Pro at 36.45 / 26.63 ms against an RTX 2000 Ada at 17.75 / 11.33 ms. **Inter added 2026-09-08**: `encode-sequence` codes 1I + 2P and `decode-sequence` round-trips it on an RTX 4000 Ada, and Mesa lavapipe produces **byte-identical** frame sizes — two Vulkan implementations sharing no compiler code. *(This row said `block_match_split.wgsl` crashes three independent drivers and that P/B coding was unreachable. All four recorded crashes were on invalid SPIR-V from an upstream naga defect, fixed in `51a9ac6`; the Windows builds that looked like independent confirmation predate that commit. BUG-25 **fixed**, `docs/decisions/0029`.)* **No inter throughput figure yet, and Intel Arc has not been re-run since the fix** |
| **DX12** | **run once, on a software adapter, and it panicked** | Microsoft Basic Render Driver (WARP, CPU): exit 101 on a single frame, 2026-09-08. Adapter *enumeration* works across Vulkan/DX12/GL. **No DX12 hardware adapter has ever been tried** |
| **WebGPU / WASM** | compiles; **not verified in a browser**, and one known blocker | both abac GPU shaders declare 18 688 B of workgroup storage against WebGPU's 16 384 B limit. Native wgpu does not enforce it; a conformant implementation must. The decoder builds the abac decoder unconditionally, so if it bites, *every* WASM decode fails, Rice files included (BUG-31, open) |

**Two of the three non-Metal rows are still not clean**, which is why the top of this file states
Metal and Vulkan and stops there: GNC is *written* to be portable, is *measured* end to end on
Metal, and is measured *correct* — not yet fast — on Vulkan.
Platforms it has run on at all: macOS/Metal, Linux/Vulkan, and — since 2026-09-08 — Windows,
where it builds clean and runs all-intra on both GPUs of a two-GPU laptop.

Cross-backend output has been compared once (2026-09-07): at `q=100` Metal and Vulkan produce
byte-identical files, and at `q=75` they differ by one byte in 1.17 MB. Every file decodes to
identical pixels on both. **The decoder is bit-exact across backends and the lossy encoder is
not** — so conformance must require decoder bit-exactness, not encoder reproducibility.

## Current Results (1080p, bbb reference, Apple M5 Pro GPU)

### Single-frame (Rice+ZRL entropy)

| q | PSNR | BPP | VMAF | levels |
|---|------|-----|------|--------|
| 25 | 35.51 dB | 1.60 | 90.25 | 5 |
| 50 | 40.30 dB | 2.73 | 95.02 | 5 |
| 75 | 44.84 dB | 4.53 | 96.58 | 5 |
| 90 | 50.06 dB | 8.07 | 97.08 | 5 |

*Single-frame, 1080p bbb reference, Rice, 4:4:4. **[BASELINE.md](BASELINE.md) is the single source
for these** — this table was three separate copies from 2026-02-27 and had drifted more than 2 dB.
Throughput columns are deliberately absent: see BASELINE's fps section for why no single "encode
fps" exists.*

### Video sequence

**12.2 fps** GPU encode phase, **5.0 fps** end to end (1080p, q=75, keyframe interval 8, Rice,
measured 2026-09-06 on a machine that was not idle).

*The 31.7 fps this line used to carry is withdrawn: it is not reproducible, it matches none of the
three quantities below, and its stated parameters are internally inconsistent — ki=8 cannot produce
B-frames. See [BASELINE.md](BASELINE.md), "How to read the fps figures in this file".*

> **On the throughput figures above.** Three different quantities have been called "encode fps" in
> this project and they differ by 2.4x — the GPU encode phase, the encoder loop, and end-to-end
> wall clock. The figures here are the encoder loop. They were also measured on a machine that is
> not reliably idle: the same workload has timed 25.2, 31.1 and 37.5 ms across three runs, a 48%
> spread on identical work. **Treat every fps number in this README as indicative to about ±25%**,
> and say which of the three quantities you mean whenever you quote one.
> The compression figures (bpp, PSNR, CIEDE2000) are deterministic and carry no such caveat.

## Architecture

Everything runs as wgpu compute shaders. The pipeline:

```
RGB → YCoCg-R → Wavelet → Quantize → Entropy Code → Bitstream
         ↕          ↕          ↕            ↕
     (lossless   (CDF 9/7   (adaptive,   (Rice+ZRL:
      integer)   or 5/3)     CfL, AQ)    256 streams)
```

Each tile (256x256) is fully independent — no cross-tile dependencies. This gives parallelism, random access, and error resilience for free. See [`docs/PIPELINE.md`](docs/PIPELINE.md) for a detailed stage-by-stage breakdown.

### Pipeline stages

1. **Color space** — YCoCg-R via lifting (integer-exact, lossless-capable)
2. **Wavelet transform** — CDF 9/7 for lossy (q=1–99), LeGall 5/3 for lossless (q=100), 5 decomposition levels at q≥25, 4 below
3. **Adaptive quantization** — Per-block variance analysis on LL subband, geometric mean normalization, 3×3 spatial smoothing
4. **Quantization** — Uniform scalar with perceptual subband weights, dead zone, adaptive QP from AQ weight map. Fused quantize+histogram kernel when CfL is off.
5. **Chroma-from-Luma (CfL)** — Per-tile per-subband least-squares alpha (14-bit), active at q=50–85. Encodes chroma residuals instead of raw coefficients.
6. **Entropy coding** — Rice+ZRL (default): significance map + Golomb-Rice + zero-run-length, 256 independent streams per tile. rANS (32 streams), Huffman (64-symbol), and Bitplane also available but parked.

### Video features

- **I/P/B frames** — motion-compensated prediction with half-pel bilinear interpolation
- **Motion estimation** — hierarchical coarse-to-fine block matching (16x16, ±32px search)
- **Container** — GNV1 format with frame index table, keyframe seeking
- **Error resilience** — per-tile CRC-32 checksums, corrupt tile detection and recovery

## Build & Run

```bash
cargo build --release
```

### Encode / decode a single image

```bash
gnc encode -i input.png -o output.gpuc -q 75
gnc decode -i output.gpuc -o output.png
```

### Benchmark

```bash
gnc benchmark -i input.png -q 75              # Rice+ZRL (default)
gnc benchmark -i input.png -q 75 --rans       # rANS entropy (see Entropy Coders)
```

### Rate-distortion curve

```bash
gnc rd-curve -i input.png                     # sweep q=10..100, output CSV
gnc rd-curve -i input.png --compare-codecs    # also compare vs JPEG, JPEG 2000
```

### Encode / decode video sequence

```bash
gnc encode-sequence -i "frames/%04d.png" -o video.gnv -q 75 --keyframe-interval 8
gnc decode-sequence -i video.gnv -o "output/%04d.png"
gnc decode-sequence -i video.gnv -o "output/%04d.png" --seek 5.0  # seek to 5s
```

### Run tests

```bash
cargo test --release    # 148 tests: unit, regression, conformance
```

## Test Material

```bash
cd test_material && bash fetch_test_frames.sh
```

Downloads representative broadcast frames from [Xiph.org](https://media.xiph.org/) (requires ffmpeg and curl).

## Entropy Coders

GNC has five entropy coding backends, all decoding as GPU compute shaders:

| Coder | Streams/tile | Coding | Rate vs Rice | Decode vs Rice | Patent risk |
|-------|-------------|--------|--------------|----------------|-------------|
| **Rice+ZRL** (default above q=20) | 256 | Golomb-Rice + zero-run | — | — | None |
| rANS (`--rans`, default at q≤20) | 32 | Range asymmetric numeral systems | −6.4% at q=10, +0.4% at q=25; cannot encode above q≈76 | ~1.15× (TUNE-3, not re-measured) | Possible (MS patent) |
| abac (`--abac`) | 1 per 64px code-block | Adaptive binary arithmetic, context-modelled | −16.6% to −18.8% at q=50–90 | **1.69×** (idle-machine bench) | None known |

abac encodes on the GPU as well as decoding there (ENT-5): one thread per code-block, bit-exact
against the CPU coder in `abac.rs` — 98 of 98 whole-file comparisons byte-identical across four
stills, q=60–100, both arithmetic engines, 4:4:4/4:2:2/4:2:0 and an 8-frame sequence.
**Its encode time per frame is not measured**: four sessions were working this Mac when it landed,
and a throughput figure taken under load is worth nothing here. `docs/decisions/0024`.
| Huffman (parked) | 256 | 64-symbol + escape | not measured | not measured | None |
| Bitplane (parked) | Per-block | Sign + magnitude bitplanes | not measured | not measured | None |

*Rate column measured by ENT-2 (2026-09-07) on four stills at one commit, mean across images,
negative meaning rANS is smaller. Entropy coding is lossless and both coders quantise identically,
so equal q decodes to the same picture — verified, 0 of 40 points differ in PSNR — which makes this
an exact rate comparison rather than a BD-rate estimate. **The mean hides the spread**: at q=25 the
same setting runs from −3.5% (touchdown) to +8.2% (kristensara), so which coder wins is
content-dependent at every quality point. rANS overflows a fixed 4 KB per-stream buffer above
q≈76 (BUG-9), which is below the contribution operating point. Full ladder in
[RESEARCH_LOG.md](RESEARCH_LOG.md); [BASELINE.md](BASELINE.md) remains Rice-only and is the single
source for absolute figures.*

*The Decode column carries only figures someone actually timed, and says which run they came
from. The "1.5–2× faster" that stood here for Rice was neither: it contradicted the only throughput
figure in the repository — TUNE-3 measured rANS at ~8% encode and ~15% decode behind Rice, not
50–100% — so it is removed rather than corrected. abac's 1.69× is from an idle-machine bench;
rANS's ~1.15× is TUNE-3's and was **not** re-measured, because up to eight sessions share this Mac
and COORDINATION rule 1 forbids timing under load.*

Rice is the default because it eliminates the sequential state chain that limits rANS. Each of the 256 streams encodes independently — no shared state, no synchronization, minimal shared memory (< 1 KB vs rANS's 16 KB frequency tables). That is a GPU-parallelism argument, and the rate figures above no longer argue against it: level with rANS where Rice is selected, and rANS keeps the range below q=20 where it is 6–7% smaller. Huffman and Bitplane are available but parked.

**abac needs no BD-rate either**, for the same reason the rANS column above is exact: entropy
coding is lossless, so abac and Rice decode to the *identical picture* and the only difference is
file size. Measured 2026-09-07 through the real
bitstream on bbb, blue_sky, kristensara and touchdown — encode to a file, decode on the GPU,
pixels compared:

| q | mean rate vs Rice, at identical pixels |
|---|---|
| 50 | **−18.8%** |
| 75 | **−16.6%** |
| 90 | **−17.3%** |
| 100 (bit-exact lossless) | **−13.4%** |

At lossless that takes GNC from +23.9% behind FFV1 to **+7.3%**.

It is opt-in rather than the default because it costs about **1.69× frame decode** — one serial
adaptive coder per code-block, against Rice's 256 branch-free streams per tile — and because its
CPU-side encoder is currently single-threaded (129 ms/frame against Rice's 23 ms). The rate result
is intra only; inter frames use the same coder with contexts that were tuned on intra
coefficients, and that has not been measured. See `docs/decisions/0017`.

## Quality Spectrum

Smooth, monotonic quality scaling from lossless to extreme compression:

```
q=100  Lossless     — bit-exact round-trip (LeGall 5/3 integer wavelet)
q=90   High quality — near-transparent
q=75   Production   — good general-purpose quality
q=50   Balanced     — CfL + adaptive quantization
q=25   Compressed   — broadcast-suitable
q=5    Extreme      — preview/thumbnail
```

*Deliberately without dB figures. This block used to carry its own set (q=75 → 42 dB, q=50 → 37,
q=25 → 33) which was a third copy of the 2026-02-27 numbers and had drifted 2–3 dB from the table
above. [BASELINE.md](BASELINE.md) is the single source; the Current Results table quotes it, and
nothing else in this file should.*

## WebGPU / WASM

The full decoder compiles to WebAssembly (263 KB) and is intended to run in browsers via WebGPU. **A browser render has never been verified** — see [Portability, as measured](#portability-as-measured), including BUG-31, which would fail every WASM decode if a conformant implementation enforces the workgroup-storage limit that native wgpu does not:

```bash
wasm-pack build --target web --release
```

Browser demo in `examples/web/index.html`.

## Project Structure

```
src/
├── lib.rs              Core types, quality_preset(), codec config
├── main.rs             CLI (encode, decode, benchmark, rd-curve, ...)
├── format.rs           Bitstream serialization (GP11 frame, GNV1 sequence)
├── encoder/
│   ├── pipeline.rs     Encoder orchestration
│   ├── sequence.rs     Video sequence, B-frames, rate control
│   ├── rice.rs         CPU Rice encoder/decoder (reference)
│   ├── rice_gpu.rs     GPU Rice encoder/decoder
│   ├── rans.rs         CPU rANS encoder/decoder
│   ├── rans_gpu_encode.rs  GPU rANS encoder
│   ├── huffman_gpu.rs  GPU Huffman encoder
│   ├── motion.rs       Motion estimation and compensation
│   ├── cfl.rs          Chroma-from-Luma prediction
│   ├── adaptive.rs     Adaptive quantization
│   ├── fused_block.rs  Block DCT-8×8 mega-kernel
│   └── ...
├── decoder/
│   ├── pipeline.rs     Decoder orchestration
│   ├── frame_data.rs   Frame data upload
│   └── gpu_work.rs     GPU dispatch
├── shaders/            WGSL compute shaders
│   ├── rice_encode.wgsl, rice_decode.wgsl
│   ├── rans_encode.wgsl, rans_decode.wgsl
│   ├── transform_97.wgsl, transform_53.wgsl
│   ├── block_match.wgsl, motion_compensate.wgsl
│   └── ...
├── bench/              BD-rate, codec comparison, quality metrics
└── experiments/        Experimental features

tests/
├── quality_regression.rs   Golden-baseline regression (q=25/50/75/90)
├── conformance.rs          5 conformance bitstreams + corruption tests
└── golden_baselines.toml   Reference PSNR/SSIM/bpp values
```

## Documentation

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — Detailed encode pipeline description
- [`docs/BITSTREAM_SPEC.md`](docs/BITSTREAM_SPEC.md) — Complete bitstream format specification (GP11 frame, GNV1 sequence)
- [`RESEARCH_LOG.md`](RESEARCH_LOG.md) — Experiment log with hypotheses, results, analysis

## License

All code is patent-free. No H.264/5/6 patent pool or MPEG-LA encumbered techniques. All dependencies are open source.
