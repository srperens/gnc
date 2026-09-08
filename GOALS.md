# GNC — Goals, Rules & Priorities

## 1. What GNC Is

GNC is a patent-free **video codec** designed from scratch for GPU parallelism. Everything runs as wgpu compute shaders (WGSL), written against the WebGPU feature set so the same source targets Metal, Vulkan, DX12 and WebGPU/WASM. The core idea: tile-independent processing with thousands of parallel threads instead of sequential CPU-era algorithms.

**That is the design, and it is not yet fully the measured state.** As of 2026-09-08 every performance figure is **Metal**; on **Vulkan** intra *and* inter now run end to end on two independent implementations, with byte-identical output between them, but no inter throughput figure exists yet (BUG-25 **fixed** — `docs/decisions/0029`; the four driver crashes on record were all one upstream naga defect on invalid SPIR-V). DX12 has never been run beyond a software adapter that panicked, and the WASM path is unverified in a browser with one known limit breach (BUG-31). The README's *Portability, as measured* table is the current evidence and should be read before any claim of cross-platform support is repeated.

### GNC is broad on purpose — that is the decision, not an unresolved question (2026-09-07)

This is the decision that sets every target below, so it comes first, and it was **re-affirmed
explicitly on 2026-09-07** after the repository had drifted into narrowing it. The market
requirements, the sourced external facts and the measurements are in
**[docs/POSITIONING.md](docs/POSITIONING.md)** — read that before changing any target here, but
read §2 knowing its "commit to one segment" recommendation was **rejected**.

**The goal is a codec that is good at many things, not excellent at one.** A codec built for a
single niche is one of many; there are already dozens. A codec that spans the range is
interesting. Concretely, all of the following are in scope at once and none of them is the
"primary" one:

| axis | the target |
|---|---|
| intra | strong — the same picture quality per bit as H.264 intra |
| inter | strong — a static studio shot should cost almost nothing, and today it does not |
| quality range | heavy compression through visually lossless to **bit-exact lossless** |
| chroma | 4:2:0, 4:2:2 and 4:4:4, at 8 and 10 bits |
| use cases | contribution, mezzanine, **archival**, low-latency preview, browser playback |
| parallelism | massively parallel, and *designed* to be portable across every GPU with WebGPU/Vulkan/Metal/DX12 — measured on Metal, see §1 |
| compression | roughly **H.264-class**, across that whole range |

**Several internal strategies, selected by quality and bitrate, is a legitimate design — not a
failure to find one.** If no single mechanism covers the range, the codec picks per operating
point, and it already does: MED prediction replaces the wavelet entirely at q=100, the entropy
coder follows quality, and the wavelet depth follows the tile size. Adding to that set is normal
engineering here, not an admission of defeat. What is *not* acceptable is a strategy that only
works at one end and is quietly measured only there.

Where GNC is meant to win outright is portability and scale, against fixed-function silicon:

| | fixed-function (NVENC/QSV/VideoToolbox) | GNC |
|---|---|---|
| where it runs | one vendor, specific silicon generations | any GPU with WebGPU/Vulkan/Metal/DX12 — by design; measured on Metal, see §1 |
| concurrent streams | a fixed number of encoder blocks per chip, plus driver session limits | limited by general compute, so it scales with the card |
| 10-bit 4:2:2 | only on recent hardware (NVENC: Blackwell and later) | a design target from the start |
| patents | licensed formats | patent-free |

The structural argument is that the number of hardware encoder blocks in a chip is roughly
constant no matter how large and expensive the GPU is, while shader throughput scales with the
card. A bigger GPU should therefore buy more GNC instances; it does not buy more NVENC blocks.
**That claim is currently unproven and is the single most important thing to measure** (MEAS-5).

Two things that follow, and they are about *measurement discipline*, not about narrowing scope:

- **Every operating point is in scope, so a result must say which one it was measured at.** The
  expensive lessons behind this stay: TUNE-5 was measured at q=15-50, shipped, and cost 10.3 dB at
  q=99; the headline gap figure was 3x wrong because it was taken at distribution bitrates. A
  ladder that stops at q=50 has not tested this codec, and neither has one that starts at q=85.
- **Encoder effort is not free here.** GNC encodes and decodes about as often as each other, so
  the distribution-codec trade — burn unbounded encoder time to shave a percent, because the
  bitrate is paid a billion times over — does not apply. That bounds how much *search* is worth
  buying; it does not lower the compression target.

**GNC is both intra and inter, and going all-intra is explicitly rejected.** Most established
contribution formats are all-intra and it would be the easy answer. Entropy coding is the biggest
single gap against H.264 on inter, and it is the one lever that pays in every row of the table
above at once — intra, inter, lossless, every chroma format — which is why it leads the priority
order rather than a format-specific feature.

## 2. Design Rules

1. **Patent-free** — No patented techniques, period. If it's patented, we don't use it.
2. **GPU-first** — Everything runs in compute shaders. No CPU fallback paths. CPU reference implementations only for validation/testing.
3. **Massive parallelism via tile independence** — No cross-tile dependencies at any stage. Each tile encodes/decodes in isolation. This is what enables thousands of parallel GPU threads.
4. **Cross-platform** — Must work on Metal, Vulkan, DX12, and WebGPU (WASM). No backend-specific features. WGSL shaders are the single source. **Half met as of 2026-09-08: Metal and Vulkan both run the whole codec; DX12 and the browser do not** — see §1. Portability is the axis the project claims to win on (§1), so a backend it cannot run on is a headline defect and not a compatibility nit.
5. **No f64 in shaders** — Apple and mobile GPUs have no hardware double precision, and WGSL has no `f64` in any case.
6. **Open source only** — All dependencies must be open source.
7. **English only** — All code, comments, docs, and commit messages in English.
8. **Measure everything** — Every change benchmarked: PSNR, SSIM, bpp, encode/decode FPS. Compare against baseline and previous best. Optionally compare against relevant codecs (H.264, H.265, AV1, MJPEG, JPEG XS, ProRes) for context.
9. **No code duplication** — Extract shared logic. Code must pass `cargo fmt` and `cargo clippy` with zero warnings. The exact clippy commands are in CLAUDE.md, "Code Style": `--all-targets` on native since BUG-20 (`docs/decisions/0062`), `--lib` on wasm. **The `cargo fmt` half of this rule is currently false** — 566 diffs in 61 files, 504 of them under `src/` — filed as **BUG-38**, not yet decided.
10. **No legacy** — Nobody runs GNC in production. We can break the bitstream format, change the container, rename fields, restructure anything. No backward compatibility constraints.
11. **Video codec first** — GNC is a video codec, not an image codec. Sequence encode/decode performance is the primary metric. Single-frame performance only matters as a component of video throughput.

## 3. Current State

**Entropy coder: Rice+ZRL** (256 independent streams/tile, fully GPU-parallel, patent-free).
rANS and Huffman exist in the codebase but are parked — Rice is the default for all paths.
**abac** (adaptive binary code-blocks) shipped 2026-09-07 as an opt-in fifth backend: −16.6% to
−18.8% of rate against Rice at *identical pixels* on intra, for ~1.69x frame decode. Opt-in and
not a default — `docs/decisions/0017`.

| q | PSNR | BPP | VMAF | levels |
|---|------|-----|------|--------|
| 25 | 35.63 dB | 1.57 | 90.31 | 5 |
| 50 | 40.25 dB | 2.60 | 95.07 | 5 |
| 75 | 44.64 dB | 4.31 | 96.55 | 5 |
| 90 | 49.89 dB | 7.21 | 97.06 | 5 |

*Single-frame, 1080p bbb reference, Rice, 4:4:4. **[BASELINE.md](BASELINE.md) is the single source
for these** — this table was three separate copies from 2026-02-27 and had drifted more than 2 dB.
Throughput columns are deliberately absent: see BASELINE's fps section for why no single "encode
fps" exists.*

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

**Key GPU architecture insight:** shared-memory occupancy dominates performance. 16KB is the budget because that is what **GNC requests** (wgpu defaults, for WebGPU portability) — the adapter here offers 32KB, so this is a self-imposed ceiling, not the chip's (BUG-29). At 16KB, 2 workgroups/core is full occupancy; Rice uses < 1KB shared, so occupancy is excellent.

**Known gaps:**
- Sequence encode: **12.2 fps GPU encode phase, 5.0 fps end to end** (BASELINE's A and C,
  1080p q=75, non-idle machine) → target 60 fps. The 31.7 fps this line used to carry is the
  figure retracted four paragraphs above; it stood here for a day after being withdrawn.
- Single-frame encode 40 fps → target 60 fps
- ~~8-bit only (10-bit not implemented) — the main format gap for broadcast contribution~~ — **wrong, corrected 2026-09-08. 10-bit shipped on 2026-09-06 (FMT-1).** `--bit-depth 10` on `encode` and `encode-sequence`; 10-bit samples live in the high bits of 16-bit PNG channels. Verified end to end on a genuine 10-bit 1080p source: **q=100 is bit-exact — max error 0 over 6 220 800 samples** — and q=90 reads 61.33 dB. This line stood for two days after the gap closed, and it was the gap FMT-1 called *"the first-order problem"*, so it was the most misleading sentence in this file
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

> **Correction, MEAS-3 (2026-09-07): the 17–27% is an equal-setting rate figure and does not
> survive as a saving.** Measured as BD-rate against all-intra on three sequences, the shipped
> inter configuration needs **+4.6% more bits on mean PSNR and +19.1% more on worst-frame PSNR**,
> winning only on low-motion animation (−24.2%). Above q≈85 the saving is gone and at q=95 inter
> costs more than all-intra. The older figure compared rates at the same q, where the inter arm is
> 7.6 dB worse on crowd_run — its quality evidence was VMAF 99.09 against 99.10, which is
> saturated. The H.264 60–70% half of the sentence stands; it was not re-measured. See
> `docs/decisions/0019` and BACKLOG INTER-1.
>
> **Corrected again, INTER-1 (2026-09-07): those figures were themselves measuring BUG-27** — the
> encoder's P-frame reference was dequantised with the intra quantiser step, so its reference
> disagreed with the decoder's at every q ≤ 80, which is most of MEAS-3's q=25–95 ladder. Re-run
> on the same harness: **−0.3% on mean PSNR and +8.0% on worst-frame**, not +4.6% / +19.1%. And at
> the contribution operating point specifically (q=85–99, where the defect could not occur) the
> shipped configuration measures **−1.9% mean / −0.2% worst-frame** — a wash, not a loss. The
> remaining deficit is content-specific: old_town_cross alone is +28.7% on the worst frame.
> **So "the inter path is a loss at contribution quality" is withdrawn**; ki=9 is the best of four
> intervals measured, and inter stays a default. See `docs/decisions/0023`.

**MEAS-1 (2026-09-05) measured the gap properly for the first time — at the wrong operating
point. QUAL-1 (2026-09-06) re-measured it at the right one.** MEAS-1 found GNC needing **5-7x**
the bitrate of H.264 (BD-rate +457% / +494% / +672%) at *distribution* bitrates, with the quality
ladder above q=92 dead. Re-run at contribution quality with that ladder working, the same harness
and the same parameters gives **+90.5% BD-rate on PSNR — about 1.9x** (+129.0% bbb, +71.9%
old_town, +70.6% crowd_run). Nothing in the coder changed between the two; the 5-7x figure was
measured somewhere GNC is not built to operate. **MEAS-10 re-took that ladder on 2026-09-08 at
`0a1b055`: +89.2%** (+128.5% / +70.2% / +68.8%). INTER-2 moved the q=85 rung; 92/96/99 did not.
**Use +89.2% for current HEAD, and do not quote a VMAF BD-rate at this end** — widening the
quality ladder moved the VMAF figure by 47.5 points on average and the PSNR figure by 1.0. The
+13.9% still-image figure is PSNR on stills, a third quantity again.
See [RESEARCH_LOG.md](RESEARCH_LOG.md), 2026-09-06, and the decision record
[docs/decisions/0013](docs/decisions/0013-the-headline-gap-figure-was-the-wrong-operating-point.md).

**The colour lead is withdrawn (CHROMA-2, decision 0020).** It read: at rate matched to 1%, GNC
beats x264 on CIEDE2000 (0.611 vs 0.684 on bbb, 0.911 vs 0.949 on old_town). The control that
settles it — hand x264 the same allocation via `--chroma-qp-offset`, re-match the rate — was run on
2026-09-07 and **x264 wins on all six runs**, on five of them without needing the offset and while
also leading luma by 4.1–7.4 dB. GNC has no measured advantage over x264 on any axis at this
operating point. The superseded text follows.

<!-- superseded 2026-09-07, kept so the reversal is visible rather than silent -->
**And luma alone misleads here.** At rate matched to 1%, GNC beats x264 on CIEDE2000 (0.611 vs
0.684 on bbb, 0.911 vs 0.949 on old_town, better 95th percentile on all three) while losing
7.4-8.8 dB of luma PSNR. The two codecs allocate rate differently between luma and chroma, so a
single luma number overstates the gap for a colour-critical use case and understates the luma
deficit. Quote both.

**And that allocation difference is not where the gap is — CHROMA-1 checked (2026-09-06).** If GNC
simply spent more on chroma than the optimum, the luma gap would be partly a config choice. It is
not: `chroma_weight` moves the file by 1.5% and luma by 0.01 dB in the shipped 4:2:0 P-chain,
because motion compensation leaves almost no chroma residual to reclaim. The knob is worth
−20.8% on an all-intra sequence and −2.9% on a ki=9 P-chain. **So the +90.5% is genuine luma
coding deficit, and intra is the only route** — by elimination now, not by assumption.

**MEAS-4 (2026-09-05) located that gap.** It is *prediction quality*, not the coding model.
Simulating both models on GNC's own motion-compensated residuals at matched distortion, an
idealised per-block DCT with oracle block skip beats GNC's wavelet by only 4–23% at broadcast
quality and *loses* by 3–18% at low bitrate; context-adaptive entropy coding is worth ≤3.4%
*(scope: **inter** residuals, and a two-signal context model — `GNC_SIG_CONTEXT`. It is not a
verdict on context modelling in general. EBCOT's nine-context model over code-blocks measures
−16.6% to −18.8% on **intra**, shipped 2026-09-07; see BACKLOG "EBCOT — evaluating in halves")*;
and
only 0–2% of blocks are skippable at q=75, meaning the prediction leaves error nearly everywhere.
An x264 ablation on the same content agrees from the other side: H.264's largest inter lever is
multi-reference and B-frame prediction (+29–32%), three times CABAC and thirty times sub-block
partitioning. Full writeup:
[docs/decisions/0005-meas4-inter-gap-decomposition.md](docs/decisions/0005-meas4-inter-gap-decomposition.md).

So meaningful temporal compression stays a goal, and the form is now much clearer than "rebuild
the inter pipeline": GNC uses **single-reference P-frames**, and the lever the measurement says
matters most is the one it does not have. Multi-reference prediction is ordinary and
GPU-parallel. That is where the inter work goes next ([BACKLOG.md](BACKLOG.md) #25); per-block
inter transforms and block skip are ruled out by measurement. **"Context entropy" is not** — that
line was written from the ≤3.4% figure above, which measured a much weaker model on inter
residuals. What is ruled out is *that* model; context modelling over code-blocks is now shipped
and is the largest single-mechanism gain in the codebase. Whether it also pays on inter residuals
is untested.

GNC's distinguishing properties hold regardless — **patent-free + GPU-native + tile-independent +
low-latency + WebGPU/WASM browser decode** (JPEG XS is patented, VC-2 is CPU-era, JPEG 2000 is
slow) — and they serve broadcast contribution, mezzanine storage, low-latency preview and
browser playback.

GNC should become a **good, robust codec** — not optimized along a single axis. We iterate across multiple dimensions simultaneously, looking for combinations of techniques that work well together. No single property is a hard blocker for the others.

**Target properties (all of these, no strict order):**

| Property | Current | Target |
|----------|---------|--------|
| **Concurrent streams per GPU** | **never measured** | beat NVENC's session/block ceiling on the same machine |
| **Latency per frame** | **~80 ms round trip at the default**, of which **0 frames** are reordering delay (MEAS-6, `docs/decisions/0033`). Below the low-latency-HEVC band's 120 ms floor, above JPEG XS. The structural half is exact; the ~80 ms coding half is a non-idle measurement and is owed on an idle machine, and glass-to-glass is still unmeasured | sub-frame, end to end — at 50 fps that is 20 ms, so ~80 ms is four frames short |
| Encode speed | 12.2 fps GPU encode phase / 5.0 fps end to end (seq, 1080p q=75, non-idle; BASELINE A and C) | 60 fps |
| Bit depth | **8-bit and 10-bit, both shipping** (FMT-1, 2026-09-06; 10-bit lossless re-verified 2026-09-08) | met — keep it met as the format changes |
| Chroma formats | 4:4:4, 4:2:2, 4:2:0 | keep all three working at 10-bit |
| Compression (intra) | **Read these three numbers with their caveats, they are not one quantity.** +46–55% vs H.264 all-I on video is **VMAF, predates the high-q ladder fix and has not been re-run** (BASELINE says so); +13.9% on stills is PSNR against H.264 all-I; and against JPEG 2000 9/7 the gap is +27.1% of which **15.1 points are not coding deficiencies at all**, so the intra *coding* gap is nearer **+12%** (INTRA-1, answered 2026-09-08) | ≤ H.264 all-I, measured at contribution quality — and re-run the VMAF figure as PSNR |
| Compression (video) | **+89.2% BD-rate on PSNR vs H.264 at contribution quality** with Rice; **+66.0% with `--abac`** at identical pixels (MEAS-10, 2026-09-08; QUAL-1's +90.5% was 2026-09-06; +457% to +672% was distribution bitrates and is superseded) | ≤ +25%, and the remaining gap is intra |
| Colour accuracy | **no lead — withdrawn 2026-09-07, decision 0020.** x264 wins dE00 on 6 of 6 rate-matched runs, on five without needing a chroma-QP offset and while also leading luma. GNC's dE00 0.54–0.92 mean is good in absolute terms, just not better | close the luma gap; there is no colour lead to keep |
| Luma/chroma split | on the frontier as of CHROMA-1 (2026-09-06) — `chroma_weight` 1.2 is the largest value that costs nothing on MEAS-8's criterion | leave it; the remaining gap is not here |
| Quality range | q=1–100 functional | smooth, predictable quality curve |
| Robustness | basic test coverage | no artifacts, stable across q and content |
| Bitstream | GNV1/GNV2 defined | well-specified, documented |

**Updated 2026-09-08 — this ordering has to be restated, because it no longer says anything.**
It read: "The two metrics at the top of that table have never been measured, and they are the ones
the whole positioning rests on. They come before further compression work." One of the two is now
measured and the other cannot be, so as written it directs every session at work that is either
done or impossible:

- **Latency per frame is measured** (MEAS-6, ~80 ms, 0 frames of reordering). What is left is the
  cheap half — re-take the coding time on an idle machine — and glass-to-glass instrumentation
  that nobody has built.
- **Concurrent streams per GPU is parked, not skipped.** Claim A (no session cap, and it runs
  where NVENC does not) is closed and sourced. Claim B (more aggregate throughput than the card's
  own NVENCs) needs hardware this project does not have — a discrete NVIDIA card with a current
  driver, or an idle Mac for the `--density-still` re-take. It is still the single most important
  thing to measure, and it is blocked on a machine rather than on effort.

**So the ordering is: unblock what is blocked when the hardware appears, finish the cheap half of
latency, and otherwise pick by value — which now means compression again**, since the gate that
was holding it back is a hardware queue and not a question of priority. `docs/decisions/0033` and
MEAS-5's entry carry the detail.

**On the compression numbers:** both objections to the +457% to +672% figures have now been
settled rather than merely noted. B-frames were defective (BUG-5) and are off by default; the
operating point was wrong, and QUAL-1 re-measured it at the contribution end, where the gap is
**+89.2%** (QUAL-1's +90.5% on 2026-09-06, re-taken MEAS-10). Treat +457% to +672% as a
historical distribution-bitrate figure only. The remaining gap is **intra** — inter breaks even
for both codecs at this quality, and does so for x264 too.

VC-2 (Dirac) demonstrates that a patent-free wavelet codec can do real temporal work (MCTF) and
reach H.264-class compression. That remains the reference point for where the inter path could go.

**How we iterate:**

- Pick the next backlog item based on what provides the most overall value right now — not what happens to be listed as P1
- Rotate between compression, speed, and robustness — progress in one area does not unlock another
- Always measure on ≥3 sequences and multiple q levels — a codec that is only good over a narrow quality band is not a good codec
- Technology choices are driven by: patent freedom, GPU parallelism, measurable improvement

## 5. Design Philosophy

**Correctness over speed.** A codec with subtle bugs is worthless. Verify every change end-to-end. A fast encoder that produces subtly wrong output is not a working encoder.

**Form first, then speed — but speed is not optional (owner, 2026-09-06).** The codec has to end up
*ruthlessly* fast; that is a headline property, not a nice-to-have. Right now, though, the binding
constraint is the shape of the thing: the algorithms, the design, and the measured gaps against
H.264/H.265 (§4). So the sequencing is deliberate:

- **Now:** close the gaps. Get the algorithms and the design right. A change that closes a real gap
  is worth taking even if it costs decode time, because an architecture that is fast and behind on
  compression is the harder of the two problems to fix later — you cannot optimise your way to a
  better transform or a better predictor.
- **Later, as a first-class push:** performance. Not incidental tuning — a dedicated effort with
  its own targets, against real profiling, on an idle machine.

Two standing consequences:

1. **Do not reject a design win purely on throughput while we are in the "form" phase.** Record the
   cost honestly, in a comparable unit, and keep the option. The abac entropy coder is the worked
   example: −16.7% rate for 1.69× frame decode at q=90. During the form phase that is a candidate,
   not a rejection — see BACKLOG "EBCOT part 6".
2. **Do not let throughput debt accumulate silently either.** Every accepted design win that costs
   speed must be logged with a *measured* cost, so the later performance push has a list to work
   from rather than a re-measurement project. Also worth knowing before that push starts: the
   entropy stage is 47% of frame decode, so it caps any entropy-side optimisation at 1.9x.

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

- **Beating AV1/H.265 on compression ratio** — H.264-class across the whole range is the target (§1); matching the codecs a generation beyond it is not. The design point is parallel, low-latency and patent-free, and the compression target is set by what that design can reach, not by the best number in the field. This is *not* licence to dismiss single-digit gains: against the corrected ~1.9x gap they accumulate, and every "too small to bother with" judgement in this repo predating 2026-09-06 was made against a denominator that was 3x wrong.
- **Narrowing GNC to one segment** — Rejected 2026-09-07, and docs/POSITIONING.md §2's recommendation to do so is superseded. A codec for one niche is one of many; spanning the range is the point.
- **CPU decode path** — GPU-only by design. No software fallback.
- **Backward compatibility** — No legacy bitstreams to support (rule 10).
- **Neural/ML compression** — Extreme complexity for marginal gains. Not worth it for GPU-native design.
- **Maximum single-thread performance** — We scale with parallelism, not clock speed.
