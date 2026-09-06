# GNC — Positioning, Market Reality, and What To Do About It

**Status:** current as of 2026-09-05. Supersedes nothing; it is the reasoning behind
[GOALS.md](../GOALS.md) §1 and the priority order in [BACKLOG.md](../BACKLOG.md).

**Shareable version:** <https://claude.ai/code/artifact/d8b12c5b-8437-40c9-9a1e-42b2fbb5bfa0>
(same content, formatted for reading outside the repo).

This document exists because GNC's targets were being set against the wrong operating point. It
records three things: what GNC is for, what the market it aims at actually requires, and where GNC
measurably stands against that. External claims are sourced. Internal claims are measured, and the
measurement is named.

---

## 1. What GNC is

**GNC is a contribution codec, not a distribution codec.**

A distribution codec (H.264, HEVC, AV1) is encoded once and decoded a billion times. Spending
enormous encoder effort to shave a percent off the bitrate is rational there, because the bitrate
is paid a billion times over. GNC is not that. GNC encodes and decodes roughly as often as each
other: contribution links, mezzanine storage, low-latency preview, browser playback.

So GNC does **not** try to beat H.264 or AV1 on compression. It aims for roughly H.264-class
quality that runs on **any** GPU and scales with the card, against fixed-function encoders that
are vendor-locked and session-limited.

The structural argument: the number of hardware video-encoder blocks in a chip is roughly constant
regardless of how large and expensive the GPU is, and driver session limits cap it further, while
general shader throughput scales with the card. A bigger GPU should therefore buy more GNC
instances; it does not buy more NVENC blocks.

### The thesis is two claims, and only one of them holds today

Measured and sourced 2026-09-05 ([BACKLOG.md](../BACKLOG.md) MEAS-5).

**Claim A — "no session cap, and it runs where NVENC does not" — holds, and is stronger than the
project has been claiming.**

- NVENC's consumer limit is **12 concurrent sessions per *system***, explicitly *"the combined
  number of encoding sessions executed on all non-qualified cards present in the system."*
  **Adding a second GeForce buys zero additional sessions.**
- **A100, H100 and B200 ship with zero NVENC.** NVIDIA's Hopper whitepaper states it outright:
  H100 GPUs *"do not include display connectors, NVIDIA RT Cores for ray-tracing acceleration, or
  an NVENC encoder."* 132 SMs and no encoder. **The most valuable GPUs in the world cannot encode
  video at all**, so any organisation with an idle AI fleet has zero encode capacity. That is a
  market, not a benchmark, and it is the most under-used fact GNC has.
- The **GeForce driver licence §2.8** prohibits datacenter deployment. Encoding at density with
  NVENC legally requires professional or datacenter SKUs, independent of the session counter — a
  commercial wall, not just a driver counter.
- Engine counts are flat or sublinear against compute: Ampere runs one NVENC from RTX 3050 (20
  SM) to RTX 3090 Ti (84 SM), **4.2× the compute with the same single encoder**. And **per-engine
  throughput grew just 14% from Turing to Blackwell** while shader FP32 grew roughly sixfold.

**Claim B — "more aggregate throughput than the card's own NVENCs" — is unproven, and the first
local measurement is sobering.** N concurrent 1080p encodes on the M1, two runs:

| instances | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| aggregate fps | 7.02 / 6.22 | 11.13 / 9.51 | 11.51 / 12.23 | 14.15 / 13.38 |

**Roughly 2× aggregate at N=8, and most of it already reached at N=2.** A single 1080p encode does
not saturate the M1, so there is real headroom — but nowhere near linear. The published
multi-tenancy literature agrees: concurrency converts *idle* GPU into *useful* GPU; it does not
create GPU. NVIDIA's own consolidation study measured time-slicing at 0.76 requests/s where MIG
gave 1.00 — a 32% *reduction*.

**What this changes:** lead with Claim A, which is defensible today and does not depend on
out-running fixed-function silicon. Do not put Claim B in front of anyone until it is measured on
a discrete card.

### The historical GPU-encoder failures do not generalise to GNC

Every documented failure was a **block-based hybrid codec with adaptive arithmetic coding**. Jason
Garrett-Glaser, x264 maintainer, 2008: *"basically everything can be reasonably done on the GPU
except CABAC (which could be done, it just couldn't be parallelized)."* NVIDIA's deprecated CUDA
encoder failed on *scope* — one reference frame, no configurable search range, no two-pass — not
on physics. BeHardware's 2011 study found the shipping GPU encoders performed identically on €100
and €330 cards because they were never compute-bound at all.

Every surviving GPU-compute codec has GNC's exact shape: wavelet, spatially independent tiles,
parallel entropy coding. NVIDIA killed its CUDA H.264 encoder and ships nvJPEG2000 in the same
product line.

The most encouraging sourced datapoint, and it is an inference rather than our measurement:
Fastvideo's JPEG 2000 encoder on an RTX 4090 reports 616 fps at 4K — about **5.1 Gpixel/s** —
against that same card's two NVENC engines at H.264 P1, about **3.8 Gpixel/s**. A CUDA wavelet
codec already out-throughputs the card's fixed-function encoders in raw pixels per second, while
carrying EBCOT, which is dramatically heavier than Rice or rANS.

---

## 2. The market is two markets, and GNC is straddling them

This is the most consequential finding of the 2026-09-05 research sweep. Two segments are
routinely conflated and they share almost no buyers, sales channels, or requirements.

| | **Live contribution** | **Cloud mezzanine / proxy / preview** |
|---|---|---|
| incumbents | JPEG XS, low-latency HEVC, VC-2, AVC-Intra | ProRes, DNxHR, J2K, ad-hoc H.264 |
| bought by | broadcast engineering | platform / post engineering |
| transport | SMPTE ST 2110-22 on managed links | object storage, HTTP, WebRTC |
| latency budget | **1–32 lines** (JPEG XS); sub-frame | 500 ms – 2 s |
| bit depth | **10-bit 4:2:2, non-negotiable** | 10-bit preferred |
| rate control | **constant bitrate, mandatory** | variable acceptable |
| entry requirements | SDI or ST 2110 I/O, NMOS IS-04/IS-05 control plane, JT-NM Tested | none |
| verdict for GNC | **probably not winnable** | **winnable** |

**Live contribution is JPEG XS's home ground.** It is embedded in ST 2110-22, has FPGA and ASIC
silicon, multi-vendor interop, a VSF interop recommendation, JT-NM certification and an Emmy.
EBU TR 091 states as an entry condition that a codec must be purchasable or rentable and must use
SDI or ST 2110 interfaces — GNC cannot currently be entered into the industry's own evaluation.

**Recommendation: commit to cloud mezzanine, proxy and preview. Stop claiming both.**
GOALS.md currently claims both, and the requirements diverge by years of work.

---

## 3. What the market actually requires

### 10-bit 4:2:2 is a gate, not a feature

- **EBU R 153 v2.0** (live contribution of UHD/HDR) specifies *"Video Format 10-bit 4:2:2
  Y′C′BC′R"*, BT.2100 HLG, BT.2020 primaries, and states that *"Use of a Standard Dynamic Range
  (SDR) transfer function is **not allowed**."*
- **EBU TR 091**'s entire codec test matrix is 1080p/50, 2160p/50, 1080p/59.94 and 2160p/59.94,
  all BT.2100 HLG. There is no 8-bit test point and no 4:2:0 test point anywhere in it.

A codec that cannot ingest 10-bit 4:2:2 cannot be entered into the industry's reference
evaluation at all. **GNC's video path is 8-bit.** See [BACKLOG.md](../BACKLOG.md) FMT-1.

### Compression is not the currency; latency and generation-survival are

From **EBU TR 092** (Oct 2025), measured on real vendor equipment:

| | 1080p/50 | 2160p/50 |
|---|---|---|
| JPEG XS, EBU-recommended 8:1–6:1 | **259–346 Mbps** | 1037–1382 Mbps |
| low-latency HEVC | **40–60 Mbps** | 60–90 Mbps |

The same picture costs roughly **seven times** as much in JPEG XS as in HEVC. Broadcasters pay
that premium purely to buy latency and multi-generation robustness. EBU also judged JPEG XS's own
vendor-suggested 10:1 ratio as *"not comparable to the source and should not be used"* — the
market runs these codecs **more conservatively than their vendors market them**.

**Implication:** "H.264-class quality" is the wrong claim to make in this market. The claim that
matters is *"visually lossless at 6:1, still clean at the third generation."*

### Latency

| | measured / quoted |
|---|---|
| JPEG XS | 1–32 lines algorithmic; EBU measured **< 1 frame** |
| VC-2 low delay | a few lines |
| NDI High Bandwidth | < 16 ms at 1080p60 |
| low-latency HEVC | EBU measured **120 ms – 3060 ms** across real vendors |

**Measured 2026-09-06 (MEAS-6):** GNC's codec round trip at 1080p on an M1 is **~80 ms** — about
47 ms encode plus 35 ms decode. On top of that, **the default B-pyramid costs 8 frames of
lookahead**: `ki=17` encodes in the order `0[I] 4[B] 8[P] 2[B] 6[B] 1[B] ...`, so frame 1 cannot
be coded until frame 8 has arrived. At 50 fps that is **160 ms of structural delay before any
coding runs**. P-only coding encodes in display order with **zero** reordering delay.

| | latency |
|---|---|
| JPEG XS | 1–32 lines; EBU measured under one frame |
| NDI High Bandwidth | under 16 ms |
| **GNC, intra or P-only** | **~80 ms** |
| **GNC, B-pyramid (current default)** | **~240 ms** |
| low-latency HEVC | 120–3060 ms |

**GNC's default configuration sits in the low-latency-HEVC band, not the JPEG XS band.** Dropping
the pyramid is worth roughly 3× — and BUG-5 independently measured that same pyramid as *costing*
7–31% in rate at contribution quality on camera content. Two measurements, one conclusion.

Note also that the often-quoted ~256-line tile floor is not currently reachable: the pipeline
processes whole frames, so the practical floor is one full frame regardless of tile size. Getting
near JPEG XS would require emitting tile rows as they complete, which the pipeline does not do.

### Error resilience: nobody appears to be buying it

The industry answer is **SMPTE ST 2022-7 seamless protection switching** — two identical streams
over diverging paths, receiver takes whichever packet arrives first — or ARQ/FEC at the transport
layer (SRT, RIST, Zixi). No source found values codec-level graceful degradation, per-tile CRC, or
partial-frame recovery. In a contribution plant a corrupted frame is replaced by the redundant
path; if both fail the operator wants an alarm, not a partially-decoded picture.

GNC's per-tile CRC is engineering the project is proud of that customers do not appear to be
asking for. *(Confidence: medium-high, partly inference from a consistent absence across EBU,
SMPTE, VSF and vendor material.)*

### How quality is actually judged

**ISO/IEC 29170-2 Annex B**, the flicker paradigm, is the formal visually-lossless test — JPEG XS
was qualified with it across three independent labs. EBU TR 091 runs two phases: automated
(PSNR / VMAF / SSIM, plus a multi-generation breakdown-point search) and then **expert viewing**.

Note for internal calibration: **VMAF is barely present in this literature.** EBU lists it as one
of three automated metrics, but no JPEG XS, ProRes or VC-2 publication reachable in the sweep
states a VMAF threshold. Every published quality claim is PSNR-based or flicker-based. GNC's
insistence on VMAF is good engineering practice but is not this market's currency; the currency is
**expert viewing at the third generation**.

---

## 4. Where GNC actually stands

Measured 2026-09-05 unless a subsection says otherwise. Details and raw numbers in
[RESEARCH_LOG.md](../RESEARCH_LOG.md).

### Compression against H.264: 1.9x at contribution quality, and the gap is intra

**Updated 2026-09-06 (QUAL-1). This supersedes the 5–7x figure used throughout earlier drafts of
this document.** Re-running the same harness at the operating point §1 says GNC is *for* —
rather than at distribution bitrates, where the earlier figure was taken and where the quality
ladder above q=92 was dead — gives **+90.5% BD-rate on PSNR**, about **1.9x**: +129.0% on bbb,
+71.9% on old_town, +70.6% on crowd_run. Nothing in the coder changed between the two
measurements.

Three consequences for this document's argument:

- **The "no published transform result exceeds ~15%, so nothing closes a 5–7x gap" reasoning still
  holds arithmetically but no longer bites as hard.** Against 1.9x, single-digit and low-double-digit
  improvements are worth having. §7's dead-end entry is corrected accordingly.
- **The remaining gap is intra, not inter.** x264's own inter saving at this quality is +0.6% at
  crf 12 and −33.3% at crf 2 — near lossless the motion-compensated residual is noise-like, so it
  costs about what the picture costs and the vectors are overhead. Both codecs break even here.
  Inter is a distribution-bitrate problem, and GNC is not a distribution codec.
- **Colour is a genuine lead, and luma-only metrics hide it.** At rate matched to 1%, GNC beats
  x264 on CIEDE2000 on all three sequences (0.611 vs 0.684 mean on bbb) with fewer pixels past the
  JND, while sitting 7.4–8.8 dB behind on luma. For a codec sold on generation survival and colour
  fidelity that is the more relevant half, and no VMAF-based comparison can see it.

**And do not quote a VMAF BD-rate above about q=85.** Widening the quality ladder moved the VMAF
figure by a mean of 47.5 points (old_town +81.1% → +191.4%) while PSNR moved 1.0 point. §3's note
that VMAF is barely present in the contribution literature turns out to be the right instinct for
a second, purely numerical reason.

### Multi-generation robustness: the falsification test does not falsify

The cheapest experiment that could have killed the whole positioning was the EBU TR 091
multi-generation test, because GNC has a fixed 256×256 tile grid and TR 091 deliberately shifts
the picture between generations. If tile-boundary artefacts accumulate, the positioning fails
regardless of anything else.

Encode → decode → pixel-shift → re-encode, 5 generations at q=75 (~6:1), with the same shifts
applied to an uncoded reference chain so only codec degradation is measured:

| sequence | gen 1 | gen 3 | gen 5 | Δ VMAF | bitrate |
|---|---|---|---|---|---|
| bbb | 96.51 | 95.33 | 94.05 | −2.46 | flat |
| touchdown | 96.32 | 94.72 | 92.90 | −3.43 | flat |
| blue_sky | 96.85 | 94.84 | 90.64 | −6.21 | flat |

**No breakdown point, no cliff, no tile-grid catastrophe.** Degradation is smooth at −0.6 to −1.5
VMAF per generation and the bitrate stays flat. blue_sky — smooth gradients, where wavelet ringing
is least maskable — degrades over twice as fast as bbb and deserves a closer look.

This is the *necessary* condition for a contribution codec. It is not yet evidence of the
*sufficient* condition, which EBU decides by expert viewing rather than by VMAF.

### Speed: below real time, and we do not agree with ourselves about by how much

At BASELINE's own stated parameters (bbb, q=75, Rice, ki=8, 10 frames), this session measured:

| what is being timed | fps |
|---|---|
| `benchmark-sequence`, GPU encode phase only | 13.6 |
| `encode-sequence`, end to end incl. PNG decode and container write | 7.8 |
| BASELINE.md, stated | 31.7 |

The binary used was built at this session's start and HEAD has moved since, so this is not yet a
regression claim. But **three different numbers are in circulation for "GNC encode fps" and GOALS
quotes one of them without saying which** — and the CLI's own help text concedes that PNG input
inflates the cost. For a codec whose thesis is real-time density, that ambiguity is not
survivable. Pin the definition before any density claim rests on it.

Either way: contribution is 50 or 59.94 fps real time, concurrency multiplies throughput by about
two rather than by eight, and **the primary use case does not yet function on the reference
platform.**

### Format coverage

4:4:4, 4:2:2 and 4:2:0 all work. Bit depth is 8 on the video path; a `--bit-depth 10` flag exists
on the still-image path only.

---

## 5. The inter-coding question, and why the transform theory is wrong

GNC's inter coding measures far behind H.264. The standing structural hypothesis was that a
wavelet over a 256×256 tile destroys the spatial sparsity of a motion-compensated residual. The
literature says that hypothesis is **right physics in the wrong order of magnitude.**

**Every published transform-choice effect on motion-compensated residuals sits in the 5–15% band.**
Kamisli & Lim's 1-D directional transforms — whose statistical model is precisely GNC's hypothesis,
that residuals are locally 1-D structures clustered at motion boundaries — measure 4.1% to 11.4%
BD-rate. OBMC measures 1–4% in mature DCT codecs. Secondary transforms in AV2 measure 1.8%.
GNC's measured gap is 400–600%. **These are not the same kind of quantity.**

Two corrections to previously held beliefs:

**Dirac did not use MCTF.** Three independent sources (BBC's own EBU Technical Review, the Dirac
survey chapter, MultimediaWiki) agree that shipped Dirac was a conventional closed-loop hybrid:
hierarchical motion estimation, OBMC, wavelet on the *residual*, RDO quantisation, arithmetic
coding. **That is exactly GNC's architecture.** With a considerably more mature encoder, Dirac
landed at roughly H.264-class rather than multiples behind. GNC is therefore not at an
architectural ceiling — it is roughly 2–3× worse than its own architecture's demonstrated
capability.

**MCTF is settled against, at standards level.** Twelve of the fourteen SVC proposals in 2004 were
3-D wavelet codecs; none overcame the AVC-based approach. Schwarz, Marpe & Wiegand's ICME 2006
head-to-head concluded that *"MCTF does not improve the coding efficiency, mainly because the
open-loop coder control of an MCTF encoder cannot compensate for quantization errors of the
reference pictures."* MPEG subsequently **deleted** the temporal update step from SVC, citing both
lower complexity *and* improved coding efficiency. The best modern revisit — a learned wavelet
MCTF codec with neural networks doing the lifting, motion and transforms — reaches roughly
VVC-low-delay, about 18 BD-rate points behind VVC random-access, and goes *worse than its anchor*
at 4K.

**Consequence for a project rule.** CLAUDE.md's hard rule *"temporal lifting operates on spatial
wavelet subbands, never raw pixels"* is, in the literature's terms, in-band MCTF (2D+t). That
ordering is strictly worse for compression than t+2D and is structurally awkward on a
critically-sampled wavelet because of shift variance; the standard remedy is a redundant
overcomplete transform. If the rule exists for GPU-dataflow reasons, CLAUDE.md should say so —
on compression grounds it is backwards.

### What is missing — corrected 2026-09-06

An earlier version of this section claimed the missing machinery was rate-distortion decisions,
reasoning from published magnitudes (RDOQ is worth 6–8% in HEVC, deblocking up to 9% in H.264).
**That claim was wrong, and this repo had already measured it.** It is recorded here rather than
quietly deleted, because the mistake is instructive: generic published magnitudes were weighted
above this project's own specific negatives.

| already measured here | result |
|---|---|
| Coefficient-level RDOQ (per coefficient, D + λR, zero among the candidates) | **+0.1%** |
| Per-tile RD bit allocation (equal-slope quantiser step per tile) | **0.00 dB at every rate** |
| Energy-based tile skip, P and B paths | **dominated by simply raising q** — −15% rate at VMAF 92.37 where the q-curve gives 94.1 at the same rate |

The RDOQ entry also supplies a mechanism that generalises: Rice+ZRL over 256 interleaved streams
has far weaker inter-coefficient dependence than x264's run-length, context-coded blocks, so there
is no "cheap to drop" structure for an RD decision to exploit. And the tile-skip result is the
direct refutation — an RD criterion would choose *which* tiles to zero, not change the
granularity, and **granularity is what the measurements say is binding.**

**What is actually binding is an architectural coupling, and it is measured end to end:**

> 256 independent entropy streams per tile → ~290 B fixed per-tile header → smaller tiles cost
> +70% → the smallest region that can decline to be coded is 256×256 → almost every tile in real
> content contains something → almost nothing skips (0–3% of tiles at q=75).

The design choice that makes GNC decode in parallel is the same one that blocks fine-grained skip.

**But that coupling is not a ceiling.** Dirac shipped this exact architecture and landed at roughly
H.264-class. Whatever separates GNC from Dirac is not the shape of the pipeline.

**Honest state: unexplained after exhausting the locally available levers.** Multi-reference,
sub-pel filters, motion search, context entropy, block transforms, sub-block masking, smaller
tiles, dead zone, QP scaling, coefficient RDOQ, per-tile allocation and tile skip have all been
measured and rejected. That is a legitimate result and it is recorded as one.

The one untested lever with a mechanism specific to *this* weakness is **OBMC**. A block-edge step
is cheap for a DCT — it lands on a transform boundary — and expensive inside a 256×256 CDF 9/7
tile, where it lights up coefficients at every scale. Dirac adopted OBMC for exactly that reason,
and it is patent-clear. Published at 1–4% in DCT codecs; plausibly more here, but that is
inference, not a measurement.

Supporting measurements, 2026-09-05:

- **The fixed-cost floor is not the problem.** Tile headers are 3–16% of an inter frame on real
  content; coefficients are 84–97%.
- **Almost nothing skips.** 0–3% of tiles reach all-skip on real content at q=75.
- **B-frames stop paying at contribution quality.** BD-rate of the B-pyramid against P-only:
  −37% on animation but **+7% to +31% on all three camera sequences at the high-quality end** —
  while the P-only configuration was handicapped by two extra I-frames. On byte-identical frames
  the B path costs 16.7× what P-frames cost, and 34% more than not coding inter frames at all.

Published values for the missing machinery: coefficient-level RDOQ is worth 6–8% BD-rate in HEVC;
H.264's in-loop deblocking up to 9%; HEVC's SAO 3.5%. x264 ships a *decimation* rule — zero a
block whose only non-zero levels are ±1 separated by long zero runs — and a deliberately wider
dead zone for inter than intra (21 vs 11 of 32). GNC's B-frame tiles are exactly the state that
decimation rule was designed for.

---

## 6. Which differentiators are real

**Real:**

- **Portable GPU-compute implementation.** Verified: no commercial cross-vendor compute-shader
  mezzanine codec exists. Every credible alternative is CUDA-locked, FPGA, or CPU.
- **No per-instance and no per-service-hour royalty, and no cloud-service exclusion.** The JPEG XS
  patent pool's published terms explicitly do not license products sold to customers who offer
  public cloud JPEG XS services or who rent JPEG XS devices. This is the single most concrete,
  defensible commercial argument GNC has.
- **Scaling with GPU size rather than fixed-function block count** — sound in principle, unmeasured.
- **In-browser decode of GNC's own bitstream** — true by construction; WebCodecs will never expose
  a proprietary wavelet format.

**Weak or overstated:**

- **"Patent-free" as a headline.** JPEG 2000 and VC-2 already are, and neither is winning. And
  GNC's legal posture would be identical to VC-2's: SMPTE's statement for VC-2 is an
  *absence-of-notice*, not an affirmative royalty-free grant, with no indemnity and no pool. The
  specific claim about cloud and rental is worth far more than the general one.
- **Per-tile CRC and graceful degradation** — see §3.
- **Browser decode as a differentiator.** Real capability, weak differentiation: the browser
  monitoring market solved its problem with WebRTC and optimises for latency tiers, not pixel
  fidelity. A custom browser decoder is the *cost* of a custom bitstream, not a benefit of it.
- **"H.264-class quality" as a phrase** — wrong currency, see §3.

**The nearest competitor is not a product.** FFmpeg 8.1 ships real video codecs implemented in
general Vulkan compute shaders, and an unmerged **Vulkan VC-2 encoder** patch exists on
ffmpeg-devel. That is a patent-free wavelet mezzanine codec on cross-vendor GPU compute, being
built by people with distribution GNC does not have.

---

## 7. What to do

In priority order. Items map to [BACKLOG.md](../BACKLOG.md).

1. **Finish MEAS-5 on a discrete card.** Claim A is settled and defensible. Claim B needs a head
   to head against NVENC at both P1 and P7 presets — P7 sits nearer GNC's quality target and is
   roughly four times easier to win — and then on an H100, where the NVENC column is a zero.
   **Pin the fps definition first**; three numbers are currently in circulation (§4).
2. **Spend optimisation effort on entropy coding, not the wavelet.** Entropy coding is 51–85% of
   runtime in every GPU wavelet codec measured, and it is local-memory-latency bound, where
   register footprint per thread is the lever. Rate control, not the transform, is also what sank
   the historical GPU encoders' quality — scrutinise it as hard as the DWT.
3. **10-bit through the video path** (FMT-1). A gate, not a feature. Cheaper than assumed: the
   still-image path already has the flag; `encode-sequence` has no bit-depth option at all.
4. **Stop working on inter at this operating point** (§4, §5, corrected 2026-09-06). Not because
   the gap is unexplained — because at contribution quality there is no inter gap to close, for
   GNC or for x264. Every locally available inter lever was measured and rejected, RD decisions
   included, and then the operating point turned out to be the whole story. OBMC remains the one
   untested wavelet-specific lever; it is now a low priority rather than the last hope. **Spend
   the effort on intra**, which is where the +90.5% lives.
5. **Decide between live contribution and cloud mezzanine, and stop straddling** (§2).
6. **Drop the B-pyramid at contribution quality.** Two independent measurements now agree it is
   the wrong default here: it costs 7–31% in rate on camera content (BUG-5) and 160 ms in latency
   (MEAS-6). A configuration change, not a bitstream change. Keep P-frames — they have zero
   reordering delay and were the better performer.
7. **Sharpen the licensing claim** from "patent-free" to the specific documented one, and back it
   with a patent search and a defensive publication.

### Dead ends — do not attempt

- **MCTF / temporal lifting as a compression win.** Settled at standards level; GNC is already on
  the winning branch and moving to MCTF is a step backwards.
- **In-band MCTF / 2D+t** specifically. Strictly worse than t+2D for compression and structurally
  broken on a critically-sampled wavelet.
- **More or better motion**: multi-reference, better sub-pel filters, better search, more
  hypotheses. Already ruled out by GNC's own measurements and now explained by Girod's bounds —
  half-pel already captures roughly 1.8 of the ~2 bits/sample available.
- **Smaller tiles.** Measured at +70% bits, and the header floor is not where the gap is.
- **Any expectation that a *single* transform change closes the gap.** No published transform
  result on motion-compensated residuals exceeds ~15%. Note the target moved on 2026-09-06: the
  gap at contribution quality is 1.9x, not 5–7x (§4), so an accumulation of single-digit wins is a
  credible route where against 5–7x it was not. What remains a dead end is expecting one of them to
  do it alone.
- **Blaming low latency for the inter gap** — and, since 2026-09-06, **treating the inter gap as
  the priority at all.** At contribution quality both codecs break even on inter; x264's own inter
  saving is +0.6% at crf 12. The 17–27% versus 40–55% comparison was drawn at distribution
  bitrates and does not describe this operating point. The gap here is intra.

---

## 8. Sources

**Standards and industry**
[EBU TR 092 — Tests of Video Processing and Encoding for Contribution (Oct 2025)](https://tech.ebu.ch/docs/techreports/tr092.pdf) ·
[EBU TR 091 — test plan (Aug 2025)](https://tech.ebu.ch/publications/tr091) ·
[EBU R 153 — Parameters for Live Contribution of UHD/HDR](https://tech.ebu.ch/docs/r/r153.pdf) ·
[SMPTE ST 2042-2:2009 (VC-2)](https://pub.smpte.org/doc/st2042-2/20091102-pub/st2042-2-2009.pdf) ·
[RFC 9134 — JPEG XS RTP payload](https://www.rfc-editor.org/rfc/rfc9134.txt) ·
[JPEG XS Patent Portfolio Licence, Overview of Terms v5.0 (Apr 2025)](https://static1.squarespace.com/static/620fe5f759d6a229a9f1e39f/t/67efde16a1cb1320a6f8db3a/1743773207321/JPEG+XS+PPL+-+Overview+of+Terms+-+v5.0+-+01+Apr+2025.pdf) ·
[Apple ProRes White Paper](https://www.apple.com/final-cut-pro/docs/Apple_ProRes.pdf)

**Wavelet and MCTF**
[Schwarz, Marpe & Wiegand — Analysis of Hierarchical B Pictures and MCTF, ICME 2006](https://www.cecs.uci.edu/~papers/icme06/pdfs/0001929.pdf) ·
[Schwarz, Marpe & Wiegand — SVC Overview, IEEE TCSVT 2007](https://eeweb.engineering.nyu.edu/~yao/EL6123old/Schwarz_SVC_CSVT2007.pdf) ·
[Borer & Davies — Dirac, EBU Technical Review 303](https://tech.ebu.ch/docs/techreview/trev_303-borer.pdf) ·
[Learned Wavelet Video Coding using MCTF (arXiv:2305.16211)](https://arxiv.org/abs/2305.16211) ·
[Variable Rate Learned Wavelet Video Coding (arXiv:2410.15873)](https://arxiv.org/html/2410.15873)

**Residual statistics, transforms, RD**
[Kamisli & Lim — 1-D Transforms for the Motion Compensation Residual](https://dspace.mit.edu/bitstream/handle/1721.1/79671/1-D%20Transforms%20for%20the%20motion%20Compensation%20Residual%20-%20jrnl_reviewed_and_submitted.pdf) ·
[Girod — Efficiency Analysis of Multihypothesis MCP, IEEE TIP 2000](https://web.stanford.edu/~bgirod/pdfs/Girod_Multihyp_Feb2000.pdf) ·
[List, Joch et al. — Adaptive deblocking filter, IEEE TCSVT 2003](https://www.semanticscholar.org/paper/Adaptive-deblocking-filter-List-Joch/845ebc98698772e7fa05570fd45be05a41ffbc95) ·
[Fast RDOQ for HEVC](https://ieeexplore.ieee.org/iel7/8014728/8049747/08050460.pdf) ·
[x264 source — analyse.c](https://raw.githubusercontent.com/mirror/x264/master/encoder/analyse.c),
[quant.c](https://raw.githubusercontent.com/mirror/x264/master/common/quant.c),
[tables.c](https://raw.githubusercontent.com/mirror/x264/master/common/tables.c)

**GPU and portability**
[Khronos — Video encoding and decoding with Vulkan compute shaders in FFmpeg](https://www.khronos.org/blog/video-encoding-and-decoding-with-vulkan-compute-shaders-in-ffmpeg) ·
[Compeg — WebGPU compute-shader JPEG decoder](https://github.com/SludgePhD/Compeg) ·
[SVT-JPEG-XS](https://github.com/OpenVisualCloud/SVT-JPEG-XS) ·
[intoPIX FastTicoXS SDKs](https://www.intopix.com/fasttico-xs-sdks) ·
[Fastvideo GPU JPEG2000](https://www.fastcompression.com/products/gpu-jpeg2000.htm)
