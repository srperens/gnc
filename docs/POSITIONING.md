# GNC — Positioning, Market Reality, and What To Do About It

**Status:** current as of 2026-09-05. Supersedes nothing; it is the reasoning behind
[GOALS.md](../GOALS.md) §1 and the priority order in [BACKLOG.md](../BACKLOG.md).

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

**That claim is unproven. It is the single most important thing to measure** ([BACKLOG.md](../BACKLOG.md)
MEAS-5). Everything below is downstream of it.

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

**GNC's 256×256 tiles impose a structural floor of ~256 lines** before the first tile row can be
emitted — roughly an order of magnitude coarser than JPEG XS. This is an architectural fact, not
a tuning problem, and it is a second reason live contribution is the harder target.

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

Everything in this section was measured on 2026-09-05. Details and raw numbers in
[RESEARCH_LOG.md](../RESEARCH_LOG.md).

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

### Speed: below real time for the use case

31.7 fps sequence encode at 1080p. Contribution is 50 or 59.94 fps real time. **The primary use
case does not yet function on the reference platform.**

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

### What is actually missing: rate-distortion decisions

Two independent literature sweeps converged on the same answer, and GNC's own measurements agree.

**GNC makes no rate-distortion decision anywhere.** It quantises at the configured step and codes
whatever comes out. Dirac had RDO mode decision. Every hybrid codec's inter advantage comes from
most of a P-frame costing approximately nothing — skip flags, coded-block-pattern hierarchies,
end-of-block. GNC has only the degenerate case: a tile that *happened* to quantise to all zeros.

Measured support, 2026-09-05:

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

1. **Measure concurrent streams per GPU against NVENC** (MEAS-5). The entire strategic argument
   rests on this and it has never been measured. Per-stream, compute will lose to fixed function —
   that is expected and is not the claim. The claim is about the aggregate. If the aggregate also
   loses, the positioning is wrong and it is better to know now than after another quarter.
2. **10-bit through the video path** (FMT-1). A gate, not a feature. Cheaper than assumed: the
   still-image path already has the flag; `encode-sequence` has no bit-depth option at all.
3. **Add a rate-distortion decision** (BUG-5 and successors). Minimal viable version: per tile,
   compare the Lagrangian cost of coding the residual against zeroing it, take the cheaper. No
   bitstream change. Then the same decision at code-block granularity, which needs one. x264's
   decimation rule and an inter-specific dead zone are cheap approximations worth measuring first.
   Note this is **pricing**, not masking — the earlier rejected experiment masked coefficients in
   the wavelet domain, which changes the signal and causes ringing. Different thing.
4. **Decide between live contribution and cloud mezzanine, and stop straddling** (§2).
5. **Measure latency** (MEAS-6), and confront the ~256-line tile floor against the target segment's
   budget.
6. **Sharpen the licensing claim** from "patent-free" to the specific documented one, and back it
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
- **Any expectation that a transform change closes a 5–7× gap.** No published transform result on
  motion-compensated residuals exceeds ~15%.
- **Blaming low latency for the inter gap.** Low-delay costs perhaps 15–25% versus random access,
  not multiples. GNC currently saves 17–27% where a low-latency-adjusted target is 40–55%.

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
