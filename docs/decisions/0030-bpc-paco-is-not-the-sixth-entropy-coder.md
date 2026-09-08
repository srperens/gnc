# 0030 — BPC-PaCo is not GNC's sixth entropy coder; three of its ideas go into abac

**Date:** 2026-09-08
**Status:** accepted
**Item:** BACKLOG ENT-7 (P2), steps 2 and 3. Step 1 (BUG-31's WGSL fix) is untouched and stays a
separate item.
**Supersedes nothing. Redirects ENT-7's part 2 into ENT-6 and a new ENT-8.**

## The question

ENT-7 asked whether GNC should stop patching abac's shaders and replace the coder with BPC-PaCo —
bitplane coding with parallel coefficient processing (Aulí-Llinàs, Enfedaque, Moure, Sanchez,
*IEEE TIP* 25(1):209–219, 2016; GPU implementation Enfedaque, Aulí-Llinàs, Moure, *IEEE TPDS*
28(8):2272–2284, 2017). It differs from abac in the two places that have cost this project: a
stationary probability model instead of per-symbol adaptation, and parallelism *inside* a code
block rather than one thread per block.

The item set its own criteria before any code was written: **rate within +2% of abac** at
identical decoded pixels, **decode ≥ 1.3x** abac's throughput, and — for preferring a sixth
backend over folding the mechanism into abac — **≥3% of total rate beyond what abac + ENT-6
reach**.

## What was measured

`src/encoder/bpc_paco_diag.rs`, a seventh model in the `GNC_COEF_ENTROPY=1` harness, on the same
four stills, the same parameters and the same shipped abac tiles decision `0024` used. Full
numbers and the per-band decomposition are in RESEARCH_LOG, "ENT-7 steps 2–3". The mean over four
images, as a percentage of what abac's bitstream actually spent:

| | q=85 | q=90 |
|---|---|---|
| BPC-PaCo's model, oracle table trained on the image being coded | **−7.76%** | **−7.87%** |
| the same, table trained on the other three images (leave-one-out) | −1.96% | −2.20% |
| **the same, plus its 32 fixed-length codeword streams per block** | **+2.81%** | **+1.85%** |
| dropping the cross-lane neighbour exchange (relative to the first row) | +9.35% | +8.63% |

## The decision

**Keep abac. Do not add BPC-PaCo as a backend.** Three reasons, in the order they bind:

1. **The rate criterion fails.** A complete BPC-PaCo is +2.81% at q=85 and +1.85% at q=90 against
   the +2% bar, and on bbb — the sequence GNC regression-tests against — it is +6.21% / +4.07%.
   The criterion was written expecting the *model* to lose; the model in fact wins by 7.9%, and
   the loss is entirely in how the model's output is collected.
2. **The throughput half's mechanism does not port, and not for the reason the item assumed.**
   The WebGPU limits do **not** bind. BPC-PaCo's released CUDA uses **20 bytes** of shared memory
   per 128-thread block and 3 storage buffers, and its whole probability model is **2.6 KB** of
   lookup table — 190 (level, subband, bitplane) triples × 14 contexts, counted from the released
   `LUTs/` — which belongs in a uniform buffer, not in workgroup storage. Two other things bind:

   - **512 bytes of dynamically-indexed private state per invocation** (128 coefficients × u32,
     in registers). This is the authors' own documented bottleneck and they did not solve it:
     coding a 4K frame reads **251 MB** for 32 MB of data, an **8× amplification**, in CUDA with a
     hand-tuned register cap. WGSL offers no better control, and holding the block in workgroup
     storage instead measured **3.5–9× slower** in their own thesis — separately impossible here,
     since 4096 × 4 B is 16 384 B, the entire budget.
   - **Two techniques need cross-lane operations, and the hot one is hot:** the cross-stripe
     neighbour exchange runs on the order of 1000 times per thread per code-block. The authors
     measured substituting shared memory for `__shfl` at **~20%** on their *DWT* kernel, which is
     far less shuffle-dense and pays no barriers, so 20% is a floor rather than an estimate.

   **And subgroups would not rescue it even where they exist**, which is the part worth recording.
   The codeword-slot reservation is `__ballot` + `__popc`, and it is not an optimisation — it *is*
   the bitstream ordering rule (TIP 2016 §III-C: left stripes take priority, for determinism).
   WGSL §15.5 states there is **no defined relationship** between `subgroup_invocation_id` and
   `local_invocation_index`, and subgroup size is only guaranteed to be a power of two in
   [4, 128] chosen by the device compiler. A ballot-based port therefore produces a
   **device-dependent bitstream**, which for a coder whose gate is byte-exactness is a
   correctness failure, not a portability preference. The deterministic
   `atomicOr` + `countOneBits` emulation is the right implementation, and it is the one that
   spends the prize. The alternative — no exchange at all — costs **8.6% of rate**, measured
   above.
3. **What does pay is not BPC-PaCo's, it is abac's defect.** The 7.9% model gap is concentrated
   exactly where `0024` and ENT-6 found abac's cold start: `Y LL` −50.4%, the sub-64px bands −12%
   to −29%, the full 64×64 blocks only −4% to −11%. An adaptive coder given the same table as a
   **prior** collects that without paying the 5.7 points stationarity costs or the 4.0 points the
   multi-codeword bitstream costs.

**Three transplants are adopted instead:**

- **Stationary per-bitplane, per-subband initial probabilities for abac's contexts** → ENT-6
  candidate 3, now sized rather than speculated. Chroma gets its own tables or none: the
  leave-one-out misses run to 2.44% on Co against ≤0.31% on Y, and every band where the trained
  table came out worse than abac is a chroma level-1 band.
- **The two-column lockstep scan** → filed as **ENT-8**. This is the finding the item did not
  expect and the most valuable thing in the papers: the scan achieves the *same* average number of
  coded neighbours as a raster scan (4) while making every stripe independent, so abac's one
  thread per code-block could become one thread per stripe — 32x more parallelism inside a block —
  by scheduling alone.
- **Not** the fixed-length multi-codeword coder.

## Facts about the artefact, since "port the reference code" was on the table

- **The reference implementation cannot be used at all.** `PabloEnfedaque/CUDA_BPC-PaCo` has **no
  licence file and no copyright header in any source file**; one commit, 2016-11-01,
  self-described as a proof of concept. No licence means no permission, so the papers are the only
  usable source. The authors' JPEG 2000 framework BOI is GPL up to 1.8 and thereafter under a
  licence forbidding commercial use and redistribution. The end-to-end GPU codec of *IEEE Access*
  2020 has no public release.
- **No patent naming BPC-PaCo, its authors or the coefficient-parallel scheme was found.** Searches
  on the three inventors, on "BPC-PaCo", on "parallel coefficient processing" and on the
  university as assignee return nothing relevant. The one topically adjacent filing is
  **US7760948B1, "Parallel coefficient bit modeling", Xilinx, filed 2006-10-13**, different
  inventors. Recorded as a search result, not as a legal opinion, and not as a reason for this
  decision — the rate number is.
- **No independent measurement of BPC-PaCo exists.** Every rate and throughput figure in
  circulation traces to the same group, eight of the GPU paper's nineteen citers are
  self-citations, and no GPU-BPC-PaCo vs GPU-HTJ2K comparison has ever been run by anyone. That is
  a reason to have measured it here rather than to have trusted it, which is what happened.
- **The one substantive external critique lands on us too.** Rossinelli et al. (*IEEE TMI* 40(2),
  2021) object that the 25x over Kakadu confounds the algorithm change with the CPU→GPU move —
  "it remains unclear why the authors did not compare against the CPU implementation of BPC-PaCo".
  **Any GNC throughput claim that compares coders across substrates has the same hole.** abac's
  figures are stated against Rice on the same GPU for exactly this reason; keep it that way.

## What was not chosen, and what it would have cost

- **Implementing BPC-PaCo behind a flag and measuring it for real.** The honest cost: a GP version,
  an entropy type, a CPU reference coder, GPU encode *and* decode shaders, a byte-exactness gate
  over the full artefact set (abac's standard is 98 of 98 whole files), and a permanent maintenance
  surface across the stills and sequence command families. Rejected on a measurement that says the
  best case is +1.85% of rate, not on the estimate of the work.
- **Implementing the 2023 adaptive sliding window instead of the 2016 LUT.** The authors
  themselves retired the stationary model: a 14-context sliding window (W = 256, updated once per
  32 coefficients) beats both JPEG 2000 and HTJ2K at medium and high rates for ~10% more compute
  (*SPIC* 112:116914, 2023). That is the version worth having — and "one update per 32
  coefficients" is a description of abac with a coarser adaptation schedule, which is why it lands
  in ENT-6 and ENT-8 rather than in a new coder. **If ENT-6 and ENT-8 both pay and someone still
  wants the backend, this paper, not the 2016 one, is the specification to build from.**
- **Truncatable embedded streams**, which BPC-PaCo keeps: 0.00 dB here at every rate from 0.05 to
  3.5 bpp (EBCOT part 1), for a structural reason. Unchanged by anything measured here.
- **Training the stationary table on more than four images.** The leave-one-out penalty of 5.7
  points is partly small-training-set error, and a larger corpus would shrink it. Not done because
  it cannot change the decision: even at *zero* stationarity cost the multi-codeword stream leaves
  BPC-PaCo at −3.8% model gap minus 4.0 points of codewords, i.e. inside noise of abac, and the
  throughput argument in reason 2 is untouched.
- **A VMAF or BD-rate figure.** Neither applies. abac and BPC-PaCo are lossless recodings of the
  same coefficients, so the decoded pixels are bit-identical and the only quantity is bytes;
  and above q≈85 a VMAF BD-rate is not a weak number but not a number (QUAL-1).

## Caveats a later reader needs

- **These are model costs, not coder costs, on both sides.** `Hbpc`/`Hbpcf` do not include
  BPC-PaCo's own arithmetic rounding beyond the codeword excess, so its honest figure is slightly
  worse than +1.85%. abac's side is measured against what it *shipped*, which does include its
  rounding and its per-block length fields — a fair comparison, and one that costs abac 0.1% at
  bbb q=90 and 2.4% at kristensara q=85.
- **The codeword excess is a stated mechanism with an assumed constant**, `⌈w/2⌉ × W/2` bits per
  block at W = 16. It is the paper's mechanism (TIP 2016 §III-C) and its own ablation puts the
  penalty in the same place, but half a codeword per stripe is an expectation, not a measurement.
  It is printed as its own column so any reader can substitute a different assumption.
- **Four images is a small training corpus** and the stationarity penalty is the number most
  likely to move. See "what was not chosen".
- **This says nothing about inter.** Every figure here is intra, on stills. abac's inter result is
  `0025`'s.
- **Step 1 is still open, and it now has a sibling.** GNC requests
  `max_storage_buffers_per_shader_stage: 10` at `src/lib.rs:1431` against
  `wgpu::Limits::default()`'s **8** — verified in `wgpu-types-24.0.0`. That contradicts CLAUDE.md's
  "GNC asks for wgpu's default limits, not the hardware's", and it is the same class of defect as
  BUG-31 on a limit nobody was watching. Filed as **BUG-34**, found while checking one of this
  brief's claims rather than by looking for it. BUG-31 — `abac_encode.wgsl` and `abac_decode.wgsl` declaring 18 688 B
  against a 16 384 B device — is untouched by this decision and is the half of ENT-7 that has a
  reachable failure on a shipped target.
