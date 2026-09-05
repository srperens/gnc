# BUG-1 Diagnosis — 4:2:0 pyramid B-frame chroma collapse (layers 2–3)

**Date:** 2026-09-04. **Method:** code-reading root-cause analysis (Researcher role); no GPU
available in the analysis environment, so all candidates come with concrete confirmation
diagnostics to run on the dev machine before implementing the fix.

**Symptom (from BACKLOG BUG-1):** With 4:2:0 + pyramid GOP, B-frames at layers 2–3 show
~22–26 dB PSNR despite 2–4 bpp. I/P and B₄ (layer 1) are fine. 4:4:4 pyramid is fine.

## Domain declaration

The primary defect operates in the **MV field domain** (the scaled chroma motion-vector
buffer used for prediction), not in the spatial residual, wavelet, or bitstream domains.
Secondary candidates operate in the **reference buffer domain** (which frame sits in the
bwd reference at prediction time).

## The B₂ chroma reference chain in 4:2:0, step by step

**Encoder** (all in `src/encoder/sequence.rs` unless noted):

1. I₀ encoded; its reconstructed chroma is stored **NN-upsampled to luma dims** in
   `gpu_ref_planes[1..2]`; saved to pyramid slot 3 (line 632).
2. B₄ encoded as forward-only P-frame via `encode_pframe` (lines 645–661) — split-grid MVs
   (`block_size: ME_SPLIT_BLOCK_SIZE`, 32640 entries at 1080p). Reconstructed B₄ →
   `gpu_ref_planes` → pyramid slot 0 (line 664). P₈ encoded against B₄, saved to slot 4 (line 820).
3. B₂ refs loaded: I₀ → fwd (line 822), B₄ → bwd (line 823).
4. `encode_bframe` (line 5082): bidir ME on luma produces **fresh 8160-entry** fwd/bwd MV
   buffers (`motion.rs:1261–1275`, one MV per 16×16 block) and `bidir_modes_scratch`
   (8160 u32, `encoder/buffer_cache.rs:663–668`).
5. 4:2:0 MV scaling (lines 5307–5339): `dispatch_mv_scale` is dispatched with
   **`split_total_blocks` = 32640** threads reading from the **8160-entry** MV buffer.
   Entries 8160..32640 of `mv_chroma_buf`/`mv_chroma_buf_bwd` are produced by
   **out-of-bounds reads** (whatever naga's bounds-check policy yields — zero or
   clamped-to-last — deterministic, but *never* real motion).
6. Chroma forward pass (lines 5368–5480): box-filter current + both refs to chroma dims,
   `compensate_bidir_chroma_cached` computes the residual at chroma dims using
   `mv_chroma_buf`, `mv_chroma_buf_bwd`, `bidir_modes_scratch`; wavelet + quantize at chroma dims.
7. Local decode to pyramid slot 1 (`local_decode_bframe_to_pyramid_slot`, lines 6165–6528;
   4:2:0 chroma branch 6223–6349): dequant → inverse wavelet → box-filter fwd/bwd refs →
   inverse chroma bidir MC **with the same `mv_chroma_buf` contents** → NN-upsample → slot,
   then **deblock** (line 6342). The encoder is self-consistent: its pyramid slot equals
   what it predicted from.
8. Bitstream stores only the **8160** fwd/bwd MVs + modes (`me_total_blocks` readback,
   lines 5624–5645, 5774–5782; `MotionField { block_size: ME_BLOCK_SIZE }` lines 5829–5836).

**Decoder** (`src/decoder/`):

1. Decode order I₀, B₄, P₈, B₂ … B₄ decoded via the P-frame path (`gpu_work.rs:72–77`),
   I₀ → `pyramid_ref_planes[4]` (`pipeline.rs:725`), B₄ → `pyramid_ref_planes[0]`
   (`pipeline.rs:732`), P₈ → `pyramid_ref_planes[3]` (`pipeline.rs:774`).
2. B₂: fwd_pool=0 → pyr[4]=I₀, bwd_pool=2 → pyr[0]=B₄ (`pipeline.rs:809–822`). The
   mapping is consistent with the encoder's slots.
3. `frame_data.rs:435–447`: B₂'s **8160** MV pairs are written into the persistent
   `mv_buf`. Crucially, `mv_buf` was **grown to ≥32640 entries** when B₄/P₈ (split MVs)
   were decoded (`ensure_var_buf`, `gpu_util.rs:6–24` — grows, never shrinks, never
   clears) — so entries 8160..32640 still hold **P₈'s (or B₄'s) split MVs**.
4. `gpu_work.rs:80–109`: `dispatch_mv_scale` with `split_total_blocks` reads `mv_buf`
   entries 0..32640 — the tail reads are **in-bounds stale P-frame MVs**, scaled into
   `mv_chroma_buf`.
5. Chroma bidir MC (`gpu_work.rs:450–513`) at chroma dims: the shader
   (`shaders/motion_compensate_bidir_chroma.wgsl:94–102`) indexes
   `block_idx = by * blocks_x + bx` over the **full 4×4-chroma-block grid = 32640 blocks**
   (blocks_x = chroma_w/4 = padded_w/8, `motion.rs:1429`). Every chroma block with
   `block_idx ≥ 8160` — spatially the **bottom ~75 % of the chroma plane** — uses tail MVs.
6. Decoded B₂ (chroma NN-upsampled) → pyr[1] (`pipeline.rs:838`), from which B₁/B₃ chroma
   is then predicted → cascade.

## Ranked root-cause candidates

### 1. Encoder/decoder mismatch in the scaled chroma-MV tail for true bidirectional B-frames — **HIGH confidence** — *MV field domain*

- **Where:** encoder `sequence.rs:5307–5339` (OOB reads from fresh 8160-entry
  `fwd_mv_buf`/`bwd_mv_buf`, `motion.rs:1261–1275`) vs decoder `gpu_work.rs:88–109`
  reading stale in-bounds data from the grown, never-cleared `mv_buf`
  (`frame_data.rs:437–446`, `gpu_util.rs:14–23`; initial cap `decoder/buffer_cache.rs:369–377`).
- **Mechanism:** the encoder's own comment (`sequence.rs:5312–5318`) asserts the decoder
  "also … gets zeros for the out-of-bounds reads from mv_buf". That is **false** once any
  P-frame (or B₄-as-P, which carries 32640 split MVs) has been decoded: the decoder's
  reads are in-bounds and return the previous frame's split MVs. Decoder prediction ≠
  encoder prediction for the bottom 75 % of Co/Cg wherever P₈'s motion is nonzero; the
  coded residual then *adds* to a wrong prediction. Note the history: the older code
  (pre-"fix", see archived item #14) dispatched `me_total_blocks` and left the encoder's
  persistent `mv_chroma_buf` tail holding the last P-frame's scaled MVs — accidentally
  *matching* the decoder. The "fix" to `split_total_blocks` made the encoder tail
  deterministic-zero/clamped and thereby **created** the mismatch.
- **Why it matches the signature exactly:**
  - I/P frames: P chroma scales from `split_mv_buf` with all 32640 entries valid
    (`sequence.rs:3317–3327`, `4515`) — fine.
  - B₄: encoded/decoded via the P path with full split MVs — fine (~38 dB).
  - B₂/B₆: first frames that use the B-frame (8160-entry) MV field after a 32640-entry
    write — broken. Luma untouched (luma MC indexes the 16×16 grid, 8160 valid blocks)
    but corrupted Co/Cg destroys all three RGB channels after YCoCg inversion →
    ~22–26 dB overall.
  - Leaves: inherit corrupted pyr[1]/pyr[2] chroma **and** suffer their own tail
    mismatch — worst.
  - 4:4:4: no chroma MV scaling path exists — fine.
  - High bpp: the encoder's tail prediction (zero/clamped MV, OOB mode) is poor, so the
    chroma residual is large — bits are spent, then wasted against a different decoder
    prediction.
- **Confirming diagnostics (dev machine):**
  - (a) per-region PSNR of decoded B₂ chroma: top 25 % of the plane should be
    dramatically better than the bottom 75 % — a one-run smoking gun;
  - (b) read back `mv_chroma_buf` entries 8160..8200 after MV-scale in encoder B₂ encode
    and decoder B₂ decode and diff (expect zeros/constant vs scaled P₈ MVs);
  - (c) static-scene clip: P₈ MVs ≈ 0 → bug should nearly vanish; high-motion clip → severe;
  - (d) hot-patch test: zero-fill `mv_buf`/`bwd_mv_buf` entries
    `vectors.len()..split_total_blocks` in `frame_data.rs` before B-frame decode (and
    clear the encoder's `mv_chroma_buf` tail if naga's policy is clamp rather than zero)
    → B₂/B₆ should jump to ≈ B₄ level.

### 2. `block_modes` tail mismatch (same mechanism, modes array) — **MEDIUM confidence** — *MV field domain*

- **Where:** shader reads `block_modes[block_idx]` up to 32640
  (`motion_compensate_bidir_chroma.wgsl:102`); encoder binds `bidir_modes_scratch` sized
  exactly 8160×4 (`encoder/buffer_cache.rs:663–668`) → OOB; decoder binds
  `block_modes_buf` grown to `required*2` = 16320 entries (`gpu_util.rs:15`, upload
  `frame_data.rs:463–477`) → entries 8160..16320 in-bounds zeros, 16320..32640 OOB.
- **Why it matters:** if wgpu/naga's bounds policy is clamp ("Restrict"), the encoder's
  tail mode = mode of block 8159 while the decoder's is 0 (forward-only) — a second,
  independent prediction mismatch in the same region. If the policy is read-zero, this
  candidate is inert and #1 alone explains everything.
- **Diagnostic:** log the effective mode used at a tail block on both sides, or simply fix
  #1 with explicit tail zeroing of MVs *and* modes on both sides and re-measure.

### 3. B₇'s encoder backward reference is stale (B₆, not P₈) — **MEDIUM confidence** — *reference buffer domain*, all chroma formats

- **Where:** `sequence.rs:972–997`. `layer3_order` runs B₁(bwd=3→slot1), B₃(bwd=2→slot0),
  B₅(bwd=4→slot2 = B₆ → `gpu_bwd_ref_planes`), then B₇(bwd=1) hits the no-op arm at line
  992 whose comment "future_P already in gpu_bwd_ref_planes (restored for B₆)" is
  **wrong** — B₅'s setup overwrote bwd with B₆. Encoder predicts B₇ from bwd=B₆; decoder
  loads pyr[3]=P₈ (`pipeline.rs:817`).
- **Why it fits/doesn't fit:** format-independent, so it cannot explain the 4:2:0-specific
  B₂/B₆ collapse, and B₆↔P₈ are temporally adjacent so the error is moderate (plausibly
  unnoticed in the 4:4:4 validation of #42/#64). It does add degradation to one leaf frame
  (B₇) and, together with #4, corrupts the end-of-group state.
- **Fix direction:** `1 => self.copy_pyramid_slot_to_bwd_ref(ctx, 4, plane_size)`.
- **Diagnostic:** per-frame PSNR in 4:4:4 pyramid — B₇ should be measurably worse than
  B₁/B₃/B₅ today.

### 4. End-of-group restore is 4:4:4-only — next GOP's P-frame encoder reference is wrong in 4:2:0 — **MEDIUM confidence** — *reference buffer domain*

- **Where:** `sequence.rs:1074–1078`:
  `if pyramid_enabled && Yuv444 { copy slot4 → fwd } else { swap_ref_planes() }`. In 4:2:0
  pyramid, after B₇ the buffers are fwd=B₆ (loaded at line 988) and bwd=B₆ (stale, per
  candidate 3) — `swap_ref_planes` (line 6064) then leaves `gpu_ref_planes` = **B₆**, not
  decoded P₈. The next group's P₁₆ is encoded against B₆ while the decoder uses P₈
  (`pipeline.rs:846–847`).
- **Prediction to test:** in multi-GOP 4:2:0 runs, P₁₆ and everything after should also
  drift. If the reported "P-frames fine ~42 dB" includes P₁₆, re-check per-frame numbers —
  either the test was effectively single-GOP or this needs re-measurement. Even if
  currently masked, this is wrong by inspection once candidate 3 is understood.
- **Fix direction:** use the slot-4 restore unconditionally (it exists precisely because
  bwd gets clobbered).

### 5. Encoder-only reference deblocking — **LOW confidence as root cause; real but small systematic mismatch** — *reference buffer domain*

- **Where:** encoder deblocks every reference and pyramid slot in place
  (`pipeline.rs:1388–1398`, enabled by default unless `GNC_REF_DEBLOCK=0`; B-frame slots
  at `sequence.rs:6342/6428/6493`), decoder has **no deblocking at all** (no hits in
  `src/decoder/`). `pipeline_tests.rs:2294–2318` explicitly acknowledges refs differ when
  it's on.
- **Why not the bug:** it affects I/P and 4:4:4 identically, which are fine; it touches
  only tile-boundary segments. But it is a drift floor that compounds down the pyramid
  chroma chain. Run all confirmation experiments with `GNC_REF_DEBLOCK=0` to remove this
  noise.

## What could NOT be verified without running the code

- The actual naga/wgpu out-of-bounds read behavior on the dev machine (zero vs clamp) —
  it changes whether candidate 2 and the encoder tail value matter, but **not** candidate
  1's conclusion (the decoder tail is stale P-motion either way; wgpu buffers are
  zero-initialized only at creation).
- Actual buffer contents / per-frame PSNR splits — diagnostics (a)–(d) above confirm or
  falsify on the GPU machine in one run each.
- Whether flat (non-pyramid) 4:2:0 B-frames are also currently broken — candidate 1
  predicts **yes** (any B decoded after a P). If a flat 4:2:0 B-frame test passes today,
  that would falsify candidate 1 and the ranking must be redone; the pipeline test noted
  at `pipeline_tests.rs:2294` covers I-frames only, so it wouldn't catch this.
- Whether the reported P-frame numbers include a second GOP (bears on candidate 4).

## Recommended fix shape for Builder (after gate confirmation)

Make the B-frame chroma MV/mode mapping explicit and identical on both sides — either
(a) deterministically zero-fill `mv_buf`/`bwd_mv_buf`/`block_modes_buf` tails up to
`split_total_blocks` on B-frame upload in `frame_data.rs` *and* have the encoder clear its
`mv_chroma_buf(_bwd)` tails, or better (b) implement the correct spatial mapping (spread
each 16×16-grid MV to its four 8×8-grid cells, à la `mv_spread_4x.wgsl`), which also fixes
the known stride mismatch noted in archived item #48 and should improve chroma prediction
quality outright. Plus the two one-line reference fixes from candidates 3 and 4.

**Canary:** log first/last nonzero index of `mv_chroma_buf` per B-frame on both sides, and
per-region chroma PSNR in the validator.
