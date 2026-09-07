# Simple performance fixes — host-side scan (2026-09-07)

Code inspection of the encode and decode hot paths, looking for *simple* wins:
unnecessary copies, extra GPU submits/readbacks, per-dispatch object creation,
leftover diagnostics, missing buffer reuse. Not algorithm changes, not shader
occupancy experiments.

No throughput number in this file is a new measurement. FPS figures are quoted
from BASELINE.md and RESEARCH_LOG.md and were taken on a non-idle machine.
COORDINATION.md: do not tune kernels against wall-clock while other sessions
hold the GPU. Host-side waste can be counted in bytes and in extra
`create_buffer` / `poll(Wait)` calls without a timing run.

**Worktree:** `gnc-perffix`. Not claimed as a BACKLOG item — this is a scan.

## Which fps

Three quantities have been called "encode fps" and they differ by ~2.4×
(BASELINE.md, 1080p, ki=8, Rice, M1, machine not idle):

| | what | figure |
|---|---|---|
| **A** | GPU encode phase (`benchmark-sequence`, Y4M) | 12.2 fps (~82 ms) |
| **B** | encoder loop (`encode-sequence` print) | 5.6 fps |
| **C** | wall clock (`encode-sequence`, PNG) | 5.0 fps |

The previously quoted 31.7 fps is not reproducible. GOALS still cites it.

The **2.4× gap between A and C is host-side work** (load, convert, clone,
extra polls). That is where the simple fixes live. Moving **A** itself is GPU
wavelet + Rice.

I-frame GPU profile (bbb 1080p q=75, Rice, `GNC_PROFILE=1`, 2026-03-10):

```
gpu_wavelet_quant ≈ 12.75 ms
gpu_rice          ≈ 12.8 ms
rice_assemble     ≈ 0.5 ms
pad/upload        ≈ 3.0 ms
total             ≈ 29.5 ms   (~34 fps I-only)
```

GPU is ~86% of I-frame time. After the Rice `to_vec` cut the compute floor was
wavelet 12.3 ms + Rice 9.1 ms ≈ 21 ms — "no amount of CPU-side optimization
can reduce below 21 ms." That was 4 wavelet levels; default is now 5.

Decode: Rice entropy is ~14.0 ms of ~29.5 ms I-frame decode (47%). BACKLOG's
"entropy coding, 51–85% of runtime" is the **decode** bottleneck, not encode.

## Already closed — do not re-propose

| item | result |
|---|---|
| Fused wavelet (#29) | Closed without impl. All 24 dispatches already one encoder; Metal barriers ~150 µs. Fusing level-0 row+col needs 256 KB LDS vs M1 32 KB. |
| Fused quantize+Rice (#33) | Closed. Bandwidth save ~0.35 ms. Gate was quantize+Rice > 30 ms; measured ~19 ms. |
| Three abac decode shader opts | Context scratch → workgroup memory; sort blocks by area; thread-interleaved LDS. All three returned null on a 48% noisy instrument (25.2 / 31.1 / 37.5 ms). Do not re-run. |
| Rice GPU `to_vec` of 192 MB | Shipped. Pack 4 ms → 0.6 ms. |
| 58 MB AQ readback | Replaced by GPU tile-energy reduce (34 ms → 4.2 ms). |
| Fused rANS encode | −20 ms per P-frame (256 threads, only 32 encode). Occupancy loss. |
| Hierarchical ME / fewer SAD | Quality OK, fps ~flat (19.6 vs 18.9). Occupancy, not SAD count. |
| `docs/MEGA_KERNEL_PLAN.md` | Stale vs #29. Do not start a mega-kernel for a 0.5 ms submit. |

Shader occupancy tweaks and kernel fusion are not "simple" on this tree.
They have been tried. The leftovers below are host-side or obviously extra
copies/passes.

---

## Ranked candidates

### 1. 24 MB RGB clones on every `load_frame` — easy, bitstream-identical

`EncoderPipeline::encode` takes `&[f32]`. Everything above it clones the
full interleaved RGB frame (~24.9 MB at 1080p).

`encode_sequence` already has the frames in memory, then copies them into a
new `Vec` to satisfy the streaming API:

```
src/encoder/sequence.rs  (~140)
    |i| frames[i].to_vec()
```

Y4M cache hits clone too:

```
src/main.rs  StreamingY4m::load  (~965)
    self.cache[i].clone()
```

Scene-cut is on by default (`scene_cut_threshold: 50.0`) whenever `ki > 1`.
The P-frame path loads for MAD, **drops** the buffer if it is not a cut, then
loads again:

```
src/encoder/sequence.rs  (~292)
    // If not a cut, preloaded_frame remains None — the P/B-frame path
    // will call load_frame again.
```

Look-ahead ME loads `display_idx + 1`, and the next iteration loads that
index again. B-GOP scene-cut scan clones every frame in `[display_idx, next_key)`,
plus `prev_luma.clone()`.

Comment at `sequence.rs:223` says `prev_frame_luma` stores "just the R
channel". The code stores the whole interleaved RGB `frame_data`.
`luma_mad` only reads R every 4th pixel.

`encode-sequence` (CLI) only loads PNG and the timer wraps
`encode_sequence_streaming` (`src/main.rs` ~3104). PNG decode + `u8→f32` is
typically 10–30 ms; two or three of those per P-frame is enough of the A→B/C
gap by itself. `benchmark-sequence` CLI help already admits PNG inflates cost.

**Fix:** `Arc<[f32]>` in the cache and in `load_frame`; always keep
`preloaded_frame` after the MAD load (P/B paths must `take()` it); store a
luma proxy or an `Arc`, not a second RGB copy. `encode()` does not change.

Domain: bitstream / host frame loader. Not coefficients.

### 2. Y4M: scalar YCbCr→RGB on CPU, then RGB→YCoCg on GPU — medium

`Y4mReader::read_frame_rgb` (`src/main.rs` ~119–211), every frame:

- allocate Y/Cb/Cr as `f32` (~10 MB at 4:2:0)
- nested `row × col` BT.601 loop to interleaved RGB
- allocate 24 MB RGB `f32`
- encoder `write_buffer` of ~24.9 MB
- GPU `color_convert` RGB → YCoCg-R

GNC codes YCoCg-R, not RGB. Y4M is already YUV. Double colour conversion plus
a 4× oversized upload (u8 RGB is 6.2 MB; u8 4:2:0 is ~3 MB).

Older profile: `pad ≈ 3.0 ms` was the CPU→GPU upload. The Y4M loop sits
*outside* GPU-encode (A) and *inside* end-to-end (C) on the streaming path.

**Smaller fix:** reuse plane buffers in the reader; skip `f32` intermediates;
SIMD/chunk the loop.

**Right fix:** upload packed u8 YUV and convert YUV→YCoCg in a shader. Same
pixels, less DMA, no CPU colour. Touches the encode input API (today RGB f32).

### 3. Extra `map_async` + `poll(Wait)` after Rice is already done — easy

Rice `finish_3planes_readback` already polls (`src/encoder/rice_gpu.rs` ~899).
I-frame encode then polls **again** for:

- AQ weight map (~pipeline.rs 2562)
- CfL alphas (~2583) — CfL is on at q=75
- intra modes if enabled

P-frames poll **again** for MVs in `finish_mv_readback_cached` (`motion.rs`
~1597), even though `split_mv_staging_buf` was copied in the **same** submit
as Rice (~sequence.rs 4259). Look-ahead ME is submitted before the Rice poll
to hide sync — then MV readback adds another Wait.

**Fix:** map Rice + MV + weight map + CfL in one poll. Driver round-trips, not
GPU work. Several milliseconds on the I/P path.

### 4. `create_buffer_init` + `create_bind_group` on every dispatch — easy

Rice encode and the wavelet already cache uniforms (`write_buffer` into a
persistent UBO; wavelet uses dynamic offsets). The rest of the hot path still
allocates a GPU buffer and a bind group per call. Pad params are cached; the
pad **bind group** is not.

| site | what |
|---|---|
| `src/encoder/color.rs:117` | new uniform + bind group every colour convert |
| `src/encoder/quantize.rs:224` | same, 6+ times per P-frame (3 encode + 3 local-decode) |
| `src/encoder/interleave.rs:134` | same |
| `src/encoder/adaptive.rs:204` | variance / AQ |
| `src/encoder/pipeline.rs:888` | pad bind group every frame |
| `src/encoder/pipeline.rs:1041, 1200, 1265` | tile_skip / zero_skip / mv_smooth uniforms |
| `src/encoder/transform.rs:255` | 3 bind groups × 3 planes, forward and inverse |
| `src/encoder/rice_gpu.rs:792` encode, `:1470` decode | bind group per plane |
| `src/decoder/gpu_work.rs:672` | crop bind group |
| `src/decoder/pipeline.rs:477, 1982` | pack bind group |
| `src/encoder/motion.rs` ~2207, `chroma_resample.rs:229`, `cfl.rs:345` | P/B helpers |

`CachedBuffers` already caches `buf_to_tex_bind_group`
(`src/decoder/buffer_cache.rs:477`). Crop/pack/colour/interleave/dequant
buffers do not change after `ensure_cached`. The template is
`GpuRiceEncoder`'s `params_buf` comment: *"updated via write_buffer, avoids
per-frame create_buffer_init"*. Decode never got that treatment.

I-frame `wq_cmd` was 0.6 ms of command recording. P-frames add ME+MC+skip+
local decode on top. Bitstream-identical.

### 5. Dummy CfL `MAP_READ` buffers every `encode()` — easy

```
src/encoder/pipeline.rs  (~1809)
    } else {
        // Dummy buffers (never used)
        std::array::from_fn(|_| {
            ctx.device.create_buffer(... usage: MAP_READ ...)
```

When CfL is off (default, and q ≥ 92) two `MAP_READ` buffers are created and
never used. When CfL is on, staging is created every frame instead of living
in `CachedEncodeBuffers`. `Option<[Buffer; 2]>` or a cache slot.

### 6. I-frame preprocess is a second `queue.submit` — easy, modest

`src/encoder/pipeline.rs` ~1598–1683 submits pad+colour+deinterleave, then
records wavelet+quant+Rice and submits again (~2389). P-frames already batch
preprocess into one encoder. Split-phase was measured at ~0.5 ms. Cheap,
obviously correct. Do not split the wavelet itself — that raced on shared
`plane_*` buffers (RESEARCH_LOG).

### 7. Decode: `pack_decode_data` alloc + memcpy every plane — easy

`src/encoder/rice_gpu.rs:1391`, called from `src/decoder/frame_data.rs:174`.
Every frame, per plane: `vec![0u32; …]` for k, offsets, and packed words,
`copy_from_slice` of all stream bytes, then four `write_buffer`s.
`ensure_var_buf` can also reallocate GPU buffers when sizes grow.

Reuse a CPU scratch (same idea as encode's cached GPU buffers) and only
`write_buffer`. Shows up in `GNC_PROFILE` as `prepare=`.

### 8. Decode: Rice bit reader loads one **byte** per refill — easy

`src/shaders/rice_decode.wgsl` `load_byte` / `read_bit` / `read_bits`
(~91–131). Unary Rice is `while (read_bit())`. Every 8 bits: storage load of
a `u32`, shift, mask. Encode already accumulates **words** (archive log
"Word-at-a-time output"). Keep a 32-bit window, refill every 4 bytes. No
bitstream change. Inner loop of ~47% of I-frame decode.

This is a shader change, but it is mechanical and not an occupancy
experiment. Still: do not quote a fps delta until an idle-machine run.
`GNC_RICE_DISPATCH_REPEAT` isolates the entropy slice.

### 9. Decode: extra full-plane `copy_buffer_to_buffer` — easy–medium

| copy | where | why it is waste |
|---|---|---|
| Inverse wavelet preamble | `src/encoder/transform.rs:320` | Whole-plane copy so ping-pong starts with all subbands. First pass can read `input_buf`. ×3 planes. |
| I-frame residual → `plane_results` | `src/decoder/gpu_work.rs:624` | Inverse wavelet already wrote `scratch_a`. Bind `plane_results[p]` as inverse output. |
| `plane_results` → `reference_planes` | `src/decoder/gpu_work.rs:637` | Needed for video; wasted on still / all-I. |
| Chroma upsample bounce | `src/decoder/gpu_work.rs:434` | Upsample can target the next consumer. |
| Intra reconstruct bounce | `src/decoder/gpu_work.rs:411` | Only if `intra_prediction` is on (off by default). |
| CfL Y snapshot + alpha slice | `src/decoder/gpu_work.rs:309` | Full luma copy; alpha copy because of a slice offset. Bind a ranged view. |

1080p luma is ~8 MB. Three I-frame copies × 3 planes is tens of MB of extra
bus traffic plus extra Metal copy passes.

### 10. Dequant as a standalone full-frame pass — easy–medium

`src/shaders/quantize.wgsl:170` (`direction == 1` is `output[idx] = val *
effective_step`), dispatched from `src/decoder/gpu_work.rs:291`. Rice already
writes every coefficient as `f32` (`rice_decode.wgsl:212`). Dequant is one
multiply + a subband/AQ lookup, then another 8 MB read + 8 MB write per
plane. Fold the multiply into the Rice store (quantize's
`compute_subband_index`, not Rice's 12-group map). One fewer dispatch and
~48 MB less traffic at 1080p 4:4:4.

Do not confuse this with fused quantize+Rice **encode** (#33), which was
closed at 0.35 ms. This is the **decode** dequant that runs *after* Rice has
already produced floats.

### 11. Interleave → inv-colour → crop → pack as four bandwidth kernels — medium

`src/shaders/{interleave,color_convert,crop,pack_u8}.wgsl`, wired in
`gpu_work.rs:648` + `pipeline.rs:475`. Four full-frame trips: planar Y/Co/Cg
→ interleaved YCoCg → RGB → cropped RGB → packed u8. Roughly 100+ MB extra
R/W at 1080p. `decode_to_texture` already skips pack and has a cached bind
group; CLI `decode_u8` still pays crop+pack. One shader: read 3 planes,
YCoCg-R inverse, crop, pack/store. Arithmetic is tiny vs bandwidth.

Not as "simple" as (4) or (7). Build behind a switch; measure on an idle
machine with the rest of the set.

### 12. Smaller leftovers

- **`gpu_util::read_buffer_*`** (`src/gpu_util.rs:26`): allocates a staging
  buffer, submit, poll, `to_vec` every call. Default GPU Rice uses cached
  staging. Bites CPU-entropy / diagnostics / abac readback, not the Rice
  default.
- **`decode_to_owned_texture`** (`src/decoder/pipeline.rs:619`): fresh
  texture every call (comment admits it). Pool N textures. Still-image
  `decode_u8` does not hit this.
- **Checkerboard two-pass Rice** (`rice_decode.wgsl:256`, same shape in
  encode): even streams, `workgroupBarrier()`, then odd streams. 50% of
  lanes idle each pass. Measured **neutral bpp** (expected 0.05–0.15, got
  0). `shared_ctx_even` is ~6 KB; GOALS "Rice uses < 1 KB shared" is stale.
  Dropping it is a product decision on a null compression feature, not a
  cleanup. Do not treat it as a free occupancy win without an idle-machine
  number.
- **Inverse wavelet occupancy at fine levels:** `workgroup_size(256)` with
  `region=16` → 8 of 256 threads work; 4 KB LDS even on inverse
  (`overlap=0`) which never uses `shared_data`. Fusion already measured
  ~150 µs. Low expected win next to Rice.
- **Intra reconstruct `@workgroup_size(1)`:** one thread per tile, serial
  raster. Default Rice I-frame path does not set `intra_modes`. Skip unless
  that flag is on.
- **`eprintln!("GNC: checkerboard k-context active…")`** in
  `GpuRiceEncoder::new` — once per encoder, not a fps item.

---

## What is *not* the encode bottleneck

- **Rate vs JPEG 2000:** INTRA-1 — GNC is within 7.5% of its own coefficient
  entropy; most of the J2K gap is transform/quant, not Rice.
- **Rice vs rANS (rate):** a few percent. rANS costs ~8% encode / ~15%
  decode. Not a throughput lever.
- **abac encode:** no GPU encoder; 75–155 ms/frame; opt-in. Default is Rice
  GPU.
- **ME on I-frames:** I-frames have no ME. P-frame ME is already look-ahead
  overlapped in P-only mode.
- **Reference deblock:** off unless `GNC_REF_DEBLOCK=1`.

## Recommended order

Do not start a timing run while other sessions hold the GPU.

1. **`Arc<[f32]>` + stop double-loading after scene-cut** (item 1). Small
   diff, counted in bytes copied, no GPU time. Give `encode-sequence` a Y4M
   path so PNG is not inside the timer.
2. **Cache uniforms and bind groups** the way Rice already does (item 4),
   drop dummy CfL staging (item 5), one poll for all readbacks (item 3),
   fold I-frame preprocess into the main submit (item 6).
3. **Y4M packed-u8 upload** (item 2). Larger, but the only host change that
   can close A→C on the streaming path for real.
4. **Decode:** CPU pack scratch (item 7), 32-bit Rice window (item 8),
   delete the extra plane copies (item 9). Measure Rice stage and whole-frame
   decode separately (`GNC_RICE_DISPATCH_REPEAT` for the entropy slice).

Items 10–11 (fold dequant; fuse colour/crop/pack) belong behind switches and
an idle-machine bench, same pattern as `GNC_ABAC_CODER`.

Canary for (1): log `load_frame` call count vs frame index; it should be 1
per display index on the P-only path, not 2–3. Canary for (4): zero
`create_buffer_init` on the steady-state I-frame path (count in
`GNC_PROFILE`).
