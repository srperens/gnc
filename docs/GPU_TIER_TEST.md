# The GPU tier test — what it is, and what it can settle

*Written 2026-09-06, for a Windows 11 laptop with an Intel integrated GPU and an
NVIDIA RTX 2000 Ada. Nothing in it is specific to that machine.*

Two numbers at the top of GOALS §4's target table have never been measured, and they
are the two the entire positioning rests on: **concurrent streams per GPU** and
**latency per frame**. Everything else in this repository — the compression figures,
the entropy coder work, the lossless path — is secondary to them, because if GNC does
not actually scale with the card then the reason to prefer it over a fixed-function
encoder mostly evaporates.

This harness measures the first one, and a cheaper thing that must be true before it
can mean anything.

## Run it

```bash
cargo build --release

python scripts/gpu_tier_bench.py --list           # what GPUs does this machine have?
python scripts/gpu_tier_bench.py --tier    -i test_material/frames/bbb_1080p.png
python scripts/gpu_tier_bench.py --density -i <clip>.y4m --adapter nvidia
python scripts/gpu_tier_bench.py --hwenc   -i <clip>.y4m --encoder h264_nvenc
```

`--all` runs the three in order. `--json report.json` keeps the raw numbers.

The binary picks its GPU from the environment, so one build serves every device:

| variable | effect |
|---|---|
| `GNC_GPU_ADAPTER=<substring>` | first adapter whose name contains it, case-insensitive |
| `GNC_GPU_BACKEND=vulkan\|dx12\|metal\|gl` | restrict to one backend |
| `GNC_GPU_POWER=high\|low` | preference when no adapter name is given |
| `GNC_GPU_INFO=1` | print the adapter actually in use to stderr |

A `GNC_GPU_ADAPTER` that matches nothing is a hard error listing what is present.
Silently falling back to the default adapter would file a measurement of one GPU under
the name of another, which is precisely the failure the canary rule exists to catch.
Every run that sets one of these variables prints a `[gpu] …` line naming the device.

## Test 1 — CANARY-1: does encode time move between GPUs at all?

`--tier` runs the single-frame encode/decode loop on each adapter in turn and reports
the best of several processes.

**Why it comes first.** In 2011 BeHardware measured the shipping GPU H.264 encoders of
the day and found them performing *identically* on a 100 EUR card and a 330 EUR card —
because they were never compute-bound at all. The GPU was doing far less than the
marketing implied. A laptop with an integrated and a discrete GPU is the cheapest
possible version of that experiment, and until it passes, no throughput number from
this project means anything.

**Pass:** encode time drops substantially on the discrete GPU. The expected direction is
strong — an RTX 2000 Ada has roughly an order of magnitude more shader throughput than a
current Intel integrated GPU.

**Fail:** the two come out within ~15% of each other. The script says so in plain words
when it happens. That would mean the measurement is dominated by something other than
GPU compute — I/O, driver overhead, per-dispatch CPU cost — and that is a bug to find
before any density claim is made.

This test needs no NVENC comparison and no quality matching. It is pure self-consistency,
and it is the one result on this machine that is unambiguous.

## Test 2 — MEAS-5: how far does concurrency scale, and against what?

`--density` launches N concurrent GNC encodes of the same clip and reports aggregate
throughput; `--hwenc` does the same sweep through the machine's fixed-function encoder.

The structural argument is that a chip's hardware encoder blocks stay roughly constant
however large the GPU is, while shader throughput scales with the card — so a bigger GPU
should buy more GNC instances and does not buy more NVENC blocks. The M1 measurement so
far gives about **2x aggregate at N=8**, most of it already at N=2: real headroom, far
from linear. The published multi-tenancy literature agrees that concurrency converts
*idle* GPU into *useful* GPU rather than creating GPU.

### What this machine can and cannot settle

**It can** establish the scaling *slope* over three tiers — M1, Intel integrated, RTX 2000
Ada — which is more informative than any single point, and it tests whether GNC keeps
completing as N rises where a fixed-function encoder might not.

**It cannot** settle the density claim in its strong form. An RTX 2000 Ada is a small
mobile part; it is the least favourable version of GNC's own argument, which is about
large GPUs. Winning here is strong evidence. Losing here is not disproof, and the
honest reading is the slope, not the endpoint.

**Two things to know before reading the NVENC rows, or they will be misread:**

- **The session cap will not appear on this card.** The documented 12-concurrent-session
  limit is a GeForce driver restriction; professional RTX Ada parts do not carry it. On
  this laptop the comparison is pure throughput, and the absence of a cap is expected
  rather than a refutation.
- **The strongest fact in the positioning cannot be tested here at all.** A100, H100 and
  B200 ship with *no* NVENC whatsoever — an idle AI fleet has zero encode capacity. That
  stays a sourced claim about hardware neither of us has.

**The rows are not quality-matched.** The script says so in its own output. Bitrate and
distortion parity is `scripts/meas1_vs_h264.py`'s job; read these tables as session-count
and scaling only. A throughput comparison between two encoders sitting at different
quality points is not a comparison.

## Test 3 — the free one: does GNC run on Windows at all?

GOALS rule 4 claims Metal, Vulkan, DX12 and WebGPU. Only Metal has ever been exercised.
Simply getting `cargo build --release` and `gnc gpu-info` to work on Windows tests a
stated design property that has never been checked, and `GNC_GPU_BACKEND=vulkan` against
`GNC_GPU_BACKEND=dx12` on the same card compares two backends of the same shaders on
identical hardware. Any divergence in output between them is a portability bug worth more
than the throughput numbers.

## Measuring honestly on a laptop

COORDINATION.md's timing rules apply, and a laptop makes two of them sharper:

- **An idle machine is necessary and not sufficient.** A GPU ramps its clocks, and three
  consecutive processes on identical input have read 66.5, 45.3 and 34.9 ms on a genuinely
  idle M1 — a 1.9x spread, monotonically decreasing. The script takes the **best** of
  several processes, because every error source only makes a reading slower, and prints
  `median / best` beside it: near 1.0 means quotable, well above means keep only the ratios.
- **Thermal throttling is the laptop-specific hazard**, and it pushes the other way from
  the clock ramp. Run on mains power, on a hard surface, with the power profile at maximum.
  Density is a question about *sustained* throughput, so if the sustained figure is well
  below the best-of-N figure, that gap is itself the result — report both.

Do not tune anything against numbers from this machine while other work is running on it.
Compression figures (bpp, PSNR, VMAF, dE00) are deterministic and safe at any load;
throughput figures are not.

## What to record

Whatever comes out, it belongs in RESEARCH_LOG.md with the machine stated, and CANARY-1's
result belongs in the regression suite as a permanent check. The failure mode this guards
against is silent: a pipeline that is not running where we think it is looks exactly like
a pipeline that is, right up until someone compares two GPUs.
