# GNC — GPU-Native Codec

A patent-free video codec designed from scratch for GPU parallelism. Everything runs as wgpu
compute shaders (WGSL), written against the WebGPU feature set so one source targets Metal,
Vulkan, DX12 and the browser.

## The idea

Established codecs were designed for CPUs and carry sequential algorithms that a GPU cannot
exploit. GNC starts from the opposite constraint: **every 256x256 tile is fully independent, at
every stage.** That single rule buys parallelism, random access, low-latency decode and error
resilience at the same time.

Where GNC is meant to win is not bitrate — it is **portability and scale**. A chip's
fixed-function encoder blocks stay roughly constant however large and expensive the GPU is, while
shader throughput scales with the card. A bigger GPU should therefore buy more GNC instances. It
should also run where no hardware encoder exists at all, in a browser, and on any vendor.

GNC is deliberately **broad**: intra and inter, 4:2:0 / 4:2:2 / 4:4:4 at 8 and 10 bits, and a
quality range from heavy compression through visually lossless to bit-exact lossless. The target
is roughly H.264-class compression across that whole range, not a record at one point of it. Its
uses — contribution links, mezzanine and archival storage, low-latency preview, browser playback —
encode about as often as they decode, which bounds how much encoder search is worth buying.

**Several internal strategies selected by quality and bitrate is the design, not a failure to pick
one.** MED prediction replaces the wavelet entirely at `q=100`, the entropy coder follows quality,
and the wavelet depth follows the tile size.

## Where it stands

Honestly: **the compression is close on stills and about 1.6x off H.264 on video, and the two
claims the positioning rests on are the ones with the least evidence.**

- **Stills** are competitive — level with ProRes 4444 and ahead of JPEG XS with the `--abac`
  coder, and about 14% behind H.264 all-intra. JPEG 2000 uses the *same* wavelet and still needs
  54% fewer bits, so the remaining intra gap is the entropy coder and the rate is demonstrably
  reachable.
- **Video** needs about 1.6x the bitrate of x264 at contribution quality with `--abac`, 1.9x with
  the default coder.
- **Lossless** is bit-exact and beats JPEG 2000 lossless and PNG, while losing to FFV1.
- **Latency** is 25.2 ms round trip with zero frames of reordering delay — below the
  low-latency-HEVC band, above JPEG XS.
- **Throughput** is roughly 4x short of the 60 fps target, and about half of the per-frame cost is
  not GPU coding work.
- **Portability is half met.** Metal and Vulkan both run the whole codec with byte-identical
  output across independent implementations. **DX12 has never produced a single frame, and the
  browser path has never been verified in a browser.** That is a headline defect, not a
  compatibility nit, because portability is the axis the project claims to win on.
- **Scale is unmeasured.** "A bigger GPU buys more GNC instances than it buys hardware encoder
  blocks" is the central structural claim and it has no number yet. Concurrency currently saturates
  on host memory and per-process startup, not on the GPU.

Every figure, with its caveats and its corrections, is in **[BASELINE.md](BASELINE.md)** —
including which quantity each throughput number is and whether the machine was idle when it was
taken. Do not quote a number from this file without reading its row there.

## Build & run

```bash
cargo build --release
cargo test --release

gnc encode -i in.png -o out.gpuc -q 75            # stills
gnc decode -i out.gpuc -o out.png

gnc encode-sequence -i "frames/%04d.png" -o v.gnv -q 75 --keyframe-interval 8
gnc decode-sequence -i v.gnv -o "out/%04d.png" --seek 5.0

gnc benchmark -i in.png -q 75 --vmaf              # measurement
gnc benchmark-sequence -i clip.y4m --throughput   # Y4M: no image decode in the timer
gnc rd-curve -i in.png --compare-codecs
gnc gpu-info                                      # device, and the limits GNC requests
gnc fingerprint                                   # what this binary produces

cd test_material && bash fetch_test_frames.sh     # Xiph.org frames; needs ffmpeg + curl
wasm-pack build --target web --release            # browser decoder
```

## Pipeline

```
RGB → YCoCg-R → Wavelet → Quantize → Entropy Code → Bitstream
         ↕          ↕          ↕            ↕
     (lossless   (CDF 9/7   (adaptive,   (Rice+ZRL:
      integer)   or 5/3)     CfL, AQ)    256 streams)
```

Video adds half-pel motion compensation with hierarchical block matching, CBR/VBR rate control and
the GNV1 container with keyframe seek and per-tile CRC-32. Five entropy backends exist, all
decoding as GPU compute shaders; Rice is the default everywhere, `--abac` trades roughly 3x decode
time for 17–19% of the rate, and `--rans` is kept for experiments below q=20.

Stage-by-stage detail is in [`docs/PIPELINE.md`](docs/PIPELINE.md); the format is in
[`docs/BITSTREAM_SPEC.md`](docs/BITSTREAM_SPEC.md).

## Documentation

| | |
|---|---|
| [GOALS.md](GOALS.md) | rules, priorities, current state, non-goals — the source of truth |
| [BASELINE.md](BASELINE.md) | every benchmark figure, with its caveats |
| [BACKLOG.md](BACKLOG.md) | open items, each with its measurements attached |
| [RESEARCH_LOG.md](RESEARCH_LOG.md) | every experiment, including the failures |
| [`docs/POSITIONING.md`](docs/POSITIONING.md) | what GNC is for, and where it stands against the market |
| [`docs/decisions/`](docs/decisions/) | why each choice was made, and what was rejected |

**This file states the current position only.** Many figures here have been corrected or withdrawn
along the way; that history is kept deliberately visible in BASELINE, RESEARCH_LOG and the decision
records rather than in the front door.

## License

Patent-free: no H.264/5/6 pool or MPEG-LA encumbered techniques. All dependencies are open source.
