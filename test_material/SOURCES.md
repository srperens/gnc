# Test Material Sources

Freely available broadcast-quality test sequences for codec research.

## Xiph.org / Derf's Collection

The de facto standard test media library for open-source codec development.

- **URL:** https://media.xiph.org/video/derf/
- **Format:** Uncompressed YUV4MPEG (y4m), xz-compressed
- **License:** Freely redistributable / CC-BY

### 1080p sequences

| Sequence | Content type | Frames |
|---|---|---|
| `touchdown_pass` | Sports/motion | 570 |
| `rush_field_cuts` | Sports/motion | 570 |
| `speed_bag` | Sports/motion | 570 |
| `blue_sky` | Nature | 217 |
| `sunflower` | Nature | 500 |
| `pedestrian_area` | Urban | 375 |
| `riverbed` | Nature/texture | 250 |
| `KristenAndSara` (720p) | Talking head | 600 |
| `Johnny` (720p) | Talking head | 600 |

### 4K sequences

| Sequence | Content type |
|---|---|
| `crowd_run` | Urban/motion |
| `ducks_take_off` | Nature/motion |
| `park_joy` | Nature |

### Download

```bash
curl -O https://media.xiph.org/video/derf/y4m/blue_sky_1080p.y4m.xz
xz -d blue_sky_1080p.y4m.xz
```

## UVG 4K Dataset (Tampere University)

Purpose-built 4K dataset for codec evaluation. Sony F65 camera source.

- **URL:** https://ultravideo.fi/dataset.html
- **GitHub:** https://github.com/ultravideo/UVG-4K-Dataset
- **Format:** Raw YUV 4:2:0, 8-bit and 10-bit, 3840x2160
- **License:** CC BY-NC (free for research, citation required)

### Sequences (120 fps, 5 seconds each)

| Sequence | Content type | Size (8-bit) |
|---|---|---|
| `Beauty` | Face closeup | 3.87 GB |
| `Bosphorus` | Yacht/panning | 2.66 GB |
| `HoneyBee` | Nature detail | 3.51 GB |
| `Jockey` | Horse racing | 3.22 GB |
| `ReadySetGo` | Fast motion | 3.06 GB |
| `ShakeNDry` | Fast motion | 1.75 GB |
| `YachtRide` | Water/motion | 2.68 GB |

### Sequences (50 fps, 12 seconds each)

CityAlley, FlowerFocus, FlowerKids, FlowerPan, RaceNight, RiverBank, SunBath, Twilight

## Netflix Open Content

Professional production content for codec and HDR research.

- **URL:** https://opencontent.netflix.com
- **S3:** `s3://download.opencontent.netflix.com/`
- **Format:** 16-bit TIFF frame sequences, OpenEXR, DPX, ProRes
- **License:** CC-BY 4.0 (most content)

### Projects

| Project | Resolution | Notes |
|---|---|---|
| Chimera | 4096x2160, 60fps, 10-bit | Varied live-action scenes |
| El Fuente | 4096x2160, 10-bit | Indoor/outdoor scenes |
| Meridian | 4K HDR | 12-minute short film |
| Sol Levante | 4K HDR | Hand-drawn anime |

### Download

```bash
aws s3 ls --no-sign-request s3://download.opencontent.netflix.com/
aws s3 sync --no-sign-request s3://download.opencontent.netflix.com/aom_test_materials/Chimera/ ./Chimera/
```

## Blender Foundation Open Movies

Full films with lossless frame sequences available via Xiph.org mirrors.

- **URL:** https://media.xiph.org/BBB/ , https://media.xiph.org/sintel/ , https://media.xiph.org/tearsofsteel/
- **Format:** PNG sequences, TIFF 16-bit, y4m, OpenEXR
- **License:** CC-BY 3.0

| Film | Resolution | Notes |
|---|---|---|
| Big Buck Bunny | 1080p, 4K | Animation, fast motion |
| Sintel | 1080p, 4K | Fantasy, VFX, varied lighting |
| Tears of Steel | 1080p, 4K | Live-action + VFX, real camera noise |

```bash
rsync -avz rsync://media.xiph.org/tearsofsteel/tearsofsteel-4k-png/ ./tos-4k-png/
```

## Format Notes

- **y4m** — text header + raw planar YUV per frame. Trivial to parse.
- **Raw YUV** — no container, just raw planar pixels. Need to know resolution and bit depth.
- **PNG sequences** — easiest to use with the `image` crate (current codec input path).
- **TIFF 16-bit** — useful for 10-bit+ pipeline testing.
