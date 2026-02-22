# Test Material

Place broadcast test content here. This directory is not tracked in git (except this README).

## Recommended test content

1. **Talking head** — Low motion, skin tones, fine detail in hair/eyes
2. **Sports/motion** — High motion, complex textures (grass, crowds)
3. **Graphics/text** — Sharp edges, solid colors, fine text

## Generating test images

A synthetic test image is included (`test_512.png`). For real broadcast content, extract frames from openly available test sequences:

```bash
# Extract a frame from a video file
ffmpeg -i input.mp4 -vf "select=eq(n\,0)" -vframes 1 frame_1080p.png

# Convert to specific resolution
ffmpeg -i input.mp4 -vf "scale=1920:1080,select=eq(n\,0)" -vframes 1 test_1080p.png
```

## Running benchmarks

```bash
# Single benchmark
cargo run --release -- benchmark --input test_material/frame.png -n 10

# Sweep quantization parameters
cargo run --release -- sweep --input test_material/frame.png
```
