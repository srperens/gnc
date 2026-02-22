#!/usr/bin/env bash
#
# Fetch broadcast-representative test frames from freely available sources.
# Requires: ffmpeg, curl
#
# Content types covered:
#   - Nature/landscape (blue_sky, 1080p)
#   - Talking head (KristenAndSara, 720p)
#   - Sports/motion (touchdown_pass, 1080p)
#   - Animation (Big Buck Bunny, 1080p)
#
# Sources: Xiph.org / Derf's collection (freely redistributable)
# Strategy: Download PNG frames directly where possible, otherwise stream
# y4m via ffmpeg to extract a single frame (avoids downloading multi-GB files).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

FRAMES_DIR="frames"
mkdir -p "$FRAMES_DIR"

for cmd in ffmpeg curl; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: $cmd is required but not found"
        exit 1
    fi
done

# Extract a single frame from a remote y4m by streaming just enough data.
# We pipe curl into ffmpeg and grab frame N, then kill the download.
fetch_y4m_frame() {
    local name="$1"
    local url="$2"
    local frame_num="${3:-0}"
    local out_png="${FRAMES_DIR}/${name}.png"

    if [ -f "$out_png" ]; then
        echo "  [skip] ${out_png} already exists"
        return
    fi

    echo "  Streaming ${name} (extracting frame ${frame_num})..."
    # Stream the y4m and extract just the frame we need.
    # For frame 0 we only need ~6MB of a 1080p y4m.
    ffmpeg -nostdin -y -loglevel error \
        -i "$url" \
        -vf "select=eq(n\\,${frame_num})" \
        -vframes 1 \
        "$out_png" </dev/null 2>&1 || true

    if [ -f "$out_png" ] && [ -s "$out_png" ]; then
        echo "  [done] ${out_png}"
    else
        echo "  [FAIL] Could not extract frame from ${url}"
        rm -f "$out_png"
    fi
}

fetch_png_frame() {
    local name="$1"
    local url="$2"
    local out_png="${FRAMES_DIR}/${name}.png"

    if [ -f "$out_png" ]; then
        echo "  [skip] ${out_png} already exists"
        return
    fi

    echo "  Downloading ${name}..."
    curl -# -L -o "$out_png" "$url"

    if [ -f "$out_png" ] && [ -s "$out_png" ]; then
        echo "  [done] ${out_png}"
    else
        echo "  [FAIL] Download failed for ${url}"
        rm -f "$out_png"
    fi
}

XIPH="https://media.xiph.org/video/derf"

echo "=== Fetching broadcast test frames ==="
echo ""

# 1. Animation - Big Buck Bunny (direct PNG, small download ~3MB)
echo "[1/4] Big Buck Bunny (1080p, animation)"
fetch_png_frame "bbb_1080p" \
    "https://media.xiph.org/BBB/BBB-1080-png/big_buck_bunny_00350.png"

# 2. Nature/landscape - blue_sky (stream y4m, grab frame 50)
echo "[2/4] blue_sky (1080p, nature/landscape)"
fetch_y4m_frame "blue_sky_1080p" \
    "${XIPH}/y4m/blue_sky_1080p25.y4m" \
    50

# 3. Talking head - KristenAndSara (stream y4m, grab frame 30)
echo "[3/4] KristenAndSara (720p, talking head)"
fetch_y4m_frame "kristensara_720p" \
    "${XIPH}/y4m/KristenAndSara_1280x720_60.y4m" \
    30

# 4. Sports/motion - touchdown_pass (stream y4m, grab frame 100)
echo "[4/4] touchdown_pass (1080p, sports/motion)"
fetch_y4m_frame "touchdown_1080p" \
    "${XIPH}/y4m/touchdown_pass_1080p.y4m" \
    100

echo ""
echo "=== Done ==="
echo ""
ls -lh "${FRAMES_DIR}/"*.png 2>/dev/null || echo "No frames downloaded"
echo ""
echo "Run benchmarks:"
echo "  cargo run --release -- benchmark -i test_material/frames/bbb_1080p.png"
echo "  cargo run --release -- sweep -i test_material/frames/blue_sky_1080p.png"
