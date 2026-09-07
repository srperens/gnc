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
FAILED=0
mkdir -p "$FRAMES_DIR"

for cmd in ffmpeg curl; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: $cmd is required but not found"
        exit 1
    fi
done

# ffmpeg 5.1 renamed `-vsync` to `-fps_mode`, and ffmpeg 9 removed `-vsync` outright.
# This script runs on more than one machine, so ask the installed binary which one it
# takes rather than pinning either. Getting this wrong is not loud: on 2026-09-07 the
# rejected option only produced "Unrecognized option 'vsync'" into a swallowed stream,
# and the blue_sky sequence went missing while the script still exited 0.
if ffmpeg -nostdin -hide_banner -loglevel quiet \
        -f lavfi -i "testsrc=d=0.1" -fps_mode vfr -f null - </dev/null 2>/dev/null; then
    VFR_OPT="-fps_mode vfr"
else
    VFR_OPT="-vsync vfr"
fi
echo "Using '${VFR_OPT}' for variable-frame-rate output"

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
        FAILED=1
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
        FAILED=1
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

# === Multi-frame sequences for temporal testing ===
echo "=== Fetching multi-frame sequences (10 frames each) ==="
echo ""

SEQ_DIR="${FRAMES_DIR}/sequences"
mkdir -p "${SEQ_DIR}/bbb" "${SEQ_DIR}/blue_sky"

# BBB: 10 consecutive frames (350-359)
echo "[seq 1/2] Big Buck Bunny sequence (8 frames, 1080p)"
for i in $(seq 0 7); do
    frame_num=$((350 + i))
    out_png="${SEQ_DIR}/bbb/frame_$(printf '%04d' $i).png"
    if [ -f "$out_png" ]; then
        echo "  [skip] ${out_png}"
        continue
    fi
    echo "  Downloading frame ${frame_num}..."
    curl -# -L -o "$out_png" \
        "https://media.xiph.org/BBB/BBB-1080-png/big_buck_bunny_$(printf '%05d' $frame_num).png"
done
# Generate Y4M for fast encoder input (avoids PNG decode overhead)
if [ ! -f "${SEQ_DIR}/bbb/bbb.y4m" ] && [ -f "${SEQ_DIR}/bbb/frame_0000.png" ]; then
    echo "  Generating bbb.y4m..."
    ffmpeg -nostdin -y -loglevel error \
        -framerate 30 -start_number 0 \
        -i "${SEQ_DIR}/bbb/frame_%04d.png" \
        -pix_fmt yuv420p -f yuv4mpegpipe \
        "${SEQ_DIR}/bbb/bbb.y4m"
    echo "  [done] bbb.y4m"
fi

# blue_sky: extract 8 consecutive frames (50-57)
echo "[seq 2/2] blue_sky sequence (8 frames, 1080p)"
first_frame="${SEQ_DIR}/blue_sky/frame_0000.png"
if [ -f "$first_frame" ]; then
    echo "  [skip] blue_sky sequence already exists"
else
    echo "  Extracting 8 frames from blue_sky y4m..."
    ffmpeg -nostdin -y -loglevel error \
        -i "${XIPH}/y4m/blue_sky_1080p25.y4m" \
        -vf "select=between(n\\,50\\,57)" \
        $VFR_OPT \
        -start_number 0 \
        "${SEQ_DIR}/blue_sky/frame_%04d.png" </dev/null || true
    if [ -f "$first_frame" ]; then
        echo "  [done] blue_sky sequence"
    else
        echo "  [FAIL] Could not extract blue_sky sequence"
        FAILED=1
    fi
fi
# Generate Y4M for fast encoder input
if [ ! -f "${SEQ_DIR}/blue_sky/blue_sky.y4m" ] && [ -f "${SEQ_DIR}/blue_sky/frame_0000.png" ]; then
    echo "  Generating blue_sky.y4m..."
    ffmpeg -nostdin -y -loglevel error \
        -framerate 25 -start_number 0 \
        -i "${SEQ_DIR}/blue_sky/frame_%04d.png" \
        -pix_fmt yuv420p -f yuv4mpegpipe \
        "${SEQ_DIR}/blue_sky/blue_sky.y4m"
    echo "  [done] blue_sky.y4m"
fi

echo ""
echo "=== Verifying ==="
echo ""

# The script used to exit 0 even when a fetch failed: on 2026-09-07 ffmpeg 9 rejected
# `-vsync` (removed upstream), the `|| true` swallowed it, and the blue_sky sequence was
# silently absent while the script reported success. Check what actually landed.
expect_file() {
    local f="$1"
    if [ -f "$f" ] && [ -s "$f" ]; then
        echo "  [ok]   $f"
    else
        echo "  [MISS] $f"
        FAILED=1
    fi
}

for n in bbb_1080p blue_sky_1080p kristensara_720p touchdown_1080p; do
    expect_file "${FRAMES_DIR}/${n}.png"
done
for i in 0 1 2 3 4 5 6 7; do
    expect_file "${SEQ_DIR}/bbb/frame_000${i}.png"
    expect_file "${SEQ_DIR}/blue_sky/frame_000${i}.png"
done
expect_file "${SEQ_DIR}/bbb/bbb.y4m"
expect_file "${SEQ_DIR}/blue_sky/blue_sky.y4m"

echo ""
if [ "$FAILED" -ne 0 ]; then
    echo "=== INCOMPLETE — some material is missing (see [MISS]/[FAIL] above) ==="
    echo ""
    echo "Re-run this script; it skips what is already present. A partial download"
    echo "satisfies the skip check, so delete a suspect file before retrying."
    exit 1
fi

echo "=== Done ==="
echo ""
ls -lh "${FRAMES_DIR}/"*.png 2>/dev/null || echo "No single frames downloaded"
echo ""
ls "${SEQ_DIR}/bbb/" 2>/dev/null | head -3 && echo "  ... (bbb sequence)" || true
ls "${SEQ_DIR}/blue_sky/" 2>/dev/null | head -3 && echo "  ... (blue_sky sequence)" || true
echo ""
echo "NOTE: Full broadcast sequences (crowd_run, rush_hour, stockholm, etc.) are not"
echo "      fetched here (multi-GB). To generate Y4M files for existing PNG sequences:"
echo "        ffmpeg -framerate 50 -i 'seq/frame_%04d.png' -pix_fmt yuv420p seq/seq.y4m"
echo ""
echo "Run benchmarks:"
echo "  cargo run --release -- benchmark -i test_material/frames/bbb_1080p.png"
echo "  cargo run --release -- sweep -i test_material/frames/blue_sky_1080p.png"
echo ""
echo "Run temporal benchmark (Y4M — fast, no PNG decode overhead):"
echo "  cargo run --release -- benchmark-sequence -i test_material/frames/sequences/bbb/bbb.y4m -n 8 -k 8 --temporal-wavelet haar"
