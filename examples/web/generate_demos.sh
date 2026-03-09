#!/usr/bin/env bash
#
# Generate demo .gnv files (I+P+B motion vectors) for the web player.
# Uses benchmark-sequence --output, which encodes, decodes, and reports per-frame
# PSNR/SSIM/bpp — same diagnostics as the research workflow.
# For temporal wavelet demos (GNV2), see generate_demos_tw.sh.
# Re-run this whenever the bitstream format changes.
#
# Requires: cargo build --release (gnc binary)
# Source material: test_material/frames/sequences/ (run fetch_test_frames.sh first)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GNC="$ROOT/target/release/gnc"
SEQ="$ROOT/test_material/frames/sequences"
OUT="$SCRIPT_DIR"

if [ ! -x "$GNC" ]; then
    echo "Building gnc..."
    (cd "$ROOT" && cargo build --release)
fi

# Some sequences start at frame_0001 instead of frame_0000.
ensure_zero_indexed() {
    local dir="$1"
    if [ -f "$dir/frame_0001.png" ] && [ ! -f "$dir/frame_0000.png" ]; then
        ln -s frame_0001.png "$dir/frame_0000.png"
        echo "  (symlinked frame_0000 → frame_0001 in $(basename "$dir"))"
    fi
}

# Encode a demo using benchmark-sequence (gives per-frame PSNR/SSIM + saves GNV1).
# Usage: bench_encode <output-name> <input-pattern> [benchmark-sequence flags...]
bench_encode() {
    local name="$1"
    local input="$2"
    shift 2

    local outfile="$OUT/${name}.gnv"
    echo ""
    echo "=== ${name}.gnv ==="
    ensure_zero_indexed "$(dirname "$input")"

    "$GNC" benchmark-sequence -i "$input" -o "$outfile" "$@"

    local size
    size=$(ls -lh "$outfile" | awk '{print $5}')
    echo "  → saved: $outfile ($size)"
}

# Clean old GNV1 demo files (not tw_* which belong to generate_demos_tw.sh)
echo "GNC demo file generator (GNV1 — I+P+B, benchmark-sequence)"
echo "============================================================"
echo "output: $OUT"
echo ""
echo "Removing old .gnv files..."
rm -f "$OUT"/*.gnv

# --- Broadcast sequences (long, q=75) ---

bench_encode "ducks_q75" \
    "$SEQ/ducks_take_off/frame_%04d.png" \
    -q 75 -n 300 --keyframe-interval 8 --fps 50

bench_encode "rush_hour" \
    "$SEQ/rush_hour/frame_%04d.png" \
    -q 75 -n 200 --keyframe-interval 8 --fps 25

bench_encode "old_town_cross" \
    "$SEQ/old_town_cross/frame_%04d.png" \
    -q 75 -n 200 --keyframe-interval 8 --fps 50

bench_encode "pedestrian_area" \
    "$SEQ/pedestrian_area/frame_%04d.png" \
    -q 75 -n 200 --keyframe-interval 8 --fps 25

# --- Long-form ---

bench_encode "bbb_2min" \
    "$SEQ/bbb_2min/frame_%04d.png" \
    -q 75 -n 1800 --keyframe-interval 24 --fps 30

bench_encode "bbb_2min_q5" \
    "$SEQ/bbb_2min/frame_%04d.png" \
    -q 5 -n 1800 --keyframe-interval 24 --fps 30

# --- Chroma subsampling: Crowd Run ---

bench_encode "crowd_444_q50" \
    "$SEQ/crowd_run/frame_%04d.png" \
    -q 50 -n 100 --keyframe-interval 8 --fps 50 --chroma-format 444

bench_encode "crowd_422_q50" \
    "$SEQ/crowd_run/frame_%04d.png" \
    -q 50 -n 100 --keyframe-interval 8 --fps 50 --chroma-format 422

bench_encode "crowd_420_q50" \
    "$SEQ/crowd_run/frame_%04d.png" \
    -q 50 -n 100 --keyframe-interval 8 --fps 50 --chroma-format 420

bench_encode "crowd_444_q25" \
    "$SEQ/crowd_run/frame_%04d.png" \
    -q 25 -n 100 --keyframe-interval 8 --fps 50 --chroma-format 444

bench_encode "crowd_422_q25" \
    "$SEQ/crowd_run/frame_%04d.png" \
    -q 25 -n 100 --keyframe-interval 8 --fps 50 --chroma-format 422

bench_encode "crowd_420_q25" \
    "$SEQ/crowd_run/frame_%04d.png" \
    -q 25 -n 100 --keyframe-interval 8 --fps 50 --chroma-format 420

# --- Chroma subsampling: Park Joy ---

bench_encode "park_joy_444_q50" \
    "$SEQ/park_joy/frame_%04d.png" \
    -q 50 -n 100 --keyframe-interval 8 --fps 50 --chroma-format 444

bench_encode "park_joy_422_q50" \
    "$SEQ/park_joy/frame_%04d.png" \
    -q 50 -n 100 --keyframe-interval 8 --fps 50 --chroma-format 422

bench_encode "park_joy_420_q50" \
    "$SEQ/park_joy/frame_%04d.png" \
    -q 50 -n 100 --keyframe-interval 8 --fps 50 --chroma-format 420

echo ""
echo "=== Summary ==="
echo ""
echo "GNV1 (I+P+B) files:"
ls -lhS "$OUT"/*.gnv 2>/dev/null || echo "  (none)"
echo ""
echo "Serve with: cd examples/web && bash serve.sh"
