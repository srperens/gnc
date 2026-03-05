#!/usr/bin/env bash
#
# Generate demo .gnv files (I+P+B motion vectors) for the web player.
# Uses encode-sequence which is the I+P motion vector encoder (GNV1 format).
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
# The encoder expects frame_0000, so create a symlink if needed.
ensure_zero_indexed() {
    local dir="$1"
    if [ -f "$dir/frame_0001.png" ] && [ ! -f "$dir/frame_0000.png" ]; then
        ln -s frame_0001.png "$dir/frame_0000.png"
        echo "  (symlinked frame_0000 → frame_0001 in $(basename "$dir"))"
    fi
}

encode() {
    local name="$1"
    local input="$2"
    shift 2

    local outfile="$OUT/${name}.gnv"
    local diagfile="$OUT/${name}.diag"
    echo ""
    echo "=== ${name}.gnv ==="
    echo "  input: ${input}"
    echo "  args:  $*"

    # Ensure source directory is zero-indexed
    ensure_zero_indexed "$(dirname "$input")"

    local start=$SECONDS
    "$GNC" encode-sequence -i "$input" -o "$outfile" --diagnostics "$@" 2>"$diagfile"
    local elapsed=$(( SECONDS - start ))

    local size
    size=$(ls -lh "$outfile" | awk '{print $5}')
    local diag_frames
    diag_frames=$(grep -c '^Frame ' "$diagfile" 2>/dev/null || echo 0)
    echo "  done: ${size} in ${elapsed}s (${diag_frames} frames diagnosed → ${name}.diag)"
}

# Clean old GNV1 demo files
echo "GNC demo file generator (GNV1 — I+P+B)"
echo "========================================"
echo "output: $OUT"
echo ""
echo "Removing old .gnv files..."
rm -f "$OUT"/*.gnv
# Remove only GNV1 diag files (not tw_*.diag from generate_demos_tw.sh)
find "$OUT" -maxdepth 1 -name '*.diag' ! -name 'tw_*.diag' -delete

# --- Quick test files (small, fast to encode) ---

encode "test_quick" \
    "$SEQ/bbb/frame_%04d.png" \
    -q 75 -n 10 --keyframe-interval 4

encode "test_animation" \
    "$SEQ/bbb_extended/frame_%04d.png" \
    -q 75 -n 30 --keyframe-interval 8

encode "test_nature" \
    "$SEQ/park_joy/frame_%04d.png" \
    -q 75 -n 30 --keyframe-interval 8 \
    --fps-num 50 --fps-den 1

encode "test_crowd" \
    "$SEQ/crowd_run/frame_%04d.png" \
    -q 75 -n 30 --keyframe-interval 8 \
    --fps-num 50 --fps-den 1

# --- Quality comparison (same content, different q) ---

encode "ducks_q25" \
    "$SEQ/ducks_take_off/frame_%04d.png" \
    -q 25 -n 300 --keyframe-interval 8 \
    --fps-num 50 --fps-den 1

encode "ducks_q50" \
    "$SEQ/ducks_take_off/frame_%04d.png" \
    -q 50 -n 300 --keyframe-interval 8 \
    --fps-num 50 --fps-den 1

encode "ducks_q75" \
    "$SEQ/ducks_take_off/frame_%04d.png" \
    -q 75 -n 300 --keyframe-interval 8 \
    --fps-num 50 --fps-den 1

# --- Broadcast test sequences ---

encode "rush_hour" \
    "$SEQ/rush_hour/frame_%04d.png" \
    -q 75 -n 200 --keyframe-interval 8 \
    --fps-num 25 --fps-den 1

encode "old_town_cross" \
    "$SEQ/old_town_cross/frame_%04d.png" \
    -q 75 -n 200 --keyframe-interval 8 \
    --fps-num 50 --fps-den 1

encode "stockholm" \
    "$SEQ/stockholm/frame_%04d.png" \
    -q 75 -n 200 --keyframe-interval 8 \
    --fps-num 60000 --fps-den 1001

encode "pedestrian_area" \
    "$SEQ/pedestrian_area/frame_%04d.png" \
    -q 75 -n 200 --keyframe-interval 8 \
    --fps-num 25 --fps-den 1

# --- Long-form demos ---

encode "bbb_2min_q5" \
    "$SEQ/bbb_2min/frame_%04d.png" \
    -q 5 -n 1800 --keyframe-interval 24 \
    --fps-num 30 --fps-den 1

encode "bbb_2min" \
    "$SEQ/bbb_2min/frame_%04d.png" \
    -q 75 -n 1800 --keyframe-interval 24 \
    --fps-num 30 --fps-den 1

echo ""
echo "=== Summary ==="
echo ""
echo "GNV1 (I+P+B) files:"
ls -lhS "$OUT"/*.gnv 2>/dev/null || echo "  (none)"
echo ""
echo "Diagnostics files:"
ls -lhS "$OUT"/*.diag 2>/dev/null || echo "  (none)"
echo ""
echo "Serve with: cd examples/web && bash serve.sh"
