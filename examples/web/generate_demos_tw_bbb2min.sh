#!/usr/bin/env bash
#
# Generate long-form bbb_2min GNV2 demos (temporal wavelet).
# Separated from generate_demos_tw.sh because these are slow/memory-intensive.
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
        echo "  (symlinked frame_0000 -> frame_0001 in $(basename "$dir"))"
    fi
}

encode_tw() {
    local name="$1"
    local input="$2"
    shift 2

    local outfile="$OUT/${name}.gnv2"
    local diagfile="$OUT/${name}.diag"
    echo ""
    echo "=== ${name}.gnv2 ==="
    echo "  input: ${input}"
    echo "  args:  $*"

    # Ensure source directory is zero-indexed
    ensure_zero_indexed "$(dirname "$input")"

    local start=$SECONDS
    "$GNC" benchmark-sequence -i "$input" -o "$outfile" --diagnostics "$@" 2>"$diagfile"
    local elapsed=$(( SECONDS - start ))

    local size
    size=$(ls -lh "$outfile" | awk '{print $5}')
    local diag_gops
    diag_gops=$(grep -c '^GOP ' "$diagfile" 2>/dev/null || echo 0)
    echo "  done: ${size} in ${elapsed}s (${diag_gops} GOPs diagnosed -> ${name}.diag)"
}

echo "GNC demo file generator — bbb_2min GNV2 (temporal wavelet)"
echo "============================================================"
echo "output: $OUT"

encode_tw "tw_bbb_2min_q5" \
    "$SEQ/bbb_2min/bbb_2min.y4m" \
    -q 5 -n 1800 -k 8 --temporal-wavelet haar \
    --fps 30

encode_tw "tw_bbb_2min" \
    "$SEQ/bbb_2min/bbb_2min.y4m" \
    -q 75 -n 1800 -k 8 --temporal-wavelet haar \
    --fps 30

echo ""
echo "=== Summary ==="
echo ""
echo "bbb_2min GNV2 files:"
ls -lhS "$OUT"/tw_bbb_2min*.gnv2 2>/dev/null || echo "  (none)"
echo ""
echo "Diagnostics files:"
ls -lhS "$OUT"/tw_bbb_2min*.diag 2>/dev/null || echo "  (none)"
