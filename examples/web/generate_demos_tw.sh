#!/usr/bin/env bash
#
# Generate demo .gnv2 files (temporal wavelet Haar) for the web player.
# Uses benchmark-sequence --temporal-wavelet haar (must be explicit; default is none/I+P+B).
# Mirrors generate_demos.sh with matching content for direct GNV1 vs GNV2 comparison.
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
    echo "  done: ${size} in ${elapsed}s (${diag_gops} GOPs diagnosed → ${name}.diag)"
}

# Clean old GNV2 demo files
echo "GNC demo file generator (GNV2 — temporal wavelet)"
echo "==================================================="
echo "output: $OUT"
echo ""
echo "Removing old tw_*.gnv2 and tw_*.diag files (excluding tw_bbb_2min*)..."
for f in "$OUT"/tw_*.gnv2 "$OUT"/tw_*.diag; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in
        tw_bbb_2min*) ;; # skip — managed by generate_demos_tw_bbb2min.sh
        *) rm -f "$f" ;;
    esac
done

# --- Quick test files (matching generate_demos.sh) ---

encode_tw "tw_test_quick" \
    "$SEQ/bbb/bbb.y4m" \
    -q 75 -n 8 -k 8 --temporal-wavelet haar \
    --fps 30

encode_tw "tw_test_animation" \
    "$SEQ/bbb_extended/bbb_extended.y4m" \
    -q 75 -n 24 -k 8 --temporal-wavelet haar \
    --fps 30

encode_tw "tw_test_nature" \
    "$SEQ/park_joy/park_joy.y4m" \
    -q 75 -n 32 -k 8 --temporal-wavelet haar \
    --fps 50

encode_tw "tw_test_crowd" \
    "$SEQ/crowd_run/crowd_run.y4m" \
    -q 75 -n 32 -k 8 --temporal-wavelet haar \
    --fps 50

# --- Quality comparison (same content, different q) ---

encode_tw "tw_ducks_q25" \
    "$SEQ/ducks_take_off/ducks_take_off.y4m" \
    -q 25 -n 300 -k 8 --temporal-wavelet haar \
    --fps 50

encode_tw "tw_ducks_q50" \
    "$SEQ/ducks_take_off/ducks_take_off.y4m" \
    -q 50 -n 300 -k 8 --temporal-wavelet haar \
    --fps 50

encode_tw "tw_ducks_q75" \
    "$SEQ/ducks_take_off/ducks_take_off.y4m" \
    -q 75 -n 300 -k 8 --temporal-wavelet haar \
    --fps 50

# --- Broadcast test sequences ---

encode_tw "tw_rush_hour" \
    "$SEQ/rush_hour/rush_hour.y4m" \
    -q 75 -n 200 -k 8 --temporal-wavelet haar \
    --fps 25

encode_tw "tw_old_town_cross" \
    "$SEQ/old_town_cross/old_town_cross.y4m" \
    -q 75 -n 200 -k 8 --temporal-wavelet haar \
    --fps 50

encode_tw "tw_stockholm" \
    "$SEQ/stockholm/stockholm.y4m" \
    -q 75 -n 200 -k 8 --temporal-wavelet haar \
    --fps 60

encode_tw "tw_pedestrian_area" \
    "$SEQ/pedestrian_area/pedestrian_area.y4m" \
    -q 75 -n 200 -k 8 --temporal-wavelet haar \
    --fps 25

# --- Long-form demos ---
# bbb_2min is in generate_demos_tw_bbb2min.sh (separate due to memory/time)

echo ""
echo "=== Summary ==="
echo ""
echo "GNV2 (temporal wavelet) files:"
ls -lhS "$OUT"/*.gnv2 2>/dev/null || echo "  (none)"
echo ""
echo "Diagnostics files:"
ls -lhS "$OUT"/tw_*.diag 2>/dev/null || echo "  (none)"
echo ""
echo "Serve with: cd examples/web && bash serve.sh"
