#!/usr/bin/env bash
#
# Generate demo .gnv files comparing 4:4:4, 4:2:2, and 4:2:0 chroma subsampling.
# Short clips — enough to verify decode + visual quality in the browser.
#
# Uses single-frame benchmark (--single-frame) to generate I-frame-only .gnv files
# since encode-sequence does not yet support non-444 chroma (I-frame local decode
# for P-frame ME reference has not been updated for subsampled chroma).
#
# Requires: cargo build --release (gnc binary)
# Source material: test_material/frames/ (bbb_1080p.png, blue_sky_1080p.png)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GNC="$ROOT/target/release/gnc"
FRAMES="$ROOT/test_material/frames"
OUT="$SCRIPT_DIR"

if [ ! -x "$GNC" ]; then
    echo "Building gnc..."
    (cd "$ROOT" && cargo build --release)
fi

# Single-frame encode → single-frame .gnv for browser decode
encode_single() {
    local name="$1"
    local input="$2"
    local q="$3"
    local fmt="$4"

    local outfile="$OUT/${name}.gnv"
    echo "=== ${name}.gnv ==="
    "$GNC" encode -i "$input" -o "$outfile" -q "$q" --chroma-format "$fmt" 2>/dev/null
    local size
    size=$(ls -lh "$outfile" | awk '{print $5}')
    echo "  done: ${size}  (${fmt}, q=${q})"
}

echo "GNC chroma subsampling demos (4:4:4 vs 4:2:2 vs 4:2:0)"
echo "========================================================="
echo "output: $OUT"
echo ""

# Remove old chroma demo files
rm -f "$OUT"/chroma_*.gnv

# --- bbb: format comparison at q=75 ---
encode_single "chroma_bbb_444" "$FRAMES/bbb_1080p.png"      75 444
encode_single "chroma_bbb_422" "$FRAMES/bbb_1080p.png"      75 422
encode_single "chroma_bbb_420" "$FRAMES/bbb_1080p.png"      75 420

# --- blue_sky: chromatic content, useful for spotting chroma artifacts ---
encode_single "chroma_sky_444" "$FRAMES/blue_sky_1080p.png" 75 444
encode_single "chroma_sky_422" "$FRAMES/blue_sky_1080p.png" 75 422
encode_single "chroma_sky_420" "$FRAMES/blue_sky_1080p.png" 75 420

# --- bbb at q=50 (lower quality — chroma artifacts more visible) ---
encode_single "chroma_bbb_q50_444" "$FRAMES/bbb_1080p.png"  50 444
encode_single "chroma_bbb_q50_422" "$FRAMES/bbb_1080p.png"  50 422
encode_single "chroma_bbb_q50_420" "$FRAMES/bbb_1080p.png"  50 420

echo ""
echo "=== Summary ==="
ls -lhS "$OUT"/chroma_*.gnv 2>/dev/null || echo "  (none)"
echo ""
echo "Serve with: cd examples/web && bash serve.sh"
