#!/usr/bin/env bash
#
# Generate the demo set for the web player.
#
# Replaces the four scripts that used to live here (generate_demos.sh, generate_demos_tw.sh,
# generate_demos_tw_bbb2min.sh, generate_demos_chroma.sh). They referenced source clips that are
# no longer fetched and built 300-1800 frame demos that took an hour; this produces a browsable
# set in a few minutes.
#
# What each group demonstrates:
#   1. Quality range — q=25 to q=100. The range above q=92 was dead until 2026-09-06 (an rANS
#      constraint capped qstep at 2.0 for every coder); q=100 is bit-exact lossless.
#   2. Temporal mode — I+P against the Haar temporal wavelet on the same clip. Measured
#      2026-09-06 on three sequences: the wavelet is +2.4 to +5.4 dB PSNR at equal q.
#   3. Chroma format — 4:4:4 / 4:2:2 / 4:2:0 at matched q.
#
# Requires: cargo build --release, and test_material/frames/sequences populated.
# Output is gitignored. Re-run after any bitstream change.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GNC="$ROOT/target/release/gnc"
SEQ="$ROOT/test_material/frames/sequences"
OUT="$SCRIPT_DIR"

[ -x "$GNC" ] || { echo "Build first: cargo build --release" >&2; exit 1; }

# Pick the first clip that exists, so the script survives a different fetch.
pick() {
    for name in "$@"; do
        [ -f "$SEQ/$name/$name.y4m" ] && { echo "$SEQ/$name/$name.y4m"; return 0; }
    done
    return 1
}

CLIP_MAIN=$(pick bbb_2min big_buck_bunny bbb ducks_take_off) \
    || { echo "No source clip with a .y4m found under $SEQ" >&2; exit 1; }
CLIP_MOTION=$(pick ducks_take_off crowd_run park_joy) || CLIP_MOTION="$CLIP_MAIN"
CLIP_CHROMA=$(pick crowd_run park_joy ducks_take_off) || CLIP_CHROMA="$CLIP_MAIN"

echo "Sources:"
echo "  quality range : $(basename "$CLIP_MAIN")"
echo "  temporal      : $(basename "$CLIP_MOTION")"
echo "  chroma        : $(basename "$CLIP_CHROMA")"
echo

encode() {                       # encode <outfile> <source> <frames> [flags...]
    local out="$1" src="$2" frames="$3"; shift 3
    printf '  %-26s ' "$(basename "$out")"
    if "$GNC" benchmark-sequence -i "$src" -n "$frames" -k 8 --output "$out" "$@" \
         > "${out}.log" 2>&1; then
        printf '%12s bytes\n' "$(wc -c < "$out" | tr -d ' ')"
    else
        printf 'FAILED (%s)\n' "$(basename "$out").log"
    fi
}

echo "Removing previous demo files..."
rm -f "$OUT"/*.gnv "$OUT"/*.gnv2 "$OUT"/*.log

# --- Watchable material: whole clips at the default preset -----------------------------------
# The comparison groups below are deliberately short; these are the ones you actually watch.
echo "1/4  Full clips at q=75 — material to watch"
watch() {                        # watch <name> <clip> <frames> [flags...]
    local name="$1" clip="$2" frames="$3"; shift 3
    local src; src=$(pick "$clip") || { printf '  %-26s skipped (no source)\n' "$name.gnv"; return; }
    encode "$OUT/$name.gnv" "$src" "$frames" -q 75 "$@"
}
watch watch_rush_hour       rush_hour       200      # 8 s at 25 fps — city traffic, slow pan
watch watch_pedestrian_area pedestrian_area 200      # 8 s at 25 fps — walking crowd
watch watch_old_town_cross  old_town_cross  200      # 4 s at 50 fps — urban pan
watch watch_bbb_2min        bbb_2min        600      # 20 s at 30 fps — animation, varied scenes
# Ducks is ~6 bpp at q=75, so a full 300 frames would be half a gigabyte. q=50 keeps it viewable.
src_ducks=$(pick ducks_take_off) && encode "$OUT/watch_ducks_q50.gnv" "$src_ducks" 300 -q 50

echo
echo "2/4  Quality range — short and uniform, for A/B"
for q in 25 50 75 92; do
    encode "$OUT/range_q${q}.gnv" "$CLIP_MAIN" 24 -q "$q"
done
# Same frame count as the rest of the group. A shorter clip here made lossless look *cheaper*
# than q=92 in an earlier version of this script — bbb_2min opens on a quiet intro, so 8 frames
# is not comparable with 24. Keep the group uniform or the comparison it exists to show is false.
encode "$OUT/range_q100_lossless.gnv" "$CLIP_MAIN" 24 -q 100

echo
echo "3/4  Temporal mode — same clip, same q"
encode "$OUT/temporal_ip.gnv"    "$CLIP_MOTION" 24 -q 75 --temporal-wavelet none
encode "$OUT/temporal_haar.gnv2" "$CLIP_MOTION" 24 -q 75 --temporal-wavelet haar

echo
echo "4/4  Chroma format — same clip, same q"
for fmt in 444 422 420; do
    encode "$OUT/chroma_${fmt}.gnv" "$CLIP_CHROMA" 24 -q 50 --chroma-format "$fmt"
done

echo
echo "Writing demos.json manifest..."
python3 "$SCRIPT_DIR/write_manifest.py" "$SCRIPT_DIR"

echo
echo "Done. Serve with ./serve.sh, then open player.html"
