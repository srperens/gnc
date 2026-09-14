#!/usr/bin/env bash
#
# Cut a full-length film into player-sized segments and encode each one.
#
# Why segments and not one file: player.html accumulates the whole container into a Uint8Array
# before it can create a decoder, so file size becomes tab memory. Big Buck Bunny at 1080p q=50
# is roughly 4 GB whole, which no browser tab should be asked to hold. Thirty-second pieces are
# ~200 MB each, and the player's auto-advance walks them in manifest order, so it plays as a film.
#
# Source: Big Buck Bunny, (c) Blender Foundation, CC-BY 3.0 — an open movie made to be used this
# way. Pass the file as $1 or in $GNC_MOVIE; it is not downloaded here and not committed.
#
#   ./generate_movie.sh ~/somewhere/big_buck_bunny_1080p_h264.mov
#   SEG_SECONDS=20 QUALITY=75 ./generate_movie.sh <file>
#
# Requires: ffmpeg, and cargo build --release. Output is gitignored.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GNC="$ROOT/target/release/gnc"
OUT="$SCRIPT_DIR"
MOVIE="${1:-${GNC_MOVIE:-}}"
SEG_SECONDS="${SEG_SECONDS:-30}"
QUALITY="${QUALITY:-50}"
PREFIX="${PREFIX:-movie_bbb}"
# One keyframe a second. `-k 8` — what the comparison demos use — spends 97 I-frames per 720 on
# a film and buys nothing a viewer can see: measured on one 30 s segment, ki 8 -> 24 is -10.5%
# of rate (94.9 -> 84.9 MB) and ki 48 only 1.5% beyond that, while making seeks coarser.
KI="${KI:-24}"

[ -n "$MOVIE" ] || { echo "usage: $0 <movie file>   (or set GNC_MOVIE)" >&2; exit 1; }
[ -f "$MOVIE" ] || { echo "No such file: $MOVIE" >&2; exit 1; }
[ -x "$GNC" ]   || { echo "Build first: cargo build --release" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }

# `-of csv=p=0` appends a trailing comma for the stream section, so `24/1` arrives as `24/1,`
# and every arithmetic use of it fails. `default=nw=1` prints `key=value` and is unambiguous.
probe() { ffprobe -v error -select_streams v:0 -show_entries "$1" -of default=nw=1 "$MOVIE" \
            | head -1 | cut -d= -f2; }
DUR=$(probe format=duration)
FPS=$(probe stream=r_frame_rate)
VW=$(probe stream=width)
VH=$(probe stream=height)
SEGMENTS=$(python3 -c "import math,sys; print(math.ceil(float(sys.argv[1])/float(sys.argv[2])))" "$DUR" "$SEG_SECONDS")
# `benchmark-sequence` encodes 10 frames when `-n` is absent, so a segment without it is a
# segment of one second. Derive the count from the source's own frame rate.
SEG_FRAMES=$(python3 -c "import sys; n,d=sys.argv[1].split('/'); print(round(float(n)/float(d)*float(sys.argv[2])))" "$FPS" "$SEG_SECONDS")

# A bad probe used to take the whole run down one line later, inside a command substitution,
# with nothing written and the exit code swallowed by whatever the caller piped into.
case "$SEG_FRAMES" in ''|*[!0-9]*) echo "Could not derive a frame count (dur='$DUR' fps='$FPS')" >&2; exit 1 ;; esac
[ "$SEG_FRAMES" -gt 0 ] || { echo "Frame count is zero (fps='$FPS')" >&2; exit 1; }

echo "Source : $(basename "$MOVIE")  ${DUR}s @ ${FPS}"
echo "Cutting: ${SEGMENTS} x ${SEG_SECONDS}s (${SEG_FRAMES} frames) at q=${QUALITY}, ki=${KI}"
echo

# One scratch y4m at a time — a whole 1080p film as Y4M is ~45 GB, a segment is ~2 GB.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

rm -f "$OUT/${PREFIX}"_*.gnv "$OUT/${PREFIX}"_*.gnv.log

total=0
for i in $(seq 0 $((SEGMENTS - 1))); do
    start=$((i * SEG_SECONDS))
    name=$(printf '%s_%02d.gnv' "$PREFIX" "$i")
    printf '  %-22s %4ds ' "$name" "$start"

    # -ss before -i seeks on the input, which is what keeps this from re-decoding the whole film
    # once per segment. Accurate enough here: the container is all-keyframe-seekable h264.
    if ! ffmpeg -v error -y -ss "$start" -t "$SEG_SECONDS" -i "$MOVIE" \
            -pix_fmt yuv420p "$TMP/seg.y4m" 2>"$OUT/$name.log"; then
        printf 'FFMPEG FAILED (%s)\n' "$name.log"; continue
    fi
    if [ ! -s "$TMP/seg.y4m" ]; then printf 'empty segment, stopping\n'; break; fi

    # Count what ffmpeg actually wrote rather than assuming a full segment. The tail is short —
    # 596.46s does not divide by 30 — and asking for more frames than the file holds is a *panic*,
    # not a short encode: "frame 663 requested but Y4M EOF was at frame 635". Derived from the
    # file because we wrote it: yuv420p is w*h*3/2 per frame plus a 6-byte "FRAME\n".
    n=$(python3 -c "
import sys
path, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, 'rb') as f:
    header = len(f.readline())
import os
print((os.path.getsize(path) - header) // (w * h * 3 // 2 + 6))
" "$TMP/seg.y4m" "$VW" "$VH")
    [ "$n" -gt 0 ] || { printf 'no frames in segment, stopping\n'; break; }
    [ "$n" -eq "$SEG_FRAMES" ] || printf '(%s frames) ' "$n"

    if "$GNC" benchmark-sequence -i "$TMP/seg.y4m" -n "$n" -k "$KI" -q "$QUALITY" \
            --output "$OUT/$name" >>"$OUT/$name.log" 2>&1; then
        sz=$(wc -c < "$OUT/$name" | tr -d ' ')
        total=$((total + sz))
        printf '%12s bytes\n' "$sz"
    else
        printf 'ENCODE FAILED (%s)\n' "$name.log"
    fi
    rm -f "$TMP/seg.y4m"
done

echo
echo "Total: $((total / 1000000)) MB across $SEGMENTS segments"
python3 "$SCRIPT_DIR/write_manifest.py" "$SCRIPT_DIR"
echo "Serve with ./serve.sh, tick Auto-advance, pick the first segment."
