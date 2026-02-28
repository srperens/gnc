#!/usr/bin/env bash
# fetch_test_media.sh — Download standard codec test sequences from media.xiph.org
#
# Usage:
#   ./scripts/fetch_test_media.sh                  # download all default sequences
#   ./scripts/fetch_test_media.sh ducks_take_off    # download one sequence
#   ./scripts/fetch_test_media.sh ducks_take_off crowd_run  # download specific ones
#   ./scripts/fetch_test_media.sh --list            # list available sequences
#
# Requires: ffmpeg, curl
#
# Streams y4m from URL, extracts first 30 frames as PNG. Does NOT download
# the entire file — ffmpeg+curl exit early once enough frames are captured.
#
# Output:
#   test_material/frames/sequences/<name>/frame_NNNN.png  (30 frames)
#   test_material/frames/<name>_1080p.png                 (1 representative still)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRAMES_DIR="$PROJECT_DIR/test_material/frames"
SEQ_DIR="$FRAMES_DIR/sequences"

FRAME_COUNT=30
BASE_URL="https://media.xiph.org/video/derf/y4m"

# ---- Sequence catalog ----
# Each entry: name|y4m_filename|frames|description
# frames=0 means use FRAME_COUNT default (30)

CATALOG="ducks_take_off|ducks_take_off_1080p50.y4m|0|High motion + fine detail (classic stress test)
crowd_run|crowd_run_1080p50.y4m|0|Complex crowd scene, many moving objects
park_joy|park_joy_1080p50.y4m|0|Foliage texture + motion (hard to compress)
sintel_trailer|sintel_trailer_2k_1080p24.y4m|0|CGI, mixed dark/bright scenes
big_buck_bunny|big_buck_bunny_1080p24.y4m.xz|2880|BBB 2 min @ 24fps — animation reference (xz-compressed)"

DEFAULT_TARGETS="ducks_take_off crowd_run park_joy sintel_trailer"

lookup_y4m() {
    echo "$CATALOG" | while IFS='|' read -r n f frames d; do
        if [ "$n" = "$1" ]; then echo "$f"; return; fi
    done
}

lookup_frames() {
    echo "$CATALOG" | while IFS='|' read -r n f frames d; do
        if [ "$n" = "$1" ]; then
            if [ "$frames" -gt 0 ] 2>/dev/null; then echo "$frames"; else echo "$FRAME_COUNT"; fi
            return
        fi
    done
}

lookup_desc() {
    echo "$CATALOG" | while IFS='|' read -r n f frames d; do
        if [ "$n" = "$1" ]; then echo "$d"; return; fi
    done
}

list_sequences() {
    echo "Available sequences:"
    echo "$CATALOG" | while IFS='|' read -r n f frames d; do
        if [ "$frames" -gt 0 ] 2>/dev/null; then
            printf "  %-20s %s (%d frames)\n" "$n" "$d" "$frames"
        else
            printf "  %-20s %s (%d frames)\n" "$n" "$d" "$FRAME_COUNT"
        fi
    done
}

# ---- Preflight checks ----

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Error: ffmpeg is required. Install with: brew install ffmpeg"
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required."
    exit 1
fi

# ---- Parse args ----

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ]; then
    list_sequences
    exit 0
fi

if [ $# -gt 0 ]; then
    TARGETS="$*"
else
    TARGETS="$DEFAULT_TARGETS"
fi

# Validate targets
for name in $TARGETS; do
    y4m=$(lookup_y4m "$name")
    if [ -z "$y4m" ]; then
        echo "Error: unknown sequence '$name'"
        list_sequences
        exit 1
    fi
done

mkdir -p "$FRAMES_DIR" "$SEQ_DIR"

# ---- Download and extract ----

total_bytes=0

for name in $TARGETS; do
    y4m_file=$(lookup_y4m "$name")
    desc=$(lookup_desc "$name")
    target_frames=$(lookup_frames "$name")
    url="$BASE_URL/$y4m_file"
    out_dir="$SEQ_DIR/$name"
    still_frame="$FRAMES_DIR/${name}_1080p.png"

    echo ""
    echo "=== $name ==="
    echo "    $desc"

    # Check if already extracted
    if [ -d "$out_dir" ]; then
        count=$(find "$out_dir" -name 'frame_*.png' | wc -l | tr -d ' ')
        if [ "$count" -ge "$target_frames" ]; then
            echo "    Already extracted ($count frames). Skipping."
            size=$(du -sk "$out_dir" | awk '{print $1}')
            total_bytes=$((total_bytes + size))
            continue
        fi
    fi

    mkdir -p "$out_dir"

    # Stream y4m directly from URL → ffmpeg, extract only target_frames.
    # curl streams the data; ffmpeg reads only what it needs and both exit early.
    # For .xz files, decompress on the fly via xz -d.
    echo "    Streaming $y4m_file → extracting $target_frames frames ..."
    set +e
    if [[ "$y4m_file" == *.xz ]]; then
        # xz-compressed: need xz to decompress the stream
        if ! command -v xz >/dev/null 2>&1; then
            echo "    ERROR: xz is required for $y4m_file. Install with: brew install xz"
            set -e
            continue
        fi
        curl -sL "$url" | xz -d | ffmpeg -hide_banner -loglevel error \
            -i pipe:0 \
            -frames:v "$target_frames" \
            -y \
            "$out_dir/frame_%04d.png"
        ffmpeg_exit=${PIPESTATUS[2]:-$?}
    else
        curl -sL "$url" | ffmpeg -hide_banner -loglevel error \
            -i pipe:0 \
            -frames:v "$target_frames" \
            -y \
            "$out_dir/frame_%04d.png"
        ffmpeg_exit=${PIPESTATUS[1]:-$?}
    fi
    set -e

    if [ "$ffmpeg_exit" -ne 0 ]; then
        echo "    WARNING: ffmpeg exited with code $ffmpeg_exit"
    fi

    count=$(find "$out_dir" -name 'frame_*.png' | wc -l | tr -d ' ')
    echo "    Extracted $count frames."

    # Copy a representative still frame (middle of extracted range)
    mid=$((target_frames / 2))
    rep_frame=$(printf "$out_dir/frame_%04d.png" "$mid")
    if [ -f "$rep_frame" ] && [ ! -f "$still_frame" ]; then
        cp "$rep_frame" "$still_frame"
        echo "    Still frame → ${name}_1080p.png"
    fi

    size=$(du -sk "$out_dir" | awk '{print $1}')
    total_bytes=$((total_bytes + size))
    echo "    Disk: $(du -sh "$out_dir" | awk '{print $1}')"
done

echo ""
echo "=== Summary ==="
echo "Sequences: $SEQ_DIR"
total_mb=$(echo "scale=1; $total_bytes / 1024" | bc 2>/dev/null || echo "${total_bytes}KB")
echo "Total disk: ${total_mb}MB"
echo "Done."
