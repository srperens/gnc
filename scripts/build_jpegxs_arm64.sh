#!/usr/bin/env bash
#
# Build SVT-JPEG-XS on Apple Silicon (or any non-x86 host) for MEAS-9.
#
# JPEG XS is the codec that decides GNC's positioning (docs/POSITIONING.md, BACKLOG MEAS-9):
# in the contribution segment the incumbents are JPEG XS, J2K, VC-2 and ProRes, and x264 is a
# sanity anchor rather than a competitor. It is not in Homebrew, and SVT-JPEG-XS's CMake assumes
# x86-64 unconditionally -- `add_definitions(-DARCH_X86_64=1)`, `enable_language(ASM_NASM)`, and
# nine ASM object libraries wired straight into every target.
#
# It is only the *build system* that assumes x86. The C sources already carry `#else /*
# ARCH_X86_64 */` fallbacks for every dispatch, so the scalar path is written and just never
# selected. svt-jpegxs-arm64.patch makes the arch a decision instead of an assumption:
#
#   * detect x86 once into SVT_ARCH_X86 and gate the nasm/yasm discovery, `-DARCH_X86_64` and
#     the nine ASM subdirectories on it;
#   * define a non-x86 `get_cpu_flags()` returning 0 (EncHandle/DecHandle call it unconditionally,
#     so it has to exist on every arch) and declare it outside the x86 guard;
#   * guard the SIMD kernel headers, which pull in <immintrin.h> and do not compile on arm64.
#
# Verified 2026-09-07 on an M1: builds clean, and a 1920x1080 yuv422p frame at --bpp 3 round-trips
# to **PSNR y 44.48 dB** -- a sane JPEG XS result, so the scalar path is correct and not merely
# present. Throughput from this build is NOT comparable to a tuned x86 build and must not be
# quoted: every SIMD kernel is disabled. Rate and quality are exact and are what MEAS-9 needs.

set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
PATCH="$REPO/scripts/svt-jpegxs-arm64.patch"
PREFIX="${1:-${TMPDIR:-/tmp}/svt-jpegxs}"
UPSTREAM=96f8f06e65011042b265221c8558a46090c679bd   # the commit the patch was made against

for cmd in git cmake ffmpeg; do
    command -v "$cmd" >/dev/null || { echo "Error: $cmd is required"; exit 1; }
done
[ -f "$PATCH" ] || { echo "Error: $PATCH not found"; exit 1; }

if [ ! -d "$PREFIX/.git" ]; then
    echo "=== cloning SVT-JPEG-XS into $PREFIX ==="
    git clone https://github.com/OpenVisualCloud/SVT-JPEG-XS.git "$PREFIX"
fi

cd "$PREFIX"
if ! git cat-file -e "$UPSTREAM^{commit}" 2>/dev/null; then
    echo "note: pinned upstream commit not in this clone; patching whatever is checked out"
else
    git checkout -q "$UPSTREAM"
fi

if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "=== applied arm64 patch ==="
else
    echo "=== patch already applied (or does not apply); continuing ==="
fi

cmake -S . -B Build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build Build -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

BIN="$PREFIX/Bin/Release"
echo ""
echo "=== verifying, because a binary that links is not a codec that works ==="
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
SRC="$REPO/test_material/frames/bbb_1080p.png"
if [ ! -f "$SRC" ]; then
    echo "  ! $SRC missing — run test_material/fetch_test_frames.sh; skipping verification"
    exit 0
fi
ffmpeg -nostdin -y -loglevel error -i "$SRC" -pix_fmt yuv422p -f rawvideo "$WORK/in.yuv"
"$BIN/SvtJpegxsEncApp" -i "$WORK/in.yuv" -w 1920 -h 1080 --colour-format yuv422 \
    --input-depth 8 --bpp 3 -n 1 -b "$WORK/out.jxs" >/dev/null 2>&1
"$BIN/SvtJpegxsDecApp" -i "$WORK/out.jxs" -o "$WORK/out.yuv" >/dev/null 2>&1
PSNR=$(ffmpeg -nostdin -v info \
    -f rawvideo -pix_fmt yuv422p -s 1920x1080 -i "$WORK/in.yuv" \
    -f rawvideo -pix_fmt yuv422p -s 1920x1080 -i "$WORK/out.yuv" \
    -lavfi psnr -f null - 2>&1 | grep -o 'y:[0-9.]*' | tail -1 | cut -d: -f2)
echo "  round-trip at --bpp 3: PSNR y ${PSNR} dB (expected ~44.5 on bbb_1080p)"
awk -v p="${PSNR:-0}" 'BEGIN{exit !(p>40)}' \
    || { echo "  ! that is not a working codec — do not measure with this build"; exit 1; }
echo ""
echo "=== ok ==="
echo "  encoder: $BIN/SvtJpegxsEncApp"
echo "  decoder: $BIN/SvtJpegxsDecApp"
echo "  Rate and quality are exact. Throughput is NOT — every SIMD kernel is off on this arch."
