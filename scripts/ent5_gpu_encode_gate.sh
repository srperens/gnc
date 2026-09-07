#!/usr/bin/env bash
# ENT-5 gate: the GPU abac encoder must produce the CPU encoder's exact bytes.
#
# Bit-exactness is the criterion, not "decodes to the same picture". A GPU encoder that emitted a
# different-but-valid stream would silently move every rate figure abac has (-16.0% over q=60-99
# intra, -13.4% lossless) and nothing would fail. Whole-file byte identity against `--cpu-encode`
# is the strongest available form of that check, and it also discharges "rate unchanged" by
# construction: identical bytes cannot be a different rate.
#
# Covers both arithmetic engines, both output-sizing modes, all three chroma formats, the lossy
# range and bit-exact lossless. Compression figures only — no wall-clock — so it is valid on a
# loaded machine (COORDINATION.md, "the machine is shared").
set -uo pipefail
REPO=$(git rev-parse --show-toplevel)
GNC=$REPO/target/release/gnc
OUT=${1:-$(mktemp -d)}
mkdir -p "$OUT"

IMAGES=(bbb_1080p blue_sky_1080p kristensara_720p touchdown_1080p)
fail=0
pass=0

check() {  # check <label> <extra encode args...>
    local label=$1; shift
    "$GNC" encode "$@" -o "$OUT/gpu.gnc" >/dev/null 2>&1 || { echo "FAIL $label: gpu encode errored"; fail=$((fail+1)); return; }
    "$GNC" encode "$@" --cpu-encode -o "$OUT/cpu.gnc" >/dev/null 2>&1 || { echo "FAIL $label: cpu encode errored"; fail=$((fail+1)); return; }
    if cmp -s "$OUT/gpu.gnc" "$OUT/cpu.gnc"; then
        printf 'ok   %-58s %9d B\n' "$label" "$(wc -c <"$OUT/gpu.gnc")"
        pass=$((pass+1))
    else
        echo "FAIL $label: GPU and CPU encoders produced different bytes"
        fail=$((fail+1))
    fi
}

for img in "${IMAGES[@]}"; do
    src=$REPO/test_material/frames/$img.png
    [ -f "$src" ] || { echo "skip $img (not in test_material)"; continue; }
    for q in 60 75 90 99 100; do
        for coder in range interval; do
            for sizing in count slots; do
                GNC_ABAC_CODER=$coder GNC_ABAC_GPU_SIZING=$sizing \
                    check "$img q=$q $coder/$sizing" -i "$src" -q "$q" --abac
            done
        done
    done
    for cf in 422 420; do
        check "$img q=90 4:$cf" -i "$src" -q 90 --abac --chroma-format "$cf"
    done
    for cb in 16 32; do
        GNC_ABAC_CB=$cb check "$img q=90 cb=$cb" -i "$src" -q 90 --abac
    done
done

# The sequence path reaches `encode_entropy` from two more call sites (I-frames and P-frames in
# `sequence.rs`), and `encode` cannot exercise either. ki=9 codes P-frame residuals, which is a
# different coefficient distribution and the only place the inter arm is checked. Both arms take
# the same frame pipeline — for abac `use_gpu_encode` is false either way, deliberately, so
# quantisation is identical and the only difference is where the entropy coding runs.
SEQ=$REPO/test_material/frames/sequences/bbb/frame_%04d.png
if [ -f "$REPO/test_material/frames/sequences/bbb/frame_0000.png" ]; then
    for ki in 1 9; do
        "$GNC" encode-sequence -i "$SEQ" -o "$OUT/seq_gpu.gnv" -q 90 -n 8 \
            --keyframe-interval "$ki" --abac >/dev/null 2>&1
        "$GNC" encode-sequence -i "$SEQ" -o "$OUT/seq_cpu.gnv" -q 90 -n 8 \
            --keyframe-interval "$ki" --abac --cpu-encode >/dev/null 2>&1
        if cmp -s "$OUT/seq_gpu.gnv" "$OUT/seq_cpu.gnv"; then
            printf 'ok   %-58s %9d B\n' "bbb sequence 8 frames ki=$ki" "$(wc -c <"$OUT/seq_gpu.gnv")"
            pass=$((pass+1))
        else
            echo "FAIL bbb sequence ki=$ki: GPU and CPU encoders produced different bytes"
            fail=$((fail+1))
        fi
    done
else
    echo "skip bbb sequence (not in test_material)"
fi

echo "---"
echo "$pass identical, $fail differing"
[ "$fail" -eq 0 ]
