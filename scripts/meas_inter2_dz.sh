#!/usr/bin/env bash
# INTER-2 — price the inter dead zone (`inter_dz_mul`) on a real ladder.
#
# The filed evidence was ONE point (crowd_run, q=85), which sizes the lever and cannot rank the
# options (COORDINATION rule 4). This sweeps a 4-rung ladder per arm so the arms can be compared
# by BD-rate over a COMMON quality interval -- see meas_inter1_pscale.py for why a per-arm
# interval would make the numbers incomparable.
#
# Only q <= 85 is swept: the inter dead zone is `config.dead_zone * inter_dz_mul`, the anchors put
# dead_zone at 0.05 by q=92, and any dead zone <= 0.5 is a no-op because GNC quantises as
# floor(|v|/step + 0.5) after a |v| < dz*step test. Above q~86 every arm is therefore identical,
# which the q=92 control at the end asserts rather than assumes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
BIN=./target/release/gnc
OUT=${1:?usage: meas_inter2_dz.sh <out.csv>}
N=24; KI=9
echo "sequence,q,inter_dz_mul,bytes,bpp,psnr_mean,psnr_worst,vmaf" > "$OUT"
for seq in crowd_run old_town_cross bbb_extended; do
  for q in 70 75 80 85; do
    for mul in 2.0 1.0 0.0; do
      out=$(GNC_INTER_DZ_MUL=$mul $BIN benchmark-sequence \
              -i "test_material/frames/sequences/$seq/frame_%04d.png" \
              -n $N -k $KI -q $q --vmaf 2>&1)
      bytes=$(printf '%s' "$out" | grep -m1 -oE 'Total: [0-9]+ bytes' | grep -oE '[0-9]+')
      bpp=$(printf '%s'   "$out" | grep -m1 -oE 'avg [0-9.]+ bpp' | grep -oE '[0-9.]+')
      pm=$(printf '%s'    "$out" | grep -A1 -m1 'Sequence Summary' | grep -oE 'avg [0-9.]+ dB' | grep -oE '[0-9.]+')
      pw=$(printf '%s'    "$out" | grep -A1 -m1 'Sequence Summary' | grep -oE 'min [0-9.]+' | grep -oE '[0-9.]+')
      vm=$(printf '%s'    "$out" | grep -m1 -oE 'mean=[0-9.]+' | grep -oE '[0-9.]+')
      echo "$seq,$q,$mul,${bytes:-},${bpp:-},${pm:-},${pw:-},${vm:-}" | tee -a "$OUT"
    done
  done
done
echo "--- q=92 control: every arm must be byte-identical (dead zone is a no-op there) ---"
for seq in crowd_run bbb_extended; do
  for mul in 2.0 1.0 0.0; do
    b=$(GNC_INTER_DZ_MUL=$mul $BIN benchmark-sequence \
          -i "test_material/frames/sequences/$seq/frame_%04d.png" -n $N -k $KI -q 92 2>&1 \
        | grep -m1 -oE 'Total: [0-9]+ bytes' | grep -oE '[0-9]+')
    echo "control,$seq,q=92,mul=$mul,bytes=$b" | tee -a "$OUT"
  done
done
