#!/usr/bin/env bash
# MEAS-10 — re-take BASELINE compression numbers against the current commit.
# Compression only (PSNR / bpp / VMAF). Does not quote fps.
# Logs and CSVs go to $OUT; this script prints a one-line status per run to stdout
# so a watcher can see progress without drowning in encoder output.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
GNC="${GNC:-$ROOT/target/release/gnc}"
OUT="${MEAS10_OUT:-$ROOT/meas10_out}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
FRAMES="$ROOT/test_material/frames"
mkdir -p "$OUT"

if [ ! -x "$GNC" ]; then
  echo "FAILED: missing $GNC" >&2
  exit 1
fi

COMMIT=$(git rev-parse HEAD)
SHORT=$(git rev-parse --short HEAD)
{
  echo "commit=$COMMIT"
  echo "short=$SHORT"
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "load=$(sysctl -n vm.loadavg)"
  echo "gnc=$GNC"
  "$GNC" --version 2>/dev/null || true
} | tee "$OUT/meta.txt"

STILLS_CSV="$OUT/stills.csv"
SEQ_CSV="$OUT/sequences.csv"
[ -f "$STILLS_CSV" ] || echo "image,q,psnr_db,bpp,bytes,vmaf,ssim,levels" >"$STILLS_CSV"
[ -f "$SEQ_CSV" ] || echo "sequence,q,n,ki,chroma,bytes,bpp,psnr_avg,psnr_min,vmaf,frame_mix,i_only_bytes,i_only_bpp" >"$SEQ_CSV"

parse_still() {
  local log=$1
  local psnr bpp bytes vmaf ssim
  psnr=$(awk '/^Quality:/{ for (i=1;i<=NF;i++) if ($i=="PSNR:") { print $(i+1); exit } }' "$log")
  bpp=$(awk '/^Quality:/{ for (i=1;i<=NF;i++) if ($i=="BPP:") { print $(i+1); exit } }' "$log")
  bytes=$(awk '/^Quality:/{ for (i=1;i<=NF;i++) if ($i=="Size:") { print $(i+1); exit } }' "$log")
  ssim=$(awk '/^Quality:/{ for (i=1;i<=NF;i++) if ($i=="SSIM:") { print $(i+1); exit } }' "$log")
  vmaf=$(awk '/^VMAF:/{ print $NF; exit }' "$log")
  printf '%s %s %s %s %s\n' "${psnr:-}" "${bpp:-}" "${bytes:-}" "${vmaf:-}" "${ssim:-}"
}

run_still() {
  local img=$1 q=$2
  local name
  name=$(basename "$img" .png)
  local stem="still_${name}_q${q}"
  local log="$OUT/${stem}.log"
  if grep -q "^${name},${q}," "$STILLS_CSV" 2>/dev/null; then
    echo "skip still $name q=$q (already in csv)"
    return 0
  fi
  echo "run still $name q=$q"
  if ! "$GNC" benchmark -i "$img" -q "$q" --vmaf -n 1 >"$log" 2>&1; then
    echo "FAILED: still $name q=$q (see $log)" >&2
    tail -20 "$log" >&2
    return 1
  fi
  local psnr bpp bytes vmaf ssim
  read -r psnr bpp bytes vmaf ssim <<<"$(parse_still "$log")"
  if [ -z "$psnr" ] || [ -z "$bpp" ]; then
    echo "FAILED: could not parse still $name q=$q" >&2
    tail -20 "$log" >&2
    return 1
  fi
  local levels=5
  if [ "$q" -lt 25 ]; then levels=4; fi
  echo "${name},${q},${psnr},${bpp},${bytes},${vmaf},${ssim},${levels}" >>"$STILLS_CSV"
  echo "  -> PSNR ${psnr} dB  bpp ${bpp}  VMAF ${vmaf}"
}

run_seq() {
  local seq=$1 q=$2 n=${3:-10} ki=${4:-9}
  local stem="seq_${seq}_q${q}_n${n}"
  local log="$OUT/${stem}.log"
  local csv="$OUT/${stem}.csv"
  if grep -q "^${seq},${q},${n}," "$SEQ_CSV" 2>/dev/null; then
    echo "skip seq $seq q=$q n=$n (already in csv)"
    return 0
  fi
  local pattern="$FRAMES/sequences/${seq}/frame_%04d.png"
  echo "run seq $seq q=$q n=$n ki=$ki"
  if ! "$GNC" benchmark-sequence \
      -i "$pattern" -n "$n" -k "$ki" -q "$q" --vmaf --csv "$csv" \
      >"$log" 2>&1; then
    echo "FAILED: seq $seq q=$q (see $log)" >&2
    tail -30 "$log" >&2
    return 1
  fi
  local bytes bpp psnr_avg psnr_min vmaf mix i_bytes i_bpp
  # First Sequence Summary is I+P (the default path). Field 5 is the byte count after
  # stripping punctuation from `Sequence Summary (N frames, BYTES bytes):`.
  psnr_avg=$(awk '/^Sequence Summary/{grab=1; next} grab && /PSNR:/{ print $3; exit }' "$log")
  psnr_min=$(awk '/^Sequence Summary/{grab=1; next} grab && /PSNR:/{ print $6; exit }' "$log")
  bpp=$(awk '/^Sequence Summary/{grab=1; next} grab && /BPP:/{ print $3; exit }' "$log")
  bytes=$(awk '/^Sequence Summary \(/{ gsub(/[(),]/,""); print $5; exit }' "$log")
  vmaf=$(awk '/VMAF: computing/{ for (i=1;i<=NF;i++) if ($i ~ /^mean=/) { sub(/^mean=/,"",$i); print $i; exit } }' "$log")
  mix=$(awk '/avg .* fps.*, .*I\+/{ for (i=1;i<=NF;i++) if ($i ~ /[0-9]+I\+[0-9]+P\+[0-9]+B/) { print $i; exit } }' "$log")
  i_bytes=$(awk '/^=== Comparison ===/{ getline; print $2; exit }' "$log")
  i_bpp=$(awk '/^=== Comparison ===/{ getline; gsub(/[()]/,""); print $4; exit }' "$log")
  echo "${seq},${q},${n},${ki},444,${bytes},${bpp},${psnr_avg},${psnr_min},${vmaf},${mix},${i_bytes},${i_bpp}" >>"$SEQ_CSV"
  echo "  -> bpp ${bpp}  PSNR ${psnr_avg} dB  VMAF ${vmaf}  mix ${mix}"
}

# --- stills: the GOALS / BASELINE table, plus the other three images ---
STILL_IMAGES=(
  "$FRAMES/bbb_1080p.png"
  "$FRAMES/blue_sky_1080p.png"
  "$FRAMES/touchdown_1080p.png"
  "$FRAMES/kristensara_720p.png"
)
STILL_QS=(25 50 75 90)

fail=0
for img in "${STILL_IMAGES[@]}"; do
  for q in "${STILL_QS[@]}"; do
    run_still "$img" "$q" || fail=1
  done
done
# lossless canary on the table image
run_still "$FRAMES/bbb_1080p.png" 100 || fail=1

# --- sequences: BASELINE table was q=75 ki=9 n=10 4:4:4; add q=90 ---
SEQS=(crowd_run old_town_cross bbb_extended)
for seq in "${SEQS[@]}"; do
  for q in 75 90; do
    run_seq "$seq" "$q" 10 9 || fail=1
  done
done

cp "$STILLS_CSV" "$OUT/stills.csv.final" 2>/dev/null || true
{
  echo
  echo "=== stills.csv ==="
  cat "$STILLS_CSV"
  echo
  echo "=== sequences.csv ==="
  cat "$SEQ_CSV"
} | tee "$OUT/summary.txt"

if [ "$fail" -ne 0 ]; then
  echo "FAILED: one or more runs failed; csvs still written"
  exit 1
fi
echo "DONE: stills and sequences at $SHORT"
exit 0
