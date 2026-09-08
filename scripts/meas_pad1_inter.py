#!/usr/bin/env python3
"""PAD-1's gate — does the padding fill cost anything on inter?

**Why this exists and why it is the whole item.** The padded region is don't-care for *output*:
the decoder crops it and no metric sees it. It is **not** don't-care for *prediction*. The decoder
keeps the padded plane in the reference buffer and motion compensation samples from it with its
reads clamped to the **padded** bounds (`p_padded_w/h` at `src/decoder/gpu_work.rs:478`), so a
block at the frame edge with a motion vector pointing outward really does read the padding. Edge
replication is the standard choice there precisely because it extends the picture plausibly, and
nothing in PAD-1's intra measurement says what a faded-flat region does to inter prediction.

So this compares `GNC_PAD_FILL=decay` (the new default) against `GNC_PAD_FILL=replicate` (the
pre-PAD-1 shader) on P-chains, and the figure that decides it is **worst-frame PSNR, not the
mean**. That is not a stylistic preference: INTRA-2's dead zone passed on the mean and failed on
the worst frame by up to 1.93 dB, because an error in a reference propagates down the chain until
the next keyframe. An edge-block prediction change has the same shape of risk.

**Pre-declared criterion (from BACKLOG's PAD-1):** no worst-frame regression above 0.3 dB on any
sequence, at ki=9, in either chroma format. Rate is expected to *fall*; a fall in rate with a flat
worst frame is the win, and a fall in both needs rate-normalising before it counts as anything.

Usage:
    scripts/meas_pad1_inter.py --sequences crowd_run old_town_cross blue_sky
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Arms:
#   "replicate"  forced off — the pre-PAD-1 / shipped-inter reference
#   "decay"      forced on  — the *hazard*, which this gate exists to keep measuring
#   "zero"       PAD-2 candidate (Dirac/Schroedinger inter zero-extend)
#   None         no override — what the codec actually does, and it must equal "replicate"
FILLS = ("replicate", "decay", "zero", None)


def repo_root():
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    return Path(out.stdout.strip()) if out.returncode == 0 else Path.cwd()


def run(gnc, pattern, frames, q, ki, chroma, fill):
    """One benchmark-sequence run. Returns (total_bytes, avg_psnr, min_psnr) or None.

    `fill=None` means no override at all, i.e. whatever the codec chooses for itself.
    """
    env = dict(os.environ)
    env.pop("GNC_PAD_FILL", None)
    if fill is not None:
        env["GNC_PAD_FILL"] = fill
    cmd = [str(gnc), "benchmark-sequence", "-i", pattern, "-n", str(frames),
           "-q", str(q), "-k", str(ki), "--chroma-format", chroma]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"      failed: {r.stderr.strip().splitlines()[-1:]}")
        return None
    # The I+P+B block comes first; its summary is the one to read. `--ab` is deliberately not
    # passed, so there is exactly one "Sequence Summary" before the all-I baseline section.
    total = re.search(r"Sequence Summary \((\d+) frames, (\d+) bytes\)", r.stdout)
    psnr = re.search(r"PSNR:\s+avg ([\d.]+) dB\s+min ([\d.]+)\s+max ([\d.]+)", r.stdout)
    if not total or not psnr:
        print("      could not parse the summary — output format changed?")
        return None
    return int(total.group(2)), float(psnr.group(1)), float(psnr.group(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", nargs="+",
                    default=["crowd_run", "old_town_cross", "bbb_extended"],
                    help="needs >= ki frames each; blue_sky and bbb ship only 8 PNG frames and "
                         "cannot carry a ki=9 chain at all")
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--ki", type=int, default=9)
    ap.add_argument("--qualities", default="85,92")
    ap.add_argument("--chroma", default="444,420")
    ap.add_argument("--gnc-binary", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    root = repo_root()
    gnc = Path(args.gnc_binary) if args.gnc_binary else root / "target/release/gnc"
    if not gnc.exists():
        sys.exit(f"no GNC binary at {gnc} — cargo build --release")

    qualities = [int(q) for q in args.qualities.split(",")]
    chromas = args.chroma.split(",")
    rows, worst = [], []

    print("PAD-1/PAD-2 inter gate — decay and zero against replicate, ki="
          f"{args.ki}, {args.frames} frames")
    print("The column that decides it is dWORST, not dAVG.\n")
    print(f"  {'':<16} {'':>6} {'':>3} {'--- forced decay ---':>19}   "
          f"{'--- zero (PAD-2) ---':>19}   {'--- default ---':>18}")
    print(f"  {'sequence':<16} {'chroma':>6} {'q':>3} {'rate':>8} {'dWORST dB':>10}   "
          f"{'rate':>8} {'dWORST dB':>10}   {'rate':>8} {'dWORST':>9}")

    for seq in args.sequences:
        pattern = str(root / f"test_material/frames/sequences/{seq}/frame_%04d.png")
        if not Path(pattern % 0).exists():
            print(f"  {seq:<16} no frames at {pattern} — skipped")
            continue
        # Clamp to what is on disk. Asking for more frames than exist makes the CLI panic in
        # `image_util.rs` with a bare NotFound, which this harness first reported as eight
        # identical "failed" lines and no reason — and a silently dropped sequence is exactly how
        # a gate ends up declared on two sequences when it asked for three.
        have = len(sorted(Path(pattern).parent.glob("frame_*.png")))
        frames = min(args.frames, have)
        if frames < args.ki:
            print(f"  {seq:<16} only {have} frames on disk, fewer than ki={args.ki} — skipped, "
                  f"because a P-chain shorter than the keyframe interval is not the thing being "
                  f"measured")
            continue
        if frames < args.frames:
            print(f"  {seq:<16} {have} frames on disk, using {frames} instead of {args.frames}")
        for chroma in chromas:
            for q in qualities:
                res = {f: run(gnc, pattern, frames, q, args.ki, chroma, f) for f in FILLS}
                if any(v is None for v in res.values()):
                    continue
                (b_rep, a_rep, w_rep) = res["replicate"]
                (b_dec, a_dec, w_dec) = res["decay"]
                (b_zro, a_zro, w_zro) = res["zero"]
                (b_def, a_def, w_def) = res[None]
                d_rate, d_worst = b_dec / b_rep - 1.0, w_dec - w_rep
                z_rate, z_worst = b_zro / b_rep - 1.0, w_zro - w_rep
                f_rate, f_worst = b_def / b_rep - 1.0, w_def - w_rep
                worst.append((seq, chroma, q, d_worst, f_worst, f_rate, d_rate,
                              z_rate, z_worst))
                rows.append({"sequence": seq, "chroma": chroma, "q": q, "ki": args.ki,
                             "frames": frames,
                             "bytes_replicate": b_rep, "bytes_forced_decay": b_dec,
                             "bytes_zero": b_zro, "bytes_default": b_def,
                             "avg_replicate": a_rep, "avg_forced_decay": a_dec,
                             "avg_zero": a_zro, "avg_default": a_def,
                             "worst_replicate": w_rep, "worst_forced_decay": w_dec,
                             "worst_zero": w_zro, "worst_default": w_def,
                             "forced_d_rate": d_rate, "forced_d_worst": d_worst,
                             "zero_d_rate": z_rate, "zero_d_worst": z_worst,
                             "default_d_rate": f_rate, "default_d_worst": f_worst})
                print(f"  {seq:<16} {chroma:>6} {q:>3} {d_rate:>+8.2%} {d_worst:>+10.3f}"
                      f"   {z_rate:>+8.2%} {z_worst:>+9.3f}"
                      f"   {f_rate:>+8.2%} {f_worst:>+9.3f}")

    if not worst:
        sys.exit("no points measured")

    print("\n--- what the codec ships: the default must not take the fill on a P-chain ---")
    print("  criterion: no worst-frame regression above 0.3 dB, and rate identical to replicate")
    ship_bad = [w for w in worst if w[4] < -0.3 or abs(w[5]) > 1e-9]
    print(f"  points measured:                 {len(worst)}")
    print(f"  worst default dWORST:            {min(w[4] for w in worst):+.3f} dB")
    print(f"  max |default rate change|:       {max(abs(w[5]) for w in worst):.4%}")
    print(f"  VERDICT: {'FAILS' if ship_bad else 'PASSES'} — the default is "
          f"{'NOT ' if ship_bad else ''}equivalent to replication on inter")
    for w in ship_bad:
        print(f"    {w[0]} {w[1]} q={w[2]}: dWORST {w[4]:+.3f} dB, rate {w[5]:+.4%}")

    print("\n--- the hazard this gate guards, measured by forcing GNC_PAD_FILL=decay ---")
    print("  This arm is EXPECTED to regress. It is why the fill is refused on any frame that")
    print("  something predicts from, and re-measuring it is how we would notice that stopping")
    print("  being true.")
    forced = [w for w in worst if w[3] < -0.3]
    big = min(worst, key=lambda w: w[3])
    print(f"  points regressing > 0.3 dB:      {len(forced)} of {len(worst)}")
    print(f"  worst forced dWORST:             {big[3]:+.3f} dB "
          f"({big[0]}, {big[1]}, q={big[2]})")
    print(f"  mean forced rate change:         {sum(w[6] for w in worst) / len(worst):+.2%}")
    print(f"  hazard still present: {'YES — refusing it is still right' if forced else 'NO — re-open PAD-1s inter half'}")

    print("\n--- PAD-2 candidate: GNC_PAD_FILL=zero (Dirac inter zero-extend) ---")
    print("  criterion: rate of the forced-on arm, worst-frame PSNR within 0.3 dB of replicate")
    z_bad = [w for w in worst if w[8] < -0.3]
    z_big = min(worst, key=lambda w: w[8])
    print(f"  points regressing > 0.3 dB:      {len(z_bad)} of {len(worst)}")
    print(f"  worst zero dWORST:               {z_big[8]:+.3f} dB "
          f"({z_big[0]}, {z_big[1]}, q={z_big[2]})")
    print(f"  mean zero rate change:           {sum(w[7] for w in worst) / len(worst):+.2%}")
    print(f"  VERDICT: {'FAILS' if z_bad else 'PASSES'} — zero-extend "
          f"{'is not' if z_bad else 'is'} a drop-in for replication on inter")

    if args.csv and rows:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
