#!/usr/bin/env python3
"""How much prediction quality does GNC's motion search leave on the table?

MEAS-4 established that GNC's inter gap is prediction quality rather than the coding model, and
a quality-matched x264 ablation put sub-pel motion compensation far ahead of every other inter
tool. GNC already has quarter-pel MC. So: is GNC's *search* finding the motion?

Compares GNC's achieved luma residual against an offline oracle motion search on the identical
(current, reference) pair — the encoder dumps both, so the oracle predicts from the same decoded
reference GNC used, not from source frames.

The oracle is deliberately handicapped: 16x16 blocks (GNC splits to 8x8), full integer search,
bilinear quarter-pel refinement (the same interpolation GNC uses). Anything it wins is a lower
bound on the headroom in GNC's search.

Dumps come from: GNC_DUMP_RESIDUAL=<dir> GNC_DIAGNOSTICS=1, 4:4:4, P-frames only (ki<=9 with
fewer than 9 frames).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meas_subpel_filter import integer_search, residual_plane, sample_bilinear, satd_blocks  # noqa: E402
from meas4_oracle import shannon_bits, to_blocks, dct2_8x8_blocks  # noqa: E402


def load(path, cw, ch):
    meta = json.load(open(path.replace(".f32", ".json")))
    w, h = meta["width"], meta["height"]
    a = np.fromfile(path, dtype="<f4", count=w * h).reshape(h, w)
    return a[:ch, :cw].astype(np.float64)


def bits_of(resid, qstep=4.0):
    b = to_blocks(resid, 8)
    q = np.rint(dct2_8x8_blocks(b) / qstep).astype(np.int32)
    return sum(shannon_bits(q[:, i, j]) for i in range(8) for j in range(8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--crop", type=int, nargs=2, default=[768, 448])
    ap.add_argument("--range", type=int, default=32)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=4)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cw, ch = args.crop
    refs = sorted(glob.glob(os.path.join(args.dump_dir, "resid_Pref_*.f32")))
    if not refs:
        sys.exit(f"no Pref dumps in {args.dump_dir}")
    refs = refs[: args.max_frames]

    print(f"\n=== GNC motion search vs offline oracle: {args.label or args.dump_dir} ===")
    print(f"{len(refs)} P-frames, {cw}x{ch} crop, oracle = {args.block}x{args.block} blocks, "
          f"+/-{args.range} integer + bilinear quarter-pel")

    g_satd = g_bits = o_satd = o_bits = 0.0
    for rp in refs:
        n = int(os.path.basename(rp).split("_")[2].split(".")[0])
        cur_p = os.path.join(args.dump_dir, f"resid_Pcur_{n + 1:05d}.f32")
        res_p = os.path.join(args.dump_dir, f"resid_Py_{n + 2:05d}.f32")
        if not (os.path.exists(cur_p) and os.path.exists(res_p)):
            continue
        ref = load(rp, cw, ch)
        cur = load(cur_p, cw, ch)
        gnc = load(res_p, cw, ch)

        _, bdx, bdy = integer_search(cur, ref, args.block, args.range)
        best = residual_plane(cur, ref, args.block, bdx, bdy)
        best_cost = satd_blocks(best, args.block)
        for fy in range(4):
            for fx in range(4):
                if fx == 0 and fy == 0:
                    continue
                r = residual_plane(cur, sample_bilinear(ref, fx, fy), args.block, bdx, bdy)
                c = satd_blocks(r, args.block)
                pick = c < best_cost
                big = np.repeat(np.repeat(pick, args.block, 0), args.block, 1)
                best = np.where(big, r, best)
                best_cost = np.minimum(c, best_cost)

        g_satd += satd_blocks(gnc, args.block).sum()
        o_satd += best_cost.sum()
        g_bits += bits_of(gnc)
        o_bits += bits_of(best)

    print(f"\n   {'':24} {'SATD':>14} {'est. bits':>14}")
    print(f"   {'GNC (as shipped)':24} {g_satd:14.0f} {g_bits:14.0f}")
    print(f"   {'offline oracle ME':24} {o_satd:14.0f} {o_bits:14.0f}")
    print(f"\n   oracle is {(1 - o_satd / g_satd) * 100:+.1f}% on SATD, "
          f"{(1 - o_bits / g_bits) * 100:+.1f}% on estimated bits")
    print()


if __name__ == "__main__":
    main()
