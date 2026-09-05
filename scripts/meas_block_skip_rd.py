#!/usr/bin/env python3
"""What is local (block-level) skip worth on GNC's inter residuals?

Background. GNC transforms a whole 256x256 tile with a wavelet, so the smallest region it can
decline to code is a tile, and each tile costs ~290 bytes of header regardless of size — measured
at 2.3% of the bitrate at 256px, 7.4% at 128px, 20.7% at 64px. Shrinking tiles to macroblock size
is therefore impossible: 16x16 tiles at 1080p would be 8100 headers, ~2.3 MB per frame. H.264
skips a macroblock for about one bit.

MEAS-4 compared coding models at matched *residual distortion* and found little difference. That
comparison structurally cannot see skip, because skipping raises distortion in exchange for rate —
it only pays under a rate-distortion criterion. This script redoes it as RD.

For each scheme it minimises D + lambda*R per block, with lambda tied to the quantiser the way
practical encoders do it, and reports the resulting (rate, distortion) point:

  tile-wavelet   — the whole plane transformed at once, every coefficient coded (GNC today)
  block-dct      — 16x16 blocks, 8x8 DCT, per-block RD skip decision, 1 bit per block signalled

Input: residual dumps from GNC_DUMP_RESIDUAL (see meas_me_quality.py for the encoder side).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meas4_oracle import (  # noqa: E402
    dct2_8x8_blocks, dwt2, idwt2, quantize, dequantize, shannon_bits, subband_gains, to_blocks,
)
from meas_subpel_filter import dct8  # noqa: E402


def idct8(blocks):
    n = 8
    k = np.arange(n)
    c = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    sc = np.full(n, np.sqrt(2.0 / n))
    sc[0] = np.sqrt(1.0 / n)
    m = (c * sc[None, :]).T
    return m.T @ blocks @ m


def load(path, cw, ch):
    meta = json.load(open(path.replace(".f32", ".json")))
    w, h = meta["width"], meta["height"]
    a = np.fromfile(path, dtype="<f4", count=w * h).reshape(h, w)
    cw, ch = cw - cw % 16, ch - ch % 16
    return a[:ch, :cw].astype(np.float64)


def tile_wavelet_cost(resid, qstep, levels=3, dead_zone=0.75):
    """GNC's scheme: one transform over the whole plane, every coefficient coded."""
    ll, bands = dwt2(resid, levels)
    g = subband_gains(resid.shape, levels)
    bits = 0.0
    qll = quantize(ll * g["LL"], qstep, dead_zone)
    bits += shannon_bits(qll.reshape(-1))
    rec_bands = []
    for lv, (lh, hl, hh) in enumerate(bands):
        names = (f"LH{lv+1}", f"HL{lv+1}", f"HH{lv+1}")
        qs = [quantize(b * g[nm], qstep, dead_zone) for b, nm in zip((lh, hl, hh), names)]
        for q in qs:
            bits += shannon_bits(q.reshape(-1))
        rec_bands.append(tuple(
            dequantize(q, qstep, dead_zone) / g[nm] for q, nm in zip(qs, names)
        ))
    rec = idwt2(dequantize(qll, qstep, dead_zone) / g["LL"], rec_bands)
    return bits, float(((resid - rec) ** 2).sum())


def block_dct_rd_cost(resid, qstep, blk=16, dead_zone=0.0):
    """16x16 blocks, 8x8 DCT, per-block RD skip. One bit per block for the flag.

    The skip decision compares, per block, the cost of coding it (its quantized distortion plus
    lambda times its bits) against the cost of not coding it (full residual energy plus one bit).
    lambda = 0.85 * qstep^2 is the usual practical tie between the quantiser and the RD slope.
    """
    h, w = resid.shape
    by, bx = h // blk, w // blk
    lam = 0.85 * qstep * qstep

    # Per-8x8 quantized DCT over the whole plane, then grouped into blk x blk blocks.
    b8 = to_blocks(resid, 8)
    q = quantize(dct2_8x8_blocks(b8), qstep, dead_zone)
    rec8 = idct8(dequantize(q, qstep, dead_zone))

    r = blk // 8
    n8x, n8y = w // 8, h // 8
    # Bits per 8x8 block, apportioned from each coefficient position's entropy.
    pos_bits = np.zeros(q.shape[0])
    for i in range(8):
        for j in range(8):
            sym = q[:, i, j]
            vals, counts = np.unique(sym, return_counts=True)
            p = counts / counts.sum()
            cost = dict(zip(vals, -np.log2(p)))
            pos_bits += np.array([cost[v] for v in sym])

    def regroup(flat):
        return flat.reshape(n8y, n8x).reshape(by, r, bx, r).sum(axis=(1, 3))

    coded_bits = regroup(pos_bits)
    err8 = ((b8 - rec8) ** 2).sum(axis=(1, 2))
    coded_dist = regroup(err8)
    energy8 = (b8 ** 2).sum(axis=(1, 2))
    skip_dist = regroup(energy8)

    cost_code = coded_dist + lam * (coded_bits + 1.0)
    cost_skip = skip_dist + lam * 1.0
    skip = cost_skip < cost_code

    bits = float(np.where(skip, 1.0, coded_bits + 1.0).sum())
    dist = float(np.where(skip, skip_dist, coded_dist).sum())
    return bits, dist, float(skip.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--max-planes", type=int, default=6)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dump_dir, "resid_Py_*.f32")))
    if not files:
        files = sorted(glob.glob(os.path.join(args.dump_dir, "resid_P_*.f32")))
    if not files:
        sys.exit(f"no luma residual dumps in {args.dump_dir}")
    files = files[: args.max_planes]

    aw = args.width - args.width % 16
    ah = args.height - args.height % 16
    px = aw * ah * len(files)

    print(f"\n=== local block skip, RD: {args.label or args.dump_dir} ===")
    print(f"{len(files)} luma residual planes at {aw}x{ah}\n")
    print(f"  {'qstep':>6}  {'tile-wavelet':>22}   {'16x16 block DCT + RD skip':>32}")
    print(f"  {'':6}  {'bpp':>10} {'PSNR':>10}   {'bpp':>10} {'PSNR':>10} {'skipped':>9}")

    for qstep in (2.0, 4.0, 8.0, 16.0):
        wb = wd = db = dd = sk = 0.0
        for f in files:
            r = load(f, args.width, args.height)
            b, d = tile_wavelet_cost(r, qstep)
            wb += b
            wd += d
            b2, d2, s = block_dct_rd_cost(r, qstep)
            db += b2
            dd += d2
            sk += s
        wpsnr = 10 * np.log10(255.0 ** 2 / (wd / px))
        dpsnr = 10 * np.log10(255.0 ** 2 / (dd / px))
        print(f"  {qstep:6.1f}  {wb / px:10.4f} {wpsnr:10.2f}   "
              f"{db / px:10.4f} {dpsnr:10.2f} {sk / len(files) * 100:8.1f}%")
    print()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Sub-tile skip inside the existing wavelet (added 2026-09-05)
# ---------------------------------------------------------------------------


def subtile_skip_cost(resid, qstep, sub=32, levels=3, dead_zone=0.75):
    """Keep GNC's tile-wide wavelet, but zero the coefficients belonging to low-energy
    spatial sub-blocks after quantisation.

    This is the one option ARCH-2 left untested: finer skip *without* changing the transform and
    *without* a new tile header. It needs no bitstream syntax at all — zeroed quantised
    coefficients simply cost whatever the entropy coder charges for zeros, and the decoder
    dequantises them to nothing.

    A `sub`x`sub` spatial region of the plane maps to a sub/2^l rectangle in each level-l
    subband, so the mask is applied per subband at the matching scale. The wavelet's synthesis
    support is wider than the nominal region, so zeroing bleeds some ringing into neighbours —
    that is a distortion cost, not a correctness problem, and it is what this measures.
    """
    h, w = resid.shape
    ll, bands = dwt2(resid, levels)
    g = subband_gains(resid.shape, levels)

    # Per-sub-block residual energy drives the decision, with the same RD form as the block
    # experiment: skip when the energy saved is not worth the bits.
    lam = 0.85 * qstep * qstep
    sy, sx = h // sub, w // sub
    energy = (
        (resid[: sy * sub, : sx * sub] ** 2)
        .reshape(sy, sub, sx, sub)
        .sum(axis=(1, 3))
    )

    def mask_for(shape, scale):
        """Nearest-neighbour expansion of the sub-block decision onto a subband's grid."""
        bh, bw = shape
        yi = (np.arange(bh) * sy // max(bh, 1)).clip(0, sy - 1)
        xi = (np.arange(bw) * sx // max(bw, 1)).clip(0, sx - 1)
        return keep[np.ix_(yi, xi)]

    # Decide per sub-block by comparing "code it" against "drop it", using the quantised cost of
    # the whole plane as the rate proxy — the same generosity the block experiment allowed.
    qll_full = quantize(ll * g["LL"], qstep, dead_zone)
    bits_full = shannon_bits(qll_full.reshape(-1))
    for lv, (lh, hl, hh) in enumerate(bands):
        for b, nm in zip((lh, hl, hh), (f"LH{lv+1}", f"HL{lv+1}", f"HH{lv+1}")):
            bits_full += shannon_bits(quantize(b * g[nm], qstep, dead_zone).reshape(-1))
    bits_per_px = bits_full / (h * w)
    sub_bits = bits_per_px * sub * sub
    keep = (energy + lam * 1.0) > (lam * sub_bits)

    # Apply the mask and measure what the kept coefficients cost and reconstruct to.
    bits = float(keep.size)  # one bit per sub-block for the decision
    qll = quantize(ll * g["LL"], qstep, dead_zone) * mask_for(ll.shape, levels)
    bits += shannon_bits(qll.reshape(-1))
    rec_bands = []
    for lv, (lh, hl, hh) in enumerate(bands):
        names = (f"LH{lv+1}", f"HL{lv+1}", f"HH{lv+1}")
        m = mask_for(lh.shape, levels - lv - 1)
        qs = [quantize(b * g[nm], qstep, dead_zone) * m for b, nm in zip((lh, hl, hh), names)]
        for q in qs:
            bits += shannon_bits(q.reshape(-1))
        rec_bands.append(tuple(
            dequantize(q, qstep, dead_zone) / g[nm] for q, nm in zip(qs, names)
        ))
    rec = idwt2(dequantize(qll, qstep, dead_zone) / g["LL"], rec_bands)
    return bits, float(((resid - rec) ** 2).sum()), float(1.0 - keep.mean())
