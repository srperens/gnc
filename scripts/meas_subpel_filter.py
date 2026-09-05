#!/usr/bin/env python3
"""How much prediction does GNC's bilinear sub-pel interpolation give away?

GNC does quarter-pel motion compensation, but interpolates sub-pel positions bilinearly
(`bilinear_ref` in motion_compensate.wgsl). H.264 uses a 6-tap Wiener filter for the half-pel
positions and only then bilinear averaging for quarter-pel. Bilinear interpolation is a low-pass
filter: it blurs the reference, so the predictor loses exactly the high-frequency detail that
would otherwise cancel in the residual.

The x264 feature ablation cannot separate this, because x264 always uses the 6-tap filter -- its
`--subme` ladder varies search *effort*, not the filter. So this measures the filter directly:
identical motion search, identical blocks, only the interpolation differs.

Method: integer-pel full search per block, then evaluate all 16 quarter-pel offsets around the
integer optimum under each interpolation scheme.

The metric matters more than usual here. **SAD is the wrong one**: it rewards a blurred
predictor, because blurring suppresses the large isolated differences SAD sums, and bilinear
interpolation *is* a blur. Measured with SAD, bilinear beats the 6-tap filter on three of four
real sequences -- which is an artefact, not a result, and it is exactly why x264 uses SATD rather
than SAD for sub-pel decisions. So this reports:

  * SATD (Hadamard) -- the standard sub-pel decision metric, blur-aware;
  * estimated bits -- Shannon entropy of the quantized 8x8 DCT of the residual, which is what
    the residual actually costs.

Source frames on both sides, so this bounds the filter's contribution to prediction quality; it
is not a bitrate prediction.

Run:  python3 scripts/meas_subpel_filter.py <seq.y4m> [--frames N] [--crop W H]
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from meas_multiref_gate import read_y4m_luma  # noqa: E402

# H.264 half-pel interpolation filter, normalised.
SIX_TAP = np.array([1, -5, 20, 20, -5, 1], dtype=np.float64) / 32.0


def filter_1d(plane, axis):
    """Apply the 6-tap half-pel filter along `axis`, producing samples halfway between."""
    p = np.moveaxis(plane, axis, -1)
    padded = np.pad(p, ((0, 0),) * (p.ndim - 1) + ((2, 3),), mode="edge")
    out = np.zeros_like(p)
    for k, c in enumerate(SIX_TAP):
        out += c * padded[..., k : k + p.shape[-1]]
    return np.moveaxis(out, -1, axis)


def build_planes_sixtap(ref):
    """The four H.264 sample grids: integer, half-right (b), half-down (h), centre (j).

    Half-pel results are clipped back to the 8-bit range, as H.264's Clip1 specifies. Without
    that clip the filter's ringing overshoot survives into the quarter-pel averages and the
    predictor measures *worse* than bilinear, which is what a first version of this script
    reported.
    """
    a = ref
    b = np.clip(filter_1d(ref, axis=1), 0.0, 255.0)
    c = np.clip(filter_1d(ref, axis=0), 0.0, 255.0)
    # The centre sample filters the unclipped intermediate, then clips once.
    d = np.clip(filter_1d(filter_1d(ref, axis=1), axis=0), 0.0, 255.0)
    return [[a, b], [c, d]]


def _shift(p, ox, oy):
    """Shift a plane by whole samples, replicating the edge."""
    if ox:
        p = np.concatenate([p[:, ox:], np.repeat(p[:, -1:], ox, axis=1)], axis=1)
    if oy:
        p = np.concatenate([p[oy:, :], np.repeat(p[-1:, :], oy, axis=0)], axis=0)
    return p


def sample_sixtap(planes, fx, fy):
    """Quarter-pel sample at (fx, fy) quarter units, following H.264's construction.

    Quarter positions are the average of exactly TWO neighbouring samples, never three. The
    diagonal quarter positions (H.264's e, g, p, r) average the two *pure* half-pel planes —
    half-right and half-down — not the centre sample.
    """
    hx, rx = fx // 2, fx % 2
    hy, ry = fy // 2, fy % 2

    if rx == 0 and ry == 0:
        return planes[hy][hx]

    if rx == 1 and ry == 1:
        # e / g / p / r: average half-right and half-down, each shifted into position.
        b = _shift(planes[0][1], 0, hy)
        h = _shift(planes[1][0], hx, 0)
        return (b + h) / 2.0

    if ry == 0:
        # a / c and their half-row variants: average across the x axis.
        p0 = planes[hy][hx]
        p1 = _shift(planes[hy][1 - hx], 1 if hx == 1 else 0, 0)
        return (p0 + p1) / 2.0

    # d / n and their half-column variants: average across the y axis.
    p0 = planes[hy][hx]
    p1 = _shift(planes[1 - hy][hx], 0, 1 if hy == 1 else 0)
    return (p0 + p1) / 2.0


def sample_bilinear(ref, fx, fy):
    """Quarter-pel sample by bilinear interpolation — what GNC's shader does."""
    ax = fx / 4.0
    ay = fy / 4.0
    r00 = ref
    r01 = np.roll(ref, -1, axis=1)
    r10 = np.roll(ref, -1, axis=0)
    r11 = np.roll(np.roll(ref, -1, axis=0), -1, axis=1)
    top = r00 * (1 - ax) + r01 * ax
    bot = r10 * (1 - ax) + r11 * ax
    return top * (1 - ay) + bot * ay


def _hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


_H8 = _hadamard(8)


def satd_blocks(resid, bs):
    """Sum of absolute Hadamard-transformed differences, per `bs`x`bs` block.

    Computed on 8x8 sub-blocks, as x264 does. Unlike SAD this does not reward a blurred
    predictor: energy the blur moves into low frequencies is still counted.
    """
    h, w = resid.shape
    by8, bx8 = h // 8, w // 8
    b = (
        resid[: by8 * 8, : bx8 * 8]
        .reshape(by8, 8, bx8, 8)
        .transpose(0, 2, 1, 3)
    )
    t = _H8 @ b @ _H8.T
    s8 = np.abs(t).sum(axis=(2, 3)) / 8.0
    r = bs // 8
    return s8.reshape(by8 // r, r, bx8 // r, r).sum(axis=(1, 3))


def dct8(blocks):
    n = 8
    k = np.arange(n)
    c = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    sc = np.full(n, np.sqrt(2.0 / n))
    sc[0] = np.sqrt(1.0 / n)
    m = (c * sc[None, :]).T
    return m @ blocks @ m.T


def residual_bits(resid, qstep=4.0):
    """Shannon entropy of the quantized 8x8 DCT of a residual — an ideal-coder rate proxy."""
    h, w = resid.shape
    by, bx = h // 8, w // 8
    b = resid[: by * 8, : bx * 8].reshape(by, 8, bx, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    q = np.rint(dct8(b) / qstep).astype(np.int32)
    bits = 0.0
    for i in range(8):
        for j in range(8):
            sym = q[:, i, j]
            _, counts = np.unique(sym, return_counts=True)
            p = counts / counts.sum()
            bits += float(-(p * np.log2(p)).sum() * sym.size)
    return bits


def residual_plane(cur, plane, bs, dxs, dys):
    """Residual of `cur` against `plane` displaced by the per-block integer MV."""
    h, w = cur.shape
    by, bx = h // bs, w // bs
    out = np.empty_like(cur)
    for j in range(by):
        for i in range(bx):
            y0, x0 = j * bs, i * bs
            sy = max(0, min(y0 + dys[j, i], h - bs))
            sx = max(0, min(x0 + dxs[j, i], w - bs))
            out[y0 : y0 + bs, x0 : x0 + bs] = (
                cur[y0 : y0 + bs, x0 : x0 + bs] - plane[sy : sy + bs, sx : sx + bs]
            )
    return out


def integer_search(cur, ref, bs, rng):
    h, w = cur.shape
    by, bx = h // bs, w // bs
    best = np.full((by, bx), np.inf)
    bdx = np.zeros((by, bx), dtype=np.int32)
    bdy = np.zeros((by, bx), dtype=np.int32)
    ref_pad = np.pad(ref, rng, mode="edge")
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            sh = ref_pad[rng + dy : rng + dy + h, rng + dx : rng + dx + w]
            sad = np.abs(cur - sh).reshape(by, bs, bx, bs).sum(axis=(1, 3))
            m = sad < best
            best = np.where(m, sad, best)
            bdx = np.where(m, dx, bdx)
            bdy = np.where(m, dy, bdy)
    return best, bdx, bdy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("y4m")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--range", type=int, default=16)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--crop", type=int, nargs=2, default=[768, 448],
                    help="centre crop analysed, keeps the sweep tractable")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    w, h, frames = read_y4m_luma(args.y4m, args.frames)
    if len(frames) < 2:
        sys.exit("need at least 2 frames")
    cw, ch = args.crop
    x0 = max(0, (w - cw) // 2)
    y0 = max(0, (h - ch) // 2)
    frames = [f[y0 : y0 + ch, x0 : x0 + cw].astype(np.float64) for f in frames]

    name = args.label or args.y4m.split("/")[-1]
    print(f"\n=== sub-pel interpolation filter: {name} ===")
    print(f"{len(frames)} frames, {cw}x{ch} centre crop, {args.block}x{args.block} blocks, "
          f"+/-{args.range} integer search")

    tot_int = tot_bil = tot_six = 0.0
    bits_int = bits_bil = bits_six = 0.0
    for n in range(1, len(frames)):
        cur, ref = frames[n], frames[n - 1]
        _, bdx, bdy = integer_search(cur, ref, args.block, args.range)

        six = build_planes_sixtap(ref)

        # For each scheme, pick the quarter-pel offset per block by SATD, then build the chosen
        # residual plane so the rate proxy is computed on what would actually be coded.
        def evaluate(sampler):
            best_cost = None
            best_res = None
            for fy in range(4):
                for fx in range(4):
                    plane = sampler(fx, fy)
                    res = residual_plane(cur, plane, args.block, bdx, bdy)
                    cost = satd_blocks(res, args.block)
                    if best_cost is None:
                        best_cost, best_res = cost, res
                    else:
                        pick = cost < best_cost
                        big = np.repeat(np.repeat(pick, args.block, 0), args.block, 1)
                        best_res = np.where(big, res, best_res)
                        best_cost = np.minimum(cost, best_cost)
            return best_cost.sum(), best_res

        # integer-pel reference point
        r_int = residual_plane(cur, ref, args.block, bdx, bdy)
        c_int = satd_blocks(r_int, args.block).sum()

        c_bil, r_bil = evaluate(lambda fx, fy: sample_bilinear(ref, fx, fy))
        c_six, r_six = evaluate(lambda fx, fy: sample_sixtap(six, fx, fy))

        tot_int += c_int
        tot_bil += c_bil
        tot_six += c_six
        bits_int += residual_bits(r_int)
        bits_bil += residual_bits(r_bil)
        bits_six += residual_bits(r_six)

    print(f"\n   {'':28} {'SATD':>14} {'vs int':>8}   {'est. bits':>12} {'vs int':>8}")
    print(f"   {'integer-pel only':28} {tot_int:14.0f} {0.0:7.1f}%   "
          f"{bits_int:12.0f} {0.0:7.1f}%")
    print(f"   {'+ bilinear qpel (GNC)':28} {tot_bil:14.0f} "
          f"{(1 - tot_bil / tot_int) * 100:6.1f}%   {bits_bil:12.0f} "
          f"{(1 - bits_bil / bits_int) * 100:6.1f}%")
    print(f"   {'+ 6-tap half + qpel (H.264)':28} {tot_six:14.0f} "
          f"{(1 - tot_six / tot_int) * 100:6.1f}%   {bits_six:12.0f} "
          f"{(1 - bits_six / bits_int) * 100:6.1f}%")
    print(f"\n   6-tap vs bilinear:  SATD {(1 - tot_six / tot_bil) * 100:+.1f}%   "
          f"est. bits {(1 - bits_six / bits_bil) * 100:+.1f}%")
    print()


if __name__ == "__main__":
    main()
