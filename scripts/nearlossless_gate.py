#!/usr/bin/env python3
"""INTRA-NEARLOSSLESS offline gate — MED prediction instead of the wavelet, above q=88.

LOSSLESS-1 established that at q=100 a per-pixel median predictor whose error is entropy-coded
directly beats GNC's wavelet by 14.9%. Priority 1 in BACKLOG is intra at the contribution
operating point, where the whole remaining +90.5% BD-rate gap lives. This gate asks the obvious
next question: does that mechanism keep paying at q=88-99, where the codec still has to be lossy?

The honest version of the question needs a **closed loop**. JPEG-LS near-lossless quantises the
prediction error and predicts from the *reconstruction*, not from the original; predicting from
originals and quantising afterwards makes the decoder drift, which is exactly the signature BUG-13
was diagnosed from. So the model marches the same anti-diagonal wavefront the shipped decoder uses
(`med_reconstruct.wgsl`), per tile, with the reconstruction feeding the predictor.

What is modelled and what is not:

- Modelled: GNC's own reversible YCoCg-R, prediction reset per 256px tile (as the shader does),
  uniform quantisation of the residual with reconstruction in the loop, and the zeroth-order
  entropy of the quantised residuals.
- Not modelled: the real entropy coder. Rice spends a significance bit per coefficient, which is
  nearly free on sparse wavelet coefficients and close to a wasted bit per pixel on a dense MED
  residual. LOSSLESS-1's gate was optimistic by exactly this and delivered 79% of what it
  predicted, so this script also reports a **calibrated** figure: the same model at delta=1
  against the real q=100 file, per image, which turns that optimism into a measured ratio instead
  of an assumption.

Quality is reported the way CLAUDE.md requires above q=85: PSNR leads, luma is taken in YCoCg-R
(the plane GNC codes — a luma computed from decoded RGB is contaminated by chroma error and
overstates the loss 3.7x), and CIEDE2000 rides along because nothing about chroma is visible to a
luma metric.

Usage:
    nearlossless_gate.py <gnc-binary> <image.png> [more.png ...]

GNC_FRAMES is not used: pass the images explicitly, because a shared frames directory can be
rewritten by another session mid-run (COORDINATION, 2026-09-07).
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chroma_metric import ciede2000, srgb_to_lab  # noqa: E402

TILE = 256

# Deltas to sweep. delta=1 is lossless by construction (round(e/1) == e) and is the calibration
# point; the rest are the near-lossless rungs. A uniform step d gives a max error of d/2 per
# coefficient in the YCoCg-R domain, so the useful range up here is small.
DELTAS = tuple(
    float(x) for x in os.environ.get("GNC_NL_DELTAS", "1,1.5,2,2.5,3,4,5,6,8,10").split(",")
)

# The codec quantises chroma at `chroma_weight` times the luma step above q=60 (CHROMA-1 measured
# 1.2 as the largest value that costs nothing on MEAS-8's criterion). The model has to spend rate
# the same way or the comparison is a luma/chroma trade dressed up as a coding result — which is
# the error CLAUDE.md records twice. GNC_NL_CHROMA_MUL=1 measures the equal-step arm instead.
CHROMA_MUL = float(os.environ.get("GNC_NL_CHROMA_MUL", "1.2"))

# The q ladder the model is compared against. Wide enough that the two curves overlap in quality
# without extrapolating either.
QUALITIES = tuple(
    int(x) for x in os.environ.get("GNC_NL_QUALITIES", "78,82,85,86,88,90,92,93,94,95,96").split(",")
)


def ycocg_r(rgb):
    """Integer-reversible YCoCg-R forward, matching src/shaders/color_convert.wgsl."""
    R, G, B = (rgb[:, :, i].astype(np.int32) for i in range(3))
    Co = R - B
    t = B + (Co >> 1)
    Cg = G - t
    Y = t + (Cg >> 1)
    return Y, Co, Cg


def ycocg_r_inv(Y, Co, Cg):
    """Inverse of the above. Exact when the planes are exact."""
    t = Y - (Cg >> 1)
    G = Cg + t
    B = t - (Co >> 1)
    R = Co + B
    return np.stack([R, G, B], axis=2)


def to_tiles(plane):
    """(H, W) -> (ntiles, TILE, TILE). Requires H, W to be multiples of TILE."""
    h, w = plane.shape
    assert h % TILE == 0 and w % TILE == 0, f"{plane.shape} is not a whole number of tiles"
    return (
        plane.reshape(h // TILE, TILE, w // TILE, TILE)
        .transpose(0, 2, 1, 3)
        .reshape(-1, TILE, TILE)
    )


def from_tiles(tiles, h, w):
    return (
        tiles.reshape(h // TILE, w // TILE, TILE, TILE)
        .transpose(0, 2, 1, 3)
        .reshape(h, w)
    )


def med(a, b, c):
    """LOCO-I median edge predictor, elementwise."""
    mx, mn = np.maximum(a, b), np.minimum(a, b)
    return np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))


def dpcm_closed_loop(plane, delta, lo, hi):
    """Quantised MED DPCM with the reconstruction in the predictor loop, per tile.

    Returns (quantised residual symbols, reconstruction), both in the plane's own shape.

    The anti-diagonal sweep is the same dependency structure as `med_reconstruct.wgsl`: pixel
    (lx, ly) needs (lx-1, ly), (lx, ly-1) and (lx-1, ly-1), all of which sit on diagonals d-1 and
    d-2. Every tile advances its diagonal together, so this is 2*TILE-1 vectorised steps rather
    than a Python loop over pixels.
    """
    src = to_tiles(plane).astype(np.int32)
    rec = np.zeros_like(src)
    quant = np.zeros_like(src)
    n = src.shape[0]
    idx = np.arange(n)[:, None]

    for d in range(2 * TILE - 1):
        lx = np.arange(max(0, d - TILE + 1), min(d, TILE - 1) + 1)
        ly = d - lx
        # Neighbours from the reconstruction, guarded at the tile edges the way the shader is.
        left = rec[idx, ly, np.maximum(lx - 1, 0)]
        above = rec[idx, np.maximum(ly - 1, 0), lx]
        upleft = rec[idx, np.maximum(ly - 1, 0), np.maximum(lx - 1, 0)]
        p = med(left, above, upleft)
        p = np.where(ly == 0, left, p)          # top row of the tile: left only
        p = np.where(lx == 0, above, p)         # left column: above only
        p = np.where((lx == 0) & (ly == 0), 0, p)  # tile origin: nothing to predict from

        x = src[idx, ly, lx]
        e = x - p
        # Round-to-nearest uniform quantiser. delta=1 makes this the identity, so the lossless
        # arm of this model is exact by construction rather than by luck.
        q = np.round(e / delta).astype(np.int32)
        # The planes are integers, so the reconstruction has to land on an integer. Rounding, not
        # truncation: `rec` is an int32 array and a bare assignment of `p + q*delta` truncates
        # toward zero for a fractional step, which feeds a biased neighbour back into the
        # predictor and made the modelled rate *rise* with a coarser step.
        r = np.clip(np.rint(p + q * delta), lo, hi)
        quant[idx, ly, lx] = q
        rec[idx, ly, lx] = r

    h, w = plane.shape
    return from_tiles(quant, h, w), from_tiles(rec, h, w)


def entropy_bits(x):
    """Zeroth-order entropy of a symbol array, in bits."""
    _, cnt = np.unique(x, return_counts=True)
    pr = cnt / cnt.sum()
    return float(-(pr * np.log2(pr)).sum() * x.size)


def psnr(a, b, peak=255.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak**2 / mse)


def bd_rate(rate_ref, psnr_ref, rate_test, psnr_test):
    """Bjontegaard BD-rate of the test curve against the reference, in percent.

    Negative means the test curve needs fewer bits for the same quality. Cubic fit of log10(rate)
    against PSNR, integrated over the quality range the two curves share — points outside that
    range are dropped rather than extrapolated, because a cubic fitted through distant points
    bends inside the interval that matters (QUAL-1 measured a ladder-width effect of 47.5 VMAF
    points from exactly this).
    """
    lo = max(min(psnr_ref), min(psnr_test))
    hi = min(max(psnr_ref), max(psnr_test))
    if hi - lo < 1.0:
        return float("nan"), lo, hi

    def fit(rate, ps):
        # Keep the points inside the overlap plus the nearest one on each side.
        order = np.argsort(ps)
        ps, rate = np.asarray(ps)[order], np.asarray(rate)[order]
        inside = np.where((ps >= lo) & (ps <= hi))[0]
        first, last = inside[0], inside[-1]
        sel = slice(max(0, first - 1), min(len(ps), last + 2))
        return np.polyfit(ps[sel], np.log10(rate[sel]), 3)

    p_ref, p_test = fit(rate_ref, psnr_ref), fit(rate_test, psnr_test)
    i_ref = np.polyval(np.polyint(p_ref), [hi, lo])
    i_test = np.polyval(np.polyint(p_test), [hi, lo])
    avg = ((i_test[0] - i_test[1]) - (i_ref[0] - i_ref[1])) / (hi - lo)
    return (10**avg - 1) * 100, lo, hi


def gnc_arm(gnc, path, rgb):
    """The real encoder's rate/quality curve on this image: (bytes, Y-PSNR in YCoCg-R, dE00)."""
    out, png = "/tmp/_nlgate.gnc", "/tmp/_nlgate.png"
    ref_y = ycocg_r(rgb)[0]
    ref_lab = srgb_to_lab(rgb)
    curve = []
    for q in QUALITIES:
        subprocess.run([gnc, "encode", "-i", str(path), "-o", out, "-q", str(q)],
                       capture_output=True, check=True)
        subprocess.run([gnc, "decode", "-i", out, "-o", png],
                       capture_output=True, check=True)
        dec = np.asarray(Image.open(png).convert("RGB"))
        curve.append((
            q,
            os.path.getsize(out),
            psnr(ref_y, ycocg_r(dec)[0]),
            float(np.mean(ciede2000(ref_lab, srgb_to_lab(dec)))),
        ))
    os.remove(out)
    os.remove(png)
    return curve


def gnc_encode_size(gnc, path, q, extra_env=None):
    out = "/tmp/_nlgate.gnc"
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    subprocess.run(
        [gnc, "encode", "-i", str(path), "-o", out, "-q", str(q)],
        capture_output=True,
        env=env,
        check=True,
    )
    n = os.path.getsize(out)
    os.remove(out)
    return n


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    gnc, images = sys.argv[1], sys.argv[2:]
    print(f"chroma step multiplier {CHROMA_MUL} (the codec's chroma_weight above q=60)\n")

    summary = []
    for path in images:
        name = Path(path).stem
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        planes = ycocg_r(rgb)
        # YCoCg-R ranges for 8-bit input: Y in [0, 255], Co and Cg in [-255, 255].
        ranges = ((0, 255), (-255, 255), (-255, 255))
        ref_lab = srgb_to_lab(rgb)

        # Calibration: the same model at delta=1 against the real lossless file. Anything the real
        # coder spends that the model does not shows up here, once, per image.
        real_lossless = gnc_encode_size(gnc, path, 100)

        model = []
        for delta in DELTAS:
            bits = 0.0
            rec_planes = []
            for i, (plane, (lo, hi)) in enumerate(zip(planes, ranges)):
                # Integer step per plane. A fractional step does not divide the integer pixel
                # lattice, so the residual alphabet grows with the step's denominator and the
                # modelled rate *rises* as the quantiser coarsens — a modelling artefact, not a
                # property of DPCM. The codec's chroma_weight is applied as a rounded integer.
                step = delta if i == 0 else max(1.0, round(delta * CHROMA_MUL))
                q, rec = dpcm_closed_loop(plane, step, lo, hi)
                bits += entropy_bits(q)
                rec_planes.append(rec)
            recon = np.clip(ycocg_r_inv(*rec_planes), 0, 255).astype(np.uint8)
            model.append((
                delta,
                int(bits / 8),
                psnr(planes[0], ycocg_r(recon)[0]),
                float(np.mean(ciede2000(ref_lab, srgb_to_lab(recon)))),
                int(np.max(np.abs(recon.astype(np.int32) - rgb.astype(np.int32)))),
            ))

        # Calibrate against a model that is actually lossless: delta=1 on every plane, chroma
        # multiplier included, so the ratio measures only what the real coder spends above the
        # zeroth-order entropy. With the multiplier applied the delta=1 rung is *not* exact
        # (chroma step 1.2), which would fold a quantisation loss into the calibration.
        bits_ll = sum(
            entropy_bits(dpcm_closed_loop(plane, 1.0, lo, hi)[0])
            for plane, (lo, hi) in zip(planes, ranges)
        )
        model_lossless = int(bits_ll / 8)
        ratio = real_lossless / model_lossless
        px = w * h

        print(f"=== {name} ({w}x{h}) — model calibration {real_lossless:,} / "
              f"{model_lossless:,} = {ratio:.3f}x")
        print(f"{'arm':>14s} {'bytes':>10s} {'bpp':>6s} {'Y-PSNR':>8s} {'dE00':>7s} {'max err':>7s}")
        gnc_curve = gnc_arm(gnc, path, rgb)
        for q, b, yp, de in gnc_curve:
            print(f"{'gnc q=' + str(q):>14s} {b:>10,} {b * 8 / px:>6.3f} {yp:>8.3f} {de:>7.4f}"
                  f" {'':>7s}")
        for delta, b, yp, de, mx in model:
            cal = int(b * ratio)
            yps = "   exact" if yp == float("inf") else f"{yp:8.3f}"
            print(f"{'MED d=' + f'{delta:g}':>14s} {cal:>10,} {cal * 8 / px:>6.3f} {yps}"
                  f" {de:>7.4f} {mx:>7d}")

        # BD-rate, calibrated model against the real encoder, on luma. Lossless rungs (infinite
        # PSNR) cannot enter a BD-rate; they are the calibration point, not a curve point.
        g_r = [b for _, b, _, _ in gnc_curve]
        g_p = [yp for _, _, yp, _ in gnc_curve]
        m_r = [int(b * ratio) for d, b, yp, _, _ in model if yp != float("inf")]
        m_p = [yp for d, b, yp, _, _ in model if yp != float("inf")]
        bd, lo, hi = bd_rate(g_r, g_p, m_r, m_p)
        # And the same on colour, treating dE00 as a distortion: -20*log10(dE00) is monotone in
        # quality, so a BD-rate over it answers "fewer bits at the same colour error?".
        g_c = [-20 * np.log10(de) for _, _, _, de in gnc_curve]
        m_c = [-20 * np.log10(de) for d, b, yp, de, _ in model if yp != float("inf")]
        bdc, clo, chi = bd_rate(g_r, g_c, m_r, m_c)
        print(f"  BD-rate luma  {bd:+7.2f}%   over {lo:.2f}-{hi:.2f} dB")
        print(f"  BD-rate dE00  {bdc:+7.2f}%   over dE00 {10 ** (-chi / 20):.3f}-"
              f"{10 ** (-clo / 20):.3f}\n")
        summary.append((name, bd, bdc))

    print(f"{'image':16s} {'BD-rate luma':>13s} {'BD-rate dE00':>13s}")
    for name, bd, bdc in summary:
        print(f"{name:16s} {bd:>12.2f}% {bdc:>12.2f}%")
    if summary:
        print(f"{'mean':16s} {np.mean([b for _, b, _ in summary]):>12.2f}% "
              f"{np.mean([c for _, _, c in summary]):>12.2f}%")


if __name__ == "__main__":
    main()
