#!/usr/bin/env python3
"""MEAS-4 — offline decomposition of GNC's inter-coding gap vs H.264.

Reads spatial-domain motion-compensated residuals dumped by the encoder
(`GNC_DUMP_RESIDUAL=<dir> GNC_DIAGNOSTICS=1`) and computes upper bounds on what a
*different coding model* could achieve on the same prediction. Nothing here changes the
codec; the point is to find out whether the gap lives in the model or in the prediction
before anyone builds a new inter pipeline.

  4a  Residual subband energy distribution   — is the wavelet the wrong transform here?
  4b  Oracle block-skip / DCT bound          — the decision experiment
  4c  Entropy context ceiling                — what context modelling could recover

Run:  python3 scripts/meas4_oracle.py <dump_dir> --width 1920 --height 1080 --qstep <q>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_residual(path, crop_w, crop_h):
    """Load one dumped plane and crop away the tile padding.

    The dump is the *padded* plane (e.g. 2048x1280 for 1080p). The padding is
    identically zero, and leaving it in would inflate every "fraction of blocks below
    threshold" statistic towards 1 — the single most important correctness detail in
    this script.
    """
    meta = json.load(open(path.replace(".f32", ".json")))
    w, h = meta["width"], meta["height"]
    a = np.fromfile(path, dtype="<f4", count=w * h).reshape(h, w)
    # Crop to a whole number of 16x16 blocks so the 16x16 skip grid and the 8x8 transform
    # grid stay aligned (1080 is not a multiple of 16; the lost 8 rows are 0.7% of the frame).
    cw, ch = crop_w - crop_w % 16, crop_h - crop_h % 16
    return a[:ch, :cw].astype(np.float64)


# ---------------------------------------------------------------------------
# 4a — subband energy of the residual
# ---------------------------------------------------------------------------


def dwt53_1d(x, axis):
    """One level of a LeGall 5/3 lifting DWT along `axis`. Integer-ish, reversible shape.

    5/3 rather than CDF 9/7: the question here is how residual energy *distributes*
    across subbands, which is a property of the multiresolution split, not of the exact
    filter taps. 5/3 keeps the lifting short and the answer is not sensitive to the
    difference (checked: 9/7 shifts the LL share by well under a percentage point).
    """
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    even = x[..., 0::2].copy()
    odd = x[..., 1::2].copy()
    # Predict: odd -= (even_l + even_r) / 2
    even_r = np.concatenate([even[..., 1:], even[..., -1:]], axis=-1)
    odd = odd - 0.5 * (even + even_r)
    # Update: even += (odd_l + odd_r) / 4
    odd_l = np.concatenate([odd[..., :1], odd[..., :-1]], axis=-1)
    even = even + 0.25 * (odd_l + odd)
    return np.moveaxis(even, -1, axis), np.moveaxis(odd, -1, axis)


def subband_energy(img, levels=3):
    """Per-subband energy shares of a 2D dyadic wavelet decomposition."""
    cur = img
    out = {}
    for lv in range(1, levels + 1):
        l, h = dwt53_1d(cur, axis=1)   # columns -> L, H
        ll, lh = dwt53_1d(l, axis=0)
        hl, hh = dwt53_1d(h, axis=0)
        out[f"LH{lv}"] = float((lh ** 2).sum())
        out[f"HL{lv}"] = float((hl ** 2).sum())
        out[f"HH{lv}"] = float((hh ** 2).sum())
        cur = ll
    out["LL"] = float((cur ** 2).sum())
    return out


def idwt53_1d(even, odd, axis):
    """Inverse of `dwt53_1d`."""
    even = np.moveaxis(even, axis, -1)
    odd = np.moveaxis(odd, axis, -1)
    odd_l = np.concatenate([odd[..., :1], odd[..., :-1]], axis=-1)
    even = even - 0.25 * (odd_l + odd)
    even_r = np.concatenate([even[..., 1:], even[..., -1:]], axis=-1)
    odd = odd + 0.5 * (even + even_r)
    n = even.shape[-1] + odd.shape[-1]
    out = np.empty(even.shape[:-1] + (n,), dtype=even.dtype)
    out[..., 0::2] = even
    out[..., 1::2] = odd
    return np.moveaxis(out, -1, axis)


def dwt2(img, levels):
    """Full 2D dyadic decomposition. Returns (LL, [(LH, HL, HH) per level])."""
    cur = img
    bands = []
    for _ in range(levels):
        l, h = dwt53_1d(cur, axis=1)
        ll, lh = dwt53_1d(l, axis=0)
        hl, hh = dwt53_1d(h, axis=0)
        bands.append((lh, hl, hh))
        cur = ll
    return cur, bands


def idwt2(ll, bands):
    cur = ll
    for lh, hl, hh in reversed(bands):
        l = idwt53_1d(cur, lh, axis=0)
        h = idwt53_1d(hl, hh, axis=0)
        cur = idwt53_1d(l, h, axis=1)
    return cur


def subband_gains(shape, levels):
    """L2 norm of each subband's synthesis basis function, measured numerically.

    The lifting DWT is not orthonormal, so a quantization error of a given size in the
    transform domain contributes a *different* amount of spatial MSE depending on which
    subband it lands in. Quantizing every subband with the same step would therefore be a
    badly normalised coder, and comparing that against an orthonormal DCT would flatter the
    DCT for no reason other than scaling. Dividing each subband by its synthesis gain
    restores the Parseval-like property the DCT already has, so the two models are compared
    on their structure rather than on their normalisation.
    """
    h, w = shape
    key = (h, w, levels)
    if key in _GAIN_CACHE:
        return _GAIN_CACHE[key]
    zero = np.zeros((h, w))
    ll0, bands0 = dwt2(zero, levels)
    gains = {}
    # LL
    imp = np.zeros_like(ll0)
    imp[imp.shape[0] // 2, imp.shape[1] // 2] = 1.0
    gains["LL"] = float(np.sqrt((idwt2(imp, [tuple(np.zeros_like(b) for b in lv)
                                             for lv in bands0]) ** 2).sum()))
    for lv, (lh, hl, hh) in enumerate(bands0):
        for bi, name in enumerate(("LH", "HL", "HH")):
            bands = [tuple(np.zeros_like(b) for b in l) for l in bands0]
            imp = np.zeros_like(bands[lv][bi])
            imp[imp.shape[0] // 2, imp.shape[1] // 2] = 1.0
            bands[lv] = tuple(imp if k == bi else bands[lv][k] for k in range(3))
            rec = idwt2(np.zeros_like(ll0), bands)
            gains[f"{name}{lv+1}"] = float(np.sqrt((rec ** 2).sum()))
    _GAIN_CACHE[key] = gains
    return gains


_GAIN_CACHE = {}


def quantize(x, qstep, dead_zone=0.0):
    """Uniform quantizer with a dead zone, matching the shape of GNC's."""
    t = np.abs(x) / qstep
    q = np.floor(np.maximum(t - dead_zone, 0.0) + 0.5)
    return (np.sign(x) * q).astype(np.int32)


def dequantize(q, qstep, dead_zone=0.0):
    mag = np.abs(q).astype(np.float64)
    mag = np.where(mag > 0, mag + dead_zone, 0.0)
    return np.sign(q) * mag * qstep


def simulate_wavelet(resid, qstep, levels=3, dead_zone=0.75):
    """GNC's own model, simulated: DWT -> quantize -> entropy, and the MSE it lands at.

    Entropy is measured per subband (GNC codes each subband with its own parameters), using
    the Shannon entropy of the quantized coefficients — i.e. an *ideal* coder for this model.
    That makes this an upper bound on GNC's own scheme, so comparing it against the DCT
    simulation compares the two models at their respective ceilings rather than comparing a
    tuned implementation against an idealised rival.
    """
    ll, bands = dwt2(resid, levels)
    g = subband_gains(resid.shape, levels)
    bits = 0.0
    # Scale by the synthesis gain so one quantizer step costs the same spatial MSE in every
    # subband (see subband_gains).
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
    mse = float(((resid - rec) ** 2).mean())
    return bits, mse


def simulate_dct(resid, qstep, skip_bs=16, dead_zone=0.0):
    """The rival model: oracle 16x16 block skip + 8x8 DCT + per-position entropy coding.

    Deliberately generous — the skip decision is an oracle, the entropy figure is an ideal
    arithmetic coder with a perfect per-coefficient-position model, and there is no side
    information beyond 1 bit per block. A real coder cannot beat this.
    """
    bs = 8
    h, w = resid.shape
    b16 = to_blocks(resid, skip_bs)
    n16 = b16.shape[0]
    peak = np.abs(b16).reshape(n16, -1).max(axis=1)
    skippable = peak < (qstep / 2.0)

    by16, bx16 = h // skip_bs, w // skip_bs
    r = skip_bs // bs
    skip8 = np.repeat(np.repeat(skippable.reshape(by16, bx16), r, axis=0), r, axis=1)

    b8 = to_blocks(resid, bs)
    keep = ~skip8.reshape(-1)
    coded = b8[keep]

    rec8 = np.zeros_like(b8)
    bits = float(n16)  # 1 bit per 16x16 skip flag
    if coded.shape[0] > 0:
        d = dct2_8x8_blocks(coded)
        q = quantize(d, qstep, dead_zone)
        for i in range(bs):
            for j in range(bs):
                bits += shannon_bits(q[:, i, j])
        dq = dequantize(q, qstep, dead_zone)
        rec8[keep] = idct2_8x8_blocks(dq)

    by8, bx8 = h // bs, w // bs
    rec = (
        rec8.reshape(by8, bx8, bs, bs)
        .transpose(0, 2, 1, 3)
        .reshape(by8 * bs, bx8 * bs)
    )
    mse = float(((resid - rec) ** 2).mean())
    return bits, mse, float(skippable.mean())


def idct2_8x8_blocks(blocks):
    n = 8
    k = np.arange(n)
    c = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    sc = np.full(n, np.sqrt(2.0 / n))
    sc[0] = np.sqrt(1.0 / n)
    m = (c * sc[None, :]).T
    return m.T @ blocks @ m


def bits_at_mse(points, target_mse):
    """Log-linear interpolation of bits at a target MSE from (mse, bits) samples."""
    pts = sorted(points)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if target_mse <= xs[0]:
        return float(ys[0]), "extrapolated-low"
    if target_mse >= xs[-1]:
        return float(ys[-1]), "extrapolated-high"
    return float(np.interp(np.log(target_mse), np.log(xs), ys)), "interpolated"


# ---------------------------------------------------------------------------
# 4b — oracle block-skip / DCT bound
# ---------------------------------------------------------------------------


def dct2_8x8_blocks(blocks):
    """Orthonormal 2D DCT-II over the last two axes of an (..., 8, 8) array."""
    n = 8
    k = np.arange(n)
    c = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    s = np.full(n, np.sqrt(2.0 / n))
    s[0] = np.sqrt(1.0 / n)
    m = (c * s[None, :]).T  # (freq, spatial)
    return m @ blocks @ m.T


def to_blocks(img, bs):
    h, w = img.shape
    h2, w2 = (h // bs) * bs, (w // bs) * bs
    return (
        img[:h2, :w2]
        .reshape(h2 // bs, bs, w2 // bs, bs)
        .transpose(0, 2, 1, 3)
        .reshape(-1, bs, bs)
    )


def shannon_bits(sym):
    """Zeroth-order entropy of an integer array, in bits (0 for an empty array)."""
    if sym.size == 0:
        return 0.0
    _, counts = np.unique(sym, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum() * sym.size)


def oracle_bound(resid, qstep, skip_bs=16):
    """Bits an idealised block-based coder would spend on this residual.

    Deliberately generous to the hypothetical coder — it is an *upper bound on the gain*,
    not a design:
      * skip decision is an oracle (free knowledge of which blocks quantize to nothing),
      * the entropy figure is the zeroth-order Shannon entropy of the quantized
        coefficients, i.e. an ideal arithmetic coder with a perfect static model,
      * no side information beyond 1 bit per block for the skip flag,
      * no MV cost (identical to GNC's, so it cancels in the comparison).
    A real coder cannot beat this. If GNC is already close to it, a rebuilt inter pipeline
    has nothing to win.
    """
    n_skip_blocks = 0
    total_skip_blocks = 0

    # --- oracle skip on 16x16 blocks ---
    b16 = to_blocks(resid, skip_bs)
    total_skip_blocks = b16.shape[0]
    # A block is skippable if every sample quantizes to zero under a uniform quantizer.
    peak = np.abs(b16).reshape(total_skip_blocks, -1).max(axis=1)
    skippable = peak < (qstep / 2.0)
    n_skip_blocks = int(skippable.sum())

    # --- energy concentration over 16x16 blocks ---
    energy = (b16.astype(np.float64) ** 2).reshape(total_skip_blocks, -1).sum(axis=1)
    order = np.sort(energy)[::-1]
    tot = order.sum()
    conc = {}
    for frac in (0.10, 0.25, 0.50):
        k = max(1, int(round(frac * total_skip_blocks)))
        conc[frac] = float(order[:k].sum() / tot) if tot > 0 else 0.0

    # --- oracle DCT bits on the non-skipped area ---
    # Map the 16x16 skip decision onto the 8x8 transform grid.
    bs = 8
    h, w = resid.shape
    bx16, by16 = w // skip_bs, h // skip_bs
    skip_map = skippable.reshape(by16, bx16)
    skip8 = np.repeat(np.repeat(skip_map, skip_bs // bs, axis=0), skip_bs // bs, axis=1)

    b8 = to_blocks(resid, bs)
    by8, bx8 = h // bs, w // bs
    keep = ~skip8.reshape(-1)[: b8.shape[0]]
    coded = b8[keep]

    if coded.shape[0] == 0:
        coeff_bits = 0.0
    else:
        d = dct2_8x8_blocks(coded)
        q = np.rint(d / qstep).astype(np.int32)
        # Entropy per coefficient position: a real coder models DC and each AC band
        # separately, so give the oracle that too.
        coeff_bits = 0.0
        for i in range(bs):
            for j in range(bs):
                coeff_bits += shannon_bits(q[:, i, j])

    skip_bits = float(total_skip_blocks)  # 1 bit per 16x16 block
    return {
        "skip_frac": n_skip_blocks / total_skip_blocks if total_skip_blocks else 0.0,
        "energy_conc": conc,
        "bits": coeff_bits + skip_bits,
        "coeff_bits": coeff_bits,
        "skip_bits": skip_bits,
        "n_blocks16": total_skip_blocks,
    }


# ---------------------------------------------------------------------------
# 4c — entropy context ceiling
# ---------------------------------------------------------------------------


def context_entropy(resid, qstep):
    """Zeroth-order vs context-conditioned entropy of quantized DCT coefficients.

    Bounds what context-adaptive entropy coding could recover relative to a context-free
    coder on the *same* symbols. Context = quantized magnitude of the same coefficient
    position in the block to the left, bucketed to 4 levels.
    """
    bs = 8
    b8 = to_blocks(resid, bs)
    h, w = resid.shape
    bx = w // bs
    by = h // bs
    d = dct2_8x8_blocks(b8)
    q = np.rint(d / qstep).astype(np.int32).reshape(by, bx, bs, bs)

    h0 = 0.0
    hc = 0.0
    for i in range(bs):
        for j in range(bs):
            band = q[:, :, i, j]
            h0 += shannon_bits(band.reshape(-1))
            left = np.concatenate([band[:, :1], band[:, :-1]], axis=1)
            mag = np.abs(left)
            ctx = np.digitize(mag, [1, 2, 5])  # 4 buckets
            for c in range(4):
                sel = band[ctx == c]
                hc += shannon_bits(sel)
    return {"h0_bits": h0, "hctx_bits": hc}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--width", type=int, required=True, help="real (uncropped) frame width")
    ap.add_argument("--height", type=int, required=True, help="real frame height")
    ap.add_argument("--qstep", type=float, required=True,
                    help="quantization step the sequence was encoded at")
    ap.add_argument("--gnc-inter-bpp", type=float, default=None,
                    help="measured GNC coefficient bpp averaged over the P/B frames")
    ap.add_argument("--label", default="")
    ap.add_argument("--max-planes", type=int, default=0,
                    help="analyse only the first N planes (0 = all)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dump_dir, "resid_*.f32")))
    if not files:
        sys.exit(f"no residual dumps in {args.dump_dir}")
    if args.max_planes:
        files = files[: args.max_planes]

    aw, ah = args.width - args.width % 16, args.height - args.height % 16
    # GNC reports bpp as (bits for all three planes) / luma pixels, so aggregate per FRAME:
    # sum the Y/Co/Cg planes and divide by the luma pixel count once. Averaging per plane
    # instead would understate every figure here by a factor of 3.
    n_frames = sum(1 for f in files if os.path.basename(f).split("_")[1] in ("Py", "By"))
    if n_frames == 0:
        n_frames = max(1, len(files) // 3)
    px = aw * ah

    print(f"\n=== MEAS-4 {args.label} ===")
    print(f"{len(files)} residual planes over {n_frames} inter frames, analysed at {aw}x{ah} "
          f"(cropped from the padded dump; padding excluded), qstep={args.qstep}")

    # RD sweep: both models over the same quantizer ladder, on the same residuals.
    # Wide enough that the rival model can reach the distortion the wavelet model lands at,
    # at every quality point we test. If it cannot, the comparison is extrapolated and the
    # script refuses to report a number rather than quoting a clamped one.
    ladder = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0]
    agg_sub = {}
    wav = {q: [0.0, 0.0] for q in ladder}   # [bits, mse-sum]
    dct = {q: [0.0, 0.0] for q in ladder}
    skipfrac = {q: 0.0 for q in ladder}
    conc_tot = {0.10: 0.0, 0.25: 0.0, 0.50: 0.0}
    h0_tot = hc_tot = 0.0

    for f in files:
        r = load_residual(f, args.width, args.height)

        sb = subband_energy(r, levels=3)
        tot = sum(sb.values())
        for k, v in sb.items():
            agg_sub[k] = agg_sub.get(k, 0.0) + (v / tot if tot > 0 else 0.0)

        b16 = to_blocks(r, 16)
        e = (b16 ** 2).reshape(b16.shape[0], -1).sum(axis=1)
        order = np.sort(e)[::-1]
        se = order.sum()
        for frac in conc_tot:
            k = max(1, int(round(frac * b16.shape[0])))
            conc_tot[frac] += float(order[:k].sum() / se) if se > 0 else 0.0

        for q in ladder:
            wb, wm = simulate_wavelet(r, q)
            wav[q][0] += wb
            wav[q][1] += wm
            db, dm, sf = simulate_dct(r, q)
            dct[q][0] += db
            dct[q][1] += dm
            skipfrac[q] += sf

        ce = context_entropy(r, args.qstep)
        h0_tot += ce["h0_bits"]
        hc_tot += ce["hctx_bits"]

    n = len(files)

    # --- 4a ---
    print("\n-- 4a: residual subband energy distribution --")
    order_k = ["LL"] + [f"{b}{l}" for l in (3, 2, 1) for b in ("LH", "HL", "HH")]
    detail = 0.0
    for k in order_k:
        if k not in agg_sub:
            continue
        share = agg_sub[k] / n * 100
        if k != "LL":
            detail += share
        print(f"   {k:>4}: {share:6.2f}%")
    print(f"   detail subbands total: {detail:.2f}%")
    print("   NOTE: an MC residual is high-pass by construction (prediction removes the low")
    print("         frequencies), so a high detail share is expected and this gate as originally")
    print("         written cannot discriminate. The RD comparison in 4b is the real test.")

    # --- 4b ---
    print("\n-- 4b: model comparison at matched distortion --")
    print("   Both models simulated offline on the same residuals, each with an ideal entropy")
    print("   coder, so this compares MODELS and not implementations.")
    print(f"   {'qstep':>6}  {'wavelet bpp':>12} {'wav MSE':>9}   {'DCT+skip bpp':>13} "
          f"{'DCT MSE':>9} {'skip%':>6}")
    wpts, dpts = [], []
    for q in ladder:
        wb = wav[q][0] / n_frames / px
        wm = wav[q][1] / n
        db = dct[q][0] / n_frames / px
        dm = dct[q][1] / n
        sf = skipfrac[q] / n * 100
        wpts.append((wm, wb))
        dpts.append((dm, db))
        print(f"   {q:6.1f}  {wb:12.4f} {wm:9.3f}   {db:13.4f} {dm:9.3f} {sf:6.1f}")

    print(f"\n   energy in top 10/25/50% of 16x16 blocks: "
          f"{conc_tot[0.10]/n*100:.1f}% / {conc_tot[0.25]/n*100:.1f}% / "
          f"{conc_tot[0.50]/n*100:.1f}%")

    # Compare at the distortion the wavelet model reaches at the encode qstep.
    ref_mse = wav[args.qstep][1] / n
    ref_bits = wav[args.qstep][0] / n_frames / px
    dct_bits, how = bits_at_mse(dpts, ref_mse)
    print(f"\n   At the wavelet model's operating point (qstep={args.qstep}, "
          f"MSE={ref_mse:.3f}):")
    print(f"     wavelet model (ideal coder): {ref_bits:.4f} bpp")
    if how != "interpolated":
        print(f"     DCT+oracle-skip model:       UNAVAILABLE ({how})")
        print("     The rival model's RD sweep does not span this distortion, so its rate here")
        print("     would be a clamped endpoint rather than a measurement. Widen the ladder.")
        print("\n   -- 4b INVALID at this quality point --")
        return
    print(f"     DCT+oracle-skip model:       {dct_bits:.4f} bpp  ({how})")
    red = (1 - dct_bits / ref_bits) * 100 if ref_bits > 0 else 0.0
    print(f"     the rival model is {red:+.1f}% vs the wavelet model at equal distortion")

    if args.gnc_inter_bpp:
        print(f"\n   Cross-check against the real encoder:")
        print(f"     GNC measured coefficient bpp: {args.gnc_inter_bpp:.4f}")
        print(f"     wavelet model, ideal coder:   {ref_bits:.4f}")
        gap = (args.gnc_inter_bpp / ref_bits - 1) * 100 if ref_bits > 0 else 0.0
        print(f"     GNC's entropy coder costs {gap:+.1f}% over an ideal coder for its own model")
        total = (1 - dct_bits / args.gnc_inter_bpp) * 100
        print(f"     rival model vs GNC as shipped: {total:.1f}% below")
        verdict = ("MODEL is the cap - a hybrid inter pipeline has real headroom" if total >= 40
                   else "PREDICTION QUALITY is the cap - a new coding model would not pay"
                   if total < 20 else "INCONCLUSIVE (20-40%) - needs a tie-breaker")
        print(f"     DECISION (MEAS-4b rule): {verdict}")

    # --- 4c ---
    print("\n-- 4c: entropy context ceiling --")
    print(f"   context-free (H0):    {h0_tot / n_frames / px:.4f} bpp")
    print(f"   1-neighbour context:  {hc_tot / n_frames / px:.4f} bpp")
    gain = (1 - hc_tot / h0_tot) * 100 if h0_tot > 0 else 0.0
    print(f"   context modelling could recover at most {gain:.1f}% of coefficient bits")
    print()


if __name__ == "__main__":
    main()
