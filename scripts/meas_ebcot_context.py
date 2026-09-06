#!/usr/bin/env python3
"""EBCOT part 2: what could a context-modelled bit-plane coder buy over GNC's Rice+ZRL?

Part 1 (`meas_ebcot_pcrd.py`) showed EBCOT's rate-allocation half is worth 0.00 dB here. This
measures the other half — the engine: three coding passes per bit-plane, with the significance bit
of each coefficient coded under a context derived from its 8-neighbourhood's significance state
and the subband's orientation.

All three figures come from the *same* quantised coefficients, so they are directly comparable:

  rice+zrl   Golomb-Rice with a per-subband optimal k plus zero-run-length, which is what GNC
             actually codes with. Simulated rather than read from the encoder so it sees exactly
             the coefficients the other two arms see.
  H0         zeroth-order entropy per subband — an ideal memoryless coder. GNC's floor.
  ebcot      empirical conditional entropy of each coded bit given its EBCOT context, summed over
             bit-planes, plus context-coded sign bits. This is the rate an adaptive arithmetic
             coder converges to, so it is a *ceiling* on what EBCOT's engine could achieve; the MQ
             coder's own adaptation loss is not modelled and would make the real thing worse.

The contexts follow JPEG 2000's significance-propagation table in spirit: the count of significant
horizontal, vertical and diagonal neighbours, bucketed, and separated by band orientation. Sign
bits get their own contexts from the signs of the horizontal and vertical neighbours. Magnitude
refinement bits are coded under three contexts as in the standard (first refinement with and
without significant neighbours, later refinements).

Run:  python3 scripts/meas_ebcot_context.py <png> [--levels 5] [--qstep 8]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meas4_oracle import dwt2, quantize, subband_gains  # noqa: E402


def cond_entropy_bits(contexts, bits):
    """Total bits an ideal adaptive coder spends on `bits` given `contexts`.

    Sum over contexts of n_c * H(p_c) — the conditional entropy, which is what an adaptive
    arithmetic coder converges to. Contexts seen once cost 0 by this measure, which flatters the
    result slightly; the alternative (a Krichevsky-Trofimov style penalty) is reported too.
    """
    if len(bits) == 0:
        return 0.0, 0.0
    contexts = np.asarray(contexts)
    bits = np.asarray(bits, dtype=np.int64)
    total = 0.0
    penalised = 0.0
    for c in np.unique(contexts):
        m = contexts == c
        n = int(m.sum())
        ones = int(bits[m].sum())
        p = ones / n
        if 0.0 < p < 1.0:
            h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        else:
            h = 0.0
        total += n * h
        # Adaptive-coder learning cost: ~0.5*log2(n) bits per context (KT bound).
        penalised += n * h + 0.5 * np.log2(max(n, 2))
    return total, penalised


def neighbour_counts(sig):
    """(horizontal, vertical, diagonal) significant-neighbour counts for every position."""
    p = np.pad(sig.astype(np.int8), 1)
    h = p[1:-1, :-2] + p[1:-1, 2:]
    v = p[:-2, 1:-1] + p[2:, 1:-1]
    d = p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
    return h, v, d


def zc_context(h, v, d, orient):
    """Zero-coding context, JPEG 2000 Table D.1 in spirit: 9 contexts per orientation class."""
    if orient == "HH":
        hv = h + v
        ctx = np.where(d >= 3, 8,
              np.where((d == 2) & (hv >= 1), 7,
              np.where((d == 2) & (hv == 0), 6,
              np.where((d == 1) & (hv >= 2), 5,
              np.where((d == 1) & (hv == 1), 4,
              np.where((d == 1) & (hv == 0), 3,
              np.where(hv >= 2, 2, np.where(hv == 1, 1, 0))))))))
    else:
        # For LL/LH/HL the standard swaps the roles of h and v by orientation; use h as the
        # "preferred" direction for HL and v for LH, which is what the table encodes.
        a, b = (v, h) if orient == "LH" else (h, v)
        ctx = np.where(a >= 2, 8,
              np.where((a == 1) & (b >= 1), 7,
              np.where((a == 1) & (b == 0) & (d >= 1), 6,
              np.where((a == 1) & (b == 0) & (d == 0), 5,
              np.where((a == 0) & (b >= 2), 4,
              np.where((a == 0) & (b == 1), 3,
              np.where((a == 0) & (b == 0) & (d >= 2), 2,
              np.where((a == 0) & (b == 0) & (d == 1), 1, 0))))))))
    return ctx


def band_bits(coef, orient, band_id):
    """Simulate EBCOT bit-plane coding of one subband; return (ebcot_bits, ebcot_penalised)."""
    mag = np.abs(coef).astype(np.int64)
    sign = (coef < 0).astype(np.int8)
    if mag.max() == 0:
        return 0.0, 0.0
    nplanes = int(mag.max()).bit_length()

    sig = np.zeros(mag.shape, dtype=bool)
    refined_once = np.zeros(mag.shape, dtype=bool)
    zc_ctx_all, zc_bit_all = [], []
    sc_ctx_all, sc_bit_all = [], []
    mr_ctx_all, mr_bit_all = [], []

    for plane in range(nplanes - 1, -1, -1):
        bit = ((mag >> plane) & 1).astype(np.int8)
        h, v, d = neighbour_counts(sig)

        # Magnitude refinement: coefficients already significant.
        mr = sig
        if mr.any():
            nb = (h + v + d) > 0
            ctx = np.where(refined_once[mr], 2, np.where(nb[mr], 1, 0)).astype(np.int64)
            mr_ctx_all.append(ctx + band_id * 3)
            mr_bit_all.append(bit[mr])
            refined_once = refined_once | mr

        # Significance coding: not yet significant. Context from the neighbourhood.
        ns = ~sig
        if ns.any():
            ctx = zc_context(h, v, d, orient)
            zc_ctx_all.append(ctx[ns].astype(np.int64) + band_id * 9)
            zc_bit_all.append(bit[ns])
            # Sign coding for those that just became significant.
            became = ns & (bit == 1)
            if became.any():
                ps = np.pad(np.where(sig, np.where(sign == 1, -1, 1), 0).astype(np.int8), 1)
                hc = np.clip(ps[1:-1, :-2] + ps[1:-1, 2:], -1, 1)
                vc = np.clip(ps[:-2, 1:-1] + ps[2:, 1:-1], -1, 1)
                sc_ctx_all.append(
                    ((hc[became].astype(np.int64) + 1) * 3 + (vc[became].astype(np.int64) + 1))
                    + band_id * 9
                )
                sc_bit_all.append(sign[became])
            sig = sig | became

    total = pen = 0.0
    for ctxs, bits in ((zc_ctx_all, zc_bit_all), (sc_ctx_all, sc_bit_all), (mr_ctx_all, mr_bit_all)):
        if ctxs:
            t, p = cond_entropy_bits(np.concatenate(ctxs), np.concatenate(bits))
            total += t
            pen += p
    return total, pen


def rice_zrl_bits(coef):
    """Golomb-Rice with the best per-band k, plus zero-run-length, as GNC codes."""
    v = coef.astype(np.int64).ravel()
    nz = v != 0
    # Zero runs between non-zeros, Rice-coded with their own k.
    runs = []
    run = 0
    for is_nz in nz:
        if is_nz:
            runs.append(run)
            run = 0
        else:
            run += 1
    runs.append(run)
    mags = np.abs(v[nz])
    if mags.size == 0:
        return float(len(runs) * 2)

    def rice_cost(vals):
        vals = np.asarray(vals, dtype=np.int64)
        best = None
        for k in range(16):
            c = int(((vals >> k) + 1 + k).sum())
            if best is None or c < best:
                best = c
        return best

    # magnitudes coded as (mag-1), plus one sign bit each
    return float(rice_cost(mags - 1) + mags.size + rice_cost(np.array(runs)))


def rice_zrl_split_bits(coef, nstreams=256):
    """Rice+ZRL as GNC actually codes it: the tile split into `nstreams` independent streams.

    GNC gives each tile 256 fully independent entropy streams for GPU parallelism. Independent
    streams cannot share a zero run across a boundary and each pays for its own length field, so
    this is strictly worse than one stream per subband — and it is what the codec really does, so
    the gap between this and `rice_zrl_bits` is the price of the parallelism.
    """
    v = coef.astype(np.int64).ravel()
    if v.size == 0:
        return 0.0
    total = 0.0
    # Interleave as the shader does: stream s takes every nstreams-th coefficient.
    for s0 in range(min(nstreams, v.size)):
        part = v[s0::nstreams]
        if part.size == 0:
            continue
        total += rice_zrl_bits(part)
        total += 8.0  # the stream's length field, Rice-coded per GP17: about a byte
    return total


def context_coefficient_bits(coef, table_bits=8.0, nctx_mag=4):
    """Conditional entropy of the whole quantised coefficient given a neighbourhood context.

    This is the cheap alternative to EBCOT: keep coding coefficients as symbols (which is what
    GNC's rANS backend already does with per-subband frequency tables) and condition the table on
    a context instead. No bit-planes, no MQ coder, no three passes — the existing interleaved
    rANS machinery with more tables.

    Context = bucketed sum of the absolute values of the four causal neighbours (left, up,
    up-left, up-right), which is available to a decoder in raster order.
    """
    a = np.abs(coef).astype(np.int64)
    p = np.pad(a, 1)
    nb = p[1:-1, :-2] + p[:-2, 1:-1] + p[:-2, :-2] + p[:-2, 2:]
    # Bucket: 0, 1, 2-3, 4-7, 8+  (five buckets; nctx_mag controls the log spread)
    ctx = np.zeros_like(nb)
    ctx = np.where(nb == 0, 0, np.minimum(np.log2(np.maximum(nb, 1)).astype(np.int64) + 1, nctx_mag + 1))
    total = 0.0
    v = coef.astype(np.int64).ravel()
    c = ctx.ravel()
    for cc in np.unique(c):
        m = c == cc
        vals = v[m]
        _, counts = np.unique(vals, return_counts=True)
        pr = counts / counts.sum()
        total += float(-(pr * np.log2(pr)).sum() * vals.size)
        # Each context needs its own frequency table. rANS pays for these: charge the alphabet.
        total += float(counts.size) * table_bits
    return total


def vertical_context_bits(coef, table_bits=8.0):
    """Context from the vertical neighbours only — the ones GNC can already reach.

    Rice and rANS map coefficient `i` of a tile to stream `i % STREAMS_PER_TILE`, so with 256
    streams in a 256-wide tile **each stream is one column**: consecutive symbols within a stream
    are vertically adjacent. The row above is therefore already decoded when the current symbol is
    decoded, in the existing architecture, with no restructuring at all. The horizontal neighbours
    live in sibling streams decoded concurrently and are not available.

    So this is the context GNC can have for free, as opposed to the full neighbourhood which needs
    EBCOT's per-code-block sequential model.
    """
    a = np.abs(coef).astype(np.int64)
    up1 = np.vstack([np.zeros((1, a.shape[1]), dtype=np.int64), a[:-1]])
    up2 = np.vstack([np.zeros((2, a.shape[1]), dtype=np.int64), a[:-2]])
    nb = up1 * 2 + up2
    ctx = np.where(nb == 0, 0, np.minimum(np.log2(np.maximum(nb, 1)).astype(np.int64) + 1, 5))
    total = 0.0
    v = coef.astype(np.int64).ravel()
    c = ctx.ravel()
    for cc in np.unique(c):
        vals = v[c == cc]
        _, counts = np.unique(vals, return_counts=True)
        pr = counts / counts.sum()
        total += float(-(pr * np.log2(pr)).sum() * vals.size)
        total += float(counts.size) * table_bits
    return total


def binary_adaptive_vertical_bits(coef):
    """The configuration actually recommended: CABAC-style binarisation, adaptive contexts, no
    frequency tables, context from the *fully decoded* vertical neighbours.

    Why this shape rather than EBCOT's:
      - GNC decodes each stream's symbols in order, and a stream is a tile column, so when a
        coefficient is decoded both coefficients above it are fully known — not partially known as
        in a plane-major bit-plane scan. That is a strictly richer context, for free.
      - Binary decisions mean an adaptive binary coder can be used, which carries no tables. The
        table cost is what kills symbol-level context coding here (see the --table-bits sweep).
      - Giving up plane-major order gives up truncatability, which part 1 measured at 0.00 dB.

    Binarisation per coefficient: significant?, then |v|>1?, |v|>2?, then the remainder as raw
    Exp-Golomb suffix bits (uncoded, as CABAC bypasses them), then the sign. Every coded decision
    gets a context from the bucketed magnitudes of the two coefficients above; suffix and sign bits
    are charged at one bit each, uncoded.
    """
    a = np.abs(coef).astype(np.int64)
    up1 = np.vstack([np.zeros((1, a.shape[1]), dtype=np.int64), a[:-1]])
    up2 = np.vstack([np.zeros((2, a.shape[1]), dtype=np.int64), a[:-2]])
    nb = up1 * 2 + up2
    ctx = np.where(nb == 0, 0, np.minimum(np.log2(np.maximum(nb, 1)).astype(np.int64) + 1, 5))

    v = a.ravel()
    c = ctx.ravel()
    total = 0.0

    # decision 1: significant?
    t, _ = cond_entropy_bits(c, (v > 0).astype(np.int64))
    total += t
    # decision 2: |v| > 1, among the significant
    m = v > 0
    if m.any():
        t, _ = cond_entropy_bits(c[m], (v[m] > 1).astype(np.int64))
        total += t
    # decision 3: |v| > 2, among those
    m2 = v > 1
    if m2.any():
        t, _ = cond_entropy_bits(c[m2], (v[m2] > 2).astype(np.int64))
        total += t
    # remainder: Exp-Golomb order 0 of (|v| - 3), bypassed at one bit per bit
    m3 = v > 2
    if m3.any():
        rem = v[m3] - 3
        total += float((2 * np.floor(np.log2(rem + 1)) + 1).sum())
    # sign, one bypassed bit per significant coefficient
    total += float((v > 0).sum())
    return total


def binary_adaptive_per_stream_bits(coef, nstreams=256, warm_start=True):
    """The same coder, but with each of the 256 streams adapting its contexts independently.

    This is what a parallel decode forces: stream `s` holds coefficients s, s+256, s+512, ... and
    decodes them in order, so its context is its own previous two symbols and its probability
    estimates can only be learned from its own history. That is roughly 256 symbols per stream to
    learn 3 decisions x 6 buckets on, which is little.

    `warm_start=True` charges a per-tile initial-probability table (one byte per context, as GNC
    already signals per-subband k) and then charges only the conditional entropy; `False` charges
    the KT learning cost instead, modelling a cold start with no table.
    """
    a = np.abs(coef).astype(np.int64)
    flat = a.ravel()
    n = flat.size
    total = 0.0
    NB = 6
    # Accumulate (context, decision) pairs per stream, then charge each stream separately.
    for s0 in range(min(nstreams, n)):
        part = flat[s0::nstreams]
        if part.size == 0:
            continue
        up1 = np.concatenate([[0], part[:-1]])
        up2 = np.concatenate([[0, 0], part[:-2]])
        nb = up1 * 2 + up2
        ctx = np.where(nb == 0, 0, np.minimum(np.log2(np.maximum(nb, 1)).astype(np.int64) + 1, NB - 1))
        for sel, dec in (
            (np.ones(part.shape, dtype=bool), part > 0),
            (part > 0, part > 1),
            (part > 1, part > 2),
        ):
            if sel.any():
                t, p = cond_entropy_bits(ctx[sel], dec[sel].astype(np.int64))
                total += t if warm_start else p
        m3 = part > 2
        if m3.any():
            rem = part[m3] - 3
            total += float((2 * np.floor(np.log2(rem + 1)) + 1).sum())
        total += float((part > 0).sum())
    if warm_start:
        total += 3 * NB * 8.0  # per-tile initial probabilities, one byte per context
    return total


def codeblock_adaptive_bits(coef, cb=64, warm_start=False):
    """EBCOT's actual design: independent code-blocks, raster scan inside each, full neighbourhood.

    This is the configuration that resolves the tension the per-stream measurement exposed. GNC's
    256-streams-per-tile gives each coder only ~256 symbols to adapt 18 context probabilities on,
    which wipes out the gain. A 64x64 code-block gives one coder 4096 symbols — enough to adapt —
    and a 1080p plane still holds about 640 independent code-blocks, which is ample GPU
    parallelism even though it is not 256-per-tile.

    Inside a block the scan is raster, so left, up, up-left and up-right are all decoded and the
    full neighbourhood context is available, not just the vertical one.

    Charged with the KT learning cost by default (`warm_start=False`), i.e. each block starts cold
    with no signalled table — the honest number for an adaptive coder.
    """
    a = np.abs(coef).astype(np.int64)
    NB = 6
    total = 0.0
    h, w = a.shape
    nblocks = 0
    for by in range(0, h, cb):
        for bx in range(0, w, cb):
            blk = a[by : by + cb, bx : bx + cb]
            if blk.size == 0:
                continue
            nblocks += 1
            p = np.pad(blk, 1)
            nb = p[1:-1, :-2] + p[:-2, 1:-1] + p[:-2, :-2] + p[:-2, 2:]
            ctx = np.where(nb == 0, 0,
                           np.minimum(np.log2(np.maximum(nb, 1)).astype(np.int64) + 1, NB - 1))
            v = blk.ravel()
            c = ctx.ravel()
            for sel, dec in (
                (np.ones(v.shape, dtype=bool), v > 0),
                (v > 0, v > 1),
                (v > 1, v > 2),
            ):
                if sel.any():
                    t, pen = cond_entropy_bits(c[sel], dec[sel].astype(np.int64))
                    total += t if warm_start else pen
            m3 = v > 2
            if m3.any():
                rem = v[m3] - 3
                total += float((2 * np.floor(np.log2(rem + 1)) + 1).sum())
            total += float((v > 0).sum())
            total += 16.0  # the block's own length field
    if warm_start:
        total += nblocks * 3 * NB * 8.0
    return total


def h0_bits(coef):
    v = coef.astype(np.int64).ravel()
    _, counts = np.unique(v, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum() * v.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--qstep", type=float, nargs="*", default=[4.0, 8.0, 16.0])
    ap.add_argument("--dead-zone", type=float, default=0.75)
    ap.add_argument("--crop", type=int, nargs=2, default=[1792, 1024])
    ap.add_argument(
        "--table-bits",
        type=float,
        default=8.0,
        help="bits charged per alphabet symbol per context for the frequency table. The gain is "
        "sensitive to this and rANS already loses to Rice on table cost, so sweep it.",
    )
    args = ap.parse_args()

    from PIL import Image

    im = np.asarray(Image.open(args.png).convert("L"), dtype=np.float64)
    cw, ch = args.crop
    y0 = max(0, (im.shape[0] - ch) // 2)
    x0 = max(0, (im.shape[1] - cw) // 2)
    img = im[y0 : y0 + ch, x0 : x0 + cw]
    px = img.size

    ll, bands = dwt2(img, args.levels)
    g = subband_gains(img.shape, args.levels)
    named = [("LL", ll, "LL")]
    for lv, (lh, hl, hh) in enumerate(bands):
        named += [(f"LH{lv+1}", lh, "LH"), (f"HL{lv+1}", hl, "HL"), (f"HH{lv+1}", hh, "HH")]

    print(f"\n=== EBCOT context-coding ceiling: {os.path.basename(args.png)} "
          f"({args.levels} levels) ===")
    print(f"  {'qstep':>6} {'rice split':>11} {'rice 1-stream':>14} {'H0':>8} "
          f"{'ebcot':>8} {'+KT':>8} {'ebcot vs':>10} {'full-ctx':>10} {'full vs':>10} "
          f"{'vert-ctx':>9} {'vert vs':>9} {'bin-adap':>9} {'bin vs':>8} "
          f"{'per-str':>9} {'cold':>9} {'cb64':>9} {'cb32':>9}")
    for q in args.qstep:
        rs = r = h = e = ep = cc = vc = ba = bs = bc = cb64 = cb32 = 0.0
        for bid, (nm, band, orient) in enumerate(named):
            qq = quantize(band * g[nm], q, args.dead_zone).astype(np.int64)
            rs += rice_zrl_split_bits(qq)
            r += rice_zrl_bits(qq)
            h += h0_bits(qq)
            cc += context_coefficient_bits(qq, args.table_bits)
            vc += vertical_context_bits(qq, args.table_bits)
            ba += binary_adaptive_vertical_bits(qq)
            bs += binary_adaptive_per_stream_bits(qq, warm_start=True)
            bc += binary_adaptive_per_stream_bits(qq, warm_start=False)
            cb64 += codeblock_adaptive_bits(qq, 64)
            cb32 += codeblock_adaptive_bits(qq, 32)
            a, b = band_bits(qq, orient, bid)
            e += a
            ep += b
        globals()["_ctx_coef_bpp"] = cc / px
        print(f"  {q:>6.1f} {rs/px:>11.4f} {r/px:>14.4f} {h/px:>8.4f} "
              f"{e/px:>8.4f} {ep/px:>8.4f} {(ep/rs-1)*100:>+9.1f}% "
              f"{cc/px:>10.4f} {(cc/rs-1)*100:>+9.1f}% {vc/px:>9.4f} {(vc/rs-1)*100:>+8.1f}% "
              f"{ba/px:>9.4f} {(ba/rs-1)*100:>+8.1f}% "
              f"{(bs/rs-1)*100:>+9.1f}% {(bc/rs-1)*100:>+9.1f}% "
              f"{(cb64/rs-1)*100:>+9.1f}% {(cb32/rs-1)*100:>+9.1f}%")
    print()
    print("  'rice split' is what GNC really codes: 256 independent streams per tile.")
    print("  'rice 1-stream' is one stream per subband — an idealised Rice that shares runs and")
    print("  statistics across the whole band, i.e. what GNC gives up for GPU parallelism.")
    print("  'ebcot' is the conditional entropy, a ceiling: the MQ coder's own adaptation loss")
    print("  is excluded. '+KT' adds a 0.5*log2(n)-per-context learning cost and is the fairer")
    print("  bound; both 'vs' columns compare against +KT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
