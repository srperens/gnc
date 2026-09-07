#!/usr/bin/env python3
"""RATE-1 — how much rate above q=90 an 8-bit output cannot emit.

The claim in BACKLOG: on the test gradient q=90 costs 0.275 bpp and q=95 costs 1.142 bpp — four
times the bits for output that is bit-identical at 8 bits. That is not a bug; the anchor ladder
halves qstep and zeroes the dead zone up there by design. But nothing tells it the output is
8-bit, so above some q the extra precision is unrepresentable and the bits are spent for nothing.

**Measured here, per image:** the lowest q whose decoded 8-bit image is *bit-exact* to the 8-bit
original. Every bit spent above that q is strictly unemittable at this output depth. A second,
softer tier is reported too — the lowest q within 1 LSB everywhere — because a change of +-1 in a
handful of pixels is not worth 4x the rate either, and a rate-control rule would plausibly target
that instead.

Deliberately *not* built yet: the rule itself. The item says measure how much is recoverable
across content first, and the gradient is the best case by construction, so a rule justified on it
alone would be justified on nothing.

Scope and honesty notes:
  * q=100 is a **different transform** (LOSSLESS-1 routes it to MED prediction, not the wavelet),
    so it is reported separately and never used as the reference decode for the wavelet ladder.
    The reference is the original PNG.
  * PNG input is 8-bit, so this measures the 8-bit output case only. For a 10-bit target the extra
    precision is real and the ladder should stay — that is why the item asks what is *recoverable*,
    not what is wasted.
  * Rate is the encoded `.gnc` file. Quality figures are computed from the decoded PNG against the
    original PNG, one metric path, per CLAUDE.md.
"""

import argparse, hashlib, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def synth(kind, size=512):
    """Synthetic content. The gradient is the item's own best case; the others bracket it."""
    y, x = np.mgrid[0:size, 0:size]
    if kind == "gradient512":
        v = (x * 255.0 / (size - 1)).astype(np.uint8)
        return np.dstack([v, v, v])
    if kind == "flat512":
        return np.full((size, size, 3), 128, np.uint8)
    if kind == "noise512":
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    if kind == "smoothramp512":            # two-axis ramp: smooth but not axis-aligned
        v = ((x + y) * 255.0 / (2 * size - 2)).astype(np.uint8)
        return np.dstack([v, v, v])
    raise ValueError(kind)


def run_one(gnc, img_path, orig, qs, tmp, label):
    h, w = orig.shape[:2]
    rows = []
    for q in qs:
        bs, out = tmp / f"{label}_q{q}.gnc", tmp / f"{label}_q{q}.png"
        r = sh([gnc, "encode", "-i", str(img_path), "-o", str(bs), "-q", str(q)])
        if r.returncode != 0:
            print(f"    q={q}: encode failed: {r.stderr.strip().splitlines()[-1:]}")
            continue
        r = sh([gnc, "decode", "-i", str(bs), "-o", str(out)])
        if r.returncode != 0:
            print(f"    q={q}: decode failed: {r.stderr.strip().splitlines()[-1:]}")
            continue
        dec = np.array(Image.open(out).convert("RGB"))
        d = np.abs(dec.astype(np.int16) - orig.astype(np.int16))
        size = os.path.getsize(bs)
        rows.append({
            "q": q, "bytes": size, "bpp": size * 8 / (w * h),
            "psnr": psnr(orig, dec), "max_err": int(d.max()),
            "pct_differing": float(np.mean(d > 0) * 100.0),
            "exact": bool(d.max() == 0),
            "within1": bool(d.max() <= 1),
        })
    return rows


def report(name, rows):
    print(f"\n=== {name} ===")
    print(f"  {'q':>4} {'bpp':>8} {'psnr':>9} {'max_err':>8} {'%px≠':>7}  verdict")
    for r in rows:
        v = "BIT-EXACT" if r["exact"] else ("within 1 LSB" if r["within1"] else "")
        p = "inf" if r["psnr"] == float("inf") else f"{r['psnr']:.2f}"
        print(f"  {r['q']:>4} {r['bpp']:>8.4f} {p:>9} {r['max_err']:>8} "
              f"{r['pct_differing']:>7.3f}  {v}")

    wavelet = [r for r in rows if r["q"] < 100]
    if not wavelet:
        return None
    top = wavelet[-1]

    lossless = next((r for r in rows if r["q"] == 100), None)
    if lossless is not None:
        dominated = [r for r in wavelet if r["bpp"] >= lossless["bpp"]]
        if dominated:
            cheapest_dom = min(dominated, key=lambda r: r["bpp"])
            print(f"  DOMINATED: q=100 (MED, bit-exact) costs {lossless['bpp']:.4f} bpp, which is "
                  f"LESS than q={cheapest_dom['q']} ({cheapest_dom['bpp']:.4f} bpp, "
                  f"{cheapest_dom['psnr']:.2f} dB, max err {cheapest_dom['max_err']}). Every "
                  f"wavelet setting from q={cheapest_dom['q']} up spends more bits for worse "
                  f"output than lossless does — {len(dominated)} of {len(wavelet)} rungs.")
        else:
            print(f"  q=100 (MED) costs {lossless['bpp']:.4f} bpp, above every wavelet rung — "
                  f"no rung is dominated")
    out = {"name": name, "top_q": top["q"], "top_bpp": top["bpp"]}
    for tier, key in (("exact", "exact"), ("within1", "within1")):
        # cheapest, not first: the rate ladder is not monotonic in q (flat512 pays more at q=86
        # than at q=99), and a rule would target the cheapest setting that reaches the tier.
        cands = [r for r in wavelet if r[key]]
        hit = min(cands, key=lambda r: r["bpp"]) if cands else None
        if hit is None:
            out[tier] = None
            print(f"  no q below 100 reaches {'bit-exact' if tier == 'exact' else 'within 1 LSB'}"
                  f" — nothing is strictly unemittable in this range")
        else:
            waste = (top["bpp"] - hit["bpp"]) / top["bpp"] * 100.0
            out[tier] = {"q": hit["q"], "bpp": hit["bpp"], "waste_pct": waste}
            print(f"  cheapest {'bit-exact' if tier == 'exact' else 'within 1 LSB'} is q={hit['q']} "
                  f"({hit['bpp']:.4f} bpp) → the rate from there to q={top['q']} "
                  f"({top['bpp']:.4f} bpp) is **{waste:.1f}% of the top rate, unemittable at "
                  f"8 bits**")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnc-binary", default="target/release/gnc")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--synth", nargs="*",
                    default=["gradient512", "smoothramp512", "flat512", "noise512"])
    ap.add_argument("--qualities", default="86,88,90,92,94,95,96,97,98,99,100")
    args = ap.parse_args()
    qs = [int(q) for q in args.qualities.split(",")]
    gnc = args.gnc_binary
    if not os.path.exists(gnc):
        sys.exit(f"no binary at {gnc} — cargo build --release")

    tmp = Path(tempfile.mkdtemp(prefix="rate1_"))
    summaries = []
    for kind in args.synth:
        a = synth(kind)
        p = tmp / f"{kind}.png"
        Image.fromarray(a).save(p)
        summaries.append(report(f"{kind} (synthetic)", run_one(gnc, p, a, qs, tmp, kind)))
    for ip in args.images:
        a = np.array(Image.open(ip).convert("RGB"))
        digest = hashlib.sha256(open(ip, "rb").read()).hexdigest()[:12]
        summaries.append(report(f"{Path(ip).name} (sha {digest})",
                                run_one(gnc, ip, a, qs, tmp, Path(ip).stem)))

    print("\n=== recoverable rate at 8-bit output, by content ===")
    print(f"  {'content':<34} {'bit-exact from':>15} {'unemittable':>12} "
          f"{'within 1 LSB from':>18} {'unemittable':>12}")
    for s in summaries:
        if not s:
            continue
        e, w = s.get("exact"), s.get("within1")
        ef = f"q={e['q']}" if e else "never"
        ep = f"{e['waste_pct']:.1f}%" if e else "-"
        wf = f"q={w['q']}" if w else "never"
        wp = f"{w['waste_pct']:.1f}%" if w else "-"
        print(f"  {s['name']:<34} {ef:>15} {ep:>12} {wf:>18} {wp:>12}")


if __name__ == "__main__":
    main()
