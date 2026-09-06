#!/usr/bin/env python3
"""LOSSLESS-1 offline gate.

GNC is 27% behind FFV1 and 43% behind x264 -qp 0 at q=100. Both win the same way: a per-pixel
median predictor (MED / LOCO-I) whose error is entropy-coded directly, with no transform. BUG-13
showed that GNC's block prediction *before* the wavelet costs 4-8%; that says nothing about
prediction *instead of* it.

This measures the mechanism itself, offline: MED predict in GNC's own colour space, then take the
zeroth-order entropy of the residual. Calibrated against FFV1's real output on the same image, so
the model's optimism is visible rather than assumed.
"""
import os, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

# Derived, never typed: scripts/ sits one level below the repo root.
# GNC_FRAMES overrides it if the test material lives elsewhere.
FRAMES = os.environ.get(
    "GNC_FRAMES", str(Path(__file__).resolve().parents[1] / "test_material" / "frames")
)

def ycocg_r(rgb):
    """Integer-reversible YCoCg-R, the same transform GNC uses."""
    R, G, B = (rgb[:, :, i].astype(np.int32) for i in range(3))
    Co = R - B
    t = B + (Co >> 1)
    Cg = G - t
    Y = t + (Cg >> 1)
    return Y, Co, Cg

def med_predict(p):
    """JPEG-LS / LOCO-I median predictor. a=left, b=above, c=upper-left."""
    a = np.zeros_like(p); a[:, 1:] = p[:, :-1]
    b = np.zeros_like(p); b[1:, :] = p[:-1, :]
    c = np.zeros_like(p); c[1:, 1:] = p[:-1, :-1]
    mx, mn = np.maximum(a, b), np.minimum(a, b)
    pred = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    pred[0, :] = a[0, :]          # first row: left only
    pred[:, 0] = b[:, 0]          # first col: above only
    pred[0, 0] = 0
    return p - pred

def entropy_bits(x):
    v, cnt = np.unique(x, return_counts=True)
    pr = cnt / cnt.sum()
    return float(-(pr * np.log2(pr)).sum() * x.size)

def gnc_lossless(path, gnc):
    out = "/tmp/_ll.gnc"
    subprocess.run([gnc, "encode", "-i", path, "-o", out, "-q", "100", "--rans"],
                   capture_output=True)
    n = os.path.getsize(out); os.remove(out); return n

def ffv1_size(path):
    out = "/tmp/_ll.mkv"
    subprocess.run(["ffmpeg", "-y", "-i", path, "-c:v", "ffv1", "-level", "3", out],
                   capture_output=True)
    n = os.path.getsize(out); os.remove(out); return n

gnc_bin = sys.argv[1] if len(sys.argv) > 1 else "./target/release/gnc"
print(f"{'image':18s} {'GNC q=100':>11s} {'FFV1 real':>11s} {'MED+H0 (model)':>15s} "
      f"{'model vs GNC':>13s} {'model vs FFV1':>14s}")
for name in ("touchdown_1080p", "bbb_1080p", "blue_sky_1080p", "kristensara_720p"):
    p = os.path.join(FRAMES, f"{name}.png")
    if not os.path.exists(p):
        continue
    rgb = np.asarray(Image.open(p).convert("RGB"))
    bits = sum(entropy_bits(med_predict(pl)) for pl in ycocg_r(rgb))
    model = int(bits / 8)
    g, f = gnc_lossless(p, gnc_bin), ffv1_size(p)
    print(f"{name:18s} {g:>11,} {f:>11,} {model:>15,} "
          f"{(model/g-1)*100:>+12.1f}% {(model/f-1)*100:>+13.1f}%")
