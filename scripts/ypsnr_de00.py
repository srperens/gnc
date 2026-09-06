#!/usr/bin/env python3
"""Luma PSNR and CIEDE2000 between two PNGs, with luma taken in the codec's own colour space.

CHROMA-1 needs luma and colour separated, and `gnc benchmark` reports PSNR over RGB, which mixes
exactly the two things under test.

**Why YCoCg-R and not BT.709 Y.** A luma computed from decoded *RGB* is contaminated by chroma
error: perturb only Co/Cg and the reconstructed RGB moves, so BT.709 Y moves with it. Measured on
kristensara at q=92, chroma_weight 1.0 -> 3.0 showed a 0.56 dB "luma" drop that way, which is
mostly leaked chroma. GNC codes YCoCg-R, so its Y plane is the channel that actually carries the
luma bits; the integer-reversible forward transform recovers it exactly from RGB. That is the
primary figure here. BT.709 Y is reported alongside as the contaminated cross-check, because it is
what a YUV-based harness would show.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chroma_metric import ciede2000, srgb_to_lab  # noqa: E402


def ycocg_r_y(rgb):
    """Y of the integer-reversible YCoCg-R, the plane GNC actually codes."""
    R, G, B = (rgb[:, :, i].astype(np.int32) for i in range(3))
    Co = R - B
    t = B + (Co >> 1)
    Cg = G - t
    return (t + (Cg >> 1)).astype(np.float64)


def y709(rgb):
    r, g, b = (rgb[:, :, i].astype(np.float64) for i in range(3))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def psnr(a, b, peak=255.0):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak ** 2 / mse)


def main():
    a = np.array(Image.open(sys.argv[1]).convert("RGB"))
    b = np.array(Image.open(sys.argv[2]).convert("RGB"))
    d = ciede2000(srgb_to_lab(a), srgb_to_lab(b))
    print(f"{psnr(ycocg_r_y(a), ycocg_r_y(b)):.4f} "
          f"{psnr(y709(a), y709(b)):.4f} "
          f"{float(np.mean(d)):.4f} {float(np.percentile(d, 95)):.4f}")


if __name__ == "__main__":
    main()
