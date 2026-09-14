#!/usr/bin/env python3
"""Build the assets for chroma.html — the 4:4:4 / 4:2:2 / 4:2:0 side-by-side.

Encodes one still at three chroma formats and two quality points, decodes each, and emits
zoomed crops plus dE00 heat maps of the same window.

**Why a crop and not the whole frame.** Full-frame dE00 understates this comparison badly: on
bbb at q=95 it reads 0.25 / 0.80 / 1.02, which sounds like a rounding error. The damage is not
spread evenly — it sits on saturated colour edges, and everywhere else there is nothing to lose.
In the window this script picks, the same three encodes read 0.27 / 1.96 / 2.68.

**How the window is picked.** By maximising `dE00(4:2:0) - dE00(4:4:4)` at the same q, so it
lands where *subsampling* costs rather than where *quantisation* does. Those are different
places, and the second one is not what the page is about.

Requires the project venv (numpy, Pillow) and `cargo build --release`.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from chroma_metric import ciede2000, srgb_to_lab  # noqa: E402

GNC = ROOT / "target" / "release" / "gnc"
SRC = ROOT / "test_material" / "frames" / "bbb_1080p.png"
OUT = HERE / "chroma_ab"
FORMATS = ("444", "422", "420")
QUALITIES = (95, 50)
CROP_W, CROP_H, ZOOM, STRIDE = 200, 150, 4, 20


def run(*args):
    subprocess.run([str(a) for a in args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def zoom_save(arr, path):
    """Nearest-neighbour, always. Any smooth resize would destroy the 2x1 and 2x2 chroma blocks
    that are the entire subject of the page."""
    im = Image.fromarray(arr.astype(np.uint8))
    im.resize((arr.shape[1] * ZOOM, arr.shape[0] * ZOOM), Image.NEAREST).save(path)


def heat(d):
    """dE00 -> ramp: 0 black, ~1 blue, ~2.5 magenta, 5+ near-white. Not a perceptual colormap,
    just a legible one; the numbers under each panel are the measurement."""
    stops = np.array([[0, 0, 0], [40, 60, 180], [200, 40, 160], [255, 230, 210]], float)
    pos = np.array([0.0, 0.25, 0.6, 1.0])
    t = np.clip(d / 5.0, 0, 1)
    out = np.zeros(d.shape + (3,))
    for i in range(3):
        m = (t >= pos[i]) & (t <= pos[i + 1])
        f = ((t[m] - pos[i]) / (pos[i + 1] - pos[i]))[:, None]
        out[m] = stops[i] * (1 - f) + stops[i + 1] * f
    return out


def main():
    if not GNC.exists():
        sys.exit("Build first: cargo build --release")
    if not SRC.exists():
        sys.exit(f"No source still at {SRC} — run test_material/fetch_test_frames.sh")
    OUT.mkdir(parents=True, exist_ok=True)

    src = np.array(Image.open(SRC).convert("RGB"))
    lab_src = srgb_to_lab(src)
    dec, de, size = {}, {}, {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for q in QUALITIES:
            for fmt in FORMATS:
                bit, png = tmp / f"{fmt}_{q}.gnc", tmp / f"{fmt}_{q}.png"
                # The RATE-2 fallback would silently hand back a lossless sibling at high q and
                # the row would stop being the arm it claims to be.
                run(GNC, "encode", "-i", SRC, "-o", bit, "-q", q, "--chroma-format", fmt)
                run(GNC, "decode", "-i", bit, "-o", png)
                img = np.array(Image.open(png).convert("RGB"))
                dec[fmt, q], size[fmt, q] = img, bit.stat().st_size
                de[fmt, q] = ciede2000(lab_src, srgb_to_lab(img))

    excess = de["420", QUALITIES[0]] - de["444", QUALITIES[0]]
    ii = np.pad(excess, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = excess.shape
    best = (-np.inf, 0, 0)
    for y in range(0, h - CROP_H, STRIDE):
        for x in range(0, w - CROP_W, STRIDE):
            s = (ii[y + CROP_H, x + CROP_W] - ii[y, x + CROP_W]
                 - ii[y + CROP_H, x] + ii[y, x])
            best = max(best, (s, x, y))
    _, bx, by = best
    print(f"crop ({bx},{by}) {CROP_W}x{CROP_H}")

    zoom_save(src[by:by + CROP_H, bx:bx + CROP_W], OUT / "crop_source.png")
    stats = {"source": SRC.name, "crop": [int(bx), int(by), CROP_W, CROP_H],
             "zoom": ZOOM, "panels": {}}
    for q in QUALITIES:
        for fmt in FORMATS:
            d = de[fmt, q][by:by + CROP_H, bx:bx + CROP_W]
            zoom_save(dec[fmt, q][by:by + CROP_H, bx:bx + CROP_W], OUT / f"crop_{fmt}_{q}.png")
            zoom_save(heat(d), OUT / f"diff_{fmt}_{q}.png")
            stats["panels"][f"{fmt}_{q}"] = {
                "crop_mean": round(float(d.mean()), 3),
                "crop_p95": round(float(np.percentile(d, 95)), 3),
                "full_mean": round(float(de[fmt, q].mean()), 4),
                "bytes": size[fmt, q],
            }
            print(f"  {fmt} q={q}: crop dE00 {d.mean():.3f} "
                  f"(p95 {np.percentile(d, 95):.2f}), {size[fmt, q]} bytes")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
