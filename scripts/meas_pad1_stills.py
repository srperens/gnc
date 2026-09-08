#!/usr/bin/env python3
"""PAD-1 — what the *shipped* encoder delivers on stills, against GNC_PAD_FILL=replicate.

`scripts/meas_intra1_padding.py --part 3` measures an **oracle**: it rebuilds the padding in Python
and encodes it as a tile-aligned picture, which prices a fill without touching the codec. This
measures the codec, which is the figure that belongs in a decision record — and the two agreeing to
two decimals (-4.63% RGB / -4.60% Y) is what says the shader implements what was modelled.

Both arms are the same binary at the same q; only `GNC_PAD_FILL` differs, so quality should not
move (it moves -0.002 to +0.000 dB) and the whole difference is the padding.

Usage:
    scripts/meas_pad1_stills.py [--images ...] [--qualities 80,85,90,94]

Keep every rung at q<=94: since RATE-2 the encoder codes q=95..99 both ways and a rung that comes
back bit-exact lossless has infinite PSNR, which a Bjontegaard fit cannot integrate.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meas9_contribution import measure  # noqa: E402
from meas1_vs_h264 import bd_rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", default=[
        "test_material/frames/bbb_1080p.png",
        "test_material/frames/blue_sky_1080p.png",
        "test_material/frames/kristensara_720p.png",
        "test_material/frames/touchdown_1080p.png"])
    ap.add_argument("--qualities", default="80,85,90,94")
    ap.add_argument("--gnc-binary", default="target/release/gnc")
    ap.add_argument("--extra", default="--abac")
    args = ap.parse_args()

    qs = [int(q) for q in args.qualities.split(",")]
    if max(qs) > 94:
        sys.exit("every rung must stay at q<=94 — see the module docstring")
    extra = args.extra.split() if args.extra else []
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "pad1_stills"
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"  {'image':<20} {'RGB BD-rate':>12} {'Y BD-rate':>11} {'dRGB@mid':>9} {'dbytes@mid':>11}")
    rgbs, ys = [], []
    for src in args.images:
        name = Path(src).stem
        orig = np.array(Image.open(src).convert("RGB"))
        px = orig.shape[0] * orig.shape[1]
        lad = {}
        for fill in ("replicate", "decay"):
            env = dict(os.environ, GNC_PAD_FILL=fill)
            pts = []
            for q in qs:
                bs, png = tmp / f"{name}_{fill}_{q}.gnc", tmp / f"{name}_{fill}_{q}.png"
                if subprocess.run([args.gnc_binary, "encode", "-i", src, "-o", str(bs),
                                   "-q", str(q), *extra],
                                  capture_output=True, env=env).returncode:
                    continue
                if subprocess.run([args.gnc_binary, "decode", "-i", str(bs), "-o", str(png)],
                                  capture_output=True, env=env).returncode:
                    continue
                m = measure(orig, png)
                pts.append((os.path.getsize(bs) * 8.0 / px, m["psnr_rgb"], m["psnr_y"],
                            os.path.getsize(bs)))
            lad[fill] = pts
        r, d = lad["replicate"], lad["decay"]
        if min(len(r), len(d)) < 4:
            print(f"  {name:<20} too few points for a BD-rate")
            continue
        rgb, _ = bd_rate([p[0] for p in r], [p[1] for p in r],
                         [p[0] for p in d], [p[1] for p in d])
        y, _ = bd_rate([p[0] for p in r], [p[2] for p in r],
                       [p[0] for p in d], [p[2] for p in d])
        mid = len(qs) // 2
        print(f"  {name:<20} {rgb:>+11.2f}% {y:>+10.2f}% {d[mid][1] - r[mid][1]:>+9.3f} "
              f"{d[mid][3] / r[mid][3] - 1:>+10.2%}")
        rgbs.append(rgb)
        ys.append(y)
    if rgbs:
        print(f"\n  mean over {len(rgbs)} images: RGB {np.mean(rgbs):+.2f}%, "
              f"Y {np.mean(ys):+.2f}%")
        print("  the oracle in meas_intra1_padding.py --part 3 predicted -4.63% / -4.60%")


if __name__ == "__main__":
    main()
