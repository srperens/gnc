#!/usr/bin/env python3
"""BUG-14 gate: Huffman's stream mapping, before against after.

Encodes each image with --huffman at several tile sizes and quality points, decodes it
back, and reports coded bytes plus reconstruction error. Two binaries can be compared in
one run so the "before" and "after" rows come from the same inputs.
"""
import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np
from PIL import Image


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    FAILED: {' '.join(cmd)}\n    {r.stderr.strip()[:400]}", file=sys.stderr)
        return None
    return r.stdout


def psnr_maxerr(ref_png, dec_png):
    a = np.array(Image.open(ref_png).convert("RGB")).astype(np.int32)
    b = np.array(Image.open(dec_png).convert("RGB")).astype(np.int32)
    d = a - b
    mse = float(np.mean(d.astype(np.float64) ** 2))
    p = float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
    return p, int(np.max(np.abs(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action="append", required=True,
                    help="label=path/to/gnc, repeatable")
    ap.add_argument("--images", required=True, help="comma-separated PNG paths")
    ap.add_argument("--tiles", default="256,512")
    ap.add_argument("--quality", default="75,90,100")
    ap.add_argument("--work", required=True)
    ap.add_argument("--cpu-encode", action="store_true",
                    help="also run the host encoder, which must agree byte for byte")
    ap.add_argument("--host-only", action="store_true",
                    help="encode with --cpu-encode only; the host writer has no fixed stream slot")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    bins = [b.split("=", 1) for b in args.bin]
    images = args.images.split(",")
    tiles = [int(t) for t in args.tiles.split(",")]
    qs = [int(q) for q in args.quality.split(",")]

    for img in images:
        with open(img, "rb") as f:
            print(f"{os.path.basename(img)}  md5={hashlib.md5(f.read()).hexdigest()}")

    print(f"\n{'image':22} {'tile':>5} {'q':>4} " +
          " ".join(f"{lbl+' bytes':>15} {lbl+' psnr':>12} {lbl+' max':>5}" for lbl, _ in bins) +
          f"{'  delta':>9}")

    for img in images:
        name = os.path.splitext(os.path.basename(img))[0]
        for tile in tiles:
            for q in qs:
                cells, sizes = [], []
                for lbl, path in bins:
                    stem = os.path.join(args.work, f"{name}_{lbl}_t{tile}_q{q}")
                    enc = ["encode", "-i", img, "-o", stem + ".gnc",
                           "-q", str(q), "-t", str(tile), "--huffman"]
                    if args.host_only:
                        enc.append("--cpu-encode")
                    if sh([path] + enc) is None:
                        cells.append(f"{'enc-fail':>15} {'':>12} {'':>5}")
                        sizes.append(None)
                        continue
                    nbytes = os.path.getsize(stem + ".gnc")
                    if args.cpu_encode:
                        cpu = stem + "_cpu.gnc"
                        if sh([path, "encode", "-i", img, "-o", cpu, "-q", str(q),
                               "-t", str(tile), "--huffman", "--cpu-encode"]) is None:
                            print("    host encoder failed")
                        else:
                            same = open(cpu, "rb").read() == open(stem + ".gnc", "rb").read()
                            print(f"    host-vs-gpu bytes: {os.path.getsize(cpu)} "
                                  f"{'identical' if same else 'DIFFER'}")
                    if sh([path, "decode", "-i", stem + ".gnc", "-o", stem + ".png"]) is None:
                        cells.append(f"{nbytes:15d} {'dec-fail':>12} {'':>5}")
                        sizes.append(nbytes)
                        continue
                    p, m = psnr_maxerr(img, stem + ".png")
                    cells.append(f"{nbytes:15d} {p:12.4f} {m:5d}")
                    sizes.append(nbytes)
                delta = ""
                if len(sizes) == 2 and None not in sizes and sizes[0]:
                    delta = f"{(sizes[1] / sizes[0] - 1) * 100:+8.2f}%"
                print(f"{name:22} {tile:5d} {q:4d} " + " ".join(cells) + delta)


if __name__ == "__main__":
    main()
