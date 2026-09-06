#!/usr/bin/env python3
"""MEAS-1 — like-for-like video comparison of GNC against H.264, scored with VMAF.

The comparison this replaces was not valid: GNC's benchmark reports PSNR computed in RGB while
x264 reports it in YUV, which are different quantities, so neither the bitrates nor the savings
could be compared. Everything here is measured the same way for both codecs:

  * one reference file, the source Y4M, used for every VMAF call;
  * each codec encodes that same source, is decoded back to Y4M, and is scored by the same
    `vmaf` binary with the same arguments;
  * rate is the actual coded bitstream size, in bits per luma pixel.

The colour path has to be identical for all three files or the comparison measures the conversion
rather than the codecs. A first version used the source Y4M directly as the VMAF reference while
GNC's output came back through PNG; GNC's own VMAF read 95 where the harness read 74, purely from
that mismatch. So everything is normalised through PNG first:

    source Y4M --ffmpeg--> reference PNGs --ffmpeg--> reference Y4M   (the one VMAF reference)
    reference PNGs --GNC--> decoded PNGs --ffmpeg--> distorted Y4M
    reference Y4M  --x264--> bitstream   --ffmpeg--> distorted Y4M

Both codecs encode the same normalised source and both distorted files are produced by the same
ffmpeg invocation as the reference.

Output: a CSV per codec plus a BD-rate summary computed on VMAF.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np


# Single VMAF reference and pixel format, set once in main().
#
# 4:2:0 puts GNC at a disadvantage the codec does not deserve: its only sequence output is PNG,
# so its chroma is subsampled twice (once inside the codec, once converting its PNGs back to
# Y4M) while x264's single subsampling matches the reference exactly. Comparing in 4:4:4 removes
# that entirely and measures the coding, not the conversion.
REFERENCE = [None]
PIX_FMT = ["yuv420p"]
# Bit depth of the whole comparison. 10-bit needs `-strict -1` on every ffmpeg *output* (the
# 10-bit Y4M colourspaces are not "official"), x264's --input-depth/--output-depth/--profile,
# and GNC's --bit-depth. vmaf scores 10-bit Y4M directly.
DEPTH = [8]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)


def y4m_geometry(path):
    with open(path, "rb") as f:
        header = f.readline().decode("ascii", "replace")
    w = int(re.search(r"\bW(\d+)", header).group(1))
    h = int(re.search(r"\bH(\d+)", header).group(1))
    return w, h


def vmaf_score(ref, dist):
    out = os.path.join(tempfile.gettempdir(), f"vmaf_{os.getpid()}.json")
    r = sh(["vmaf", "--reference", ref, "--distorted", dist, "--json", "--output", out,
            "--quiet", "--feature", "psnr"])
    if r.returncode != 0 or not os.path.exists(out):
        print(f"    vmaf failed: {r.stderr.strip()[:200]}", file=sys.stderr)
        return None
    with open(out) as f:
        j = json.load(f)
    os.remove(out)
    pooled = j["pooled_metrics"]
    psnr = None
    for k in ("psnr_y", "psnr"):
        if k in pooled:
            psnr = pooled[k]["mean"]
            break
    return j["pooled_metrics"]["vmaf"]["mean"], psnr


def bd_rate(rate_a, q_a, rate_b, q_b):
    """BD-rate of B relative to A, in percent. Negative = B is more efficient.

    Standard Bjontegaard: cubic fit of log10(rate) against quality, integrated over the
    overlapping quality range.
    """
    la, lb = np.log10(rate_a), np.log10(rate_b)
    pa = np.polyfit(q_a, la, min(3, len(q_a) - 1))
    pb = np.polyfit(q_b, lb, min(3, len(q_b) - 1))
    lo = max(min(q_a), min(q_b))
    hi = min(max(q_a), max(q_b))
    if hi <= lo:
        return None, (lo, hi)
    ia = np.polyval(np.polyint(pa), hi) - np.polyval(np.polyint(pa), lo)
    ib = np.polyval(np.polyint(pb), hi) - np.polyval(np.polyint(pb), lo)
    return (10 ** ((ib - ia) / (hi - lo)) - 1) * 100, (lo, hi)


def strictly(args):
    """ffmpeg refuses the 10-bit Y4M colourspaces without -strict -1 on the output."""
    return args + (["-strict", "-1"] if DEPTH[0] == 10 else [])


def png_fmt():
    return "rgb48le" if DEPTH[0] == 10 else "rgb24"


def normalise_source(src, work, n, pix_fmt="yuv420p"):
    """Decode the source once to PNG and back, giving one canonical reference for everything."""
    pngdir = os.path.join(work, "ref_png")
    os.makedirs(pngdir, exist_ok=True)
    if not os.path.exists(os.path.join(pngdir, "0000.png")):
        sh(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", src,
            "-frames:v", str(n), "-start_number", "0", "-pix_fmt", png_fmt(),
            os.path.join(pngdir, "%04d.png")])
    ref_y4m = os.path.join(work, f"reference_{pix_fmt}.y4m")
    if not os.path.exists(ref_y4m):
        sh(strictly(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-start_number", "0",
                     "-i", os.path.join(pngdir, "%04d.png"), "-pix_fmt", pix_fmt])
           + ["-f", "yuv4mpegpipe", ref_y4m])
    return pngdir, ref_y4m


def run_gnc(src, work, n, ki, q, chroma, gnc_bin):
    tag = f"gnc_q{q}"
    gnv = os.path.join(work, f"{tag}.gnv2")
    r = sh([gnc_bin, "benchmark-sequence", "-i", src, "-n", str(n), "-k", str(ki),
            "-q", str(q), "--chroma-format", chroma, "-o", gnv]
           + (["--bit-depth", "10"] if DEPTH[0] == 10 else []),
           env={**os.environ, "GNC_REF_DEBLOCK": "0"})
    if not os.path.exists(gnv):
        print(f"    gnc encode failed at q={q}: {r.stderr.strip()[:200]}", file=sys.stderr)
        return None
    size = os.path.getsize(gnv)

    pngdir = os.path.join(work, tag)
    os.makedirs(pngdir, exist_ok=True)
    sh([gnc_bin, "decode-sequence", "-i", gnv, "-o", os.path.join(pngdir, "%04d.png")])
    dist = os.path.join(work, f"{tag}.y4m")
    sh(strictly(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i",
                 os.path.join(pngdir, "%04d.png"), "-pix_fmt", PIX_FMT[0]])
       + ["-f", "yuv4mpegpipe", dist])
    shutil.rmtree(pngdir, ignore_errors=True)
    if not os.path.exists(dist):
        return None
    v = vmaf_score(REFERENCE[0], dist)
    os.remove(dist)
    os.remove(gnv)
    return size, v


def run_x264(src, work, n, ki, crf, extra):
    tag = f"x264_crf{crf}"
    bs = os.path.join(work, f"{tag}.264")
    csp = ["--output-csp", "i444"] if PIX_FMT[0].startswith("yuv444") else []
    if DEPTH[0] == 10:
        csp += ["--input-depth", "10", "--output-depth", "10",
                "--profile", "high444" if PIX_FMT[0].startswith("yuv444") else "high10"]
    sh(["x264", "--crf", str(crf), "--frames", str(n), "--keyint", str(ki),
        "--tune", "psnr", *csp, *extra, "-o", bs, src])
    if not os.path.exists(bs):
        return None
    size = os.path.getsize(bs)
    dist = os.path.join(work, f"{tag}.y4m")
    sh(strictly(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", bs,
                 "-pix_fmt", PIX_FMT[0]]) + ["-f", "yuv4mpegpipe", dist])
    if not os.path.exists(dist):
        return None
    v = vmaf_score(REFERENCE[0], dist)
    os.remove(dist)
    os.remove(bs)
    return size, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--keyint", type=int, default=9)
    ap.add_argument("--chroma", default="420", help="420 or 444; sets both codecs and the reference")
    ap.add_argument("--depth", type=int, default=8, choices=(8, 10),
                    help="bit depth of the whole comparison; 10 needs 10-bit source material")
    ap.add_argument("--q", default="25,40,55,70,85")
    ap.add_argument("--crf", default="18,23,28,33,38")
    ap.add_argument("--gnc", default="./target/release/gnc")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    w, h = y4m_geometry(args.src)
    px = w * h * args.frames
    work = args.workdir or tempfile.mkdtemp(prefix="meas1_")
    os.makedirs(work, exist_ok=True)
    DEPTH[0] = args.depth
    base = "yuv444p" if args.chroma == "444" else "yuv420p"
    PIX_FMT[0] = base + ("10le" if args.depth == 10 else "")
    ref_pngs, ref_y4m = normalise_source(args.src, work, args.frames, PIX_FMT[0])
    REFERENCE[0] = ref_y4m
    gnc_input = os.path.join(ref_pngs, "%04d.png")

    name = args.label or os.path.basename(args.src)
    print(f"\n=== MEAS-1 {name} — {w}x{h}, {args.frames} frames, ki={args.keyint}, "
          f"chroma {args.chroma}, {args.depth}-bit ===")

    rows = []
    print(f"\n  {'codec':6} {'setting':>8} {'bpp':>9} {'VMAF':>8} {'PSNR-Y':>8}")
    gr, gq, gp = [], [], []
    for q in [int(x) for x in args.q.split(",")]:
        res = run_gnc(gnc_input, work, args.frames, args.keyint, q, args.chroma, args.gnc)
        if not res or res[1] is None:
            continue
        size, (v, ps) = res
        bpp = size * 8 / px
        gr.append(bpp)
        gq.append(v)
        rows.append(("gnc", q, bpp, v, ps))
        gp.append(ps)
        print(f"  {'gnc':6} {q:8} {bpp:9.4f} {v:8.2f} {ps if ps is None else f'{ps:8.2f}'}")

    xr, xq, xp = [], [], []
    for crf in [int(x) for x in args.crf.split(",")]:
        res = run_x264(ref_y4m, work, args.frames, args.keyint, crf, [])
        if not res or res[1] is None:
            continue
        size, (v, ps) = res
        bpp = size * 8 / px
        xr.append(bpp)
        xq.append(v)
        rows.append(("x264", crf, bpp, v, ps))
        xp.append(ps)
        print(f"  {'x264':6} {crf:8} {bpp:9.4f} {v:8.2f} {ps if ps is None else f'{ps:8.2f}'}")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("codec,setting,bpp,vmaf,psnr\n")
            for c, st, b, v, ps in rows:
                f.write(f"{c},{st},{b:.6f},{v:.4f},{'' if ps is None else f'{ps:.4f}'}\n")

    if len(gq) >= 2 and len(xq) >= 2:
        # BD-rate of GNC relative to x264: positive means GNC needs more bits.
        bd, (lo, hi) = bd_rate(np.array(xr), np.array(xq), np.array(gr), np.array(gq))
        if bd is None:
            print(f"\n  no overlapping VMAF range ({lo:.1f}..{hi:.1f}) — widen the sweeps")
        else:
            print(f"\n  BD-rate GNC vs H.264 (VMAF {lo:.1f}-{hi:.1f}): {bd:+.1f}%")
            print("  (positive = GNC needs more bits for the same VMAF)")
        if all(p is not None for p in gp + xp) and len(gp) >= 2 and len(xp) >= 2:
            bdp, (plo, phi) = bd_rate(np.array(xr), np.array(xp), np.array(gr), np.array(gp))
            if bdp is not None:
                print(f"  BD-rate on PSNR-Y   ({plo:.1f}-{phi:.1f} dB): {bdp:+.1f}%")
                print("  (the repo's +13.9% figure is PSNR-based; VMAF is the stated primary metric)")
    print()


if __name__ == "__main__":
    main()
