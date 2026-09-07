#!/usr/bin/env python3
"""MEAS-9 — GNC against the codecs it actually competes with.

GNC is positioned as a *contribution* codec (GOALS §1, docs/POSITIONING.md). Every cross-codec
number in this repo so far is against x264, which POSITIONING itself calls a sanity anchor rather
than a competitor: nobody ships H.264 as a mezzanine format. The incumbents in that segment are
JPEG XS, VC-2, JPEG 2000 and ProRes, and none of them had ever been measured here.

What this harness does, and why each piece is the way it is:

  * **One metric path for every codec.** Each arm produces a decoded 8-bit RGB PNG, and every
    figure is computed from that PNG against the original PNG. The repo has been burned twice by
    comparing a number from one harness with a number from another (`byte_size()` vs the real
    bitstream; GNC's RGB PSNR vs x264's YUV PSNR), so no arm is allowed to report its own quality.
  * **Rate is coded bytes, never file size.** For the ffmpeg arms that is the sum of the video
    packet sizes from ffprobe, which excludes MOV/Matroska container overhead; for GNC it is the
    `.gnc` file; for JPEG 2000 the `.j2k` codestream.
  * **Three quality figures, because one cannot describe this trade.** Y-PSNR in YCoCg-R is the
    plane GNC codes and leads above q=85 (CLAUDE.md's metric table); RGB PSNR is the cross-check;
    CIEDE2000 is the only figure that sees colour at all, and the 4:2:2 arms exist to be judged on
    it. VMAF is deliberately absent: it is luma-only and saturated at this operating point, where
    widening a ladder moved it 47.5 points on average (QUAL-1).
  * **A conversion ceiling per pixel format, measured, not assumed.** Each ffmpeg arm's pixel
    format is round-tripped through lossless FFV1 first. `rgb24 -> yuv444p10le -> rgb24` is exact
    (PSNR inf), so the 4:4:4 arms are clean; `yuv422p10le` caps at ~39 dB on RGB PSNR, which is
    *chroma subsampling alone* and is larger than any coding difference measured here. An arm can
    never be scored better than its own ceiling, and quoting a 4:2:2 codec's RGB PSNR against a
    4:4:4 codec measures the format, not the codec.

Not measured: **JPEG XS**, the one the positioning most wants. No implementation is installable on
this machine — SVT-JPEG-XS ships Linux/Windows build trees and x86 assembly, and libjxs is not in
the package manager. VC-2 stands in as the nearest available relative: intra-only, low-latency,
broadcast contribution, and a 9/7 wavelet like GNC's own.

Usage:
    scripts/meas9_contribution.py --images test_material/frames/bbb_1080p.png ... \
        [--gnc-binary target/release/gnc] [--csv out.csv] [--arms gnc,prores444,vc2,j2k,prores422]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chroma_metric import ciede2000, srgb_to_lab  # noqa: E402
from meas1_vs_h264 import bd_rate  # noqa: E402  (same Bjontegaard code as the H.264 comparison)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def repo_root():
    out = sh(["git", "rev-parse", "--show-toplevel"])
    return Path(out.stdout.strip()) if out.returncode == 0 else Path.cwd()


# ---------------------------------------------------------------------------
# metrics — one implementation, applied to every arm
# ---------------------------------------------------------------------------

def ycocg_r_y(rgb):
    """Y of the integer-reversible YCoCg-R: the plane GNC actually codes."""
    R, G, B = (rgb[:, :, i].astype(np.int32) for i in range(3))
    Co = R - B
    t = B + (Co >> 1)
    Cg = G - t
    return (t + (Cg >> 1)).astype(np.float64)


def psnr(a, b, peak=255.0):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak**2 / mse)


def measure(orig_rgb, decoded_png):
    """Every quality figure for one decoded image, from one place."""
    dec = np.array(Image.open(decoded_png).convert("RGB"))
    if dec.shape != orig_rgb.shape:
        raise RuntimeError(f"{decoded_png}: {dec.shape} != original {orig_rgb.shape}")
    d = ciede2000(srgb_to_lab(orig_rgb), srgb_to_lab(dec))
    return {
        "psnr_rgb": psnr(orig_rgb.astype(np.float64), dec.astype(np.float64)),
        "psnr_y": psnr(ycocg_r_y(orig_rgb), ycocg_r_y(dec)),
        "de00_mean": float(np.mean(d)),
        "de00_p95": float(np.percentile(d, 95)),
    }


# ---------------------------------------------------------------------------
# ffmpeg plumbing
# ---------------------------------------------------------------------------

def coded_bytes(path):
    """Sum of video packet sizes — the codestream, without container overhead."""
    out = sh(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "packet=size", "-of", "csv=p=0", str(path)])
    sizes = [int(s) for s in out.stdout.split() if s.strip().isdigit()]
    return sum(sizes) if sizes else None


def ffmpeg_encode(src_png, dst, pix_fmt, encoder, extra, container):
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-framerate", "1", "-i", str(src_png),
           "-pix_fmt", pix_fmt, "-c:v", encoder, *extra, "-frames:v", "1",
           "-f", container, str(dst)]
    r = sh(cmd)
    return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0


def ffmpeg_decode_png(src, dst_png):
    r = sh(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src),
            "-pix_fmt", "rgb24", "-frames:v", "1", str(dst_png)])
    return r.returncode == 0 and os.path.exists(dst_png)


def conversion_ceiling(src_png, pix_fmt, orig_rgb, tmp):
    """Quality of a lossless round trip through `pix_fmt`. An arm cannot beat this."""
    mkv, png = tmp / f"ceil_{pix_fmt}.mkv", tmp / f"ceil_{pix_fmt}.png"
    if not ffmpeg_encode(src_png, mkv, pix_fmt, "ffv1", [], "matroska"):
        return None
    if not ffmpeg_decode_png(mkv, png):
        return None
    return measure(orig_rgb, png)


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------

def arm_gnc(gnc_binary, src_png, orig_rgb, tmp, qualities):
    rows = []
    for q in qualities:
        bs, out_png = tmp / f"gnc_q{q}.gnc", tmp / f"gnc_q{q}.png"
        r = sh([str(gnc_binary), "encode", "-i", str(src_png), "-o", str(bs), "-q", str(q)])
        if r.returncode != 0:
            print(f"    GNC q={q} encode failed: {r.stderr.strip().splitlines()[-1:]}")
            continue
        r = sh([str(gnc_binary), "decode", "-i", str(bs), "-o", str(out_png)])
        if r.returncode != 0:
            print(f"    GNC q={q} decode failed: {r.stderr.strip().splitlines()[-1:]}")
            continue
        rows.append(("GNC", f"q{q}", os.path.getsize(bs), measure(orig_rgb, out_png)))
    return rows


def arm_ffmpeg(name, encoder, pix_fmt, container, rungs, src_png, orig_rgb, tmp):
    rows = []
    for label, extra in rungs:
        stem = f"{name.replace(' ', '_')}_{label}"
        enc, out_png = tmp / f"{stem}.{container}", tmp / f"{stem}.png"
        if not ffmpeg_encode(src_png, enc, pix_fmt, encoder, extra, container):
            print(f"    {name} {label}: encode failed")
            continue
        size = coded_bytes(enc)
        if not size or not ffmpeg_decode_png(enc, out_png):
            print(f"    {name} {label}: no packets or decode failed")
            continue
        rows.append((name, label, size, measure(orig_rgb, out_png)))
    return rows


def arm_j2k(src_png, orig_rgb, tmp, rates):
    if shutil.which("opj_compress") is None:
        print("    JPEG 2000: opj_compress not in PATH, skipped")
        return []
    rows = []
    for rate in rates:
        j2k, out_png = tmp / f"j2k_r{rate}.j2k", tmp / f"j2k_r{rate}.png"
        r = sh(["opj_compress", "-i", str(src_png), "-o", str(j2k), "-r", str(rate)])
        if r.returncode != 0 or not os.path.exists(j2k):
            print(f"    JPEG 2000 r={rate}: encode failed")
            continue
        r = sh(["opj_decompress", "-i", str(j2k), "-o", str(out_png)])
        if r.returncode != 0 or not os.path.exists(out_png):
            print(f"    JPEG 2000 r={rate}: decode failed")
            continue
        rows.append(("JPEG 2000", f"r{rate}", os.path.getsize(j2k), measure(orig_rgb, out_png)))
    return rows


def prores444_rungs():
    """ProRes 4444/4444XQ. `-qscale:v` is the real rate knob; the named profiles are two points
    on it. bits_per_mb reaches the top of the contribution range that qscale 1 cannot."""
    return [(f"4444_q{q}", ["-profile:v", "4444", "-qscale:v", str(q)])
            for q in (16, 12, 8, 6, 4, 2)] + [
        ("4444xq", ["-profile:v", "4444xq"]),
        ("4444_bpm5000", ["-profile:v", "4444", "-bits_per_mb", "5000"]),
    ]


def prores422_rungs():
    """The profiles a broadcast engineer actually picks. 4:2:2, so judge these on dE00 and on
    Y-PSNR, never on RGB PSNR — their subsampling ceiling is below any coding difference here."""
    return [(p, ["-profile:v", p]) for p in ("proxy", "lt", "standard", "hq")]


def vc2_rungs(w, h, qm="default"):
    """VC-2 is rate-driven: low-delay assigns fixed bytes per slice. `-framerate 1` on the input
    makes `-b:v` bits per frame, so the ladder is written directly in bpp. slice_height 8 divides
    1080 and 720; the default 16 does not divide 1080.

    `qm` selects the quantisation matrix. The spec default is perceptually weighted, which costs
    it PSNR against a codec measured on PSNR; `flat` is the encoder's own "optimize for PSNR"
    setting. Measure both before quoting a VC-2 number — scoring a competitor through a matrix
    tuned for a different metric is the mirror image of the VMAF/chroma error this repo keeps
    making against itself."""
    return [(f"{bpp}bpp", ["-slice_height", "8", "-qm", qm, "-b:v", str(int(bpp * w * h))])
            for bpp in (1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 10.0)]


# ---------------------------------------------------------------------------

ARM_ORDER = ["gnc", "prores444", "vc2", "j2k", "prores422"]


def run_image(img_path, gnc_binary, arms, qualities, tmp, vc2_qm="default"):
    orig = np.array(Image.open(img_path).convert("RGB"))
    h, w = orig.shape[:2]
    print(f"\n=== {Path(img_path).name}  {w}x{h} ===")

    ceilings = {}
    for pix_fmt in ("yuv444p10le", "yuv422p10le"):
        c = conversion_ceiling(img_path, pix_fmt, orig, tmp)
        ceilings[pix_fmt] = c
        if c:
            print(f"  ceiling {pix_fmt:14s} rgb {c['psnr_rgb']:7.2f} dB   "
                  f"Y {c['psnr_y']:7.2f} dB   dE00 {c['de00_mean']:.4f}")

    rows = []
    if "gnc" in arms:
        rows += arm_gnc(gnc_binary, img_path, orig, tmp, qualities)
    if "prores444" in arms:
        rows += arm_ffmpeg("ProRes 4444", "prores_ks", "yuv444p10le", "mov",
                           prores444_rungs(), img_path, orig, tmp)
    if "vc2" in arms:
        rows += arm_ffmpeg(f"VC-2 {vc2_qm}", "vc2", "yuv444p10le", "matroska",
                           vc2_rungs(w, h, vc2_qm), img_path, orig, tmp)
    if "j2k" in arms:
        rows += arm_j2k(img_path, orig, tmp, (40, 20, 12, 8, 5, 3, 2))
    if "prores422" in arms:
        rows += arm_ffmpeg("ProRes 422", "prores_ks", "yuv422p10le", "mov",
                           prores422_rungs(), img_path, orig, tmp)

    out = []
    for codec, label, size, m in rows:
        out.append({
            "image": Path(img_path).name, "codec": codec, "rung": label,
            "bytes": size, "bpp": size * 8 / (w * h), **m,
        })
    return out, ceilings


def print_table(rows):
    print(f"  {'codec':<12} {'rung':<14} {'bpp':>7} {'psnr_rgb':>9} {'psnr_Y':>9} "
          f"{'dE00':>7} {'dE00p95':>8}")
    for r in sorted(rows, key=lambda r: (r["codec"], r["bpp"])):
        print(f"  {r['codec']:<12} {r['rung']:<14} {r['bpp']:>7.3f} {r['psnr_rgb']:>9.2f} "
              f"{r['psnr_y']:>9.2f} {r['de00_mean']:>7.4f} {r['de00_p95']:>8.4f}")


def bd_summary(rows, metric):
    """BD-rate of GNC against each other arm, on `metric`. Negative = GNC needs fewer bits."""
    by_codec = {}
    for r in rows:
        if np.isfinite(r[metric]):
            by_codec.setdefault(r["codec"], []).append((r["bpp"], r[metric]))
    if "GNC" not in by_codec:
        return []
    g = sorted(by_codec["GNC"])
    out = []
    for codec, pts in sorted(by_codec.items()):
        if codec == "GNC" or len(pts) < 4 or len(g) < 4:
            continue
        p = sorted(pts)
        bd, (lo, hi) = bd_rate([x[0] for x in p], [x[1] for x in p],
                               [x[0] for x in g], [x[1] for x in g])
        out.append((codec, bd, lo, hi))
    return out


def main():
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--gnc-binary", default=str(root / "target/release/gnc"))
    ap.add_argument("--qualities", default="60,75,85,90,95,99",
                    help="GNC quality ladder (default spans the contribution operating point)")
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    ap.add_argument("--vc2-qm", default="default", choices=("default", "color", "flat"),
                    help="VC-2 quantisation matrix; 'flat' is its own optimise-for-PSNR setting")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the decoded PNGs for inspection")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARM_ORDER)
    if unknown:
        sys.exit(f"unknown arm(s): {', '.join(sorted(unknown))}; known: {', '.join(ARM_ORDER)}")
    qualities = [int(q) for q in args.qualities.split(",")]
    if "gnc" in arms and not os.path.exists(args.gnc_binary):
        sys.exit(f"GNC binary not found: {args.gnc_binary} (cargo build --release)")

    all_rows = []
    tmpdir = tempfile.mkdtemp(prefix="meas9_")
    try:
        for img in args.images:
            rows, _ = run_image(img, args.gnc_binary, arms, qualities, Path(tmpdir),
                                vc2_qm=args.vc2_qm)
            print_table(rows)
            for metric, name in (("psnr_y", "Y-PSNR (YCoCg-R)"), ("psnr_rgb", "RGB PSNR")):
                bds = bd_summary(rows, metric)
                if bds:
                    print(f"  BD-rate on {name}, GNC vs:")
                    for codec, bd, lo, hi in bds:
                        if bd is None:
                            print(f"    {codec:<12} n/a (no overlap)")
                        else:
                            verdict = "fewer" if bd < 0 else "more"
                            print(f"    {codec:<12} {bd:+8.1f}%  "
                                  f"(GNC needs {abs(bd):.1f}% {verdict} bits, "
                                  f"overlap {lo:.1f}-{hi:.1f} dB)")
            all_rows += rows

        if len(args.images) > 1:
            print("\n=== mean BD-rate across images ===")
            for metric, name in (("psnr_y", "Y-PSNR (YCoCg-R)"), ("psnr_rgb", "RGB PSNR")):
                per_codec = {}
                for img in args.images:
                    rows = [r for r in all_rows if r["image"] == Path(img).name]
                    for codec, bd, _, _ in bd_summary(rows, metric):
                        if bd is not None:
                            per_codec.setdefault(codec, []).append(bd)
                print(f"  on {name}:")
                for codec, vals in sorted(per_codec.items()):
                    print(f"    {codec:<12} {np.mean(vals):+8.1f}%   "
                          f"({', '.join(f'{v:+.1f}' for v in vals)})")

        if args.csv:
            import csv as csvmod
            with open(args.csv, "w", newline="") as f:
                wtr = csvmod.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                wtr.writeheader()
                wtr.writerows(all_rows)
            print(f"\nCSV written to {args.csv}")
    finally:
        if args.keep:
            print(f"decoded images kept in {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
