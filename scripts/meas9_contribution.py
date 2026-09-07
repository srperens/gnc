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
from meas1_vs_h264 import bd_rate as bd_rate_raw  # noqa: E402  (the H.264 comparison's Bjontegaard)


def sh(cmd, env=None):
    """`env` adds to the inherited environment rather than replacing it — a replaced environment
    loses PATH and every arm that shells out to a tool stops working."""
    e = None
    if env:
        e = dict(os.environ)
        e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(cmd, capture_output=True, text=True, env=e)


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

def arm_gnc(gnc_binary, src_png, orig_rgb, tmp, qualities, extra=(), name="GNC", env=None):
    """The GNC arm. `extra` selects a non-default coder — `["--abac"]` for the code-block
    arithmetic coder shipped 2026-09-07.

    abac re-codes the *same quantised coefficients* losslessly, so an abac arm must reproduce the
    Rice arm's pixels exactly at every q. That is checked in `entropy_identity_check`, and it is
    checked before any rate figure is believed: if quality moves at all, the arm is measuring
    something other than the entropy coder and the rate saving is not a rate saving.
    """
    rows = []
    tag = name.replace(" ", "_")
    for q in qualities:
        bs, out_png = tmp / f"{tag}_q{q}.gnc", tmp / f"{tag}_q{q}.png"
        r = sh([str(gnc_binary), "encode", "-i", str(src_png), "-o", str(bs), "-q", str(q),
                *extra], env=env)
        if r.returncode != 0:
            print(f"    {name} q={q} encode failed: {r.stderr.strip().splitlines()[-1:]}")
            continue
        r = sh([str(gnc_binary), "decode", "-i", str(bs), "-o", str(out_png)])
        if r.returncode != 0:
            print(f"    {name} q={q} decode failed: {r.stderr.strip().splitlines()[-1:]}")
            continue
        rows.append((name, f"q{q}", os.path.getsize(bs), measure(orig_rgb, out_png)))

    # The ffmpeg arms are checked for honouring their requested rate; this is the same guard for
    # the one arm whose rate is requested indirectly. GNC's ladder is not guaranteed monotonic in
    # rate: RATE-1 measured flat512 at 0.0450 bpp for q=86 and 0.0370 for q=90. Interpolating
    # through an inversion is silent, so say it out loud.
    for a, b in zip(rows, rows[1:]):
        if b[2] < a[2]:
            print(f"    {name}: rate falls as quality rises, {a[1]} -> {b[1]} "
                  f"({a[2]} -> {b[2]} bytes) — the ladder is not monotonic here")
    return rows


def requested_bpp(extra, pixels):
    """The bpp a rate-driven rung asked for, from its own `-b:v`. None if it is quality-driven."""
    for i, a in enumerate(extra):
        if a == "-b:v" and i + 1 < len(extra):
            return int(extra[i + 1]) / pixels
    return None


def arm_ffmpeg(name, encoder, pix_fmt, container, rungs, src_png, orig_rgb, tmp, pixels,
               rate_tolerance=0.15):
    """One ffmpeg arm. A rate-driven rung whose achieved bpp misses its request is DROPPED.

    VC-2 low-delay assigns a fixed byte count per slice, and below a floor it cannot honour the
    request at all: on kristensara_720p the 1.5, 2.5 and 3.5 bpp rungs all emitted the same
    3.438 bpp and decoded to **10.3 dB / dE00 28** — visibly destroyed output. Scored naively that
    reads as GNC winning by +36.7 dB, which is not a coding result, it is a broken encoder
    configuration being quoted as a competitor. An arm may only contribute a point at a rate it
    actually hit.
    """
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
        want = requested_bpp(extra, pixels)
        got = size * 8 / pixels
        if want is not None and abs(got - want) / want > rate_tolerance:
            print(f"    {name} {label}: DROPPED — asked {want:.3f} bpp, encoder produced "
                  f"{got:.3f} bpp ({(got - want) / want * 100:+.0f}%); rate not honoured")
            continue
        rows.append((name, label, size, measure(orig_rgb, out_png)))
    return rows


def arm_j2k(src_png, orig_rgb, tmp, rates, irreversible=True, tile=None, name=None):
    """OpenJPEG. `-I` selects the irreversible 9/7 transform, and it is not optional here.

    opj_compress defaults to the **reversible 5/3** path, which is the wrong configuration for a
    lossy comparison and costs JPEG 2000 2-3 dB at the same rate (bbb at 4.80 bpp: 45.55 dB
    reversible against 48.58 dB irreversible). The first run of this harness used the default and
    therefore flattered GNC by that margin. The reversible arm stays available because it is a
    real J2K mode and it explains an otherwise baffling reading: reversible 5/3 + RCT codes a
    luma numerically identical to YCoCg-R's, so once its luma subbands are fully coded Y-PSNR
    runs off to 79-105 dB while colour error remains, which is what put a 105 dB point in the
    first run's ladder.
    """
    if shutil.which("opj_compress") is None:
        print("    JPEG 2000: opj_compress not in PATH, skipped")
        return []
    if name is None:
        name = "J2K 9/7" if irreversible else "J2K 5/3rev"
    rows = []
    for rate in rates:
        stem = f"{name.replace(' ', '').replace('/', '')}_r{rate}"
        j2k, out_png = tmp / f"{stem}.j2k", tmp / f"{stem}.png"
        cmd = ["opj_compress", "-i", str(src_png), "-o", str(j2k), "-r", str(rate)]
        if irreversible:
            cmd.append("-I")
        if tile:
            cmd += ["-t", f"{tile},{tile}"]
        r = sh(cmd)
        if r.returncode != 0 or not os.path.exists(j2k):
            print(f"    {name} r={rate}: encode failed")
            continue
        r = sh(["opj_decompress", "-i", str(j2k), "-o", str(out_png)])
        if r.returncode != 0 or not os.path.exists(out_png):
            print(f"    {name} r={rate}: decode failed")
            continue
        rows.append((name, f"r{rate}", os.path.getsize(j2k), measure(orig_rgb, out_png)))
    return rows


def jpegxs_binaries():
    """Where scripts/build_jpegxs_arm64.sh leaves the apps, overridable for another host."""
    root = Path(os.environ.get("GNC_JPEGXS_BIN",
                               Path(os.environ.get("TMPDIR", "/tmp")) / "svt-jpegxs/Bin/Release"))
    enc, dec = root / "SvtJpegxsEncApp", root / "SvtJpegxsDecApp"
    return (enc, dec) if enc.exists() and dec.exists() else (None, None)


def png_to_raw(src_png, pix_fmt, dst):
    r = sh(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src_png),
            "-pix_fmt", pix_fmt, "-frames:v", "1", "-f", "rawvideo", str(dst)])
    return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0


def raw_to_png(src_raw, pix_fmt, w, h, dst_png):
    r = sh(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo",
            "-pix_fmt", pix_fmt, "-s", f"{w}x{h}", "-i", str(src_raw),
            "-pix_fmt", "rgb24", "-frames:v", "1", str(dst_png)])
    return r.returncode == 0 and os.path.exists(dst_png)


def arm_jpegxs(src_png, orig_rgb, tmp, w, h, bpps, colour_format="yuv444", depth=10):
    """JPEG XS through SVT-JPEG-XS — the codec MEAS-9 was filed to measure.

    Not in Homebrew and its CMake assumes x86 unconditionally; `scripts/build_jpegxs_arm64.sh`
    plus `scripts/svt-jpegxs-arm64.patch` build it on arm64 by gating the nasm discovery and the
    nine ASM object libraries, leaving the scalar C fallbacks that were always in the sources.
    Verified there by round trip, not by linking: bbb at --bpp 3 gives PSNR y 44.48 dB.

    **Rate and quality from this build are exact. Throughput is not** — every SIMD kernel is off
    on this architecture, so nothing here may be quoted about JPEG XS speed. That matters because
    speed is the whole reason the format exists (1-32 lines of latency, per EBU TR 092), and a
    rate comparison against it is therefore only half the story.

    `--bpp` is JPEG XS's CBR target and it hits it almost exactly, which is what its market
    requires: constant bitrate is mandatory in live contribution, not a preference.
    """
    enc_bin, dec_bin = jpegxs_binaries()
    if enc_bin is None:
        print("    JPEG XS: SvtJpegxsEncApp not found "
              "(run scripts/build_jpegxs_arm64.sh, or set GNC_JPEGXS_BIN), skipped")
        return []
    pix_fmt = {("yuv444", 10): "yuv444p10le", ("yuv444", 8): "yuv444p",
               ("yuv422", 10): "yuv422p10le", ("yuv422", 8): "yuv422p"}[(colour_format, depth)]
    name = f"JPEG XS {colour_format[3:]}"
    raw = tmp / f"jxs_{colour_format}_{depth}.yuv"
    if not png_to_raw(src_png, pix_fmt, raw):
        print(f"    {name}: could not make {pix_fmt} raw input")
        return []
    rows = []
    for bpp in bpps:
        stem = f"jxs_{colour_format}_{bpp}"
        jxs, out_raw, out_png = tmp / f"{stem}.jxs", tmp / f"{stem}.yuv", tmp / f"{stem}.png"
        r = sh([str(enc_bin), "-i", str(raw), "-w", str(w), "-h", str(h),
                "--colour-format", colour_format, "--input-depth", str(depth),
                "--bpp", str(bpp), "-n", "1", "-b", str(jxs), "--no-progress", "1"])
        if r.returncode != 0 or not os.path.exists(jxs) or os.path.getsize(jxs) == 0:
            print(f"    {name} {bpp}bpp: encode failed "
                  f"{r.stderr.strip().splitlines()[-1:] or r.stdout.strip().splitlines()[-1:]}")
            continue
        r = sh([str(dec_bin), "-i", str(jxs), "-o", str(out_raw)])
        if r.returncode != 0 or not os.path.exists(out_raw):
            print(f"    {name} {bpp}bpp: decode failed")
            continue
        if not raw_to_png(out_raw, pix_fmt, w, h, out_png):
            print(f"    {name} {bpp}bpp: could not convert output back to PNG")
            continue
        size = os.path.getsize(jxs)
        got = size * 8 / (w * h)
        if abs(got - bpp) / bpp > 0.15:
            print(f"    {name} {bpp}bpp: DROPPED — produced {got:.3f} bpp, rate not honoured")
            continue
        rows.append((name, f"{bpp}bpp", size, measure(orig_rgb, out_png)))
        for f in (jxs, out_raw):
            f.unlink(missing_ok=True)
    raw.unlink(missing_ok=True)
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
            for bpp in (3.5, 4.5, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0)]


# ---------------------------------------------------------------------------

CHROMA_WEIGHT_ARMS = ["1.0", "1.6", "2.0", "2.45"]
# INTRA-1 step 2c. GNC quantises as `floor(|v|/step + 0.5)` after a `|v| < dead_zone*step` test,
# so any dead_zone <= 0.5 is a no-op — the rounding already zeroes those. Production interpolates
# 0.5 at q=85 to 0.0 at q>=96, which means **GNC has no dead zone at all in the contribution
# range**: its zero bin is 1.0*step wide. JPEG 2000's irreversible quantiser truncates
# (`floor(|v|/step)`), so its zero bin is **2.0*step** — twice as wide. These arms price that.
DEAD_ZONE_ARMS = ["0.6", "0.75", "0.9", "1.0"]
ARM_ORDER = (["gnc", "gnc_abac", "gnc_abac_t512"] + [f"gnc_abac_cw{c}" for c in CHROMA_WEIGHT_ARMS]
             + [f"gnc_abac_dz{d}" for d in DEAD_ZONE_ARMS]
             + ["jpegxs", "jpegxs422", "prores444", "vc2", "j2k", "j2k_t256",
                "j2k_t512", "j2k_rev", "prores422"])
DEFAULT_ARMS = ["gnc", "jpegxs", "jpegxs422", "prores444", "j2k", "prores422"]


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
    if "gnc_abac" in arms:
        rows += arm_gnc(gnc_binary, img_path, orig, tmp, qualities,
                        extra=["--abac"], name="GNC abac")
    if "gnc_abac_t512" in arms:
        # INTRA-1 step 2 candidate 1, measured inside GNC. Only meaningful on content whose
        # dimensions are a multiple of *both* tile sizes: 1920x1080 pads to 2048x1280 at tile 256
        # and to 2048x1536 at tile 512, so a full-frame comparison charges tile 512 for 20% more
        # coefficients and reads as +6% rate that is entirely padding.
        rows += arm_gnc(gnc_binary, img_path, orig, tmp, qualities,
                        extra=["--abac", "-t", "512"], name="GNC abac t512")
    for dz in DEAD_ZONE_ARMS:
        if f"gnc_abac_dz{dz}" in arms:
            rows += arm_gnc(gnc_binary, img_path, orig, tmp, qualities,
                            extra=["--abac"], name=f"GNC abac dz{dz}",
                            env={"GNC_DEAD_ZONE": dz})
    for cw in CHROMA_WEIGHT_ARMS:
        if f"gnc_abac_cw{cw}" in arms:
            # INTRA-1 step 2: GNC's YCoCg-R has synthesis norms (Y 1.732, Co 0.707, Cg 0.866), so
            # RGB-MSE-optimal chroma steps are 2.45x / 2.00x the luma step. Production runs 1.2
            # (CHROMA-1, chosen on a colour-aware criterion). These arms price that difference on
            # the RGB-PSNR metric the JPEG 2000 comparison is scored with. Decoding needs no
            # matching env — the weight is carried in the bitstream's quantiser, not re-derived.
            rows += arm_gnc(gnc_binary, img_path, orig, tmp, qualities,
                            extra=["--abac"], name=f"GNC abac cw{cw}",
                            env={"GNC_CHROMA_WEIGHT": cw})
    if "prores444" in arms:
        rows += arm_ffmpeg("ProRes 4444", "prores_ks", "yuv444p10le", "mov",
                           prores444_rungs(), img_path, orig, tmp, w * h)
    if "vc2" in arms:
        rows += arm_ffmpeg(f"VC-2 {vc2_qm}", "vc2", "yuv444p10le", "matroska",
                           vc2_rungs(w, h, vc2_qm), img_path, orig, tmp, w * h)
    if "j2k" in arms:
        rows += arm_j2k(img_path, orig, tmp, (40, 20, 12, 8, 5, 4, 3), irreversible=True)
    if "j2k_t256" in arms:
        # The same codec, the same transform, the same five levels — but cut into GNC's 256px
        # tiles instead of one tile over the whole picture. INTRA-1 step 2 candidate 1: this
        # prices the tiling difference in an implementation that does tiling properly, which is
        # far cheaper than making GNC's wavelet shader span a frame (its shared memory is sized
        # for a 510px region, so tile 512 is its hard ceiling).
        rows += arm_j2k(img_path, orig, tmp, (40, 20, 12, 8, 5, 4, 3), irreversible=True,
                        tile=256, name="J2K 9/7 t256")
    if "j2k_t512" in arms:
        rows += arm_j2k(img_path, orig, tmp, (40, 20, 12, 8, 5, 4, 3), irreversible=True,
                        tile=512, name="J2K 9/7 t512")
    if "j2k_rev" in arms:
        rows += arm_j2k(img_path, orig, tmp, (40, 20, 12, 8, 5, 4, 3), irreversible=False)
    if "jpegxs" in arms:
        rows += arm_jpegxs(img_path, orig, tmp, w, h, (1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 10.0),
                           colour_format="yuv444", depth=10)
    if "jpegxs422" in arms:
        rows += arm_jpegxs(img_path, orig, tmp, w, h, (1.5, 2.5, 3.5, 4.5, 6.0, 8.0),
                           colour_format="yuv422", depth=10)
    if "prores422" in arms:
        rows += arm_ffmpeg("ProRes 422", "prores_ks", "yuv422p10le", "mov",
                           prores422_rungs(), img_path, orig, tmp, w * h)

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


MIN_OVERLAP_DB = 3.0


def entropy_identity_check(rows, a="GNC", b="GNC abac"):
    """An entropy coder must not change a single pixel. Verify before believing its rate saving.

    Both coders quantise identically and differ only in how the coefficients are serialised, so
    every quality figure must match to the last digit. A mismatch means the arm is not isolating
    the entropy stage — the exact failure mode that made three of this repo's retracted results
    look like coding wins.
    """
    ra = {r["rung"]: r for r in rows if r["codec"] == a}
    rb = {r["rung"]: r for r in rows if r["codec"] == b}
    shared = sorted(set(ra) & set(rb), key=lambda k: int(k[1:]))
    if not shared:
        return []
    out, bad = [], 0
    for k in shared:
        same = all(abs(ra[k][m] - rb[k][m]) < 1e-9
                   for m in ("psnr_rgb", "psnr_y", "de00_mean", "de00_p95"))
        if not same:
            bad += 1
            out.append(f"{b} {k}: pixels DIFFER from {a} "
                       f"(Y {ra[k]['psnr_y']:.4f} vs {rb[k]['psnr_y']:.4f}) — "
                       f"this arm is not measuring the entropy coder")
        else:
            out.append(f"{b} {k}: pixels identical to {a}, "
                       f"rate {ra[k]['bpp']:.3f} -> {rb[k]['bpp']:.3f} bpp "
                       f"({(rb[k]['bpp'] / ra[k]['bpp'] - 1) * 100:+.1f}%)")
    out.insert(0, f"entropy identity {a} vs {b}: "
                  f"{len(shared) - bad}/{len(shared)} rungs bit-identical in pixels"
                  + ("" if not bad else f" — {bad} DIFFER, rate figures below are not comparable"))
    return out


def saturation_warnings(rows):
    """Arms whose quality stops responding to rate — an encoder limit, not a codec property.

    ffmpeg's VC-2 encoder is why this exists: it saturates near 41-43 dB RGB PSNR on every
    pixel format, bit depth and slice geometry tried, so 12 bpp buys 0.1 dB over 6 bpp. A
    BD-rate against a saturated arm reports the encoder's ceiling and reads as a landslide for
    whoever is not saturated. Anything flagged here cannot be quoted as a statement about the
    *format*.
    """
    out = []
    by_codec = {}
    for r in rows:
        by_codec.setdefault(r["codec"], []).append(r)
    for codec, rs in sorted(by_codec.items()):
        rs = sorted(rs, key=lambda r: r["bpp"])
        if len(rs) < 2:
            continue
        a, b = rs[-2], rs[-1]
        d_rate = (b["bpp"] - a["bpp"]) / a["bpp"] if a["bpp"] else 0.0
        d_q = b["psnr_rgb"] - a["psnr_rgb"]
        if d_rate > 0.20 and np.isfinite(d_q) and d_q < 0.5:
            out.append(f"{codec}: +{d_rate * 100:.0f}% rate buys {d_q:+.2f} dB at the top of its "
                       f"ladder ({a['bpp']:.2f} -> {b['bpp']:.2f} bpp) — rate-insensitive, so its "
                       f"ceiling is the encoder's, not the format's")
    return out


def _window(pts, lo, hi):
    """Points inside [lo, hi] on the quality axis, plus the nearest one on each side.

    A cubic is fitted to whatever it is given, so points far outside the integration range bend
    the curve *inside* it. Two arms here make that concrete: JPEG 2000 codes a reversible RCT
    luma, so its top rung reaches 79-105 dB Y-PSNR, and ffmpeg's VC-2 emits 10 dB garbage below
    ~4 bpp. Fitting either curve globally and integrating over a 15 dB window is not the same
    quantity as fitting the window.
    """
    inside = [p for p in pts if lo <= p[1] <= hi]
    below = [p for p in pts if p[1] < lo]
    above = [p for p in pts if p[1] > hi]
    if below:
        inside.append(max(below, key=lambda p: p[1]))
    if above:
        inside.append(min(above, key=lambda p: p[1]))
    return sorted(inside, key=lambda p: p[1])


def bd_summary(rows, metric, min_overlap=MIN_OVERLAP_DB, test="GNC"):
    """BD-rate of GNC against each other arm on `metric`. Negative = GNC needs fewer bits.

    Returns (codec, bd_percent_or_None, lo, hi, note) per arm. `note` says why a number is
    missing, because a silent n/a reads as "no difference" and a BD-rate integrated over a
    fraction of a dB reads as a landslide — the first run of this harness produced -58.6% and
    -65.7% from overlaps of 0.1 and 2.5 dB.
    """
    by_codec = {}
    for r in rows:
        if np.isfinite(r[metric]):
            by_codec.setdefault(r["codec"], []).append((r["bpp"], r[metric]))
    if test not in by_codec:
        return []
    g_all = sorted(by_codec[test], key=lambda p: p[1])
    out = []
    for codec, pts in sorted(by_codec.items()):
        if codec == test or codec.startswith("GNC"):
            continue
        p_all = sorted(pts, key=lambda p: p[1])
        lo = max(p_all[0][1], g_all[0][1])
        hi = min(p_all[-1][1], g_all[-1][1])
        if hi - lo < min_overlap:
            out.append((codec, None, lo, hi,
                        f"overlap {max(hi - lo, 0.0):.1f} dB < {min_overlap:.0f} dB required"))
            continue
        p, g = _window(p_all, lo, hi), _window(g_all, lo, hi)
        if len(p) < 4 or len(g) < 4:
            out.append((codec, None, lo, hi,
                        f"only {min(len(p), len(g))} points in the overlap, 4 needed"))
            continue
        bd, _ = bd_rate_raw([x[0] for x in p], [x[1] for x in p],
                            [x[0] for x in g], [x[1] for x in g])
        out.append((codec, bd, lo, hi, ""))
    return out


def drop_gnc_above(rows, cap):
    """The same rows with GNC's rungs above q=`cap` removed.

    GNC's top rungs are **dominated by its own lossless path**: LOSSLESS-1 made q=100 code MED
    residuals instead of wavelet coefficients and 14.9% cheaper, which moved the bit-exact price
    *below* the top of the lossy ladder. Measured on these four stills (RATE-2, another session):
    q=99 costs +9.3% (bbb), +40.6% (blue_sky), +35.9% (kristensara), +29.6% (touchdown) more than
    q=100 for output that is worse than bit-exact, and domination starts at q=98/95/96/96.

    So a BD-rate integrated over the full ladder scores GNC partly through settings a user should
    never choose. The rungs stay in the table — they are what the encoder produces, and removing
    them would flatter GNC — but the BD-rate is reported twice, and the difference between the two
    is the self-inflicted part.
    """
    out = []
    for r in rows:
        if r["codec"].startswith("GNC") and r["rung"].startswith("q"):
            try:
                if int(r["rung"][1:]) > cap:
                    continue
            except ValueError:
                pass
        out.append(r)
    return out


def matched_rate_table(rows, gnc_rows):
    """Each incumbent rung against GNC interpolated to the *same* bpp.

    BD-rate cannot compare arms whose quality ranges barely overlap, and it cannot say anything
    about colour at all. This can: it is the comparison CLAUDE.md asks for when the question is a
    luma/chroma trade, and it is what the 4:2:2 arms have to be judged on, since their subsampling
    ceiling sits below the range where a BD-rate against GNC would be computable.
    """
    if len(gnc_rows) < 2:
        return []
    g = sorted(gnc_rows, key=lambda r: r["bpp"])
    bpps = [r["bpp"] for r in g]
    out = []
    for r in sorted(rows, key=lambda r: (r["codec"], r["bpp"])):
        if r["codec"].startswith("GNC") or not (bpps[0] <= r["bpp"] <= bpps[-1]):
            continue  # outside GNC's measured ladder: extrapolation, not measurement
        at = {k: float(np.interp(r["bpp"], bpps, [x[k] for x in g]))
              for k in ("psnr_y", "psnr_rgb", "de00_mean")}
        out.append((r, at))
    return out


def selftest():
    """Prove the BD-rate machinery on cases with known answers before trusting it on codecs.

    chroma_metric.py validates dE00 against Sharma's reference pairs; this is the same idea for
    the other half of the instrument. Three cases, and the third is the one that caught a real
    defect: a curve fitted globally and integrated over a narrow window is not the same quantity
    as one fitted to the window.
    """
    ok = True

    def check(name, got, want, tol=0.15):
        nonlocal ok
        good = got is not None and abs(got - want) <= tol
        ok = ok and good
        print(f"  {'pass' if good else 'FAIL'}  {name}: got "
              f"{'None' if got is None else f'{got:+.3f}%'}, expected {want:+.1f}%")

    base = [(1.0, 40.0), (2.0, 44.0), (4.0, 48.0), (8.0, 52.0), (16.0, 56.0)]
    rows = [{"codec": "GNC", "bpp": b, "psnr_y": q, "psnr_rgb": q, "de00_mean": 0.0}
            for b, q in base]

    # 1. A curve against itself is 0% by construction.
    same = [dict(r, codec="Same") for r in rows]
    bd = bd_summary(rows + same, "psnr_y")
    check("identical curves", bd[0][1], 0.0, tol=1e-6)

    # 2. Scaling every rate by 1.25 must read as exactly +25%: BD-rate is a ratio of integrated
    #    log-rate, so a constant factor comes through untouched whatever the fit does.
    cheap = [dict(r, codec="Cheap", bpp=r["bpp"] / 1.25) for r in rows]
    bd = bd_summary(rows + cheap, "psnr_y")
    check("reference 20% cheaper everywhere", bd[0][1], 25.0, tol=1e-6)

    # 3. An arm that overlaps by less than 3 dB must return no number at all, not a landslide.
    #    The first run of this harness reported -58.6% and -65.7% from overlaps of 0.1 and 2.5 dB.
    narrow = [{"codec": "Narrow", "bpp": b, "psnr_y": q, "psnr_rgb": q, "de00_mean": 0.0}
              for b, q in [(1.0, 55.0), (2.0, 55.5), (4.0, 56.0), (8.0, 56.4)]]
    bd = bd_summary(rows + narrow, "psnr_y")
    got = bd[0][1]
    print(f"  {'pass' if got is None else 'FAIL'}  1.4 dB overlap: "
          f"{'refused, ' + bd[0][4] if got is None else f'returned {got:+.1f}%'}")
    ok = ok and got is None

    # 4. Matched-rate interpolation must land on a measured point exactly when asked for its rate.
    incumbent = [{"image": "x", "codec": "Other", "rung": "r", "bpp": 4.0,
                  "psnr_y": 40.0, "psnr_rgb": 40.0, "de00_mean": 1.0}]
    m = matched_rate_table(incumbent, [dict(r, image="x", rung="q") for r in rows])
    at = m[0][1]["psnr_y"] if m else None
    check("interpolation at a measured rate", at, 48.0, tol=1e-9)

    print("selftest:", "all pass" if ok else "FAILURES — do not trust the BD-rate column")
    return 0 if ok else 1


def main():
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", help="reference PNGs (not needed with --selftest)")
    ap.add_argument("--gnc-binary", default=str(root / "target/release/gnc"))
    ap.add_argument("--qualities", default="60,75,85,90,95,99",
                    help="GNC quality ladder (default spans the contribution operating point)")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help=f"any of: {', '.join(ARM_ORDER)}")
    ap.add_argument("--gnc-bdrate-cap", type=int, default=94,
                    help="also report BD-rate with GNC rungs above this q removed; above it GNC "
                         "is dominated by its own q=100 lossless path (RATE-2), so the full-ladder "
                         "figure scores it through settings nobody should choose")
    ap.add_argument("--vc2-qm", default="default", choices=("default", "color", "flat"),
                    help="VC-2 quantisation matrix; 'flat' is its own optimise-for-PSNR setting")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the decoded PNGs for inspection")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the BD-rate and interpolation machinery and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.images:
        ap.error("--images is required unless --selftest is given")

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
            for w in saturation_warnings(rows):
                print(f"  CANARY  {w}")
            for w in entropy_identity_check(rows):
                print(f"  CANARY  {w}")
            capped = drop_gnc_above(rows, args.gnc_bdrate_cap)
            tests = [t for t in ("GNC", "GNC abac") if any(r["codec"] == t for r in rows)]
            for metric, name in (("psnr_y", "Y-PSNR (YCoCg-R)"), ("psnr_rgb", "RGB PSNR")):
                for test in tests:
                    bds = bd_summary(rows, metric, test=test)
                    cap_bds = {c: bd for c, bd, _, _, _
                               in bd_summary(capped, metric, test=test)}
                    if not bds:
                        continue
                    print(f"  BD-rate on {name}, {test} vs (full ladder | q<="
                          f"{args.gnc_bdrate_cap} only, see RATE-2):")
                    for codec, bd, lo, hi, note in bds:
                        if bd is None:
                            print(f"    {codec:<12} n/a — {note}")
                            continue
                        cb = cap_bds.get(codec)
                        cb_txt = f"{cb:+8.1f}%" if cb is not None else "     n/a"
                        print(f"    {codec:<12} {bd:+8.1f}% | {cb_txt}  "
                              f"(fitted over {lo:.1f}-{hi:.1f} dB)")
            matched = matched_rate_table(rows, [r for r in rows if r["codec"] == "GNC"])
            if matched:
                print("  At matched rate, GNC minus the incumbent "
                      "(+dB and -dE00 mean GNC is better):")
                print(f"    {'codec':<12} {'rung':<14} {'bpp':>7} {'dY':>7} {'dRGB':>7} {'ddE00':>8}")
                for r, at in matched:
                    print(f"    {r['codec']:<12} {r['rung']:<14} {r['bpp']:>7.3f} "
                          f"{at['psnr_y'] - r['psnr_y']:>+7.2f} "
                          f"{at['psnr_rgb'] - r['psnr_rgb']:>+7.2f} "
                          f"{at['de00_mean'] - r['de00_mean']:>+8.4f}")
            all_rows += rows

        if len(args.images) > 1:
            print("\n=== mean BD-rate across images ===")
            for metric, name in (("psnr_y", "Y-PSNR (YCoCg-R)"), ("psnr_rgb", "RGB PSNR")):
                for test in [t for t in ("GNC", "GNC abac")
                             if any(r["codec"] == t for r in all_rows)]:
                    per_codec, per_codec_cap = {}, {}
                    for img in args.images:
                        rows = [r for r in all_rows if r["image"] == Path(img).name]
                        for codec, bd, _, _, _ in bd_summary(rows, metric, test=test):
                            if bd is not None:
                                per_codec.setdefault(codec, []).append(bd)
                        for codec, bd, _, _, _ in bd_summary(
                                drop_gnc_above(rows, args.gnc_bdrate_cap), metric, test=test):
                            if bd is not None:
                                per_codec_cap.setdefault(codec, []).append(bd)
                    print(f"  on {name}, {test} (full ladder | q<={args.gnc_bdrate_cap} only):")
                    for codec, vals in sorted(per_codec.items()):
                        cap_vals = per_codec_cap.get(codec)
                        cap_txt = f"{np.mean(cap_vals):+8.1f}%" if cap_vals else "     n/a"
                        print(f"    {codec:<12} {np.mean(vals):+8.1f}% | {cap_txt}   "
                              f"full: ({', '.join(f'{v:+.1f}' for v in vals)})"
                              + (f"  capped: ({', '.join(f'{v:+.1f}' for v in cap_vals)})"
                                 if cap_vals else ""))

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
