#!/usr/bin/env python3
"""INTRA-1 — how much of the JPEG 2000 gap is GNC coding its own tile padding?

**The question.** GNC pads every plane up to a whole multiple of `tile_size` with edge
replication (`src/shaders/pad.wgsl`) and then codes the padded plane. A 1920x1080 frame becomes
2048x1280: **26.4% more coefficients, 20.9% of the coded samples outside the picture.** The
decoder crops them away. JPEG 2000 in whole-picture mode codes 1920x1080 exactly, and every
figure in the INTRA-1 accounting divides both codecs' bytes by the *visible* pixel count. So the
GNC arm has been carrying a tax the J2K arm does not pay, and nothing in the record prices it.

RESEARCH_LOG records padding only as a *tile-size comparison* confound ("1920x1080 pads to
2048x1280 at tile 256 and 2048x1536 at tile 512"), which is a different question — that one is
about comparing two GNC configurations, this one is about comparing GNC to a codec that does not
pad at all.

**Two measurements, and they have to agree.**

Part 1 — *content-controlled*, GNC only. Take the largest tile-aligned centred crop of each image
(`A`), then crops one pixel wider (`B`), one taller (`C`), and both (`D`). Those three differ from
`A` by ~0.15% of content and by a whole tile row/column of padding. Any rate difference at equal
PSNR is the padding, with content held fixed. `C` is the geometry that matches a native 1080p
frame most closely (padded/real 1.249 against 1.264, and bottom-heavy the same way).

Part 2 — *cross-codec*, on padding-free content. Run the ENT-4 comparison (GNC `--abac` against
OpenJPEG 9/7, whole picture) on the `A` crops, where GNC pays no padding at all, and read the gap
against ENT-4's native +27.1%. This one is confounded by the content change; Part 1 is not. They
answer the same question from opposite ends, and the point of running both is that neither is
trusted alone.

**Canary that the padding is really coded** (`--canary`): rebuild the edge-replicated padded plane
in Python, encode *that* as a picture in its own right, and check the byte count against the
unpadded crop's. Byte-identical means GNC spent exactly those bits on samples the decoder throws
away. Anything else means this harness has the mechanism wrong.

Usage:
    scripts/meas_intra1_padding.py --images test_material/frames/*.png --csv padding.csv
    scripts/meas_intra1_padding.py --images ... --part 1 --qualities 85,90,95
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meas9_contribution import arm_j2k, measure, sh  # noqa: E402
from meas1_vs_h264 import bd_rate as bd_rate_raw  # noqa: E402

TILE = 256

# The four crop variants. `A` is tile-aligned; the others add one pixel of picture and therefore a
# whole tile row and/or column of padding.
VARIANTS = ("A", "B", "C", "D")
VARIANT_NOTE = {
    "A": "tile-aligned, no padding",
    "B": "+1 column  -> one tile column of padding",
    "C": "+1 row     -> one tile row of padding",
    "D": "+1 both    -> a tile row and a tile column",
}


def ceil_tile(n):
    return ((n + TILE - 1) // TILE) * TILE


def aligned_base(w, h):
    """Largest tile-aligned size *strictly* smaller than the image in both axes.

    Strictly smaller, because the `B`/`C`/`D` variants need one pixel of headroom to crop into.
    kristensara_720p is 1280 wide, an exact multiple of 256, so `w // TILE * TILE` would leave
    nothing to add.
    """
    return ((w - 1) // TILE) * TILE, ((h - 1) // TILE) * TILE


def variant_size(base_w, base_h, variant):
    return {
        "A": (base_w, base_h),
        "B": (base_w + 1, base_h),
        "C": (base_w, base_h + 1),
        "D": (base_w + 1, base_h + 1),
    }[variant]


def pad_ratio(w, h):
    """Coded samples per visible sample."""
    return (ceil_tile(w) * ceil_tile(h)) / (w * h)


def replicate_pad(arr, pw, ph):
    """Edge-replicate to (ph, pw) — the same extension `pad.wgsl` performs on the GPU."""
    h, w = arr.shape[:2]
    return arr[np.minimum(np.arange(ph), h - 1)][:, np.minimum(np.arange(pw), w - 1)]


def gnc_point(gnc, src_png, orig_rgb, tmp, q, extra, tag):
    """One (rate, quality) point from the GNC arm, or None if either half failed."""
    bs, out_png = tmp / f"{tag}_q{q}.gnc", tmp / f"{tag}_q{q}.png"
    r = sh([str(gnc), "encode", "-i", str(src_png), "-o", str(bs), "-q", str(q), *extra])
    if r.returncode != 0:
        print(f"      encode failed q={q}: {r.stderr.strip().splitlines()[-1:]}")
        return None
    r = sh([str(gnc), "decode", "-i", str(bs), "-o", str(out_png)])
    if r.returncode != 0:
        print(f"      decode failed q={q}: {r.stderr.strip().splitlines()[-1:]}")
        return None
    m = measure(orig_rgb, out_png)
    return {"q": q, "bytes": os.path.getsize(bs), **m}


def bpp(byts, w, h):
    """Bits per *visible* pixel — the padding is charged to the picture, which is the point."""
    return byts * 8.0 / (w * h)


def bd(ref, test, metric):
    """BD-rate of `test` against `ref`, positive meaning `test` needs more bits.

    The overlapping-quality window comes back with the figure and is worth watching: two ladders
    that barely overlap produce a number from almost no data.

    **A rung whose quality is infinite is refused, not integrated.** Since RATE-2 (2026-09-08) the
    encoder codes q=95..99 both ways and keeps the smaller, so a rung in that range can come back
    *bit-exact lossless* — `psnr()` returns `inf` and `np.polyfit` then returns `nan` for the whole
    curve. That is a silent nonsense number, and the padded crops in this harness are exactly the
    arms that reach lossless first, because a flat padding region is cheap to code losslessly. So a
    ladder that touches the dual-path range cannot be used for a BD-rate here: keep every rung at
    **q <= 94** and this guard never fires.
    """
    ra = [p["bpp"] for p in ref]
    qa = [p[metric] for p in ref]
    rb = [p["bpp"] for p in test]
    qb = [p[metric] for p in test]
    if min(len(ra), len(rb)) < 4:
        return None, (0.0, 0.0)
    if not all(np.isfinite(q) for q in qa + qb):
        return None, (float("nan"), float("nan"))
    return bd_rate_raw(ra, qa, rb, qb)


def canary(gnc, img_path, tmp, q, extra):
    """Prove the padded plane is what GNC codes, by coding it as a picture in its own right."""
    im = np.array(Image.open(img_path).convert("RGB"))
    h, w = im.shape[:2]
    pw, ph = ceil_tile(w), ceil_tile(h)
    if (pw, ph) == (w, h):
        return None
    pre = tmp / "canary_prepadded.png"
    Image.fromarray(replicate_pad(im, pw, ph)).save(pre)
    a = gnc_point(gnc, img_path, im, tmp, q, extra, "canary_crop")
    b = gnc_point(gnc, pre, np.array(Image.open(pre)), tmp, q, extra, "canary_pre")
    if a is None or b is None:
        return None
    return {
        "crop": f"{w}x{h}", "padded": f"{pw}x{ph}",
        "crop_bytes": a["bytes"], "prepadded_bytes": b["bytes"],
        "identical": a["bytes"] == b["bytes"],
    }


def part1(gnc, images, qualities, extra, tmp, rows):
    """Content-controlled: what one extra pixel of picture costs, when it forces a tile of padding."""
    print("\n=== Part 1 — content-controlled padding cost (GNC only) ===")
    per_image = {}
    for img in images:
        name = Path(img).stem
        full = Image.open(img).convert("RGB")
        W, H = full.size
        bw, bh = aligned_base(W, H)
        ox, oy = (W - bw) // 2, (H - bh) // 2
        print(f"\n  {name}: {W}x{H}, aligned base {bw}x{bh} at ({ox},{oy})")
        ladders = {}
        for v in VARIANTS:
            w, h = variant_size(bw, bh, v)
            crop_png = tmp / f"{name}_{v}.png"
            crop = full.crop((ox, oy, ox + w, oy + h))
            crop.save(crop_png)
            orig = np.array(crop)
            ratio = pad_ratio(w, h)
            pts = []
            for q in qualities:
                p = gnc_point(gnc, crop_png, orig, tmp, q, extra, f"{name}_{v}")
                if p is None:
                    continue
                p["bpp"] = bpp(p["bytes"], w, h)
                pts.append(p)
                rows.append({
                    "part": 1, "image": name, "variant": v, "w": w, "h": h,
                    "coded_w": ceil_tile(w), "coded_h": ceil_tile(h), "pad_ratio": ratio,
                    "arm": "GNC abac", "q": q, "bytes": p["bytes"], "bpp": p["bpp"],
                    "psnr_rgb": p["psnr_rgb"], "psnr_y": p["psnr_y"], "de00_mean": p["de00_mean"],
                })
            ladders[v] = pts
            print(f"    {v} {w:5d}x{h:<5d} coded {ceil_tile(w)}x{ceil_tile(h)} "
                  f"ratio {ratio:.4f}  {VARIANT_NOTE[v]}")
            for p in pts:
                print(f"        q{p['q']:<3d} {p['bytes']:>9d} B  {p['bpp']:.4f} bpp  "
                      f"RGB {p['psnr_rgb']:.3f} dB  Y {p['psnr_y']:.3f} dB")
        per_image[name] = ladders

    print("\n  Padding cost, BD-rate against the tile-aligned crop of the same content.")
    print("  `dRGB@mid` is the quality difference at the middle rung — it must be ~0, or the two")
    print("  ladders are not at the same operating point and the rate difference is not padding.")
    print(f"  {'image':<20} {'variant':<8} {'pad ratio':>9} {'RGB BD-rate':>12} {'Y BD-rate':>11}"
          f" {'dRGB@mid':>9} {'dbytes@mid':>11}")
    summary = {}
    for img in images:
        name = Path(img).stem
        ladders = per_image[name]
        bw, bh = aligned_base(*Image.open(img).size)
        ref = ladders["A"]
        for v in VARIANTS[1:]:
            test = ladders[v]
            if len(ref) < 4 or len(test) < 4:
                print(f"  {name:<20} {v:<8} too few points for a BD-rate")
                continue
            (rgb, wr), (y, _) = bd(ref, test, "psnr_rgb"), bd(ref, test, "psnr_y")
            mid = len(ref) // 2
            dq = test[mid]["psnr_rgb"] - ref[mid]["psnr_rgb"]
            db = test[mid]["bytes"] / ref[mid]["bytes"] - 1.0
            w, h = variant_size(bw, bh, v)
            summary.setdefault(name, {})[v] = (pad_ratio(w, h), rgb, y, db)
            print(f"  {name:<20} {v:<8} {pad_ratio(w, h):>9.4f} {fmt(rgb):>12} {fmt(y):>11}"
                  f" {dq:>+9.3f} {db:>+10.2%}")

    print("\n  Is the cost separable? D should equal B + C if a padding column and a padding row")
    print("  are independent, which is what lets the figure be carried to another frame size.")
    for name, s in summary.items():
        if all(v in s for v in ("B", "C", "D")) and all(s[v][1] is not None for v in "BCD"):
            print(f"  {name:<20} B {s['B'][1]:+.2f}% + C {s['C'][1]:+.2f}% = "
                  f"{s['B'][1] + s['C'][1]:+.2f}%  against D {s['D'][1]:+.2f}%")
    return summary


def part1_control(images, j2k_rates, tmp, rows):
    """Part 1's control: the same crop pair through a codec that does not pad.

    Part 1 attributes the whole `A` -> `D` rate difference to padding on the grounds that the two
    crops differ by ~0.15% of picture. That is an argument, not a measurement. JPEG 2000 in
    whole-picture mode codes both crops exactly as given, so its `A` -> `D` BD-rate is the size of
    the content term on its own — and it has to be near zero for Part 1 to mean what it says.
    """
    print("\n=== Part 1 control — the same crop pair through JPEG 2000 9/7 (no padding) ===")
    if shutil.which("opj_compress") is None:
        print("  opj_compress not in PATH — skipped")
        return
    print(f"  {'image':<20} {'A -> D, RGB':>12} {'A -> D, Y':>11}  (near 0 = the extra row and "
          f"column of content is negligible)")
    for img in images:
        name = Path(img).stem
        full = Image.open(img).convert("RGB")
        W, H = full.size
        bw, bh = aligned_base(W, H)
        ox, oy = (W - bw) // 2, (H - bh) // 2
        ladders = {}
        for v in ("A", "D"):
            w, h = variant_size(bw, bh, v)
            crop_png = tmp / f"{name}_ctrl_{v}.png"
            crop = full.crop((ox, oy, ox + w, oy + h))
            crop.save(crop_png)
            orig = np.array(crop)
            pts = []
            for arm_name, label, byts, m in arm_j2k(crop_png, orig, tmp, j2k_rates,
                                                    name=f"J2K 9/7 ctrl {v}"):
                pts.append({"bpp": bpp(byts, w, h), "bytes": byts, **m})
                rows.append({"part": "1c", "image": name, "variant": v, "w": w, "h": h,
                             "coded_w": w, "coded_h": h, "pad_ratio": 1.0, "arm": arm_name,
                             "q": label, "bytes": byts, "bpp": pts[-1]["bpp"],
                             "psnr_rgb": m["psnr_rgb"], "psnr_y": m["psnr_y"],
                             "de00_mean": m["de00_mean"]})
            ladders[v] = pts
        if min(len(ladders["A"]), len(ladders["D"])) < 4:
            print(f"  {name:<20} too few points")
            continue
        (rgb, _), (y, _) = (bd(ladders["A"], ladders["D"], "psnr_rgb"),
                            bd(ladders["A"], ladders["D"], "psnr_y"))
        print(f"  {name:<20} {fmt(rgb):>12} {fmt(y):>11}")


def fmt(x):
    return "n/a" if x is None else f"{x:+.2f}%"


def _cross_codec_gap(gnc, src_png, orig, w, h, gnc_tag, qualities, extra, j2k_rates, tmp,
                     rows, image, variant):
    """One GNC-vs-J2K-9/7 BD-rate on one picture, both arms through the same metric path."""
    g = []
    for q in qualities:
        p = gnc_point(gnc, src_png, orig, tmp, q, extra, gnc_tag)
        if p is None:
            continue
        p["bpp"] = bpp(p["bytes"], w, h)
        g.append(p)
        rows.append({"part": 2, "image": image, "variant": variant, "w": w, "h": h,
                     "coded_w": ceil_tile(w), "coded_h": ceil_tile(h),
                     "pad_ratio": pad_ratio(w, h), "arm": "GNC abac", "q": q,
                     "bytes": p["bytes"], "bpp": p["bpp"], "psnr_rgb": p["psnr_rgb"],
                     "psnr_y": p["psnr_y"], "de00_mean": p["de00_mean"]})
    j = []
    for arm_name, label, byts, m in arm_j2k(src_png, orig, tmp, j2k_rates,
                                            name=f"J2K 9/7 {variant}"):
        pt = {"bpp": bpp(byts, w, h), "bytes": byts, **m}
        j.append(pt)
        rows.append({"part": 2, "image": image, "variant": variant, "w": w, "h": h,
                     "coded_w": ceil_tile(w), "coded_h": ceil_tile(h),
                     "pad_ratio": pad_ratio(w, h), "arm": arm_name, "q": label,
                     "bytes": byts, "bpp": pt["bpp"], "psnr_rgb": m["psnr_rgb"],
                     "psnr_y": m["psnr_y"], "de00_mean": m["de00_mean"]})
    if len(g) < 4 or len(j) < 4:
        return None, None, None
    (rgb, win), (y, _) = bd(j, g, "psnr_rgb"), bd(j, g, "psnr_y")
    return rgb, y, win


def part2(gnc, images, qualities, extra, j2k_rates, tmp, rows):
    """Cross-codec, native against padding-free, **on one ladder**.

    The first version of this part read its padding-free gap against ENT-4's published +27.1%,
    which was taken on a different GNC ladder (q=60-99 against q=80-98 here). A BD-rate is
    integrated over the *overlapping* quality range, so two ladders with different ends give two
    different numbers on identical content, and the difference would have been quoted as padding.
    So the native arm is re-run here rather than cited: same q ladder, same J2K rates, same metric
    path, and only the picture changes. The overlap window is printed so the reader can see it.
    """
    print("\n=== Part 2 — GNC against JPEG 2000 9/7, native vs padding-free, one ladder ===")
    if shutil.which("opj_compress") is None:
        print("  opj_compress not in PATH — skipped")
        return {}
    out = {}
    for img in images:
        name = Path(img).stem
        full = Image.open(img).convert("RGB")
        W, H = full.size
        bw, bh = aligned_base(W, H)
        ox, oy = (W - bw) // 2, (H - bh) // 2
        crop_png = tmp / f"{name}_A.png"
        full.crop((ox, oy, ox + bw, oy + bh)).save(crop_png)
        native_png = tmp / f"{name}_native.png"
        full.save(native_png)
        print(f"\n  {name}: native {W}x{H} (pad ratio {pad_ratio(W, H):.4f}) against "
              f"padding-free {bw}x{bh}")

        nat = _cross_codec_gap(gnc, native_png, np.array(full), W, H, f"{name}_p2nat",
                               qualities, extra, j2k_rates, tmp, rows, name, "native")
        crp = _cross_codec_gap(gnc, crop_png, np.array(Image.open(crop_png)), bw, bh,
                               f"{name}_p2crop", qualities, extra, j2k_rates, tmp, rows,
                               name, "padding-free")
        if nat[0] is None or crp[0] is None:
            print("    too few points on one arm")
            continue
        out[name] = (nat[0], crp[0], nat[1], crp[1])
        print(f"    native       RGB {fmt(nat[0]):>9}  Y {fmt(nat[1]):>9}   "
              f"overlap {nat[2][0]:.2f}-{nat[2][1]:.2f} dB")
        print(f"    padding-free RGB {fmt(crp[0]):>9}  Y {fmt(crp[1]):>9}   "
              f"overlap {crp[2][0]:.2f}-{crp[2][1]:.2f} dB")
        print(f"    drop         RGB {nat[0] - crp[0]:+9.2f}  Y {nat[1] - crp[1]:+9.2f}")
    if out:
        dr = [v[0] - v[1] for v in out.values()]
        dy = [v[2] - v[3] for v in out.values()]
        print(f"\n  mean over {len(dr)} images: native RGB "
              f"{np.mean([v[0] for v in out.values()]):+.2f}%, padding-free "
              f"{np.mean([v[1] for v in out.values()]):+.2f}%, "
              f"**drop {np.mean(dr):+.2f} points**")
        print(f"  Y: native {np.mean([v[2] for v in out.values()]):+.2f}%, padding-free "
              f"{np.mean([v[3] for v in out.values()]):+.2f}%, drop {np.mean(dy):+.2f} points")
        print("  For reference only, not the comparison: ENT-4's native figures on its own "
              "q=60-99 ladder were RGB +27.1%, Y +48.3%.")
    return out


def project_native(csv_path, images):
    """Carry Part 1's crop measurement to the frame sizes the gap was actually quoted at.

    Part 1 measures a *whole* tile row or column of padding on a crop; a native 1920x1080 frame
    carries 200 padding rows and 128 padding columns instead of 255 of each, on a slightly larger
    picture. The model is the one Part 1 tests rather than assumes: the cost of a padding row and
    the cost of a padding column are independent and each scales with the strip's area (`D` came
    out within 0.3-1.1 points of `B + C` on all four images).

        cost% ~= [k_v * W * padrows + k_h * H * padcols] / visible_cost
        k_v    = c_C * A_bytes / (bw * 255)      (bytes per padding sample, from a padding row)
        k_h    = c_B * A_bytes / (bh * 255)      (from a padding column)

    The corner is counted in both strips, which overstates the projection by ~5% of itself. Left
    that way deliberately: an overstated tax is the conservative direction for a claim that the
    recorded gap is too large.
    """
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    p1 = [r for r in rows if r["part"] == "1"]
    if not p1:
        print("  no Part 1 rows in the CSV — nothing to project")
        return {}
    qs = sorted({int(r["q"]) for r in p1})
    midq = qs[len(qs) // 2]
    out = {}
    print(f"\n=== Projection to native frame size (from Part 1, q={midq}) ===")
    print(f"  {'image':<20} {'native':>11} {'padrows':>8} {'padcols':>8} "
          f"{'projected':>10} {'measured':>10}")
    for img in images:
        name = Path(img).stem
        W, H = Image.open(img).size
        mine = [r for r in p1 if r["image"] == name and int(r["q"]) == midq]
        by_v = {r["variant"]: r for r in mine}
        if not all(v in by_v for v in ("A", "B", "C")):
            continue
        bw, bh = int(by_v["A"]["w"]), int(by_v["A"]["h"])
        a = float(by_v["A"]["bytes"])
        c_b = float(by_v["B"]["bytes"]) / a - 1.0
        c_c = float(by_v["C"]["bytes"]) / a - 1.0
        k_v = c_c * a / (bw * 255.0)
        k_h = c_b * a / (bh * 255.0)
        padrows, padcols = ceil_tile(H) - H, ceil_tile(W) - W
        visible = a * (W * H) / (bw * bh)
        proj = (k_v * W * padrows + k_h * H * padcols) / visible
        out[name] = proj
        print(f"  {name:<20} {f'{W}x{H}':>11} {padrows:>8} {padcols:>8} {proj:>+9.2%}")
    if out:
        print(f"  mean projected padding tax over {len(out)} images: "
              f"{np.mean(list(out.values())):+.2%}")
        print("  Compare with Part 2's native-against-padding-free drop on the same images.")
    return out


# ---------------------------------------------------------------------------
# Part 3 — the fill oracle
# ---------------------------------------------------------------------------
#
# The padded samples are **don't-care**: the decoder crops them and nothing ever looks at them. So
# the encoder is free to put whatever is cheapest there, and `pad.wgsl`'s edge replication is a
# *choice*, not a constraint. Replication is flat along the direction it extends — the cheapest
# possible in that axis — but it carries the edge line's full transverse detail across every
# padded row, and that detail is coded at every decomposition level.
#
# This part prices the choice without touching the codec, by exploiting the canary above: GNC
# codes exactly the edge-replicated padded plane, so feeding it a *tile-aligned* plane that was
# padded in Python is the same encode. Build the padded plane with different fills, encode each as
# a picture in its own right, and score quality on the visible region only.
#
# `replicate` must reproduce the production encode byte for byte. That is what ties this oracle to
# the shipped path; if it does not, the rest of the part means nothing.


def fill_replicate(vis, pw, ph):
    return replicate_pad(vis, pw, ph)


def fill_mirror(vis, pw, ph):
    """Whole-point symmetric extension — what the wavelet does at a tile edge, applied to the
    picture edge. Smooth at the seam and therefore cheap *there*, but it copies real detail into
    the padding, so it should be the expensive end."""
    h, w = vis.shape[:2]
    iy = np.array([whole_point(y, h) for y in range(ph)])
    ix = np.array([whole_point(x, w) for x in range(pw)])
    return vis[iy][:, ix]


def whole_point(i, n):
    """Index into [0, n) by whole-point reflection, the extension JPEG 2000 specifies."""
    if n == 1:
        return 0
    period = 2 * (n - 1)
    i %= period
    return i if i < n else period - i


def fill_decay(vis, pw, ph, ramp=32):
    """Replicate, then fade to a single scalar over `ramp` pixels.

    Continuous at the seam like replication, so it does not pollute the coefficients that straddle
    the picture edge, but beyond the ramp the padding is *constant in both axes* and its detail
    bands go to zero. This is the cheap-and-safe candidate: it is what a fill-choice fix would do,
    and it needs no bitstream change and no cross-tile dependency.
    """
    out = replicate_pad(vis, pw, ph).astype(np.float64)
    h, w = vis.shape[:2]
    flat = vis.reshape(-1, vis.shape[2]).mean(axis=0)
    if ph > h:
        t = np.clip((np.arange(ph - h) + 1) / ramp, 0.0, 1.0)[:, None, None]
        out[h:] = out[h:] * (1 - t) + flat[None, None, :] * t
    if pw > w:
        t = np.clip((np.arange(pw - w) + 1) / ramp, 0.0, 1.0)[None, :, None]
        out[:, w:] = out[:, w:] * (1 - t) + flat[None, None, :] * t
    return np.rint(out).clip(0, 255).astype(np.uint8)


def fill_flat(vis, pw, ph):
    """A single scalar in the whole padding region. Cheapest padding there is, and it puts a step
    discontinuity at the picture edge — which the wavelet then has to code, in coefficients that
    also reconstruct visible pixels. The point of measuring it is to see whether the seam costs
    more than the detail it saves."""
    out = replicate_pad(vis, pw, ph)
    h, w = vis.shape[:2]
    flat = np.rint(vis.reshape(-1, vis.shape[2]).mean(axis=0)).astype(np.uint8)
    out = out.copy()
    out[h:] = flat
    out[:, w:] = flat
    return out


FILLS = {
    "replicate": fill_replicate,          # production (pad.wgsl)
    "decay32": lambda v, pw, ph: fill_decay(v, pw, ph, 32),
    "decay8": lambda v, pw, ph: fill_decay(v, pw, ph, 8),
    "flat": fill_flat,
    "mirror": fill_mirror,
}


def part3(gnc, images, qualities, extra, tmp, rows):
    """How much of the padding tax is a fill choice, measured at native resolution."""
    print("\n=== Part 3 — fill oracle: how much of the padding is removable without a "
          "bitstream change? ===")
    print("  Rate is charged to the visible pixels and quality is scored on the visible pixels")
    print("  only, so a fill can only win by making the don't-care region cheaper to code.")
    out = {}
    for img in images:
        name = Path(img).stem
        vis = np.array(Image.open(img).convert("RGB"))
        H, W = vis.shape[:2]
        pw, ph = ceil_tile(W), ceil_tile(H)
        if (pw, ph) == (W, H):
            print(f"\n  {name}: already tile-aligned, no padding to reallocate")
            continue
        print(f"\n  {name}: {W}x{H} -> {pw}x{ph}, {1 - (W * H) / (pw * ph):.1%} of coded samples "
              f"outside the picture")
        ladders = {}
        for fname, fn in FILLS.items():
            plane = fn(vis, pw, ph)
            src = tmp / f"{name}_fill_{fname}.png"
            Image.fromarray(plane).save(src)
            pts = []
            for q in qualities:
                bs, dec = tmp / f"{name}_{fname}_q{q}.gnc", tmp / f"{name}_{fname}_q{q}.png"
                r = sh([str(gnc), "encode", "-i", str(src), "-o", str(bs), "-q", str(q), *extra])
                if r.returncode != 0:
                    continue
                r = sh([str(gnc), "decode", "-i", str(bs), "-o", str(dec)])
                if r.returncode != 0:
                    continue
                d = np.array(Image.open(dec).convert("RGB"))[:H, :W]
                vis_png = tmp / f"{name}_{fname}_q{q}_vis.png"
                Image.fromarray(d).save(vis_png)
                m = measure(vis, vis_png)
                p = {"q": q, "bytes": os.path.getsize(bs),
                     "bpp": bpp(os.path.getsize(bs), W, H), **m}
                pts.append(p)
                rows.append({"part": 3, "image": name, "variant": fname, "w": W, "h": H,
                             "coded_w": pw, "coded_h": ph, "pad_ratio": pad_ratio(W, H),
                             "arm": f"GNC abac fill={fname}", "q": q, "bytes": p["bytes"],
                             "bpp": p["bpp"], "psnr_rgb": m["psnr_rgb"], "psnr_y": m["psnr_y"],
                             "de00_mean": m["de00_mean"]})
            ladders[fname] = pts
            for p in pts:
                print(f"    {fname:<10} q{p['q']:<3d} {p['bytes']:>9d} B  {p['bpp']:.4f} bpp  "
                      f"RGB {p['psnr_rgb']:.3f} dB  Y {p['psnr_y']:.3f} dB")
        ref = ladders.get("replicate", [])
        for fname, pts in ladders.items():
            if fname == "replicate" or len(ref) < 4 or len(pts) < 4:
                continue
            (rgb, _), (y, _) = bd(ref, pts, "psnr_rgb"), bd(ref, pts, "psnr_y")
            out.setdefault(name, {})[fname] = (rgb, y)
            print(f"    {fname:<10} against production replicate: RGB {fmt(rgb)}, Y {fmt(y)}")
    if out:
        print("\n  Mean over images, BD-rate against the shipped edge-replicate fill:")
        for fname in FILLS:
            if fname == "replicate":
                continue
            vs = [v[fname][0] for v in out.values() if fname in v and v[fname][0] is not None]
            ys = [v[fname][1] for v in out.values() if fname in v and v[fname][1] is not None]
            if vs:
                print(f"    {fname:<10} RGB {np.mean(vs):+.2f}%   Y {np.mean(ys):+.2f}%  "
                      f"({len(vs)} images)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--gnc-binary", default=None)
    ap.add_argument("--qualities", default="80,85,90,94",
                    help="every rung must stay at q<=94: since RATE-2 the encoder "
                         "codes q=95..99 both ways and keeps the smaller, and a rung "
                         "that comes back bit-exact lossless has infinite quality, "
                         "which a BD-rate cannot integrate")
    ap.add_argument("--j2k-rates", default="4,6,8,12,20,40",
                    help="opj_compress -r compression ratios")
    ap.add_argument("--gnc-extra", default="--abac",
                    help="extra GNC encode flags; --abac is the arm the gap is quoted for")
    ap.add_argument("--part", default="1,2,3", help="which parts to run")
    ap.add_argument("--j2k-control", action="store_true",
                    help="Part 1's control: run JPEG 2000 on the same crop pair. Part 1 assumes "
                         "the one extra row and column of *content* is negligible; a codec that "
                         "does not pad should read ~0 between A and D, and if it does not, Part "
                         "1's rate difference is not all padding.")
    ap.add_argument("--canary", action="store_true",
                    help="only prove that the padded plane is what GNC codes, then exit")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--project-from", default=None, metavar="CSV",
                    help="skip every encode and only carry a previous run's Part 1 to the native "
                         "frame sizes")
    args = ap.parse_args()

    root = Path(sh(["git", "rev-parse", "--show-toplevel"]).stdout.strip() or ".")
    gnc = Path(args.gnc_binary) if args.gnc_binary else root / "target/release/gnc"
    if not gnc.exists():
        sys.exit(f"no GNC binary at {gnc} — cargo build --release")
    qualities = [int(q) for q in args.qualities.split(",")]
    j2k_rates = [float(r) for r in args.j2k_rates.split(",")]
    extra = args.gnc_extra.split() if args.gnc_extra else []
    parts = {int(p) for p in args.part.split(",")}
    rows = []

    if args.project_from:
        project_native(args.project_from, args.images)
        return

    with tempfile.TemporaryDirectory(prefix="intra1_pad_") as td:
        tmp = Path(td)
        if args.canary:
            print("=== Canary — is the padded plane really what GNC codes? ===")
            for img in args.images:
                c = canary(gnc, img, tmp, qualities[len(qualities) // 2], extra)
                if c is None:
                    print(f"  {Path(img).stem}: already tile-aligned, nothing to check")
                    continue
                verdict = "IDENTICAL" if c["identical"] else "DIFFER — mechanism is wrong"
                print(f"  {Path(img).stem}: crop {c['crop']} -> {c['padded']}  "
                      f"{c['crop_bytes']} B vs pre-padded {c['prepadded_bytes']} B  {verdict}")
            return

        if 1 in parts:
            part1(gnc, args.images, qualities, extra, tmp, rows)
            if args.j2k_control:
                part1_control(args.images, j2k_rates, tmp, rows)
        if 2 in parts:
            part2(gnc, args.images, qualities, extra, j2k_rates, tmp, rows)
        if 3 in parts:
            part3(gnc, args.images, qualities, extra, tmp, rows)

    print("\n=== Native padding fractions, for reference ===")
    print(f"  {'image':<20} {'picture':>12} {'coded':>12} {'coded/visible':>14} {'outside':>9}")
    for img in args.images:
        W, H = Image.open(img).size
        pw, ph = ceil_tile(W), ceil_tile(H)
        print(f"  {Path(img).stem:<20} {f'{W}x{H}':>12} {f'{pw}x{ph}':>12} "
              f"{pad_ratio(W, H):>14.4f} {1 - (W * H) / (pw * ph):>8.1%}")

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")
        if 1 in parts:
            # The projection is the number the harness exists to produce, so a Part 1 run should
            # not need a second invocation to see it.
            project_native(args.csv, args.images)


if __name__ == "__main__":
    main()
