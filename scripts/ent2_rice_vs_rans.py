#!/usr/bin/env python3
"""ENT-2 — Rice against rANS on one commit, with the coder read out of the bitstream.

Why this exists: the README carried "Rice 4.01 bpp vs rANS 4.22 bpp @ q=75" until 2026-09-06.
That pair was taken across three changes (uniform weights, 5 wavelet levels, GP17 Rice-coded
length tables) and q=75 has since moved to 4.53 bpp, so the comparison never held on any single
commit. The 2026-09-06 re-sweep fixed that for q=25-70; this harness reproduces it on today's
main and extends it **below** q=25, which is the range where rANS is the codec's own default
(`quality_preset`: rANS at q<=20, Rice above) and where nobody has ever compared the two on one
commit after ENT-1 landed.

Three things it does that the earlier sweeps did not:

  * **The coder is read out of the encoded file, not taken from the flag.** BACKLOG's BUG-9 entry
    records `--rans` as "a no-op flag kept for backward compatibility"; if that were true every
    number in this table would be Rice against Rice. `entropy_type` is a u32 in the frame header
    (0 = rANS, 2 = per-subband rANS, 3 = Rice), so `bitstream_coder()` walks the header and reads
    it. Every row is asserted against the coder that was requested, and a mismatch is reported as
    a mismatch rather than silently averaged in.
  * **Equal q is checked to be equal picture, not assumed.** Entropy coding is lossless and both
    coders quantise identically, so the same q must decode to the same image and only the rate may
    differ. The harness decodes both arms and compares PSNR; a nonzero delta means the premise is
    broken and the bpp comparison is void.
  * **The q>=80 crash is a row, not an omission.** rANS overflows its fixed 4 KB per-stream buffer
    from q=80 up (BUG-9). The stderr line is captured and printed, because "rANS cannot reach the
    contribution operating point" is the load-bearing half of the recommendation.

Metrics: Y-PSNR in YCoCg-R (the plane GNC codes, and the metric that leads above q=85 per
CLAUDE.md) plus RGB PSNR as the cross-check. VMAF is deliberately absent — it is luma-only and
saturated at the top of this ladder, and nothing here is a perceptual question: at equal q the two
arms decode to the same picture, so quality is a control, not a result.

Usage:
    scripts/ent2_rice_vs_rans.py --images frames_pinned/*.png [--qualities 5,10,...] [--csv out.csv]
"""

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# entropy_type as written by src/format.rs::serialize_compressed. Type 5 (abac) arrived with GP18
# on 2026-09-07; the header layout is unchanged from GP17, only the magic differs, so the parser
# below reads both. Listing it matters because an unknown type raises "header layout moved?", which
# would send the next reader hunting for a format bug that is not there.
CODER_NAMES = {0: "rANS", 1: "bitplane", 2: "rANS-subband", 3: "Rice", 4: "Huffman", 5: "abac"}
CODER_FAMILY = {0: "rans", 1: "bitplane", 2: "rans", 3: "rice", 4: "huffman", 5: "abac"}


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def repo_root():
    out = sh(["git", "rev-parse", "--show-toplevel"])
    return Path(out.stdout.strip()) if out.returncode == 0 else Path.cwd()


# ---------------------------------------------------------------------------
# the canary: which coder is actually in the file
# ---------------------------------------------------------------------------

def bitstream_coder(path):
    """Walk the GP1x frame header and return (entropy_type, name).

    Mirrors `deserialize_compressed_validated` in src/format.rs up to the entropy_type word. It
    parses rather than pattern-matches, so it fails loudly if the layout moves — which is the
    point: this is the only figure in the table the encoder does not get to self-report.
    """
    d = Path(path).read_bytes()
    magic = d[0:4].decode("ascii", "replace")
    if not magic.startswith("GP") or not magic[2:].isdigit():
        raise RuntimeError(f"{path}: unexpected magic {magic!r}")
    gen = int(magic[2:])
    if gen < 13:
        raise RuntimeError(f"{path}: generation {gen} predates the chroma byte; parser is GP13+")

    p = 32                      # width, height, bit_depth, tile_size, qstep, dead_zone, levels
    p += 2                      # wavelet_type, transform_type
    p += 1                      # per_subband_entropy   (gen >= 9)
    p += 1                      # chroma_format         (gen >= 13)

    p += 4                      # subband_weights.ll
    num_detail = struct.unpack_from("<I", d, p)[0]
    p += 4 + 12 * num_detail    # per-level [LH, HL, HH]
    p += 4                      # chroma_weight

    cfl_flag = d[p]
    p += 1
    if cfl_flag:
        nsb, num_cfl_tiles = struct.unpack_from("<II", d, p)
        p += 8
        p += 2 * (2 * num_cfl_tiles * nsb)   # i16 alphas

    p += 4                      # aq_flag
    p += 4                      # aq_strength
    wm_len = struct.unpack_from("<I", d, p)[0]
    p += 4 + 4 * wm_len         # f32 weight map

    if d[p]:                    # intra_flag
        p += 1
        p += 4                  # exact block count
        packed_len = struct.unpack_from("<I", d, p)[0]
        p += 4 + packed_len
    else:
        p += 1

    frame_type = d[p]
    p += 1
    if frame_type != 0:
        raise RuntimeError(f"{path}: frame type {frame_type} has a motion field; stills only")

    etype = struct.unpack_from("<I", d, p)[0]
    if etype not in CODER_NAMES:
        raise RuntimeError(f"{path}: entropy_type {etype} at offset {p} — header layout moved?")
    return etype, CODER_NAMES[etype]


# ---------------------------------------------------------------------------
# metrics — one implementation for both arms
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
    dec = np.array(Image.open(decoded_png).convert("RGB"))
    if dec.shape != orig_rgb.shape:
        raise RuntimeError(f"{decoded_png}: {dec.shape} != original {orig_rgb.shape}")
    return {
        "psnr_rgb": psnr(orig_rgb.astype(np.float64), dec.astype(np.float64)),
        "psnr_y": psnr(ycocg_r_y(orig_rgb), ycocg_r_y(dec)),
    }


# ---------------------------------------------------------------------------
# one (image, q, coder) point
# ---------------------------------------------------------------------------

def failure_line(r):
    """The line that says *why*, not the last line printed.

    A Rust panic ends with "note: run with `RUST_BACKTRACE=1`...", so taking the last non-empty
    stderr line reports the note and throws away the message — which is how the first run of this
    harness recorded four identical useless rows for the q>=80 overflow.
    """
    lines = [ln.strip() for ln in (r.stderr + "\n" + r.stdout).splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        # A Rust panic header ("thread 'main' panicked at file:line:col:") carries no reason —
        # the reason is the line after it. Matching the header and stripping it leaves the empty
        # string, which is exactly how this function returned nothing the first time round.
        if re.match(r"thread '[^']*'.*panicked at ", ln):
            return lines[i + 1] if i + 1 < len(lines) else ln
        if re.search(r"overflow|error", ln, re.I):
            return ln
    return lines[-1] if lines else "no output"


def encode_point(gnc, img, q, coder, orig_rgb, tmp):
    """Encode, verify the coder from the bitstream, decode, measure. None if the arm failed."""
    stem = f"{Path(img).stem}_q{q}_{coder}"
    bs, png = tmp / f"{stem}.gnc", tmp / f"{stem}.png"
    flag = "--rice" if coder == "rice" else "--rans"
    r = sh([str(gnc), "encode", "-i", str(img), "-o", str(bs), "-q", str(q), flag])
    if r.returncode != 0 or not os.path.exists(bs):
        return {"failed": failure_line(r)}

    etype, ename = bitstream_coder(bs)
    r = sh([str(gnc), "decode", "-i", str(bs), "-o", str(png)])
    if r.returncode != 0 or not os.path.exists(png):
        return {"failed": "decode: " + failure_line(r)}

    h, w = orig_rgb.shape[:2]
    size = os.path.getsize(bs)
    return {
        "bytes": size,
        "bpp": size * 8 / (w * h),
        "coder_in_file": ename,
        "coder_matches": CODER_FAMILY[etype] == coder,
        **measure(orig_rgb, png),
    }


def run_image(gnc, img, qualities, tmp):
    orig = np.array(Image.open(img).convert("RGB"))
    h, w = orig.shape[:2]
    print(f"\n=== {Path(img).name}  {w}x{h} ===")
    print(f"  {'q':>4}  {'Rice bpp':>9} {'rANS bpp':>9} {'rANS vs Rice':>13}  "
          f"{'Rice Y-PSNR':>11} {'rANS Y-PSNR':>11}  {'in file':>13}")
    rows = []
    for q in qualities:
        a = encode_point(gnc, img, q, "rice", orig, tmp)
        b = encode_point(gnc, img, q, "rans", orig, tmp)
        row = {"image": Path(img).name, "q": q}
        for name, res in (("rice", a), ("rans", b)):
            if "failed" in res:
                row[f"{name}_failed"] = res["failed"]
            else:
                row[f"{name}_bpp"] = res["bpp"]
                row[f"{name}_bytes"] = res["bytes"]
                row[f"{name}_psnr_y"] = res["psnr_y"]
                row[f"{name}_psnr_rgb"] = res["psnr_rgb"]
                row[f"{name}_coder_in_file"] = res["coder_in_file"]
                row[f"{name}_coder_matches"] = res["coder_matches"]
        if "rice_bpp" in row and "rans_bpp" in row:
            row["delta_pct"] = 100.0 * (row["rans_bpp"] - row["rice_bpp"]) / row["rice_bpp"]
            row["psnr_y_delta"] = row["rans_psnr_y"] - row["rice_psnr_y"]
            flag = "" if abs(row["psnr_y_delta"]) < 0.005 else f"  !! dY {row['psnr_y_delta']:+.3f}"
            mism = "" if (row["rice_coder_matches"] and row["rans_coder_matches"]) else "  !! MISMATCH"
            print(f"  {q:>4}  {row['rice_bpp']:>9.4f} {row['rans_bpp']:>9.4f} "
                  f"{row['delta_pct']:>+12.2f}%  {row['rice_psnr_y']:>11.2f} "
                  f"{row['rans_psnr_y']:>11.2f}  "
                  f"{row['rice_coder_in_file']}/{row['rans_coder_in_file']:>4}{flag}{mism}")
        else:
            for name in ("rice", "rans"):
                if f"{name}_failed" in row:
                    print(f"  {q:>4}  {name} FAILED: {row[f'{name}_failed']}")
        rows.append(row)
    return rows


def main():
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--gnc-binary", default=str(root / "target/release/gnc"))
    ap.add_argument("--qualities", default="5,10,15,20,25,40,55,70,80,90",
                    help="q ladder; spans the rANS default range (<=20), the 2026-09-06 re-sweep "
                         "points (25-70) and the overflow threshold (80,90)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    qualities = [int(q) for q in args.qualities.split(",")]
    if not os.path.exists(args.gnc_binary):
        sys.exit(f"GNC binary not found: {args.gnc_binary} (cargo build --release)")

    all_rows = []
    with tempfile.TemporaryDirectory(prefix="ent2_") as td:
        for img in args.images:
            all_rows += run_image(args.gnc_binary, img, qualities, Path(td))

    print("\n=== rANS against Rice, bpp, negative means rANS is smaller ===")
    images = [Path(i).name for i in args.images]
    print(f"  {'q':>4}  " + "".join(f"{i.replace('.png',''):>18}" for i in images) + f"{'mean':>10}")
    for q in qualities:
        cells, vals = [], []
        for im in images:
            r = next((r for r in all_rows if r["image"] == im and r["q"] == q), None)
            if r and "delta_pct" in r:
                cells.append(f"{r['delta_pct']:>17.1f}%")
                vals.append(r["delta_pct"])
            else:
                cells.append(f"{'crash':>18}")
        mean = f"{np.mean(vals):>9.1f}%" if vals else f"{'—':>10}"
        print(f"  {q:>4}  " + "".join(cells) + mean)

    bad = [r for r in all_rows if r.get("psnr_y_delta") is not None
           and abs(r["psnr_y_delta"]) >= 0.005]
    mismatched = [r for r in all_rows
                  if r.get("rice_coder_matches") is False or r.get("rans_coder_matches") is False]
    print(f"\nquality control: {len(bad)} of {len(all_rows)} points differ in Y-PSNR between "
          f"coders (must be 0 — equal q is equal picture)")
    print(f"coder canary:    {len(mismatched)} points where the bitstream's coder != the flag "
          f"(must be 0)")
    for r in mismatched:
        print(f"  {r['image']} q={r['q']}: rice->{r.get('rice_coder_in_file')} "
              f"rans->{r.get('rans_coder_in_file')}")

    if args.csv:
        import csv as csvmod
        keys = sorted({k for r in all_rows for k in r})
        with open(args.csv, "w", newline="") as f:
            wtr = csvmod.DictWriter(f, fieldnames=keys)
            wtr.writeheader()
            wtr.writerows(all_rows)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
