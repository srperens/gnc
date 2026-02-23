#!/usr/bin/env python3
"""Benchmark JPEG, JPEG 2000, and GNC on the same test images.

Outputs CSV with: codec, image, quality_param, psnr_db, ssim, bpp, compressed_bytes, encode_ms
"""

import csv
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_image(path):
    """Load image as numpy array (H, W, 3) uint8."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def compute_metrics(original, decoded):
    """Return (psnr_db, ssim) for two uint8 RGB images."""
    psnr = peak_signal_noise_ratio(original, decoded, data_range=255)
    ssim = structural_similarity(original, decoded, channel_axis=2, data_range=255)
    return float(psnr), float(ssim)


def file_bpp(file_path, width, height):
    """Compute bits per pixel."""
    size_bytes = os.path.getsize(file_path)
    return (size_bytes * 8) / (width * height)


def png_to_ppm(img_path):
    """Convert PNG to PPM for cjpeg input. Returns temp file path."""
    img = Image.open(img_path).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
        ppm_path = f.name
    img.save(ppm_path)
    return ppm_path


def benchmark_jpeg(img_path, original, qualities):
    """Benchmark libjpeg-turbo at various quality levels."""
    h, w = original.shape[:2]
    results = []
    ppm_path = png_to_ppm(img_path)

    for q in qualities:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            jpg_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
            dec_path = f.name

        try:
            # Encode
            t0 = time.perf_counter()
            subprocess.run(
                ["cjpeg", "-quality", str(q), "-optimize", "-outfile", jpg_path, ppm_path],
                check=True, capture_output=True,
            )
            encode_ms = (time.perf_counter() - t0) * 1000

            # Decode
            subprocess.run(
                ["djpeg", "-ppm", "-outfile", dec_path, jpg_path],
                check=True, capture_output=True,
            )

            decoded = load_image(dec_path)
            psnr, ssim = compute_metrics(original, decoded)
            bpp = file_bpp(jpg_path, w, h)
            compressed_bytes = os.path.getsize(jpg_path)

            results.append({
                "codec": "jpeg",
                "quality_param": f"q{q}",
                "psnr_db": round(psnr, 2),
                "ssim": round(ssim, 4),
                "bpp": round(bpp, 3),
                "compressed_bytes": compressed_bytes,
                "encode_ms": round(encode_ms, 1),
            })
            print(f"  JPEG q={q}: {psnr:.2f} dB, {bpp:.3f} bpp")
        finally:
            for p in [jpg_path, dec_path]:
                if os.path.exists(p):
                    os.unlink(p)

    os.unlink(ppm_path)
    return results


def benchmark_jpeg2000(img_path, original, rates):
    """Benchmark OpenJPEG JPEG 2000 at various compression rates.

    opj_compress -r N means target compression ratio N:1.
    """
    h, w = original.shape[:2]
    results = []

    for rate in rates:
        with tempfile.NamedTemporaryFile(suffix=".j2k", delete=False) as f:
            j2k_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            dec_path = f.name

        try:
            # Encode (lossy with CDF 9/7 by default at given rate)
            t0 = time.perf_counter()
            subprocess.run(
                ["opj_compress", "-i", img_path, "-o", j2k_path, "-r", str(rate)],
                check=True, capture_output=True,
            )
            encode_ms = (time.perf_counter() - t0) * 1000

            # Decode
            subprocess.run(
                ["opj_decompress", "-i", j2k_path, "-o", dec_path],
                check=True, capture_output=True,
            )

            decoded = load_image(dec_path)
            psnr, ssim = compute_metrics(original, decoded)
            bpp = file_bpp(j2k_path, w, h)
            compressed_bytes = os.path.getsize(j2k_path)

            results.append({
                "codec": "jpeg2000",
                "quality_param": f"r{rate}",
                "psnr_db": round(psnr, 2),
                "ssim": round(ssim, 4),
                "bpp": round(bpp, 3),
                "compressed_bytes": compressed_bytes,
                "encode_ms": round(encode_ms, 1),
            })
            print(f"  JP2 r={rate}: {psnr:.2f} dB, {bpp:.3f} bpp")
        finally:
            for p in [j2k_path, dec_path]:
                if os.path.exists(p):
                    os.unlink(p)

    return results


def main():
    images = [
        "test_material/frames/bbb_1080p.png",
        "test_material/frames/blue_sky_1080p.png",
        "test_material/frames/touchdown_1080p.png",
    ]

    # JPEG quality levels (spanning the useful range)
    jpeg_qualities = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 98]

    # JPEG 2000 compression ratios (higher = more compression)
    # ratio = uncompressed_size / compressed_size
    # 24 bpp raw / ratio = target bpp, so ratio 5 = ~4.8 bpp, ratio 50 = ~0.48 bpp
    j2k_rates = [2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50, 80, 100]

    all_results = []

    for img_path in images:
        if not os.path.exists(img_path):
            print(f"Skipping {img_path} (not found)")
            continue

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        original = load_image(img_path)
        h, w = original.shape[:2]
        print(f"\n=== {img_name} ({w}x{h}) ===")

        print("JPEG (libjpeg-turbo):")
        jpeg_results = benchmark_jpeg(img_path, original, jpeg_qualities)
        for r in jpeg_results:
            r["image"] = img_name
            r["width"] = w
            r["height"] = h
        all_results.extend(jpeg_results)

        print("JPEG 2000 (OpenJPEG):")
        j2k_results = benchmark_jpeg2000(img_path, original, j2k_rates)
        for r in j2k_results:
            r["image"] = img_name
            r["width"] = w
            r["height"] = h
        all_results.extend(j2k_results)

    # Write CSV
    out_path = "results/codec_comparison.csv"
    os.makedirs("results", exist_ok=True)
    fieldnames = ["codec", "image", "width", "height", "quality_param", "psnr_db", "ssim", "bpp", "compressed_bytes", "encode_ms"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
