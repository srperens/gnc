#!/usr/bin/env python3
"""Measure GNC across the GPUs a machine has, and against its fixed-function encoder.

This is the harness for two open questions that no amount of code review can settle
(see docs/GPU_TIER_TEST.md for what each one proves and what it does not):

  CANARY-1  Does encode time move between GPU tiers at all? If a discrete GPU is no
            faster than an integrated one, the pipeline is not GPU-bound and every
            throughput claim in the project rests on nothing.

  MEAS-5    How does aggregate throughput scale with concurrent encode processes, and
            how does that compare with the machine's fixed-function encoder (NVENC,
            QSV, VideoToolbox) under the same load?

Stdlib only, and it runs on Windows and macOS alike.

Timing rules, taken from COORDINATION.md — an idle machine is necessary and not
sufficient. Every error source (clock ramp, another process, scheduler noise) only
makes a reading slower, so the *minimum* over repeats is the least contaminated
estimate. `median / best` is printed beside it as a free settled-or-not diagnostic:
near 1.0 means the number is quotable, well above means keep only the ratios.

Examples
--------
    python scripts/gpu_tier_bench.py --list
    python scripts/gpu_tier_bench.py --tier   -i frames/bbb_1080p.png
    python scripts/gpu_tier_bench.py --density -i clip.y4m --adapter nvidia
    python scripts/gpu_tier_bench.py --hwenc  -i clip.y4m --encoder h264_nvenc
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# "  Apple M1 Pro [Metal, IntegratedGpu]"
ADAPTER_LINE = re.compile(r"^\s{2}(?P<name>.+?)\s+\[(?P<backend>\w+),\s*(?P<kind>\w+)\]\s*$")


# ---------------------------------------------------------------- process helpers


def default_binary() -> Path:
    exe = "gnc.exe" if platform.system() == "Windows" else "gnc"
    return REPO / "target" / "release" / exe


def run(cmd: list[str], env: dict[str, str] | None = None, timeout: int = 3600):
    """Run to completion. Returns (returncode, stdout, stderr, wall_seconds)."""
    merged = dict(os.environ)
    if env:
        merged.update(env)
    start = time.perf_counter()
    proc = subprocess.run(
        cmd, env=merged, capture_output=True, text=True, timeout=timeout
    )
    return proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - start


def run_concurrent(cmds: list[list[str]], env: dict[str, str] | None = None):
    """Launch every command at once; return (wall_seconds, [returncodes], [stderr]).

    Wall time runs from the first launch to the last exit, which is what an aggregate
    throughput figure has to be measured over.
    """
    merged = dict(os.environ)
    if env:
        merged.update(env)
    start = time.perf_counter()
    procs = [
        subprocess.Popen(c, env=merged, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for c in cmds
    ]
    errs = []
    for p in procs:
        _, err = p.communicate()
        errs.append(err)
    return time.perf_counter() - start, [p.returncode for p in procs], errs


def settled(values: list[float]) -> tuple[float, float]:
    """(best, median/best). Ratio near 1.0 means the measurement settled."""
    best = min(values)
    return best, (statistics.median(values) / best if best > 0 else float("nan"))


# ---------------------------------------------------------------- adapters


def list_adapters(binary: Path) -> list[dict]:
    code, out, err, _ = run([str(binary), "gpu-info"])
    if code != 0:
        sys.exit(f"gpu-info failed:\n{err}")
    adapters = []
    for line in out.splitlines():
        m = ADAPTER_LINE.match(line)
        if m:
            adapters.append(m.groupdict())
    return adapters


def selector_for(adapter: dict) -> str:
    """A GNC_GPU_ADAPTER substring that picks this adapter and, ideally, only it.

    Vendor word first — 'nvidia', 'intel' — since that is what distinguishes the
    tiers on the machines this test targets. Falls back to the full name.
    """
    name = adapter["name"].lower()
    for vendor in ("nvidia", "intel", "amd", "radeon", "apple", "llvmpipe", "microsoft"):
        if vendor in name:
            return vendor
    return adapter["name"]


# ---------------------------------------------------------------- CANARY-1


def tier_bench(binary: Path, image: Path, adapters: list[dict], repeats: int,
               iterations: int, quality: int) -> list[dict]:
    """Single-frame encode/decode on each adapter in turn, best of `repeats` processes.

    Uses `gnc benchmark`, which loads the image once and then loops the GPU work, so
    the figure is close to GPU-bound: the PNG decode is paid once per process and
    amortised over `iterations`.
    """
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "bench.csv"
        for adapter in adapters:
            sel = selector_for(adapter)
            enc, dec, bpp, psnr = [], [], None, None
            failed = None
            for _ in range(repeats):
                code, _, err, _ = run(
                    [str(binary), "benchmark", "-i", str(image), "-n", str(iterations),
                     "-q", str(quality), "--csv", str(csv_path)],
                    env={"GNC_GPU_ADAPTER": sel},
                )
                if code != 0:
                    failed = err.strip().splitlines()[-1] if err.strip() else f"exit {code}"
                    break
                with open(csv_path, newline="") as fh:
                    row = list(csv.DictReader(fh))[-1]
                enc.append(float(row["encode_ms"]))
                dec.append(float(row["decode_ms"]))
                bpp, psnr = float(row["bpp"]), float(row["psnr_db"])

            if failed or not enc:
                results.append({"adapter": adapter, "selector": sel, "error": failed})
                continue
            best_enc, ratio_enc = settled(enc)
            best_dec, ratio_dec = settled(dec)
            results.append({
                "adapter": adapter, "selector": sel,
                "encode_ms": best_enc, "encode_settle": ratio_enc,
                "decode_ms": best_dec, "decode_settle": ratio_dec,
                "encode_fps": 1000.0 / best_enc, "decode_fps": 1000.0 / best_dec,
                "bpp": bpp, "psnr_db": psnr,
            })
    return results


# ---------------------------------------------------------------- MEAS-5


def density(binary: Path, clip: Path, frames: int, quality: int, ki: int,
            levels: list[int], adapter: str | None) -> list[dict]:
    """Aggregate throughput with N concurrent GNC encodes of the same clip."""
    env = {"GNC_GPU_ADAPTER": adapter} if adapter else {}
    cmd = [str(binary), "benchmark-sequence", "-i", str(clip),
           "-n", str(frames), "-q", str(quality), "-k", str(ki), "--rice"]
    rows = []
    for n in levels:
        wall, codes, errs = run_concurrent([cmd] * n, env=env)
        ok = sum(1 for c in codes if c == 0)
        rows.append({
            "instances": n,
            "completed": ok,
            "wall_s": wall,
            "aggregate_fps": (ok * frames / wall) if wall > 0 and ok else 0.0,
            "error": None if ok == n else (errs[codes.index(next(c for c in codes if c != 0))]
                                           .strip().splitlines() or [""])[-1],
        })
    return rows


def hwenc_density(ffmpeg: str, clip: Path, encoder: str, preset: str, qp: int,
                  frames: int, levels: list[int]) -> list[dict]:
    """The same sweep through the machine's fixed-function encoder.

    A failure at some N is a result, not an error: a driver session cap is exactly
    what the positioning argument claims exists.
    """
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for n in levels:
            cmds = []
            for i in range(n):
                out = str(Path(tmp) / f"out_{n}_{i}.264")
                cmd = [ffmpeg, "-y", "-v", "error", "-i", str(clip),
                       "-frames:v", str(frames), "-c:v", encoder]
                if "nvenc" in encoder:
                    cmd += ["-preset", preset, "-rc", "constqp", "-qp", str(qp)]
                elif "qsv" in encoder:
                    cmd += ["-preset", preset, "-global_quality", str(qp)]
                elif "videotoolbox" in encoder:
                    cmd += ["-q:v", str(qp)]
                cmds.append(cmd + [out])
            wall, codes, errs = run_concurrent(cmds)
            ok = sum(1 for c in codes if c == 0)
            first_err = ""
            if ok != n:
                idx = codes.index(next(c for c in codes if c != 0))
                first_err = (errs[idx].strip().splitlines() or [""])[-1]
            rows.append({
                "instances": n,
                "completed": ok,
                "wall_s": wall,
                "aggregate_fps": (ok * frames / wall) if wall > 0 and ok else 0.0,
                "error": first_err or None,
            })
    return rows


# ---------------------------------------------------------------- reporting


def print_tier(rows: list[dict]) -> None:
    print("\n## CANARY-1 — encode time across GPU tiers")
    print("\n| GPU | backend | encode ms | fps | settle | decode ms | fps | settle |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        a = r["adapter"]
        if r.get("error"):
            print(f"| {a['name']} | {a['backend']} | — | — | — | — | — | {r['error']} |")
            continue
        print(f"| {a['name']} | {a['backend']} | {r['encode_ms']:.2f} | {r['encode_fps']:.1f} "
              f"| {r['encode_settle']:.2f} | {r['decode_ms']:.2f} | {r['decode_fps']:.1f} "
              f"| {r['decode_settle']:.2f} |")

    good = [r for r in rows if not r.get("error")]
    if len(good) >= 2:
        fastest = min(good, key=lambda r: r["encode_ms"])
        slowest = max(good, key=lambda r: r["encode_ms"])
        spread = slowest["encode_ms"] / fastest["encode_ms"]
        print(f"\nSpread across tiers: **{spread:.2f}x** "
              f"({slowest['adapter']['name']} vs {fastest['adapter']['name']}).")
        if spread < 1.15:
            print("\n**This is the CANARY-1 failure signal.** Encode time barely moved between "
                  "GPUs of different capability, so the measurement is not GPU-bound — it is "
                  "dominated by I/O, driver overhead or CPU work. Fix that before quoting any "
                  "throughput number from this machine.")
    if any(r.get("encode_settle", 1.0) > 1.15 for r in good):
        print("\n**At least one reading did not settle** (median/best > 1.15). Close other work, "
              "keep the machine on mains power, and raise --repeats before quoting absolute "
              "figures; ratios within a run are still usable.")


def print_density(title: str, rows: list[dict], note: str = "") -> None:
    print(f"\n## {title}")
    print("\n| instances | completed | wall s | aggregate fps | scaling vs N=1 |")
    print("|---|---|---|---|---|")
    base = next((r["aggregate_fps"] for r in rows if r["instances"] == 1 and r["aggregate_fps"]), None)
    for r in rows:
        scale = f"{r['aggregate_fps'] / base:.2f}x" if base else "—"
        note_col = f" ({r['error']})" if r.get("error") else ""
        print(f"| {r['instances']} | {r['completed']}/{r['instances']}{note_col} | {r['wall_s']:.1f} "
              f"| {r['aggregate_fps']:.2f} | {scale} |")
    if note:
        print(f"\n{note}")
    failed = [r for r in rows if r["completed"] < r["instances"]]
    if failed:
        print(f"\n**Stopped completing at N={failed[0]['instances']}.** For a fixed-function "
              "encoder that is the session cap, and it is the number the positioning argument "
              "turns on — record it. For GNC it is a bug or an out-of-memory condition.")


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", type=Path, default=default_binary())
    ap.add_argument("-i", "--input", type=Path, help="PNG frame for --tier, Y4M clip otherwise")
    ap.add_argument("--list", action="store_true", help="list adapters and exit")
    ap.add_argument("--tier", action="store_true", help="CANARY-1: encode time per GPU")
    ap.add_argument("--density", action="store_true", help="MEAS-5: concurrent GNC encodes")
    ap.add_argument("--hwenc", action="store_true", help="MEAS-5: the same sweep through NVENC/QSV")
    ap.add_argument("--all", action="store_true", help="tier, then density, then hwenc")
    ap.add_argument("--adapter", help="GNC_GPU_ADAPTER substring for --density")
    ap.add_argument("--encoder", default="h264_nvenc", help="ffmpeg hardware encoder for --hwenc")
    ap.add_argument("--preset", default="p7", help="hardware encoder preset (p7 = slowest/best)")
    ap.add_argument("--qp", type=int, default=18, help="hardware encoder constant QP")
    ap.add_argument("--quality", type=int, default=90, help="GNC quality (90 = contribution)")
    ap.add_argument("--keyframe-interval", type=int, default=9)
    ap.add_argument("--frames", type=int, default=120,
                    help="frames per encode. Process startup (GPU init, shader compilation) is "
                         "a fixed cost, so short clips flatter whichever encoder starts faster")
    ap.add_argument("--iterations", type=int, default=24, help="GPU loops inside one --tier process")
    ap.add_argument("--repeats", type=int, default=5, help="processes per --tier point; best wins")
    ap.add_argument("--levels", default="1,2,4,8", help="concurrency levels to sweep")
    ap.add_argument("--json", type=Path, help="write raw results here")
    args = ap.parse_args()

    if not args.binary.exists():
        sys.exit(f"No GNC binary at {args.binary} — run `cargo build --release` first.")

    adapters = list_adapters(args.binary)
    if args.list or not (args.tier or args.density or args.hwenc or args.all):
        print(f"{len(adapters)} adapter(s) on {platform.system()} {platform.release()}:")
        for a in adapters:
            print(f"  {a['name']} [{a['backend']}, {a['kind']}]  "
                  f"→ GNC_GPU_ADAPTER={selector_for(a)}")
        if not args.list:
            print("\nPick a mode: --tier, --density, --hwenc or --all. See --help.")
        return

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if (args.density or args.hwenc or args.all) and args.frames < 60:
        print(f"\n**--frames {args.frames} is short for a density measurement.** Per-process "
              "startup is a fixed cost paid once per instance, and GNC pays more of it than a "
              "fixed-function encoder does, so a short clip understates GNC. Use 120 or more "
              "for anything quotable.")
    out: dict = {
        "machine": {"system": platform.system(), "release": platform.release(),
                    "processor": platform.processor()},
        "adapters": adapters,
        "params": vars(args) | {"binary": str(args.binary), "input": str(args.input),
                                "json": str(args.json)},
    }

    print(f"# GNC GPU tier report — {platform.system()} {platform.release()}")
    print(f"\n{len(adapters)} adapter(s): " + "; ".join(
        f"{a['name']} [{a['backend']}]" for a in adapters))

    if args.tier or args.all:
        if not args.input:
            sys.exit("--tier needs -i <png frame>")
        # Deduplicate: one row per selector, so a card exposed on two backends is not
        # measured twice under two names.
        seen, unique = set(), []
        for a in adapters:
            key = selector_for(a)
            if key not in seen:
                seen.add(key)
                unique.append(a)
        out["tier"] = tier_bench(args.binary, args.input, unique,
                                 args.repeats, args.iterations, args.quality)
        print_tier(out["tier"])

    if args.density or args.all:
        if not args.input:
            sys.exit("--density needs -i <y4m clip>")
        out["density"] = density(args.binary, args.input, args.frames, args.quality,
                                 args.keyframe_interval, levels, args.adapter)
        print_density(f"MEAS-5 — GNC, {args.frames} frames at q={args.quality}"
                      + (f", adapter={args.adapter}" if args.adapter else ""),
                      out["density"],
                      "Concurrency converts idle GPU into useful GPU; it does not create GPU. "
                      "Sub-linear scaling is expected — the question is how far it goes before "
                      "it flattens.")

    if args.hwenc or args.all:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print("\n(skipping --hwenc: no ffmpeg on PATH)")
        elif not args.input:
            sys.exit("--hwenc needs -i <y4m clip>")
        else:
            out["hwenc"] = hwenc_density(ffmpeg, args.input, args.encoder, args.preset,
                                         args.qp, args.frames, levels)
            uses_preset = "nvenc" in args.encoder or "qsv" in args.encoder
            label = (f"{args.encoder} preset {args.preset}, qp {args.qp}" if uses_preset
                     else f"{args.encoder}, q {args.qp}")
            print_density(f"MEAS-5 — {label}",
                          out["hwenc"],
                          "**Not quality-matched to the GNC rows.** Bitrate and distortion are "
                          "not compared here; scripts/meas1_vs_h264.py is the harness for that. "
                          "Read these rows as a session-count and scaling comparison only.")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nRaw results → {args.json}")


if __name__ == "__main__":
    main()
