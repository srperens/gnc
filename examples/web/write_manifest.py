#!/usr/bin/env python3
"""Write demos.json for the web player from whatever generate_demos.sh actually produced.

The player used to carry a hardcoded list, which is how it ended up offering twenty files none of
which existed. Deriving the manifest from the directory means the two cannot disagree.
"""
import glob, json, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

# name -> (group, label, description). Frame counts and fps mirror generate_demos.sh.
CATALOG = [
    ("watch_rush_hour.gnv",       "Watch — full clips at q=75", "Rush Hour",        "200f 25fps 1080p — city traffic, slow pan"),
    ("watch_pedestrian_area.gnv", "Watch — full clips at q=75", "Pedestrian Area",  "200f 25fps 1080p — walking crowd"),
    ("watch_old_town_cross.gnv",  "Watch — full clips at q=75", "Old Town Cross",   "200f 50fps 1080p — urban pan"),
    ("watch_bbb_2min.gnv",        "Watch — full clips at q=75", "Big Buck Bunny",   "600f 30fps 1080p — 20 s, varied scenes"),
    ("watch_ducks_q50.gnv",       "Watch — full clips at q=75", "Ducks Take Off (q=50)", "300f 50fps 1080p — water and feathers; q=50 because q=75 is ~6 bpp here"),

    ("range_q25.gnv",             "Quality range (24f, same clip)", "q=25",  "heavy compression, visible artefacts"),
    ("range_q50.gnv",             "Quality range (24f, same clip)", "q=50",  "balanced"),
    ("range_q75.gnv",             "Quality range (24f, same clip)", "q=75",  "the default preset"),
    ("range_q92.gnv",             "Quality range (24f, same clip)", "q=92",  "contribution quality — this range was unreachable before 2026-09-06"),
    ("range_q100_lossless.gnv",   "Quality range (24f, same clip)", "q=100 lossless", "bit-exact; verified zero error against the source"),

    ("temporal_ip.gnv",           "Temporal mode (24f, same clip and q)", "I+P motion vectors", "GNV1 — predict from the previous frame, code the difference"),
    ("temporal_haar.gnv2",        "Temporal mode (24f, same clip and q)", "Haar temporal wavelet", "GNV2 — transform across frame pairs, no motion vectors. +3.2 dB PSNR here"),

    ("watch_crowd_run.gnv",       "Watch — full clips at q=75", "Crowd Run",        "24f 50fps 1080p — dense motion, the hardest clip here"),
    ("watch_bbb_extended.gnv",    "Watch — full clips at q=75", "Big Buck Bunny",   "24f 30fps 1080p — animation"),

    ("chroma_444.gnv",            "Chroma format (24f, same clip and q)", "4:4:4", "full chroma resolution"),
    ("chroma_422.gnv",            "Chroma format (24f, same clip and q)", "4:2:2", "half horizontal chroma"),
    ("chroma_420.gnv",            "Chroma format (24f, same clip and q)", "4:2:0", "quarter chroma"),

    ("lossless_444.gnv",          "Lossless at subsampled chroma (q=100) — BUG-49", "4:4:4", "the control: bit-exact, and always was"),
    ("lossless_422.gnv",          "Lossless at subsampled chroma (q=100) — BUG-49", "4:2:2", "was drifting across every tile until 2026-09-10; now bit-exact for the plane it codes"),
    ("lossless_420.gnv",          "Lossless at subsampled chroma (q=100) — BUG-49", "4:2:0", "same fix; the subsample itself is still lossy, which is what the remaining dE00 is"),
]

def human(n):
    return f"{n/1e9:.1f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB" if n >= 1e6 else f"{n/1e3:.0f} KB"

entries = []
for name, group, label, desc in CATALOG:
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        continue
    entries.append({"name": name, "group": group, "label": label,
                    "desc": f"{desc} ({human(os.path.getsize(path))})"})

catalogued = len(entries)

# Full-length films are cut into however many segments the runtime divides into, so they cannot
# be catalogued by hand the way the comparison groups are. Globbed in filename order, which
# generate_movie.sh makes chronological, because that is the order auto-advance walks.
for prefix, title in (("movie_bbb", "Big Buck Bunny \u2014 full film, auto-advance through it"),):
    segs = sorted(glob.glob(os.path.join(OUT, prefix + "_*.gnv")))
    for n, path in enumerate(segs):
        entries.append({
            "name": os.path.basename(path),
            "group": title,
            "label": f"Part {n + 1} of {len(segs)}",
            "desc": f"{human(os.path.getsize(path))} \u2014 tick Auto-advance and it plays straight on",
        })

with open(os.path.join(OUT, "demos.json"), "w") as f:
    json.dump(entries, f, indent=1)
segments = len(entries) - catalogued
print(f"  demos.json: {catalogued} of {len(CATALOG)} catalogued files present"
      + (f", plus {segments} film segment(s)" if segments else ""))
