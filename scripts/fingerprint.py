#!/usr/bin/env python3
"""COORD-6: the two lines that make a number carry its codec.

Five of the seven recorded measurement failures in this repo are one session's table decaying
because `main` moved under it — early rows and late rows from different encoders, no error
anywhere. `gnc fingerprint` answers "would this binary produce different bytes?" by encoding a
pinned 10-configuration matrix and digesting the result: 0.52 s, exact on what it covers, and
insensitive to changes that do not move output (it is unchanged under
`GNC_REF_FROM_SOURCE=0`, which is byte-identical by measurement, and moves under
`GNC_PAD_FILL`, `GNC_DEAD_ZONE` and `GNC_REF_DEBLOCK`).

Two uses, and a harness wants both:

    fp = read(GNC)                    # print it beside the numbers, so a later reader can compare
    ...sweep...
    check_unchanged(GNC, fp)          # refuse if the binary was rebuilt mid-sweep

In a shell loop, the same thing without importing anything:

    FP=$(gnc fingerprint 2>/dev/null | grep -o 'v1 [0-9a-f]*')
    ... your loop ...
    [ "$FP" = "$(gnc fingerprint 2>/dev/null | grep -o 'v1 [0-9a-f]*')" ] || echo "REBUILT MID-RUN"

It cannot say *why* two fingerprints differ and says nothing about paths outside its matrix. It
also only protects numbers that carry it — the same adoption problem prose has, met with one
command instead of a paragraph.
"""

import re
import subprocess
import sys

LINE = re.compile(r"codec-fingerprint (v\d+ [0-9a-f]+)")


def read(gnc):
    """-> 'v1 abf86a50'. Exits if the binary cannot produce one, because a sweep that cannot say
    which encoder it measured is the failure this exists to prevent."""
    p = subprocess.run([str(gnc), "fingerprint"], capture_output=True, text=True)
    m = LINE.search(p.stdout) or LINE.search(p.stderr)
    if p.returncode != 0 or not m:
        sys.exit(f"could not read a codec fingerprint from {gnc}:\n{(p.stderr or p.stdout)[-2000:]}")
    return m.group(1)


def check_unchanged(gnc, before):
    """Refuse if the binary's output moved since `before` — a mid-sweep rebuild mixes two
    encoders across the rows of one table, which is the shape that cost the most re-runs here."""
    after = read(gnc)
    if after != before:
        sys.exit(
            f"REFUSING to report: the codec fingerprint changed mid-run, {before} -> {after}.\n"
            f"  The rows of this table came from two different encoders. Rebuild once, then\n"
            f"  re-run the whole sweep — do not splice the halves."
        )
    return after
