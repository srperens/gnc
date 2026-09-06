#!/usr/bin/env python3
"""Write a 16-bit RGB PNG. Pillow cannot (it has no 16-bit-per-channel RGB mode).

Needed to build genuine >8-bit test content and to round-trip GNC's 10-bit output, which stores
10-bit samples in the high bits of 16-bit channels.
"""

import struct
import zlib

import numpy as np


def write_rgb16(path, arr):
    """arr: (h, w, 3) uint16."""
    h, w, c = arr.shape
    assert c == 3 and arr.dtype == np.uint16, "expected (h, w, 3) uint16"
    raw = bytearray()
    be = arr.astype(">u2")
    for y in range(h):
        raw.append(0)  # filter type 0
        raw += be[y].tobytes()

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 6))) 
        f.write(chunk(b"IEND", b""))


def read_rgb16(path):
    """Read an RGB PNG as uint16 samples, 8-bit files scaled up by 257.

    Decoded here rather than through Pillow because Pillow *reads* 16-bit RGB PNGs by truncating
    them to 8 bits on open — silently, and with no mode that preserves them. Only filter type 0
    and non-interlaced files are handled, which is what `write_rgb16` and the `image` crate both
    produce.
    """
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path}: not a PNG"
    pos = 8
    w = h = depth = ctype = None
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            w, h, depth, ctype, comp, filt, inter = struct.unpack(">IIBBBBB", body)
            assert inter == 0, f"{path}: interlaced PNGs are not handled"
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break

    assert ctype == 2, f"{path}: expected RGB (colour type 2), got {ctype}"
    raw = zlib.decompress(bytes(idat))
    bypp = 3 * (2 if depth == 16 else 1)
    stride = w * bypp
    out = np.empty((h, w, 3), dtype=np.uint16)
    prev = np.zeros(stride, dtype=np.uint8)
    off = 0
    for y in range(h):
        ftype = raw[off]
        off += 1
        cur = bytearray(raw[off : off + stride])
        off += stride
        if ftype == 1:  # Sub
            for i in range(bypp, stride):
                cur[i] = (cur[i] + cur[i - bypp]) & 0xFF
        elif ftype == 2:  # Up
            cur = bytearray((np.frombuffer(bytes(cur), dtype=np.uint8) + prev).astype(np.uint8))
        elif ftype == 3:  # Average
            for i in range(stride):
                left = cur[i - bypp] if i >= bypp else 0
                cur[i] = (cur[i] + ((left + int(prev[i])) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a_ = cur[i - bypp] if i >= bypp else 0
                b_ = int(prev[i])
                c_ = int(prev[i - bypp]) if i >= bypp else 0
                p = a_ + b_ - c_
                pa, pb, pc = abs(p - a_), abs(p - b_), abs(p - c_)
                pred = a_ if (pa <= pb and pa <= pc) else (b_ if pb <= pc else c_)
                cur[i] = (cur[i] + pred) & 0xFF
        elif ftype != 0:
            raise AssertionError(f"{path}: unknown PNG filter type {ftype}")
        prev = np.frombuffer(bytes(cur), dtype=np.uint8)
        if depth == 16:
            out[y] = np.frombuffer(bytes(cur), dtype=">u2").reshape(w, 3)
        else:
            out[y] = prev.reshape(w, 3).astype(np.uint16) * 257
    return out
