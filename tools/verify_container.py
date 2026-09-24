#!/usr/bin/env python3
"""verify_container - cross-check a .laqt container against a reference GGUF.

Checks, per tensor:
  * directory: the container's engine-name set equals the GGUF's
  * same-format tensors (q8_0 vs Q8_0, f32 vs F32): payloads must be byte-identical
  * format-changed tensors (GGUF F32 vs container q8_0): dequantized values must
    agree within the q8_0 block error bound (its f16 scale also introduces error)
  * tied aliases: lm.output.weight must share bytes with lm.tok_embd.weight

Usage: verify_container.py container.laqt reference.gguf
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laformat as L

# GGML_TYPE_* values we care about
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_TYPE_NAMES = {GGML_F32: "f32", GGML_F16: "f16", GGML_Q8_0: "q8_0"}
GGML_TYPE_SIZES = {GGML_F32: 4, GGML_F16: 2}
GGML_BLOCK = {GGML_Q8_0: (32, 34)}


def gguf_directory(path: Path):
    """Return ({name: dict(dims, type_id, offset, nbytes)}, data_start)."""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF"
        struct.unpack("<I", f.read(4))
        nt = struct.unpack("<Q", f.read(8))[0]
        nkv = struct.unpack("<Q", f.read(8))[0]

        def rd_str():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def skip_val(t):
            if t == 8:
                rd_str()
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                if et == 8:
                    for _ in range(n):
                        rd_str()
                else:
                    f.read(SZ[et] * n)
            else:
                f.read(SZ[t])

        SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        for _ in range(nkv):
            rd_str()
            t = struct.unpack("<I", f.read(4))[0]
            skip_val(t)
        alignment = 32
        tensors = {}
        for _ in range(nt):
            name = rd_str()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            tid = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            tensors[name] = {"dims": dims, "type": tid, "offset": off}
        data_start = f.tell()
        data_start = (data_start + alignment - 1) // alignment * alignment
    return tensors, data_start


def gguf_payload(path: Path, tensors, data_start, name):
    t = tensors[name]
    n = 1
    for d in t["dims"]:
        n *= d
    tid = t["type"]
    if tid in GGML_TYPE_SIZES:
        nb = n * GGML_TYPE_SIZES[tid]
    elif tid in GGML_BLOCK:
        qk, bb = GGML_BLOCK[tid]
        nb = n // qk * bb
    else:
        raise SystemExit(f"{name}: unsupported ggml type {tid}")
    with open(path, "rb") as f:
        f.seek(data_start + t["offset"])
        return f.read(nb), nb


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    cpath, gpath = Path(sys.argv[1]), Path(sys.argv[2])
    c = L.Container(str(cpath))
    gtensors, gstart = gguf_directory(gpath)

    names_c = set(c.names())
    names_g = set(gtensors)
    print(f"container tensors: {len(names_c)}   gguf tensors: {len(names_g)}")
    if names_c != names_g:
        print("  only in container:", sorted(names_c - names_g)[:8])
        print("  only in gguf:", sorted(names_g - names_c)[:8])
    else:
        print("  directory names match exactly")

    same_format_exact = 0
    quant_agree = 0
    mismatch = []
    worst_q = 0.0
    worst_name = ""
    for name in sorted(names_c & names_g):
        ct = c.tensors[name]
        gt = gtensors[name]
        gdims = [int(d) for d in gt["dims"]]
        cdims = [int(d) for d in ct["shape"]]
        gtname = GGML_TYPE_NAMES.get(gt["type"], str(gt["type"]))
        if gdims != cdims:
            mismatch.append((name, "shape", cdims, gdims))
            continue
        gp, gnb = gguf_payload(gpath, gtensors, gstart, name)
        cp = c.raw(name)
        if gtname == ct["dtype"]:
            if gp == cp:
                same_format_exact += 1
            else:
                mismatch.append((name, "bytes differ", ct["dtype"]))
        elif gtname == "f32" and ct["dtype"] == "q8_0":
            g = np.frombuffer(gp, dtype="<f4")
            q = L.dequantize_q8_0(cp, cdims)
            err = float(np.abs(g - q).max())
            # bound: step = d/2, d = amax/127 per block, plus f16 scale rounding
            blk = np.abs(g.reshape(-1, 32)).max(axis=1) / 127.0
            bound = float((blk * 0.6).max())
            if err <= bound + 1e-6:
                quant_agree += 1
                if err > worst_q:
                    worst_q, worst_name = err, name
            else:
                mismatch.append((name, "quant error", err, bound))

    print(f"  identical payloads ({same_format_exact} tensors)")
    print(f"  quantized-agree within bound ({quant_agree} tensors), worst {worst_q:.3e} @ {worst_name}")
    if mismatch:
        print(f"  MISMATCHES: {len(mismatch)}")
        for m in mismatch[:15]:
            print("   ", m)
    else:
        print("  no mismatches")

    # alias check
    if "lm.output.weight" in c.tensors and "lm.tok_embd.weight" in c.tensors:
        same = c.raw("lm.output.weight") == c.raw("lm.tok_embd.weight")
        print(f"  tied lm_head payload identical to tok_embd: {same}")
        assert same
    return 0 if not mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main())
