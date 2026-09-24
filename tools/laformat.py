"""laformat - reader/writer for the locate-anything quantized container.

Container layout (little-endian):

    magic       8 bytes  b"LAQTGGUF"
    version     u32      format version (1)
    header_len  u64      length of the JSON header in bytes
    header      header_len bytes, UTF-8 JSON
    <pad to alignment>
    payloads    tensor data, each `alignment`-byte aligned

Header JSON keys:
    config       object        model hyperparameters
    tokens       array[str]    BPE token table, id == index
    token_types  array[int]    per-token type (1 normal, 4 control/added)
    merges       array[str]    BPE merge rules in rank order
    alignment    int           payload alignment in bytes (4096)
    data_offset  int           absolute file offset where payloads begin
    tensors      array[obj]    directory, in file order
        name     str           engine tensor name
        dtype    "f32" | "q8_0"
        shape    array[int]    ggml-style shape, shape[0] contiguous
        offset   int           absolute file offset (may alias another entry)
        nbytes   int           payload length

`offset`/`nbytes` may be shared by more than one entry, which is how tied
weights (lm_head == embed_tokens) are stored once.

Tensors are written and read raw; f32 payloads are little-endian f32 and q8_0
payloads use the ggml block layout (34 bytes: f16 d + 32 x int8).

Writing is two-phase and streaming: `plan()` fixes the directory (it needs only
names/shapes/dtypes), `begin()` writes the header, `append()` streams payloads
straight to disk, `finish()` patches nothing (the header was written from the
plan, so it is already correct).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"LAQTGGUF"
VERSION = 1
ALIGNMENT = 4096
QK8_0 = 32
BLOCK_Q8_0_BYTES = 34
DTYPES = ("f32", "q8_0")
BLOCK_DTYPE = np.dtype([("d", "<f2"), ("qs", "i1", (QK8_0,))])


def align_up(n: int, a: int = ALIGNMENT) -> int:
    return (n + a - 1) // a * a


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def dtype_nbytes(dtype: str, shape) -> int:
    shape = [int(x) for x in shape]
    if dtype == "f32":
        return numel(shape) * 4
    if dtype == "q8_0":
        if not shape or shape[0] % QK8_0 != 0:
            raise ValueError(f"q8_0 tensor ne0={shape[:1]} is not a multiple of {QK8_0}")
        return (numel(shape) // QK8_0) * BLOCK_Q8_0_BYTES
    raise ValueError(f"unknown dtype {dtype!r}")


def quantizable(shape) -> bool:
    """Can this logical shape be stored as q8_0? (2-D, contiguous dim / 32)"""
    return len(shape) == 2 and int(shape[0]) % QK8_0 == 0


# --------------------------------------------------------------------------
# quantization (matches ggml quantize_row_q8_0_ref bit-for-bit)
# --------------------------------------------------------------------------

def _roundf(x: np.ndarray) -> np.ndarray:
    """C roundf: round half away from zero (numpy rounds half to even)."""
    return np.where(x >= 0, np.floor(x + 0.5), -np.floor(-x + 0.5)).astype(np.float32)


def quantize_q8_0(data: np.ndarray, shape=None) -> bytes:
    """Quantize f32 data (ggml shape [ne0, ne1, ...], ne0 contiguous) to q8_0.

    Matches ggml's quantize_row_q8_0_ref: per 32-value block,
        d = amax / 127 ; id = 1/d ; qs[j] = roundf(x[j] * id)
    with d stored as f16.
    """
    x = np.ascontiguousarray(data, dtype=np.float32)
    if shape is not None:
        shape = [int(s) for s in shape]
        if list(x.shape) == shape:
            ne0 = shape[0]
            rows = x.reshape(-1, ne0) if x.ndim > 1 else x.reshape(1, -1)
        else:
            # caller passed the ggml shape of a flat buffer
            assert numel(shape) == x.size, (shape, x.size)
            ne0 = shape[0]
            rows = x.reshape(-1, ne0)
    else:
        rows = x.reshape(1, -1) if x.ndim == 1 else x
        ne0 = rows.shape[-1]
    if ne0 % QK8_0 != 0:
        raise ValueError(f"ne0={ne0} is not a multiple of {QK8_0}")
    nrows = rows.shape[0]
    blocks = rows.reshape(nrows, ne0 // QK8_0, QK8_0)
    amax = np.abs(blocks).max(axis=2)
    d = (amax / ((1 << 7) - 1)).astype(np.float32)
    d16 = d.astype(np.float16)
    inv = np.where(d > 0, np.float32(1.0) / np.where(d > 0, d, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
    qs = np.clip(_roundf(blocks * inv[:, :, None]), -128, 127).astype(np.int8)
    out = np.empty((nrows, ne0 // QK8_0), dtype=BLOCK_DTYPE)
    out["d"] = d16
    out["qs"] = qs
    return out.tobytes()


def dequantize_q8_0(raw: bytes, shape, as_flat=True) -> np.ndarray:
    """Inverse of quantize_q8_0.  Returns the flat f32 values in file order."""
    shape = [int(s) for s in shape]
    nb = numel(shape) // QK8_0
    blk = np.frombuffer(raw, dtype=BLOCK_DTYPE, count=nb)
    vals = blk["qs"].astype(np.float32) * blk["d"].astype(np.float32)[:, None]
    flat = vals.reshape(-1)
    return flat if as_flat else flat.reshape(shape)


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

class ContainerWriter:
    def __init__(self, path, config=None, tokens=None, token_types=None, merges=None,
                 alignment=ALIGNMENT):
        self.path = Path(path)
        self.config = config or {}
        self.tokens = tokens
        self.token_types = token_types
        self.merges = merges
        self.alignment = alignment
        self.entries = []      # planned directory entries
        self._fh = None
        self._pos = 0

    # ---- phase 1: plan -------------------------------------------------
    def plan(self, name: str, dtype: str, shape, alias_of: str | None = None):
        """Add a directory entry and return the absolute payload offset.

        `alias_of` shares the payload of an already-planned tensor (tied
        weights) and stores nothing new.
        """
        shape = [int(x) for x in shape]
        if alias_of is not None:
            src = next(e for e in self.entries if e["name"] == alias_of)
            if src["shape"] != shape or src["dtype"] != dtype:
                raise ValueError(f"alias {name} shape/dtype mismatch with {alias_of}")
            e = {"name": name, "dtype": dtype, "shape": shape,
                 "offset": None, "nbytes": src["nbytes"], "alias_of": alias_of}
            self.entries.append(e)
            return None
        n = dtype_nbytes(dtype, shape)
        e = {"name": name, "dtype": dtype, "shape": shape,
             "offset": None, "nbytes": n, "alias_of": None}
        self.entries.append(e)
        return None

    def _layout(self, header_len: int):
        """Recompute every payload offset from scratch; returns (data_offset, end)."""
        off = align_up(8 + 4 + 8 + header_len, self.alignment)
        data_offset = off
        by_name = {}
        for e in self.entries:
            e["offset"] = None
        for e in self.entries:
            if e["alias_of"] is not None:
                src = by_name.get(e["alias_of"])
                if src is None:
                    raise ValueError(f"alias {e['name']} refers to {e['alias_of']} "
                                     f"which must be planned earlier")
                e["offset"] = src["offset"]
                e["nbytes"] = src["nbytes"]
            else:
                e["offset"] = off
                off = align_up(off + e["nbytes"], self.alignment)
            by_name[e["name"]] = e
        return data_offset, off

    def _header(self, data_offset):
        h = {
            "config": self.config,
            "alignment": self.alignment,
            "data_offset": data_offset,
            "tensors": self.entries,
        }
        if self.tokens is not None:
            h["tokens"] = self.tokens
            h["token_types"] = self.token_types
            h["merges"] = self.merges
        return h

    # ---- phase 2: write ------------------------------------------------
    def begin(self):
        """Fix the layout and write the header; returns the expected file size."""
        # header length depends on offsets which depend on header length; both
        # are multiples of `alignment`, so iterate to a fixed point.
        data_offset, end = self._layout(0)
        for _ in range(8):
            hb = json.dumps(self._header(data_offset), separators=(",", ":")).encode("utf-8")
            data_offset2, end2 = self._layout(len(hb))
            if data_offset2 == data_offset:
                end = end2
                break
            data_offset, end = data_offset2, end2
        hb = json.dumps(self._header(data_offset), separators=(",", ":")).encode("utf-8")
        assert align_up(8 + 4 + 8 + len(hb), self.alignment) == data_offset, \
            (len(hb), data_offset)
        self._fh = open(self.path, "wb")
        self._fh.write(MAGIC)
        self._fh.write(struct.pack("<I", VERSION))
        self._fh.write(struct.pack("<Q", len(hb)))
        self._fh.write(hb)
        self._fh.write(b"\0" * (data_offset - 8 - 4 - 8 - len(hb)))
        self._pos = data_offset
        return end

    def append(self, name: str, payload: bytes):
        """Write one tensor payload into its planned slot."""
        e = next(x for x in self.entries if x["name"] == name)
        assert self._pos == e["offset"], (name, self._pos, e["offset"], "append out of order")
        assert len(payload) == e["nbytes"], (name, len(payload), e["nbytes"])
        self._fh.write(payload)
        self._pos = align_up(self._pos + e["nbytes"], self.alignment)
        pad = self._pos - (e["offset"] + len(payload))
        if pad:
            self._fh.write(b"\0" * pad)

    def append_chunks(self, name: str, chunks):
        """Stream a payload in pieces (bounded memory).  Yields nothing."""
        e = next(x for x in self.entries if x["name"] == name)
        assert self._pos == e["offset"], (name, self._pos, e["offset"], "append out of order")
        written = 0
        for chunk in chunks:
            self._fh.write(chunk)
            written += len(chunk)
        assert written == e["nbytes"], (name, written, e["nbytes"])
        self._pos = align_up(self._pos + written, self.alignment)
        pad = self._pos - (e["offset"] + written)
        if pad:
            self._fh.write(b"\0" * pad)

    def flush(self):
        if self._fh is not None:
            self._fh.flush()

    def finish(self):
        self._fh.close()
        return self.path


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

class Container:
    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            magic = f.read(8)
            if magic != MAGIC:
                raise ValueError(f"{path}: bad magic {magic!r} (not a laformat container)")
            (version,) = struct.unpack("<I", f.read(4))
            if version != VERSION:
                raise ValueError(f"{path}: unsupported version {version}")
            (hlen,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(hlen).decode("utf-8"))
        self.config = self.header["config"]
        self.tensors = {t["name"]: t for t in self.header["tensors"]}
        self.tokens = self.header.get("tokens")
        self.merges = self.header.get("merges")
        self._fh = None

    def names(self):
        return list(self.tensors)

    def raw(self, name):
        t = self.tensors[name]
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(t["offset"])
        return self._fh.read(t["nbytes"])

    def f32(self, name):
        t = self.tensors[name]
        raw = self.raw(name)
        if t["dtype"] == "f32":
            return np.frombuffer(raw, dtype="<f4")
        return dequantize_q8_0(raw, t["shape"])

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
