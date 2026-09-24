#!/usr/bin/env python3
"""prequant - quantize a LocateAnything safetensors checkpoint into a .laqt container.

Dependency-free: numpy only (no torch, no safetensors, no gguf package).

    prequant --model-dir /path/to/hf_snapshot --out model.laqt \
             [--from-gguf existing.gguf] [--config-json c.json] [--tokenizer-json t.json]

--model-dir is the upstream HuggingFace snapshot, so config.json,
preprocessor_config.json, vocab.json, added_tokens.json and merges.txt are read
from it when present; --config-json / --tokenizer-json / --from-gguf override
that (the GGUF path is for cross-checking against the reference engine).

Policy (the all-q8_0 experiment):
  * every rank-2 tensor whose contiguous dim (ne0) is a multiple of 32  -> q8_0
  * everything else (norms, biases, 1-D tensors, conv kernels, and the few
    weights whose ne0 is not a multiple of 32, e.g. vit fc1 [1152,4304]) -> f32
Tied weights are stored once: lm.output.weight aliases lm.tok_embd.weight.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laformat as L
from namemap import rename_tensor

BF16 = np.dtype("<u2")


def log(msg):
    print(f"[prequant] {msg}", flush=True)


# --------------------------------------------------------------------------
# safetensors reading (headers only, then raw payload slices)
# --------------------------------------------------------------------------

def read_safetensors_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n).decode("utf-8"))
    base = 8 + n
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out[k] = {"dtype": v["dtype"], "shape": [int(s) for s in v["shape"]],
                  "start": base + a, "nbytes": b - a}
    return out


def bf16_to_f32(raw: bytes) -> np.ndarray:
    """Lossless bf16 -> f32 (left shift by 16)."""
    u = np.frombuffer(raw, dtype=BF16).astype(np.uint32) << 16
    return u.view(np.float32)


def rows_to_q8_0(raw: bytes, dtype: str, ne0: int) -> bytes:
    """Convert a chunk of raw rows into q8_0 blocks (streaming path)."""
    if dtype == "BF16":
        x = bf16_to_f32(raw)
    elif dtype == "F32":
        x = np.frombuffer(raw, dtype="<f4")
    else:
        raise SystemExit(f"streaming only supports BF16/F32, got {dtype}")
    return L.quantize_q8_0(x, [ne0, x.size // ne0])


def rows_to_f32_bytes(raw: bytes, dtype: str) -> bytes:
    if dtype == "BF16":
        return bf16_to_f32(raw).tobytes()
    if dtype == "F32":
        return raw
    raise SystemExit(f"streaming only supports BF16/F32, got {dtype}")


class Shards:
    def __init__(self, model_dir: Path):
        self.dir = model_dir
        idx_path = model_dir / "model.safetensors.index.json"
        if idx_path.exists():
            idx = json.loads(idx_path.read_text())
            self.weight_map = idx["weight_map"]
        else:
            sts = sorted(model_dir.glob("*.safetensors"))
            if not sts:
                raise SystemExit(f"no *.safetensors in {model_dir}")
            self.weight_map = {}
            for st in sts:
                for k in read_safetensors_header(st):
                    self.weight_map[k] = st.name
        self.headers = {sh: read_safetensors_header(model_dir / sh)
                        for sh in sorted(set(self.weight_map.values()))}
        for name, sh in self.weight_map.items():
            if name not in self.headers[sh]:
                raise SystemExit(f"{name} missing from {sh}")
        self._fh = {}

    def names(self):
        return list(self.weight_map)

    def info(self, name):
        return self.headers[self.weight_map[name]][name]

    def raw_slice(self, name, start: int, nbytes: int) -> bytes:
        """Read `nbytes` at payload-relative `start` (for streaming large tensors)."""
        sh = self.weight_map[name]
        if sh not in self._fh:
            self._fh[sh] = open(self.dir / sh, "rb")
        info = self.info(name)
        if start < 0 or start + nbytes > info["nbytes"]:
            raise SystemExit(f"{name}: slice {start}+{nbytes} out of range {info['nbytes']}")
        f = self._fh[sh]
        f.seek(info["start"] + start)
        return f.read(nbytes)

    def f32(self, name) -> np.ndarray:
        sh = self.weight_map[name]
        if sh not in self._fh:
            self._fh[sh] = open(self.dir / sh, "rb")
        info = self.info(name)
        f = self._fh[sh]
        f.seek(info["start"])
        raw = f.read(info["nbytes"])
        if info["dtype"] == "BF16":
            return bf16_to_f32(raw)
        if info["dtype"] == "F32":
            return np.frombuffer(raw, dtype="<f4").copy()
        if info["dtype"] in ("I64", "I32"):
            dt = "<i8" if info["dtype"] == "I64" else "<i4"
            return np.frombuffer(raw, dtype=dt).astype(np.float32)
        raise SystemExit(f"{name}: unsupported dtype {info['dtype']}")

    def close(self):
        for f in self._fh.values():
            f.close()


# --------------------------------------------------------------------------
# GGUF metadata bootstrap (config + tokenizer) - KVs only, no tensors
# --------------------------------------------------------------------------

TY = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
      7: "bool", 8: "str", 9: "arr", 10: "u64", 11: "i64", 12: "f64"}
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FM = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def gguf_kvs(path: Path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        struct.unpack("<I", f.read(4))
        struct.unpack("<Q", f.read(8))
        nkv = struct.unpack("<Q", f.read(8))[0]

        def rd_str():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def rd_val(t):
            if t == 8:
                return rd_str()
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                return [rd_val(et) for _ in range(n)]
            return struct.unpack("<" + FM[t], f.read(SZ[t]))[0]

        out = {}
        for _ in range(nkv):
            k = rd_str()
            t = struct.unpack("<I", f.read(4))[0]
            out[k] = rd_val(t)
        return out


def config_from_gguf(kvs):
    cfg = {}
    for k, v in kvs.items():
        if k.startswith("locateanything.") and k not in ("locateanything.tokenizer.tokens",
                                                         "locateanything.tokenizer.token_types",
                                                         "locateanything.tokenizer.merges"):
            short = k[len("locateanything."):]
            cfg[short] = v
    return cfg


def tokenizer_from_gguf(kvs):
    return (kvs.get("locateanything.tokenizer.tokens"),
            kvs.get("locateanything.tokenizer.token_types"),
            kvs.get("locateanything.tokenizer.merges"))


# --------------------------------------------------------------------------
# HuggingFace snapshot bootstrap (config + tokenizer) - read straight from the
# upstream files, so no GGUF is needed to build a container.
# --------------------------------------------------------------------------

# Where each special token id lives in config.json: 8 at the top level, 5 nested
# in text_config (the same mapping the reference converter uses).
SPECIAL_TOKEN_SOURCES = {
    "token.image":       ("top",  "image_token_index"),
    "token.box_start":   ("top",  "box_start_token_id"),
    "token.box_end":     ("top",  "box_end_token_id"),
    "token.coord_start": ("top",  "coord_start_token_id"),
    "token.coord_end":   ("top",  "coord_end_token_id"),
    "token.ref_start":   ("top",  "ref_start_token_id"),
    "token.ref_end":     ("top",  "ref_end_token_id"),
    "token.none":        ("top",  "none_token_id"),
    "token.null":        ("text", "null_token_id"),
    "token.switch":      ("text", "switch_token_id"),
    "token.text_mask":   ("text", "text_mask_token_id"),
    "token.eos":         ("text", "eos_token_id"),
    "token.bos":         ("text", "bos_token_id"),
}


def _f32(x):
    """Round through f32: the reference engine stores these in its GGUF as f32,
    so rounding keeps the header identical to one built from that GGUF."""
    return float(np.float32(x))


def config_from_hf(model_dir: Path):
    """Container config from the upstream config.json + preprocessor_config.json.

    Image preprocessing (mean/std/in_token_limit) lives in
    preprocessor_config.json, not config.json; the special token ids are spread
    over both levels of config.json."""
    cfg = json.loads((model_dir / "config.json").read_text())
    prep = json.loads((model_dir / "preprocessor_config.json").read_text())
    t, v = cfg["text_config"], cfg["vision_config"]
    out = {
        "lm.hidden_size":       t["hidden_size"],
        "lm.n_layers":          t["num_hidden_layers"],
        "lm.n_heads":           t["num_attention_heads"],
        "lm.n_kv_heads":        t["num_key_value_heads"],
        "lm.head_dim":          t.get("head_dim", t["hidden_size"] // t["num_attention_heads"]),
        "lm.intermediate_size": t["intermediate_size"],
        "lm.vocab_size":        t["vocab_size"],
        "lm.rope_theta":        _f32(t["rope_theta"]),
        "lm.rms_norm_eps":      _f32(t["rms_norm_eps"]),
        "lm.block_size":        int(t["block_size"]),
        "vit.hidden_size":      v["hidden_size"],
        "vit.n_layers":         v["num_hidden_layers"],
        "vit.n_heads":          v["num_attention_heads"],
        "vit.head_dim":         v.get("head_dim", v["hidden_size"] // v["num_attention_heads"]),
        "vit.intermediate_size": v["intermediate_size"],
        "vit.patch_size":       v["patch_size"],
        "vit.merge_kernel_size": [int(x) for x in v["merge_kernel_size"]],
        "vit.init_pos_emb_hw":  v.get("init_pos_emb_height", 64),
        "vit.rope_theta":       _f32(10000.0),
    }
    for key, (scope, cfg_key) in SPECIAL_TOKEN_SOURCES.items():
        src = cfg if scope == "top" else t
        if cfg_key not in src:
            raise SystemExit(f"config.json: missing {scope}-level key '{cfg_key}' (needed for {key})")
        out[key] = int(src[cfg_key])
    # Image preprocessing comes from preprocessor_config.json (keys image_mean,
    # image_std, in_token_limit), not config.json. Key order matches the
    # reference converter's, so the header bytes are identical too.
    out["image.mean"] = [_f32(x) for x in prep.get("image_mean", [0.5, 0.5, 0.5])]
    out["image.std"] = [_f32(x) for x in prep.get("image_std", [0.5, 0.5, 0.5])]
    out["image.in_token_limit"] = int(prep.get("in_token_limit", 25600))
    out["tokenizer.model"] = "gpt2"
    return out


def tokenizer_from_hf(model_dir: Path):
    """Embed the Qwen2 BPE table (gpt2 byte-level BPE) from the upstream
    vocab.json / added_tokens.json / merges.txt."""
    vocab = json.loads((model_dir / "vocab.json").read_text())
    added = {}
    p = model_dir / "added_tokens.json"
    if p.exists():
        added = json.loads(p.read_text())
    merged = dict(vocab)
    merged.update(added)
    id_to_tok = {int(i): tok for tok, i in merged.items()}
    n = max(id_to_tok) + 1
    tokens = [id_to_tok.get(i, f"<unused_{i}>") for i in range(n)]
    added_ids = {int(i) for i in added.values()}
    types = [4 if i in added_ids else 1 for i in range(n)]   # 1=normal, 4=control
    merges = []
    with open(model_dir / "merges.txt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if i == 0 and line.startswith("#version"):
                continue
            if line:
                merges.append(line)
    return tokens, types, merges


# --------------------------------------------------------------------------

def build_plan(shards, reserved_f32=()):
    """Return the ordered list of (hf_name, engine_name, dtype, shape, alias_of)."""
    plan = []
    seen_engine = {}
    for hf in sorted(shards.names()):
        eng = rename_tensor(hf)
        if eng is None:
            raise SystemExit(f"no name mapping for {hf}")
        if eng in seen_engine:
            raise SystemExit(f"duplicate engine name {eng} from {hf} and {seen_engine[eng]}")
        seen_engine[eng] = hf
        info = shards.info(hf)
        # safetensors stores PyTorch order [out, in, ...]; ggml stores the reverse
        # with ne0 (the contiguous / reduction dim) first.  The flat byte order is
        # identical - only the metadata shape is reversed - which is why the
        # existing converter needs no transpose.
        shape = list(reversed(info["shape"]))
        if eng == "lm.output.weight":
            # tied weights: same payload, second directory entry (planned last)
            plan.append((hf, eng, "q8_0", shape, "lm.tok_embd.weight"))
            continue
        if L.quantizable(shape) and eng not in reserved_f32:
            plan.append((hf, eng, "q8_0", shape, None))
        else:
            plan.append((hf, eng, "f32", shape, None))
    # aliases must come after their source entry
    normal = [p for p in plan if p[4] is None]
    aliases = [p for p in plan if p[4] is not None]
    return normal + aliases


def dtype_nbytes_of(dtype, shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n * 4 if dtype == "f32" else (n // 32) * 34


def verify_tensor_error(shards, hf, eng, shape, chunk_rows, writer):
    """Re-read what was just written and compare against the source in chunks."""
    path = writer.path
    ne0 = int(shape[0])
    nrows = int(np.prod(shape)) // ne0
    entry = next(e for e in writer.entries if e["name"] == eng)
    row_bytes = ne0 * (2 if shards.info(hf)["dtype"] == "BF16" else 4)
    blk = 0
    worst = 0.0
    dtype_src = shards.info(hf)["dtype"]
    with open(path, "rb") as f:
        f.seek(entry["offset"])
        done = 0
        while done < nrows:
            n = min(chunk_rows, nrows - done)
            raw_src = shards.raw_slice(hf, done * row_bytes, n * row_bytes)
            src = (bf16_to_f32(raw_src) if dtype_src == "BF16"
                   else np.frombuffer(raw_src, dtype="<f4").copy())
            pay = f.read((n * ne0 // 32) * 34)
            got = L.dequantize_q8_0(pay, [ne0, n])
            worst = max(worst, float(np.abs(src - got).max()))
            done += n
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="HF snapshot dir with *.safetensors")
    ap.add_argument("--out", required=True, help="output .laqt path")
    ap.add_argument("--from-gguf", help="borrow config + tokenizer from this GGUF")
    ap.add_argument("--config-json", help="config JSON (overrides --from-gguf config)")
    ap.add_argument("--tokenizer-json", help="tokenizer JSON {'tokens':[],'token_types':[],'merges':[]}")
    ap.add_argument("--verify-tensors", action="store_true",
                    help="dequantize each q8_0 tensor back and report the error")
    ap.add_argument("--tensors", help="comma-separated engine-name globs to include "
                                      "(smoke testing); default: all")
    ap.add_argument("--chunk-rows", type=int, default=4096,
                    help="rows processed per chunk (bounds peak RAM; default 4096)")
    args = ap.parse_args()

    t0 = time.time()
    model_dir = Path(args.model_dir)
    shards = Shards(model_dir)
    log(f"{len(shards.names())} tensors in {len(shards.headers)} shard(s)")

    config, tokens, token_types, merges = {}, None, None, None
    if args.from_gguf:
        kvs = gguf_kvs(Path(args.from_gguf))
        config = config_from_gguf(kvs)
        tokens, token_types, merges = tokenizer_from_gguf(kvs)
        log(f"config from {args.from_gguf}: {len(config)} keys; "
            f"tokenizer {len(tokens) if tokens else 0} tokens")
    elif (model_dir / "config.json").exists():
        config = config_from_hf(model_dir)
        log(f"config from {model_dir}/config.json: {len(config)} keys")
    if args.config_json:
        config.update(json.loads(Path(args.config_json).read_text()))
    if args.tokenizer_json:
        tj = json.loads(Path(args.tokenizer_json).read_text())
        tokens, token_types, merges = tj["tokens"], tj["token_types"], tj["merges"]
    elif tokens is None and (model_dir / "vocab.json").exists():
        tokens, token_types, merges = tokenizer_from_hf(model_dir)
        log(f"tokenizer from {model_dir}/vocab.json: {len(tokens)} tokens, "
            f"{sum(1 for x in token_types if x == 4)} added, {len(merges)} merges")
    if not config:
        log("WARNING: no config provided; container will have an empty config")
    if tokens is None:
        log("WARNING: no tokenizer provided")

    plan = build_plan(shards)
    if args.tensors:
        import fnmatch
        pats = [p.strip() for p in args.tensors.split(",")]
        keep = [e for e in plan if any(fnmatch.fnmatch(e[1], p) for p in pats)]
        # keep alias sources even if not matched
        srcs = {e[4] for e in keep if e[4]}
        need = [e for e in plan if e[1] in srcs and e not in keep]
        plan = keep + need
        log(f"--tensors: {len(plan)} entries (including alias sources)")
    nq = sum(1 for _, _, d, _, _ in plan if d == "q8_0")
    nf = len(plan) - nq
    log(f"plan: {nq} q8_0 tensors, {nf} f32 tensors")

    w = L.ContainerWriter(args.out, config=config, tokens=tokens,
                          token_types=token_types, merges=merges)
    for hf, eng, dtype, shape, alias in plan:
        w.plan(eng, dtype, shape, alias_of=alias)
    total = w.begin()
    log(f"container header written; payload bytes end at {total/1e9:.3f} GB")

    worst = 0.0
    written = 0
    chunk_rows = args.chunk_rows
    for hf, eng, dtype, shape, alias in plan:
        if alias is not None:
            continue
        info = shards.info(hf)
        ne0 = int(shape[0])                 # ggml ne0 == rows length in file order
        nrows = int(np.prod(shape)) // ne0
        if info["dtype"] not in ("BF16", "F32"):
            raise SystemExit(f"{hf}: streaming path cannot read {info['dtype']}")
        row_bytes = ne0 * (2 if info["dtype"] == "BF16" else 4)

        def chunks():
            nonlocal worst
            done = 0
            while done < nrows:
                n = min(chunk_rows, nrows - done)
                raw = shards.raw_slice(hf, done * row_bytes, n * row_bytes)
                if dtype == "q8_0":
                    yield rows_to_q8_0(raw, info["dtype"], ne0)
                else:
                    yield rows_to_f32_bytes(raw, info["dtype"])
                done += n

        w.append_chunks(eng, chunks())
        written += dtype_nbytes_of(dtype, shape)
        if args.verify_tensors and dtype == "q8_0":
            w.flush()
            worst = max(worst, verify_tensor_error(shards, hf, eng, shape, chunk_rows, w))
        if written % (256 << 20) < (chunk_rows * ne0 * 4):
            log(f"  {written/1e9:.2f} GB planned, t={time.time()-t0:.0f}s")
    w.finish()
    size = Path(args.out).stat().st_size
    log(f"done: {args.out} {size/1e9:.3f} GB in {time.time()-t0:.1f}s"
        + (f"; worst |dq-x| = {worst:.3e}" if args.verify_tensors else ""))
    shards.close()


if __name__ == "__main__":
    main()
