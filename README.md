# locate-anything-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).
The family also includes [rmbg-rs](https://github.com/jacobsparts/rmbg-rs),
[realesrgan-rs](https://github.com/jacobsparts/realesrgan-rs) and
[lama-inpaint-rs](https://github.com/jacobsparts/lama-inpaint-rs); they share the
[lightgpu toolkit](https://github.com/jacobsparts/lightgpu).
[pixeldeck](https://github.com/jacobsparts/pixeldeck) is a local web app for
cleaning up product photos that drives all of these engines.

A self-contained command-line tool for open-vocabulary object detection with
[NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B).
Give it a PNG image and a phrase, it prints labeled bounding boxes as JSON:

```
$ ./locate-anything detect photo.png \
    "Locate all the instances that matches the following description: cat."
{"detections":[{"label":"cat","box":[0.00,50.18,221.76,443.07]}]}
```

It runs on an NVIDIA GPU or on the CPU. The binary has no runtime dependencies
beyond the NVIDIA driver (no Python, no PyTorch, no CUDA toolkit); it is a
single Linux x86-64 executable that links only `libc`, `libm` and `libgcc_s`.

LocateAnything-3B is the model; this repository is an independent Rust
inference engine for it. On the tested fixture its JSON output is byte-identical
to the C++/ggml reference port ([Verification](#verification)).

## Requirements

- Linux x86-64.
- **GPU:** an NVIDIA GPU with compute capability 6.1, 7.5 or 8.0 (the fatbin
  embeds SASS for sm_61/75/80, plus PTX for 8.0 so later drivers on those
  architectures JIT). Developed and tested on a GTX 1080. Roughly 6 GB of free
  VRAM for the test image; more for larger images.
- **CPU fallback:** no GPU needed; set `LA_CPU=1` to force it.
- **Release binary:** glibc 2.34 or newer (Ubuntu 22.04, Debian 12, Fedora 35+).
- **Building the model (once):** Python 3, numpy and curl, plus about 12 GB of
  free disk space.

## Install

The release is the binaries themselves - no archive - and **not** the model
weights. Two of them, identical in behaviour:

* `locate-anything-linux-x86_64` - CUDA build; probes for a driver at startup and
  falls back to the CPU path when it cannot be initialised.
* `locate-anything-linux-x86_64-cpu-only` - pure Rust; no CUDA toolkit, no driver
  and no NVIDIA GPU involved at all.

Download one, make it executable, and build the model once:

```sh
curl -LO https://github.com/jacobsparts/locate-anything-rs/releases/latest/download/locate-anything-linux-x86_64
chmod +x locate-anything-linux-x86_64

# one command: downloads 7.7 GB and builds a 4.16 GB model next to the binary
curl -fsSL https://raw.githubusercontent.com/jacobsparts/locate-anything-rs/main/get-model.sh | bash
# or, from a checkout:  ./get-model.sh

./locate-anything-linux-x86_64 detect photo.png \
    "Locate all the instances that matches the following description: cat."
```

`get-model.sh` needs Python 3, numpy and curl. It downloads NVIDIA's upstream
checkpoint into `model-source/` (7.7 GB) and quantizes it into a single
`locate-anything-allq8_0.laqt` (4.16 GB) in about a minute, using about 0.6 GB
of RAM. The converter is three Python files in `tools/`
(`prequant.py`, `laformat.py`, `namemap.py`); piped into a shell there is no
checkout to find them in, so the script fetches them from the same revision of
this repository. Pass an output path as the first argument or set
`LA_MODEL_OUT`; `LA_MODEL_CACHE` moves the 7.7 GB download, and
`LA_REPO`/`LA_REF`/`LA_RAW_BASE`/`LA_BASE` point the script at a fork or a
mirror. A download that fails or is interrupted leaves nothing behind: the
partially written file is removed on the way out, so the only files that ever
appear are complete ones, and the container itself is written under a temporary
name and moved into place only once the converter has finished.

**The 4.16 GB container is not in the releases.** It is past GitHub's 2 GiB
per-asset limit, so this script is the way to get it.

Both files are needed only during conversion — afterwards you can delete
`model-source/` to reclaim 7.7 GB. The build is deterministic: rebuilding from
the same snapshot reproduces the verified container byte for byte (md5
`85909dfebaf0d9efaf973137007940c2`).

The model file is found automatically when it sits next to the binary (or in a
`models/` subdirectory). Point elsewhere with `--model <path>`. Inference itself
needs no Python.

> **Model licence:** LocateAnything-3B is under the NVIDIA License, limited to
> **non-commercial research or evaluation use**. The weights are not in this
> repository or its releases. Read the
> [upstream licence](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE)
> before running `get-model.sh`.

## Usage

```
./locate-anything detect <image.png> <query>        # boxes as JSON on stdout
./locate-anything detect --model model.laqt <image.png> <query>
./locate-anything inspect                           # print the container directory
./locate-anything gpuinfo                           # probe the GPU + driver
```

`detect` writes the result to stdout as one line of compact JSON and every
diagnostic to stderr, so stdout pipes straight into a parser:

```
$ ./locate-anything detect photo.png \
    "Locate all the instances that matches the following description: cat</c>remote." \
    2>/dev/null | python3 -m json.tool
{
    "detections": [
        {"label": "cat",    "box": [3.14, 50.62, 221.76, 443.52]},
        {"label": "cat",    "box": [241.92, 23.30, 447.10, 347.20]},
        {"label": "remote", "box": [27.78, 68.99, 122.75, 109.76]},
        {"label": "remote", "box": [233.41, 71.68, 258.94, 174.27]}
    ]
}
```

### The query

The query is free text, but the model was instruction-tuned with a fixed prompt
template and works best when you use it:

```
"Locate all the instances that matches the following description: <description>."
```

Separate multiple classes with `</c>` to detect them in one pass
(`...description: cat</c>remote.`). The binary wraps your query in the chat
template for you; you supply only the sentence. Short bare queries such as
`"cat"` run, but the full template is what the model was trained on and what the
reference outputs here were validated against.

### Output coordinates

Each `box` is `[x_min, y_min, x_max, y_max]`. Coordinates are in the
**preprocessed image's** pixel space, not necessarily the input's. Before
inference the image is bicubic-scaled so each side is a multiple of 28 (and, for
large inputs, downscaled to a token budget of 25600 patches). The exact resized
dimensions are printed on stderr, e.g.:

```
image     : photo.png (448x448) -> target 448x448 grid 32x32 (1024 patches, 256 merged)
```

To draw a box back on your original image, scale it by `orig_w / target_w` and
`orig_h / target_h`. For the 448x448 test image the target equals the input, so
no rescaling is needed.

### Environment variables

```
LA_CPU=1      force the CPU backend (also the automatic fallback when no
              CUDA driver initialises)
LA_MAX_NEW    decode token budget (default 1024)
```

- There is no silent backend switch: if the driver cannot be initialised the
  reason is printed and the run continues on the CPU. The `backend   :` line on
  stderr says which engine ran.
- `LA_MAX_NEW` just caps generation; the model stops at EOS on its own. On the
  test fixture that was 31 tokens, but the count depends on the query and image.

## Performance

Measured on this machine (i7-13700K, 24 threads; GTX 1080, sm_61) on the 448x448
test fixture (1024 patches, 297-token prompt, 31 generated tokens), with the
model already in the OS page cache. Each `detect` is a complete, standalone run
that loads the model from disk and, on the GPU, uploads it to VRAM.

| | CUDA (GTX 1080) | CPU (24 threads) |
| --- | --- | --- |
| model load (read + upload) | 7.9 s | ~1.5 s |
| ViT (MoonViT, 27 layers) | 1.14 s | 5.97 s |
| prefill | 1.07 s | 5.62 s |
| decode (31 tokens) | 1.2 s (~40 ms/tok) | 2.2 s (~66 ms/tok) |
| **wall, whole command** | **10.7 s** | **14.2 s** |

- The GPU **model load** is dominated by reading the 4.16 GB container and
  streaming it into VRAM. With the file warm in the page cache that is ~2 s of
  upload; the first run after boot adds seconds while it is read from storage.
- The CPU backend maps the container in place and faults pages in on demand, so
  its model-load cost is much smaller but its compute is slower.
- "decode" is the 30 steps after prefill; the per-token figure excludes the
  one-time prefill.
- Peak weight footprint is the 4.36 GB GPU arena (or the 4.16 GB mmap on CPU).
  Total process memory is that plus activations and the KV cache, which grow
  with image size.

These numbers are for one hardware/configuration; treat them as indicative, not
a benchmark.

## The model container

The model is a single `.laqt` file. The format (magic `LAQTGGUF`) is a JSON
header followed by 4096-byte-aligned tensor payloads; the header carries the
model config and the BPE tokenizer tables, so there are no sidecar files.

Weights are quantized with one rule: a rank-2 tensor whose contiguous dimension
is a multiple of 32 is stored as **q8_0** (the ggml block layout), everything
else stays **f32**. That is more aggressive than the C++ reference's GGUF and is
what makes the file smaller:

| | reference q8_0 GGUF | this container |
| --- | --- | --- |
| file size | 6.26 GB | **4.16 GB** |
| q8_0 payloads | 253 (3.28 GB) | 336 (3.59 GB) |
| f32 payloads | 517 (2.97 GB) | 433 (0.56 GB) |
| tied output head | stored twice | stored once, aliased |

So the container is about 34% smaller than the reference q8_0 GGUF and about 12%
smaller than its q4_k file while carrying 8-bit weights. The savings come from
quantizing 84 tensors the GGUF leaves in f32 (the ViT `wqkv`/`wo`/`fc0` weights,
the token embedding and the projector) and from storing the tied LM head once.
Tensors that cannot be quantized by the rule (e.g. `vit fc1`, whose inner
dimension is 4304) stay f32 in both. On the tested fixture the smaller file
produces the same detections as the reference ([Verification](#verification)).

## Building from source

```
cargo build --release
```

The build compiles the toolkit's `cuda/kernels.cu` with `nvcc` into a fatbin
embedded in the executable (SASS for sm_61/75/80 plus PTX for 8.0), generating
code for only the 24 kernels this engine calls. `NVCC=/path/to/nvcc`
overrides the compiler. The CUDA toolkit is a **build-time** dependency only;
the resulting binary needs only the driver. For a machine without any CUDA
toolchain:

```
cargo build --release --no-default-features
```

That omits the fatbin entirely and always runs the CPU backend.

The model tools are dependency-light Python (numpy only): `tools/prequant.py`
builds a container from a HuggingFace snapshot, `tools/verify_container.py`
cross-checks one against a reference GGUF, `tools/laformat.py` is the format
reader/writer and `tools/namemap.py` the tensor-name map. Useful `prequant`
flags: `--verify-tensors` reports the worst dequantization error, `--tensors
'proj.*,lm.blk.0.*'` builds a small smoke-test container, `--from-gguf ref.gguf`
borrows the config and tokenizer from an existing GGUF.

## How it works

Two independent backends share one graph:

- **CUDA** (`src/graph.rs`, plus the toolkit's `cuda/kernels.cu`): hand-written
  kernels — a dp4a q8_0 GEMM over a 36-byte-aligned block layout, warp-per-row
  decode GEMV, and shared-memory-staged flash attention. Weights are repacked
  from the native 34-byte ggml block to a 4-byte-aligned 36-byte stride on upload
  (measured ~3x faster on the large GEMMs) and streamed into a single VRAM
  arena.
- **CPU** (`src/cpu.rs`): the same kernels as rayon-parallel Rust over the
  native 34-byte blocks. Nothing is uploaded or repacked; the container is read
  directly from the mmap. It is the fallback for a machine with no GPU, not a
  test harness: it is tuned on its own terms, and the two backends are held to
  the tolerance published in [Verification](#verification).

The engine: the image is patchified and run through the 27-layer MoonViT, a
2-layer MLP connector projects the merged features into the language model's
embedding space, and a 36-layer Qwen2 decoder generates box tokens greedily,
which are parsed into coordinates.

## GPU requirements

The fatbin carries SASS for compute capability 6.1, 7.5 and 8.0 plus PTX for
8.0, so a GPU of one of those architectures is required; later architectures are
not guaranteed to have embedded code. It was developed and tuned on a GTX 1080
(sm_61). For the 448x448 test image the weights arena (4.36 GB) plus scratch and
KV cache fit in about 6 GB of VRAM; larger images need more. The loader checks
the arena against `cuMemGetInfo` before allocating and reports a sentence rather
than an out-of-memory error if it will not fit.

## Verification

The engine was validated against the C++ reference port on a fixed fixture (the
448x448 test image, the `cat</c>remote.` query). Claims here are scoped to that
fixture, not a general accuracy benchmark.

| comparison | result |
| --- | --- |
| GPU stdout vs the C++ reference CLI (`--mode slow`) | byte-identical JSON |
| GPU stdout vs `tests/ref_cli_q8_slow.json` | byte-identical body |
| container vs reference GGUF (`tools/verify_container.py`) | 686 payloads byte-identical; the 84 requantized tensors within the q8_0 block error (worst 2.8e-03) |
| GPU vs CPU, `cat` boxes | exactly equal |
| GPU vs CPU, `remote` boxes | differ by ~1 coordinate unit (~0.45 px); the CPU boxes match the C++ CPU-only reference |

The GPU and CPU ViT differ by about 2e-5 relative per layer, amplified through
27 layers of int8 activation quantization (a chaotic rounding difference, not an
accumulation-order one). On the fixture this does not change the predicted
boxes; it is the known limit of CPU/GPU numerical agreement, not a correctness
guarantee for arbitrary inputs.

`tests/test_laformat.py` covers the container format (quantization, layout,
aliasing):

```
python3 tests/test_laformat.py
```

## Credits

The model is NVIDIA's
[`LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B), an
open-vocabulary detection / visual-grounding VLM built from Qwen2.5-3B, the
MoonViT vision encoder and a 2-layer MLP connector.

The container format and tensor naming follow
[`locate-anything.cpp`](https://github.com/mudler/locate-anything.cpp), a
C++17/ggml port by Ettore Di Giacinto and the LocalAI team, which let this
engine be validated tensor-for-tensor and output-for-output against a known-good
reference.

The software here is MIT-licensed. NVIDIA's separate non-commercial licence
applies to the model weights downloaded by `get-model.sh`.
