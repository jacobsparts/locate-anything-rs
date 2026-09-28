# locate-anything-rs

One of the [lightgpu inference engines](https://github.com/jacobsparts/lightgpu).

Open-vocabulary object detection in one self-contained binary: give it a PNG and
a phrase and it prints labeled bounding boxes as JSON. No Python, PyTorch, or
CUDA toolkit needed.

```
$ ./locate-anything detect photo.png \
    "Locate all the instances that matches the following description: cat."
{"detections":[{"label":"cat","box":[0.00,50.18,221.76,443.07]}]}
```

* Both backends in one executable: a pure-Rust CPU path and a CUDA path with
  hand-written kernels. The GPU is used when the CUDA driver can be brought up
  and the CPU path when it cannot (no silent switch - the reason is printed and
  the `backend:` line on stderr says which ran); `LA_CPU=1` forces the CPU.
* A single Linux x86-64 executable that links only `libc`, `libm` and
  `libgcc_s`.
* The model is NVIDIA
  [LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B); this
  repository is an independent Rust inference engine for it, validated
  output-for-output against the C++/ggml reference port.

On the tested fixture the JSON output is byte-identical to the reference port.

## Download

The prebuilt binary is attached to the
[release](https://github.com/jacobsparts/locate-anything-rs/releases).

| asset | what it is |
|---|---|
| `locate-anything-linux-x86_64` | the engine: x86-64 Linux with glibc >= 2.34 (Ubuntu 22.04+, Debian 12+); the GPU path needs an NVIDIA GPU of compute capability 6.1/7.5/8.0 and ~6 GB VRAM for the test image |

```sh
chmod +x locate-anything-linux-x86_64
./locate-anything-linux-x86_64 detect photo.png "Locate all ... : cat."
```

The model is **not** in the release - the 4.16 GB container is past GitHub's
asset limit. Build it once (needs Python 3, numpy and curl, ~12 GB disk during
the build):

```sh
curl -fsSL https://raw.githubusercontent.com/jacobsparts/locate-anything-rs/main/get-model.sh | bash
# or, from a checkout:  ./get-model.sh
```

It downloads NVIDIA's 7.7 GB checkpoint into `model-source/` and quantizes it
into a single `locate-anything-allq8_0.laqt` next to the binary, then
`model-source/` can be deleted. The build is deterministic (md5
`85909dfebaf0d9efaf973137007940c2`). The model file is found automatically next
to the binary or in `models/`; point elsewhere with `--model <path>`.

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

`detect` writes one line of compact JSON to stdout and every diagnostic to
stderr, so stdout pipes straight into a parser:

```
$ ./locate-anything detect photo.png "...description: cat</c>remote." 2>/dev/null
{
  "detections": [
    {"label": "cat",    "box": [3.14, 50.62, 221.76, 443.52]},
    {"label": "remote", "box": [27.78, 68.99, 122.75, 109.76]}
  ]
}
```

### The query

The model was instruction-tuned with a fixed prompt template; the binary wraps
your query in it, so you supply only the sentence:

```
"Locate all the instances that matches the following description: <description>."
```

Separate multiple classes with `</c>` to detect them in one pass
(`...description: cat</c>remote.`).

### Output coordinates

Each `box` is `[x_min, y_min, x_max, y_max]` in the **preprocessed image's**
pixel space: before inference the image is bicubic-scaled so each side is a
multiple of 28 (and large inputs downscaled to a 25600-patch token budget). The
exact resized dimensions are printed on stderr; scale a box by
`orig_w / target_w`, `orig_h / target_h` to draw it on the original.

### Environment variables

| variable | effect |
|---|---|
| `LA_CPU=1` | force the CPU backend (also the automatic fallback) |
| `LA_MAX_NEW` | decode token budget (default 1024; the model stops at EOS on its own) |

## The model container

The model is a single `.laqt` file - a JSON header carrying the model config and
the BPE tokenizer tables, then 4096-byte-aligned tensor payloads. Weights follow
one quantization rule: a rank-2 tensor whose contiguous dimension is a multiple
of 32 is stored as q8_0, everything else stays f32. That makes the container
4.16 GB where the reference q8_0 GGUF is 6.26 GB (and its q4_k file 4.7 GB) - the
savings come from quantizing 84 tensors the GGUF leaves in f32 and from storing
the tied LM head once.

## Licence and attribution

The software here is MIT-licensed. The model is NVIDIA's
[`LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B), an
open-vocabulary detection VLM built from Qwen2.5-3B, the MoonViT vision encoder
and a 2-layer MLP connector. The container format and tensor naming follow
[`locate-anything.cpp`](https://github.com/mudler/locate-anything.cpp) (Ettore
Di Giacinto and the LocalAI team), which let this engine be validated
tensor-for-tensor and output-for-output against a known-good reference. NVIDIA's
non-commercial licence applies to the model weights downloaded by
`get-model.sh`.
