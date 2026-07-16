"""Prepare + upload one Inkling MLX quant to the Hub as pipenetwork/Inkling-MLX-<name>.

Bundles the `inkling_mlx` loader into the repo (the arch is not in stock mlx-lm/mlx-vlm)
and writes an accurate model card, then uploads the whole folder (resumable)."""

import glob
import os
import shutil
import sys

from huggingface_hub import HfApi, create_repo

REPO_OWNER = "pipenetwork"
SIZES = {"8bit": "~937 GB", "6bit": "~717 GB", "4bit": "~490 GB"}
NOTES = {"8bit": "near-lossless", "6bit": "high quality", "4bit": "balanced default"}
PKG_DIR = os.path.join(os.path.dirname(__file__), "..", "inkling_mlx")


def model_card(name: str) -> str:
    bits = name.replace("bit", "")
    rows = "\n".join(
        f"| [{n}](https://huggingface.co/{REPO_OWNER}/Inkling-MLX-{n}) | {SIZES[n]} | {NOTES[n]} |"
        for n in ("8bit", "6bit", "4bit")
    )
    return f"""---
license: apache-2.0
base_model: thinkingmachines/Inkling
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx
tags:
- mlx
- moe
- multimodal
- inkling
- thinking-machines
---

# Inkling-MLX-{name}

**Built with Inkling (Thinking Machines Lab).**

MLX (Apple Silicon) conversion of
[thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling),
quantized to **{bits}-bit** (affine group quant, group size 64).

**Code / loader:** [github.com/PipeNetwork/inkling-mlx](https://github.com/PipeNetwork/inkling-mlx)

Inkling is a **975B-total / 41B-active** sparse-MoE, natively multimodal model
(text + image/video + audio → text). This is the **full multimodal** conversion:
all three towers (text backbone, HMLP vision, dMel audio) are ported; the
multi-token-prediction head is dropped (inference-irrelevant).

## Quantizations

| Variant | Size | Notes |
|---|---|---|
{rows}

## ⚠️ Loading requires the bundled `inkling_mlx` loader

The `inkling_mm_model` architecture is **not** in stock `mlx-lm` / `mlx-vlm`, so this
repo bundles a minimal, numerically-validated MLX implementation under `inkling_mlx/`.

```bash
pip install mlx mlx-lm transformers
```
```python
from inkling_mlx.load import load
from inkling_mlx.generate import greedy_generate
from transformers import AutoTokenizer

model, config = load("/path/to/this/repo")
tok = AutoTokenizer.from_pretrained("/path/to/this/repo", trust_remote_code=True)
ids = tok("The capital of France is")["input_ids"]
print(tok.decode(greedy_generate(model, config, ids, max_new_tokens=64)))
```

Needs an Apple-Silicon Mac with enough unified memory to hold the weights (≈ the
size above).

## Status & caveats

- **Text generation** works end-to-end via an incremental KV + short-convolution cache.
- **Multimodal** is supported end-to-end: the vision/audio towers and their
  preprocessing (`InklingProcessor` — image patchify/normalize, audio log-mel→dMel,
  validated ~1e-7 vs the reference) are included. Pass images/audio via the processor.
- Quantized: attention / MLP / expert projections, token embed+unembed, and the
  vision/audio matmuls. Kept in higher precision: the MoE router, RMSNorms, the four
  short-convolutions per layer, and the relative-position bias.

Conversion is streaming (tensor-by-tensor; the ~1.9 TB bf16 model never fully loads
into RAM) and was validated with fp32 numerical parity against transformers PR #47347.
License: Apache-2.0 (inherits the base model).
"""


def main():
    name = sys.argv[1]                       # e.g. "8bit"
    src = sys.argv[2]                         # local quant dir
    repo = f"{REPO_OWNER}/Inkling-MLX-{name}"

    # fail fast on auth/repo before any large transfer
    create_repo(repo, repo_type="model", private=False, exist_ok=True)

    # bundle the loader package (only .py, no __pycache__)
    pkg_dst = os.path.join(src, "inkling_mlx")
    os.makedirs(pkg_dst, exist_ok=True)
    for f in glob.glob(os.path.join(PKG_DIR, "*.py")):
        shutil.copy2(f, pkg_dst)

    with open(os.path.join(src, "README.md"), "w") as fh:
        fh.write(model_card(name))

    api = HfApi()
    api.upload_large_folder(repo_id=repo, folder_path=src, repo_type="model")
    print(f"UPLOADED {repo}")


if __name__ == "__main__":
    main()
