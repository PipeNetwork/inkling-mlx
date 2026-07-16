"""Prepare + upload a REAP-pruned Inkling MLX build to pipenetwork/Inkling-MLX-<name>.

REAP (Cerebras, arXiv:2510.13999) drops the lowest-saliency routed experts per MoE
layer, where saliency = mean over active tokens of gate_weight * ||expert_output||.
Bundles the `inkling_mlx` loader and writes a model card with the measured results.

    python scripts/upload_reap.py REAP12-4bit /path/Inkling-REAP12-4bit
"""
import glob
import os
import shutil
import sys

from huggingface_hub import HfApi, create_repo

REPO_OWNER = "pipenetwork"
PKG_DIR = os.path.join(os.path.dirname(__file__), "..", "inkling_mlx")

# name -> (kept experts, prune %, on-disk size, ppl, delta vs unpruned, saliency retained, tag)
BUILDS = {
    "REAP12-4bit": (225, 12, "~470 GB", 3.815, "-0.4%", "97.4%", "free lunch — no measurable quality loss"),
    "REAP25-4bit": (192, 25, "~402 GB", 3.896, "+1.7%", "91.8%", "sweet spot — small cost, clears the 512 GB memory cliff"),
    "REAP50-4bit": (128, 50, "~272 GB", 4.589, "+19.8%", "74.3%", "aggressive / experimental — visibly degraded"),
}


def _table():
    rows = ["| Build | Experts kept | Size | Perplexity | vs unpruned |",
            "|---|---:|---:|---:|---:|",
            "| [Inkling-MLX-4bit](https://huggingface.co/pipenetwork/Inkling-MLX-4bit) (unpruned) | 256 | ~490 GB | 3.830 | — |"]
    for n, (k, _p, sz, ppl, dl, _r, _t) in BUILDS.items():
        rows.append(f"| [Inkling-MLX-{n}](https://huggingface.co/{REPO_OWNER}/Inkling-MLX-{n}) | {k} | {sz} | {ppl} | {dl} |")
    return "\n".join(rows)


def model_card(name: str) -> str:
    kept, prune, size, ppl, delta, retained, tag = BUILDS[name]
    warn = ""
    if name == "REAP50-4bit":
        warn = ("\n> **⚠️ Experimental / aggressive build.** At 50% pruning perplexity rises "
                "~20% over the unpruned 4-bit. It still answers simple prompts coherently, but "
                "quality is visibly reduced on prose and longer reasoning. Prefer **REAP12** or "
                "**REAP25** unless you specifically need the smallest footprint.\n")
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
- reap
- pruned
---

# Inkling-MLX-{name}

**Built with Inkling (Thinking Machines Lab).**

A **REAP-pruned**, 4-bit MLX build of
[thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling):
each MoE layer keeps its **{kept} highest-saliency routed experts** (of 256), a
**{prune}% expert prune**. {tag.capitalize()}.
{warn}
**Code / loader:** [github.com/PipeNetwork/inkling-mlx](https://github.com/PipeNetwork/inkling-mlx)

## What is REAP pruning?

[REAP (Router-weighted Expert Activation Pruning, Cerebras, arXiv:2510.13999)](https://arxiv.org/abs/2510.13999)
ranks each routed expert by **saliency** = mean over the tokens that route to it of
`router_gate_weight × ‖expert_output‖₂` — its actual contribution to the residual
stream. The lowest-saliency experts are dropped; the router simply renormalizes over
the survivors (no weight surgery). The **2 shared "sink" experts, attention, and
embeddings are untouched.**

Saliency here was measured over a ~50k-token calibration mix (real code + 15 languages
+ math/reasoning). Inkling routes **very uniformly** (routing entropy 0.922; only ~3
truly-cold experts per layer), so it is only *lightly* prunable — reflected below.

## Measured quality (4-bit, perplexity on a fixed held-out set)

{_table()}

This build: **perplexity {ppl} ({delta} vs the unpruned 4-bit)**, {retained} of
router-weighted expert contribution retained. Pruning is applied to the already-quantized
build; because expert subsetting is along the expert axis and affine-quant groups run
along the hidden axis, it is **bit-identical to pruning the bf16 source then requantizing**.

## ⚠️ Loading requires the bundled `inkling_mlx` loader

The `inkling_mm_model` architecture is **not** in stock `mlx-lm` / `mlx-vlm`, so this
repo bundles a minimal, numerically-validated MLX implementation under `inkling_mlx/`.
The reduced expert count is recorded in `config.json` (`n_routed_experts = {kept}`) and
the loader builds the model to match automatically.

```bash
pip install mlx mlx-lm transformers
```
```python
from inkling_mlx.load import load
from inkling_mlx.generate import greedy_generate
from transformers import AutoTokenizer

model, config = load("/path/to/this/repo")            # eager wired load fits comfortably
tok = AutoTokenizer.from_pretrained("/path/to/this/repo", trust_remote_code=True)
ids = tok("The capital of France is")["input_ids"]
print(tok.decode(greedy_generate(model, config, ids, max_new_tokens=64)))
```

Needs an Apple-Silicon Mac with unified memory ≥ the size above. The smaller footprint
(vs the 496 GB unpruned 4-bit) is the practical point: **{size}** loads eager/wired-resident
on a 512 GB machine without the memory-ceiling thrash.

## Details

- Multimodal (HMLP vision + dMel audio towers + preprocessing) is included, same as the
  base MLX build; the multi-token-prediction head is dropped.
- Quantized: attention / MLP / expert projections, embed+unembed, vision/audio matmuls.
  Kept higher precision: MoE router, RMSNorms, the four short-convolutions per layer,
  relative-position bias.

License: Apache-2.0 (inherits the base model).
"""


def main():
    name = sys.argv[1]                       # e.g. "REAP12-4bit"
    src = sys.argv[2]                         # local build dir
    assert name in BUILDS, f"unknown build {name}"
    repo = f"{REPO_OWNER}/Inkling-MLX-{name}"

    create_repo(repo, repo_type="model", private=False, exist_ok=True)

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
