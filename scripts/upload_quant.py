"""Prepare + upload one Inkling MLX build to the Hub as pipenetwork/<Family>-MLX-<name>.

Handles both family members — `Inkling` (975B-A41B) and `Inkling-Small` (276B-A12B) —
detected from the build's own config.json, or forced with `--family`.

Bundles the `inkling_mlx` loader into the repo (the arch is not in stock mlx-lm/mlx-vlm)
and writes an accurate model card, then uploads the whole folder (resumable).

Usage:  python scripts/upload_quant.py <variant> <build-dir> [--family Inkling-Small]
"""

import argparse
import glob
import json
import os
import shutil

from huggingface_hub import HfApi, create_repo

REPO_OWNER = "pipenetwork"
PKG_DIR = os.path.join(os.path.dirname(__file__), "..", "inkling_mlx")

NOTES = {
    "bf16": "reference precision",
    "8bit": "near-lossless",
    "6bit": "high quality",
    "4bit": "balanced default",
    "3bit": "⚠️ experimental — visibly degraded",
}

# Measured text perplexity per build, one fixed held-out set (identical inputs
# across builds, so the deltas are directly comparable). None = not measured.
PPL = {
    "Inkling": {},
    "Inkling-Small": {"8bit": 5.569, "6bit": 5.569, "4bit": 5.452, "3bit": 6.706},
}

# Per-family facts. `variants` is the model-card table order; `sizes` is the
# fallback when a sibling build dir isn't on disk to measure.
FAMILIES = {
    "Inkling": {
        "base_model": "thinkingmachines/Inkling",
        "params": "975B-total / 41B-active",
        "variants": ["8bit", "6bit", "4bit"],
        "sizes": {"8bit": "~937 GB", "6bit": "~717 GB", "4bit": "~490 GB"},
        "bf16_note": "~1.9 TB",
        "hidden_size": 6144,
    },
    "Inkling-Small": {
        "base_model": "thinkingmachines/Inkling-Small",
        "params": "276B-total / 12B-active",
        "variants": ["bf16", "8bit", "6bit", "4bit", "3bit"],
        "sizes": {"bf16": "~527 GB", "8bit": "~280 GB", "6bit": "~214 GB",
                  "4bit": "~148 GB", "3bit": "~116 GB"},
        "bf16_note": "~527 GB",
        "hidden_size": 4096,
    },
}


def detect_family(src: str) -> str:
    """Identify the family from the build's config (hidden_size is unambiguous)."""
    cfg = json.load(open(os.path.join(src, "config.json")))
    h = cfg.get("text_config", {}).get("hidden_size")
    for fam, meta in FAMILIES.items():
        if meta["hidden_size"] == h:
            return fam
    raise SystemExit(f"unknown Inkling family: text_config.hidden_size={h}")


def _dir_size_gb(path: str) -> float:
    return sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(path)
        for f in fs
        if f.endswith(".safetensors")
    ) / 1e9


def measured_sizes(family: str, src: str) -> dict:
    """Prefer real on-disk sizes of the sibling builds; fall back to the table."""
    sizes = dict(FAMILIES[family]["sizes"])
    outroot = os.path.dirname(os.path.abspath(src))
    for v in FAMILIES[family]["variants"]:
        d = os.path.join(outroot, f"{family}-{v}")
        if os.path.isdir(d):
            gb = _dir_size_gb(d)
            if gb > 0:
                sizes[v] = f"~{gb:.0f} GB"
    return sizes


def model_card(family: str, name: str, sizes: dict) -> str:
    meta = FAMILIES[family]
    ppl = PPL.get(family, {})
    ppl_col = " Text ppl |" if ppl else ""
    ppl_sep = " ---: |" if ppl else ""
    rows = "\n".join(
        f"| [{n}](https://huggingface.co/{REPO_OWNER}/{family}-MLX-{n}) | {sizes[n]} |"
        + (f" {ppl[n]:.3f} |" if ppl.get(n) else (" — |" if ppl else ""))
        + f" {NOTES[n]} |"
        for n in meta["variants"]
    )
    header = f"| Variant | Size |{ppl_col} Notes |\n|---|---:|{ppl_sep}---|"

    warning = ""
    if name == "3bit":
        warning = """
> ### ⚠️ This build is experimental
>
> 3-bit measures **+20% text perplexity** vs 8-bit (6.706 vs 5.569). It answers direct
> factual and coding questions correctly, but after the answer it tends to fall into
> repetition loops and emit stray glyphs. At ~116 GB it also only *just* fits a 128 GB
> Mac, needing `iogpu.wired_limit_mb` raised close to the ceiling.
>
> **For the same ~112 GB footprint, take
> [REAP25-4bit](https://huggingface.co/pipenetwork/Inkling-Small-MLX-REAP25-4bit) instead.**
> It keeps 4-bit precision and drops 25% of the routed experts instead, which measured
> as *no* perplexity cost (vs this build's +20%), with vision and speech intact. This
> 3-bit build is kept only for the case where you want the full 256-expert set at that
> size. With more memory, plain
> [4-bit](https://huggingface.co/pipenetwork/Inkling-Small-MLX-4bit) shows no measurable
> loss vs 8-bit.
"""
    if name == "bf16":
        precision = (
            "converted to MLX at **bfloat16** — no quantization, the reference build "
            "for evaluating the quantized ones."
        )
        quant_para = """## Quantization

None. This is the unquantized bf16 conversion — useful as an evaluation reference or
as the source for your own quantization sweep. It does **not** fit in a single Mac's
unified memory; for builds that do, see the table above."""
    else:
        bits = name.replace("bit", "")
        precision = f"quantized to **{bits}-bit** (affine group quant, group size 64)."
        quant_para = """## Quantization scheme: affine int4 (not NVFP4 / MXFP4)

MLX supports FP4 modes and Thinking Machines ships an
[Inkling-NVFP4](https://huggingface.co/thinkingmachines/Inkling-NVFP4) checkpoint — so for
the record, we benchmarked round-trip reconstruction error (‖W − Ŵ‖ / ‖W‖ vs bf16) on real
Inkling expert weights:

| Scheme | bits/weight | reconstruction error |
|---|---:|---:|
| **affine int4** (group 64) | 4.50 | **~9.1%** |
| nvfp4 (group 16) | 4.50 | ~10.2% |
| mxfp4 (group 32) | 4.25 | ~12.3% |

Affine int4 is the most faithful: it is *asymmetric* (per-group scale **and** zero-point, 16
uniform levels), which centers on Inkling's near-Gaussian expert weights better than
symmetric FP4's fixed non-uniform levels. FP4's real payoff is heavy-tailed *activations* and
native Blackwell FP4 tensor cores — neither helps weight fidelity on Apple Silicon, where MLX
would dequantize FP4 anyway. So these builds use affine int4."""

    return f"""---
license: apache-2.0
base_model: {meta["base_model"]}
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

# {family}-MLX-{name}

**Built with Inkling (Thinking Machines Lab).**

MLX (Apple Silicon) conversion of
[{meta["base_model"]}](https://huggingface.co/{meta["base_model"]}),
{precision}

**Code / loader:** [github.com/PipeNetwork/inkling-mlx](https://github.com/PipeNetwork/inkling-mlx)
{warning}
{family.replace("-", " ") if family != "Inkling" else "Inkling"} is a **{meta["params"]}**
sparse-MoE, natively multimodal model (text + image/video + audio → text). This is the
**full multimodal** conversion: all three towers (text backbone, HMLP vision, dMel audio)
are ported; the multi-token-prediction head is dropped (inference-irrelevant).

## Builds

{header}
{rows}

{"Perplexity is teacher-forcing over one fixed held-out set (prose / code / reasoning / multilingual) — identical inputs across builds, so the columns compare directly. 4-bit shows no measurable loss vs 8-bit." if ppl else ""}

{"There is also a **REAP-pruned** build: [REAP25-4bit](https://huggingface.co/pipenetwork/Inkling-Small-MLX-REAP25-4bit) keeps 4-bit precision with 192 of 256 routed experts, fitting a **128 GB Mac** at ~112 GB for no measurable perplexity cost, with vision and speech intact." if family == "Inkling-Small" else ""}

{quant_para}

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

Conversion is streaming (tensor-by-tensor; the {meta["bf16_note"]} bf16 model never fully
loads into RAM) and was validated with fp32 numerical parity against transformers PR #47347.
License: Apache-2.0 (inherits the base model).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="variant, e.g. 4bit / 6bit / 8bit / bf16")
    ap.add_argument("src", help="local build dir")
    ap.add_argument("--family", choices=sorted(FAMILIES), default=None,
                    help="override the family detected from the build's config.json")
    args = ap.parse_args()

    family = args.family or detect_family(args.src)
    if args.name not in FAMILIES[family]["variants"]:
        raise SystemExit(f"{family}: unknown variant {args.name!r}")
    repo = f"{REPO_OWNER}/{family}-MLX-{args.name}"

    # fail fast on auth/repo before any large transfer
    create_repo(repo, repo_type="model", private=False, exist_ok=True)

    # bundle the loader package (only .py, no __pycache__)
    pkg_dst = os.path.join(args.src, "inkling_mlx")
    os.makedirs(pkg_dst, exist_ok=True)
    for f in glob.glob(os.path.join(PKG_DIR, "*.py")):
        shutil.copy2(f, pkg_dst)

    with open(os.path.join(args.src, "README.md"), "w") as fh:
        fh.write(model_card(family, args.name, measured_sizes(family, args.src)))

    api = HfApi()
    api.upload_large_folder(repo_id=repo, folder_path=args.src, repo_type="model")
    print(f"UPLOADED {repo}")


if __name__ == "__main__":
    main()
