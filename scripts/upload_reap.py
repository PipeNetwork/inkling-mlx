"""Prepare + upload a REAP-pruned Inkling MLX build to pipenetwork/Inkling-MLX-<name>.

REAP (Cerebras, arXiv:2510.13999) drops the lowest-saliency routed experts per MoE
layer, where saliency = mean over active tokens of gate_weight * ||expert_output||.
Bundles the `inkling_mlx` loader and writes a model card with the measured results.

    python scripts/upload_reap.py REAP12-4bit /path/Inkling-REAP12-4bit
"""
import glob
import json
import os
import shutil
import sys

from huggingface_hub import HfApi, create_repo

REPO_OWNER = "pipenetwork"
PKG_DIR = os.path.join(os.path.dirname(__file__), "..", "inkling_mlx")

# Per-family REAP results. Every number here is MEASURED on the published build —
# nothing is carried over between family members, since routing statistics and
# prunability differ. Build entry:
#   name -> (kept, prune %, size, text ppl, ppl delta, saliency retained, vision, audio, tag)
# Calibration is MULTIMODAL (text + images + audio) — see model card.
FAMILIES = {
    "Inkling": {
        "base_model": "thinkingmachines/Inkling",
        "hidden_size": 6144,
        "unpruned": ("4bit", 256, "~490 GB", 3.830),
        "routing": "routing entropy 0.922; only ~1 cold expert per layer under multimodal calibration",
        "footprint_note": "**{size}** loads eager/wired-resident on a 512 GB machine "
                          "without the memory-ceiling thrash (vs the 496 GB unpruned 4-bit)",
        "calibration_note": """Inkling is multimodal, and expert saliency was profiled over a mixed corpus of **text
(code + 15 languages + reasoning), 200 real images, and 180 speech clips** run through
the full vision and audio paths. This is deliberate: a **text-only** calibration prunes
experts that ground *visual* features (a Pallas's cat → *"brown bear"*, a golf ball →
*"butterfly"*); adding only text+image then leaves *audio*-grounding experts unprotected
(speech transcription word-overlap fell from 0.88 to 0.57 at 25% pruning) — all while
text perplexity looked fine the whole time. Profiling over all three modalities keeps
every expert that matters to any of them. On held-out tests this build scores **vision
{vis}** (vs 2/6 text-only) and **audio {aud}** overlap (vs 0.57 text+image), at no extra
text cost.""",
        "builds": {
            "REAP12-4bit": (225, 12, "~470 GB", 3.806, "-0.6%", "96.2%", "6/6", "0.88", "free lunch — text, vision AND audio intact"),
            "REAP25-4bit": (192, 25, "~402 GB", 3.946, "+3.0%", "90.3%", "6/6", "0.87", "sweet spot — clears the 512 GB memory cliff"),
            "REAP50-4bit": (128, 50, "~272 GB", 4.682, "+22.2%", "75.0%", "5/6", "0.87", "aggressive / experimental — text degraded"),
        },
    },
    # Inkling-Small entries are populated from the measured eval — see
    # scripts/eval_build.py. Left empty until then so nothing is published unmeasured.
    "Inkling-Small": {
        "base_model": "thinkingmachines/Inkling-Small",
        "hidden_size": 4096,
        "unpruned": ("4bit", 256, "~148 GB", 5.452),
        "routing": "routing entropy 0.908; only ~0.15 cold experts per layer once audio "
                   "is included in the calibration",
        "footprint_note": "**{size}** fits a 128 GB Mac, where the unpruned 148 GB "
                          "4-bit build does not",
        "calibration_note": """Inkling-Small is multimodal, and expert saliency was profiled over a mixed corpus of
**text (code + 15 languages + reasoning), 200 real images, and 180 speech clips** run
through the full vision and audio paths. On this model that is not optional: **47.7
experts per layer are >50% audio-driven and 22.3 are >50% image-driven** — about 27% of
all experts serve primarily non-text input, and adding audio to the calibration dropped
the cold-expert count from 0.65 to 0.15 per layer. Experts that only ever fire on speech
look worthless to a text-only profiler and get pruned first, which is how the 975B model
lost speech transcription (word-overlap 0.88 → 0.57) while its text perplexity still
looked fine. Profiling all three modalities keeps them. On held-out tests this build
scores **vision {vis}** and **audio {aud}** word-overlap — both at the unpruned build's
level.""",
        "ppl_caveat": """
**Read that −8.4% as "no measurable change", not as pruning improving the model.**
Perplexity was measured on two independent held-out sets: this one has REAP-25 at
**−8.4%** vs the unpruned 4-bit, a second one has it at **+0.55%**. Sets that small
disagree by a few percent in either direction, so the defensible claim is that a 25%
expert prune costs nothing measurable here — not that it helps.

**50% pruning is a different story and is deliberately not published.** It was built and
evaluated: text perplexity roughly **doubled** (+88% / +105% on the two sets) and speech
transcription fell from **0.874 to 0.702**, so it was dropped rather than shipped with a
warning. Inkling-Small is meaningfully less prunable than the 975B model, whose REAP-50
cost ~22%.""",
        "builds": {
            "REAP25-4bit": (192, 25, "~112 GB", 4.992, "−8.4%", "87.5%", "6/6", "0.896",
                            "no measurable loss — the build to take for a 128 GB Mac"),
        },
    },
}


def detect_family(src: str) -> str:
    cfg = json.load(open(os.path.join(src, "config.json")))
    h = cfg.get("text_config", {}).get("hidden_size")
    for fam, meta in FAMILIES.items():
        if meta["hidden_size"] == h:
            return fam
    raise SystemExit(f"unknown Inkling family: text_config.hidden_size={h}")


def _table(family: str):
    meta = FAMILIES[family]
    uname, ukept, usize, uppl = meta["unpruned"]
    rows = ["| Build | Experts kept | Size | Text ppl | vs unpruned | Vision (image ID) | Audio (speech overlap) |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| [{family}-MLX-{uname}](https://huggingface.co/{REPO_OWNER}/{family}-MLX-{uname}) (unpruned) | {ukept} | {usize} | {uppl} | — | ✓ | ✓ |"]
    for n, (k, _p, sz, ppl, dl, _r, vis, aud, _t) in meta["builds"].items():
        rows.append(f"| [{family}-MLX-{n}](https://huggingface.co/{REPO_OWNER}/{family}-MLX-{n}) | {k} | {sz} | {ppl} | {dl} | {vis} | {aud} |")
    return "\n".join(rows)


def model_card(family: str, name: str) -> str:
    meta = FAMILIES[family]
    kept, prune, size, ppl, delta, retained, vis, aud, tag = meta["builds"][name]
    warn = ""
    if name == "REAP50-4bit":
        warn = ("\n> **⚠️ Experimental / aggressive build.** At 50% pruning **text** perplexity "
                "rises ~22% over the unpruned 4-bit, and fine-grained image ID slips a little "
                "(5/6 vs 6/6). Audio transcription still holds (0.87). It answers simple prompts "
                "coherently but text quality is visibly reduced on prose and longer reasoning. "
                "Prefer **REAP12** or **REAP25** unless you specifically need the smallest footprint.\n")
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
- reap
- pruned
- audio-text-to-text
---

# {family}-MLX-{name}

**Built with Inkling (Thinking Machines Lab).**

A **REAP-pruned**, 4-bit MLX build of
[{meta["base_model"]}](https://huggingface.co/{meta["base_model"]}):
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
embeddings are untouched.** Inkling routes **very uniformly** ({meta["routing"]}), so it
is only *lightly* prunable — reflected below.

## Calibrated on text, images **and audio** (this matters)

{meta["calibration_note"].format(vis=vis, aud=aud)}

## Measured quality (4-bit)

{_table(family)}

This build: **text perplexity {ppl} ({delta} vs the unpruned 4-bit)**, **vision {vis}**
(held-out image ID), **audio {aud}** (held-out speech transcription word-overlap),
{retained} of router-weighted expert contribution retained.
{meta.get("ppl_caveat", "")} Pruning is applied to the
already-quantized build; because expert subsetting is along the expert axis and
affine-quant groups run along the hidden axis, it is **bit-identical to pruning the bf16
source then requantizing**.

## Quantization scheme: affine int4 (not NVFP4 / MXFP4)

MLX supports FP4 modes and Thinking Machines ships an
[Inkling-NVFP4](https://huggingface.co/thinkingmachines/Inkling-NVFP4) checkpoint — so
for the record, we benchmarked round-trip reconstruction error (‖W − Ŵ‖ / ‖W‖ vs bf16)
on real Inkling expert weights:

| Scheme | bits/weight | reconstruction error |
|---|---:|---:|
| **affine int4** (group 64) | 4.50 | **~9.1%** |
| nvfp4 (group 16) | 4.50 | ~10.2% |
| mxfp4 (group 32) | 4.25 | ~12.3% |

Affine int4 is the most faithful: it is *asymmetric* (per-group scale **and** zero-point,
16 uniform levels), which centers on Inkling's near-Gaussian expert weights better than
symmetric FP4's fixed non-uniform levels (scale only, no zero-point). FP4's real payoff is
heavy-tailed *activations* and native Blackwell FP4 tensor cores — neither helps weight
fidelity on Apple Silicon, where MLX would dequantize FP4 anyway. So these builds use
affine int4; a Mac port of the NVFP4 checkpoint would be *lower* quality at best-equal size.

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

Needs an Apple-Silicon Mac with unified memory ≥ the size above. The smaller footprint is
the practical point: {meta["footprint_note"].format(size=size)}.

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
    family = sys.argv[3] if len(sys.argv) > 3 else detect_family(src)
    builds = FAMILIES[family]["builds"]
    if name not in builds:
        raise SystemExit(
            f"{family}: no measured results for {name!r}. Populate FAMILIES[{family!r}]"
            f"['builds'] from scripts/eval_build.py output first — these cards state "
            f"measured perplexity/vision/audio numbers and must not be published unmeasured."
        )
    repo = f"{REPO_OWNER}/{family}-MLX-{name}"

    create_repo(repo, repo_type="model", private=False, exist_ok=True)

    pkg_dst = os.path.join(src, "inkling_mlx")
    os.makedirs(pkg_dst, exist_ok=True)
    for f in glob.glob(os.path.join(PKG_DIR, "*.py")):
        shutil.copy2(f, pkg_dst)

    with open(os.path.join(src, "README.md"), "w") as fh:
        fh.write(model_card(family, name))

    api = HfApi()
    api.upload_large_folder(repo_id=repo, folder_path=src, repo_type="model")
    print(f"UPLOADED {repo}")


if __name__ == "__main__":
    main()
