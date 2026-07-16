"""REAP-prune an ALREADY-CONVERTED MLX build (bit-identical to pruning the bf16
source then re-quantizing, because expert subsetting is along axis 0 while affine
quant groups run along the hidden dim — the two are independent).

Streams the source build ONCE and writes several pruned builds at different keep
ratios in the same pass. For each MoE layer it drops the lowest-saliency routed
experts (REAP), subsetting the quantized weight+scales+biases together and the
router rows (kept routed + shared), and writes a reduced ``n_routed_experts``.

    python scripts/prune_build.py            # real 3-way run (REAP12/25/50)
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import mlx.core as mx

_LRE = re.compile(r"model\.llm\.layers\.(\d+)\.")
_N_SHARED = 2
_SHARD_CAP = 5_000_000_000


def subset(name: str, w: mx.array, keep, dmi: int) -> mx.array:
    """Subset one converted-build tensor to the kept experts. ``keep`` maps
    sparse-layer index -> kept routed-expert indices. Routed experts (weight/
    scales/biases) and router rows are subset; shared experts are untouched."""
    m = _LRE.search(name)
    if m is None:
        return w
    L = int(m.group(1))
    if L < dmi:
        return w
    kidx = mx.array(keep[L - dmi])
    leaf = name.rsplit(".", 1)[-1]
    if ".mlp.experts." in name and leaf in ("weight", "scales", "biases"):
        return w[kidx]                                   # [E, ...] -> [K, ...]
    if name.endswith("mlp.gate.weight"):                 # [n_routed + n_shared, hidden]
        n_routed = w.shape[0] - _N_SHARED
        return mx.concatenate([w[kidx], w[n_routed:]], axis=0)
    if name.endswith("mlp.gate.bias"):
        return w[kidx]
    return w                                             # shared_experts.*, norms, attn, ...


def keeps_from_usage(usage_path, ratios: dict):
    """ratios: {name: prune_ratio} -> {name: (keep[NL,K], K)} using REAP saliency."""
    d = np.load(usage_path)
    counts, sal = d["counts"].astype(np.float64), d["saliency"].astype(np.float64)
    dmi = int(d["dense_mlp_idx"]); NL, NE = sal.shape
    S = np.where(counts == 0, -1.0, sal / np.maximum(counts, 1.0))
    order = np.argsort(-S, axis=1)
    out = {}
    for name, r in ratios.items():
        K = int(round(NE * (1.0 - r)))
        keep = np.sort(order[:, :K], axis=1).astype(np.int32)
        retained = np.take_along_axis(sal, order[:, :K], 1).sum(1) / np.maximum(sal.sum(1), 1e-9)
        out[name] = (keep, K, float(retained.mean()), float(retained.min()))
    return out, dmi, NE


def _aux(src, dst):
    for pat in ("tokenizer*", "special_tokens_map.json", "*.tiktoken", "tiktoken",
                "chat_template.jinja", "processor_config.json", "preprocessor_config.json"):
        for p in glob.glob(os.path.join(src, pat)):
            t = os.path.join(dst, os.path.basename(p))
            shutil.copytree(p, t, dirs_exist_ok=True) if os.path.isdir(p) else shutil.copy2(p, t)


def _write_cfg(src, dst, new_ne):
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["text_config"]["n_routed_experts"] = new_ne
    cfg.setdefault("reap", {})["kept_experts"] = new_ne
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)


def prune_builds(src, outputs, usage_path):
    """outputs: {name: out_dir}. Single streaming pass over ``src``."""
    ratios = {n: r for n, (r, _) in outputs.items()}
    keeps, dmi, NE = keeps_from_usage(usage_path, ratios)
    print(f"[prune] source experts={NE}, dense_mlp_idx={dmi}", flush=True)
    for name in outputs:
        keep, K, rmean, rmin = keeps[name]
        print(f"[prune] {name}: keep {K}/{NE}  retained mean {rmean*100:.1f}% min {rmin*100:.1f}%", flush=True)

    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shard_to_names: dict[str, list[str]] = {}
    for n, s in weight_map.items():
        shard_to_names.setdefault(s, []).append(n)

    # per-output streaming write state
    state = {name: dict(dir=outputs[name][1], idx={}, buf={}, bytes=0, sid=0) for name in outputs}
    for st in state.values():
        os.makedirs(st["dir"], exist_ok=True)

    def flush(st):
        if not st["buf"]:
            return
        st["sid"] += 1
        f = f"model-{st['sid']:05d}.safetensors"
        mx.save_safetensors(os.path.join(st["dir"], f), st["buf"], metadata={"format": "mlx"})
        for k in st["buf"]:
            st["idx"][k] = f
        st["buf"] = {}; st["bytes"] = 0

    for shard in sorted(shard_to_names):
        tensors = mx.load(os.path.join(src, shard))  # mmap
        for nm in shard_to_names[shard]:
            w = tensors[nm]
            for name in outputs:
                keep = keeps[name][0]
                arr = subset(nm, w, keep, dmi)
                mx.eval(arr)
                st = state[name]
                st["buf"][nm] = arr
                st["bytes"] += arr.nbytes
                if st["bytes"] >= _SHARD_CAP:
                    flush(st)
        del tensors
    for name in outputs:
        st = state[name]
        flush(st)
        # finalize index with -of- names
        n = st["sid"]
        remap = {}
        for i in range(1, n + 1):
            old = f"model-{i:05d}.safetensors"; new = f"model-{i:05d}-of-{n:05d}.safetensors"
            if os.path.exists(os.path.join(st["dir"], old)):
                os.rename(os.path.join(st["dir"], old), os.path.join(st["dir"], new))
            remap[old] = new
        wm = {k: remap[v] for k, v in st["idx"].items()}
        total = sum(os.path.getsize(os.path.join(st["dir"], f)) for f in set(wm.values()))
        json.dump({"metadata": {"total_size": total}, "weight_map": wm},
                  open(os.path.join(st["dir"], "model.safetensors.index.json"), "w"), indent=2)
        _write_cfg(src, st["dir"], keeps[name][1])
        _aux(src, st["dir"])
        print(f"[prune] wrote {name} -> {st['dir']} ({n} shards, {total/1e9:.0f} GB)", flush=True)


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)   # CPU indexing: no Metal watchdog on the long stream
    SRC = "/Users/david/llm/inkling-mlx-out/Inkling-4bit"
    USAGE = "/Users/david/llm/inkling-mlx-out/expert_usage.npz"
    B = "/Users/david/llm/inkling-mlx-out"
    OUTPUTS = {
        "REAP12": (0.12, f"{B}/Inkling-REAP12-4bit"),
        "REAP25": (0.25, f"{B}/Inkling-REAP25-4bit"),
        "REAP50": (0.50, f"{B}/Inkling-REAP50-4bit"),
    }
    prune_builds(SRC, OUTPUTS, USAGE)
    print("[prune] ALL DONE", flush=True)
