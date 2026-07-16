"""Numerical parity: MLX TextModel vs standalone PyTorch reference, tiny random
config, shared weights, fp32. Validates the novel forward math (hybrid attention,
relative-position bias, log-scaling, short-convs, sigmoid MoE router + shared sink,
dense/sparse MLP, muP logits)."""

import numpy as np
import torch
import mlx.core as mx

import ref_torch as R
from inkling_mlx.config import TextConfig
from inkling_mlx.text import TextModel


def build_cfg():
    return {
        "hidden_size": 128, "num_hidden_layers": 6, "vocab_size": 300, "unpadded_vocab_size": 290,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "swa_head_dim": 32,
        "sliding_window_size": 6, "d_rel": 8, "rel_extent": 12,
        "log_scaling_n_floor": 128000, "log_scaling_alpha": 0.1,
        "sconv_kernel_size": 4, "dense_mlp_idx": 2,
        "dense_intermediate_size": 256, "moe_intermediate_size": 64,
        "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 2,
        "route_scale": 8.0, "rms_norm_eps": 1e-6, "logits_mup_width_multiplier": 24.0,
        # layers 0,1,2,4 sliding; 3,5 global (mix of both)
        "local_layer_ids": [0, 1, 2, 4],
    }


def cfg_with_layer_types(c):
    local = set(c["local_layer_ids"])
    c = dict(c)
    c["layer_types"] = ["hybrid_sliding" if i in local else "hybrid" for i in range(c["num_hidden_layers"])]
    c["mlp_layer_types"] = ["dense" if i < c["dense_mlp_idx"] else "sparse" for i in range(c["num_hidden_layers"])]
    return c


def t2m(t):
    return mx.array(t.detach().float().numpy())


def transfer(ref: "R.TextModel", mlx_model: TextModel, moe_inter: int):
    """Copy torch reference params into the MLX model (applying converter-style transforms)."""
    updates = {}
    for name, p in ref.named_parameters():
        arr = p
        # short-conv: torch conv1d weight [C,1,K] -> MLX [C,K,1]; drop ".conv1d"
        if name.endswith(".conv1d.weight"):
            mlx_name = name.replace(".conv1d.weight", ".weight")
            updates[mlx_name] = t2m(arr).swapaxes(1, 2)
            continue
        # fused routed experts gate_up_proj [E,2I,H] -> gate_proj/up_proj
        if name.endswith("experts.gate_up_proj"):
            base = name[: -len("gate_up_proj")]
            g = t2m(arr[:, :moe_inter, :])
            u = t2m(arr[:, moe_inter:, :])
            updates[base + "gate_proj.weight"] = g
            updates[base + "up_proj.weight"] = u
            continue
        # routed experts down_proj [E,H,I]
        if name.endswith("experts.down_proj") and "shared" not in name:
            updates[name.replace("experts.down_proj", "experts.down_proj.weight")] = t2m(arr)
            continue
        # shared experts separate gate/up/down (params, not Linears in ref)
        if name.endswith("shared_experts.gate_proj"):
            updates[name + ".weight"] = t2m(arr); continue
        if name.endswith("shared_experts.up_proj"):
            updates[name + ".weight"] = t2m(arr); continue
        if name.endswith("shared_experts.down_proj"):
            updates[name + ".weight"] = t2m(arr); continue
        # everything else identity
        updates[name] = t2m(arr)

    # apply into MLX model via update
    from mlx.utils import tree_unflatten
    mlx_model.update(tree_unflatten(list(updates.items())))
    return updates


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    raw = build_cfg()
    rcfg = cfg_with_layer_types(raw)

    ref = R.TextModel(rcfg).eval()
    # init params to reasonable random values
    for p in ref.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if "global_scale" in n:
                p.fill_(1.2)
            if n.endswith("gate.bias") or n.endswith(".bias"):
                p.normal_(std=0.02)

    tcfg = TextConfig.from_dict(raw)
    mlx_model = TextModel(tcfg)
    transfer(ref, mlx_model, raw["moe_intermediate_size"])
    mx.eval(mlx_model.parameters())

    ids_np = np.array([[3, 17, 42, 8, 100, 5, 9, 200, 1, 33]])
    with torch.no_grad():
        logits_ref = ref(torch.tensor(ids_np)).float().numpy()[0]
    logits_mlx = np.array(mlx_model(mx.array(ids_np)).astype(mx.float32))[0]

    diff = np.abs(logits_ref - logits_mlx)
    denom = np.abs(logits_ref).mean()
    print(f"logits shape ref={logits_ref.shape} mlx={logits_mlx.shape}")
    print(f"max|Δ|      = {diff.max():.3e}")
    print(f"mean|Δ|     = {diff.mean():.3e}")
    print(f"mean|ref|   = {denom:.3e}   (rel max Δ = {diff.max()/denom:.3e})")
    # argmax agreement per position (next-token consistency)
    agree = (logits_ref.argmax(-1) == logits_mlx.argmax(-1)).mean()
    print(f"argmax agreement over positions = {agree*100:.1f}%")
    ok = diff.max() < 1e-2 and agree == 1.0
    print("\nPARITY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
