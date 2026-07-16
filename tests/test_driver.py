"""End-to-end test of the convert_model DRIVER: build a tiny model, write it out in
real *checkpoint* format (fused w13, [C,1,K] convs, model.llm/visual/audio names),
then convert_model -> load -> forward. Exercises sharding/flush/index/config plumbing."""

import json, os, shutil
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from inkling_mlx.config import InklingConfig
from inkling_mlx.model import InklingForConditionalGeneration
from inkling_mlx import convert as C
from inkling_mlx.load import load

SCRATCH = "/private/tmp/claude-501/-Users-david-llm/90565c15-6afa-4b84-8e85-b7fc0d50f667/scratchpad"
SRC = f"{SCRATCH}/fake-inkling-src"


def to_checkpoint(mlx_params: dict):
    """Reverse of convert.map_name/transform: MLX param dict -> checkpoint tensors."""
    ck = {}
    # group gate/up siblings to fuse back into w13
    keys = set(mlx_params)
    used = set()
    for k, v in mlx_params.items():
        if k in used:
            continue
        # convs: [C,K,1] -> [C,1,K]
        if k.endswith((".k_sconv.weight", ".v_sconv.weight", ".attn_sconv.weight", ".mlp_sconv.weight")):
            ck[k] = mx.swapaxes(v, 1, 2); continue
        # fuse gate_proj/up_proj -> w13 variant
        if k.endswith("gate_proj.weight"):
            base = k[: -len("gate_proj.weight")]
            up = mlx_params[base + "up_proj.weight"]
            # interleave gate/up rows -> [g0,u0,g1,u1,...] (checkpoint layout)
            fused = mx.stack([v, up], axis=-2).reshape(*v.shape[:-2], v.shape[-2] * 2, v.shape[-1])
            used.add(base + "up_proj.weight")
            if base.endswith("experts."):
                if base.endswith("shared_experts."):
                    ck[base + "shared_w13_weight"] = fused
                else:
                    ck[base + "w13_weight"] = fused
            else:  # dense mlp
                ck[base + "w13_dn.weight"] = fused
            continue
        if k.endswith("down_proj.weight"):
            base = k[: -len("down_proj.weight")]
            if base.endswith("shared_experts."):
                ck[base + "shared_w2_weight"] = v
            elif base.endswith("experts."):
                ck[base + "w2_weight"] = v
            else:
                ck[base + "w2_md.weight"] = v
            continue
        ck[k] = v
    return ck


def main():
    cfg_json = {
      "model_type":"inkling_mm_model","image_token_id":50,"audio_token_id":51,"eos_token_id":42,
      "text_config": {"hidden_size":128,"num_hidden_layers":4,"vocab_size":256,"unpadded_vocab_size":250,
        "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
        "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":32,
        "sliding_window_size":8,"d_rel":8,"rel_extent":16,"log_scaling_n_floor":128000,
        "sconv_kernel_size":4,"dense_mlp_idx":2,"dense_intermediate_size":256,"intermediate_size":128,
        "n_routed_experts":8,"num_experts_per_tok":2,"n_shared_experts":2,"route_scale":8.0,
        "logits_mup_width_multiplier":24.0,"local_layer_ids":[0,1,2]},
      "vision_config": {"patch_size":40,"temporal_patch_size":2,"n_channels":3,"n_layers":4},
      "audio_config": {"n_mel_bins":80,"mel_vocab_size":16},
    }
    cfg = InklingConfig.from_dict(cfg_json)
    m = InklingForConditionalGeneration(cfg); mx.eval(m.parameters())
    ids = mx.array([[3,7,1,42,8,9,2,5]])
    ref_logits = np.array(m(ids).astype(mx.float32))

    # write fake checkpoint (checkpoint naming, bf16), forcing MTP keys to ensure they're dropped
    params = dict(tree_flatten(m.parameters()))
    ck = to_checkpoint(params)
    ck["model.mtp.layers.0.transformer_block.attn.q_norm.weight"] = mx.zeros((32,))  # must be dropped
    os.makedirs(SRC, exist_ok=True)
    for f in os.listdir(SRC):
        p=os.path.join(SRC,f); shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    ck = {k: v.astype(mx.bfloat16) if v.dtype==mx.float32 and "global_scale" not in k and "gate.bias" not in k else v for k,v in ck.items()}
    mx.save_safetensors(os.path.join(SRC,"model-00001-of-00001.safetensors"), ck, metadata={"format":"pt"})
    json.dump({"metadata":{"total_size":0},"weight_map":{k:"model-00001-of-00001.safetensors" for k in ck}},
              open(os.path.join(SRC,"model.safetensors.index.json"),"w"))
    json.dump(cfg_json, open(os.path.join(SRC,"config.json"),"w"))
    open(os.path.join(SRC,"tokenizer.json"),"w").write("{}")  # aux copy check

    # force multi-shard output to exercise flush/rename
    C._SHARD_CAP_BYTES = 200_000

    for bits, name in [(None,"bf16"),(8,"8bit"),(4,"4bit")]:
        dst=f"{SCRATCH}/fake-inkling-{name}"
        if os.path.exists(dst): shutil.rmtree(dst)
        C.convert_model(SRC, dst, bits=bits)
        idx=json.load(open(os.path.join(dst,"model.safetensors.index.json")))
        nshards=len(set(idx["weight_map"].values()))
        has_tok=os.path.exists(os.path.join(dst,"tokenizer.json"))
        assert not any(k.startswith("model.mtp.") for k in idx["weight_map"]), "MTP not dropped!"
        model2,_ = load(dst)
        out=np.array(model2(ids).astype(mx.float32))
        d=np.abs(out-ref_logits).max()
        print(f"[{name:5s}] shards={nshards} tok_copied={has_tok} tensors={len(idx['weight_map'])} maxΔ_vs_fp={d:.3e} finite={np.isfinite(out).all()}")
        assert np.isfinite(out).all()
        if bits is None:
            assert d < 0.2, f"bf16 convert drifted too much: {d}"
    print("\nDRIVER TEST PASS")


if __name__ == "__main__":
    main()
