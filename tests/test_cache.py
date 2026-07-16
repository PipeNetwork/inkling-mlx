"""Verify the incremental KV + conv-state cache reproduces the cache-free forward
exactly (same logits at every position), for a mix of sliding/global layers and
dense/sparse MLPs."""

import numpy as np
import mlx.core as mx
from inkling_mlx.config import InklingConfig
from inkling_mlx.model import InklingForConditionalGeneration
from inkling_mlx.cache import make_cache


def build():
    cfg = InklingConfig.from_dict({
      "model_type":"inkling_mm_model","image_token_id":50,"audio_token_id":51,
      "text_config": {"hidden_size":128,"num_hidden_layers":5,"vocab_size":200,"unpadded_vocab_size":190,
        "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
        "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":32,
        "sliding_window_size":4,"d_rel":8,"rel_extent":8,"log_scaling_n_floor":128000,
        "sconv_kernel_size":4,"dense_mlp_idx":2,"dense_intermediate_size":256,"intermediate_size":64,
        "n_routed_experts":8,"num_experts_per_tok":2,"n_shared_experts":2,"route_scale":8.0,
        "logits_mup_width_multiplier":24.0,"local_layer_ids":[0,2,3]},  # 0,2,3 sliding; 1,4 global
      "vision_config": {"patch_size":40,"temporal_patch_size":2,"n_channels":3,"n_layers":4},
      "audio_config": {"n_mel_bins":80,"mel_vocab_size":16},
    })
    m = InklingForConditionalGeneration(cfg); mx.eval(m.parameters())
    return m, cfg


def main():
    mx.random.seed(0)
    m, cfg = build()
    L = 12
    seq = np.random.randint(0, 190, size=(L,)).tolist()

    # reference: full cache-free forward
    full = np.array(m(mx.array([seq])).astype(mx.float32))[0]   # [L, vocab]

    # (a) fully incremental: one token at a time
    caches = make_cache(m)
    inc = []
    for t in range(L):
        lg = m(mx.array([[seq[t]]]), caches=caches, start_pos=t, last_logit_only=True)
        inc.append(np.array(lg.astype(mx.float32))[0, -1])
    inc = np.stack(inc)
    d_inc = np.abs(inc - full).max()

    # (b) prefill a prompt then decode the rest
    P = 5
    caches2 = make_cache(m)
    lg = m(mx.array([seq[:P]]), caches=caches2, start_pos=0, last_logit_only=True)
    pref = [np.array(lg.astype(mx.float32))[0, -1]]  # predicts token at pos P-1's next == full[P-1]
    for t in range(P, L):
        lg = m(mx.array([[seq[t]]]), caches=caches2, start_pos=t, last_logit_only=True)
        pref.append(np.array(lg.astype(mx.float32))[0, -1])
    pref = np.stack(pref)  # positions P-1 .. L-1
    d_pref = np.abs(pref - full[P - 1:]).max()

    print(f"L={L}, sliding_window={cfg.text.sliding_window_size}")
    print(f"(a) incremental  vs full: max|Δ| = {d_inc:.3e}")
    print(f"(b) prefill+decode vs full: max|Δ| = {d_pref:.3e}")
    ok = d_inc < 1e-3 and d_pref < 1e-3
    print("\nCACHE PARITY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
