"""Select experts to keep per MoE layer using REAP saliency, and report the
contribution retained. Emits keep-indices for the prune-aware converter.

REAP saliency:  S_j = mean_{x in X_j} g_j(x) * ||f_j(x)||_2   (X_j = tokens where
expert j is in top-k). The profiler saved `saliency` = sum of g*||f|| and `counts`
= |X_j|, so S_j = saliency / counts. We keep the top-K by S_j per layer (uniform K,
per-layer selection = REAP's per-layer p% pruning).

    python scripts/prune_experts.py <prune_ratio e.g. 0.5>
"""
import sys
import numpy as np

USAGE = "/Users/david/llm/inkling-mlx-out/expert_usage.npz"
OUT = "/Users/david/llm/inkling-mlx-out/keep_indices.npz"
ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

d = np.load(USAGE)
counts, sal = d["counts"].astype(np.float64), d["saliency"].astype(np.float64)
dmi = int(d["dense_mlp_idx"]); NL, NE = sal.shape
S = sal / np.maximum(counts, 1.0)             # REAP saliency; cold experts -> 0
S = np.where(counts == 0, -1.0, S)            # ensure never-used prune first

order = np.argsort(-S, axis=1)                 # experts ranked by saliency, per layer
def retained_at(K):
    kp = order[:, :K]
    return (np.take_along_axis(sal, kp, axis=1).sum(1) / np.maximum(sal.sum(1), 1e-9))

# sweep: how much router-weighted contribution survives at each prune ratio
print("=== ratio sweep (contribution retained) ===")
for r in (0.10, 0.20, 0.25, 0.375, 0.50, 0.60, 0.75):
    ret = retained_at(int(round(NE * (1 - r))))
    print(f"  prune {r:.0%} (keep {int(round(NE*(1-r)))}): retained mean {ret.mean()*100:.1f}%  min {ret.min()*100:.1f}%")

K = int(round(NE * (1.0 - ratio)))
keep = np.sort(order[:, :K], axis=1)           # [NL, K] per-layer top-K
retained = retained_at(K)
cold = (counts == 0).sum(1)

print(f"=== REAP prune ratio {ratio:.0%}  ->  keep {K}/{NE} experts per layer ===")
print(f"contribution retained: mean {retained.mean()*100:.1f}%  (min {retained.min()*100:.1f}%, max {retained.max()*100:.1f}%)")
print(f"cold experts / layer:  mean {cold.mean():.0f}  (so at 50% many pruned experts were already ~unused)")
print(f"experts kept: {K} routed + 2 shared;  new n_routed_experts = {K}")
# size implication
print(f"routed-expert params scale ~x{K/NE:.2f} (experts are ~95% of the model)")
np.savez(OUT, keep=keep.astype(np.int32), K=K, dense_mlp_idx=dmi, ratio=ratio)
print(f"saved {OUT}")
