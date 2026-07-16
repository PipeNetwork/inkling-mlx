"""MoE expert-usage profiler (REAP saliency). For each MoE layer, over a wide
calibration corpus, accumulates per-expert: (a) selection count, (b) summed router
weight, and (c) REAP saliency = sum(gate_weight * ||expert_output||) — the actual
contribution to the residual stream, which is what REAP ranks by. Reports per-layer
skew and saves arrays for the pruning pass.

    python scripts/profile_experts.py [MODEL_DIR]
"""
import json, sys, os, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inkling_mlx.load import load
from inkling_mlx.moe import MoE
from transformers import AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/Users/david/llm/inkling-mlx-out/Inkling-4bit"
CALIB = "/Users/david/llm/inkling-mlx-out/calib_wide.json"
OUT = "/Users/david/llm/inkling-mlx-out/expert_usage.npz"
MAX_TOK = 256
try: mx.set_wired_limit(int(500e9))
except Exception as e: print("[warn]", e, flush=True)

print(f"[prof] loading {MODEL}", flush=True)
model, config = load(MODEL, lazy=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.chat_template = open(f"{MODEL}/chat_template.jinja").read()
texts = json.load(open(CALIB))

sparse = [l for l in model.model.llm.layers if isinstance(l.mlp, MoE)]
NE = config.text.n_routed_experts
counts = np.zeros((len(sparse), NE))
rweight = np.zeros((len(sparse), NE))
saliency = np.zeros((len(sparse), NE))


class MoERec:
    """Wraps a MoE: records count / router-weight / REAP saliency, then does the
    normal MoE forward (re-implemented so we can see per-expert outputs)."""
    def __init__(self, moe, li):
        self.moe, self.li = moe, li

    def __call__(self, x):
        B, L, H = x.shape
        tw, ti, sg = self.moe.gate(x)
        xf = x.reshape(-1, H)
        routed = self.moe.experts(xf, ti)                       # [T, top_k, H]
        idx = np.array(ti).reshape(-1)
        gate = np.array(tw.astype(mx.float32)).reshape(-1)
        # ||expert output|| computed ON-DEVICE; only the small [T,k] norm is synced
        onorm = np.array(mx.sqrt((routed.astype(mx.float32) ** 2).sum(-1))).reshape(-1)
        np.add.at(counts[self.li], idx, 1.0)
        np.add.at(rweight[self.li], idx, gate)
        np.add.at(saliency[self.li], idx, gate * onorm)
        routed = (routed * tw[..., None]).sum(axis=1)
        T = xf.shape[0]
        shared_idx = mx.broadcast_to(mx.arange(self.moe.n_shared)[None], (T, self.moe.n_shared))
        shared = self.moe.shared_experts(xf, shared_idx)
        shared = (shared.astype(mx.float32) * sg[..., None].astype(mx.float32)).sum(axis=1).astype(routed.dtype)
        return (routed + shared).reshape(B, L, H)


for li, l in enumerate(sparse):
    l.mlp = MoERec(l.mlp, li)

def save(nt):
    np.savez(OUT, counts=counts, rweight=rweight, saliency=saliency, n_tokens=nt, n_experts=NE,
             dense_mlp_idx=config.text.dense_mlp_idx)

ntok = 0
t0 = time.time()
for i, txt in enumerate(texts):
    ids = tok(txt)["input_ids"][:MAX_TOK]
    if len(ids) < 8:
        continue
    lg = model(mx.array([ids]), last_logit_only=True)
    mx.eval(lg)
    mx.clear_cache()                         # release per-prompt buffers (near-capacity model)
    ntok += len(ids)
    if i % 15 == 0 or i == len(texts) - 1:
        save(ntok)                           # incremental checkpoint (crash-safe)
        print(f"[prof] {i+1}/{len(texts)} chunks, {ntok} tokens ({time.time()-t0:.0f}s)", flush=True)

save(ntok)
print(f"[prof] saved {OUT}", flush=True)

# ---- report (by REAP saliency) ----
d = config.text.dense_mlp_idx
cold = (counts == 0).sum(1)
p = counts / np.clip(counts.sum(1)[:, None], 1, None)
ent = -(np.where(p > 0, p * np.log(p), 0)).sum(1) / np.log(NE)
ssort = np.sort(saliency, axis=1)[:, ::-1]
scum = np.cumsum(ssort, axis=1) / np.clip(saliency.sum(1)[:, None], 1e-9, None)
keep = {thr: (scum < thr).sum(1) + 1 for thr in (0.90, 0.95, 0.99)}
print(f"\n=== REAP saliency over {ntok} tokens, {len(sparse)} MoE layers x {NE} experts ===", flush=True)
print(f"cold experts / layer: mean {cold.mean():.0f} (min {cold.min()}, max {cold.max()})", flush=True)
print(f"routing entropy (1=uniform): mean {ent.mean():.3f} (min {ent.min():.3f}, max {ent.max():.3f})", flush=True)
for thr, k in keep.items():
    print(f"experts for {int(thr*100)}% saliency / layer: mean {k.mean():.0f} (max {k.max()})  -> keep {int(np.ceil(k.max()))} covers {int(thr*100)}% in ALL layers", flush=True)
print("[prof] DONE", flush=True)
