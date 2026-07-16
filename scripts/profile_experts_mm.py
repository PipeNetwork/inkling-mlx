"""MULTIMODAL MoE expert-saliency profiler — fixes the text-only blind spot.

The first profiler (profile_experts.py) ran text only, so experts that specialize in
grounding VISION tokens were never activated, ranked cold, and got pruned — which
degraded image understanding (a Pallas's cat -> "bear"). This runs real images through
the full multimodal forward ALONGSIDE the text calibration, so vision-relevant experts
accumulate real saliency and survive pruning. Same REAP saliency:
    S_j = mean over active tokens of gate_weight_j * ||expert_output_j||_2

    python scripts/profile_experts_mm.py [MODEL_DIR]
"""
import json, sys, os, time, glob
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inkling_mlx.load import load
from inkling_mlx.moe import MoE
from inkling_mlx.processing import InklingProcessor
from transformers import AutoTokenizer
from PIL import Image

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/Users/david/llm/inkling-mlx-out/Inkling-4bit"
CALIB = "/Users/david/llm/inkling-mlx-out/calib_wide.json"
IMGROOT = "/Users/david/llm/inkling-mlx-out/imagenette2-320/train"
OUT = "/Users/david/llm/inkling-mlx-out/expert_usage_mm.npz"
MAX_TOK = 256
N_IMG = 200
IMG_PROMPTS = ["Describe this image in detail.",
               "What objects and animals are in this image?",
               "What is happening in this picture? Be specific."]
try: mx.set_wired_limit(int(500e9))
except Exception as e: print("[warn]", e, flush=True)

print(f"[prof-mm] loading {MODEL}", flush=True)
model, config = load(MODEL, lazy=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
ctmpl = open(f"{MODEL}/chat_template.jinja").read()
tok.chat_template = ctmpl
proc = InklingProcessor(tok, ctmpl)
texts = json.load(open(CALIB))

# sample images spread across all classes (deterministic: sorted + strided)
classes = sorted(glob.glob(f"{IMGROOT}/*"))
per = max(1, N_IMG // max(1, len(classes)))
imgs = []
for c in classes:
    files = sorted(glob.glob(f"{c}/*.JPEG"))
    step = max(1, len(files) // per)
    imgs += files[::step][:per]
imgs = imgs[:N_IMG]
print(f"[prof-mm] calib: {len(texts)} text chunks + {len(imgs)} images across {len(classes)} classes", flush=True)

sparse = [l for l in model.model.llm.layers if isinstance(l.mlp, MoE)]
NE = config.text.n_routed_experts
counts = np.zeros((len(sparse), NE))
rweight = np.zeros((len(sparse), NE))
saliency = np.zeros((len(sparse), NE))
img_counts = np.zeros((len(sparse), NE))   # activations attributable to image inputs (diagnostic)
_MODE = {"img": False}


class MoERec:
    def __init__(self, moe, li):
        self.moe, self.li = moe, li

    def __call__(self, x):
        B, L, H = x.shape
        tw, ti, sg = self.moe.gate(x)
        xf = x.reshape(-1, H)
        routed = self.moe.experts(xf, ti)                       # [T, top_k, H]
        idx = np.array(ti).reshape(-1)
        gate = np.array(tw.astype(mx.float32)).reshape(-1)
        onorm = np.array(mx.sqrt((routed.astype(mx.float32) ** 2).sum(-1))).reshape(-1)
        np.add.at(counts[self.li], idx, 1.0)
        np.add.at(rweight[self.li], idx, gate)
        np.add.at(saliency[self.li], idx, gate * onorm)
        if _MODE["img"]:
            np.add.at(img_counts[self.li], idx, 1.0)
        routed = (routed * tw[..., None]).sum(axis=1)
        T = xf.shape[0]
        shared_idx = mx.broadcast_to(mx.arange(self.moe.n_shared)[None], (T, self.moe.n_shared))
        shared = self.moe.shared_experts(xf, shared_idx)
        shared = (shared.astype(mx.float32) * sg[..., None].astype(mx.float32)).sum(axis=1).astype(routed.dtype)
        return (routed + shared).reshape(B, L, H)


for li, l in enumerate(sparse):
    l.mlp = MoERec(l.mlp, li)


def save(nt, ni):
    np.savez(OUT, counts=counts, rweight=rweight, saliency=saliency, img_counts=img_counts,
             n_tokens=nt, n_images=ni, n_experts=NE, dense_mlp_idx=config.text.dense_mlp_idx)


ntok = 0
t0 = time.time()
# --- text pass (keeps text experts credited) ---
for i, txt in enumerate(texts):
    ids = tok(txt)["input_ids"][:MAX_TOK]
    if len(ids) < 8:
        continue
    mx.eval(model(mx.array([ids]), last_logit_only=True)); mx.clear_cache()
    ntok += len(ids)
    if i % 20 == 0 or i == len(texts) - 1:
        save(ntok, 0)
        print(f"[prof-mm] text {i+1}/{len(texts)} ({ntok} tok, {time.time()-t0:.0f}s)", flush=True)

# --- image pass (credits vision-grounding experts) ---
_MODE["img"] = True
nimg = 0
for j, ipath in enumerate(imgs):
    try:
        img = Image.open(ipath).convert("RGB")
        p = IMG_PROMPTS[j % len(IMG_PROMPTS)]
        inp = proc.apply([{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": p}]}], reasoning_effort="none")
        ids = inp["input_ids"]
        mx.eval(model(mx.array([ids]), last_logit_only=True, pixel_values=inp.get("pixel_values")))
        mx.clear_cache()
        nimg += 1; ntok += len(ids)
    except Exception as e:
        print(f"[prof-mm] img skipped {os.path.basename(ipath)}: {str(e)[:60]}", flush=True)
        continue
    if j % 10 == 0 or j == len(imgs) - 1:
        save(ntok, nimg)
        print(f"[prof-mm] img {j+1}/{len(imgs)} ({nimg} ok, {ntok} tok, {time.time()-t0:.0f}s)", flush=True)

save(ntok, nimg)
print(f"[prof-mm] saved {OUT} ({ntok} tokens, {nimg} images)", flush=True)

# ---- report: how many experts would flip from cold->credited by adding images ----
cold_all = (counts == 0).sum(1)
img_touch = (img_counts > 0).sum(1)                      # experts activated by images / layer
img_only = ((img_counts > 0) & (counts - img_counts == 0)).sum(1)  # ONLY images activated them
print(f"\n=== MM profile: {config.text.num_hidden_layers} layers, {NE} experts, {nimg} imgs + {len(texts)} text ===", flush=True)
print(f"cold experts / layer (mm): mean {cold_all.mean():.1f} (max {cold_all.max()})", flush=True)
print(f"experts activated by images / layer: mean {img_touch.mean():.0f}", flush=True)
print(f"experts ONLY images activated (text-blind would prune these): mean {img_only.mean():.1f} (max {img_only.max()})", flush=True)
print("[prof-mm] DONE", flush=True)
