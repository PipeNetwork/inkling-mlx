"""MULTIMODAL MoE expert-saliency profiler — text + images + audio.

Text-only calibration prunes experts that ground non-text modalities (they rarely fire
on text), silently wrecking image AND audio understanding while text perplexity looks
fine. This runs real images and real speech through the full multimodal forward
alongside the text calibration, so vision- and audio-grounding experts accumulate real
saliency and survive pruning. REAP saliency:
    S_j = mean over active tokens of gate_weight_j * ||expert_output_j||_2

    python scripts/profile_experts_mm.py [MODEL_DIR]
"""
import json, sys, os, time, glob
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from inkling_mlx.load import load
from inkling_mlx.moe import MoE
from inkling_mlx.processing import InklingProcessor
from transformers import AutoTokenizer
from PIL import Image
from audio_util import load_wav, librispeech_index, content_words

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/Users/david/llm/inkling-mlx-out/Inkling-4bit"
CALIB = "/Users/david/llm/inkling-mlx-out/calib_wide.json"
IMGROOT = "/Users/david/llm/inkling-mlx-out/imagenette2-320/train"
LSROOT = "/Users/david/llm/inkling-mlx-out/LibriSpeech/dev-clean"
OUT = "/Users/david/llm/inkling-mlx-out/expert_usage_mm.npz"
MAX_TOK = 256
N_IMG = 200
N_AUD = 180
IMG_PROMPTS = ["Describe this image in detail.",
               "What objects and animals are in this image?",
               "What is happening in this picture? Be specific."]
AUD_PROMPTS = ["Transcribe the speech in this audio.",
               "What is being said in this audio?"]
try: mx.set_wired_limit(int(500e9))
except Exception as e: print("[warn]", e, flush=True)

print(f"[prof-mm] loading {MODEL}", flush=True)
model, config = load(MODEL, lazy=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
ctmpl = open(f"{MODEL}/chat_template.jinja").read()
tok.chat_template = ctmpl
proc = InklingProcessor(tok, ctmpl)
texts = json.load(open(CALIB))

# images: spread across classes (deterministic strided)
classes = sorted(glob.glob(f"{IMGROOT}/*"))
per = max(1, N_IMG // max(1, len(classes)))
imgs = []
for c in classes:
    files = sorted(glob.glob(f"{c}/*.JPEG"))
    imgs += files[::max(1, len(files) // per)][:per]
imgs = imgs[:N_IMG]

# audio: short distinctive LibriSpeech clips (deterministic strided across the index)
ls = librispeech_index(LSROOT) if os.path.isdir(LSROOT) else []
ls = [(fp, t) for fp, t in ls if len(content_words(t)) >= 8]
auds = ls[::max(1, len(ls) // max(1, N_AUD))][:N_AUD]
print(f"[prof-mm] calib: {len(texts)} text + {len(imgs)} images + {len(auds)} audio clips", flush=True)

sparse = [l for l in model.model.llm.layers if isinstance(l.mlp, MoE)]
NE = config.text.n_routed_experts
counts = np.zeros((len(sparse), NE)); rweight = np.zeros((len(sparse), NE))
saliency = np.zeros((len(sparse), NE))
img_counts = np.zeros((len(sparse), NE)); aud_counts = np.zeros((len(sparse), NE))
_MODE = {"m": "text"}


class MoERec:
    def __init__(self, moe, li):
        self.moe, self.li = moe, li

    def __call__(self, x):
        B, L, H = x.shape
        tw, ti, sg = self.moe.gate(x)
        xf = x.reshape(-1, H)
        routed = self.moe.experts(xf, ti)
        idx = np.array(ti).reshape(-1)
        gate = np.array(tw.astype(mx.float32)).reshape(-1)
        onorm = np.array(mx.sqrt((routed.astype(mx.float32) ** 2).sum(-1))).reshape(-1)
        np.add.at(counts[self.li], idx, 1.0)
        np.add.at(rweight[self.li], idx, gate)
        np.add.at(saliency[self.li], idx, gate * onorm)
        if _MODE["m"] == "img":
            np.add.at(img_counts[self.li], idx, 1.0)
        elif _MODE["m"] == "aud":
            np.add.at(aud_counts[self.li], idx, 1.0)
        routed = (routed * tw[..., None]).sum(axis=1)
        T = xf.shape[0]
        shared_idx = mx.broadcast_to(mx.arange(self.moe.n_shared)[None], (T, self.moe.n_shared))
        shared = self.moe.shared_experts(xf, shared_idx)
        shared = (shared.astype(mx.float32) * sg[..., None].astype(mx.float32)).sum(axis=1).astype(routed.dtype)
        return (routed + shared).reshape(B, L, H)


for li, l in enumerate(sparse):
    l.mlp = MoERec(l.mlp, li)


def save(nt, ni, na):
    np.savez(OUT, counts=counts, rweight=rweight, saliency=saliency,
             img_counts=img_counts, aud_counts=aud_counts,
             n_tokens=nt, n_images=ni, n_audio=na, n_experts=NE,
             dense_mlp_idx=config.text.dense_mlp_idx)


ntok = 0; t0 = time.time()
# --- text ---
_MODE["m"] = "text"
for i, txt in enumerate(texts):
    ids = tok(txt)["input_ids"][:MAX_TOK]
    if len(ids) < 8:
        continue
    mx.eval(model(mx.array([ids]), last_logit_only=True)); mx.clear_cache()
    ntok += len(ids)
    if i % 40 == 0 or i == len(texts) - 1:
        save(ntok, 0, 0); print(f"[prof-mm] text {i+1}/{len(texts)} ({ntok} tok, {time.time()-t0:.0f}s)", flush=True)

# --- images ---
_MODE["m"] = "img"; nimg = 0
for j, ipath in enumerate(imgs):
    try:
        inp = proc.apply([{"role": "user", "content": [
            {"type": "image", "image": Image.open(ipath).convert("RGB")},
            {"type": "text", "text": IMG_PROMPTS[j % len(IMG_PROMPTS)]}]}], reasoning_effort="none")
        mx.eval(model(mx.array([inp["input_ids"]]), last_logit_only=True, pixel_values=inp.get("pixel_values")))
        mx.clear_cache(); nimg += 1; ntok += len(inp["input_ids"])
    except Exception as e:
        print(f"[prof-mm] img skipped {os.path.basename(ipath)}: {str(e)[:60]}", flush=True); continue
    if j % 25 == 0 or j == len(imgs) - 1:
        save(ntok, nimg, 0); print(f"[prof-mm] img {j+1}/{len(imgs)} ({nimg} ok, {time.time()-t0:.0f}s)", flush=True)

# --- audio ---
_MODE["m"] = "aud"; naud = 0
for j, (fp, text) in enumerate(auds):
    try:
        wav = load_wav(fp)
        inp = proc.apply([{"role": "user", "content": [
            {"type": "audio", "audio": wav, "sampling_rate": 16000},
            {"type": "text", "text": AUD_PROMPTS[j % len(AUD_PROMPTS)]}]}], reasoning_effort="none")
        mx.eval(model(mx.array([inp["input_ids"]]), last_logit_only=True, audio_input_ids=inp.get("audio_input_ids")))
        mx.clear_cache(); naud += 1; ntok += len(inp["input_ids"])
    except Exception as e:
        print(f"[prof-mm] aud skipped {os.path.basename(fp)}: {str(e)[:60]}", flush=True); continue
    if j % 20 == 0 or j == len(auds) - 1:
        save(ntok, nimg, naud); print(f"[prof-mm] aud {j+1}/{len(auds)} ({naud} ok, {time.time()-t0:.0f}s)", flush=True)

save(ntok, nimg, naud)
print(f"[prof-mm] saved {OUT} ({ntok} tokens, {nimg} images, {naud} audio)", flush=True)

# ---- report ----
cold = (counts == 0).sum(1)
img_only = ((img_counts > 0) & (counts - img_counts == 0)).sum(1)
aud_only = ((aud_counts > 0) & (counts - aud_counts == 0)).sum(1)
print(f"\n=== MM profile: {config.text.num_hidden_layers} layers, {NE} experts; {nimg} imgs + {naud} audio + {len(texts)} text ===", flush=True)
print(f"cold experts / layer: mean {cold.mean():.1f} (max {cold.max()})", flush=True)
print(f"experts activated by images / layer: mean {(img_counts>0).sum(1).mean():.0f}", flush=True)
print(f"experts activated by audio / layer:  mean {(aud_counts>0).sum(1).mean():.0f}", flush=True)
print(f"experts ONLY images activated / layer: mean {img_only.mean():.1f} (max {img_only.max()})", flush=True)
print(f"experts ONLY audio activated / layer:  mean {aud_only.mean():.1f} (max {aud_only.max()})", flush=True)
print("[prof-mm] DONE", flush=True)
