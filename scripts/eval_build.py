"""Combined per-build eval in ONE model load: audio transcription (LibriSpeech) +
image ID (imagenette-val + Pallas's cat) + text perplexity. Writes build_eval.json
keyed by build name for before/after comparison.

    python scripts/eval_build.py <MODEL_DIR>
"""
import json, os, sys, math, glob
import mlx.core as mx
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from inkling_mlx.load import load
from inkling_mlx.generate import greedy_generate
from inkling_mlx.processing import InklingProcessor
from transformers import AutoTokenizer
from audio_util import load_wav, librispeech_index, transcription_overlap, content_words

MODEL = sys.argv[1]
OUT = "/Users/david/llm/inkling-mlx-out/build_eval.json"
VAL = "/Users/david/llm/inkling-mlx-out/imagenette2-320/val"
LS = "/Users/david/llm/inkling-mlx-out/LibriSpeech/dev-clean"
LAZY = os.environ.get("INKLING_LAZY") == "1"
try: mx.set_wired_limit(int(500e9))
except Exception as e: print("[warn]", e, flush=True)

PASSAGES = [
    "The lighthouse keeper had not spoken to another human being in forty-one days.",
    "In distributed systems, a consensus protocol must guarantee that all non-faulty "
    "nodes agree on a single value even when some nodes fail or messages are delayed.",
    "def merge_sort(a):\n    if len(a) <= 1:\n        return a\n    mid = len(a)//2\n"
    "    return merge(merge_sort(a[:mid]), merge_sort(a[mid:]))",
    "A train leaves at 60 km/h; a second leaves an hour later at 90 km/h. Setting "
    "60(t+1)=90t gives 30t=60, so t=2 hours.",
    "La mémoire est le gardien de toutes choses. Sans elle, la raison ne pourrait "
    "accomplir son office, ni les arts fleurir.",
    "量子力学描述了微观粒子的行为，其中最著名的是不确定性原理。",
]


def img_first(w):
    fs = sorted(glob.glob(f"{VAL}/{w}/*.JPEG"))
    return fs[0] if fs else None


IMG_SPECS = [
    ("/tmp/test_cat.jpeg", ["cat", "feline", "lynx", "manul"], "Pallas's cat"),
    (img_first("n02102040"), ["dog", "spaniel", "springer", "pupp", "canine"], "dog"),
    (img_first("n03028079"), ["church", "cathedral", "chapel", "building"], "church"),
    (img_first("n03445777"), ["golf", "ball"], "golf ball"),
    (img_first("n03417042"), ["truck", "garbage", "vehicle", "lorry"], "garbage truck"),
    (img_first("n03394916"), ["horn", "instrument", "brass", "trumpet", "tuba"], "French horn"),
]

# same 6 audio clips as eval_audio.py (deterministic)
aidx = librispeech_index(LS)
AUD = []
for fp, text in aidx:
    if len(content_words(text)) < 10:
        continue
    wav = load_wav(fp)
    if 3.0 <= len(wav) / 16000.0 <= 10.0:
        AUD.append((fp, text, wav))
    if len(AUD) >= 6:
        break

print(f"[beval] loading {os.path.basename(MODEL)} ({'lazy' if LAZY else 'eager'})", flush=True)
model, config = load(MODEL, lazy=LAZY)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
proc = InklingProcessor(tok, open(f"{MODEL}/chat_template.jinja").read())

# ---- audio ----
a_pass, a_ovs = 0, []
for fp, text, wav in AUD:
    inp = proc.apply([{"role": "user", "content": [
        {"type": "audio", "audio": wav, "sampling_rate": 16000},
        {"type": "text", "text": "Transcribe the speech in this audio."}]}], reasoning_effort="none")
    out = greedy_generate(model, config, inp["input_ids"], max_new_tokens=64,
                          audio_input_ids=inp.get("audio_input_ids"))
    ov = transcription_overlap(tok.decode(out[len(inp["input_ids"]):]), text)
    mx.clear_cache(); a_ovs.append(ov); a_pass += ov >= 0.30
mean_ov = round(sum(a_ovs) / max(1, len(a_ovs)), 3)
print(f"[beval] AUDIO {a_pass}/{len(AUD)}  mean_overlap {mean_ov}", flush=True)

# ---- image ----
i_pass = 0
for path, kws, label in IMG_SPECS:
    if not path or not os.path.exists(path):
        continue
    inp = proc.apply([{"role": "user", "content": [
        {"type": "image", "image": Image.open(path).convert("RGB")},
        {"type": "text", "text": "What is the main subject of this image? Answer in one sentence."}]}],
        reasoning_effort="none")
    out = greedy_generate(model, config, inp["input_ids"], max_new_tokens=48,
                          pixel_values=inp.get("pixel_values"))
    resp = tok.decode(out[len(inp["input_ids"]):]).lower()
    mx.clear_cache(); i_pass += any(k in resp for k in kws)
print(f"[beval] IMAGE {i_pass}/{len(IMG_SPECS)}", flush=True)

# ---- text perplexity ----
tot_nll, tot_tok = 0.0, 0
for p in PASSAGES:
    ids = tok(p)["input_ids"][:400]
    if len(ids) < 8:
        continue
    lp = model(mx.array([ids]), last_logit_only=False)[0].astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    nll = -lp[mx.arange(len(ids) - 1), mx.array(ids[1:])]
    tot_nll += float(nll.sum().item()); tot_tok += len(ids) - 1
    mx.clear_cache()
ppl = round(math.exp(tot_nll / tot_tok), 3)
print(f"[beval] PPL {ppl}", flush=True)

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res[os.path.basename(MODEL)] = {"audio_pass": a_pass, "audio_mean_overlap": mean_ov,
                                "image_pass": i_pass, "image_n": len(IMG_SPECS), "ppl": ppl}
json.dump(res, open(OUT, "w"), indent=2)
print(f"[beval] saved -> {OUT}", flush=True)
