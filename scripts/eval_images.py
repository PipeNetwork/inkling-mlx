"""Image acceptance test: run a fixed set of held-out images through one build and
check whether the response names the right subject. Held-out = imagenette VAL split
(calibration used TRAIN) + the Pallas's-cat that exposed the vision regression.
Appends to a shared JSON so builds can be compared side by side.

    python scripts/eval_images.py <MODEL_DIR>
"""
import json, os, sys, glob, time
import mlx.core as mx
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inkling_mlx.load import load
from inkling_mlx.generate import greedy_generate
from inkling_mlx.processing import InklingProcessor
from transformers import AutoTokenizer

MODEL = sys.argv[1]
OUT = "/Users/david/llm/inkling-mlx-out/image_eval.json"
VAL = "/Users/david/llm/inkling-mlx-out/imagenette2-320/val"
LAZY = os.environ.get("INKLING_LAZY") == "1"


def first(wnid):
    fs = sorted(glob.glob(f"{VAL}/{wnid}/*.JPEG"))
    return fs[0] if fs else None


# (image, accept-keywords, label). Keywords are lowercase substrings; any match = pass.
SPECS = [
    ("/tmp/test_cat.jpeg", ["cat", "feline", "lynx", "manul"], "Pallas's cat (snow)"),
    (first("n02102040"), ["dog", "spaniel", "springer", "puppy", "canine"], "English springer dog"),
    (first("n03028079"), ["church", "cathedral", "chapel", "building"], "church"),
    (first("n03445777"), ["golf", "ball"], "golf ball"),
    (first("n03417042"), ["truck", "garbage", "vehicle", "lorry"], "garbage truck"),
    (first("n03394916"), ["horn", "instrument", "brass", "trumpet", "tuba"], "French horn"),
]
Q = "What is the main subject of this image? Answer in one sentence."

print(f"[imeval] loading {os.path.basename(MODEL)} ({'lazy' if LAZY else 'eager'})", flush=True)
model, config = load(MODEL, lazy=LAZY)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
proc = InklingProcessor(tok, open(f"{MODEL}/chat_template.jinja").read())

results = []
npass = 0
for path, kws, label in SPECS:
    if not path or not os.path.exists(path):
        print(f"[imeval] SKIP missing {label}", flush=True); continue
    img = Image.open(path).convert("RGB")
    inp = proc.apply([{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": Q}]}], reasoning_effort="none")
    out = greedy_generate(model, config, inp["input_ids"], max_new_tokens=48,
                          pixel_values=inp.get("pixel_values"))
    resp = tok.decode(out[len(inp["input_ids"]):])
    low = resp.lower()
    ok = any(k in low for k in kws)
    npass += ok
    results.append({"label": label, "pass": ok, "expect": kws, "resp": resp})
    print(f"[imeval] {'PASS' if ok else 'FAIL'} {label}: {resp.strip()[:100]!r}", flush=True)

score = f"{npass}/{len(results)}"
print(f"[imeval] SCORE {score}", flush=True)
allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
allres[os.path.basename(MODEL)] = {"score": score, "n_pass": npass, "n": len(results), "items": results}
json.dump(allres, open(OUT, "w"), indent=2)
print(f"[imeval] saved -> {OUT}", flush=True)
