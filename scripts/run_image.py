import sys, os, time
import mlx.core as mx
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inkling_mlx.load import load
from inkling_mlx.generate import greedy_generate
from inkling_mlx.processing import InklingProcessor
from transformers import AutoTokenizer

MODEL = "/Users/david/llm/inkling-mlx-out/Inkling-4bit"
try: mx.set_wired_limit(int(500e9))
except Exception as e: print("[warn]", e, flush=True)

print("[img] loading 4-bit model (eager: weights wired-resident)...", flush=True)
t0 = time.time()
model, config = load(MODEL)   # eager (lazy=False default): materialize + wire weights so
                              # forwards don't re-read/thrash the mmap near the memory ceiling
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
proc = InklingProcessor(tok, open(f"{MODEL}/chat_template.jinja").read())
print(f"[img] ready in {time.time()-t0:.0f}s", flush=True)

img = Image.open("/tmp/test_cat.jpeg").convert("RGB")
q = "What animal is in this image, and what is around it? Answer in one sentence."
inputs = proc.apply([{"role":"user","content":[
    {"type":"image","image":img}, {"type":"text","text":q}]}], reasoning_effort="none")
print(f"[img] {img.size} -> {inputs['pixel_values'].shape[0]} patches; prompt {len(inputs['input_ids'])} tokens", flush=True)

t0 = time.time()
out = greedy_generate(model, config, inputs["input_ids"], max_new_tokens=96,
                      pixel_values=inputs.get("pixel_values"),
                      audio_input_ids=inputs.get("audio_input_ids"))
dt = time.time() - t0
print(f"\n[img] QUESTION: {q}", flush=True)
print(f"[img] RESPONSE: {tok.decode(out[len(inputs['input_ids']):])!r}", flush=True)
print(f"[img] {len(out)-len(inputs['input_ids'])} tokens in {dt:.0f}s", flush=True)
print("[img] DONE", flush=True)
