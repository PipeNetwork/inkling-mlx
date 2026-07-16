"""End-to-end smoke test for a converted Inkling MLX model: load the real quantized
weights, run the KV-cache generation loop on a few prompts, and report coherence +
speed + peak memory. Intended for the 4-bit quant on a 512 GB M3 Ultra.
"""

import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inkling_mlx.load import load
from inkling_mlx.generate import greedy_generate, load_tokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/Users/david/llm/inkling-mlx-out/Inkling-4bit"
SRC = "/Users/david/llm/Inkling-src"
PROMPTS = [
    "The capital of France is",
    "Here is a Python function that returns the nth Fibonacci number:\n\ndef fib(n):",
    "Q: In one sentence, what is a transformer in machine learning?\nA:",
]
MAX_NEW = 48


def main():
    # wire up to ~500 GB (requires `sudo sysctl iogpu.wired_limit_mb=500000` first,
    # else the kernel caps the working set at ~75% and this warns + the model can't
    # fully reside). Value stays under the raised cap to leave the OS headroom.
    try:
        mx.set_wired_limit(int(500e9))
    except Exception as e:
        print(f"[warn] set_wired_limit: {e}")

    print(f"[smoke] loading {MODEL} (lazy; materializes during generation)", flush=True)
    t0 = time.time()
    model, config = load(MODEL, lazy=True)
    load_s = time.time() - t0
    peak = mx.get_peak_memory() / 1e9
    print(f"[smoke] loaded in {load_s:.0f}s | weights resident ~{mx.get_active_memory()/1e9:.0f} GB | peak {peak:.0f} GB")

    # tokenizer: prefer the model dir, fall back to the source checkpoint
    try:
        tok = load_tokenizer(MODEL)
    except Exception:
        tok = load_tokenizer(SRC)

    for p in PROMPTS:
        ids = tok(p)["input_ids"]
        t0 = time.time()
        out = greedy_generate(model, config, ids, max_new_tokens=MAX_NEW)
        dt = time.time() - t0
        n_new = len(out) - len(ids)
        text = tok.decode(out)
        print("\n" + "=" * 70)
        print(f"PROMPT: {p!r}")
        print(f"OUTPUT: {text}")
        print(f"[{n_new} new tokens in {dt:.1f}s = {n_new/dt:.2f} tok/s]")

    print(f"\n[smoke] peak memory {mx.get_peak_memory()/1e9:.0f} GB")
    print("[smoke] DONE")


if __name__ == "__main__":
    main()
