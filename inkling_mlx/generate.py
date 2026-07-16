"""Greedy generation for an Inkling MLX model, using an incremental KV + conv-state
cache: the prompt is prefilled once, then each new token is a single-position step.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from .cache import make_cache
from .load import load


def load_tokenizer(path: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        from transformers import PreTrainedTokenizerFast
        import os
        return PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json"))


def greedy_generate(model, config, input_ids, max_new_tokens=32, eos_id=None,
                    pixel_values=None, audio_input_ids=None):
    """Greedy decode. For multimodal, pass ``pixel_values`` / ``audio_input_ids``
    (from ``InklingProcessor``); they are consumed only by the prompt prefill."""
    eos_id = eos_id if eos_id is not None else config.eos_token_id
    caches = make_cache(model)
    prompt = list(input_ids)

    # prefill the whole prompt (with any media) in one pass
    logits = model(mx.array([prompt]), caches=caches, start_pos=0, last_logit_only=True,
                   pixel_values=pixel_values, audio_input_ids=audio_input_ids)
    next_id = int(mx.argmax(logits[0, -1]).item())
    out = [next_id]
    pos = len(prompt)

    for _ in range(max_new_tokens - 1):
        if next_id == eos_id:
            break
        logits = model(mx.array([[next_id]]), caches=caches, start_pos=pos, last_logit_only=True)
        next_id = int(mx.argmax(logits[0, -1]).item())
        out.append(next_id)
        pos += 1

    return prompt + out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="converted MLX model dir")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    print(f"[load] {args.model}")
    t0 = time.time()
    model, config = load(args.model)
    print(f"[load] ready in {time.time()-t0:.0f}s")

    tok = load_tokenizer(args.model)
    input_ids = tok(args.prompt)["input_ids"]
    print(f"[prompt] {args.prompt!r} -> {len(input_ids)} tokens")

    t0 = time.time()
    out_ids = greedy_generate(model, config, input_ids, args.max_new_tokens)
    dt = time.time() - t0
    text = tok.decode(out_ids)
    n_new = len(out_ids) - len(input_ids)
    print(f"\n{text}\n")
    print(f"[gen] {n_new} tokens in {dt:.1f}s ({n_new/dt:.2f} tok/s)")


if __name__ == "__main__":
    main()
