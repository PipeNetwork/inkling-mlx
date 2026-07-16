"""Evaluate one Inkling build: teacher-forcing perplexity over a fixed held-out
set + a short chat generation for a coherence read. Identical inputs across builds,
so REAP-vs-unpruned perplexity deltas are directly comparable. Appends to a shared
JSON so all builds land in one table.

    python scripts/ppl_eval.py <MODEL_DIR>
"""
import json, os, sys, time, math
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inkling_mlx.load import load
from inkling_mlx.generate import greedy_generate
from transformers import AutoTokenizer

MODEL = sys.argv[1]
OUT = "/Users/david/llm/inkling-mlx-out/ppl_results.json"
try: mx.set_wired_limit(int(500e9))
except Exception as e: print("[warn]", e, flush=True)

# Fixed held-out passages (prose / code / reasoning / multilingual). The absolute
# ppl matters less than that every build sees the SAME text -> fair relative delta.
PASSAGES = [
    "The lighthouse keeper had not spoken to another human being in forty-one days. "
    "Each morning he climbed the spiral stair, wound the great lamp, and recorded the "
    "weather in a leather book whose pages had begun to smell of salt and kerosene.",
    "In distributed systems, a consensus protocol must guarantee that all non-faulty "
    "nodes agree on a single value even when some nodes fail or messages are delayed. "
    "Paxos achieves this through a two-phase prepare-and-accept exchange among proposers, "
    "acceptors, and learners.",
    "def merge_sort(a):\n    if len(a) <= 1:\n        return a\n    mid = len(a) // 2\n"
    "    left = merge_sort(a[:mid])\n    right = merge_sort(a[mid:])\n    out = []\n"
    "    i = j = 0\n    while i < len(left) and j < len(right):\n"
    "        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n"
    "        else:\n            out.append(right[j]); j += 1\n"
    "    return out + left[i:] + right[j:]",
    "A train leaves the station at 60 km/h. A second train leaves the same station one "
    "hour later at 90 km/h on the same track. To find when the second train catches the "
    "first, set 60(t+1) = 90t, so 60t + 60 = 90t, giving 30t = 60 and t = 2 hours.",
    "La mémoire est le gardien de toutes choses. Sans elle, la raison ne pourrait "
    "accomplir son office, ni les arts fleurir, ni les cités subsister dans la concorde "
    "et la justice qui font la dignité des peuples libres.",
    "量子力学描述了微观粒子的行为，其中最著名的是不确定性原理，它指出粒子的位置和动量"
    "不能同时被精确测量。这一原理揭示了经典物理与量子世界之间的根本差异。",
]

print(f"[ppl] loading {MODEL}", flush=True)
t0 = time.time()
model, config = load(MODEL, lazy=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
ct = f"{MODEL}/chat_template.jinja"
if os.path.exists(ct):
    tok.chat_template = open(ct).read()
print(f"[ppl] loaded in {time.time()-t0:.0f}s", flush=True)

total_nll = 0.0; total_tok = 0; per = []
for p in PASSAGES:
    ids = tok(p)["input_ids"][:400]
    if len(ids) < 8:
        continue
    logits = model(mx.array([ids]), last_logit_only=False)  # [1,L,V]
    lp = logits[0].astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    tgt = mx.array(ids[1:])
    nll = -lp[mx.arange(len(ids) - 1), tgt]
    s = float(nll.sum().item()); n = len(ids) - 1
    per.append(math.exp(s / n)); total_nll += s; total_tok += n
    mx.clear_cache()
ppl = math.exp(total_nll / total_tok)
print(f"[ppl] {MODEL}: perplexity {ppl:.3f}  (per-passage {[round(x,2) for x in per]})", flush=True)

# coherence: chat generation
gen_txt = ""
try:
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Inkling reasoning prefix (thinking effort) — mirror processing.py
    ids = tok(prompt)["input_ids"]
    out_ids = greedy_generate(model, config, ids, max_new_tokens=32)
    gen_txt = tok.decode(out_ids[len(ids):])
    print(f"[ppl] gen: {gen_txt!r}", flush=True)
except Exception as e:
    gen_txt = f"<gen error: {str(e)[:120]}>"
    print(f"[ppl] gen skipped: {e}", flush=True)

res = {}
if os.path.exists(OUT):
    res = json.load(open(OUT))
res[os.path.basename(MODEL)] = {"perplexity": ppl, "per_passage": per, "gen": gen_txt,
                                "n_tokens": total_tok}
json.dump(res, open(OUT, "w"), indent=2)
print(f"[ppl] saved -> {OUT}", flush=True)
