#!/bin/zsh
# Evaluate the unpruned 4-bit + all three REAP builds sequentially. Each runs in a
# fresh process so its 263-496 GB of weights fully releases before the next loads.
set -u
OUT=/Users/david/llm/inkling-mlx-out
cd /Users/david/llm/inkling-mlx
rm -f "$OUT/ppl_results.json"
for M in Inkling-4bit Inkling-REAP12-4bit Inkling-REAP25-4bit Inkling-REAP50-4bit; do
  D="$OUT/$M"
  if [ ! -d "$D" ]; then echo "[eval] SKIP missing $M"; continue; fi
  echo "===== [eval] $M ====="
  python3 scripts/ppl_eval.py "$D" 2>&1 | grep -vE "^You are using|PyTorch|resume_download"
  echo "===== [eval] $M done ====="
done
echo "[eval] ALL DONE"
python3 - <<'PY'
import json
r = json.load(open("/Users/david/llm/inkling-mlx-out/ppl_results.json"))
base = r.get("Inkling-4bit", {}).get("perplexity")
print("\n=== REAP perplexity comparison ===")
print(f"{'build':<22}{'ppl':>10}{'vs unpruned':>14}")
for k in ("Inkling-4bit","Inkling-REAP12-4bit","Inkling-REAP25-4bit","Inkling-REAP50-4bit"):
    if k not in r: continue
    p = r[k]["perplexity"]
    d = f"{(p/base-1)*100:+.1f}%" if base else "-"
    print(f"{k:<22}{p:>10.3f}{d:>14}")
    print(f"    gen: {r[k]['gen'][:90]!r}")
PY
