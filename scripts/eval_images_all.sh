#!/bin/zsh
# Image acceptance test across builds. Fast eager (pruned) builds first for the
# key old-vs-mm comparison; unpruned gold reference last (slow lazy near-capacity load).
set -u
OUT=/Users/david/llm/inkling-mlx-out
cd /Users/david/llm/inkling-mlx
rm -f "$OUT/image_eval.json"
FILT='^You are using|PyTorch|resume_download|warnings.warn'

for M in Inkling-REAP25-4bit Inkling-REAP25mm-4bit Inkling-REAP12mm-4bit Inkling-REAP50mm-4bit; do
  [ -d "$OUT/$M" ] || { echo "[imeval] SKIP missing $M"; continue; }
  echo "===== [imeval] $M ====="
  python3 scripts/eval_images.py "$OUT/$M" 2>&1 | grep -vE "$FILT"
done

# gold reference (near-capacity, lazy)
echo "===== [imeval] Inkling-4bit (gold, lazy) ====="
INKLING_LAZY=1 python3 scripts/eval_images.py "$OUT/Inkling-4bit" 2>&1 | grep -vE "$FILT"

echo "[imeval] ALL DONE"
python3 - <<'PY'
import json
r = json.load(open("/Users/david/llm/inkling-mlx-out/image_eval.json"))
order = ["Inkling-4bit","Inkling-REAP12mm-4bit","Inkling-REAP25mm-4bit","Inkling-REAP50mm-4bit","Inkling-REAP25-4bit"]
print("\n=== image acceptance (held-out) ===")
for k in order:
    if k not in r: continue
    print(f"\n{k}: {r[k]['score']}")
    for it in r[k]["items"]:
        print(f"  [{'PASS' if it['pass'] else 'FAIL'}] {it['label']:<24} {it['resp'].strip()[:80]!r}")
PY
