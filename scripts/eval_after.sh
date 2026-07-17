#!/bin/zsh
# Re-eval the audio-inclusive mm builds (audio + image + text ppl) and compare to the
# audio-blind "before". REAP-25mm first (the headline sweet spot).
set -u
OUT=/Users/david/llm/inkling-mlx-out
cd /Users/david/llm/inkling-mlx
rm -f "$OUT/build_eval.json"
for M in Inkling-REAP25mm-4bit Inkling-REAP12mm-4bit Inkling-REAP50mm-4bit; do
  echo "===== [after] $M ====="
  python3 scripts/eval_build.py "$OUT/$M" 2>&1 | grep -vE "^You are using|PyTorch|resume_download|warnings.warn"
done
echo "[after] ALL DONE"
python3 - <<'PY'
import json
r = json.load(open("/Users/david/llm/inkling-mlx-out/build_eval.json"))
# audio-blind "before" (from audio_eval_before.json + earlier text-only ppl)
before_audio = {"Inkling-REAP12mm-4bit": 0.875, "Inkling-REAP25mm-4bit": 0.567}
print("\n=== audio-inclusive recalibration: before -> after ===")
print(f"{'build':<24}{'audio ov (before->after)':>28}{'image':>10}{'ppl':>9}")
for k in ("Inkling-REAP12mm-4bit","Inkling-REAP25mm-4bit","Inkling-REAP50mm-4bit"):
    if k not in r: continue
    a = r[k]; b = before_audio.get(k)
    ba = f"{b:.3f}->{a['audio_mean_overlap']:.3f}" if b else f"->{a['audio_mean_overlap']:.3f}"
    print(f"{k:<24}{ba:>28}{str(a['image_pass'])+'/'+str(a['image_n']):>10}{a['ppl']:>9}")
PY
