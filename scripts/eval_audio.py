"""Audio acceptance test: transcribe held-out LibriSpeech clips and score by content-word
overlap with the ground-truth transcript. Analogous to eval_images.py. Detects the same
failure mode for audio that text-only/text+image calibration can cause (audio-grounding
experts pruned -> transcription collapses).

    python scripts/eval_audio.py <MODEL_DIR>
"""
import json, os, sys
import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inkling_mlx.load import load
from inkling_mlx.generate import greedy_generate
from inkling_mlx.processing import InklingProcessor
from transformers import AutoTokenizer
from audio_util import load_wav, librispeech_index, transcription_overlap, content_words

MODEL = sys.argv[1]
OUT = "/Users/david/llm/inkling-mlx-out/audio_eval.json"
LS = "/Users/david/llm/inkling-mlx-out/LibriSpeech/dev-clean"
LAZY = os.environ.get("INKLING_LAZY") == "1"
MAXTOK = int(os.environ.get("AUDIO_MAXTOK", "64"))  # lower for near-capacity lazy runs
PASS_THRESH = 0.30                                # 4-bit ASR isn't verbatim; lenient overlap

# pick 6 deterministic clips: distinctive (>=10 content words), 3-10 s long
idx = librispeech_index(LS)
picked = []
for fp, text in idx:
    if len(content_words(text)) < 10:
        continue
    wav = load_wav(fp)
    dur = len(wav) / 16000.0
    if 3.0 <= dur <= 10.0:
        picked.append((fp, text, wav))
    if len(picked) >= 6:
        break

print(f"[aeval] loading {os.path.basename(MODEL)} ({'lazy' if LAZY else 'eager'}); {len(picked)} clips", flush=True)
model, config = load(MODEL, lazy=LAZY)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
proc = InklingProcessor(tok, open(f"{MODEL}/chat_template.jinja").read())
Q = "Transcribe the speech in this audio."

results = []
npass = 0
for fp, text, wav in picked:
    inp = proc.apply([{"role": "user", "content": [
        {"type": "audio", "audio": wav, "sampling_rate": 16000},
        {"type": "text", "text": Q}]}], reasoning_effort="none")
    out = greedy_generate(model, config, inp["input_ids"], max_new_tokens=MAXTOK,
                          audio_input_ids=inp.get("audio_input_ids"))
    resp = tok.decode(out[len(inp["input_ids"]):])
    mx.clear_cache()                              # release per-clip buffers (near-capacity)
    ov = transcription_overlap(resp, text)
    ok = ov >= PASS_THRESH
    npass += ok
    results.append({"ref": text, "resp": resp, "overlap": round(ov, 2), "pass": ok})
    print(f"[aeval] {'PASS' if ok else 'FAIL'} ov={ov:.2f}\n    REF : {text[:80]}\n    HYP : {resp.strip()[:80]}", flush=True)

score = f"{npass}/{len(results)}"
mean_ov = round(sum(r["overlap"] for r in results) / max(1, len(results)), 3)
print(f"[aeval] SCORE {score}  mean_overlap {mean_ov}", flush=True)
allres = json.load(open(OUT)) if os.path.exists(OUT) else {}
allres[os.path.basename(MODEL)] = {"score": score, "mean_overlap": mean_ov, "items": results}
json.dump(allres, open(OUT, "w"), indent=2)
print(f"[aeval] saved -> {OUT}", flush=True)
