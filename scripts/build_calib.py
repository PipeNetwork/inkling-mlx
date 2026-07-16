"""Build a wide, diverse calibration corpus for MoE expert profiling and save it as
a JSON list of text chunks. Sources: local real code (Python/Rust), streamed
multilingual Wikipedia (many languages -> exercises language-specialist experts),
and curated math/reasoning/structured prompts. Robust to dataset/network failure.
"""
import glob, json, os, sys

OUT = "/Users/david/llm/inkling-mlx-out/calib_wide.json"
chunks: list[str] = []


def add(text, max_chars=2400):
    text = (text or "").strip()
    if len(text) < 200:
        return
    chunks.append(text[:max_chars])


# 1) local real code (diverse: python + rust)
for pat in ("/Users/david/llm/inkling-mlx/inkling_mlx/*.py",
            "/Users/david/llm/inkling-mlx/scripts/*.py",
            "/Users/david/llm/hi/**/*.rs", "/Users/david/llm/hi/**/*.py",
            "/Users/david/llm/longcat2-mlx/*.py"):
    for f in glob.glob(pat, recursive=True)[:40]:
        try:
            add(open(f, encoding="utf-8", errors="ignore").read())
        except Exception:
            pass
print(f"[calib] after local code: {len(chunks)} chunks", flush=True)

# 2) streamed multilingual wikipedia (best-effort)
LANGS = ["en", "zh", "es", "fr", "de", "ru", "ja", "ar", "hi", "pt", "ko", "it", "tr", "vi", "fa"]
try:
    from datasets import load_dataset
    for lang in LANGS:
        try:
            ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train",
                              streaming=True)
            n = 0
            for ex in ds:
                add(ex.get("text", ""))
                n += 1
                if n >= 8:
                    break
        except Exception as e:
            print(f"[calib] wiki {lang} skipped: {str(e)[:60]}", flush=True)
    print(f"[calib] after wikipedia: {len(chunks)} chunks", flush=True)
except Exception as e:
    print(f"[calib] datasets unavailable ({str(e)[:60]}); curated only", flush=True)

# 3) curated math / reasoning / structured / dialogue (add breadth the above may miss)
CURATED = [
    "Solve step by step: integrate x^2 * e^x dx.",
    "Prove by induction that the sum of the first n odd numbers is n^2.",
    "A committee of 3 is chosen from 10 people. How many ways, and why?",
    "Explain the Fourier transform intuitively, then give the definition.",
    "Compute the eigenvalues of [[2,1],[1,2]] and show the work.",
    "Write a proof that there are infinitely many primes.",
    "Derive the quadratic formula from ax^2+bx+c=0.",
    "Explain Bayes' theorem with a medical-testing example and numbers.",
    "Return JSON: a weather report with city, temp_c, conditions, forecast[3].",
    "Write an XML document describing a library with 2 books.",
    "Write a bash script that finds and deletes files older than 30 days.",
    "Explain the CAP theorem and give an example system for each pair.",
    "Write a Haskell function that computes the Fibonacci sequence lazily.",
    "Translate to German, French, and Japanese: 'Knowledge is power.'",
    "Write a short dialogue between a customer and a support agent about a refund.",
    "Explain gradient descent and write the update rule in LaTeX.",
    "Describe the TCP three-way handshake in detail.",
    "Write a SQL schema for a blog with users, posts, and comments.",
]
for c in CURATED:
    chunks.append(c)

json.dump(chunks, open(OUT, "w"))
# rough token estimate (chars/4)
tot_chars = sum(len(c) for c in chunks)
print(f"[calib] saved {len(chunks)} chunks (~{tot_chars//4} tokens est) -> {OUT}", flush=True)
