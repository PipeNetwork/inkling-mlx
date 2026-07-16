"""Minimal chat example for an Inkling MLX build.

    python examples/chat.py /path/to/Inkling-MLX-4bit "What is the capital of France?"
"""
import sys
from inkling_mlx.load import load
from inkling_mlx.generate import greedy_generate
from transformers import AutoTokenizer

model_dir = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "What is the capital of France?"

model, config = load(model_dir)
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tok.chat_template = open(f"{model_dir}/chat_template.jinja").read()

ids = tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True, reasoning_effort="none", tokenize=True)["input_ids"]
out = greedy_generate(model, config, ids, max_new_tokens=128)
print(tok.decode(out[len(ids):]))
