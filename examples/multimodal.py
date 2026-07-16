"""Multimodal example: ask Inkling about an image (and/or audio).

    python examples/multimodal.py /path/to/Inkling-MLX-4bit cat.jpg "What's in this image?"
"""
import sys
from PIL import Image
from inkling_mlx.load import load
from inkling_mlx.generate import greedy_generate
from inkling_mlx.processing import InklingProcessor
from transformers import AutoTokenizer

model_dir, image_path = sys.argv[1], sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else "Describe this image."

model, config = load(model_dir)
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
proc = InklingProcessor(tok, open(f"{model_dir}/chat_template.jinja").read())

inputs = proc.apply([{"role": "user", "content": [
    {"type": "image", "image": Image.open(image_path)},
    {"type": "text", "text": prompt},
]}], reasoning_effort="none",
    max_long_edge=512)   # cap image resolution -> fewer patches -> faster prefill

out = greedy_generate(model, config, inputs["input_ids"], max_new_tokens=128,
                      pixel_values=inputs.get("pixel_values"),
                      audio_input_ids=inputs.get("audio_input_ids"))
print(tok.decode(out[len(inputs["input_ids"]):]))

# Audio works the same way: {"type": "audio", "audio": <16kHz mono np.ndarray>}
