#!/bin/zsh
# Fetch the image + audio corpora used by the multimodal REAP calibration
# (scripts/profile_experts_mm.py) and the per-build evals (scripts/eval_build.py).
#
# Text calibration comes from scripts/build_calib.py; this covers the other two
# modalities, which matter: a text-only calibration prunes the experts that ground
# vision and audio while text perplexity still looks fine.
#
# Usage: scripts/fetch_calib_assets.sh [DEST]      (default /Users/david/llm/inkling-mlx-out)
set -e
DEST=${1:-/Users/david/llm/inkling-mlx-out}
mkdir -p "$DEST"
cd "$DEST"

# imagenette2-320 (~325 MB) — 10-class ImageNet subset, train + val splits
if [[ ! -d imagenette2-320 ]]; then
  echo "==== imagenette2-320 -> $DEST ===="
  curl -fL -o imagenette2-320.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
  tar xzf imagenette2-320.tgz && rm imagenette2-320.tgz
fi

# LibriSpeech dev-clean (~337 MB) — read speech with transcripts, for audio saliency
if [[ ! -d LibriSpeech/dev-clean ]]; then
  echo "==== LibriSpeech dev-clean -> $DEST ===="
  curl -fL -o dev-clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz
  tar xzf dev-clean.tar.gz && rm dev-clean.tar.gz
fi

# Pallas's cat (manul) — the fine-grained image-ID probe in scripts/eval_build.py.
# Imagenette's 10 classes are coarse; this one separates a model that really grounds
# visual features from one that pattern-matches ("brown bear" was the failure mode
# when expert pruning was calibrated on text alone).
# "Otocolobus manul by Henry Söderlund", CC BY 2.0 — https://commons.wikimedia.org/wiki/File:Otocolobus_manul_by_Henry_S%C3%B6derlund.jpg
if [[ ! -f pallas_cat.jpg ]]; then
  echo "==== Pallas's cat probe image -> $DEST ===="
  curl -fL -A "inkling-mlx-eval/1.0" -o pallas_cat.jpg \
    "https://commons.wikimedia.org/wiki/Special:FilePath/Otocolobus%20manul%20by%20Henry%20S%C3%B6derlund.jpg?width=800" || \
    echo "  (optional probe image unavailable — eval_build.py will skip it)"
fi

echo "images: $(find imagenette2-320 -name '*.JPEG' | wc -l | tr -d ' ')"
echo "audio:  $(find LibriSpeech/dev-clean -name '*.flac' | wc -l | tr -d ' ')"
echo "probe:  $([[ -f pallas_cat.jpg ]] && echo 'pallas_cat.jpg ok' || echo 'absent')"
echo "CALIB ASSETS OK -> $DEST"
