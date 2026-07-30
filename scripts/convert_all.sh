#!/bin/zsh
# Standard MLX quant sweep for an Inkling checkpoint: 4/6/8-bit + bf16.
# Smallest first, so a coherence smoke test can run while the big ones convert.
#
# Usage: scripts/convert_all.sh [SRC] [OUTROOT] [PREFIX]
#   scripts/convert_all.sh /Users/david/llm/Inkling-src       ~/llm/inkling-mlx-out    Inkling
#   scripts/convert_all.sh /Users/david/llm/Inkling-Small-src ~/llm/inkling-small-out  Inkling-Small
set -e
SRC=${1:-/Users/david/llm/Inkling-src}
OUT=${2:-/Users/david/llm/inkling-mlx-out}
PREFIX=${3:-Inkling}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"

run() {  # name bits-args...
  local name=$1; shift
  local dst="$OUT/$PREFIX-$name"
  echo "==== $name -> $dst ===="
  python3 -m inkling_mlx.convert_cli --src "$SRC" --dst "$dst" "$@" 2>&1 | tee "$OUT/convert_$name.log"
}

run 4bit --bits 4
run 6bit --bits 6
run 8bit --bits 8
run bf16
echo "ALL QUANTS DONE -> $OUT"
