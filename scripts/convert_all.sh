#!/bin/zsh
# Standard MLX quant sweep for Inkling: bf16 + 4/6/8-bit.
# Usage: scripts/convert_all.sh [SRC] [OUTROOT]
set -e
SRC=${1:-/Users/david/llm/Inkling-src}
OUT=${2:-/Users/david/llm/inkling-mlx-out}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"

run() {  # name bits-args...
  local name=$1; shift
  local dst="$OUT/Inkling-$name"
  echo "==== $name -> $dst ===="
  python3 -m inkling_mlx.convert_cli --src "$SRC" --dst "$dst" "$@" 2>&1 | tee "$OUT/convert_$name.log"
}

run bf16
run 8bit --bits 8
run 6bit --bits 6
run 4bit --bits 4
echo "ALL QUANTS DONE -> $OUT"
