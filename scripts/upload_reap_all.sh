#!/bin/zsh
# Upload the three REAP builds sequentially. Resumable (upload_large_folder).
set -u
cd /Users/david/llm/inkling-mlx
OUT=/Users/david/llm/inkling-mlx-out
for spec in "REAP12-4bit Inkling-REAP12-4bit" "REAP25-4bit Inkling-REAP25-4bit" "REAP50-4bit Inkling-REAP50-4bit"; do
  set -- ${=spec}
  echo "===== [upload] $1 begin ====="
  python3 scripts/upload_reap.py "$1" "$OUT/$2" 2>&1 | grep -vE "^You are using|PyTorch|resume_download|warnings.warn"
  echo "===== [upload] $1 end (rc=$?) ====="
done
echo "[upload] ALL DONE"
