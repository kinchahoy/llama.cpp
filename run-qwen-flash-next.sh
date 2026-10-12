#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL=${QWEN_FLASH_MODEL:-/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}

exec "$ROOT/build/bin/llama-cli" -m "$MODEL" -ngl 999 -sm tensor -ts 1,1 -c "${CTX_SIZE:-8192}" -fa on -ctk f16 -ctv f16 -lm dio -lzm auto --no-host --no-warmup -t "${THREADS:-16}" --simple-io "$@"
