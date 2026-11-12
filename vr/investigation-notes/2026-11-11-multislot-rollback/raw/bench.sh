#!/usr/bin/env bash
# llama-bench PP2048 / TG128, old vs new, Qwen 27B Q8_0 and Flash Next (single sequence, no draft)
source /home/raistlin/amd-clean/env/therock.sh
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
pgrep -x llama-server >/dev/null && { echo "llama-server running, abort"; exit 1; }
Q=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
F=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
for B in build-play-b1 build-mslot-ring; do
  BIN=/home/raistlin/infer/vrversion-llama.cpp/$B/bin/llama-bench
  echo "== $B 27B"
  $BIN -m $Q -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa 1 -b 2048 -ub 2048 -t 20 -p 2048 -n 128 -r 3 -o md 2>/dev/null | grep -E "pp2048|tg128"
  echo "== $B Flash Next"
  $BIN -m $F -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 0.52/0.48 -fa 1 -b 1024 -ub 512 -t 20 -p 2048 -n 128 -r 3 -o md 2>/dev/null | grep -E "pp2048|tg128"
done
