#!/usr/bin/env bash
# usage: q3_target.sh OUT BUILD   UD-Q3_K_XL (Q8_0 n-gram table) at ctx 100352: plain, MTP2 with the shared Q8_0 draft, MTP2 with the Q8_0 sidecar
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
M=/home/raistlin/models/flashnext/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf
SNAP=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c
SHARED=$SNAP/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
SIDECAR=$SNAP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
PROMPT=/home/raistlin/infer/vrversion-llama.cpp/flake.nix
SHA=/home/raistlin/infer/vrversion-llama.cpp/vr/old-ignore/2026-11-05/catchup-4b77dee05/scripts/reply_sha.py
for run in plain mtp2-shared mtp2-sidecar; do
  case $run in
    plain)        SPEC=();;
    mtp2-shared)  SPEC=(-md "$SHARED"  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling);;
    mtp2-sidecar) SPEC=(-md "$SIDECAR" --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling);;
  esac
  f=$OUT/$run
  timeout --kill-after=5s 600s "$B/bin/llama-cli" --offline -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on \
    -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 -c 100352 -t 20 --temp 0 --top-k 1 --seed 42 \
    -f "$PROMPT" -n 512 --no-display-prompt --simple-io "${SPEC[@]}" > "$f.txt" 2> "$f.err" < /dev/null
  rc=$?
  echo "$run rc=$rc $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') $(python3 -I "$SHA" "$f.txt" 2>/dev/null | cut -d' ' -f1-3)"
  grep -iE 'out of memory|failed to (allocate|load|create)|error|abort|unknown model arch|missing tensor' "$f.err" | grep -v 'error: invalid' | head -4
done
