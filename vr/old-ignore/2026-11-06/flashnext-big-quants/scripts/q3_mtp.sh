#!/usr/bin/env bash
# usage: q3_mtp.sh OUT BUILD   UD-Q3_K_XL (Q8_0 n-gram table) at ctx 100352, MTP2 with the Q8_0 sidecar placed three ways
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
M=/home/raistlin/models/flashnext/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf
SIDECAR=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
PROMPT=/home/raistlin/infer/vrversion-llama.cpp/flake.nix
SHA=/home/raistlin/infer/vrversion-llama.cpp/vr/old-ignore/2026-11-05/catchup-4b77dee05/scripts/reply_sha.py
for run in rocm1 rocm1-kvq8 cpu-experts; do
  case $run in
    rocm1)       EXTRA=(-devd ROCm1);;
    rocm1-kvq8)  EXTRA=(-devd ROCm1 -ctkd q8_0 -ctvd q8_0);;
    cpu-experts) EXTRA=(-cmoed);;
  esac
  f=$OUT/mtp2-$run
  timeout --kill-after=5s 600s "$B/bin/llama-cli" --offline -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on \
    -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 -c 100352 -t 20 --temp 0 --top-k 1 --seed 42 \
    -f "$PROMPT" -n 512 --no-display-prompt --simple-io \
    -md "$SIDECAR" --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling "${EXTRA[@]}" \
    > "$f.txt" 2> "$f.err" < /dev/null
  rc=$?
  echo "mtp2-$run rc=$rc $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') $(python3 -I "$SHA" "$f.txt" 2>/dev/null | cut -d' ' -f1-3)"
  grep -iE 'out of memory|failed to (allocate|load|create)|abort' "$f.err" | head -3
done
