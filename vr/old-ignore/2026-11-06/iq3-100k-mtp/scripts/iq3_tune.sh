#!/usr/bin/env bash
# usage: iq3_tune.sh OUT BUILD   UD-IQ3_XXS (Q8_0 table), 100K, F16 KV, ub 512, Q8_0 sidecar on ROCm1, -ts 0.52,0.48: MTP2 + grouped MoE, MTP3, MTP3 + grouped MoE
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
M=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
DQ8=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
PROMPT=/home/raistlin/infer/vrversion-llama.cpp/flake.nix
SHA=/home/raistlin/infer/vrversion-llama.cpp/vr/old-ignore/2026-11-05/catchup-4b77dee05/scripts/reply_sha.py
run() {  # label moe nmax
  local label=$1 moe=$2 nmax=$3
  local f=$OUT/$label
  GGML_CUDA_FLASHNEXT_MOE=$moe timeout --kill-after=5s 600s "$B/bin/llama-cli" --offline -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 0.52,0.48 -fa on \
    -b 1024 -ub 512 -c 100352 -t 20 --temp 0 --top-k 1 --seed 42 \
    -f "$PROMPT" -n 512 --no-display-prompt --simple-io \
    -md "$DQ8" -devd ROCm1 --spec-draft-n-ubatch 64 --spec-type draft-mtp --spec-draft-n-max $nmax --spec-draft-backend-sampling \
    > "$f.txt" 2> "$f.err" < /dev/null
  echo "$label rc=$? $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') $(python3 -I "$SHA" "$f.txt" 2>/dev/null | cut -d' ' -f1-3)"
  grep -iE 'out of memory|failed to (allocate|load)|abort' "$f.err" | head -1
}
run mtp2-moe1 1 2
run mtp3-moe0 0 3
run mtp3-moe1 1 3
