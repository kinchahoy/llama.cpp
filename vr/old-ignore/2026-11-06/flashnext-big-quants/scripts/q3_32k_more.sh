#!/usr/bin/env bash
# usage: q3_32k_more.sh OUT BUILD   32K: MTP3 at ub 256, MTP2 at ub 512, both with the full Q4_0 draft on ROCm1 and -ts 0.515,0.485
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
M=/home/raistlin/models/flashnext/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf
D4=/home/raistlin/.cache/huggingface/hub/models--bartowski--Qwen3.8-Flash-Next-GGUF/snapshots/db3a185731e162be9ffa4d52d20d7509939e60e8/mtp-Qwen3.8-Flash-Next-Q4_0.gguf
PROMPT=/home/raistlin/infer/vrversion-llama.cpp/flake.nix
SHA=/home/raistlin/infer/vrversion-llama.cpp/vr/old-ignore/2026-11-05/catchup-4b77dee05/scripts/reply_sha.py
run() {  # label b ub nmax
  local label=$1 b=$2 ub=$3 nmax=$4
  local f=$OUT/$label
  timeout --kill-after=5s 600s "$B/bin/llama-cli" --offline -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 0.515,0.485 -fa on \
    -b $b -ub $ub -c 32768 -t 20 --temp 0 --top-k 1 --seed 42 \
    -f "$PROMPT" -n 512 --no-display-prompt --simple-io \
    -md "$D4" -devd ROCm1 --spec-draft-n-ubatch 64 --spec-type draft-mtp --spec-draft-n-max $nmax --spec-draft-backend-sampling \
    > "$f.txt" 2> "$f.err" < /dev/null
  echo "$label rc=$? $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') $(python3 -I "$SHA" "$f.txt" 2>/dev/null | cut -d' ' -f1-3)"
  grep -iE 'out of memory|failed to (allocate|load)|abort' "$f.err" | head -2
}
run mtp3-ub256 512 256 3
run mtp2-ub512 1024 512 2
