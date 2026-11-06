#!/usr/bin/env bash
# usage: q3_mtp_ts.sh OUT BUILD   UD-Q3_K_XL (Q8_0 table) at ctx 65536, F16 KV, -ub 256, MTP2 Q4_0 draft on ROCm1 (experts on CPU), target -ts skewed toward ROCm0
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
M=/home/raistlin/models/flashnext/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf
D4=$(ls /home/raistlin/.cache/huggingface/hub/models--bartowski--Qwen3.8-Flash-Next-GGUF/snapshots/*/mtp-Qwen3.8-Flash-Next-Q4_0.gguf | head -1)
PROMPT=/home/raistlin/infer/vrversion-llama.cpp/flake.nix
SHA=/home/raistlin/infer/vrversion-llama.cpp/vr/old-ignore/2026-11-05/catchup-4b77dee05/scripts/reply_sha.py
for ts in 0.505,0.495 0.51,0.49; do
  f=$OUT/mtp2-ts$ts
  timeout --kill-after=5s 600s "$B/bin/llama-cli" --offline -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts $ts -fa on \
    -b 512 -ub 256 -c 65536 -t 20 --temp 0 --top-k 1 --seed 42 \
    -f "$PROMPT" -n 512 --no-display-prompt --simple-io \
    -md "$D4" -devd ROCm1 -cmoed --spec-draft-n-ubatch 64 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling \
    > "$f.txt" 2> "$f.err" < /dev/null
  rc=$?
  echo "mtp2 ts=$ts rc=$rc $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') $(python3 -I "$SHA" "$f.txt" 2>/dev/null | cut -d' ' -f1-3)"
  grep -iE 'out of memory|failed to (allocate|load)|abort' "$f.err" | head -2
  [ $rc -eq 0 ] && break
done
