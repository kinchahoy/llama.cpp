#!/usr/bin/env bash
# usage: q3_mtp2.sh OUT BUILD   UD-Q3_K_XL (Q8_0 table), F16 KV, -b 512 -ub 256; plain vs MTP2 with the Q4_0 draft on ROCm1 (experts on CPU) at ctx 100352 and 65536
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
M=/home/raistlin/models/flashnext/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf
D4=$(ls /home/raistlin/.cache/huggingface/hub/models--bartowski--Qwen3.8-Flash-Next-GGUF/snapshots/*/mtp-Qwen3.8-Flash-Next-Q4_0.gguf | head -1)
PROMPT=/home/raistlin/infer/vrversion-llama.cpp/flake.nix
SHA=/home/raistlin/infer/vrversion-llama.cpp/vr/old-ignore/2026-11-05/catchup-4b77dee05/scripts/reply_sha.py
echo "draft: $D4"
for ctx in 100352 65536; do
  for run in mtp2; do
    SPEC=()
    [ $run = mtp2 ] && SPEC=(-md "$D4" -devd ROCm1 -cmoed --spec-draft-n-ubatch 64 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling)
    f=$OUT/$run-c$ctx
    timeout --kill-after=5s 600s "$B/bin/llama-cli" --offline -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on \
      -b 512 -ub 256 -c $ctx -t 20 --temp 0 --top-k 1 --seed 42 \
      -f "$PROMPT" -n 512 --no-display-prompt --simple-io "${SPEC[@]}" > "$f.txt" 2> "$f.err" < /dev/null
    rc=$?
    echo "$run c=$ctx rc=$rc $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') $(python3 -I "$SHA" "$f.txt" 2>/dev/null | cut -d' ' -f1-3)"
    grep -iE 'out of memory|failed to (allocate|load)|abort' "$f.err" | head -2
  done
done
