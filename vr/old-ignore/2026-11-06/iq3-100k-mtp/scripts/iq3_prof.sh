#!/usr/bin/env bash
# usage: iq3_prof.sh OUT BUILD   kernel traces: (A) llama-bench PP2048 on UD-IQ3_XXS, (B) MTP3 + grouped MoE generation at 100K
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
P=/home/raistlin/amd-clean/therock-venv/.venv/bin/rocprofv3
M=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
DQ8=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
timeout 900s $P --kernel-trace --stats -f csv -d "$OUT/pp" -o pp -- "$B/bin/llama-bench" -m "$M" -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on \
  -b 1024 -ub 512 -t 20 -p 2048 -n 0 -r 1 > "$OUT/pp.log" 2>&1
echo "pp rc=$?"
GGML_CUDA_FLASHNEXT_MOE=1 timeout 900s $P --kernel-trace --stats -f csv -d "$OUT/tg" -o tg -- "$B/bin/llama-cli" --offline -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 0.52,0.48 -fa on \
  -b 1024 -ub 512 -c 100352 -t 20 --temp 0 --top-k 1 --seed 42 -p 'Write a detailed history of the printing press.' -n 256 --no-display-prompt --simple-io \
  -md "$DQ8" -devd ROCm1 --spec-draft-n-ubatch 64 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling < /dev/null > "$OUT/tg.txt" 2> "$OUT/tg.log"
echo "tg rc=$?"
find "$OUT" -name '*kernel_stats*.csv' | head
