#!/usr/bin/env bash
# Flash Next np2 MTP3 with the draft on ROCm0: load-only probes over pool x split, then report VRAM.
cd "$(dirname "$0")"
export BUILD_DIR=build-mslot-ring SKIP=1
HF=/home/raistlin/.cache/huggingface/hub
FN=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
DQ8=$HF/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
A="-ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off -ctk f16 -ctv f16 -m $FN -b 1024 -ub 512 -md $DQ8 -devd ROCm0 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 64 --spec-draft-backend-sampling -np 2 --kv-unified"
for spec in "65536 0.44,0.56" "65536 0.40,0.60" "81920 0.44,0.56" "81920 0.40,0.60" "98304 0.40,0.60" "98304 0.36,0.64"; do
  set -- $spec
  ./run2.sh "pr_d0_$1_$2" 2 -- $A -ts $2 -c $1 2>&1
done
