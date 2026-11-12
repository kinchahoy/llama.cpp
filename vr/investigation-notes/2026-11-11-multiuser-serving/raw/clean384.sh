#!/usr/bin/env bash
# flashnext ub384, 2 slots MTP3: descending pools, full checks; stop at the first that passes the 2-concurrent long test.
cd "$(dirname "$0")"
export BUILD_DIR=build-mslot-ring SKIP=1
HF=/home/raistlin/.cache/huggingface/hub
FN=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
DQ8=$HF/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
A="-ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off -ctk f16 -ctv f16 -m $FN -b 1024 -ub 384 -md $DQ8 -devd ROCm1 -ts 0.52,0.48 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 64 --spec-draft-backend-sampling -np 2 --kv-unified"
for c in 94208 90112; do
  CL=30000 ./run3.sh "clean_u384_$c" 2 -- $A -c $c 2>&1 | tee out/last.txt
  :
done
