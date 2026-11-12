#!/usr/bin/env bash
# Flash Next np2 MTP3 context sweep on build-mslot-ring: descending pools, long fill at ~90% of the pool.
cd "$(dirname "$0")"
export BUILD_DIR=build-mslot-ring LF=longfill2.py
HF=/home/raistlin/.cache/huggingface/hub
FN=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
DQ8=$HF/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
A="-ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off -ctk f16 -ctv f16 -m $FN -b 1024 -ub 512 -md $DQ8 -devd ROCm1 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 64 --spec-draft-backend-sampling -np 2 --kv-unified"
for spec in "73728 0.52,0.48" "65536 0.52,0.48"; do
  set -- $spec
  FILL=$(( $1 * 9 / 10 )) ./run2.sh "sw3_np2mtp_$1" 2 -- $A -ts $2 -c $1 2>&1 | tee out/last.txt
  # stop at the first pool that loads, serves and answers the long fill
  :
done
