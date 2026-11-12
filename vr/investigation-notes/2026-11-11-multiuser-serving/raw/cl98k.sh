#!/usr/bin/env bash
# flashnext preset settings: 2 concurrent ~44K-token requests in the 98304 pool.
cd "$(dirname "$0")"
while pgrep -f ub256.sh >/dev/null; do sleep 10; done
export BUILD_DIR=build-mslot-ring SKIP=1 CL=44000
HF=/home/raistlin/.cache/huggingface/hub
FN=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
DQ8=$HF/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
A="-ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off -ctk f16 -ctv f16 -m $FN -b 1024 -ub 256 -md $DQ8 -devd ROCm1 -ts 0.52,0.48 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 64 --spec-draft-backend-sampling -np 2 --kv-unified -c 98304"
./run3.sh cl_ub256_np2mtp_98304 2 -- $A 2>&1
