#!/usr/bin/env bash
# Flash Next np2 MTP3, draft ROCm1, -ub 256: full checks at 80K and 98K, then llama-bench PP ub256 vs ub512.
cd "$(dirname "$0")"
while pgrep -f probe.sh >/dev/null; do sleep 10; done
export BUILD_DIR=build-mslot-ring LF=longfill2.py
HF=/home/raistlin/.cache/huggingface/hub
FN=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
DQ8=$HF/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
A="-ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off -ctk f16 -ctv f16 -m $FN -b 1024 -ub 256 -md $DQ8 -devd ROCm1 -ts 0.52,0.48 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 64 --spec-draft-backend-sampling -np 2 --kv-unified"
for c in 98304 81920; do
  FILL=$(( c * 9 / 10 )) ./run2.sh "ub256_np2mtp_$c" 2 -- $A -c $c 2>&1
done
source /home/raistlin/amd-clean/env/therock.sh
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
echo "== llama-bench FN PP ub256 vs ub512 (build-mslot-ring)"
/home/raistlin/infer/vrversion-llama.cpp/build-mslot-ring/bin/llama-bench -m $FN -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa 1 -b 1024 -ub 256,512 -t 20 -p 2048 -n 0 -d 0,32768 -r 3 2>&1 | grep -E "^\|"
