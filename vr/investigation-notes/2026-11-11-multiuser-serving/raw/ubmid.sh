#!/usr/bin/env bash
# llama-bench Flash Next PP2048 at ubatch 384 and 448 (256 and 512 measured earlier: 534.5 / 598.2 at d0, 388.8 / 434.0 at d32768)
source /home/raistlin/amd-clean/env/therock.sh
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
FN=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
/home/raistlin/infer/vrversion-llama.cpp/build-mslot-ring/bin/llama-bench -m $FN -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa 1 -b 1024 -ub 384,448 -t 20 -p 2048 -n 0 -d 0,32768 -r 3 2>&1 | grep -E "^\|"
