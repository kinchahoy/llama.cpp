#!/bin/bash
source /home/raistlin/amd-clean/env/therock.sh
B=/home/raistlin/infer/vrversion-llama.cpp/build-exp-q3
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP
for r in 8 4 8 4; do
  GGML_RP_MMV_ROWS=$r timeout --kill-after=5s 150s $B/bin/llama-bench -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
    -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on -b 2048 -ub 2048 -t 20 -p 0 -n 128 -d 0 -r 3 -o csv 2>/dev/null | tail -1 | awk -F, -v r=$r '{print "ROWS=" r, $(NF-1), $NF}'
done
