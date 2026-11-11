#!/bin/bash
source /home/raistlin/amd-clean/env/therock.sh
B=/home/raistlin/infer/vrversion-llama.cpp/build-exp-q3
OUT=/tmp/exp-q3/trace
rm -rf $OUT; mkdir -p $OUT
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP
timeout --kill-after=5s 240s rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d $OUT -o tg -- \
  $B/bin/llama-bench -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on \
  -b 2048 -ub 2048 -t 20 -p 0 -n 64 -d 0 -r 1 > $OUT/bench.txt 2> $OUT/bench.log
echo exit $?
find $OUT -name "*.csv" | head
