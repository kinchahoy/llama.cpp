#!/bin/bash
# usage: prof.sh BUILD_DIR OUTNAME [ENV=VAL ...]
source /home/raistlin/amd-clean/env/therock.sh
B=$1; N=$2; shift 2
for kv in "$@"; do export "$kv"; done
OUT=/home/raistlin/.claude/jobs/5a8c59bb/tmp/p3/prof
mkdir -p $OUT
M=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
timeout --kill-after=5s 300s rocprofv3 --kernel-trace --output-format csv -d $OUT/$N -o $N -- \
  "$B/bin/llama-bench" -m $M -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts ${TS:-1/1} -fa on -b 2048 -ub 2048 -t 20 -p 0 -n 64 -r 1 > $OUT/$N.log 2>&1
echo rc=$?
