#!/bin/bash
# usage: greedy.sh LABEL BUILD_DIR [ENV=VAL ...]; 64-token greedy 27B text
source /home/raistlin/amd-clean/env/therock.sh
L=$1; B=$2; shift 2
for kv in "$@"; do export "$kv"; done
OUT=/home/raistlin/.claude/jobs/5a8c59bb/tmp/q1/greedy
mkdir -p $OUT
i=0
M=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
for p in "The history of the printing press begins in" "Write a C function that reverses a linked list."; do
  i=$((i+1))
  timeout --kill-after=5s 200s "$B/bin/llama-completion" -m $M -ngl 999 -sm tensor -dev ROCm0,ROCm1 -ts 1,1 -fa on -b 2048 -ub 2048 \
    -c 4096 -n 64 --temp 0 --seed 1 -no-cnv --no-display-prompt -p "$p" > $OUT/$L-$i.txt 2> $OUT/$L-$i.stderr < /dev/null
  echo "$L p$i exit=$? sha=$(sha256sum < $OUT/$L-$i.txt | cut -c1-12) bytes=$(wc -c < $OUT/$L-$i.txt)"
done
