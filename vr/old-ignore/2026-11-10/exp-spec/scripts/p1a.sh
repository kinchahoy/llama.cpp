#!/usr/bin/env bash
# usage: p1a.sh LABEL BUILD OUT   greedy partial offload, share-view on/off
LABEL=$1; B=$2; OUT=$3; mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP
M27=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
for sv in 1 0; do
  f=$OUT/$LABEL-sv$sv
  GGML_SCHED_SHARE_VIEW_INPUTS=$sv timeout --kill-after=5s 240s "$B/bin/llama-completion" -m "$M27" --offline \
    -ngl ${NGL:-56} -dev ROCm0,ROCm1 -sm layer -fa on -b 512 -ub 512 -c 4096 -t 20 -tb 20 \
    --temp 0 --top-k 1 -s 42 -n 48 -no-cnv --simple-io --no-display-prompt \
    -p 'The history of the printing press begins in' < /dev/null > "$f.txt" 2> "$f.log"
  echo "$LABEL sv=$sv rc=$? sha=$(sha256sum < "$f.txt" | cut -c1-12) $(grep -E 'eval time' "$f.log" | grep -oE '[0-9.]+ tokens per second' | tr '\n' ' ')"
done
