#!/usr/bin/env bash
# usage: output_check.sh LABEL BUILD_DIR OUT_DIR
# greedy completion on both target models under -sm tensor; output text is compared across stages
set -u
LABEL=$1; BUILD=$2; OUT=$3
mkdir -p "$OUT"
HUB=/home/raistlin/.cache/huggingface/hub
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
MFN=$HUB/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
BIN=$BUILD/bin/llama-completion
PROMPT="The history of the printing press begins in"
unset GGML_CUDA_TP_OVERLAP

for m in 27b fn; do
    if [ $m = 27b ]; then M=$M27; EXTRA="-b 2048 -ub 2048"; else M=$MFN; EXTRA="-ctk q8_0 -ctv q8_0 -b 1024 -ub 512"; fi
    timeout --kill-after=5s 300s "$BIN" -m "$M" -ngl 999 -sm tensor -dev ROCm0,ROCm1 -ts 1,1 -fa on $EXTRA \
        -c 4096 -n 96 --temp 0 --seed 1 -no-cnv --no-display-prompt -p "$PROMPT" \
        > "$OUT/$LABEL-$m.txt" 2> "$OUT/$LABEL-$m.stderr" < /dev/null
    echo "$LABEL $m exit=$? sha=$(sha256sum < "$OUT/$LABEL-$m.txt" | cut -c1-12) bytes=$(wc -c < "$OUT/$LABEL-$m.txt")"
done
