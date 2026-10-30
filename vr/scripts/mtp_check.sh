#!/usr/bin/env bash
# usage: mtp_check.sh LABEL BUILD_DIR OUT_DIR
# one run each of plain, MTP2 and MTP3 for 27B Q8_0 and Flash Next IQ3_XXS (Q8_0 MTP sidecar), greedy, flake.nix prompt
set -u
LABEL=$1; BUILD=$2; OUT=$3
mkdir -p "$OUT"
HUB=/home/raistlin/.cache/huggingface/hub
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
MFN=$HUB/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
MTP_FN=/home/raistlin/.cache/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
PROMPT=/home/raistlin/infer/mx-llama.cpp/flake.nix
BIN=$BUILD/bin/llama-cli
unset GGML_CUDA_TP_OVERLAP

COMMON=(--offline -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -c 16384 --temp 0 --top-k 1 --seed 42
        -f "$PROMPT" -n 512 --no-display-prompt --simple-io)

for m in 27b fn; do
    if [ $m = 27b ]; then
        MODEL=(-m "$M27" -b 2048 -ub 2048); DRAFT=()
    else
        MODEL=(-m "$MFN" -b 1024 -ub 512 -ctk q8_0 -ctv q8_0); DRAFT=(-md "$MTP_FN")
    fi
    for mode in plain mtp2 mtp3; do
        SPEC=()
        [ $mode = mtp2 ] && SPEC=("${DRAFT[@]}" --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling)
        [ $mode = mtp3 ] && SPEC=("${DRAFT[@]}" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling)
        f=$OUT/$LABEL-$m-$mode
        timeout --kill-after=5s 420s "$BIN" "${MODEL[@]}" "${COMMON[@]}" "${SPEC[@]}" > "$f.txt" 2> "$f.stderr" < /dev/null
        rc=$?
        pp=$(grep -oE 'prompt eval time.*\(.*([0-9.]+) tokens per second' "$f.stderr" | grep -oE '[0-9.]+ tokens per second' | head -1)
        tg=$(grep -E '^\s*(common_perf_print|llama_perf_context_print):\s+eval time' "$f.stderr" | grep -oE '[0-9.]+ tokens per second' | head -1)
        acc=$(grep -iE 'accept' "$f.stderr" | tail -2 | tr '\n' ' ' | cut -c1-200)
        echo "$LABEL $m $mode exit=$rc PP=[$pp] TG=[$tg] sha=$(sha256sum < "$f.txt" | cut -c1-12) acc=[$acc]"
    done
done
