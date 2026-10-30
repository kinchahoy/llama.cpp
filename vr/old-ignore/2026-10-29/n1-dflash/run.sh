#!/usr/bin/env bash
# usage: run.sh BUILD OUT MODE   (MODE: plain mtp2 d5 d7)
BUILD=$1; OUT=$2; MODE=$3; shift 3
HUB=/home/raistlin/.cache/huggingface/hub
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
DFL=$HUB/models--z-lab--Qwen3.8-27B-DFlash2-GGUF/snapshots/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q8_0.gguf
unset GGML_CUDA_TP_OVERLAP
SPEC=()
case $MODE in
 d5) SPEC=(--spec-type draft-dflash --spec-draft-n-max 5 -md $DFL);;
 d7) SPEC=(--spec-type draft-dflash --spec-draft-n-max 7 -md $DFL);;
 mtp2) SPEC=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling);;
esac
timeout --kill-after=5s 420s $BUILD/bin/llama-cli --offline -m $M27 -b 2048 -ub 2048 -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -c 16384 --temp 0 --top-k 1 --seed 42 -f /home/raistlin/infer/mx-llama.cpp/flake.nix -n 512 --no-display-prompt --simple-io "${SPEC[@]}" "$@" > $OUT.txt 2> $OUT.err < /dev/null
echo "$OUT rc=$? $(grep -o 'Generation: [0-9.]* t/s' $OUT.txt) sha=$(grep -v '^\[ Prompt' $OUT.txt | sha256sum | cut -c1-12)"
