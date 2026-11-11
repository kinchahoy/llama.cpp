#!/usr/bin/env bash
# usage: cli.sh NAME BUILD mtp2|d5 [ENV=VAL ...]   standard 27B llama-cli workload (vr sweep.sh)
NAME=$1; B=$2; MODE=$3; shift 3
OUT=/home/raistlin/infer/vrversion-llama.cpp/build-exp-spec/out/cli; mkdir -p $OUT
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP
for kv in "$@"; do export "$kv"; done
HUB=/home/raistlin/.cache/huggingface/hub
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
DFL=$HUB/models--z-lab--Qwen3.8-27B-DFlash2-GGUF/snapshots/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q8_0.gguf
PROMPT=/home/raistlin/infer/vrversion-llama.cpp/flake.nix
case $MODE in
  mtp2) SPEC=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling);;
  d5)   SPEC=(--spec-type draft-dflash --spec-draft-n-max 5 -md "$DFL");;
esac
f=$OUT/$NAME
timeout --kill-after=5s 300s "$B/bin/llama-cli" --offline -m "$M27" \
  -b 2048 -ub 2048 -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -c 16384 --temp 0 --top-k 1 --seed 42 \
  -f "$PROMPT" -n 512 --no-display-prompt --simple-io "${SPEC[@]}" > "$f.txt" 2> "$f.err" < /dev/null
echo "$NAME rc=$? $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') sha=$(grep -v '^\[' "$f.txt" | sha256sum | cut -c1-12)"
grep -oE "p\(tgtsync,gather,inject\).*|dur\(b,p,g,a\)[^,]*,[^,]*,[^,]*,[^,]*|draft acceptance rate.*|#gen drafts.*" "$f.err" "$f.txt" | head -5
