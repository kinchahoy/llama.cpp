#!/usr/bin/env bash
# usage: depth.sh OUT BUILD   27B Q8_0 at ~32K prompt depth: plain, MTP2, DFlash5 (+ plain at the standard flake.nix prompt)
OUT=$1; B=$2
mkdir -p "$OUT"
REPO=/home/raistlin/infer/vrversion-llama.cpp
P32=$OUT/prompt-32k.txt
head -c 116000 "$REPO/src/llama-context.cpp" | sed '$d' > "$P32"
printf '\n\nExplain in a few paragraphs what this C++ file does and how its main pieces fit together.\n' >> "$P32"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP
HUB=/home/raistlin/.cache/huggingface/hub
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
DFL=$HUB/models--z-lab--Qwen3.8-27B-DFlash2-GGUF/snapshots/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q8_0.gguf
for run in plain-2k plain-32k mtp2-32k d5-32k; do
  case $run in
    *-2k)  PROMPT=$REPO/flake.nix; CTX=16384;;
    *-32k) PROMPT=$P32; CTX=40960;;
  esac
  case $run in
    plain*) SPEC=();;
    mtp2*)  SPEC=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling);;
    d5*)    SPEC=(--spec-type draft-dflash --spec-draft-n-max 5 -md "$DFL");;
  esac
  f=$OUT/$run
  timeout --kill-after=5s 420s "$B/bin/llama-cli" --offline -m "$M27" \
    -b 2048 -ub 2048 -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -c $CTX --temp 0 --top-k 1 --seed 42 \
    -f "$PROMPT" -n 512 --no-display-prompt --simple-io "${SPEC[@]}" > "$f.txt" 2> "$f.err" < /dev/null
  echo "$run rc=$? $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ')"
done
