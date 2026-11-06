#!/usr/bin/env bash
# usage: spec_check.sh OUT_DIR BASE CAND   27B Q8_0 MTP2 and DFlash5, greedy, flake.nix prompt, 512 tokens (from vr/old-ignore/2026-10-29/n1-dflash/run.sh)
OUT=$1; BASE=$2; CAND=$3
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP
HUB=/home/raistlin/.cache/huggingface/hub
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
DFL=$HUB/models--z-lab--Qwen3.8-27B-DFlash2-GGUF/snapshots/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q8_0.gguf
PROMPT=/home/raistlin/infer/vrversion-llama.cpp/flake.nix
for mode in mtp2 d5; do
  case $mode in
    mtp2) SPEC=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling);;
    d5)   SPEC=(--spec-type draft-dflash --spec-draft-n-max 5 -md "$DFL");;
  esac
  for label in base cand; do
    if [ $label = base ]; then B=$BASE; LDP=$BASE/bin; else B=$CAND; LDP=""; fi
    f=$OUT/$label-$mode
    LD_LIBRARY_PATH="$LDP${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" timeout --kill-after=5s 420s "$B/bin/llama-cli" --offline -m "$M27" \
      -b 2048 -ub 2048 -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -c 16384 --temp 0 --top-k 1 --seed 42 \
      -f "$PROMPT" -n 512 --no-display-prompt --simple-io "${SPEC[@]}" > "$f.txt" 2> "$f.err" < /dev/null
    rc=$?
    echo "$label $mode rc=$rc $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') sha=$(grep -v '^\[ Prompt' "$f.txt" | sha256sum | cut -c1-12) bytes=$(grep -v '^\[ Prompt' "$f.txt" | wc -c)"
    grep -iE 'accept|n_drafted|draft.*rate' "$f.err" | tail -3
  done
done
