#!/bin/bash
set -u
cd /home/raistlin/infer/mx-llama.cpp
source /home/raistlin/amd-clean/env/therock.sh
set +e
record=vr/old-ignore/2026-10-29/head-merge-validation/terra-port-fix
model=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
draft=/home/raistlin/.cache/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
export GGML_CUDA_FLASHNEXT_MOE=1 LLAMA_QWEN4_MTP_DEVICE=0
unset GGML_CUDA_TP_OVERLAP
env | sort | rg '^(GGML|LLAMA|HIP|HSA|ROCR|LD_LIBRARY_PATH)_' > "$record/environment.txt"
for variant in baseline candidate; do
  if [[ $variant == baseline ]]; then binary=$PWD/build-ilp/bin/llama-cli; else binary=$PWD/.worktrees/experiment/catchup-418f905a4/build-catchup/bin/llama-cli; fi
  sha256sum "$binary" >> "$record/binaries.sha256"
  ldd "$binary" > "$record/$variant.ldd"
  args=("$binary" -m "$model" --offline -lm dio -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 1,1 -fa on -b 1024 -ub 512 -c 100096 -ctk q8_0 -ctv q8_0 --temp 0 --top-k 1 --seed 42 -f "$PWD/flake.nix" -n 128 -st --no-display-prompt --simple-io --log-verbosity 3 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling -md "$draft" -devd ROCm1)
  printf '%q ' "${args[@]}" > "$record/$variant.command"
  printf '\n' >> "$record/$variant.command"
  /usr/bin/time -o "$record/$variant.time" -f '%e seconds, exit %x' timeout --kill-after=1s 55s "${args[@]}" > "$record/$variant.stdout" 2> "$record/$variant.stderr"
  printf '%s exit=%s\n' "$variant" "$?" >> "$record/status.txt"
done
