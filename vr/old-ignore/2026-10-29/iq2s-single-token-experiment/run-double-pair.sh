#!/bin/bash
set -u
cd /home/raistlin/infer/mx-llama.cpp
source /home/raistlin/amd-clean/env/therock.sh
set +e
record=$PWD/vr/old-ignore/2026-10-29/iq2s-single-token-experiment
candidate=$PWD/.worktrees/experiment/catchup-418f905a4/build-catchup/bin
model=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
export GGML_CUDA_FLASHNEXT_MOE=1
unset GGML_CUDA_TP_OVERLAP LLAMA_QWEN4_MTP_DEVICE
base_library_path=$LD_LIBRARY_PATH
rg '^Cpus_allowed_list:' /proc/self/status > "$record/cpus-allowed.txt"
env | sort | rg '^(GGML|LLAMA|HIP|HSA|ROCR|LD_LIBRARY_PATH|ROCBLAS|CMAKE_HIP)' > "$record/environment.txt"
printf "" > "$record/pair-status.txt"
for variant in baseline candidate; do
  if [[ $variant == baseline ]]; then bin=$record/baseline-bin; else bin=$candidate; fi
  export LD_LIBRARY_PATH="$bin:$base_library_path"
  sha256sum "$bin/llama-cli" "$bin/libggml-hip.so.0.28.0" >> "$record/binaries.sha256"
  ldd "$bin/llama-cli" > "$record/$variant.ldd"
  args=("$bin/llama-cli" -m "$model" --offline -lm dio -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 1,1 -fa on -b 1024 -ub 512 -c 100096 -ctk q8_0 -ctv q8_0 -t 20 -tb 20 --temp 0 --top-k 1 --seed 42 -f "$record/prompt-double.txt" -n 256 -st --no-display-prompt --simple-io --log-verbosity 3)
  printf '%q ' "${args[@]}" > "$record/$variant.command"
  printf '\n' >> "$record/$variant.command"
  /usr/bin/time -f '%e seconds, exit %x, CPU %P, user %U s, system %S s' -o "$record/$variant.time" timeout --kill-after=1s 60s "${args[@]}" > "$record/$variant.stdout" 2> "$record/$variant.stderr"
  printf '%s exit=%s\n' "$variant" "$?" >> "$record/pair-status.txt"
done
