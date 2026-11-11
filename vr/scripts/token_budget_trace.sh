#!/bin/bash
# usage: token_budget_trace.sh OUT_DIR BUILD_DIR N_GEN|pN_PROMPT DEPTH -- MODEL_AND_BENCH_ARGS...
# One TOKEN-BUDGET datapoint: llama-bench unprofiled (-r 3), then the same test once under rocprofv3 --kernel-trace.
# N_GEN (e.g. 64) runs TG (-p 0 -n N); pN_PROMPT (e.g. p2048) runs PP (-p N -n 0) and also traces memory copies.
# Writes OUT_DIR/plain.json, OUT_DIR/prof.json and OUT_DIR/trace/t_kernel_trace.csv; then run token_budget.py on them.
# Example (27B at depth 0):
#   token_budget_trace.sh /tmp/tb-27b build-ilp 64 0 -- -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
#     -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on -b 2048 -ub 2048 -t 20
set +e
O=$1; B=$2; N=$3; D=$4
shift 4
[ "$1" = "--" ] && shift
P=${ROCPROFV3:-/home/raistlin/amd-clean/therock-venv/.venv/bin/rocprofv3}
LOCK=${GPU_LOCK:-/tmp/gpu.lock}
if [ "${N#p}" != "$N" ]; then
  T=(-p "${N#p}" -n 0); CP=(--memory-copy-trace)
else
  T=(-p 0 -n "$N"); CP=()
fi
mkdir -p "$O"
source /home/raistlin/amd-clean/env/therock.sh
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE

flock "$LOCK" timeout --kill-after=5s 900s "$B/bin/llama-bench" "$@" "${T[@]}" -d "$D" -r 3 -o json \
  > "$O/plain.json" 2> "$O/plain.log"
echo "plain rc=$?"
flock "$LOCK" timeout --kill-after=5s 1200s "$P" --kernel-trace "${CP[@]}" --output-format csv -d "$O/trace" -o t -- \
  "$B/bin/llama-bench" "$@" "${T[@]}" -d "$D" -r 1 -o json > "$O/prof.json" 2> "$O/prof.log"
echo "prof rc=$?"
find "$O/trace" -name 't_kernel_trace.csv'
