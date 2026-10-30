#!/usr/bin/env bash
# usage: sweep.sh  -- runs the TG sweep, one locked llama-bench call per run
set -u
R=/home/raistlin/infer/a1-results
BIN=/home/raistlin/infer/mx-final/build-rb/bin/llama-bench
LOCK=/home/raistlin/infer/gpu.lock
HUB=/home/raistlin/.cache/huggingface/hub
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
MFN=$HUB/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
unset GGML_CUDA_TP_OVERLAP
mkdir -p "$R"
[ -f "$R/summary.md" ] || echo "# A1 TG sweep summary (TG128, -r 1, -sm tensor -dev ROCm0/ROCm1 -fa on)" > "$R/summary.md"

# run LABEL MODEL(27|fn) "ENV=V ..." ARGS...
run() {
  local label=$1 model=$2 envs=$3; shift 3
  local mpath=$M27 extra=(-ts 1/1 -b 2048 -ub 2048 -t 20)
  if [ "$model" = fn ]; then mpath=$MFN; extra=(-ts 1/1 -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 -t 8); fi
  # command-line args given by caller override the defaults (last flag wins in llama-bench)
  flock -w 3600 "$LOCK" env $envs "$BIN" -m "$mpath" -ngl 999 -sm tensor -dev ROCm0/ROCm1 -fa on \
      "${extra[@]}" "$@" -p 0 -n 128 -r 1 -o json > "$R/$label.json" 2> "$R/$label.stderr"
  local rc=$?
  python3 -I - "$R/$label.json" "$label" "$model" "$envs" "$*" "$rc" >> "$R/summary.md" <<'PY'
import json, sys
path, label, model, envs, args, rc = sys.argv[1:7]
try:
    r = [x for x in json.load(open(path)) if x["n_gen"]][0]
    res = f"{r['avg_ts']:.2f} +- {r['stddev_ts']:.2f}"
except Exception:
    res = f"FAILED (rc={rc})"
print(f"- {label} | {model} | env[{envs or '-'}] | args[{args or 'baseline'}] | TG128 {res} t/s")
PY
}

for m in 27 fn; do
  if [ "$m" = 27 ]; then
    run b27-1 27 ""
    run c27-ts052 27 "" -ts 0.52/0.48
    run b27-2 27 ""
    run c27-ts048 27 "" -ts 0.48/0.52
    run b27-3 27 ""
    run c27-ts056 27 "" -ts 0.56/0.44
    run b27-4 27 ""
    run c27-t16 27 "" -t 16
    run b27-5 27 ""
    run c27-t8 27 "" -t 8
    run b27-6 27 ""
    run c27-slabs0 27 "GGML_CUDA_AR_SLABS=0"
    run b27-7 27 ""
    run c27-share0 27 "GGML_SCHED_SHARE_VIEW_INPUTS=0"
    run b27-8 27 ""
    run c27-pd0 27 "GGML_META_PARALLEL_DISPATCH=0"
    run b27-9 27 ""
  else
    run bfn-1 fn ""
    run cfn-ts052 fn "" -ts 0.52/0.48
    run bfn-2 fn ""
    run cfn-ts048 fn "" -ts 0.48/0.52
    run bfn-3 fn ""
    run cfn-t16 fn "" -t 16
    run bfn-4 fn ""
    run cfn-slabs0 fn "GGML_CUDA_AR_SLABS=0"
    run bfn-5 fn ""
    run cfn-shexp0 fn "LLAMA_SHEXP_SPLIT=0"
    run bfn-6 fn ""
    run cfn-share0 fn "GGML_SCHED_SHARE_VIEW_INPUTS=0"
    run bfn-7 fn ""
    run cfn-pd0 fn "GGML_META_PARALLEL_DISPATCH=0"
    run bfn-8 fn ""
  fi
done
echo "DONE" >> "$R/summary.md"
