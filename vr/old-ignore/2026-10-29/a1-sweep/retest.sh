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
      "${extra[@]}" "$@" -p 0 -n 128 -r 2 -o json > "$R/$label.json" 2> "$R/$label.stderr"
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

# alternating retest: baseline, candidate, baseline, candidate
run r2-b27-1 27 ""
run r2-c27-ts052 27 "" -ts 0.52/0.48
run r2-b27-2 27 ""
run r2-c27-ts052b 27 "" -ts 0.52/0.48
run r2-bfn-1 fn ""
run r2-cfn-ts048 fn "" -ts 0.48/0.52
run r2-bfn-2 fn ""
run r2-cfn-ts048b fn "" -ts 0.48/0.52
echo "DONE2" >> "$R/summary.md"
