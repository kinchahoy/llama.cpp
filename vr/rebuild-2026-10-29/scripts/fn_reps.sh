#!/usr/bin/env bash
# usage: fn_reps.sh LABEL BUILD_DIR OUT_DIR  - Flash Next only, 5 repeats
set -u
LABEL=$1; BUILD=$2; OUT=$3
MFN=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
unset GGML_CUDA_TP_OVERLAP
timeout --kill-after=5s 600s "$BUILD/bin/llama-bench" -m "$MFN" -ngl 999 -sm tensor -dev ROCm0/ROCm1 -ts 1/1 -fa on \
    -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 -t 8 -p 2048 -n 128 -r 5 -o json > "$OUT/$LABEL-fn5.json" 2> "$OUT/$LABEL-fn5.stderr"
python3 -I -c "
import json,sys
for r in json.load(open(sys.argv[1])): print(sys.argv[2], 'fn5', ('PP%d'%r['n_prompt']) if r['n_prompt'] else ('TG%d'%r['n_gen']), '%.2f +- %.2f'%(r['avg_ts'],r['stddev_ts']))
" "$OUT/$LABEL-fn5.json" "$LABEL"
