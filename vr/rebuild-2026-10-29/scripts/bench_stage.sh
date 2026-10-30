#!/usr/bin/env bash
# usage: bench_stage.sh LABEL BUILD_DIR OUT_DIR
set -u
LABEL=$1; BUILD=$2; OUT=$3
mkdir -p "$OUT"
HUB=/home/raistlin/.cache/huggingface/hub
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
MFN=$HUB/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
BIN=$BUILD/bin/llama-bench
unset GGML_CUDA_TP_OVERLAP
sha256sum "$BIN" "$BUILD"/bin/libggml-hip.so* 2>/dev/null > "$OUT/$LABEL-binary.sha256"

timeout --kill-after=5s 300s "$BIN" -m "$M27" -ngl 999 -sm tensor -dev ROCm0/ROCm1 -ts 1/1 -fa on \
    -b 2048 -ub 2048 -t 20 -p 2048 -n 128 -r 2 -o json > "$OUT/$LABEL-27b.json" 2> "$OUT/$LABEL-27b.stderr"
echo "27b exit=$?"

timeout --kill-after=5s 400s "$BIN" -m "$MFN" -ngl 999 -sm tensor -dev ROCm0/ROCm1 -ts 1/1 -fa on \
    -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 -t 8 -p 2048 -n 128 -r 2 -o json > "$OUT/$LABEL-fn.json" 2> "$OUT/$LABEL-fn.stderr"
echo "fn exit=$?"

python3 -I - "$OUT" "$LABEL" <<'EOF'
import json, sys, os
out, label = sys.argv[1], sys.argv[2]
for m in ("27b", "fn"):
    p = os.path.join(out, f"{label}-{m}.json")
    try:
        rows = json.load(open(p))
    except Exception as e:
        print(f"{label} {m}: no result ({e})"); continue
    for r in rows:
        kind = "PP" if r["n_prompt"] else "TG"
        print(f"{label} {m} {kind}{r['n_prompt'] or r['n_gen']}: {r['avg_ts']:.2f} +- {r['stddev_ts']:.2f} t/s")
EOF
