#!/usr/bin/env bash
# usage: fn8k.sh OUT BUILD   PP2048/TG128 at depth 8192, r3, plain: Flash Next UD-IQ3_XXS, UD-Q3_K_XL, UD-IQ4_XS (smallest -ncmoe that fits), 27B Q8_0 reference
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
HUB=/home/raistlin/.cache/huggingface/hub
SNAP=$HUB/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
COMMON=(-ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on -t 20 -p 2048 -n 128 -d 8192 -r 3 -o json)
FN=(-b 1024 -ub 512 -ctk q8_0 -ctv q8_0)

run() {  # label, extra args...
  local label=$1; shift
  timeout --kill-after=5s 900s "$B/bin/llama-bench" "${COMMON[@]}" "$@" > "$OUT/$label.json" 2> "$OUT/$label.log"
  local rc=$?
  echo "$label rc=$rc"
  return $rc
}

run fn-iq3xxs -m "$SNAP/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf" "${FN[@]}"
run fn-q3kxl  -m "$SNAP/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf" "${FN[@]}"
for n in 2 3 4 5 6; do
  run fn-iq4xs-ncmoe$n -m "$SNAP/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf" "${FN[@]}" -ncmoe $n && break
  grep -iE 'out of memory|failed to allocate|alloc' "$OUT/fn-iq4xs-ncmoe$n.log" | head -2
done
run 27b-q8 -m "$M27" -b 2048 -ub 2048

python3 -I - "$OUT" <<'PY'
import json, sys, glob, os
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    try:
        rows = json.load(open(f))
    except Exception:
        continue
    for r in rows:
        k = f"PP{r['n_prompt']}" if r["n_prompt"] else f"TG{r['n_gen']}"
        print(f"{os.path.basename(f)[:-5]:18s} {k:7s} @d{r['n_depth']}: {r['avg_ts']:8.2f} +- {r['stddev_ts']:.2f}")
PY
