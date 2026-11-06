#!/usr/bin/env bash
# usage: cpy_ab.sh OUT BUILD   GGML_CUDA_CPY_2D=1 (old) vs default (copy kernels): CPY/CONT op tests, llama-bench @8K r2 x2 alternating, 100K MTP3 + grouped MoE
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE GGML_CUDA_CPY_2D
M=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
DQ8=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
PROMPT=/home/raistlin/infer/vrversion-llama.cpp/flake.nix
SHA=/home/raistlin/infer/vrversion-llama.cpp/vr/old-ignore/2026-11-05/catchup-4b77dee05/scripts/reply_sha.py

timeout 900s "$B/bin/test-backend-ops" test -b ROCm0 -o CPY,CONT,DUP > "$OUT/ops.log" 2>&1
echo "ops rc=$? ok=$(grep -cE '\bOK|OK\b' "$OUT/ops.log") fail=$(grep -c FAIL "$OUT/ops.log")"

for r in 1 2; do
  for v in 1 0; do
    GGML_CUDA_CPY_2D=$v timeout 900s "$B/bin/llama-bench" -m "$M" -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on \
      -b 1024 -ub 512 -t 20 -p 2048 -n 128 -d 8192 -r 2 -o json > "$OUT/bench-cpy2d$v-r$r.json" 2> "$OUT/bench-cpy2d$v-r$r.log"
    echo "bench cpy2d=$v r$r rc=$?"
  done
done
python3 -I - "$OUT" <<'PY'
import json, sys, glob, re, os, statistics as st, collections
res = collections.defaultdict(list)
for f in glob.glob(sys.argv[1] + "/bench-*.json"):
    lab = re.sub(r'-r\d+$', '', os.path.basename(f)[:-5])
    for r in json.load(open(f)):
        res[(lab, "PP2048" if r["n_prompt"] else "TG128")] += r["samples_ts"]
for (l, k), v in sorted(res.items()):
    print(f"  {l:14s} {k:7s} @d8192: {st.mean(v):8.2f} +- {st.pstdev(v):.2f} n={len(v)}")
PY

for v in 1 0; do
  f=$OUT/mtp3-cpy2d$v
  GGML_CUDA_CPY_2D=$v GGML_CUDA_FLASHNEXT_MOE=1 timeout --kill-after=5s 600s "$B/bin/llama-cli" --offline -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 0.52,0.48 -fa on \
    -b 1024 -ub 512 -c 100352 -t 20 --temp 0 --top-k 1 --seed 42 -f "$PROMPT" -n 512 --no-display-prompt --simple-io \
    -md "$DQ8" -devd ROCm1 --spec-draft-n-ubatch 64 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling \
    > "$f.txt" 2> "$f.err" < /dev/null
  echo "mtp3 cpy2d=$v rc=$? $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') $(python3 -I "$SHA" "$f.txt" 2>/dev/null | cut -d' ' -f1-3)"
done
