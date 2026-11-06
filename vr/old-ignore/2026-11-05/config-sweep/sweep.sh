#!/usr/bin/env bash
# usage: sweep.sh OUT BUILD   27B: speculative modes (llama-cli, standard workload) and ubatch 2048 vs 4096 at PP4096 (llama-bench r3)
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP
HUB=/home/raistlin/.cache/huggingface/hub
M27=$HUB/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
DFL=$HUB/models--z-lab--Qwen3.8-27B-DFlash2-GGUF/snapshots/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q8_0.gguf
PROMPT=/home/raistlin/infer/vrversion-llama.cpp/flake.nix
SHA=/home/raistlin/infer/vrversion-llama.cpp/vr/old-ignore/2026-11-05/catchup-4b77dee05/scripts/reply_sha.py

for mode in d7 mtp3 d5 mtp2; do
  case $mode in
    mtp2) SPEC=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling);;
    mtp3) SPEC=(--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling);;
    d5)   SPEC=(--spec-type draft-dflash --spec-draft-n-max 5 -md "$DFL");;
    d7)   SPEC=(--spec-type draft-dflash --spec-draft-n-max 7 -md "$DFL");;
  esac
  f=$OUT/$mode
  timeout --kill-after=5s 420s "$B/bin/llama-cli" --offline -m "$M27" \
    -b 2048 -ub 2048 -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -c 16384 --temp 0 --top-k 1 --seed 42 \
    -f "$PROMPT" -n 512 --no-display-prompt --simple-io "${SPEC[@]}" > "$f.txt" 2> "$f.err" < /dev/null
  echo "$mode rc=$? $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') $(python3 -I "$SHA" "$f.txt" | cut -d' ' -f1-3)"
done

timeout --kill-after=5s 300s "$B/bin/llama-bench" -m "$M27" \
  -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on -t 20 \
  -p 4096 -n 0 -b 4096 -ub 2048,4096 -r 3 -o json > "$OUT/ub.json" 2> "$OUT/ub.log"
echo "ub rc=$?"
python3 -I - "$OUT/ub.json" <<'PY'
import json, sys
for r in json.load(open(sys.argv[1])):
    print(f"PP{r['n_prompt']} ub={r['n_ubatch']}: {r['avg_ts']:.2f} +- {r['stddev_ts']:.2f} t/s  samples={['%.1f' % x for x in r['samples_ts']]}")
PY
