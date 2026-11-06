#!/usr/bin/env bash
# usage: bench2.sh <base> <cand>  matched llama-bench from vr/how-to-catchup.md
# The baseline RUNPATH points at a removed directory, so it gets LD_LIBRARY_PATH=<base>/bin (its own libraries).
BASE="$1"
CAND="$2"
OUT="$CAND/validation"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP

for label in baseline candidate; do
  if [ "$label" = baseline ]; then BUILD="$BASE"; LDP="$BASE/bin"; else BUILD="$CAND"; LDP=""; fi
  LD_LIBRARY_PATH="$LDP${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$BUILD/bin/llama-bench" > "$OUT/$label-libraries-run.txt"
  LD_LIBRARY_PATH="$LDP${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" timeout --kill-after=5s 110s "$BUILD/bin/llama-bench" \
    -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
    -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on \
    -b 2048 -ub 2048 -t 20 -p 2048 -n 128 -d 0 -r 1 -o json \
    > "$OUT/$label.json" 2> "$OUT/$label.log"
  rc=$?
  printf '%s\n' "$rc" > "$OUT/$label.exit"
  echo "$label EXIT=$rc"
  [ $rc -eq 0 ] || { tail -5 "$OUT/$label.log"; exit $rc; }
done

python3 -I - "$OUT/baseline.json" "$OUT/candidate.json" <<'PY'
import json, sys
r = {}
for lab, p in zip(("baseline", "candidate"), sys.argv[1:]):
    for row in json.load(open(p)):
        k = "PP2048" if row["n_prompt"] else "TG128"
        r.setdefault(k, {})[lab] = row["avg_ts"]
for k, v in r.items():
    print(f"{k}: baseline {v['baseline']:.2f} candidate {v['candidate']:.2f} change {(v['candidate']/v['baseline']-1)*100:+.2f}%")
PY
