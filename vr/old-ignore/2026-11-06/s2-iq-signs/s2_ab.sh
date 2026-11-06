#!/usr/bin/env bash
# usage: s2_ab.sh OUT BASE CAND   Flash Next UD-IQ3_XXS (rev c8b5954a88c2): greedy output once per build, then PP2048/TG128 r5 alternating
OUT=$1; BASE=$2; CAND=$3
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
MFN=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
FN=(-m "$MFN" -ngl 999 -sm tensor -fa on -ctk q8_0 -ctv q8_0)

for label in base cand; do
  if [ $label = base ]; then B=$BASE; else B=$CAND; fi
  timeout --kill-after=5s 300s "$B/bin/llama-completion" "${FN[@]}" -dev ROCm0,ROCm1 -ts 1,1 -b 1024 -ub 512 -c 4096 -t 20 \
    --temp 0 --top-k 1 -s 42 -n 96 -no-cnv --simple-io --no-display-prompt \
    -p 'The history of the printing press begins in' < /dev/null > "$OUT/$label-gen.txt" 2> "$OUT/$label-gen.log"
  echo "$label gen rc=$? sha=$(sha256sum < "$OUT/$label-gen.txt" | cut -c1-12) bytes=$(wc -c < "$OUT/$label-gen.txt")"
done

for r in 1 2 3 4 5; do
  for label in base cand; do
    if [ $label = base ]; then B=$BASE; else B=$CAND; fi
    timeout --kill-after=5s 300s "$B/bin/llama-bench" "${FN[@]}" -dev ROCm0/ROCm1 -ts 1/1 -b 1024 -ub 512 -t 20 \
      -p 2048 -n 128 -r 1 -o json > "$OUT/$label-r$r.json" 2> "$OUT/$label-r$r.log"
    echo "$label r$r rc=$?"
  done
done

python3 -I - "$OUT" <<'PY'
import json, sys, glob, statistics as st
out = sys.argv[1]
res = {}
for lab in ("base", "cand"):
    for f in sorted(glob.glob(f"{out}/{lab}-r*.json")):
        for row in json.load(open(f)):
            k = "PP2048" if row["n_prompt"] else "TG128"
            res.setdefault((lab, k), []).append(row["avg_ts"])
for k in ("PP2048", "TG128"):
    b, c = res.get(("base", k), []), res.get(("cand", k), [])
    if b and c:
        mb, mc = st.mean(b), st.mean(c)
        print(f"{k}: base {mb:.2f} +- {st.pstdev(b):.2f} {['%.1f' % x for x in b]} | cand {mc:.2f} +- {st.pstdev(c):.2f} {['%.1f' % x for x in c]} | change {(mc/mb-1)*100:+.2f}%")
PY
