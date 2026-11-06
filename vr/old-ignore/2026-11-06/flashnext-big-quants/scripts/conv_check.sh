#!/usr/bin/env bash
# usage: conv_eval.sh OUT BUILD
# 1) GGML_CUDA_REPACK_CONVERT=0 vs 1: PP2048/TG128 at depth 8192, 2 alternating rounds of -r 2, UD-Q3_K_XL and UD-IQ4_XS (-ncmoe 3)
# 2) perplexity, conversion off vs on (tools/server/README.md, 8 chunks of 2048)
# 3) UD-Q3_K_XL with the Q8_0 n-gram table (conversion on): speed and perplexity
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
SNAP=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c
Q8E=/home/raistlin/models/flashnext
TEXT=/home/raistlin/infer/vrversion-llama.cpp/tools/server/README.md
FN=(-ngl 999 -sm tensor -fa on -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 -t 20)

model() { echo "$SNAP/$1/Qwen3.8-Flash-Next-$1-00001-of-00004.gguf"; }
extra() { [ "$1" = UD-IQ4_XS ] && echo "-ncmoe 3"; }

bench() {  # label model conv extra...
  local label=$1 m=$2 conv=$3; shift 3
  GGML_CUDA_REPACK_CONVERT=$conv timeout --kill-after=5s 900s "$B/bin/llama-bench" -m "$m" "${FN[@]}" -dev ROCm0/ROCm1 -ts 1/1 \
    -p 2048 -n 128 -d 8192 -r 2 -o json "$@" > "$OUT/$label.json" 2> "$OUT/$label.log"
  echo "$label rc=$?"
}
ppl() {  # label model conv extra...
  local label=$1 m=$2 conv=$3; shift 3
  GGML_CUDA_REPACK_CONVERT=$conv timeout --kill-after=5s 900s "$B/bin/llama-perplexity" -m "$m" "${FN[@]}" -dev ROCm0,ROCm1 -ts 1,1 \
    -f "$TEXT" -c 2048 --chunks 8 "$@" > "$OUT/$label.ppl.txt" 2>&1
  echo "$label ppl rc=$? $(grep -oE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' "$OUT/$label.ppl.txt")"
}

for q in UD-Q3_K_XL UD-IQ4_XS; do
  for r in 1 2; do
    for conv in 0 1; do
      bench "$q-conv$conv-r$r" "$(model $q)" $conv $(extra $q)
    done
  done
  for conv in 0 1; do
    ppl "$q-conv$conv" "$(model $q)" $conv $(extra $q)
  done
done

Q3E=$Q8E/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf
if [ -f "$Q3E" ]; then
  for r in 1 2; do
    bench "q3kxl-q8engram-r$r" "$Q3E" 1
  done
  ppl "q3kxl-q8engram" "$Q3E" 1
else
  echo "q8 engram set missing: $Q3E"
fi

python3 -I - "$OUT" <<'PY'
import json, sys, glob, os, re, statistics as st, collections
res = collections.defaultdict(list)
for f in glob.glob(sys.argv[1] + "/*.json"):
    label = re.sub(r'-r\d+$', '', os.path.basename(f)[:-5])
    try:
        rows = json.load(open(f))
    except Exception:
        continue
    for r in rows:
        k = "PP2048" if r["n_prompt"] else "TG128"
        res[(label, k)] += r["samples_ts"]
for (label, k), v in sorted(res.items()):
    print(f"{label:24s} {k:7s} @d8192: {st.mean(v):8.2f} +- {st.pstdev(v):.2f}  n={len(v)}")
PY
