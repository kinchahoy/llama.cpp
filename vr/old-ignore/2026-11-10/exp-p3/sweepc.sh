#!/bin/bash
# usage: sweepc.sh TAG TS...; waits for both GPUs to cool below 55 C (edge) before each run
source /home/raistlin/amd-clean/env/therock.sh
B=${BUILD:-/home/raistlin/infer/vrversion-llama.cpp/build-catchup-20261110-151625}
OUT=/home/raistlin/.claude/jobs/5a8c59bb/tmp/p3/sweep
mkdir -p $OUT
M=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
temps() { rocm-smi --showtemp 2>/dev/null | grep edge | awk '{print $NF}' | tr '\n' ' '; }
TAG=$1; shift
for ts in "$@"; do
  for i in $(seq 1 60); do
    hot=$(temps | awk '{for(i=1;i<=NF;i++) if ($i+0 >= 55) h=1; print h+0}')
    [ "$hot" = "0" ] && break
    sleep 3
  done
  t0=$(temps)
  n=$(echo $ts | tr '/' '_')
  timeout --kill-after=5s 240s "$B/bin/llama-bench" -m $M -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts $ts -fa on -b 2048 -ub 2048 -t 20 -p 2048 -n 128 -r 3 -o json > $OUT/$TAG-$n.json 2> $OUT/$TAG-$n.err
  rc=$?
  echo "$ts rc=$rc temp before [$t0] after [$(temps)]"
  python3 -c "
import json
for r in json.load(open('$OUT/$TAG-$n.json')):
    print('  ts=$ts', 'p', r['n_prompt'], 'n', r['n_gen'], '%.2f +- %.2f' % (r['avg_ts'], r['stddev_ts']), r['samples_ts'])
"
done
