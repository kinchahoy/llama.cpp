#!/usr/bin/env bash
# usage: routercheck.sh WORKTREE OUTDIR - start the router presets, query qwen27b and flashnext, stop
WT=$1; T=$2
PORT=18099 HOST=127.0.0.1 "$WT/vr/server/serve.sh" > "$T/router.log" 2>&1 &
R=$!
for i in $(seq 1 200); do curl -sf localhost:18099/health >/dev/null && break; sleep 3; done
ask() {
  curl -s -m 900 localhost:18099/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":300}" |
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("model"), "|", repr(d["choices"][0]["message"].get("content",""))[:150], "|", d.get("timings",{}).get("predicted_per_second"))'
}
ask qwen27b
ask flashnext
grep -hE "n_slots = |rollback currently|speculative decoding will use| E " "$T/router.log" | sort -u | head
kill $R; wait $R 2>/dev/null; sleep 5
pgrep -af llama-server | grep -v pgrep | head -3
