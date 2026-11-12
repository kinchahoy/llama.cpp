#!/usr/bin/env bash
# usage: run.sh NAME NCONC -- server args...
set -u
NAME=$1; NC=$2; shift 3
OUT=$(dirname "$0")/out; mkdir -p $OUT
REPO=/home/raistlin/infer/vrversion-llama.cpp; P=$REPO/vr/investigation-notes/2026-11-11-unified-kv-unverified/verification/prompts
[ -d $P ] || { mkdir -p $OUT/prompts; P=$OUT/prompts; head -c 6000 $REPO/flake.nix > $P/p1.txt; head -c 6000 $REPO/src/llama-context.cpp > $P/p2.txt; head -c 4000 $REPO/tools/server/README.md > $P/p3.txt; }
source /home/raistlin/amd-clean/env/therock.sh; set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
PORT=18095
$REPO/build-play-b1/bin/llama-server --offline --host 127.0.0.1 --port $PORT --metrics "$@" > $OUT/$NAME.log 2>&1 &
S=$!
ok=0; for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && { ok=1; break; }; kill -0 $S 2>/dev/null || break; sleep 2; done
echo "== $NAME"; grep -E "n_rs_seq|speculative decoding will use|snapshot ring" $OUT/$NAME.log | sed 's/^/   /' | sort -u
if [ $ok = 1 ]; then
  [ -z "${SKIP:-}" ] && python3 -I $REPO/vr/scripts/spec_accept.py $PORT $P 256
  [ -z "${SKIP:-}" ] && python3 -I $(dirname "$0")/conc.py $PORT $P 256 $NC
  [ -n "${FILL:-}" ] && python3 -I $(dirname "$0")/${LF:-longfill.py} $PORT $FILL
  rocm-smi --showmeminfo vram | grep "Used Memory" | sed 's/^/   /'
else echo "   FAILED to start:"; grep -iE "error|out of memory|failed" $OUT/$NAME.log | head -5; fi
kill $S 2>/dev/null; wait $S 2>/dev/null; sleep 3
