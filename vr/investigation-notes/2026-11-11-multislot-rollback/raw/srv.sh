#!/usr/bin/env bash
# usage: srv.sh BUILD_DIR NAME "CLIENT CMD (use \$PORT)" -- server args...
# Starts llama-server, runs the client command, prints rollback lines, stops the server.
set -u
BUILD=$1; NAME=$2; CMD=$3; shift 4
OUT=$(cd "$(dirname "$0")" && pwd)/out; mkdir -p $OUT
source /home/raistlin/amd-clean/env/therock.sh; set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
export PORT=18096
if pgrep -x llama-server >/dev/null; then echo "llama-server already running, abort"; exit 1; fi
$BUILD/bin/llama-server --offline --host 127.0.0.1 --port $PORT --metrics "$@" > $OUT/$NAME.log 2>&1 &
S=$!
ok=0; for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && { ok=1; break; }; kill -0 $S 2>/dev/null || break; sleep 2; done
echo "== $NAME ($BUILD)"
grep -E "n_rs_seq|speculative decoding will use|snapshot ring|rs_seq\)|clamping" $OUT/$NAME.log | sed 's/^/   /' | sort -u | head -8
if [ $ok = 1 ]; then eval "$CMD"; rocm-smi --showmeminfo vram | grep "Used Memory" | sed 's/^/   /'
else echo "   FAILED to start:"; grep -iE "error|out of memory|failed" $OUT/$NAME.log | head -5; fi
grep -E "refusing rollback|invalidated|non-consecutive" $OUT/$NAME.log | sort | uniq -c | head -5 | sed 's/^/   WARN /'
kill $S 2>/dev/null; wait $S 2>/dev/null; sleep 3
