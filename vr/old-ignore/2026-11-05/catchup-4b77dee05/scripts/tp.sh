#!/usr/bin/env bash
# usage: tp.sh <base> <cand>
BASE="$1"
CAND="$2"
OUT="$CAND/validation"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
cp "$CAND/CMakeCache.txt" "$OUT/CMakeCache.txt"
sha256sum "$BASE/bin/llama-bench" "$CAND/bin/llama-bench" > "$OUT/binaries.sha256"
ldd "$BASE/bin/llama-bench" > "$OUT/baseline-libraries.txt"
ldd "$CAND/bin/llama-bench" > "$OUT/candidate-libraries.txt"
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP
timeout --kill-after=5s 110s bash -eu -c '
  "$1/bin/vr-test-gfx906-tp" --n 1 --check
  "$1/bin/vr-test-gfx906-tp" --n 2048 --check
' bash "$CAND" > "$OUT/tp.log" 2>&1
rc=$?
echo "TP_EXIT=$rc"
grep -E 'check=|n=|error|fail' "$OUT/tp.log" | head -20
