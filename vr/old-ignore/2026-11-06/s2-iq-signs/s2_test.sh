#!/usr/bin/env bash
# usage: s2_test.sh BUILD OUT   test-backend-ops MUL_MAT / MUL_MAT_ID for iq2_s and iq3_xxs on ROCm0 vs CPU
B=$1; OUT=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
timeout 900s "$B/bin/test-backend-ops" test -b ROCm0 -o MUL_MAT,MUL_MAT_ID -p 'type_a=(iq2_s|iq3_xxs)' > "$OUT/test-backend-ops.log" 2>&1
echo "rc=$?"
grep -cE '\bOK\b' "$OUT/test-backend-ops.log" | sed 's/^/OK lines: /'
grep -E 'FAIL|not supported' "$OUT/test-backend-ops.log" | grep -v 'not supported' | head -5
tail -4 "$OUT/test-backend-ops.log"
