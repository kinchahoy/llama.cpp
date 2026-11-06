#!/usr/bin/env bash
# usage: conv_test.sh BUILD OUT   repacked MoE vs CPU (IQ4_NL, Q5_0, IQ4_XS) and the S2 test-backend-ops subset
B=$1; OUT=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
timeout 1200s "$B/bin/vr-test-flashnext-moe" > "$OUT/repack-moe.log" 2>&1
echo "repack-moe rc=$?  cases=$(grep -c nmse= "$OUT/repack-moe.log")"
grep -oE 'repacked [a-z0-9_]+' "$OUT/repack-moe.log" | sort | uniq -c
python3 -I - "$OUT/repack-moe.log" <<'PY'
import re, sys, collections
worst = collections.defaultdict(float)
for l in open(sys.argv[1]):
    m = re.search(r'repacked (\S+) .* nmse=([0-9.eE+-]+)', l)
    if m:
        worst[m.group(1)] = max(worst[m.group(1)], float(m.group(2)))
for t, v in worst.items():
    print(f"  worst nmse {t}: {v:.3g}")
PY
tail -2 "$OUT/repack-moe.log"
timeout 900s "$B/bin/test-backend-ops" test -b ROCm0 -o MUL_MAT,MUL_MAT_ID -p 'type_a=(iq2_s|iq3_xxs|q5_0|iq4_xs)' > "$OUT/test-backend-ops.log" 2>&1
echo "test-backend-ops rc=$? ok=$(grep MUL_MAT "$OUT/test-backend-ops.log" | grep -c OK) fail=$(grep MUL_MAT "$OUT/test-backend-ops.log" | grep -c FAIL)"
