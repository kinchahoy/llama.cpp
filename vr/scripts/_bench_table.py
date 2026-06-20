#!/usr/bin/env python3
"""Parse llama-bench A/B jsonl under <OUT>/<build>/<config>.jsonl into a
control-vs-candidate t/s table. Used by watch-bench.sh; safe to run directly:
    python3 vr/scripts/_bench_table.py <OUT_DIR>
"""
import json, glob, os, sys

out = sys.argv[1] if len(sys.argv) > 1 else "."
data = {}    # (config, test) -> {build: (ts, sd)}
order = []   # first-seen order of (config, test)

for f in sorted(glob.glob(os.path.join(out, "*", "*.jsonl"))):
    build = os.path.basename(os.path.dirname(f))
    config = os.path.basename(f)[:-6]
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        np_, ng, nd = d.get("n_prompt", 0), d.get("n_gen", 0), d.get("n_depth", 0)
        if ng and not np_:
            test = f"tg{ng}"
        elif np_ and not ng:
            test = f"pp{np_}"
        else:
            test = f"pp{np_}+tg{ng}"
        if nd:
            test += f"@d{nd}"
        key = (config, test)
        if key not in data:
            data[key] = {}
            order.append(key)
        data[key][build] = (d.get("avg_ts"), d.get("stddev_ts"))

if not order:
    print("  (no results yet)")
    sys.exit(0)

print(f"  {'config':16} {'test':20} {'control':>14} {'candidate':>14} {'delta%':>7}")
print("  " + "-" * 76)
for cfg, test in order:
    bv = data[(cfg, test)]
    c = bv.get("control")
    k = bv.get("candidate")
    cs = f"{c[0]:8.2f}+-{c[1]:.1f}" if c else "-"
    ks = f"{k[0]:8.2f}+-{k[1]:.1f}" if k else "-"
    dl = f"{(k[0]-c[0])/c[0]*100:+6.1f}" if (c and k and c[0]) else "-"
    print(f"  {cfg:16} {test:20} {cs:>14} {ks:>14} {dl:>7}")
