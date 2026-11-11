# usage: python3 -I spec.py PORT PROMPT_DIR N_PREDICT OUTPREFIX
# like vr/scripts/spec_accept.py, plus prompt ms and the sha of each reply
import glob, hashlib, json, sys, urllib.request

port, d, n, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rows = []
for f in sorted(glob.glob(d + '/p*.txt')):
    body = json.dumps({"prompt": open(f).read(), "n_predict": n, "temperature": 0, "top_k": 1, "seed": 42,
                       "cache_prompt": False, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    t = r["timings"]
    name = f.split('/')[-1]
    open(outp + "-" + name, "w").write(r["content"])
    dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
    steps = max(1, t["predicted_n"] - da)
    sha = hashlib.sha256(r["content"].encode()).hexdigest()[:12]
    rows.append((name, t["predicted_per_second"], da / dn if dn else 0.0, t["predicted_ms"] / steps, t["prompt_per_second"], t["prompt_ms"], sha))
for r in rows:
    print("  %s tg %.1f acc %.3f ms/step %.2f pp %.0f prompt_ms %.0f sha %s" % r)
k = len(rows)
print("MEAN tg %.2f acc %.3f ms/step %.2f prompt_ms %.0f" % (sum(r[1] for r in rows) / k, sum(r[2] for r in rows) / k, sum(r[3] for r in rows) / k, sum(r[5] for r in rows) / k))
