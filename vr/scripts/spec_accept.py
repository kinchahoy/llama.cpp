# usage: python3 -I spec_accept.py PORT PROMPT_DIR N_PREDICT
# Sends each PROMPT_DIR/p*.txt to a running llama-server (greedy, no prompt cache) and prints
# TG, draft acceptance and ms per verify step. ms/step does not depend on acceptance, so use it to A/B kernels.
import glob, json, sys, urllib.request

port, d, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
for f in sorted(glob.glob(d + '/p*.txt')):
    body = json.dumps({"prompt": open(f).read(), "n_predict": n, "temperature": 0, "top_k": 1, "seed": 42,
                       "cache_prompt": False, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=900))["timings"]
    dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
    steps = max(1, t["predicted_n"] - da)
    rows.append((f.split('/')[-1], t["predicted_per_second"], da / dn if dn else 0.0, t["predicted_ms"] / steps, t["prompt_per_second"]))
for r in rows:
    print("  %s tg %.1f acc %.3f ms/step %.2f pp %.0f" % r)
k = len(rows)
print("MEAN tg %.2f acc %.3f ms/step %.2f" % (sum(r[1] for r in rows) / k, sum(r[2] for r in rows) / k, sum(r[3] for r in rows) / k))
