# usage: python3 -I conc.py PORT PROMPT_DIR N_PREDICT N_CONC
import glob, json, sys, time, threading, urllib.request
port, d, n, c = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
files = sorted(glob.glob(d + '/p*.txt'))[:3]
res = [None]*c
def run(i):
    body = json.dumps({"prompt": open(files[i % len(files)]).read(), "n_predict": n, "temperature": 0, "top_k": 1,
                       "seed": 42, "cache_prompt": False, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
    res[i] = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
t0 = time.time(); th = [threading.Thread(target=run, args=(i,)) for i in range(c)]
[x.start() for x in th]; [x.join() for x in th]; wall = time.time() - t0
tot = sum(r["predicted_n"] for r in res)
print("CONC %d: per-stream tg %s | aggregate gen %.1f t/s (incl prefill, wall %.1fs)" % (
    c, " ".join("%.1f" % r["predicted_per_second"] for r in res), tot / wall, wall))
