# usage: python3 -I longfill.py PORT TARGET_TOKENS  - one request filling the pool, then a short answer
import glob, json, sys, time, urllib.request
port, target = sys.argv[1], int(sys.argv[2])
def post(path, obj, to=7200):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=to))
files = sorted(glob.glob('/home/raistlin/infer/vrversion-llama.cpp/src/*.cpp')) + sorted(glob.glob('/home/raistlin/infer/vrversion-llama.cpp/tools/server/*.cpp'))
text, n = "", 0
for f in files:
    text += "\n// FILE: " + f.split('/')[-1] + "\n" + open(f, errors='ignore').read()
    if len(text) > target * 2.6:
        n = len(post("/tokenize", {"content": text})["tokens"])
        if n >= target: break
toks = post("/tokenize", {"content": text})["tokens"][:target]
q = post("/tokenize", {"content": "\n\nQuestion: name three of the source files above (they start with // FILE:) and say in one sentence what each does.\nAnswer:"})["tokens"]
t0 = time.time()
res = post("/completion", {"prompt": toks + q, "n_predict": 64, "temperature": 0, "top_k": 1, "cache_prompt": False, "ignore_eos": True, "n_probs": 3})
t = res["timings"]
print("LONGFILL prompt %d tok, pp %.1f t/s, tg %.1f t/s, wall %.0fs" % (t["prompt_n"], t["prompt_per_second"], t["predicted_per_second"], time.time() - t0))
print("ANSWER:", res["content"][:400].replace("\n", " "))
pr = res.get("completion_probabilities", [])[:6]
for p in pr:
    print("  TOP:", [(c.get("token"), round(c.get("prob", c.get("logprob", 0)), 3)) for c in p.get("top_probs", p.get("probs", []))])
