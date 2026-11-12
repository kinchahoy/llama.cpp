# usage: python3 -I conclong.py PORT TOKENS_EACH - two concurrent long requests with different content, then a short answer each
import glob, json, sys, threading, time, urllib.request
port, each = sys.argv[1], int(sys.argv[2])
R = '/home/raistlin/infer/vrversion-llama.cpp/'
def post(path, obj, to=7200):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=to))
def build(files):
    text = ""
    for f in files:
        text += "\n// FILE: " + f.split('/')[-1] + "\n" + open(f, errors='ignore').read()
        if len(text) > each * 4:
            break
    toks = post("/tokenize", {"content": text})["tokens"][:each]
    q = post("/tokenize", {"content": "\n\nQuestion: name three of the source files above (they start with // FILE:) and say in one sentence what each does.\nAnswer:"})["tokens"]
    return toks + q
sets = [sorted(glob.glob(R + 'src/*.cpp')), sorted(glob.glob(R + 'ggml/src/ggml-cuda/*.cu')) + sorted(glob.glob(R + 'common/*.cpp'))]
prompts = [build(s) for s in sets]
res = [None, None]
def run(i):
    try:
        res[i] = post("/completion", {"prompt": prompts[i], "n_predict": 96, "temperature": 0, "top_k": 1, "cache_prompt": False, "ignore_eos": True})
    except Exception as e:
        res[i] = {"error": repr(e)}
t0 = time.time()
th = [threading.Thread(target=run, args=(i,)) for i in range(2)]
[t.start() for t in th]; [t.join() for t in th]
for i, r in enumerate(res):
    if "error" in r:
        print("CONCLONG %d ERROR %s" % (i, r["error"])); continue
    t = r["timings"]
    print("CONCLONG %d prompt %d tok, pp %.1f t/s, tg %.1f t/s | %s" % (i, t["prompt_n"], t["prompt_per_second"], t["predicted_per_second"], r["content"][:200].replace("\n", " ")))
print("CONCLONG wall %.0fs" % (time.time() - t0))
print("HEALTH", urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10).read().decode())
