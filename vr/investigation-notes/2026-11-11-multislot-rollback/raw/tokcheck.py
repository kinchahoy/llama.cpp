# usage: python3 -I tokcheck.py PORT PROMPT_DIR N_PREDICT OUT.json [N_SLOTS]
# Greedy token ids for a fixed schedule of slot / concurrency cases; compare two OUT.json files with tokdiff.py.
import glob, json, sys, threading, urllib.request
port, d, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
P = [open(f).read() for f in sorted(glob.glob(d + '/p*.txt'))[:3]]
NS = int(sys.argv[5]) if len(sys.argv) > 5 else 2


def req(prompt, slot, cache=False, n_cmpl=1):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 42,
            "cache_prompt": cache, "ignore_eos": True, "return_tokens": True, "id_slot": slot if NS > 1 else 0}
    if n_cmpl > 1:
        body["n_cmpl"] = n_cmpl
    r = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(r, timeout=1800))
    if isinstance(j, list):
        return [x["tokens"] for x in j]
    return j["tokens"]


def conc(jobs):
    res = [None] * len(jobs)

    def run(i):
        res[i] = req(*jobs[i])
    th = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
    [t.start() for t in th]
    [t.join() for t in th]
    return res


R = {}
R["serial_p0_s0"] = req(P[0], 0)
R["serial_p1_s1"] = req(P[1], 1)
R["serial_p2_s0_reuse"] = req(P[2], 0)
R["serial_p0_s1_swap"] = req(P[0], 1)
R["conc_p0s0_p1s1"] = conc([(P[0], 0), (P[1], 1)])
R["conc_p1s0_p0s1_swap"] = conc([(P[1], 0), (P[0], 1)])
R["conc_p2s0_p2s1_same"] = conc([(P[2], 0), (P[2], 1)])
# prompt cache: the same prompt again on slot 0, then with a suffix (cache on)
R["cache_p1_s0_a"] = req(P[1], 0, True)
R["cache_p1_s0_b"] = req(P[1], 0, True)
R["cache_p1x_s0_c"] = req(P[1] + "\n\nSummarize the above in one line.", 0, True)
# n_cmpl=2: the parent prompt state is copied to the child slot (seq_cp)
if NS > 1:
    R["ncmpl2_p0"] = req(P[0], -1, False, 2)
json.dump(R, open(out, "w"))
print("wrote", out)
