# usage: python3 -I tiecheck.py PORT PROMPT_DIR A.json B.json
# For each case where A and B diverge, ask the (no-draft) server for the top tokens at the divergence point,
# given the prompt plus the common prefix. A floating-point near-tie shows both tokens on top with close probability;
# a broken state would pick an unlikely token.
import glob, json, math, sys, urllib.request
port, d = sys.argv[1], sys.argv[2]
a, b = json.load(open(sys.argv[3])), json.load(open(sys.argv[4]))
P = [open(f).read() for f in sorted(glob.glob(d + '/p*.txt'))[:3]]
SUFFIX = "\n\nSummarize the above in one line."
# prompt of each case (for concurrent cases: per entry)
CASE_P = {"serial_p0_s0": [0], "serial_p1_s1": [1], "serial_p2_s0_reuse": [2], "serial_p0_s1_swap": [0],
          "conc_p0s0_p1s1": [0, 1], "conc_p1s0_p0s1_swap": [1, 0], "conc_p2s0_p2s1_same": [2, 2],
          "cache_p1_s0_a": [1], "cache_p1_s0_b": [1], "cache_p1x_s0_c": [3], "ncmpl2_p0": [0, 0]}


def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))


def top(prompt_text, prefix):
    toks = post("/tokenize", {"content": prompt_text, "add_special": True})["tokens"]
    j = post("/completion", {"prompt": toks + prefix, "n_predict": 1, "temperature": 0, "top_k": 1,
                             "n_probs": 5, "cache_prompt": False, "return_tokens": True})
    return {t["id"]: math.exp(t["logprob"]) for t in j["completion_probabilities"][0]["top_logprobs"]}


for k, ids in CASE_P.items():
    if k not in a or k not in b:
        continue
    xa, xb = a[k], b[k]
    if isinstance(xa[0], int):
        xa, xb = [xa], [xb]
    for e, (u, v, pi) in enumerate(zip(xa, xb, ids)):
        i = next((i for i, (s, t) in enumerate(zip(u, v)) if s != t), None)
        if i is None:
            print("%-22s[%d] same" % (k, e))
            continue
        text = P[1] + SUFFIX if pi == 3 else P[pi]
        pr = top(text, u[:i])
        print("%-22s[%d] diff@%d  A tok %d p=%.3f  B tok %d p=%.3f  top=%s" % (
            k, e, i, u[i], pr.get(u[i], 0), v[i], pr.get(v[i], 0),
            ", ".join("%d:%.3f" % kv for kv in sorted(pr.items(), key=lambda kv: -kv[1])[:3])))
