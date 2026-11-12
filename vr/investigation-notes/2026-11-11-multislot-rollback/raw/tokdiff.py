# usage: python3 -I tokdiff.py A.json B.json   (first differing token index per case)
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))


def first_diff(x, y):
    for i, (u, v) in enumerate(zip(x, y)):
        if u != v:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


for k in a:
    xa, xb = a[k], b.get(k)
    if xb is None:
        print(k, "missing")
        continue
    if isinstance(xa[0], int):
        xa, xb = [xa], [xb]
    print("%-22s %s" % (k, " ".join(
        "same(%d)" % len(u) if first_diff(u, v) is None else "DIFF@%d/%d" % (first_diff(u, v), len(u))
        for u, v in zip(xa, xb))))
