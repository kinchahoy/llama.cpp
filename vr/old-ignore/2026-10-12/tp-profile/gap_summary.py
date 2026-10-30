import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(list)
for r in rows:
    by[r['Agent_Id']].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name'].split('(')[0].split('<')[0].replace('void ','')[:40]))
for a, ks in by.items():
    ks.sort()
    lm = [k for k in ks if k[2].startswith('mul_mat_vec_q') and not 'repacked' in k[2]]
    t0, t1 = lm[1][1], lm[-1][1]   # steady tokens between lm_head calls
    ntok = len(lm) - 2
    ks2 = [k for k in ks if t0 <= k[0] < t1]
    gaps = collections.defaultdict(lambda: [0, 0])
    tot = 0
    for p, n in zip(ks2, ks2[1:]):
        g = n[0] - p[1]
        if g > 0:
            tot += g
            key = (p[2], n[2])
            gaps[key][0] += g; gaps[key][1] += 1
    print(f"{a}: {ntok} tokens, {(t1-t0)/ntok/1e3:.0f} us/token, idle {tot/ntok/1e3:.0f} us/token")
    for k, (g, c) in sorted(gaps.items(), key=lambda x: -x[1][0])[:8]:
        print(f"   {g/ntok/1e3:7.0f} us/tok  n/tok={c/ntok:5.1f}  avg={g/c/1e3:6.1f}us  {k[0]} -> {k[1]}")
