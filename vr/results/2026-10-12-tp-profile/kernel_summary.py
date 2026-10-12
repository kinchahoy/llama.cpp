import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
# agent ids
for r in rows[:1]: print(r.keys())
by_agent = collections.defaultdict(list)
for r in rows:
    by_agent[r['Agent_Id']].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name']))
for a, ks in by_agent.items():
    ks.sort()
    # take last 40% of timeline (steady TG)
    t0 = ks[0][0]; t1 = ks[-1][1]
    cut = t0 + (t1 - t0) * 0.6
    ks2 = [k for k in ks if k[0] >= cut]
    span = ks2[-1][1] - ks2[0][0]
    busy = sum(e - s for s, e, _ in ks2)
    agg = collections.defaultdict(lambda: [0, 0])
    for s, e, n in ks2:
        n = n.split('(')[0].split('<')[0][:70]
        agg[n][0] += e - s; agg[n][1] += 1
    print(f"agent {a}: kernels {len(ks2)} span {span/1e6:.1f} ms busy {busy/1e6:.1f} ms ({100*busy/span:.0f}%)")
    for n, (t, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:14]:
        print(f"   {100*t/span:5.1f}%  {t/1e6:8.2f} ms  n={c:6d}  avg={t/c/1e3:7.1f}us  {n}")
