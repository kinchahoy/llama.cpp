import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(list)
for r in rows:
    by[r['Agent_Id']].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name'].split('(')[0].split('<')[0].replace('void ','')[:50]))
for a, ks in by.items():
    ks.sort()
    big = [k for k in ks if 'mul_mat' in k[2] or 'mmq' in k[2] or 'gemm' in k[2].lower()]
    # last ubatch: last 1/2 of matmul-active timeline (warmup + 1 rep)
    t1 = ks[-1][1]; t0 = big[len(big)//2][0]
    ks2 = [k for k in ks if k[0] >= t0]
    span = t1 - t0; busy = sum(e-s for s,e,_ in ks2)
    agg = collections.defaultdict(lambda: [0,0])
    for s,e,n in ks2: agg[n][0]+=e-s; agg[n][1]+=1
    print(f"{a}: span {span/1e6:.0f} ms busy {busy/1e6:.0f} ms kernels {len(ks2)}")
    for n,(t,c) in sorted(agg.items(), key=lambda x:-x[1][0])[:16]:
        print(f"  {100*t/span:5.1f}% {t/1e6:8.1f} ms n={c:5d} {n}")
