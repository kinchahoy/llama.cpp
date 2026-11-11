import csv, sys, re, collections
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
agents = sorted(set(r['Agent_Id'] for r in rows))
by = collections.defaultdict(list)
for r in rows:
    by[r['Agent_Id']].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name'], r['Grid_Size_X'], r['Workgroup_Size_X'] + 'x' + r['Workgroup_Size_Y'] + ' v' + r['VGPR_Count']))

def short(n):
    n = re.sub(r'\(.*', '', n)
    n = re.sub(r'^void ', '', n)
    return n[:70]

for a in agents:
    ks = sorted(by[a])
    # find the TG window: kernels after the last gap > 20 ms, minus the tail
    gaps = [(ks[i][0] - ks[i - 1][1], i) for i in range(1, len(ks))]
    big = [i for g, i in gaps if g > 20_000_000]
    start = big[-1] if big else 0
    # tokens: count mat-vec launches of the output head (largest grid of q8 mat-vec)
    w = ks[start:]
    t0, t1 = w[0][0], max(e for _, e, *_ in w)
    span = t1 - t0
    busy = 0; cur_s, cur_e = None, None
    for s, e, *_ in w:
        if cur_e is None or s > cur_e:
            if cur_e is not None: busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    tot = collections.Counter(); cnt = collections.Counter()
    for s, e, n, g, wg in w:
        k = short(n)
        tot[k] += e - s; cnt[k] += 1
    # tokens: number of output-head mat-vecs (grid = largest)
    mv = [x for x in w if 'mul_mat_vec_q8_0_repacked' in x[2]]
    gmax = max(int(x[3]) for x in mv) if mv else 0
    ntok = sum(1 for x in mv if int(x[3]) == gmax)
    print(f"agent {a}: window {span/1e6:.1f} ms, kernels {len(w)}, tokens~{ntok}, busy {busy/span*100:.1f}%, per token {span/ntok/1e6:.2f} ms, idle/token {(span-busy)/ntok/1e6:.2f} ms, launches/token {len(w)/ntok:.0f}")
    for k, v in tot.most_common(14):
        print(f"   {v/ntok/1e3:8.1f} us/tok {cnt[k]/ntok:6.1f}/tok {v/cnt[k]/1e3:7.1f} us avg  {v/span*100:5.1f}%  {k}")
    # gap histogram between consecutive kernels
    gl = []
    for i in range(1, len(w)):
        gl.append(w[i][0] - w[i - 1][1])
    gl_pos = [g for g in gl if g > 0]
    import statistics
    print(f"   gaps: n={len(gl_pos)}/tok {len(gl_pos)/ntok:.0f}, median {statistics.median(gl_pos)/1e3:.2f} us, sum/tok {sum(gl_pos)/ntok/1e6:.2f} ms, >20us sum/tok {sum(g for g in gl_pos if g > 20000)/ntok/1e6:.2f} ms")
    # gap preceding each kernel type
    gp = collections.Counter(); gc = collections.Counter()
    for i in range(1, len(w)):
        g = w[i][0] - w[i - 1][1]
        if g > 0:
            gp[short(w[i][2])] += g; gc[short(w[i][2])] += 1
    print("   idle before kernel (top):")
    for k, v in gp.most_common(6):
        print(f"     {v/ntok/1e3:8.1f} us/tok  n/tok {gc[k]/ntok:5.1f}  {k}")
    # mat-vec bandwidth by grid
    mvg = collections.defaultdict(list)
    for s, e, n, g, wg in mv:
        mvg[(g, wg, 'fused' if 'true>' in n.split('(')[0][-12:] or ', true>' in n else '')].append(e - s)
    print("   mat-vec by grid:")
    for (g, wg, f), v in sorted(mvg.items(), key=lambda kv: -sum(kv[1]))[:10]:
        print(f"     grid {g:>8} wg {wg:>5} {f:5} n/tok {len(v)/ntok:5.1f} avg {sum(v)/len(v)/1e3:7.1f} us  sum/tok {sum(v)/ntok/1e3:7.1f} us")
