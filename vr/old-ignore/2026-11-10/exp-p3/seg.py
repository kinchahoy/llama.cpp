import csv, sys, glob, collections
# per-AR segment analysis: usage seg.py TRACE_DIR NTOK
d, ntok = sys.argv[1], int(sys.argv[2])
f = glob.glob(d + '/**/*kernel_trace.csv', recursive=True)[0]
rows = list(csv.DictReader(open(f)))
rows.sort(key=lambda r: int(r['Start_Timestamp']))
start = 0
for i in range(1, len(rows)):
    if int(rows[i]['Start_Timestamp']) - int(rows[i-1]['End_Timestamp']) > 20e6:
        start = i
rows = rows[start:]
ag = collections.defaultdict(list)
for r in rows:
    ag[r['Agent_Id']].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name'].split('(')[0]))
names = sorted(ag)
segs = {}
for a in names:
    ks = ag[a]
    out = []  # per AR: (ar_start, ar_end, busy_before, mmv_before, other_before, gap_before, nk)
    prev_end = None; busy = mmv = gap = 0; nk = 0; last_end = None
    for s, e, n in ks:
        if last_end is not None and s > last_end:
            gap += s - last_end
        last_end = e if last_end is None else max(last_end, e)
        if 'ggml_cuda_ar_kernel' in n:
            out.append((s, e, busy, mmv, gap, nk))
            busy = mmv = gap = 0; nk = 0
        else:
            busy += e - s; nk += 1
            if 'mul_mat_vec' in n: mmv += e - s
    segs[a] = out
a0, a1 = names
n = min(len(segs[a0]), len(segs[a1]))
print('ARs', len(segs[a0]), len(segs[a1]))
S = lambda x: x/1e3
tot = collections.defaultdict(float)
skew = []
for i in range(1, n):
    x, y = segs[a0][i], segs[a1][i]
    skew.append(y[0] - x[0])
    tot['busy0'] += x[2]; tot['busy1'] += y[2]
    tot['mmv0'] += x[3]; tot['mmv1'] += y[3]
    tot['gap0'] += x[4]; tot['gap1'] += y[4]
    tot['ar0'] += x[1]-x[0]; tot['ar1'] += y[1]-y[0]
    tot['endskew'] += abs(y[1]-x[1])
for k, v in tot.items():
    print(f'{k}: {v/1e6/ntok:.3f} ms/tok')
skew.sort()
m = len(skew)
print('AR start skew (ROCm1 - ROCm0) us: p10 %.1f p50 %.1f p90 %.1f mean %.1f' % (S(skew[m//10]), S(skew[m//2]), S(skew[9*m//10]), S(sum(skew)/m)))
# per position-in-token pattern (130 ARs/token)
per = 130
pos = collections.defaultdict(list)
for i in range(1, n):
    x, y = segs[a0][i], segs[a1][i]
    pos[i % per].append((y[0]-x[0], x[2], y[2], x[3], y[3], x[4], y[4], x[5]))
print('pos  skew  busy0 busy1  mmv0 mmv1  gap0 gap1 nk (us)')
for p in sorted(pos)[:12]:
    v = pos[p]; k = len(v)
    print(p, ' '.join('%6.1f' % (S(sum(t[j] for t in v)/k)) for j in range(7)), v[0][7])
