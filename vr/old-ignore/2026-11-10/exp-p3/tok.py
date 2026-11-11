import csv, sys, glob, collections, statistics as st
d = sys.argv[1]
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
    ag[r['Agent_Id']].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name'].split('(')[0][:40]))
names = sorted(ag)
ar = {a: [k for k in ag[a] if 'ar_kernel' in k[2]] for a in names}
a0, a1 = names
n = len(ar[a0])
sk = [(ar[a1][i][0] - ar[a0][i][0])/1e3 for i in range(n)]
pos = collections.defaultdict(list)
for i in range(130, n):
    pos[i % 130].append(sk[i])
print('median skew by AR position (us):')
print(' '.join('%d:%.0f' % (p, st.median(v)) for p, v in sorted(pos.items())))
print('sum of median skews per token (ms): %.3f   sum of means: %.3f' % (sum(st.median(v) for v in pos.values())/1e3, sum(st.mean(v) for v in pos.values())/1e3))
print('positions where mean - median > 3 us:')
for p, v in sorted(pos.items()):
    if st.mean(v) - st.median(v) > 3:
        print('  pos %d mean %.1f median %.1f max %.1f' % (p, st.mean(v), st.median(v), max(v)))
big = [s for s in sk[130:] if s > 100]
print('ARs with skew > 100 us: %d over %d tokens, sum %.2f ms/token' % (len(big), (n-130)//130, sum(big)/1e3/((n-130)/130)))
small = [s for s in sk[130:] if s <= 100]
print('rest: sum %.2f ms/token' % (sum(small)/1e3/((n-130)/130)))
# for big-skew ARs: did ROCm1 start its token late? find the last big gap on ROCm1 before that AR
k1 = ag[a1]
starts1 = [k1[i][0] for i in range(1, len(k1)) if k1[i][0]-k1[i-1][1] > 200e3]
import bisect
idx = [i for i in range(130, n) if sk[i] > 100]
first_after = set()
for t in starts1:
    j = bisect.bisect_left([x[0] for x in ar[a1]], t)
    first_after.add(j)
print('big-skew ARs that are the first AR after a ROCm1 idle gap > 200 us: %d of %d' % (sum(1 for i in idx if i in first_after), len(idx)))
fs = {a: [k for k in ag[a] if k[2].startswith('void rms_norm_f32<1024, 4')] for a in names}
print('rms_norm<1024,4> count', len(fs[a0]), len(fs[a1]))
d0 = [(fs[a1][i][0]-fs[a0][i][0])/1e3 for i in range(min(len(fs[a0]), len(fs[a1])))]
print('token-start (rms_norm<1024,4>) skew ROCm1-ROCm0 us: median %.1f mean %.1f min %.1f max %.1f' % (st.median(d0), st.mean(d0), min(d0), max(d0)))
cb = {a: [k for k in ag[a] if 'copyBuffer' in k[2]] for a in names}
print('copyBuffer count', len(cb[a0]), len(cb[a1]))
bb = {}
for a in names:
    ks = ag[a]
    b = [(ks[i][0], ks[i][2], (ks[i][0]-ks[i-1][1])/1e3) for i in range(1, len(ks)) if ks[i][0]-ks[i-1][1] > 200e3]
    print(a, 'big gaps', len(b), 'median gap us %.0f' % st.median([x[2] for x in b]), 'kernels after:', collections.Counter(x[1] for x in b).most_common(3))
    bb[a] = b
b0, b1 = bb[a0], bb[a1]
m = min(len(b0), len(b1))
print('token-start skew ROCm1-ROCm0 us: median %.1f' % st.median([(b1[i][0]-b0[i][0])/1e3 for i in range(m)]))
