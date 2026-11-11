import csv, sys, glob, collections
# usage: summ.py DIR NTOK
d, ntok = sys.argv[1], int(sys.argv[2])
f = glob.glob(d + '/**/*kernel_trace.csv', recursive=True)[0]
rows = list(csv.DictReader(open(f)))
rows.sort(key=lambda r: int(r['Start_Timestamp']))
# window: after last gap > 20 ms
start = 0
for i in range(1, len(rows)):
    if int(rows[i]['Start_Timestamp']) - int(rows[i-1]['End_Timestamp']) > 20e6:
        start = i
rows = rows[start:]
span = (int(rows[-1]['End_Timestamp']) - int(rows[0]['Start_Timestamp'])) / 1e6
print(f'window {len(rows)} dispatches, span {span:.1f} ms, ms/tok {span/ntok:.2f}')
agents = collections.defaultdict(list)
for r in rows:
    agents[r['Agent_Id']].append(r)
for a, rs in sorted(agents.items()):
    busy = sum(int(r['End_Timestamp']) - int(r['Start_Timestamp']) for r in rs) / 1e6
    print(f'agent {a}: busy {busy:.1f} ms ({100*busy/span:.1f}%), dispatches {len(rs)}')
    k = collections.defaultdict(lambda: [0, 0])
    for r in rs:
        n = r['Kernel_Name'].split('(')[0][:60]
        k[n][0] += 1
        k[n][1] += int(r['End_Timestamp']) - int(r['Start_Timestamp'])
    for n, (c, t) in sorted(k.items(), key=lambda x: -x[1][1])[:14]:
        print(f'  {t/1e6/ntok:7.3f} ms/tok {c/ntok:7.1f}/tok avg {t/c/1e3:6.1f} us  {n}')
