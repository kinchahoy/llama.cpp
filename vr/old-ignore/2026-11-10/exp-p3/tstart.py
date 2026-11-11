import csv, sys, glob, collections
d = sys.argv[1]; tok = int(sys.argv[2]); nb = int(sys.argv[3]) if len(sys.argv) > 3 else 12
f = glob.glob(d + '/**/*kernel_trace.csv', recursive=True)[0]
rows = list(csv.DictReader(open(f)))
rows.sort(key=lambda r: int(r['Start_Timestamp']))
ag = collections.defaultdict(list)
for r in rows:
    ag[r['Agent_Id']].append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name'].split('(')[0][:45], r['Queue_Id'], r['Grid_Size_X']))
names = sorted(ag)
ref = None
for a in names:
    ks = ag[a]
    idx = [i for i, k in enumerate(ks) if k[2].startswith('void rms_norm_f32<1024, 4')]
    i = idx[-64 + tok]
    if ref is None:
        ref = ks[i][0]
    print('==', a)
    prev = ks[i - nb - 1][1]
    for k in ks[i - nb:i + 4]:
        print('%9.1f gap %7.1f dur %6.1f q%s %s grid %s' % ((k[0]-ref)/1e3, (k[0]-prev)/1e3, (k[1]-k[0])/1e3, k[3], k[2], k[4]))
        prev = k[1]
