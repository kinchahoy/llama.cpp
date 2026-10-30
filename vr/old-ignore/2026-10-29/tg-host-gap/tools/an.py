import csv,sys,collections
d=sys.argv[1]; n=sys.argv[2]
api=[(int(r['Start_Timestamp']),int(r['End_Timestamp']),r['Function'],r['Thread_Id']) for r in csv.DictReader(open(f'{d}/{n}_hip_api_trace.csv'))]
ker=[(int(r['Start_Timestamp']),int(r['End_Timestamp']),r['Agent_Id']) for r in csv.DictReader(open(f'{d}/{n}_kernel_trace.csv'))]
cp=[(int(r['Start_Timestamp']),int(r['End_Timestamp']),r['Direction']) for r in csv.DictReader(open(f'{d}/{n}_memory_copy_trace.csv'))]
api.sort()
gl=[a for a in api if a[2]=='hipGraphLaunch']
print('graph launches',len(gl))
# kernel union
ks=sorted(ker)
# idle gaps: both agents idle
merged=[]
for s,e,_ in ks:
    if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],e)
    else: merged.append([s,e])
# take window: last 'N' gaps after last 20 graph launch groups; define window from the Kth-from-last graph launch
K=int(sys.argv[3]) if len(sys.argv)>3 else 20
# per-token: cluster graph launches (gap>2ms)
toks=[];cur=[gl[0]]
for g in gl[1:]:
    if g[0]-cur[-1][1]>1e6: toks.append(cur);cur=[g]
    else: cur.append(g)
toks.append(cur)
print('graph-launch clusters',len(toks),collections.Counter(len(t) for t in toks))
# window from last K clusters
t0=toks[-K][0][0]; t1=toks[-1][-1][1]+1
gaps=[(merged[i][1],merged[i+1][0]) for i in range(len(merged)-1) if merged[i][1]>=t0 and merged[i+1][0]<=t1+30e6]
gaps=[g for g in gaps if g[1]-g[0]>20e3]
tot=sum(b-a for a,b in gaps)
print(f'window {(t1-t0)/1e6:.1f} ms for {K} tokens, {(t1-t0)/1e6/K:.2f} ms/tok; idle(both) {tot/1e6:.1f} ms = {tot/1e6/K:.2f}/tok, ngaps {len(gaps)}')
big=sorted(gaps,key=lambda g:g[0])
# show last 3 tokens detail
print('gap sizes ms',[round((b-a)/1e6,2) for a,b in big][-30:])
agg=collections.defaultdict(lambda:[0,0.0])
for a,b in gaps:
    for s,e,f,t in api:
        if e<a or s>b: continue
        if f in('hipGetDevice','hipGetLastError','hipSetDevice'): 
            k=f
        else: k=f
        agg[(f,t)][0]+=1; agg[(f,t)][1]+=(min(e,b)-max(s,a))/1e3
for k,v in sorted(agg.items(),key=lambda x:-x[1][1])[:15]: print(k,v[0],f'{v[1]/K:.0f} us/tok')
cc=collections.Counter()
for a,b in gaps:
    for s,e,dd in cp:
        if a<=s<=b: cc[dd]+=1
print('copies in gaps',dict(cc))
# show sequence for 2 last gaps
for a,b in big[-6:-3]:
    print('--- gap',(b-a)/1e3,'us')
    for s,e,f,t in api:
        if e>=a-200e3 and s<=b+200e3 and f not in('hipGetDevice',): print(f'  {(s-a)/1e3:9.1f} {(e-s)/1e3:8.1f} {f} {t}')
