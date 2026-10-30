import csv,sys
d,n=sys.argv[1:3]
api=sorted((int(r['Start_Timestamp']),int(r['End_Timestamp']),r['Function']) for r in csv.DictReader(open(f'{d}/{n}_hip_api_trace.csv')))
ker=sorted((int(r['Start_Timestamp']),int(r['End_Timestamp']),r['Agent_Id']) for r in csv.DictReader(open(f'{d}/{n}_kernel_trace.csv')))
gl=[a for a in api if a[2]=='hipGraphLaunch']
i0=gl[-7][0]; i1=gl[-5][0]  # one token period
print('period ms',(i1-i0)/1e6, 'launch durs us',[(round((a[1]-a[0])/1e3)) for a in gl[-8:]])
for s,e,f in api:
    if i0-30e6<=s<=i1 and f!='hipGetDevice':
        if (e-s)>20e3 or f in('hipGraphLaunch','hipMemcpyAsync'):
            print(f'{(s-i0)/1e3:9.1f} {(e-s)/1e3:8.1f} {f}')
