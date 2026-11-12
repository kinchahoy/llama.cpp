import concurrent.futures, json, pathlib, subprocess, time, urllib.request, hashlib
root=pathlib.Path('/home/raistlin/infer/vrversion-llama.cpp'); out=pathlib.Path('/tmp/vr-unverified-2026-11-11/verification');out.mkdir(exist_ok=True)
prompts=out/'prompts';prompts.mkdir(exist_ok=True)
for i,path,n in [(1,'flake.nix',6000),(2,'src/llama-context.cpp',6000),(3,'tools/server/README.md',4000)]:
 (prompts/f'p{i}.txt').write_bytes((root/path).read_bytes()[:n])
(prompts/'p4.txt').write_text('Write a detailed history of the printing press, covering Gutenberg, the spread across Europe, and its social effects.')
(prompts/'p5.txt').write_text('Write a Python function that parses an ISO 8601 duration string like P3DT4H5M into seconds, with tests.')
def req(path,body=None):
 data=None if body is None else json.dumps(body).encode()
 return json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18093'+path,data,{'Content-Type':'application/json'}),timeout=600))
def alter(cmd, changes):
 cmd=cmd.copy()
 for k,v in changes.items():
  if k in cmd:cmd[cmd.index(k)+1]=str(v)
  else:cmd +=[k,str(v)]
 return cmd
q=json.loads((out.parent/'unified-four-mtp274k/qwen27b.command.json').read_text())
f=json.loads((out.parent/'unified-four-fn-draft0-64k/flashnext.command.json').read_text())
# Reference server settings reproduce spec_accept.sh; unified variants keep model batch sizes.
qb=alter(q,{'--parallel':1,'--ctx-size':16384});qb[qb.index('--kv-unified')]='--no-kv-unified'
fb=alter(f,{'--parallel':1,'--ctx-size':100352,'--device-draft':'ROCm1','--tensor-split':'0.52,0.48'});fb[fb.index('--kv-unified')]='--no-kv-unified'
# Leave margin for the isolation tests instead of reusing the 64K allocation with only 10 MiB spare.
fu=alter(f,{'--ctx-size':32768})
bench=[str(root/'build-play-b1/bin/llama-bench'),'-m',q[q.index('--model')+1],'-ngl','999','-dev','ROCm0,ROCm1','-sm','tensor','-ts','1,1','-fa','on','-b','2048','-ub','2048','-t','20','-p','2048','-n','128','-r','3','-o','json']
(out/'bench.command.json').write_text(json.dumps(bench,indent=2))
print('BENCH Qwen PP2048/TG128 r3 2048/2048',flush=True)
with (out/'qwen.bench.json').open('w') as stdout,(out/'qwen.bench.stderr').open('w') as stderr:
 result=subprocess.run(bench,stdout=stdout,stderr=stderr);print('BENCH exit',result.returncode,flush=True)
for name,cmd,check in [('qwen-reference',qb,False),('qwen-unified',q,True),('flashnext-reference',fb,False),('flashnext-unified',fu,True)]:
 print('START',name,flush=True);(out/(name+'.command.json')).write_text(json.dumps(cmd,indent=2));summary={}
 with (out/(name+'.log')).open('w') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  try:
   start=time.monotonic()
   while True:
    if proc.poll() is not None:raise RuntimeError(f'server exit {proc.returncode}')
    try:
     if req('/health').get('status')=='ok':break
    except Exception:pass
    if time.monotonic()-start>300:raise TimeoutError('startup')
    time.sleep(1)
   with (out/(name+'.performance.txt')).open('w') as dest:
    result=subprocess.run(['python3','-I',str(root/'vr/scripts/spec_accept.py'),'18093',str(prompts),'256'],stdout=dest,stderr=subprocess.STDOUT)
   summary['performance_exit']=result.returncode
   print(name,(out/(name+'.performance.txt')).read_text(),flush=True)
   if check:
    texts=[(prompts/'p1.txt').read_text()+'\nExplain the compiler and package choices in this flake.\nExplanation:\n',(prompts/'p3.txt').read_text()+'\nExplain the API endpoints and server options described above.\nExplanation:\n']
    def generate(i,slot,label):
     d=req('/completion',{'prompt':texts[i],'id_slot':slot,'n_predict':128,'temperature':0,'top_k':1,'seed':42,'cache_prompt':False,'ignore_eos':True,'n_probs':1,'return_tokens':True})
     (out/(name+'.'+label+f'.prompt{i}.json')).write_text(json.dumps(d,indent=2));return d
    serial=[generate(i,i,'serial') for i in range(2)]
    waves=[]
    for label,slots in [('concurrent',[0,1]),('swapped',[1,0])]:
     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
      futures=[pool.submit(generate,i,slots[i],label) for i in range(2)];results=[f.result() for f in futures]
     rows=[]
     for i,d in enumerate(results):
      probs=d.get('completion_probabilities',d.get('probs',[])); mismatches=[]
      for j,row in enumerate(probs):
       top=row.get('top_logprobs',row.get('probs',[]))
       if top and 'id' in row and 'id' in top[0] and row['id']!=top[0]['id']:mismatches.append(j)
      a=serial[i].get('tokens',[]);b=d.get('tokens',[])
      prefix=0
      for x,y in zip(a,b):
       if x!=y:break
       prefix+=1
      rows.append({'prompt':i,'slot':slots[i],'text_equal_serial':d.get('content')==serial[i].get('content'),'token_prefix_equal':prefix,'token_count':len(b),'probability_rows':len(probs),'non_argmax_rows':mismatches,'predicted':d.get('tokens_predicted')})
     waves.append({'wave':label,'results':rows})
    summary['isolation']=waves;print(name,'ISOLATION',json.dumps(waves),flush=True)
   summary['health_after']=req('/health');summary['status']='completed'
  except Exception as e:summary['status']='failed';summary['error']=str(e);print(name,summary,flush=True)
  finally:
   if proc.poll() is None:
    proc.terminate()
    try:proc.wait(timeout=20)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
 (out/(name+'.summary.json')).write_text(json.dumps(summary,indent=2))
