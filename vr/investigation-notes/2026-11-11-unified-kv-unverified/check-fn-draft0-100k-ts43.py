import concurrent.futures, configparser, hashlib, json, pathlib, subprocess, time, urllib.request
root=pathlib.Path('/home/raistlin/infer/vrversion-llama.cpp')
out=root/'vr/old-ignore/2026-11-11/unified-four-fn-draft0-100k-ts43'
out.mkdir(parents=True,exist_ok=True)
cfg=configparser.ConfigParser(interpolation=None); cfg.read_string('[router]\n'+(root/'vr/server/models.ini').read_text())
prompt=(root/'flake.nix').read_text()
base=['--offline','--host','127.0.0.1','--port','18093','--parallel','4','--kv-unified','--cache-type-k','f16','--cache-type-v','f16','--metrics','--verbosity','4']
def request(path,body=None):
    data=None if body is None else json.dumps(body).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18093'+path,data,{'Content-Type':'application/json'}),timeout=600))
def memory():
    return {str(p.parent.parent):int(p.read_text()) for p in pathlib.Path('/sys/class/drm').glob('card[0-9]/device/mem_info_vram_used')}
for name,ctx in [('flashnext',100352)]:
    args=base.copy()
    opts=dict(cfg['*']);opts.update(cfg[name])
    opts['tensor-split']='0.43,0.57'
    opts['device-draft']='ROCm0'
    opts['ubatch-size']='128'
    opts.update({'ctx-size':str(ctx),'parallel':'4','kv-unified':'true','cache-type-k':'f16','cache-type-v':'f16'})
    for key,value in opts.items():
        if key in ('parallel','kv-unified','cache-type-k','cache-type-v','load-on-startup'):continue
        args+=['--'+key] if value == 'true' else ['--'+key,value]
    command=[str(root/'build-play-b1/bin/llama-server')]+args
    (out/(name+'.command.json')).write_text(json.dumps(command,indent=2))
    print('START',name,'ctx',ctx,flush=True)
    summary={'preset':name,'ctx':ctx,'memory_before':memory()}
    with (out/(name+'.server.log')).open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            start=time.monotonic()
            while True:
                if proc.poll() is not None:raise RuntimeError('server exited '+str(proc.returncode))
                try:
                    if request('/health').get('status')=='ok':break
                except Exception:pass
                if time.monotonic()-start>300:raise TimeoutError('startup')
                time.sleep(1)
            summary['startup_seconds']=time.monotonic()-start
            summary['memory_ready']=memory()
            def run(i):
                body={'prompt':f'Request {i}. Explain the following Nix configuration in detail, including dependencies and build choices.\n'+prompt,'n_predict':300,'ignore_eos':True,'cache_prompt':False,'id_slot':i}
                result=request('/completion',body)
                (out/(name+f'.slot{i}.json')).write_text(json.dumps(result,indent=2))
                return {'slot':i,'timings':result.get('timings'),'tokens_predicted':result.get('tokens_predicted'),'content_length':len(result.get('content','')),'truncated':result.get('truncated')}
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                jobs=[pool.submit(run,i) for i in range(4)]
                peak=memory()
                while not all(j.done() for j in jobs):
                    for k,v in memory().items():peak[k]=max(peak.get(k,0),v)
                    time.sleep(.5)
                summary['requests']=[j.result() for j in jobs]
                summary['memory_peak']=peak
            summary['health_after']=request('/health')
            summary['status']='completed'
        except Exception as exc:
            summary['status']='failed';summary['error']=str(exc)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
    (out/(name+'.summary.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
