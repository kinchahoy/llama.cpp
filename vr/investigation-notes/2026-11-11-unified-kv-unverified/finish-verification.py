import pathlib,json,subprocess,time
p=pathlib.Path('/tmp/vr-unverified-2026-11-11/verification')
start=time.monotonic()
while not (p/'flashnext-unified.summary.json').exists():
 if time.monotonic()-start>600:raise TimeoutError('preceding tests')
 time.sleep(1)
cmd=json.loads((p/'bench.corrected.command.json').read_text())
print('CORRECTED two-GPU PP2048/TG128 r3',flush=True)
with (p/'qwen.corrected.bench.json').open('w') as output,(p/'qwen.corrected.bench.stderr').open('w') as error:
 result=subprocess.run(cmd,stdout=output,stderr=error);print('BENCH EXIT',result.returncode,flush=True)
if result.returncode==0:
 for row in json.loads((p/'qwen.corrected.bench.json').read_text()):print(json.dumps({k:row[k] for k in ['devices','tensor_split','n_batch','n_ubatch','n_prompt','n_gen','avg_ts','stddev_ts']}),flush=True)
subprocess.run(['python3','-I','/tmp/vr-unverified-2026-11-11/control.py'],check=True)
