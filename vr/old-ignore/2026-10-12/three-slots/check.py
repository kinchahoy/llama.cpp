import concurrent.futures
import glob
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8093'

def api(path, data=None):
    req = urllib.request.Request(BASE + path, data=None if data is None else json.dumps(data).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)

def memory():
    return [int(Path(p).read_text()) for p in sorted(glob.glob('/sys/class/drm/card*/device/mem_info_vram_used'))]

def run(mode, ctx, test):
    name = f'{mode}-{ctx}' + os.environ.get('CHECK_LABEL', '')
    cmd = [str(ROOT / 'build-ilp/bin/llama-server'), '-hf', 'unsloth/Qwen3.8-27B-GGUF:Q8_0',
           '--offline', '--no-mmproj', '-ngl', '999', '-dev', 'ROCm0,ROCm1', '-sm', 'tensor',
           '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16', '-b', os.environ.get('CHECK_BATCH', '2048'), '-ub', os.environ.get('CHECK_BATCH', '2048'),
           '-c', str(ctx), '-np', '3', '--host', '127.0.0.1', '--port', '8093', '--slots', '-lv', '4',
           '--spec-type', {'plain': 'none', 'mtp': 'draft-mtp', 'dflash': 'draft-dflash'}[mode]]
    if mode == 'mtp':
        cmd += ['--spec-draft-n-max', '2', '--spec-draft-backend-sampling']
    if os.environ.get('CHECK_SPLIT'):
        cmd += ['-ts', os.environ['CHECK_SPLIT']]
    if mode == 'dflash':
        cmd += ['--spec-draft-hf', 'z-lab/Qwen3.8-27B-DFlash2-GGUF:Q8_0',
                '--spec-draft-model', 'Qwen3.8-27B-DFlash2-Q8_0.gguf', '--spec-draft-n-max', '5']
    env = dict(os.environ)
    env.pop('GGML_CUDA_TP_OVERLAP', None)
    result = {'mode': mode, 'ctx': ctx, 'command': cmd,
              'environment': {k: v for k, v in env.items() if k.startswith(('LLAMA_', 'GGML_', 'CHECK_'))}}
    with (OUT / (name + '.log')).open('w') as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=log, env=env)
        try:
            for _ in range(600):
                if p.poll() is not None:
                    raise RuntimeError(f'server exited {p.returncode}')
                try:
                    if api('/health').get('status') == 'ok':
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('startup timeout')
            result['idle_bytes'] = memory()
            result['slots'] = api('/slots')
            print(json.dumps(result), flush=True)
            if test:
                peak = memory()
                stop = threading.Event()
                observations = []
                def monitor():
                    while not stop.wait(.1):
                        for i, used in enumerate(memory()):
                            peak[i] = max(peak[i], used)
                        try:
                            slots = api('/slots')
                            if sum(s['is_processing'] for s in slots) == 3 and not observations:
                                observations.append(slots)
                        except Exception:
                            pass
                t = threading.Thread(target=monitor)
                t.start()
                try:
                    for wave in range(2):
                        prompts = [
                            'Summarize this build documentation and explain the key choices:\n' + (ROOT / 'docs/build.md').read_text(),
                            'Review this CUDA code for correctness and explain your findings:\n' + (ROOT / 'ggml/src/ggml-cuda/allreduce.cu').read_text()[:18000],
                            'Write a Python LRU cache with expiration. Explain concurrency, testing, and complexity in detail.'
                        ]
                        if wave:
                            prompts = [f'For project {s}, explain and prove the Pythagorean theorem, then give worked examples.' for s in ('AMBER', 'COBALT', 'JADE')]
                        def request(i):
                            data = api('/completion', {'prompt': prompts[i], 'id_slot': i, 'n_predict': 256 if wave == 0 else 128,
                                       'temperature': 0, 'top_k': 1, 'seed': 42, 'cache_prompt': False,
                                       'n_probs': 2, 'ignore_eos': True})
                            (OUT / f'{name}-wave{wave}-slot{i}.json').write_text(json.dumps(data, indent=2))
                            summary = {k: data.get(k) for k in ('id_slot', 'tokens_predicted', 'tokens_evaluated', 'timings', 'stop_type')}
                            bad = []
                            for j, token in enumerate(data.get('completion_probabilities', [])):
                                allowed = [v for v in token['top_logprobs'] if v['id'] not in (248044, 248046, 248063, 248064, 248065)]
                                if not math.isfinite(token['logprob']) or (allowed and token['logprob'] < max(v['logprob'] for v in allowed) - 1e-5):
                                    bad.append(j)
                            summary['non_argmax_positions'] = bad
                            return summary
                        start = time.monotonic()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                            responses = list(pool.map(request, range(3)))
                        entry = {'wave': wave, 'elapsed': time.monotonic() - start, 'responses': responses}
                        result.setdefault('waves', []).append(entry)
                        print(json.dumps(entry), flush=True)
                    result['peak_bytes'] = peak
                    result['three_active_slots'] = observations
                    result['final_slots'] = api('/slots')
                finally:
                    stop.set()
                    t.join()
        except Exception as e:
            result['error'] = str(e)
            print(json.dumps(result), flush=True)
        finally:
            p.send_signal(signal.SIGINT) if p.poll() is None else None
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            log.flush()
            log_text = (OUT / (name + '.log')).read_text()
            result['decode_init_errors'] = log_text.count('E decode: failed to initialize batch')
            result['assertion'] = 'GGML_ASSERT' in log_text
            (OUT / (name + '-summary.json')).write_text(json.dumps(result, indent=2))

if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]), len(sys.argv) > 3)
