import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path('/home/raistlin/infer/mx-llama.cpp')
work = root / '.worktrees/flashnext-fast'
out = work / 'vr/results/flashnext-fast'
models = {
    '27b': '/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf',
    'next': '/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf',
}
mtp = '/home/raistlin/.cache/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf'

def run(name, model, original=False, device=False, grouped=False, single=False, depth=2, count=256):
    binary = root/'build-ilp/bin/llama-cli' if original else work/'build-fast/bin/llama-cli'
    args = [str(binary), '-m', models[model], '--offline', '-ngl', '999', '-dev', 'ROCm0,ROCm1', '-sm', 'tensor', '-ts', '1,1', '-fa', 'on', '-b', '1024' if model == 'next' else '2048', '-ub', '512' if model == 'next' else '2048', '-c', '100096', '-ctk', 'q8_0', '-ctv', 'q8_0', '--temp', '0', '--top-k', '1', '--seed', '42', '-f', str(root/'flake.nix'), '-n', str(count), '-st', '--no-display-prompt', '--simple-io', '--log-verbosity', '3']
    if depth:
        args += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(depth), '--spec-draft-backend-sampling']
        if model == 'next':
            args += ['-md', mtp]
            if single:
                args += ['-devd', 'ROCm1']
    env = os.environ.copy()
    env['LLAMA_QWEN4_MTP_DEVICE'] = '1' if device else '0'
    env['GGML_CUDA_FLASHNEXT_MOE'] = '1' if grouped else '0'
    started = time.time()
    with (out/f'{name}.stdout').open('wb') as stdout, (out/f'{name}.stderr').open('wb') as stderr:
        p = subprocess.run(args, env=env, stdout=stdout, stderr=stderr, timeout=600)
    stdout_text = (out/f'{name}.stdout').read_text(errors='replace')
    text = (out/f'{name}.stderr').read_text(errors='replace')
    rates = re.findall(r'Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+) t/s', stdout_text)
    result = {'name':name, 'rates':rates, 'returncode':p.returncode, 'seconds':time.time()-started, 'command':args, 'device':device, 'grouped':grouped, 'prompt_sha256':hashlib.sha256((root/'flake.nix').read_bytes()).hexdigest(), 'stdout_sha256':hashlib.sha256((out/f'{name}.stdout').read_bytes()).hexdigest(), 'timings':[line for line in text.splitlines() if any(s in line for s in ['eval time', 'draft', 'accept', 'device MTP'])]}
    (out/f'{name}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return p.returncode

if sys.argv[1] == 'original':
    run('original-27b-plain', '27b', original=True, depth=0)
    run('original-27b-mtp2', '27b', original=True)
elif sys.argv[1] == 'smoke':
    sys.exit(run('next-device-smoke', 'next', device=True, single=True, count=16))
elif sys.argv[1] == 'combined':
    sys.exit(run('next-combined-check', 'next', device=True, grouped=True, single=True))
elif sys.argv[1] == 'final':
    for repeat in range(2):
        for enabled in ([False,True] if repeat==0 else [True,False]):
            if run(f'27b-mtp2-enabled{int(enabled)}-r{repeat}', '27b', device=enabled, grouped=enabled):
                sys.exit(1)
    run('candidate-27b-plain', '27b', depth=0, device=True, grouped=True)
    for repeat, settings in enumerate([[(False,False),(False,True),(True,False),(True,True)],[(True,True),(True,False),(False,True),(False,False)]]):
        for device,grouped in settings:
            if run(f'next-single-d{int(device)}g{int(grouped)}-r{repeat}', 'next', device=device, grouped=grouped, single=True):
                sys.exit(1)
    run('next-tp-control', 'next')
elif sys.argv[1] == 'matrix':
    for repeat, settings in enumerate([[(False,False),(False,True),(True,False),(True,True)],[(True,True),(True,False),(False,True),(False,False)]]):
        for device,grouped in settings:
            if run(f'next-single-d{int(device)}g{int(grouped)}-r{repeat}', 'next', device=device, grouped=grouped, single=True):
                sys.exit(1)
    run('next-tp-control', 'next')
    for repeat in range(2):
        for enabled in ([False,True] if repeat==0 else [True,False]):
            run(f'27b-mtp2-enabled{int(enabled)}-r{repeat}', '27b', device=enabled, grouped=enabled)
    run('candidate-27b-plain', '27b', depth=0, device=True, grouped=True)
