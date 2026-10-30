import hashlib, pathlib, re, sys
for name in ('baseline', 'flashnext-fast'):
    raw = pathlib.Path(f'/tmp/mx-flashnext-fast-compare-20261029/{name}.stdout').read_text()
    start = raw.rfind('[Start thinking]')
    end = raw.find('[ Prompt:', start)
    if start < 0 or end < 0:
        raise SystemExit(f'could not find generated response boundaries for {name}')
    text = raw[start + len('[Start thinking]'):end].strip()
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    path = pathlib.Path(f'/tmp/mx-flashnext-fast-compare-20261029/{name}.generated.txt')
    path.write_text(text + '\n')
    print(name, 'chars=', len(text), 'sha256=', hashlib.sha256(text.encode()).hexdigest())
