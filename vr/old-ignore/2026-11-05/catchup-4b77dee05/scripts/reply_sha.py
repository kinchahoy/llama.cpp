# usage: python3 reply_sha.py FILE...   sha256 (12 hex) of the llama-cli reply only: after the echoed "(truncated)" prompt, before "[ Prompt"
import hashlib, sys
for p in sys.argv[1:]:
    lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
    start = next(i for i, l in enumerate(lines) if "(truncated)" in l) + 1
    end = next(i for i, l in enumerate(lines) if l.startswith("[ Prompt"))
    reply = "\n".join(lines[start:end]).strip()
    print(f"{hashlib.sha256(reply.encode()).hexdigest()[:12]} {len(reply):5d} bytes  {p}")
