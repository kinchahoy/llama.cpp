#!/usr/bin/env python3
"""Export the MX/gfx906 runtime overlay as five full-index patches against an upstream llama.cpp commit.

Patches 0001-0004 are file-disjoint (base -> mid). Patch 0005 applies on top of them (mid -> target).
Run from anywhere: python3 vr/patches/<kit>/regenerate.py [--base B --mid M --target T]
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], text=True).strip())
DEFAULT_BASE = "344be0a60074e6395da3c4a4f300f7361df5a67d"
DEFAULT_MID = "00224c01e87a837c7630b024219a0224a43e1398"
DEFAULT_TARGET = "5f6a45b574595e809c83a03f343ba8b801431111"

PATCHES = (
    "0001-ggml-core-and-tp-meta-backend.patch",
    "0002-ggml-hip-gfx906-kernels.patch",
    "0003-llama-model-runtime.patch",
    "0004-common-tools-tests-harness.patch",
)
PATCH_TOP = "0005-tp-token-graph-fusions-ar-overlap.patch"


def git(*args):
    return subprocess.check_output(("git", "-C", str(REPO), *args))


def group(path):
    if path.startswith(("ggml/src/ggml-cuda/", "ggml/src/ggml-hip/")):
        return 1
    if path.startswith("ggml/"):
        return 0
    if path.startswith(("src/", "include/", "gguf-py/")):
        return 2
    if path.startswith(("common/", "examples/", "tools/", "tests/")) or \
            path in ("CMakeLists.txt", "vr/CMakeLists.txt", "vr/test-gfx906-tp.cpp", "vr/test-flashnext-moe.cpp"):
        return 3
    return None


def tree(rev):
    return git("rev-parse", "--verify", rev + "^{tree}").decode().strip()


def diff(a, b, files):
    return git("diff", "--binary", "--full-index", "--no-renames", "--no-ext-diff", a, b, "--", *files)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--mid", default=DEFAULT_MID)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    args = ap.parse_args()
    base, mid, target = tree(args.base), tree(args.mid), tree(args.target)

    paths = git("diff", "--name-only", "--no-renames", "-z", base, mid).decode().split("\0")[:-1]
    grouped = [[] for _ in PATCHES]
    excluded = []
    for p in paths:
        g = group(p)
        (excluded if g is None else grouped[g]).append(p)
    top = [p for p in git("diff", "--name-only", "--no-renames", "-z", mid, target).decode().split("\0")[:-1]
           if group(p) is not None]

    out = HERE / "patches"
    out.mkdir(exist_ok=True)
    sums, manifest = [], ["patch\tpath"]
    for name, data, files in [(n, diff(base, mid, f), f) for n, f in zip(PATCHES, grouped)] + [(PATCH_TOP, diff(mid, target, top), top)]:
        if not files:
            raise SystemExit(f"empty patch: {name}")
        (out / name).write_bytes(data)
        sums.append(f"{hashlib.sha256(data).hexdigest()}  patches/{name}")
        manifest += [f"{name}\t{p}" for p in files]
    (HERE / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (HERE / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n")
    (HERE / "EXCLUDED.txt").write_text("\n".join(excluded) + "\n")
    print(f"patches: {[len(f) for f in grouped] + [len(top)]} files; excluded {len(excluded)} non-runtime paths")


if __name__ == "__main__":
    main()
