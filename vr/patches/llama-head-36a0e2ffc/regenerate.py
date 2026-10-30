#!/usr/bin/env python3
"""Export the MX/gfx906 runtime overlay as five full-index patches against an upstream llama.cpp commit.

Patches 0001-0004 are file-disjoint (base -> mid). Patch 0005 applies on top of them (mid -> target).
Run from anywhere: python3 vr/patches/<kit>/regenerate.py [--base B --mid M --target T --full F]
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_BASE = "36a0e2ffc0b352b774f8d71c43668b75b3061ef5"
DEFAULT_MID = "14fb69bd91d3048dffd7bc20f489e607f28fc8fe"
DEFAULT_TARGET = "3875d3850b45825cc12d5aecb22487b7377ad598"
# full stage D tree, incl. fork docs; only used to list EXCLUDED.txt
DEFAULT_FULL = "b8f809e2fae2cbc929196479cc60fe29d4c7fd0d"

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
    ap.add_argument("--full", default=DEFAULT_FULL)
    args = ap.parse_args()
    base, mid, target = tree(args.base), tree(args.mid), tree(args.target)

    paths = git("diff", "--name-only", "--no-renames", "-z", base, mid).decode().split("\0")[:-1]
    grouped = [[] for _ in PATCHES]
    for p in paths:
        g = group(p)
        if g is None:
            raise SystemExit(f"non-runtime path in mid: {p}")
        grouped[g].append(p)
    excluded = [p for p in git("diff", "--name-only", "--no-renames", "-z", base, tree(args.full)).decode().split("\0")[:-1]
                if group(p) is None]
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
