#!/usr/bin/env python3
"""Export the runtime delta between Git trees as file-disjoint, full-index patches."""

import argparse
import hashlib
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_BASE = "e358d59178377be4c58ba567925e05faadbccb57"
DEFAULT_TARGET = "8a6351ddf09c444d342b671bc9a26559735f19e1"
PATCHES = (
    "0001-core-ggml.patch",
    "0002-gpu-backends.patch",
    "0003-model-runtime.patch",
    "0004-common-tools.patch",
    "0005-build-harness.patch",
)


def git(*args):
    return subprocess.check_output(("git", "-C", str(REPO), *args))


def group(path):
    if path.startswith(("ggml/src/ggml-cuda/", "ggml/src/ggml-hip/")):
        return 1
    if path.startswith("ggml/"):
        return 0
    if path.startswith(("src/", "include/", "gguf-py/", "conversion/")):
        return 2
    if path.startswith(("common/", "examples/", "tools/", "tests/")):
        return 3
    if path in ("CMakeLists.txt", "vr/CMakeLists.txt", "vr/test-gfx906-tp.cpp"):
        return 4
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()
    base = git("rev-parse", "--verify", args.base + "^{tree}").decode().strip()
    target = git("rev-parse", "--verify", args.target + "^{tree}").decode().strip()
    paths = git("diff", "--name-only", "--no-renames", "-z", base, target).decode().split("\0")[:-1]
    grouped = [[] for _ in PATCHES]
    excluded = []
    for path in paths:
        index = group(path)
        if index is None:
            excluded.append(path)
        else:
            grouped[index].append(path)

    patch_dir = HERE / "patches"
    patch_dir.mkdir(exist_ok=True)
    checksums = []
    manifest = ["patch\tpath"]
    for name, files in zip(PATCHES, grouped):
        if not files:
            raise SystemExit(f"empty patch group: {name}")
        data = git("diff", "--binary", "--full-index", "--no-renames", "--no-ext-diff", base, target, "--", *files)
        (patch_dir / name).write_bytes(data)
        checksums.append(f"{hashlib.sha256(data).hexdigest()}  patches/{name}")
        manifest.extend(f"{name}\t{path}" for path in files)
    (HERE / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    (HERE / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n")
    (HERE / "EXCLUDED.txt").write_text("\n".join(excluded) + "\n")
    print(f"Exported {sum(map(len, grouped))} runtime files from {base} to {target}; excluded {len(excluded)} non-runtime files")


if __name__ == "__main__":
    main()
