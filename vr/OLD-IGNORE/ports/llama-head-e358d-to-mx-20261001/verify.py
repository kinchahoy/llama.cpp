#!/usr/bin/env python3
"""Replay the patch kit on a clean archive and compare it with the target."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import regenerate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=regenerate.DEFAULT_BASE)
    parser.add_argument("--target", default=regenerate.DEFAULT_TARGET)
    args = parser.parse_args()
    base = regenerate.git("rev-parse", "--verify", args.base + "^{tree}").decode().strip()
    target = regenerate.git("rev-parse", "--verify", args.target + "^{tree}").decode().strip()
    files = [line.split("\t", 1)[1] for line in (regenerate.HERE / "MANIFEST.tsv").read_text().splitlines()[1:]]

    with tempfile.TemporaryDirectory(prefix="mx-port-check-") as directory:
        root = Path(directory)
        archive = subprocess.Popen(("git", "-C", str(regenerate.REPO), "archive", base), stdout=subprocess.PIPE)
        try:
            subprocess.run(("tar", "-x", "-C", directory), stdin=archive.stdout, check=True)
        finally:
            archive.stdout.close()
        if archive.wait():
            raise SystemExit("git archive failed")

        for name in regenerate.PATCHES:
            patch = regenerate.HERE / "patches" / name
            subprocess.run(("git", "apply", "--check", str(patch)), cwd=root, check=True)
            subprocess.run(("git", "apply", str(patch)), cwd=root, check=True)

        for name in files:
            expected = subprocess.run(("git", "-C", str(regenerate.REPO), "show", f"{target}:{name}"), capture_output=True)
            actual = root / name
            if expected.returncode:
                if actual.exists():
                    raise SystemExit(f"expected deleted file: {name}")
            elif not actual.is_file() or actual.read_bytes() != expected.stdout:
                raise SystemExit(f"file differs from target: {name}")

    print(f"Verified {len(files)} patched files from {base} against {target}")


if __name__ == "__main__":
    main()
