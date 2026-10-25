#!/usr/bin/env python3
"""Replay the kit on a clean checkout of the base commit and compare every patched file with the target tree."""

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regenerate as kit  # noqa: E402


def main():
    rows = [l.split("\t") for l in (HERE / "MANIFEST.tsv").read_text().splitlines()[1:]]
    with tempfile.TemporaryDirectory() as tmp:
        wt = Path(tmp) / "wt"
        kit.git("worktree", "add", "--detach", str(wt), kit.DEFAULT_BASE)
        try:
            for name in kit.PATCHES + (kit.PATCH_TOP,):
                subprocess.check_call(("git", "-C", str(wt), "apply", "--whitespace=nowarn", str(HERE / "patches" / name)))
            bad = []
            for path in sorted({p for _, p in rows}):
                want = subprocess.run(("git", "-C", str(kit.REPO), "rev-parse", f"{kit.DEFAULT_TARGET}:{path}"),
                                      capture_output=True, text=True).stdout.strip()
                f = wt / path
                have = subprocess.check_output(("git", "hash-object", str(f)), text=True).strip() if f.exists() else ""
                if want != have:
                    bad.append(path)
            print(f"verified {len(rows)} files, {len(bad)} mismatches" + ("".join("\n  " + b for b in bad)))
            return 1 if bad else 0
        finally:
            kit.git("worktree", "remove", "--force", str(wt))


if __name__ == "__main__":
    sys.exit(main())
