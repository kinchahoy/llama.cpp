#!/usr/bin/env python3

import argparse
import json
import pathlib
import sys


def load(path, prompt):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["n_prompt"] == prompt and row["n_gen"] == 0:
                return row["avg_ts"]
    raise RuntimeError(f"pp{prompt} result not found in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, type=int)
    parser.add_argument("--minimum-gain", type=float, default=5.0)
    parser.add_argument("control", type=pathlib.Path)
    parser.add_argument("candidate", type=pathlib.Path)
    args = parser.parse_args()

    control = load(args.control, args.prompt)
    candidate = load(args.candidate, args.prompt)
    change = 100.0 * (candidate / control - 1.0)
    passed = change >= args.minimum_gain

    print(f"{'test':10} {'control':>10} {'candidate':>10} {'change':>9}  gate")
    print(f"pp{args.prompt:<8} {control:10.2f} {candidate:10.2f} {change:+8.2f}%  {'pass' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        sys.exit(2)
