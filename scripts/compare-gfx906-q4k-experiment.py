#!/usr/bin/env python3

import argparse
import json
import pathlib
import statistics
import sys


def load(path, prompt, generation):
    samples = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if prompt is not None and row["n_prompt"] == prompt and row["n_gen"] == 0:
                samples.extend(row.get("samples_ts", [row["avg_ts"]]))
            elif generation is not None and row["n_prompt"] == 0 and row["n_gen"] == generation:
                samples.extend(row.get("samples_ts", [row["avg_ts"]]))
    if not samples:
        test = f"pp{prompt}" if prompt is not None else f"tg{generation}"
        raise RuntimeError(f"{test} result not found in {path}")
    return statistics.median(samples), len(samples)


def main():
    parser = argparse.ArgumentParser()
    test = parser.add_mutually_exclusive_group(required=True)
    test.add_argument("--prompt", type=int)
    test.add_argument("--generation", type=int)
    parser.add_argument("--minimum-gain", type=float, default=5.0)
    parser.add_argument("control", type=pathlib.Path)
    parser.add_argument("candidate", type=pathlib.Path)
    args = parser.parse_args()

    control, control_count = load(args.control, args.prompt, args.generation)
    candidate, candidate_count = load(args.candidate, args.prompt, args.generation)
    change = 100.0 * (candidate / control - 1.0)
    passed = change >= args.minimum_gain

    print(f"{'test':10} {'control':>10} {'candidate':>10} {'change':>9} {'samples':>9}  gate")
    label = f"pp{args.prompt}" if args.prompt is not None else f"tg{args.generation}"
    print(f"{label:<10} {control:10.2f} {candidate:10.2f} {change:+8.2f}% "
          f"{control_count}/{candidate_count:>3}  {'pass' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, RuntimeError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        sys.exit(2)
