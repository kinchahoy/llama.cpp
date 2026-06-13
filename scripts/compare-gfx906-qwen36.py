#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import sys


def load_results(directory):
    results = {}
    for path in sorted(pathlib.Path(directory).glob("*.jsonl")):
        name = path.stem
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (name, row["n_prompt"], row["n_gen"])
                results[key] = row["avg_ts"]
    return results


def quant_from_name(name):
    return name.rsplit("__", 1)[-1]


def test_name(n_prompt, n_gen):
    if n_prompt and n_gen:
        return f"pp{n_prompt}+tg{n_gen}"
    if n_prompt:
        return f"pp{n_prompt}"
    return f"tg{n_gen}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control_dir")
    parser.add_argument("candidate_dir")
    args = parser.parse_args()

    control = load_results(args.control_dir)
    candidate = load_results(args.candidate_dir)
    common = sorted(control.keys() & candidate.keys())
    if not common:
        print("No matching benchmark rows found", file=sys.stderr)
        return 2

    missing = sorted(control.keys() ^ candidate.keys())
    if missing:
        print("Control and candidate result sets differ", file=sys.stderr)
        return 2

    quick_mode = any(n_prompt in (512, 2048) and n_gen == 0 for _, n_prompt, n_gen in common)
    if quick_mode:
        names = {
            f"{model}__{quant}"
            for model in ("base", "mtp")
            for quant in ("q4_0", "q4_k_m", "q8_0")
        }
        required = {
            (name, n_prompt, n_gen)
            for name in names
            for n_prompt, n_gen in ((512, 0), (2048, 0), (0, 128))
        }
        absent = required - set(common)
        if absent:
            print(f"Quick result set is incomplete: {len(absent)} rows missing", file=sys.stderr)
            return 2

    min_q8 = float(os.environ.get("MIN_Q8_GAIN", "10"))
    min_q4k = float(os.environ.get("MIN_Q4K_GAIN", "5"))
    max_regression = float(os.environ.get("MAX_REGRESSION", "3"))
    passed = True

    print(f"{'model':42} {'test':16} {'control':>10} {'candidate':>10} {'change':>9}  gate")
    for key in common:
        name, n_prompt, n_gen = key
        before = control[key]
        after = candidate[key]
        change = 100.0 * (after / before - 1.0)
        quant = quant_from_name(name)
        gate = "info"

        if n_gen > 0 and change < -max_regression:
            gate = "FAIL"
            passed = False
        elif n_gen > 0:
            gate = "pass"
        elif quant == "q8_0" and n_prompt in (512, 2048):
            gate = "pass" if change >= min_q8 else "FAIL"
            passed = passed and gate == "pass"
        elif quant == "q4_k_m" and n_prompt in (512, 2048):
            gate = "pass" if change >= min_q4k else "FAIL"
            passed = passed and gate == "pass"
        elif quant == "q4_0" and change < -max_regression:
            gate = "FAIL"
            passed = False
        elif quant == "q4_0":
            gate = "pass"

        print(
            f"{name:42} {test_name(n_prompt, n_gen):16} "
            f"{before:10.2f} {after:10.2f} {change:+8.2f}%  {gate}"
        )

    print("PASS: long benchmark is enabled" if passed else "FAIL: skip the long benchmark")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
