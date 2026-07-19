#!/usr/bin/env python3
"""Summarize one-sample llama-bench A/B JSONL output."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    tokens_per_second: float
    sample_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", default=".")
    parser.add_argument(
        "--require-pairs",
        action="store_true",
        help="fail when a row lacks either control or candidate",
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        metavar="PCT",
        help="fail when any paired candidate delta is below PCT",
    )
    return parser.parse_args()


def test_name(row: dict) -> str:
    n_prompt = int(row.get("n_prompt", 0))
    n_gen = int(row.get("n_gen", 0))
    n_depth = int(row.get("n_depth", 0))

    if n_prompt and not n_gen:
        name = f"ipp{n_prompt}" if n_depth else f"pp{n_prompt}"
    elif n_gen and not n_prompt:
        name = f"tg{n_gen}"
    else:
        name = f"pp{n_prompt}+tg{n_gen}"

    if n_depth:
        name += f"@d{n_depth}"
    return name


def load_results(output_dir: str) -> tuple[dict, list, list[str]]:
    data: dict[tuple[str, str], dict[str, Result]] = {}
    order: list[tuple[str, str]] = []
    errors: list[str] = []

    pattern = os.path.join(output_dir, "*", "*.jsonl")
    for filename in sorted(glob.glob(pattern)):
        build = os.path.basename(os.path.dirname(filename))
        config = os.path.basename(filename)[:-6]
        with open(filename, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{filename}:{line_number}: {exc}")
                    continue

                key = (config, test_name(row))
                if key not in data:
                    data[key] = {}
                    order.append(key)
                if build in data[key]:
                    errors.append(
                        f"{filename}:{line_number}: duplicate {build} result for "
                        f"{config} {key[1]}"
                    )
                    continue

                value = row.get("avg_ts")
                if value is None:
                    errors.append(f"{filename}:{line_number}: missing avg_ts")
                    continue
                try:
                    tokens_per_second = float(value)
                except (TypeError, ValueError):
                    errors.append(
                        f"{filename}:{line_number}: invalid avg_ts={value!r}"
                    )
                    continue
                if not math.isfinite(tokens_per_second) or tokens_per_second <= 0:
                    errors.append(
                        f"{filename}:{line_number}: invalid avg_ts={value!r}"
                    )
                    continue
                samples = row.get("samples_ts") or []
                if len(samples) != 1:
                    errors.append(
                        f"{filename}:{line_number}: expected one timed sample, "
                        f"found {len(samples)}"
                    )
                    continue
                data[key][build] = Result(tokens_per_second, len(samples))

    return data, order, errors


def main() -> int:
    args = parse_args()
    data, order, errors = load_results(args.output_dir)

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2

    if not order:
        print("  (no results yet)")
        return 0

    config_width = max(16, min(28, max(len(config) for config, _ in order)))
    print(
        f"  {'config':{config_width}} {'test':20} "
        f"{'control':>11} {'candidate':>11} {'delta':>9}  decision"
    )
    print("  " + "-" * (config_width + 65))

    missing_pair = False
    failed_delta = False
    paired = 0
    for config, test in order:
        builds = data[(config, test)]
        control = builds.get("control")
        candidate = builds.get("candidate")

        control_text = f"{control.tokens_per_second:.2f}" if control else "-"
        candidate_text = f"{candidate.tokens_per_second:.2f}" if candidate else "-"
        if control and candidate and control.tokens_per_second:
            delta = (
                candidate.tokens_per_second / control.tokens_per_second - 1.0
            ) * 100.0
            delta_text = f"{delta:+.2f}%"
            decision = "measured"
            paired += 1
            if args.fail_below is not None:
                if delta < args.fail_below:
                    decision = "stop"
                    failed_delta = True
                else:
                    decision = "pass"
        else:
            delta_text = "-"
            decision = "pending"
            missing_pair = True

        print(
            f"  {config:{config_width}} {test:20} "
            f"{control_text:>11} {candidate_text:>11} "
            f"{delta_text:>9}  {decision}"
        )

    print(f"  paired={paired} pending={sum(1 for key in order if len(data[key]) < 2)}")

    if args.require_pairs and missing_pair:
        return 3
    if failed_delta:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
