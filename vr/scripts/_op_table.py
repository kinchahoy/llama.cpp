#!/usr/bin/env python3
"""Compare exact-shape test-backend-ops console timing logs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class OpResult:
    time_us: float
    runs: int
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_log")
    parser.add_argument("candidate_log")
    parser.add_argument(
        "--fail-below",
        type=float,
        default=0.0,
        metavar="PCT",
        help="fail when any candidate shape speedup is below PCT (default: 0)",
    )
    return parser.parse_args()


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
PERF_LINE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\((.*)\):\s+"
    r"([0-9]+) runs -\s+([0-9]+(?:\.[0-9]+)?) us/run -"
)
OP_LINE = re.compile(r"^\s*([A-Za-z0-9_]+)\((.*)\):\s+(.*)$")


def compact_description(op_name: str, params: str) -> str:
    output_shape = re.search(
        r"(?:^|,)ne=\[([0-9]+),([0-9]+),([0-9]+),([0-9]+)\]",
        params,
    )
    source_shape = re.search(
        r"(?:^|,)sources=([A-Za-z0-9_]+)"
        r"\[([0-9]+),([0-9]+),([0-9]+),([0-9]+)\]",
        params,
    )
    if output_shape and source_shape:
        return (
            f"{op_name} type_a={source_shape.group(1)} "
            f"m={output_shape.group(1)} n={output_shape.group(2)} "
            f"k={source_shape.group(2)}"
        )

    fields = {}
    for name in ("type_a", "m", "n", "k"):
        match = re.search(rf"(?:^|,){name}=([^,]+)", params)
        if match:
            fields[name] = match.group(1)
    pieces = [op_name]
    pieces.extend(f"{name}={fields[name]}" for name in ("type_a", "m", "n", "k") if name in fields)
    return " ".join(pieces)


def load_log(filename: str) -> dict[tuple[str, str], OpResult]:
    results: dict[tuple[str, str], OpResult] = {}
    with open(filename, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = ANSI_ESCAPE.sub("", raw_line.rstrip())
            match = PERF_LINE.match(line)
            if match:
                key = (match.group(1), match.group(2))
                if key in results:
                    raise ValueError(f"{filename}:{line_number}: duplicate operation")
                time_us = float(match.group(4))
                if time_us <= 0:
                    raise ValueError(
                        f"{filename}:{line_number}: invalid time_us={time_us}"
                    )
                results[key] = OpResult(
                    time_us=time_us,
                    runs=int(match.group(3)),
                    description=compact_description(*key),
                )
                continue

            operation = OP_LINE.match(line)
            if operation and "not supported" in operation.group(3).lower():
                raise ValueError(f"{filename}:{line_number}: unsupported operation")

    return results


def main() -> int:
    args = parse_args()
    try:
        control = load_log(args.control_log)
        candidate = load_log(args.candidate_log)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not control or not candidate:
        print("error: no timed operations found", file=sys.stderr)
        return 2

    missing_candidate = control.keys() - candidate.keys()
    missing_control = candidate.keys() - control.keys()
    if missing_candidate or missing_control:
        print(
            f"error: operation sets differ: missing_candidate={len(missing_candidate)} "
            f"missing_control={len(missing_control)}",
            file=sys.stderr,
        )
        return 2

    description_width = min(
        72, max(28, max(len(result.description) for result in control.values()))
    )
    print(
        f"  {'operation':{description_width}} "
        f"{'control us':>12} {'candidate us':>13} {'speedup':>9}  decision"
    )
    print("  " + "-" * (description_width + 51))

    failed = False
    for key, base in control.items():
        new = candidate[key]
        speedup = (base.time_us / new.time_us - 1.0) * 100.0
        decision = "pass" if speedup >= args.fail_below else "stop"
        failed = failed or decision == "stop"
        description = base.description[:description_width]
        print(
            f"  {description:{description_width}} "
            f"{base.time_us:12.2f} {new.time_us:13.2f} "
            f"{speedup:+8.2f}%  {decision}"
        )

    print(f"  shapes={len(control)} aggregation=none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
