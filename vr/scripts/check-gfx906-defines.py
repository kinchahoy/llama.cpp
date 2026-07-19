#!/usr/bin/env python3
"""Verify the exact source-local gfx906 compile-definition profile."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys


BRANCHLESS = "GGML_CUDA_MMVQ_Q4K_GFX906_BRANCHLESS_SCALES"
Q4_PP = "GGML_CUDA_MMQ_Q4K_GFX906_PRECOMPUTE"
Q6_PP = "GGML_CUDA_MMQ_Q6K_GFX906_MIN_BLOCKS_1"

SOURCES = (
    "mmvq.cu",
    "mmq-instance-q4_k.cu",
    "mmq-instance-q6_k.cu",
)

PROFILES = {
    "none": {
        "mmvq.cu": set(),
        "mmq-instance-q4_k.cu": set(),
        "mmq-instance-q6_k.cu": set(),
    },
    "common": {
        "mmvq.cu": {BRANCHLESS},
        "mmq-instance-q4_k.cu": set(),
        "mmq-instance-q6_k.cu": set(),
    },
    "q4": {
        "mmvq.cu": {BRANCHLESS},
        "mmq-instance-q4_k.cu": {Q4_PP},
        "mmq-instance-q6_k.cu": set(),
    },
    "q6": {
        "mmvq.cu": {BRANCHLESS},
        "mmq-instance-q4_k.cu": set(),
        "mmq-instance-q6_k.cu": {Q6_PP},
    },
    "combined": {
        "mmvq.cu": {BRANCHLESS},
        "mmq-instance-q4_k.cu": {Q4_PP},
        "mmq-instance-q6_k.cu": {Q6_PP},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir")
    parser.add_argument("profile", choices=sorted(PROFILES))
    return parser.parse_args()


def command_tokens(entry: dict) -> list[str]:
    if "arguments" in entry:
        return list(entry["arguments"])
    return shlex.split(entry["command"])


def definitions(entry: dict) -> set[str]:
    found = set()
    tokens = command_tokens(entry)
    for index, token in enumerate(tokens):
        value = ""
        if token == "-D" and index + 1 < len(tokens):
            value = tokens[index + 1]
        elif token.startswith("-D"):
            value = token[2:]
        if value:
            found.add(value.split("=", 1)[0])
    return found


def main() -> int:
    args = parse_args()
    filename = os.path.join(args.build_dir, "compile_commands.json")
    try:
        with open(filename, encoding="utf-8") as handle:
            commands = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {filename}: {exc}", file=sys.stderr)
        return 2

    actual = {source: set() for source in SOURCES}
    entries = {source: [] for source in SOURCES}
    seen = {source: 0 for source in SOURCES}
    all_gfx906: dict[str, set[str]] = {}

    for entry in commands:
        source = os.path.basename(entry["file"])
        source_defines = {
            define
            for define in definitions(entry)
            if define.startswith("GGML_CUDA_") and "GFX906" in define
        }
        if source_defines:
            all_gfx906.setdefault(source, set()).update(source_defines)
        if source in actual:
            seen[source] += 1
            actual[source].update(source_defines)
            entries[source].append(source_defines)

    errors = []
    expected = PROFILES[args.profile]
    for source in SOURCES:
        if not seen[source]:
            errors.append(f"missing compile command for {source}")
        if actual[source] != expected[source]:
            errors.append(
                f"{source}: expected={sorted(expected[source])} "
                f"actual={sorted(actual[source])}"
            )
        for index, source_defines in enumerate(entries[source], 1):
            if source_defines != expected[source]:
                errors.append(
                    f"{source} entry {index}: expected={sorted(expected[source])} "
                    f"actual={sorted(source_defines)}"
                )

    expected_location = {
        define: source
        for source, source_defines in expected.items()
        for define in source_defines
    }
    for source, source_defines in all_gfx906.items():
        for define in source_defines:
            if expected_location.get(define) != source:
                errors.append(
                    f"{source}: unexpected gfx906 definition or location: {define}"
                )

    print(f"profile={args.profile} compile_commands={filename}")
    for source in SOURCES:
        values = ",".join(sorted(actual[source])) or "none"
        print(f"  {source}: {values}")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
