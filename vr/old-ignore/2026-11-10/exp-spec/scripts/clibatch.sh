#!/usr/bin/env bash
for spec in "$@"; do
  set -- $spec
  bash /tmp/exp-spec-x/cli.sh "$@"
done
