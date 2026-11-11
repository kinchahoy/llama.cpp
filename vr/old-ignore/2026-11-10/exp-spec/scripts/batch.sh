#!/usr/bin/env bash
# usage: batch.sh "NAME BUILD MODE [ENV=VAL]" ...   runs spec.sh for each spec string
O=/home/raistlin/infer/vrversion-llama.cpp/build-exp-spec/out/spec
for spec in "$@"; do
  set -- $spec
  NAME=$1; B=$2; MODE=$3; shift 3
  bash /tmp/exp-spec-x/spec.sh "$NAME" "$B" "$O" "$MODE" "$@"
done
