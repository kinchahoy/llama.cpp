#!/usr/bin/env bash
# usage: probe.sh NAME BUILD mtp2|d5 [ENV=VAL ...]   start server verbose, one short request, grep dev path logs
NAME=$1; BUILD=$2; MODE=$3; shift 3
O=/home/raistlin/infer/vrversion-llama.cpp/build-exp-spec/out/probe
mkdir -p $O/prompts
printf 'Write a Python function that parses an ISO 8601 duration string like P3DT4H5M into seconds, with tests.' > $O/prompts/p5.txt
EXTRA="-lv 4" bash /tmp/exp-spec-x/spec.sh "$NAME" "$BUILD" "$O" "$MODE" "$@" > /dev/null 2>&1
grep -E "target h rows|h_nextn|layer input buffer|features via|state_src|failed|extracting" "$O/$NAME.server.log" | head -12
cat "$O/$NAME.txt"
