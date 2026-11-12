#!/usr/bin/env bash
# usage: build.sh BUILD_DIR [targets...]   (build dir must already be configured as in vr/README.md)
B=$1; shift
source /home/raistlin/amd-clean/env/therock.sh
cmake --build "$B" --target ${@:-llama-server llama-bench llama-cli} -j 8 2>&1 | grep -E "error|Linking CXX executable" | tail -8
