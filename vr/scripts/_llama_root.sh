#!/usr/bin/env bash

resolve_llama_root() {
    local root

    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 2
    if [[ -f "$root/CMakeLists.txt" ]]; then
        printf '%s\n' "$root"
        return 0
    fi

    echo "Error: expected this script under llama.cpp/vr/scripts." >&2
    echo "Move the vr directory into the llama.cpp tree you want to build." >&2
    return 2
}
