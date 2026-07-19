#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script so the environment remains active:" >&2
    echo "  source ${BASH_SOURCE[0]}" >&2
    exit 2
fi

THEROCK_VENV="${THEROCK_VENV:-$HOME/amd-clean/therock-venv/.venv}"

if ! command -v rocm-sdk >/dev/null 2>&1; then
    if [[ ! -r "$THEROCK_VENV/bin/activate" ]]; then
        echo "Error: TheRock virtual environment not found at $THEROCK_VENV." >&2
        echo "Set THEROCK_VENV to the virtual environment path and source this script again." >&2
        return 2
    fi

    # shellcheck disable=SC1090
    source "$THEROCK_VENV/bin/activate"
fi

if ! ROCM_SDK_VERSION="$(rocm-sdk version)"; then
    echo "Error: unable to query the active TheRock SDK version." >&2
    return 2
fi

ROCM_PATH="$(rocm-sdk path --root)"
ROCM_PACKAGE_ROOT="$(dirname "$ROCM_PATH")"

if [[ ! -x "$ROCM_PATH/llvm/bin/clang" || ! -x "$ROCM_PATH/llvm/bin/clang++" ]]; then
    echo "Error: TheRock clang and clang++ were not found under $ROCM_PATH/llvm/bin." >&2
    return 2
fi

export ROCM_PATH
export HIP_PATH="$ROCM_PATH"
export HIP_PLATFORM=amd
export HIP_CLANG_PATH="$ROCM_PATH/llvm/bin"
export CC="$ROCM_PATH/llvm/bin/clang"
export CXX="$ROCM_PATH/llvm/bin/clang++"
export PATH="$ROCM_PATH/llvm/bin:$(rocm-sdk path --bin):$PATH"
export CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$ROCM_PACKAGE_ROOT/_rocm_sdk_devel/lib:$ROCM_PACKAGE_ROOT/_rocm_sdk_libraries/lib:$ROCM_PACKAGE_ROOT/_rocm_sdk_core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

unset ROCM_PACKAGE_ROOT

echo "TheRock environment configured:"
echo "  version=$ROCM_SDK_VERSION"
echo "  ROCM_PATH=$ROCM_PATH"
echo "  CC=$CC"
echo "  CXX=$CXX"

unset ROCM_SDK_VERSION THEROCK_VENV
