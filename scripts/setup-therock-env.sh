#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this script so the environment remains active:" >&2
    echo "  source ${BASH_SOURCE[0]}" >&2
    exit 2
fi

if ! command -v rocm-sdk >/dev/null 2>&1; then
    echo "Error: rocm-sdk not found. Activate the TheRock virtual environment first." >&2
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
echo "  ROCM_PATH=$ROCM_PATH"
echo "  CC=$CC"
echo "  CXX=$CXX"
