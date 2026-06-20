#!/bin/bash
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═════╝╚═╝     ╚═╝
            ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗
           ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝
           ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗
           ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗
           ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝
            ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚═════╝  ╚═════╝           


EOF

set -e

# 1. Check location
[[ ! -f "CMakeLists.txt" ]] && echo "Error: Not in llama.cpp root directory" && exit 1

# 2. Setup ROCm Environment Variables

if ! command -v rocm-sdk >/dev/null 2>&1; then
    echo "Error: rocm-sdk not found."
    echo "Activate the TheRock virtual environment first."
    exit 1
fi

ROCM_PATH="$(rocm-sdk path --root)"

AMDGPU_ARCH="${AMDGPU_ARCH:-gfx906}"
echo "AMD GPU Architecture: $AMDGPU_ARCH"

export ROCM_PATH
export HIP_PATH="$ROCM_PATH"
export HIP_PLATFORM=amd
export CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# TheRock/uv ROCm may provide amdclang/amdclang++ in .venv/bin rather than
# /opt/rocm/llvm/bin/clang.
if command -v amdclang >/dev/null 2>&1 && command -v amdclang++ >/dev/null 2>&1; then
    export CC="$(command -v amdclang)"
    export CXX="$(command -v amdclang++)"
elif [[ -x "$ROCM_PATH/llvm/bin/clang" && -x "$ROCM_PATH/llvm/bin/clang++" ]]; then
    export CC="$ROCM_PATH/llvm/bin/clang"
    export CXX="$ROCM_PATH/llvm/bin/clang++"
else
    echo "Error: Could not find ROCm clang compiler."
    echo "Tried:"
    echo "  amdclang / amdclang++ on PATH"
    echo "  $ROCM_PATH/llvm/bin/clang"
    echo "  $ROCM_PATH/llvm/bin/clang++"
    exit 1
fi

# Only set HIP_CLANG_PATH if the directory exists.
if [[ -d "$ROCM_PATH/llvm/bin" ]]; then
    export HIP_CLANG_PATH="$ROCM_PATH/llvm/bin"
fi

export PATH="$(rocm-sdk path --bin):$PATH"

echo "ROCM_PATH=$ROCM_PATH"
echo "CC=$CC"
echo "CXX=$CXX"


rm -rf build/specialrocm

# ============================================================================
# CMAKE FLAGS DOCUMENTATION 
# ============================================================================
# 
#  CMAKE BUILD CONFIGURATION ===                                            
#
# CMAKE_BUILD_TYPE                 Build type: Release, Debug, RelWithDebInfo, MinSizeRel
# CMAKE_C_COMPILER                 C compiler path (use ROCm clang for HIP)
# CMAKE_CXX_COMPILER               C++ compiler path (use ROCm clang++ for HIP)
# CMAKE_HIP_ARCHITECTURES          Target GPU arch: gfx906 (MI50/60)
#
#  COMPILER FLAGS ===                                                             
#
# -O3                              Maximum optimization level
# -march=native                    Optimize for host CPU architecture
# -mtune=native                    Tune instruction scheduling for host CPU
# -DNDEBUG                         Disable assert() checks (release mode)
# -Wno-ignored-attributes          Suppress CUDA __host__/__device__ attribute warnings
# -Wno-cuda-compat                 Suppress CUDA compatibility warnings in HIP
# -Wno-unused-result               Suppress unused return value warnings
#
#  GGML GENERAL OPTIONS ===                                                       
#
# GGML_STATIC=OFF                  Static link libraries (ON=static, OFF=shared/dynamic)
# GGML_NATIVE=ON                   Enable CPU-native optimizations (AVX, AVX2, etc)
# GGML_LTO=OFF                     Link Time Optimization (slower build, faster binary)
# GGML_CCACHE=ON                   Use ccache for faster rebuilds if available
# GGML_OPENMP=ON                   Enable OpenMP for CPU parallelization
# GGML_CPU=ON                      Enable CPU backend
# GGML_CPU_HBM=OFF                 Use memkind for High Bandwidth Memory (HBM)
# GGML_CPU_REPACK=ON               Runtime weight conversion Q4_0 -> Q4_X_X
# GGML_BACKEND_DL=OFF              Build backends as dynamic libraries
# GGML_SCHED_NO_REALLOC=OFF        Disable reallocations in ggml-alloc (debug)
#
#  CPU SIMD INSTRUCTION SETS ===                                                  
#
# GGML_SSE42=ON                    Enable SSE 4.2 instructions
# GGML_AVX=ON                      Enable AVX instructions
# GGML_AVX2=ON                     Enable AVX2 instructions
# GGML_AVX_VNNI=OFF                Enable AVX-VNNI (Alder Lake+)
# GGML_AVX512=OFF                  Enable AVX-512F instructions
# GGML_AVX512_VBMI=OFF             Enable AVX-512 VBMI
# GGML_AVX512_VNNI=OFF             Enable AVX-512 VNNI
# GGML_AVX512_BF16=OFF             Enable AVX-512 BF16
# GGML_FMA=ON                      Enable FMA (Fused Multiply-Add)
# GGML_F16C=ON                     Enable F16C (half-float conversions)
# GGML_BMI2=ON                     Enable BMI2 bit manipulation
# GGML_AMX_TILE=OFF                Enable Intel AMX tile instructions
# GGML_AMX_INT8=OFF                Enable Intel AMX INT8
# GGML_AMX_BF16=OFF                Enable Intel AMX BF16
#
#  AMD HIP/ROCm BACKEND ===                                                       
#
# GGML_HIP=ON                      Enable AMD ROCm/HIP backend
# GGML_HIP_GRAPHS=OFF              Use HIP graphs for kernel batching (experimental)
# GGML_HIP_NO_VMM=ON               Disable Virtual Memory Management (required for MI50)
# GGML_HIP_ROCWMMA_FATTN=OFF       Use rocWMMA for Flash Attention (CDNA2+ only)
# GGML_HIP_MMQ_MFMA=ON             Use MFMA matrix instructions for MMQ (CDNA GPUs)
# GGML_HIP_EXPORT_METRICS=OFF      Export kernel performance metrics
#
#  NVIDIA CUDA BACKEND ===                                                        
#
# GGML_CUDA=OFF                    Enable NVIDIA CUDA backend
# GGML_CUDA_FORCE_MMQ=OFF          Force MMQ kernels instead of cuBLAS
# GGML_CUDA_FORCE_CUBLAS=OFF       Force cuBLAS instead of MMQ kernels
# GGML_CUDA_NO_PEER_COPY=OFF       Disable peer-to-peer GPU copies (multi-GPU)
# GGML_CUDA_NO_VMM=OFF             Disable CUDA Virtual Memory Management
# GGML_CUDA_GRAPHS=ON              Use CUDA graphs for kernel batching
#
#  FLASH ATTENTION  ===                                                           
#
# GGML_CUDA_FA=ON                  Enable Flash Attention CUDA/HIP kernels
# GGML_CUDA_FA_ALL_QUANTS=OFF      Compile FA for all quant types (Q4, Q5, Q8, etc)
#                                  ON = slower build, supports all quants
#                                  OFF = faster build, only F16 FA
#
#  OTHER GPU BACKENDS ===                                                         
#
# GGML_VULKAN=OFF                  Enable Vulkan backend (cross-platform GPU)
# GGML_VULKAN_DEBUG=OFF            Enable Vulkan debug output
# GGML_VULKAN_VALIDATE=OFF         Enable Vulkan validation layers
# GGML_METAL=OFF                   Enable Apple Metal backend (macOS/iOS)
# GGML_METAL_EMBED_LIBRARY=ON      Embed Metal shaders in binary
# GGML_SYCL=OFF                    Enable Intel SYCL backend (oneAPI)
# GGML_OPENCL=OFF                  Enable OpenCL backend (Adreno GPUs)
# GGML_MUSA=OFF                    Enable Moore Threads MUSA backend
# GGML_WEBGPU=OFF                  Enable WebGPU backend (browsers)
# GGML_RPC=OFF                     Enable RPC for distributed inference
#
#  OTHER ACCELERATORS ===                                                         
#
# GGML_BLAS=OFF                    Use BLAS library (OpenBLAS, MKL, etc)
# GGML_ACCELERATE=ON               Use Apple Accelerate framework (macOS)
# GGML_LLAMAFILE=ON                Use llamafile SGEMM kernels
# GGML_HEXAGON=OFF                 Enable Qualcomm Hexagon DSP backend
# GGML_ZENDNN=OFF                  Enable AMD ZenDNN for Zen CPUs
# GGML_ZDNN=OFF                    Enable IBM zDNN for Z mainframes
#
#  LLAMA.CPP BUILD TARGETS ===                                                    
#
# LLAMA_BUILD_SERVER=ON            Build llama-server (OpenAI-compatible HTTP API)
# LLAMA_BUILD_EXAMPLES=ON          Build example programs (simple, batched, etc)
# LLAMA_BUILD_TOOLS=ON             Build tools (quantize, bench, perplexity, etc)
# LLAMA_BUILD_TESTS=OFF            Build test suite (slower, for development)
# LLAMA_BUILD_COMMON=ON            Build common utilities library
# LLAMA_TOOLS_INSTALL=ON           Install tools to system
#
#  LLAMA.CPP FEATURES ===                                                         
#
# LLAMA_HTTPLIB=ON                 Use cpp-httplib if curl disabled
# LLAMA_OPENSSL=OFF                Use OpenSSL for HTTPS support
# LLAMA_LLGUIDANCE=OFF             Include LLGuidance for structured output
#
#  DEBUG & SANITIZERS ===                                                         
#
# GGML_ALL_WARNINGS=ON             Enable all compiler warnings
# GGML_FATAL_WARNINGS=OFF          Treat warnings as errors (-Werror)
# GGML_SANITIZE_THREAD=OFF         Enable ThreadSanitizer (race detection)
# GGML_SANITIZE_ADDRESS=OFF        Enable AddressSanitizer (memory errors)
# GGML_SANITIZE_UNDEFINED=OFF      Enable UndefinedBehaviorSanitizer
# GGML_GPROF=OFF                   Enable gprof profiling
#
# ============================================================================

cmake -S . -B build/specialrocm \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_HIP_ARCHITECTURES="$AMDGPU_ARCH" \
    -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
    -DCMAKE_HIP_FLAGS="-Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result" \
    -DGGML_HIP=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_NO_VMM=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DGGML_VULKAN=ON \
    -DBUILD_SHARED_LIBS=ON

cmake --build build/specialrocm -j"$(nproc)"

echo ""
echo "Build complete: ./build/specialrocm/bin/llama-cli, llama-server, llama-bench"
