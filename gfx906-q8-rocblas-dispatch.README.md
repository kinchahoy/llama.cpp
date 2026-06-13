# gfx906 Q8_0 rocBLAS dispatch patch

Patch: `gfx906-q8-rocblas-dispatch.patch`

SHA-256: `98d399c4fd248d262e74b547f3a21619eed20cf3945238e4d6c099298418a12b`

Tested base commit: `d8a24ccee207a1ff24c513fe1c7d3222b3ccd837`

## What it does

On exact Vega 20/gfx906, the patch keeps dense Q8_0 `MUL_MAT` batches through
256 on MMQ and selects the existing rocBLAS path above 256. It does not change
Q4_K, Q8_0 `MUL_MAT_ID`, CUDA, or other AMD architectures.

This split preserves the low-overhead fused path for small batches while using
rocBLAS where a large F16 GEMM repays conversion overhead. A global forced
rocBLAS build is not equivalent and materially regresses Q4_K.

At `m=4096,k=14336,n=512`, the change improved throughput from 8.49 to 10.80
TFLOPS on ROCm0 and from 6.96 to 8.63 TFLOPS on ROCm1. Batch 2048 completed at
11.68 and 10.12 TFLOPS respectively.

## Apply

From another llama.cpp repository root:

```bash
git apply --check /path/to/gfx906-q8-rocblas-dispatch.patch
git apply /path/to/gfx906-q8-rocblas-dispatch.patch
```

To reverse an uncommitted application:

```bash
git apply -R /path/to/gfx906-q8-rocblas-dispatch.patch
```

Build normally. Do not enable `GGML_CUDA_FORCE_CUBLAS`; the patch performs the
type- and batch-specific selection itself.

If `git apply --check` fails on a newer tree, inspect
`ggml_cuda_should_use_mmq()` in `ggml/src/ggml-cuda/mmq.cu`. The condition
belongs after the `GGML_CUDA_FORCE_MMQ` block and before the NVIDIA dispatch
block.
