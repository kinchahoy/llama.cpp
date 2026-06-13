# gfx906 Q8_0 rocBLAS dispatch patch

Patch: `gfx906-q8-rocblas-dispatch.patch`

SHA-256: `261647ed76164d91d5ef90ff72d24afd7f08d90449418a3da9e8afca050abbf7`

Tested base commit: `d8a24ccee207a1ff24c513fe1c7d3222b3ccd837`

The patch changes one function in one source file. On exact gfx906 hardware it
keeps dense Q8_0 `MUL_MAT` batches through 256 on MMQ and selects the existing
rocBLAS path above 256. It does not change Q4_K or `MUL_MAT_ID` dispatch.

The rule has no upper batch limit. The tested `m=4096,k=14336,n=2048` case
completed at 11.68 TFLOPS on GPU 0 and 10.12 TFLOPS on GPU 1.

Apply from another llama.cpp repository root:

```bash
git apply --check /path/to/gfx906-q8-rocblas-dispatch.patch
git apply /path/to/gfx906-q8-rocblas-dispatch.patch
```

Revert an uncommitted application:

```bash
git apply -R /path/to/gfx906-q8-rocblas-dispatch.patch
```

Build normally. Do not enable `GGML_CUDA_FORCE_CUBLAS`; the patch performs the
type- and batch-specific selection itself.

If `git apply --check` fails on a newer tree, inspect
`ggml_cuda_should_use_mmq()` in `ggml/src/ggml-cuda/mmq.cu` and place the four
line condition after the `GGML_CUDA_FORCE_MMQ` block and before the NVIDIA
dispatch block.
