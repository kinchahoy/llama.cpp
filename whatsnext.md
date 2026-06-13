# MI50/gfx906 Optimization Handoff

Date: 2026-06-13

This file is the concise operational handoff for the dual-gfx906 system. The
full investigation log remains in `improvements-mi50-ideas.md`.

## Current source changes

Two independent optimizations are present in the working tree.

### Q8_0 large-prefill dispatch

File: `ggml/src/ggml-cuda/mmq.cu`

On exact gfx906, dense Q8_0 `MUL_MAT` stays on MMQ through batch 256 and uses
the existing rocBLAS path above 256. `MUL_MAT_ID`, Q4_K, CUDA, and other AMD
architectures are unchanged.

Measured at `m=4096,k=14336,n=512`:

| Device | MMQ control | Selective rocBLAS | Gain |
| --- | ---: | ---: | ---: |
| ROCm0 | 8.49 TFLOPS | 10.80 TFLOPS | 27.2% |
| ROCm1 | 6.96 TFLOPS | 8.63 TFLOPS | 23.9% |

The rocBLAS path also completed batch 2048 at 11.68 TFLOPS on ROCm0 and 10.12
TFLOPS on ROCm1. Do not build with `GGML_CUDA_FORCE_CUBLAS`; selective dispatch
is required because forced rocBLAS materially hurts Q4_K.

Portable artifacts:

- `gfx906-q8-rocblas-dispatch.patch`
- `gfx906-q8-rocblas-dispatch.README.md`

### Q4_K eight-wave MMQ

Files:

- `ggml/src/ggml-cuda/mmq.cuh`
- `ggml/src/ggml-hip/CMakeLists.txt`

The Q4_K MMQ translation unit receives a private compile definition. On exact
gfx906 only, it launches eight wave64 waves instead of four while retaining the
normal `mmq_y=128,mmq_x=64` output tile.

Compiler evidence for the exact batch-512 kernel:

| Property | Four-wave control | Eight-wave candidate |
| --- | ---: | ---: |
| Threads/workgroup | 256 | 512 |
| Accumulators/thread | 32 | 16 |
| VGPRs/thread | 128 | 111 |
| Scratch/thread | 44 bytes | 0 bytes |
| VGPR spills | 10 | 0 |
| Static instructions/thread | 3319 | 1834 |

The complete isolated `test-backend-ops` target compiled and linked. GPU
correctness and runtime performance have not yet been measured for this final
candidate. Treat it as promising but unaccepted until the gate below passes.

Portable artifact:

- `gfx906-q4k-8wave.patch`

## Findings that drive the Q4_K design

The baseline Q4_K prefill kernel is register-pressure limited. Its 32 float
accumulators plus packed-value, scale, minimum, and Q8_1 temporaries push gfx906
to 128 VGPRs and private scratch.

Reducing `mmq_y` to 64 removed spills but slowed ROCm0 from about 15.89 to 14.45
TFLOPS because it doubled row-tile work. Reject that approach.

At `mmq_y=128`, the four-wave tile-width sweep produced:

| X tile | ROCm0 TFLOPS | ROCm1 TFLOPS | Scratch/thread |
| ---: | ---: | ---: | ---: |
| 32 | 16.37 | 13.78 | 0 bytes |
| 40 | 16.24 | 15.05 | 0 bytes |
| 48 | 16.34 | 15.00 | 20 bytes |
| 56 | 13.93 | 11.36 | 36 bytes |
| 64 | 15.89 | 11.88 | 44 bytes |

`X=40` is the largest spill-free four-wave tile and is the fallback if the
eight-wave candidate fails. Do not combine the X=40 cap with eight waves.

Forced rocBLAS is not a Q4_K solution. It pays for dequantization into a dense
matrix and lost roughly 30 to 34 percent versus fused MMQ prefill.

## Immediate validation

The isolated candidate binary is:

```text
/tmp/llama-q8-gfx906-build/bin/test-backend-ops
```

Set the TheRock runtime paths:

```bash
ROCM_PKGS=/home/raistlin/amd-clean/therock-venv/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH="$ROCM_PKGS/_rocm_sdk_devel/lib:$ROCM_PKGS/_rocm_sdk_libraries/lib:$ROCM_PKGS/_rocm_sdk_core/lib"
```

First run correctness on both cards:

```bash
for backend in ROCm0 ROCm1; do
  /tmp/llama-q8-gfx906-build/bin/test-backend-ops test \
    -b "$backend" -o MUL_MAT \
    -p 'type_a=q4_K,type_b=f32,m=4096,n=(1|128|256|512|1024|2048),k=14336'
done
```

Then collect five candidate runs per card:

```bash
for backend in ROCm0 ROCm1; do
  for run in 1 2 3 4 5; do
    /tmp/llama-q8-gfx906-build/bin/test-backend-ops perf \
      -b "$backend" -o MUL_MAT \
      -p 'type_a=q4_K,type_b=f32,m=4096,n=(1|128|256|512|1024|2048),k=14336'
  done
done
```

Run the same command against the unchanged control binary in the same session.
Use medians, not best runs. Keep clocks, thermals, library paths, and other GPU
load constant.

Accept the eight-wave change only if:

1. All correctness cases pass on both GPUs.
2. Batch 512 improves on both GPUs.
3. Batches 1024 and 2048 do not regress by more than 3 percent.
4. Batch 1 does not regress by more than 3 percent.
5. Profiling confirms the selected kernel has 512 threads and zero scratch.

The desired result is at least a 10 percent median batch-512 gain on both cards.
A smaller gain can still be retained privately if it is stable and has no
regressions, but it is not strong evidence for an upstreamable specialization.

## Decision tree

If eight waves pass:

1. Keep `gfx906-q4k-8wave.patch` as the base Q4_K optimization.
2. Rebuild the normal working-tree binary and repeat correctness plus five-run
   medians.
3. Run `llama-bench` with representative Q4_K models at `pp512`, `pp2048`, and
   `tg128` to verify end-to-end impact.
4. Profile aligned LDS reads before attempting another kernel edit.

If eight waves fail correctness, revert the two-file Q4_K change immediately.
The host and device wave counts must always agree.

If eight waves are correct but slower, use the measured X=40 four-wave cap as
the conservative fallback, then validate batch 128 through 2048. Do not infer
that zero spills alone guarantees higher throughput; the rejected Y=64 result
already disproved that.

## Next kernel experiment

After an eight-wave win, the next controlled experiment is vectorized Q4_K LDS
reads. The current Q4_K inner loop still emits scalar `ds_read_b32` operations.
The existing Q4_0/Q4_1 code uses `ggml_cuda_memcpy_1<16>` to induce
`ds_read_b128`, and that pattern was previously tested on MI50.

Constraints:

- change only `vec_dot_q4_K_q8_1_dp4a()`;
- stage a small aligned group at a time to avoid restoring VGPR pressure;
- compile and inspect VGPR/scratch metadata before running benchmarks;
- require zero scratch to remain true;
- compare instruction counts and batch-512 medians on both cards;
- reject the change if either card regresses materially.

Do not pursue a custom Q4_K-to-INT8 rocBLAS path first. It requires format
conversion and loses the fused block-scale/minimum arithmetic that currently
makes MMQ faster.

## Patch handling

Check and apply on another matching llama.cpp tree:

```bash
git apply --check /path/to/gfx906-q8-rocblas-dispatch.patch
git apply /path/to/gfx906-q8-rocblas-dispatch.patch

git apply --check /path/to/gfx906-q4k-8wave.patch
git apply /path/to/gfx906-q4k-8wave.patch
```

The patches are independent. Apply the Q8_0 patch for the measured large-batch
win. Apply the Q4_K patch only after the pending runtime gate passes on the
target machine.

## Source of record

- Detailed investigation: `improvements-mi50-ideas.md`
- MI50 and RTX 3090 comparison: `benchmark-mi50-rtx3090.md`
- Q8_0 portable patch: `gfx906-q8-rocblas-dispatch.patch`
- Q4_K portable patch: `gfx906-q4k-8wave.patch`
