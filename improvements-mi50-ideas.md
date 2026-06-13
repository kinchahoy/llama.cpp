# gfx906 Q4_K and Q8_0 improvement ideas

Current operational handoff and validation order: `whatsnext.md`.

## Scope

This note targets the two gfx906 devices on this server, using the current tree at
`d8a24ccee` and the TheRock ROCm installation. The priority is prompt processing
(prefill), while preserving small-batch decode performance and compatibility with
the older wave64, DP4A-capable gfx906 architecture.

The recommendations are deliberately narrow. gfx906 has DP4A and packed FP16 dot
instructions, but it is not CDNA and has no MFMA accumulator-register path. Any
change that treats gfx906 as gfx908/CDNA is incorrect.

## Measured baseline

Shape: `m=4096`, `k=14336`, `n=512`, F32 activations.

| Type | Path | GPU 0 TFLOPS | GPU 1 TFLOPS | Change vs MMQ |
| --- | --- | ---: | ---: | ---: |
| Q8_0 | current DP4A MMQ | 8.52 | 6.94 | baseline |
| Q8_0 | dequantize to F16 + rocBLAS | 10.85 | 8.68 | +27.3%, +25.1% |
| Q4_K | current DP4A MMQ | 15.91 | 11.90 | baseline |
| Q4_K | dequantize to F16 + rocBLAS | 10.52 | 8.35 | -33.9%, -29.8% |

The forced-rocBLAS build used `GGML_CUDA_FORCE_CUBLAS=ON`. At `n=1`, the results
were effectively unchanged because the small-batch MMVQ path is selected before
the MMQ/rocBLAS decision.

## Why Q8_0 and Q4_K react differently

The rocBLAS fallback expands both weight formats to F16, converts the F32
activation matrix to F16, runs an F16 GEMM, and converts the result to F32. Once
expanded, both formats present rocBLAS with the same dense matrix dimensions.

Q8_0 benefits because its current fused DP4A MMQ path is relatively inefficient
at this large batch size:

- Q8_0 moves twice as many quantized weight bytes as a 4-bit format.
- Its DP4A MMQ tile has a larger LDS footprint than Q4_K.
- gfx906 uses generic AMD tile choices: `mmq_y=128`, four wave64 wavefronts, and
  an `mmq_x` chosen primarily to minimize the number of tiles rather than measured
  gfx906 occupancy.
- rocBLAS gets a large, regular F16 GEMM with enough work to amortize conversion
  and launch overhead.

Q4_K is the opposite. The fused MMQ kernel retains the compressed weight layout,
unpacks values and scales near the DP4A operations, and avoids writing and reading
a full F16 weight matrix. The fallback discards that advantage and pays roughly a
4x weight expansion before doing the same dense F16 GEMM.

The forced-rocBLAS Q8_0 and Q4_K times on GPU 0 are close: 5.54 ms and 5.72 ms.
Therefore the more complicated Q4_K dequantizer is only a small part of the Q4_K
loss. Even eliminating its conversion cost would not close the approximately
1.94 ms gap to the 3.78 ms fused Q4_K MMQ result.

Conclusion: use a type-specific rocBLAS cutoff for Q8_0, but improve the fused
DP4A MMQ path for Q4_K. A global gfx906 rocBLAS switch is wrong.

## Profiler findings

`rocprofv3 --kernel-trace --stats` provides a more useful split than the total
graph time:

| Path | Main kernel | Main kernel share | VGPRs | Scratch/thread | Q8_1 quantize share |
| --- | --- | ---: | ---: | ---: | ---: |
| Q4_K MMQ | `mul_mat_q<type=12,x=64>` | 97.05% | 128 | 44 bytes | 2.95% |
| Q8_0 MMQ | `mul_mat_q<type=8,x=64>` | 98.16% | 104 | 0 | 1.84% |

The Q4_K kernel is spilling registers. The trace reports 44 bytes of scratch per
thread, or about 11 KiB of scratch traffic allocation per 256-thread workgroup.
This makes register-pressure reduction the first Q4_K kernel experiment. Adding
large register staging arrays before removing the spill is likely to make the
kernel worse.

Disassembly of the exact `x=64,need_check=false` gfx906 kernels supports the
profiler result:

| Kernel | Scalar LDS reads | 128-bit LDS reads | Private buffer loads | Private buffer stores |
| --- | ---: | ---: | ---: | ---: |
| Q4_K | 16 `ds_read_b32` | 0 | 10 | 10 |
| Q8_0 | 20 `ds_read_b32` | 0 | 0 | 0 |

The Q4_K `buffer_load_dword` and `buffer_store_dword` instructions use the
kernel's private-segment descriptor and correspond to the reported scratch
allocation. The absence of `ds_read_b128` in both exact kernels also confirms
that vectorized LDS reads are a real opportunity, but removing Q4_K private
memory traffic remains the prerequisite.

The dynamic LDS allocation calculated by `mmq_get_nbytes_shared()` is:

| Type | `mmq_y` | `mmq_x` | LDS/workgroup | Accumulators/thread |
| --- | ---: | ---: | ---: | ---: |
| Q4_K | 128 | 64 | 28.3 KiB | 32 floats |
| Q8_0 | 128 | 64 | 45.9 KiB | 32 floats |
| Q4_K | 64 | 64 | 18.8 KiB | 16 floats |
| Q8_0 | 64 | 64 | 27.6 KiB | 16 floats |

On a 64 KiB LDS CU, the current Q8_0 configuration permits only one workgroup by
LDS capacity. A gfx906 `mmq_y=64` variant permits two. For Q4_K, `mmq_y=64` can
permit three workgroups by LDS capacity and halves the accumulator array, which
should reduce VGPR pressure and may eliminate the observed spill. More row tiles
will be launched, so this must be measured rather than assumed to win.

The forced-rocBLAS profile confirms the dense GEMM is almost the same for both
types:

| Stage | Q8_0 | Q4_K |
| --- | ---: | ---: |
| F16 rocBLAS GEMM | 5.17 ms, 93.2% | 5.06 ms, 88.7% |
| Weight dequantization | 0.244 ms, 4.4% | 0.519 ms, 9.1% |
| F32 activation to F16 | 0.100 ms, 1.8% | 0.099 ms, 1.7% |
| F16 result to F32 | 0.030 ms, 0.5% | 0.030 ms, 0.5% |

Q4_K dequantization is about 0.275 ms slower than Q8_0 dequantization, but the
fused Q4_K MMQ path remains roughly 1.9 ms faster than the complete rocBLAS path.
Optimizing only the Q4_K fallback dequantizer cannot recover the fused-MMQ lead.

## Priority 1: type-specific Q8_0 dispatch

Relevant function: `ggml_cuda_should_use_mmq()` in
`ggml/src/ggml-cuda/mmq.cu`.

The final generic rule currently keeps all non-CDNA AMD devices on MMQ for every
batch size. The measurements show that this is too broad for Q8_0 on gfx906.

Action:

1. Add a gfx906-specific Q8_0 large-batch decision in
   `ggml_cuda_should_use_mmq()`.
2. Keep MMVQ/MMQ for decode and small batches.
3. Determine the cutoff from a sweep, not from the single `n=512` point. Test at
   least `n=32,64,128,256,512,1024` over several common `(m,k)` shapes.
4. Do not change Q4_K dispatch based on the current data.

The current exact-shape `test-backend-ops` case for `m=4096,k=14336` only
instantiates batch sizes `1,2,3,4,5,8,512`. Its `-p` filter selects existing
cases; it does not synthesize the missing `32,64,128,256,1024` cases. A local,
human-reviewed benchmark-only extension to `tests/test-backend-ops.cpp`, or an
equivalent model-level benchmark, is required for the full cutoff sweep.

Initial hypothesis: Q8_0 should switch to rocBLAS somewhere between `n=128` and
`n=512`, but the exact cutoff needs measurement on both cards and with real model
shapes.

This is the lowest-risk improvement because it uses existing paths and already
shows a 25-27% operator-level gain at `n=512`.

## Priority 2: remove Q4_K spills with a smaller gfx906 tile

Relevant functions in `ggml/src/ggml-cuda/mmq.cuh`:

- `get_mmq_y_host()` and `get_mmq_y_device()`
- `mul_mat_q_case()`
- `launch_mul_mat_q()`

Action:

1. Add a private experimental gfx906-only `mmq_y=64` selection. Host and device
   values must agree; do not change only one helper.
2. Keep `mmq_x=64` for the first comparison so the experiment isolates the Y
   tile and accumulator reduction.
3. Rebuild with resource reporting and confirm that Q4_K scratch becomes zero.
4. Benchmark both `n=1` and `n=512` on both cards. The `n=1` operation normally
   uses MMVQ, but verify dispatch rather than relying on that assumption.
5. If `mmq_y=64,x=64` removes spills but loses performance due to extra row
   tiles, test `mmq_y=128,x=32`. That also halves the accumulator array while
   preserving the current number of row tiles.
6. Only after those two points, test the cross product of Y size, X size, and
   wavefront count.

The primary success criterion for this experiment is zero scratch with a lower
kernel duration. Lower VGPR count without lower runtime is not enough.

## Priority 3: vectorize Q4_K LDS reads

Relevant functions:

- `vec_dot_q4_K_q8_1_dp4a()` in `ggml/src/ggml-cuda/mmq.cuh`
- `vec_dot_q4_K_q8_1_impl_mmq()` in `ggml/src/ggml-cuda/vecdotq.cuh`

The Q4_K inner loop passes pointers into LDS to a helper that consumes eight Q4
words and sixteen contiguous Q8_1 words per dot group. The compiler can emit many
scalar LDS reads from these pointer accesses.

Action:

1. Do this only after the baseline Q4_K kernel reports zero scratch. Follow
   upstream commit `66c4f9ded` (`ds_read_b128` for Q4_0/Q4_1), using
   `ggml_cuda_memcpy_1<ggml_cuda_get_max_cpy_bytes()>` rather than raw `int4 *`
   casts.
2. First stage the sixteen contiguous Q8_1 words into registers using four
   16-byte copies, then pass the register array to the dot helper.
3. Separately test staging the eight Q4 words with two 16-byte copies. Do not
   combine both changes initially because the added VGPR pressure may offset the
   LDS improvement.
4. Inspect generated gfx906 ISA and kernel resource reports. The desired change
   is fewer scalar `ds_read_b32` operations and more `ds_read_b128`, without
   spills or a loss of resident workgroups.

Risk: Q4_K already performs well, and staging 24 words can consume enough VGPRs
to reduce occupancy or reintroduce scratch. Treat Q8_1-only staging as the first
experiment and reject it if scratch returns.

## Priority 4: tune gfx906 MMQ tiles by quant type

Current generic choices in `ggml/src/ggml-cuda/mmq.cuh` are:

- wave size: 64
- wavefronts per block: `256 / 64 = 4`
- `mmq_y`: 128 for gfx906
- maximum `mmq_x`: 64
- selected `mmq_x`: the value that minimizes X tile count while fitting LDS

These settings are shared with newer AMD families and are not selected from a
gfx906 performance model.

Action:

1. Add an experimental compile-time gfx906 table, not runtime branches in the
   hot kernel.
2. Sweep Q4_K combinations such as:

   | `mmq_y` | wavefronts | `mmq_x` candidates |
   | ---: | ---: | --- |
   | 64 | 2 | 16, 32, 48, 64 |
   | 64 | 4 | 16, 32, 48, 64 |
   | 128 | 2 | 16, 32, 48, 64 |
   | 128 | 4 | 16, 32, 48, 64 |

3. Measure VGPRs, SGPRs, LDS bytes, spills, and achieved occupancy for every
   variant. `GGML_HIP_EXPORT_METRICS=ON` enables resource reports and saved
   compiler intermediates.
4. Select by quant type. Q8_0 has a larger X tile than Q4_K and may need a
   different `mmq_y` even if Q4_K remains at 128.
5. Validate boundary shapes where `m` is not divisible by `mmq_y`; the
   `need_check` path must remain correct and should not serialize LDS writes.

Do not assume two wavefronts is better. Fewer wavefronts reduce per-block
resources but also reduce cooperative loading and latency hiding.

## Profile result: activation quantization is not the first target

The MMQ path first quantizes the F32 activation matrix to Q8_1 with
`quantize_mmq_q8_1_cuda()`. `test-backend-ops` reports the whole graph time, so
the current number combines activation quantization and matrix multiplication.

The trace shows activation quantization is 2.95% of Q4_K time and 1.84% of Q8_0
time for this shape. Even eliminating it entirely would not meet the desired 10%
operator gain. It remains a model-graph optimization rather than a first kernel
target.

Action:

1. Reuse a prequantized activation only where graph ownership proves the same
   activation feeds multiple MMQ operations. Avoid a global cache with unclear
   lifetime or synchronization.

For model graphs with multiple projections from the same normalized activation,
fusing or reusing Q8_1 quantization can save work without changing Q4_K math.
This is more promising than repeatedly dequantizing weights to F16.

## Priority 5: optional gfx906-friendly Q4_K device layout

This is a larger experiment for a private fork after the lower-risk work.

Q4_K stores packed 6-bit scales and minima. `load_tiles_q4_K()` unpacks them each
time a weight tile is loaded. A device-only prepacked representation could:

- retain 4-bit weight values;
- expand the 12 packed scale/minimum bytes into directly indexed bytes or
  precombined half values;
- align Q4 values and Q8_1 dot groups for 16-byte loads;
- be produced once when weights are uploaded, not during every inference.

This trades modest VRAM growth for less scale unpacking and simpler addressing.
It must remain a backend-private representation rather than a new GGUF type.

Reject the idea if the profile shows Q4_K is dominated by DP4A or activation
traffic rather than scale unpacking. The format conversion and extra VRAM are not
justified without at least a 5-10% model-level prefill gain.

## Why not use rocBLAS INT8 for Q4_K

An INT8 GEMM does not directly represent Q4_K. Q4_K has block-local scales and
minima that vary along K. Expanding the nibbles to INT8 still requires applying
those block parameters during accumulation. Splitting K into many independently
scaled GEMMs would add excessive launches and reductions. A fused custom kernel
is the appropriate mechanism on gfx906, and that is what MMQ already is.

## Audit of ML-gfx906 and its linked source fork

### `mixa3607/ML-gfx906`

Repository: <https://github.com/mixa3607/ML-gfx906/tree/master/llama.cpp>

The default llama.cpp image does not contain gfx906 llama.cpp kernel patches. It
clones upstream llama.cpp and builds it for `gfx906`. Its useful work is in the
base ROCm image:

- rebuilding rocBLAS and Tensile with gfx906 enabled after upstream packages
  dropped that target;
- rebuilding RCCL for gfx906;
- packaging matching Tensile assets.

That work is operationally important. A missing or mismatched Tensile library can
make the rocBLAS path fail or perform poorly. It does not, by itself, optimize the
Q4_K or Q8_0 fused MMQ kernels.

The repository also has presets that point to separate forks. Those claims must
be evaluated against the referenced source fork, not the container repository.

### `arte-fact/llamacpp-gfx-906-turbo`

Repository: <https://github.com/arte-fact/llamacpp-gfx-906-turbo>

This fork contains real gfx906 source changes, but it should not be merged or
cherry-picked wholesale into the current tree.

Potentially useful ideas:

- vectorized LDS loads;
- Q8_0 weight-load staging;
- next-iteration cache warming;
- gfx906-specific MMVQ kernels for Q4_0, Q4_1, and Q8_0;
- fixing out-of-bounds tile loads so threads write unique LDS slots rather than
  clamping multiple writers to one slot.

Concerns requiring proof:

- The fork does not provide `test-backend-ops` before/after results for the Q4_K
  prefill shape used here.
- Its README says Q4_K remains on the generic path; most advertised MMVQ work is
  for Q4_0, Q4_1, and Q8_0 decode, not Q4_K prefill.
- Its `GFX906_MMQ_NWARPS` value is presented as two, but the active
  `mmq_get_nwarps_*()` helpers still compute four wavefronts for wave64. The
  advertised setting appears ineffective.
- The Q8_0 load staging adds large per-thread register arrays. It may improve
  memory-level parallelism or may reduce occupancy; compiler resource data is
  required.
- Its cache-warming code issues speculative global loads and keeps dummy values
  live. That is not true double buffering and can add traffic and VGPR pressure.
- Its X-tile prefetch computes a `const char *` address as
  `x + (offset_x + kb0_next) + row * stride_row_x`. In the surrounding MMQ code,
  `offset_x`, `kb0`, and `stride_row_x` are quant-block indices passed to
  type-specific loaders, not documented byte offsets. This arithmetic appears
  unit-inconsistent for a byte pointer and may warm unrelated addresses. Verify
  the intended units and generated address before copying this prefetch code.
- Some vector loads use raw `int4 *` casts. The newer upstream
  `ggml_cuda_memcpy_1` pattern is safer for alignment and code generation.
- The fork combines many unrelated TurboQuant, attention, graph-fusion, and
  kernel changes on an older upstream base, making attribution difficult.

Verdict: the ROCm/Tensile packaging is credible and useful. The source fork is a
catalog of experiments, not evidence that its MMQ changes are faster or correct
for this current tree. Port one idea at a time and require operator-level
correctness and performance evidence.

## Validation matrix

For every experiment:

1. Run `test-backend-ops test` for the modified operation and quant types on both
   gfx906 devices.
2. Benchmark at least five repetitions and report median plus spread.
3. Cover `n=1` for decode regression and `n=32,64,128,256,512,1024` for dispatch
   and prefill behavior.
4. Cover several real dimensions, including the current
   `m=4096,k=14336` case and smaller attention/MLP matrices.
5. Run `llama-bench` with representative Q4_K_M and Q8_0 models using prompt
   tests such as `pp512` and `pp2048`, plus `tg128`.
6. Test each card independently before testing tensor split. GPU 1 has a lower
   power cap and different PCIe topology, so a result that only helps GPU 0 is
   not sufficient for the pair.

Suggested acceptance gates:

- no correctness failures;
- no more than 2% decode regression;
- at least 10% Q4_K operator gain at `n=512` on both cards before accepting a
  specialized kernel change;
- model-level prompt throughput gain that survives repeated runs;
- no regression for non-gfx906 HIP targets or CUDA builds.

## Reproducible commands

The runtime library path below is required for this TheRock installation:

```bash
export THEROCK_ROOT=/home/raistlin/amd-clean/therock-venv/.venv/lib/python3.14/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$THEROCK_ROOT/lib"
```

Baseline MMQ measurement:

```bash
./build/specialrocm/bin/test-backend-ops perf \
  -b ROCm0 -o MUL_MAT \
  -p 'type_a=(q8_0|q4_K),type_b=f32,m=4096,n=(1|512),k=14336'
```

The comparison build was configured with the existing HIP toolchain plus:

```bash
-DGGML_CUDA_FORCE_CUBLAS=ON
```

The equivalent forced-rocBLAS measurement is:

```bash
./build/specialrocm-cublas/bin/test-backend-ops perf \
  -b ROCm0 -o MUL_MAT \
  -p 'type_a=(q8_0|q4_K),type_b=f32,m=4096,n=(1|512),k=14336'
```

Kernel timing and resource trace for Q4_K:

```bash
rocprofv3 --kernel-trace --stats \
  -d /tmp/mi50-q4k-prof -o q4k -f csv -- \
  env LD_LIBRARY_PATH="$THEROCK_ROOT/lib" \
  ./build/specialrocm/bin/test-backend-ops perf \
    -b ROCm0 -o MUL_MAT \
    -p 'type_a=q4_K,type_b=f32,m=4096,n=512,k=14336'
```

Important CSV columns are `VGPR_Count`, `Scratch_Size`, `LDS_Block_Size`, and
the per-kernel durations. This rocprofiler build prints repeated
`register fat binary failed` diagnostics for unrelated code objects, but it still
records the active HIP kernels and produces valid timing/resource rows. Keep this
tool limitation in the benchmark log.

To inspect the generated gfx906 ISA, extract `.hip_fatbin` from the relevant
template-instance object with `llvm-objcopy`, unbundle the
`hipv4-amdgcn-amd-amdhsa--gfx906` target with `clang-offload-bundler`, and use
`llvm-objdump -d --mcpu=gfx906`. Count instructions only inside the exact kernel
symbol being benchmarked; each template-instance object contains several tile
and boundary-check variants.

For compiler resource reports and saved intermediates, configure a dedicated
build with:

```bash
-DGGML_HIP_EXPORT_METRICS=ON
```

Do not compare a metrics build directly against the release build without first
checking that the extra compiler flags have not changed code generation.

## Recommended order

1. Implement and sweep the Q8_0-only rocBLAS dispatch cutoff.
2. Test Q4_K `mmq_y=64,x=64` and require zero scratch.
3. If needed, test Q4_K `mmq_y=128,x=32` as the alternate spill-reduction point.
4. Sweep gfx906 Q4_K tile geometry and wavefront count around the better point.
5. Test Q4_K Q8_1 LDS vector loads only after eliminating spills.
6. Only then test prepacked scales or speculative prefetching.

This sequence preserves the already strong Q4_K fused path while taking the
measured Q8_0 win immediately and keeping risky gfx906 specialization isolated.

## Implemented result: Q8_0 large-batch gfx906 dispatch

Status: implemented in `ggml/src/ggml-cuda/mmq.cu`, built, correctness-tested,
profiled, and benchmarked on both gfx906 devices.

The final rule is intentionally narrow:

```cpp
if (cc == GGML_CUDA_CC_VEGA20 && type == GGML_TYPE_Q8_0 && n_experts == 0) {
    return ne11 <= 256;
}
```

This keeps MMQ for Q8_0 batches through 256 and selects the existing
dequantize-to-F16 plus rocBLAS path above 256. It applies only to ordinary dense
`MUL_MAT`; `MUL_MAT_ID` and other expert operations retain the existing MMQ
dispatch because no MoE evidence was collected. Exact `GGML_CUDA_CC_VEGA20`
matching limits the change to gfx906 rather than all GCN hardware.

The condition is after `GGML_CUDA_FORCE_MMQ`, so the explicit build override
still wins, and after the supported-type check, so it does not alter unsupported
format handling.

### Isolated build method

The repository's `.git` directory was read-only, so `git worktree add` could not
create its administrative entry. An equivalent detached source snapshot was
created without copying build output:

```bash
mkdir -p /tmp/llama-q8-gfx906
git archive HEAD | tar -x -C /tmp/llama-q8-gfx906
```

The four-line dispatch patch was applied there and configured in
`/tmp/llama-q8-gfx906-build`. Only HIP, tests, and their required libraries were
built; Vulkan, server, examples, and tools were disabled. TheRock requires its
CMake package path explicitly:

```bash
export ROCM_PATH="$(rocm-sdk path --root)"
export CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)"
export LD_LIBRARY_PATH="$ROCM_PATH/lib"
export CCACHE_DIR=/tmp/ccache

cmake -S /tmp/llama-q8-gfx906 -B /tmp/llama-q8-gfx906-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(command -v amdclang)" \
  -DCMAKE_CXX_COMPILER="$(command -v amdclang++)" \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_C_FLAGS='-O3 -march=native -mtune=native -DNDEBUG' \
  -DCMAKE_CXX_FLAGS='-O3 -march=native -mtune=native -DNDEBUG' \
  -DCMAKE_HIP_FLAGS='-Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result' \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_VULKAN=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DBUILD_SHARED_LIBS=ON

cmake --build /tmp/llama-q8-gfx906-build -j"$(nproc)" \
  --target test-backend-ops
```

The initial build attempt failed before compilation because ccache's default
directory was read-only. Setting `CCACHE_DIR=/tmp/ccache` fixed it.

### Correctness result

The exact large performance shape is present only in `make_test_cases_perf()`,
so trying it in `test` mode correctly selected zero cases. Correctness was
instead checked with the evaluation suite's full Q8_0/F32 `MUL_MAT` coverage:

```bash
./test-backend-ops test -b ROCm0 -o MUL_MAT \
  -p 'type_a=q8_0,type_b=f32'
./test-backend-ops test -b ROCm1 -o MUL_MAT \
  -p 'type_a=q8_0,type_b=f32'
```

Both devices passed 44/44 cases. The suite includes
`m=6,n=4096,k=5120`, which exceeds the new threshold and therefore validates
the rocBLAS path against the CPU reference, not only the retained MMQ path.

### Five-run `n=512` result

Shape: `m=4096,k=14336`, Q8_0 weights, F32 activations. Each table value is the
median of five independent `test-backend-ops perf` invocations.

| Device | Control MMQ | New dispatch | Throughput gain |
| --- | ---: | ---: | ---: |
| GPU 0 | 7.082 ms, 8.49 TFLOPS | 5.566 ms, 10.80 TFLOPS | +27.2% |
| GPU 1 | 8.635 ms, 6.96 TFLOPS | 6.969 ms, 8.63 TFLOPS | +23.9% |

GPU 0 samples ranged from 5.555 to 5.573 ms with the new dispatch. GPU 1 ranged
from 6.952 to 6.988 ms. The result is stable and agrees with the earlier global
forced-rocBLAS experiment.

### Cutoff sweep

A temporary benchmark-only change added batch sizes 128, 256, 384, and 1024 to
the exact `m=4096,k=14336` perf case. The same temporary executable was run
against either the control or experimental shared libraries using
`LD_LIBRARY_PATH`; `ldd` confirmed which `libggml-hip.so` was loaded.

| Batch | GPU 0 control | GPU 0 new | GPU 1 control | GPU 1 new |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 8.49 TFLOPS | 8.48 TFLOPS | 4.40 TFLOPS | 4.40 TFLOPS |
| 256 | 8.49 TFLOPS | 8.49 TFLOPS | 5.85 TFLOPS | 5.85 TFLOPS |
| 384 | 8.48 TFLOPS | 10.22 TFLOPS | 6.55 TFLOPS | 8.11 TFLOPS |
| 512 | 8.49 TFLOPS | 10.80 TFLOPS | 6.96 TFLOPS | 8.63 TFLOPS |
| 1024 | 8.48 TFLOPS | 11.28 TFLOPS | 7.68 TFLOPS | 9.58 TFLOPS |
| 2048 | not rerun | 11.68 TFLOPS | not rerun | 10.12 TFLOPS |

The identical 128 and 256 results confirm that those batches remain on MMQ.
The gain is already 20.5% on GPU 0 and 23.8% on GPU 1 at batch 384, supporting
the conservative `ne11 > 256` cutoff.

The dispatch has no upper threshold. A later exact-shape batch-2048 run
completed successfully in 20.584 ms on GPU 0 and 23.774 ms on GPU 1. This used
roughly 240.52 GFLOP per operation and reached 11.68 and 10.12 TFLOPS. The
control MMQ backend was not rerun at 2048 because the already-linear MMQ results
through 1024 and the successful rocBLAS run were sufficient to verify support;
the table does not imply a measured 2048 speedup percentage.

### Path confirmation and guards

`rocprofv3 --kernel-trace --stats` on GPU 0 at batch 512 reported:

| Stage | Average | Share |
| --- | ---: | ---: |
| gfx906 Tensile F16 GEMM | 5.155 ms | 93.24% |
| Q8_0 to F16 dequantization | 0.244 ms | 4.41% |
| F32 activation to F16 | 0.100 ms | 1.81% |
| F16 result to F32 | 0.030 ms | 0.55% |

No `mul_mat_q` kernel appeared, confirming that the dispatch changed rather than
an unrelated MMQ code-generation difference.

Non-target guard measurements were unchanged within noise:

| Case | GPU 0 control/new | GPU 1 control/new |
| --- | ---: | ---: |
| Q8_0, `n=1` | 101.26 / 101.26 us | 112.46 / 111.17 us |
| Q4_K, `n=512` | 3.785 / 3.800 ms | 5.062 / 5.069 ms |

After promoting the patch, `build/specialrocm/bin/test-backend-ops` was rebuilt
and rerun directly. It again passed 44/44 Q8_0 tests per GPU. Its final perf
check reported Q8_0 batch 512 at 10.79 TFLOPS on GPU 0 and 8.65 TFLOPS on GPU 1,
Q8_0 batch 1 at 1.16 and 1.05 TFLOPS, and Q4_K batch 512 at 15.88 and
11.79 TFLOPS. Backend filters must be the exact names `ROCm0` and `ROCm1`;
`-b ROCm` matches neither and silently runs zero selected backends.

The temporary benchmark case additions were not promoted. Only the type- and
architecture-specific dispatch rule belongs in the main source tree.

## Experiment log: Q4_K `mmq_y=64,x=64`

Status: source patch designed, isolated build configuration attempted, no binary
produced, and no performance or correctness result measured. Do not treat this
as evidence that the smaller tile wins.

### Question being tested

The current gfx906 Q4_K `mmq_y=128,x=64` kernel uses 128 VGPRs and 44 bytes of
scratch per thread. Reducing `mmq_y` to 64 halves the per-thread accumulator
array from 32 to 16 floats and reduces calculated dynamic LDS from 28.3 KiB to
18.8 KiB. The experiment asks whether this removes private-memory spills and
lowers kernel time enough to compensate for launching twice as many row tiles.

The intended comparison changes only Q4_K on exact gfx906. Q8_0 and all other
quant types and architectures must remain controls.

### Patch design

The existing `get_mmq_y_host()` and `get_mmq_y_device()` helpers are shared by
all quant types. A global return-value change would alter every HIP MMQ kernel,
making the result impossible to attribute. The experimental patch instead:

1. Adds `ggml_type type` to both tile-height helpers.
2. Returns 64 on the host only for
   `type == GGML_TYPE_Q4_K && cc == GGML_CUDA_CC_VEGA20`.
3. Returns 64 on the device only for
   `type == GGML_TYPE_Q4_K` when compiling `__gfx906__`.
4. Passes `type` through all five call sites: tile processing, the main kernel,
   stream-K fixup, launch geometry/LDS sizing, and host X-tile selection.

Host and device values must agree. Changing only the kernel value would make
grid geometry and shared-memory sizing inconsistent. Changing only the host
value would leave the compiled kernel indexing at 128 rows.

The patch was deliberately not left active after the interrupted experiment
because it had not compiled or passed correctness testing.

Literal helper and call-site change to reapply:

```diff
-static int get_mmq_y_host(const int cc) {
+static int get_mmq_y_host(const ggml_type type, const int cc) {
+    if (type == GGML_TYPE_Q4_K && cc == GGML_CUDA_CC_VEGA20) {
+        return 64;
+    }
+
     return GGML_CUDA_CC_IS_AMD(cc) ? (GGML_CUDA_CC_IS_RDNA1(cc) ? 64 : 128) :
         ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ? 128 : 64);
 }

-static constexpr __device__ int get_mmq_y_device() {
+static constexpr __device__ int get_mmq_y_device(const ggml_type type) {
 #if defined(GGML_USE_HIP)
+#if defined(__gfx906__)
+    if (type == GGML_TYPE_Q4_K) {
+        return 64;
+    }
+#endif // defined(__gfx906__)
```

Replace all three device calls with `get_mmq_y_device(type)` and both host calls
with `get_mmq_y_host(type, cc)`. At this tree revision the call sites are in
`mul_mat_q_process_tile()`, `mul_mat_q()`, `mul_mat_q_stream_k_fixup()`,
`launch_mul_mat_q()`, and `mul_mat_q_case()`.

### Isolated build attempt

The control binary remains `build/specialrocm/bin/test-backend-ops`. The intended
experimental tree is `build/specialrocm-q4ky64`, configured with the same release
compiler flags, `gfx906` target, HIP graphs, VMM setting, Vulkan backend, and
test/tool options as the control.

The first configuration attempt stopped at `find_package(hip)` because this
shell did not have `CMAKE_PREFIX_PATH` set. This was not a source compilation
failure. The required file exists at:

```text
/home/raistlin/amd-clean/therock-venv/.venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib/cmake/hip/hip-config.cmake
```

Resume from a shell with the TheRock virtual environment active:

```bash
source /home/raistlin/amd-clean/therock-venv/.venv/bin/activate

export ROCM_PATH="$(rocm-sdk path --root)"
export HIP_PATH="$ROCM_PATH"
export HIP_PLATFORM=amd
export CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PATH="$(rocm-sdk path --bin):$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S . -B build/specialrocm-q4ky64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(command -v amdclang)" \
  -DCMAKE_CXX_COMPILER="$(command -v amdclang++)" \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DCMAKE_C_FLAGS='-O3 -march=native -mtune=native -DNDEBUG' \
  -DCMAKE_CXX_FLAGS='-O3 -march=native -mtune=native -DNDEBUG' \
  -DCMAKE_HIP_FLAGS='-Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result' \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DGGML_VULKAN=ON \
  -DBUILD_SHARED_LIBS=ON

cmake --build build/specialrocm-q4ky64 -j"$(nproc)" \
  --target test-backend-ops
```

Before configuring, reapply the type-aware patch described above. Keep the
control build untouched.

### Required measurements

First run correctness for Q4_K on each device. Then collect at least five perf
runs per build and device using the same process state and environment:

```bash
for backend in ROCm0 ROCm1; do
  ./build/specialrocm-q4ky64/bin/test-backend-ops test \
    -b "$backend" -o MUL_MAT \
    -p 'type_a=q4_K,type_b=f32,m=4096,n=(1|512),k=14336'

  ./build/specialrocm/bin/test-backend-ops perf \
    -b "$backend" -o MUL_MAT \
    -p 'type_a=q4_K,type_b=f32,m=4096,n=(1|512),k=14336'

  ./build/specialrocm-q4ky64/bin/test-backend-ops perf \
    -b "$backend" -o MUL_MAT \
    -p 'type_a=q4_K,type_b=f32,m=4096,n=(1|512),k=14336'
done
```

Profile the experimental `n=512` kernel with the `rocprofv3` command above.
Record median kernel duration, spread, VGPR count, scratch bytes per thread, LDS
bytes per workgroup, and exact kernel symbol. The primary gate is
`Scratch_Size == 0`; the final gate is lower median runtime on both devices.

Expected structural changes, which still require measurement:

| Property | Control Y=128 | Experiment Y=64 |
| --- | ---: | ---: |
| Accumulators/thread | 32 floats | 16 floats |
| Calculated dynamic LDS | 28.3 KiB | 18.8 KiB |
| Row tiles for `m=4096` | 32 | 64 |
| Current scratch/thread | 44 bytes | unknown |

Reject the variant if scratch remains, `n=512` slows down, correctness fails,
or either GPU regresses materially. If scratch reaches zero but runtime loses,
the next controlled experiment is `mmq_y=128,x=32`, which also halves the
accumulator array without doubling the row-tile count.

## Superseding Q4_K results: use eight waves on gfx906

Status as of 2026-06-13: implemented in the working tree and fully compiled in
an isolated tree. Compiler resource and ISA checks passed. Runtime correctness
and performance tests are still pending because further GPU command approvals
were unavailable after the tile-width sweep.

This section supersedes the interrupted status in the preceding section.

### Baseline diagnosis

For the exact prefill shape used throughout this investigation:

```text
MUL_MAT(type_a=q4_K,type_b=f32,m=4096,n=512,k=14336)
```

the normal gfx906 MMQ choice is `mmq_y=128`, `mmq_x=64`, and four 64-lane
waves per workgroup. The GPU 0 profile showed:

| Property | Four-wave X=64 baseline |
| --- | ---: |
| VGPRs/thread | 128 |
| Private scratch/thread | 44 bytes |
| Reported VGPR spills | 10 |
| Main kernel time | about 3.66 ms |
| Whole operator time | about 3.78 ms |
| Throughput | about 15.89 TFLOPS |

The exact kernel ISA contains ten `buffer_load_dword` and ten
`buffer_store_dword` instructions. This is register spill traffic, not useful
matrix data movement. Q4_K needs more temporary state than simpler quant types
because each dot product combines packed nibbles, per-group scales, per-group
minimum values, Q8_1 scales, and 32 live float accumulators per thread.

### Rejected experiment: Y=64, X=64

Reducing `mmq_y` from 128 to 64 reduced the accumulator array from 32 to 16
floats per thread. The compiled kernel used 76 VGPRs and zero scratch, but it
had to launch twice as many row tiles.

GPU 0 result:

| Variant | Time | Throughput | Result |
| --- | ---: | ---: | --- |
| Y=128, X=64 | about 3.78 ms | 15.89 TFLOPS | baseline |
| Y=64, X=64 | about 4.16 ms | 14.45 TFLOPS | reject |

Eliminating spills was not enough to repay the extra row blocks and repeated
tile setup. This is why simply making the output tile smaller is not the final
solution.

### Tile-width sweep at Y=128

The isolated experiment added an environment-controlled maximum `mmq_x` only
to select already-existing template instances. That selector was not promoted
to the working tree.

Single-run exact-shape results:

| X tile | GPU 0 ms | GPU 0 TFLOPS | GPU 1 ms | GPU 1 TFLOPS |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 3.672 | 16.37 | 4.363 | 13.78 |
| 40 | 3.702 | 16.24 | 3.996 | 15.05 |
| 48 | 3.679 | 16.34 | 4.010 | 15.00 |
| 56 | 4.318 | 13.93 | 5.295 | 11.36 |
| 64 | 3.784 | 15.89 | 5.063 | 11.88 |

GPU 0 repeated `X=40` samples were 3702.71, 3707.54, 3705.09, 3731.56, and
3702.85 us. The median was 3705.09 us, or about 16.23 TFLOPS.

Compiler metadata explains the discontinuity:

| X tile | VGPRs/thread | Scratch/thread | VGPR spills |
| ---: | ---: | ---: | ---: |
| 32 | 123 | 0 bytes | 0 |
| 40 | 127 | 0 bytes | 0 |
| 48 | 128 | 20 bytes | 4 |
| 56 | 128 | 36 bytes | 8 |
| 64 | 128 | 44 bytes | 10 |

`X=40` is the largest spill-free four-wave tile. It is a useful fallback and
was particularly effective on GPU 1, but it leaves output-tile reuse on the
table and only improved GPU 0 by about 2 percent.

### Selected design: Y=128, X=64, eight waves

The better approach retains the original `128x64` output tile and doubles the
workgroup from four to eight wave64 waves. Each thread owns half as many output
accumulators. The total arithmetic per workgroup is unchanged, but register
pressure is distributed across twice as many threads.

The isolated gfx906 object compiled as follows:

| Property | Four waves | Eight waves |
| --- | ---: | ---: |
| Threads/workgroup | 256 | 512 |
| Output tile | 128x64 | 128x64 |
| Accumulators/thread | 32 | 16 |
| VGPRs/thread | 128 | 111 |
| Scratch/thread | 44 bytes | 0 bytes |
| VGPR spills | 10 | 0 |
| Static instruction count/thread | 3319 | 1834 |
| `v_dot4_i32_i8` instructions/thread | 1024 | 512 |
| `ds_read2_b32` instructions/thread | 304 | 160 |

The per-thread instruction count is expected to roughly halve because each
thread computes half the output cells. Doubling the thread count preserves the
aggregate math. The important change is that private buffer load/store traffic
falls to zero.

On gfx906, the baseline is already limited to about two `28.3 KiB` LDS blocks
per CU, giving eight active waves. The eight-wave variant is expected to fit
one block per CU, also giving eight active waves. Therefore this does not rely
on increasing nominal wave occupancy. It removes spills while preserving the
large tile and supplies more independent lanes for cooperative loads.

### Scope of the implementation

The working-tree implementation changes two files:

1. `ggml/src/ggml-hip/CMakeLists.txt` applies a compile definition only to the
   Q4_K MMQ template translation unit.
2. `ggml/src/ggml-cuda/mmq.cuh` selects eight waves only when that definition is
   present and the host/device architecture is exact Vega20/gfx906.

Consequences:

- CUDA builds are unchanged.
- Other quant types are unchanged.
- gfx900 and newer AMD targets are unchanged.
- Q4_K on gfx906 uses the existing MMQ implementation with different launch
  geometry; no quantization format, numerical operation, or output layout is
  changed.
- Autogenerated template-instance files are unchanged.

Do not combine this with the experimental `X=40` cap. Eight waves are intended
to make the normal `X=64` tile spill-free. Keep rocBLAS dispatch disabled for
Q4_K prefill; rocBLAS dequantization lost roughly 30 to 34 percent in the prior
measurements.

### Isolated build verification

The experiment was built in:

```text
/tmp/llama-q8-gfx906
/tmp/llama-q8-gfx906-build
```

The complete target built successfully:

```bash
CCACHE_DIR=/tmp/ccache-q4k \
  cmake --build /tmp/llama-q8-gfx906-build -j8 \
  --target test-backend-ops
```

The executable linked and printed its help text after supplying all three
TheRock library roots. This verifies compilation and linking, not GPU
correctness.

### Required runtime gate

Use the isolated binary before accepting the optimization:

```bash
ROCM_PKGS=/home/raistlin/amd-clean/therock-venv/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH="$ROCM_PKGS/_rocm_sdk_devel/lib:$ROCM_PKGS/_rocm_sdk_libraries/lib:$ROCM_PKGS/_rocm_sdk_core/lib"

for backend in ROCm0 ROCm1; do
  /tmp/llama-q8-gfx906-build/bin/test-backend-ops test \
    -b "$backend" -o MUL_MAT \
    -p 'type_a=q4_K,type_b=f32,m=4096,n=(1|128|256|512|1024|2048),k=14336'

  for run in 1 2 3 4 5; do
    /tmp/llama-q8-gfx906-build/bin/test-backend-ops perf \
      -b "$backend" -o MUL_MAT \
      -p 'type_a=q4_K,type_b=f32,m=4096,n=(128|256|512|1024|2048),k=14336'
  done
done
```

Compare against the unchanged control binary with the same command, process
state, clocks, and library paths. Record medians rather than the best run.

Acceptance criteria:

- all Q4_K correctness cases pass on both GPUs;
- `n=512` improves on both GPUs;
- `n=1024` and `n=2048` do not regress materially;
- decode-sized `n=1` does not regress materially;
- a profiler trace confirms the selected kernel has 512 threads, 111 VGPRs,
  and zero scratch.

If the eight-wave runtime result loses despite the resource improvement, revert
the two-file specialization and use the measured `X=40` four-wave cap as the
conservative fallback. The next kernel-level optimization after a successful
eight-wave result is vectorizing aligned Q4_K LDS reads with
`ggml_cuda_memcpy_1<16>`, following the existing Q4_0/Q4_1 MI50-tested pattern.
