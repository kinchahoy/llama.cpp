# MI50/gfx906 validation handoff

Date: 2026-06-13

This branch contains two independent gfx906 optimizations. The Q8_0 change has
an operator-level measured win. The Q4_K change has strong compiler evidence but
still needs runtime acceptance testing.

## What changed

### Q8_0 large prefill

`ggml/src/ggml-cuda/mmq.cu` keeps dense Q8_0 batches through 256 on MMQ and uses
the existing rocBLAS path above 256 on exact Vega 20. This preserves the small
batch path and avoids the Q4_K regression caused by globally forcing rocBLAS.

At batch 512, the operator gain was 27.2% on ROCm0 and 23.9% on ROCm1.

### Q4_K prefill

`ggml/src/ggml-cuda/mmq.cuh` and `ggml/src/ggml-hip/CMakeLists.txt` use eight
wave64 waves only for the Q4_K MMQ translation unit on exact gfx906. The output
tile remains `128x64`; per-thread accumulators fall from 32 to 16 and compiled
scratch falls from 44 bytes to zero.

This should remove spill traffic while preserving tile reuse, but it is not
accepted until correctness and end-to-end benchmarks pass.

## Progressive test plan

Use `scripts/bench-gfx906-qwen36.sh` with an unchanged `origin/master` build and
a build from this branch. The script uses exact model files from these repos:

- `unsloth/Qwen3.6-27B-MTP-GGUF`
- `unsloth/Qwen3.6-27B-GGUF`

The tested quant files are Q4_0, Q4_K_M, and Q8_0. In this document, Q4_K means
the repository's Q4_K_M file.

The defaults follow `configs/server-models.ini`: all GPU layers, flash
attention, batch and microbatch 2048, and device order `ROCm1/ROCm0`. The slash
separator is required by `llama-bench`; the server config uses commas.

Build both revisions with identical settings:

```bash
source scripts/setup-therock-env.sh
scripts/build-gfx906-comparison.sh
```

`setup-therock-env.sh` contains the TheRock-specific path discovery and exports.
It must be sourced so `CC`, `CXX`, ROCm CMake discovery, and runtime library
paths remain active in the shell.

The script creates a detached, persistent mainline worktree at
`build/.mainline-src`, checked out at `origin/master`. It configures mainline in
`build/mainline` and the current working tree in `build/gfx906-2026-06`, using
the active environment and the same normal Release CMake options. The build
script does not discover or export toolchain paths. It builds `llama-bench`,
`test-backend-ops`, `llama-cli`, and `llama-server`. The worktree and build
directories are retained for incremental rebuilds. Set `MAINLINE_REF` to compare
against another commit. Set `CONFIGURE_ONLY=1` to validate both configurations
without compiling the targets.

Run the quick gate. Each test has one measured repetition:

```bash
scripts/bench-gfx906-qwen36.sh quick
```

This loads each model once per binary and measures `pp32`, `pp512`, `pp2048`,
and `tg128`. The measured `pp32` case is a cheap kernel warmup; the script
disables llama-bench's full-size warmup to avoid duplicating every test. Control
and candidate runs are interleaved per model. Results are written under
`bench-results/gfx906-qwen36/quick`. A concise test description and comparison
are saved to `report.md` in the same directory, including failed gates.

Review or rerun the gate without benchmarking:

```bash
scripts/compare-gfx906-qwen36.py \
  bench-results/gfx906-qwen36/quick/control \
  bench-results/gfx906-qwen36/quick/candidate
```

The default gate requires:

- at least 10% Q8_0 prefill gain at 512 and 2048 tokens;
- at least 5% Q4_K_M prefill gain at 512 and 2048 tokens;
- no Q4_0 prefill regression greater than 3%;
- no generation regression greater than 3% for any quant.

Override thresholds with `MIN_Q8_GAIN`, `MIN_Q4K_GAIN`, and `MAX_REGRESSION`.
Single-run data is intentionally a coarse gate, not publication-quality
statistics.

Only after the quick gate passes, run the long stage:

```bash
scripts/bench-gfx906-qwen36.sh long
```

`long` rechecks the quick gate and exits without running if it fails. It then
runs a cheap `pp32` warmup followed by one combined `pp20000+tg5000` test for
every repo and quant. A combined test avoids separate long prefill passes, but
its reported throughput is aggregate prompt-plus-generation throughput. Use the
quick results to attribute gains to prefill versus generation.

To deliberately run long tests after a failed quick gate:

```bash
scripts/bench-gfx906-qwen36.sh long --force
```

`FORCE_LONG=1` provides the same override. The long report records that the
quick gate was bypassed.

To run both stages in sequence:

```bash
scripts/bench-gfx906-qwen36.sh all
```

Model files are reused from the Hugging Face cache across binaries and stages.
Set `HF_TOKEN` if required and `RESULTS_DIR` to move the output directory.
Continue using the shell where `setup-therock-env.sh` was sourced so the ROCm
runtime libraries remain available. `CONTROL_BIN` and `CANDIDATE_BIN` can still
override the default comparison binaries. Vulkan may print device-discovery
messages at startup, but `-dev ROCm1/ROCm0` explicitly restricts benchmark work
to the two ROCm devices. `llama-bench` uses `/`, not `,`, between device names.

For a short backend comparison on the candidate build:

```bash
scripts/bench-gfx906-rocm-vulkan.sh
```

This runs `pp32`, `pp512`, and `tg128` once per model on `ROCm1/ROCm0` and
`Vulkan2/Vulkan1`. The order skips `Vulkan0`, which is the Intel GPU on this
machine. Results are informational and stored separately under
`bench-results/gfx906-rocm-vulkan/quick`, with the setup and comparison in
`report.md`. Override `VULKAN_DEVICE` if device enumeration changes.

## Important limitation

`llama-bench` measures the loaded base model path and does not configure MTP
speculative decoding. The MTP repository is still useful for checking model
loading and base-model kernel performance, but speculative acceptance rate and
end-to-end MTP speed require a separate `llama-server` workload with
`spec-type=draft-mtp`.

## Acceptance and fallback

Keep the Q8_0 dispatch if correctness passes and Q8_0 prefill improves without
generation regressions.

The Q4_K eight-wave change is rejected: it loses about 9-11% at pp512 and
pp2048. Keep the four-wave mainline geometry while investigating register
pressure and LDS waits. Any replacement should reduce scratch and first pass
the pp8192 core gate before running pp20000. The Q4_K `mmq_y=64` experiment is
also rejected: it loses 9.27% at pp8192.

The active Q4_K experiment keeps the mainline y128/x64/four-wave tile and
changes the gfx906 launch-bound minimum resident blocks from two to one. The
compiled x64 kernel uses 175 VGPRs with zero private segment, replacing the
profiled 128 VGPR plus 44-byte/thread scratch allocation. Its pp8192 result is
pending; pp20000 remains gated on at least a 5% gain.

## Real-shape profiling

The Q4_0 and Q4_K_M `pp512` and `pp2048` traces and counters have been
collected on one MI50. Regenerate the report with:

```bash
scripts/profile-gfx906-q4k-shapes.sh report
```

The generated `gfx906-q4k-profile-diff.md` records dominant `(m,n,k)` MMQ
shapes, calls, time, VGPRs, scratch, workgroup geometry, waves, calculated
dynamic LDS, and shape-matched hardware counters. The main result is that the
dominant Q4_K kernel is about 20% slower per dispatch with 128 VGPRs,
44 bytes/thread of scratch, 65% more LDS instructions, and about 6.1x more LDS
wait instructions than Q4_0.

To recollect either stage:

```bash
source scripts/setup-therock-env.sh
scripts/profile-gfx906-q4k-shapes.sh trace
scripts/profile-gfx906-q4k-shapes.sh counters
```

Use `all` to run both stages or `report` to regenerate the document from
existing CSV files. The default binary is the mainline build so the rejected
eight-wave Q4_K specialization does not distort the baseline.

## Files

- `improvements-mi50-ideas.md`: technical rationale and rejected experiments
- `benchmark-mi50-rtx3090.md`: isolated MI50 and RTX 3090 comparison
- `gfx906-q8-rocblas-dispatch.patch`: portable Q8_0 source patch
- `gfx906-q4k-8wave.patch`: portable Q4_K source patch
- `gfx906-q8-rocblas-dispatch.README.md`: Q8_0 patch application notes


# Other ideas from other sources to evaluate incorporating:
Since reducing occupancy to mitigate register pressure causes execution starvation on the `gfx906` architecture, the optimization strategy must shift toward hiding instruction latencies and reducing memory fetch overhead while maintaining at least two resident blocks per CU.

## 1. Scale Workgroup Geometry (`nwarps`)

The baseline heuristic of `nwarps=4` (256 threads per block) limits instruction-level parallelism on the MI60. Profiling of the `gfx906` architecture within the `llama.cpp` HIP backend indicates that scaling to `nwarps=8` (512 threads) or `nwarps=16` (1024 threads) for matrix multiplication (MMQ) significantly improves throughput. Larger workgroups alter the ratio of shared memory to compute, allowing the hardware scheduler to better hide `s_waitcnt` barriers by multiplexing execution across a larger pool of active wavefronts within the same block.

## 2. Software Pipelining (LDS Double Buffering)

The high concentration of `s_waitcnt` instructions is the result of strict read-after-write dependencies between Local Data Share (LDS) loads and VALU execution. Address this by manually unrolling the inner loop and pipelining the execution:

* Issue the LDS load for iteration $i+1$ into a secondary set of VGPRs.
* Execute the vector arithmetic for iteration $i$.
* Await the completion of the $i+1$ load.

This overlaps the memory latency of the next tile with the arithmetic execution of the current tile. To prevent scratch memory spilling, this requires precise tuning of the unroll factor to stay strictly under the 128 VGPR limit.

## 3. Data Parallel Primitives (DPP) for Warp Reductions

If the kernel utilizes LDS for intra-warp accumulator reductions, replace that mechanism with AMD Data Parallel Primitives (DPP).

* Instructions like `v_add_f32_dpp` or lane permutations via `__shfl_xor` / `_bpermute` allow threads to share register data directly across the SIMD unit without routing through local memory.
* Bypassing LDS for the reduction phase eliminates the final synchronization barriers in the computation loop and reduces the overall LDS instruction count.

## 4. Warp-Cooperative Memory Fetches

The profiling data previously showed a 21% increase in VMEM reads and a degraded TCC hit rate (49%) for `Q4_K`. This indicates that individual threads are fetching fragmented sub-block scales and minimums independently.

* Restructure the load phase to execute cooperatively across the wavefront.
* Utilize vectorized loads (`buffer_load_dwordx4`) to fetch contiguous 16-byte chunks of the super-block metadata from High Bandwidth Memory (HBM).
* Distribute the fetched metadata to the appropriate threads using cross-lane operations (`_bpermute`).

This ensures full utilization of the 64-byte cache lines per memory transaction, resolving the TCC thrashing and minimizing standard memory latency.
