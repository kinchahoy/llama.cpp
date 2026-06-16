# MI50/gfx906 next steps

Date: 2026-06-16

The accepted work is now split into three quant-specific patch files:

1. `gfx906-q8-rocblas-dispatch.patch`
2. `gfx906-q4k-metadata-precompute.patch`
3. `gfx906-q6k-min-blocks-1.patch`

The consolidated package notes are in `gfx906-optimizations.README.md`.

## Current Accepted State

| Quant | Accepted change | Gate |
| --- | --- | --- |
| Q8_0 | selective rocBLAS dispatch for dense gfx906 batches above 256 | pp512 +21% to +25%, pp2048 +53% to +57% |
| Q4_K | gfx906 DP4A metadata precompute with stride9 LDS layout | pp8192 +6.62% over accepted precompute control, pp20000 neutral |
| Q6_K | gfx906 Q6_K-only `min_blocks=1` launch bound | pp8192 +4.00% median |

The Q4_K stride9 source is the current working-tree state. It is the best
validated Q4_K layout even though group4pad1 had better LDS bank-conflict
counters.

## Open Work

1. Run a direct Q6_K scratch-memory trace to confirm the expected 52 B/thread
   spill removal.
2. Re-test Q4_K group4pad1 only if we can control run-order and clock noise
   better. Its counter result is interesting, but throughput was not stable
   enough to accept.
3. Try Q4_K metadata slot permutation on top of stride9. This is the next
   clean bank-conflict experiment because it should not increase LDS footprint.
4. Validate the accepted patch set on the intended dual-MI50 setup after
   preserving single-GPU gates.
5. Validate real MTP speculative decoding with `llama-server`; current
   `llama-bench` runs measure the loaded base-model path, not speculative
   acceptance behavior.

## Do Not Reopen Without New Evidence

| Idea | Reason |
| --- | --- |
| Q4_K `min_blocks=1` | pp8192 -33.74%; Q4_K is LDS-wait bound and needs occupancy |
| Q4_K y64 | pp8192 -9.27%; more tile-launch work dominated |
| Q4_K forced rocBLAS | operator-level -30% to -34% |
| Q4_K stride10/stride11 | quick tests lost at both pp8192 and pp20000 |
| Q4_K group2pad1 | LDS bank conflicts returned to the pre-stride level |
| Q6_K metadata precompute | pp8192 -5.3%; wrong bottleneck |

## Useful Commands

Build exact gfx906 HIP:

```bash
source scripts/setup-therock-env.sh
CCACHE_DISABLE=1 cmake -S . -B build/gfx906-final \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DGGML_VULKAN=ON \
  -DBUILD_SHARED_LIBS=ON
CCACHE_DISABLE=1 cmake --build build/gfx906-final -j --target llama-bench test-backend-ops
```

Correctness:

```bash
source scripts/setup-therock-env.sh
build/gfx906-final/bin/test-backend-ops -b ROCm0 -o MUL_MAT
```

Q4_K gate:

```bash
RUN_ID=q4k-final scripts/run-gfx906-q4k-experiment.sh core
RUN_ID=q4k-final scripts/run-gfx906-q4k-experiment.sh long
RUN_ID=q4k-final scripts/run-gfx906-q4k-experiment.sh short
```
