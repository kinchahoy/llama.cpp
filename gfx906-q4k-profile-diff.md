# gfx906 Q4_0 vs Q4_K real-shape profile

Generated: 2026-06-13T20:14:53.584825+00:00

## Test setup

- Binary: `/home/raistlin/infer/llama.cpp/build/mainline/bin/llama-bench`
- Device: `ROCm0`
- Repo: `unsloth/Qwen3.6-27B-GGUF`
- Contexts: `512,2048`
- Quants: `Q4_0,Q4_K_M`
- Batch Size: `2048`
- Ubatch Size: `2048`
- Repetitions: `1`
- Warmup: `disabled`

Each context and quantization is a separate `llama-bench` process under `rocprofv3`.
The trace pairs each `mul_mat_q` dispatch with the preceding Q8_1 quantization dispatch to recover `(m,n,k)`.

## Dominant MMQ shapes

| Quant | Test | Shape `(m,n,k)` | Calls | Total ms | Share | MMQ X | Workgroup | Waves | VGPR | Scratch B/thread | Dynamic LDS KiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Q4_0 | pp512 | `17408x512x5120` | 128 | 605.853 | 45.4% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_0 | pp512 | `5120x512x17408` | 56 | 271.344 | 20.3% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_0 | pp512 | `10240x512x5120` | 48 | 128.549 | 9.6% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q5_K | pp512 | `5120x512x6144` | 48 | 122.754 | 9.2% | 64 | 64x4 | 4 | 128 | 64 | 44.3 |
| Q4_0 | pp512 | `6144x512x5120` | 48 | 72.820 | 5.5% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_0 | pp512 | `12288x512x5120` | 16 | 51.126 | 3.8% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_1 | pp512 | `5120x512x17408` | 8 | 40.608 | 3.0% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_0 | pp512 | `5120x512x6144` | 16 | 27.103 | 2.0% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_0 | pp512 | `1024x512x5120` | 32 | 14.804 | 1.1% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_0 | pp2048 | `17408x2048x5120` | 128 | 2321.524 | 45.5% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_0 | pp2048 | `5120x2048x17408` | 56 | 1004.103 | 19.7% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_0 | pp2048 | `10240x2048x5120` | 48 | 513.993 | 10.1% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q5_K | pp2048 | `5120x2048x6144` | 48 | 483.309 | 9.5% | 64 | 64x4 | 4 | 128 | 64 | 44.3 |
| Q4_0 | pp2048 | `6144x2048x5120` | 48 | 298.592 | 5.9% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_0 | pp2048 | `12288x2048x5120` | 16 | 205.262 | 4.0% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_1 | pp2048 | `5120x2048x17408` | 8 | 142.116 | 2.8% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_0 | pp2048 | `5120x2048x6144` | 16 | 99.985 | 2.0% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_0 | pp2048 | `1024x2048x5120` | 32 | 32.510 | 0.6% | 64 | 64x4 | 4 | 100 | 0 | 29.9 |
| Q4_K | pp512 | `17408x512x5120` | 128 | 701.754 | 40.8% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q6_K | pp512 | `5120x512x17408` | 32 | 285.657 | 16.6% | 64 | 64x4 | 4 | 128 | 52 | 44.3 |
| Q4_K | pp512 | `5120x512x17408` | 32 | 182.872 | 10.6% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q6_K | pp512 | `10240x512x5120` | 24 | 135.016 | 7.9% | 64 | 64x4 | 4 | 128 | 52 | 44.3 |
| Q5_K | pp512 | `5120x512x6144` | 48 | 122.903 | 7.1% | 64 | 64x4 | 4 | 128 | 64 | 44.3 |
| Q4_K | pp512 | `6144x512x5120` | 48 | 101.221 | 5.9% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q4_K | pp512 | `10240x512x5120` | 24 | 77.946 | 4.5% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q4_K | pp512 | `12288x512x5120` | 16 | 62.077 | 3.6% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q4_K | pp512 | `5120x512x6144` | 16 | 32.564 | 1.9% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q4_K | pp512 | `1024x512x5120` | 24 | 12.519 | 0.7% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q6_K | pp512 | `1024x512x5120` | 8 | 4.413 | 0.3% | 64 | 64x4 | 4 | 128 | 52 | 44.3 |
| Q4_K | pp2048 | `17408x2048x5120` | 128 | 2782.834 | 41.5% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q6_K | pp2048 | `5120x2048x17408` | 32 | 1149.624 | 17.1% | 64 | 64x4 | 4 | 128 | 52 | 44.3 |
| Q4_K | pp2048 | `5120x2048x17408` | 32 | 690.325 | 10.3% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q6_K | pp2048 | `10240x2048x5120` | 24 | 516.560 | 7.7% | 64 | 64x4 | 4 | 128 | 52 | 44.3 |
| Q5_K | pp2048 | `5120x2048x6144` | 48 | 483.335 | 7.2% | 64 | 64x4 | 4 | 128 | 64 | 44.3 |
| Q4_K | pp2048 | `6144x2048x5120` | 48 | 364.212 | 5.4% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q4_K | pp2048 | `10240x2048x5120` | 24 | 305.845 | 4.6% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q4_K | pp2048 | `12288x2048x5120` | 16 | 244.288 | 3.6% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q4_K | pp2048 | `5120x2048x6144` | 16 | 122.335 | 1.8% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q4_K | pp2048 | `1024x2048x5120` | 24 | 30.668 | 0.5% | 64 | 64x4 | 4 | 128 | 44 | 28.3 |
| Q6_K | pp2048 | `1024x2048x5120` | 8 | 16.921 | 0.3% | 64 | 64x4 | 4 | 128 | 52 | 44.3 |

## Matching-shape comparison

This compares average dispatch time because `Q4_K_M` assigns some tensors to Q6_K.

| Test | Shape `(m,n,k)` | Q4_0 calls | Q4_K calls | Q4_0 us/call | Q4_K us/call | Q4_K slower |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pp512 | `1024x512x5120` | 32 | 24 | 462.6 | 521.6 | +12.8% |
| pp512 | `5120x512x6144` | 16 | 16 | 1693.9 | 2035.3 | +20.2% |
| pp512 | `5120x512x17408` | 56 | 32 | 4845.4 | 5714.8 | +17.9% |
| pp512 | `6144x512x5120` | 48 | 48 | 1517.1 | 2108.8 | +39.0% |
| pp512 | `10240x512x5120` | 48 | 24 | 2678.1 | 3247.8 | +21.3% |
| pp512 | `12288x512x5120` | 16 | 16 | 3195.4 | 3879.8 | +21.4% |
| pp512 | `17408x512x5120` | 128 | 128 | 4733.2 | 5482.5 | +15.8% |
| pp2048 | `1024x2048x5120` | 32 | 24 | 1015.9 | 1277.8 | +25.8% |
| pp2048 | `5120x2048x6144` | 16 | 16 | 6249.1 | 7645.9 | +22.4% |
| pp2048 | `5120x2048x17408` | 56 | 32 | 17930.4 | 21572.6 | +20.3% |
| pp2048 | `6144x2048x5120` | 48 | 48 | 6220.7 | 7587.7 | +22.0% |
| pp2048 | `10240x2048x5120` | 48 | 24 | 10708.2 | 12743.5 | +19.0% |
| pp2048 | `12288x2048x5120` | 16 | 16 | 12828.9 | 15268.0 | +19.0% |
| pp2048 | `17408x2048x5120` | 128 | 128 | 18136.9 | 21740.9 | +19.9% |

## Findings

- Q4_K_M MMQ time is 31.5% higher than Q4_0 at pp2048.
- The dominant shape is `17408xN x5120`; it accounts for about 41% of Q4_K_M MMQ time and Q4_K is about 16-20% slower per dispatch there.
- Q4_0 uses 100 VGPRs with no scratch. Q4_K uses 128 VGPRs and 44 bytes/thread of scratch at the same four-wave geometry. This makes register pressure and spill traffic the first hypothesis to test.
- Q4_K uses slightly less dynamic LDS than Q4_0, so LDS capacity alone does not explain the loss. LDS instruction count, bank conflicts, and waits may still matter.
- Q6_K contributes about one quarter of Q4_K_M MMQ time, uses 128 VGPRs, 52 bytes/thread of scratch, and 44.3 KiB of dynamic LDS. It needs its own optimization path rather than being treated as a minor tail.
- At the dominant pp2048 shape, Q4_K executes 65% more LDS instructions, 30% more VALU instructions, 21% more VMEM reads, and 31% more VMEM writes with the same wave count.
- The dominant shape reports no LDS bank conflicts for either quant, but Q4_K has about 6.1x as many LDS wait instructions. The first LDS investigation should target dependency chains and synchronization rather than padding.
- Q4_K has fewer total TCC accesses at the dominant shape but a much lower hit rate, about 49% versus 66%. Raw memory bandwidth is therefore unlikely to be the only limiter.

## Hardware counters

Counters are totals over all MMQ dispatches in each workload.

| Counter | Q4_0 pp512 | Q4_K_M pp512 | Q4_0 pp2048 | Q4_K_M pp2048 | pp2048 change |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SQ_INSTS_LDS` | 9924468736 | 15792911744 | 39697882624 | 63171559424 | +59.1% |
| `SQ_INSTS_VALU` | 81088890880 | 103410336000 | 324355677696 | 413639639040 | +27.5% |
| `SQ_INSTS_VMEM_RD` | 945704960 | 1191824128 | 3782886400 | 4766334976 | +26.0% |
| `SQ_INSTS_VMEM_WR` | 32116736 | 41678208 | 128525824 | 165838848 | +29.0% |
| `SQ_LDS_BANK_CONFLICT` | 94371840 | 353894400 | 377487360 | 1415577600 | +275.0% |
| `SQ_WAIT_INST_LDS` | 962950359 | 4296984875 | 3977104948 | 17407258102 | +337.7% |
| `SQ_WAVES_sum` | 974848 | 976384 | 3899904 | 3899392 | -0.0% |
| `TCC_EA_RDREQ_32B_sum` | 0 | 0 | 0 | 0 | +0.0% |
| `TCC_EA_WRREQ_64B_sum` | 58289455 | 77412005 | 251114054 | 325718755 | +29.7% |
| `TCC_HIT_sum` | 4430422369 | 3330147967 | 17418025173 | 13283924761 | -23.7% |
| `TCC_MISS_sum` | 1675385968 | 2322078125 | 7060229369 | 9368970520 | +32.7% |

At pp2048 the aggregate TCC hit rate falls from 71.2% to 58.6%.

### Dominant pp2048 shape

Counters below are for matching `17408x2048x5120` Q4_0 and Q4_K dispatches only.

| Counter | Q4_0 | Q4_K | Change |
| --- | ---: | ---: | ---: |
| `SQ_INSTS_LDS` | 17774542848 | 29316743168 | +64.9% |
| `SQ_INSTS_VALU` | 149168455680 | 193363050496 | +29.6% |
| `SQ_INSTS_VMEM_RD` | 1693450240 | 2056650752 | +21.4% |
| `SQ_INSTS_VMEM_WR` | 71303168 | 93585408 | +31.2% |
| `SQ_LDS_BANK_CONFLICT` | 0 | 0 | +0.0% |
| `SQ_WAIT_INST_LDS` | 1482535654 | 9009270398 | +507.7% |
| `SQ_WAVES_sum` | 2228224 | 2228224 | +0.0% |
| `TCC_EA_RDREQ_32B_sum` | 0 | 0 | +0.0% |
| `TCC_EA_WRREQ_64B_sum` | 140712185 | 185260962 | +31.7% |
| `TCC_HIT_sum` | 7622903107 | 5045817422 | -33.8% |
| `TCC_MISS_sum` | 3918497216 | 5163979707 | +31.8% |

## Interpretation limits

- `LDS_Block_Size` in the runtime trace reports static LDS and is zero for these kernels; dynamic LDS is calculated from `mmq_get_nbytes_shared()`.
- Hardware counters quantify VALU, LDS, VMEM, cache, waits, conflicts, and waves, but do not distinguish DP4A from unpacking instructions.
- Exact DP4A, unpacking, and LDS instruction counts require saved compiler intermediates or targeted thread-trace/ISA analysis for the dominant kernel variants.
- `TCC_EA_RDREQ_32B_sum` is zero on this stack, so the report does not claim a global-read bandwidth value. TCC hit/miss behavior and VMEM instruction counts are still usable.
- This first pass uses the base Qwen3.6 model on one MI50 to avoid mixing unequal devices. MTP and dual-GPU validation come after dominant shapes are understood.

## Next tests

1. Save and disassemble the Q4_0, Q4_K, and Q6_K `mmq_x=64` kernels. Count DP4A, unpack/scale, LDS, synchronization, and scratch instructions for the dominant shape variants.
2. Reduce Q4_K live ranges or accumulator pressure until scratch reaches zero or drops materially, then run the pp8192 core gate before pp20000.
3. Inspect the Q4_K LDS producer/consumer sequence around scale/minimum unpacking. The dominant-shape wait increase is large even without bank conflicts.
4. Treat Q6_K separately: its larger LDS tile and 52-byte scratch footprint make a shared Q4_K geometry unlikely to be optimal.

## Rejected attack: y64

The first attempted solution was a gfx906-only Q4_K `mmq_y=64` specialization retaining four waves and `mmq_x=64`. Halving `mmq_y` reduced per-thread accumulators but duplicated activation-tile and workgroup overhead.

This is preferable to the rejected eight-wave variant. Both approaches halve the accumulators per thread, but `mmq_y=64` keeps the 256-thread workgroup. It doubles the number of row tiles, but does not duplicate weight loads because each workgroup owns different weight rows. It duplicates the smaller Q8_1 activation tile and adds workgroup/synchronization overhead. Dynamic LDS falls from about 28.3 KiB to 18.8 KiB, allowing more residency if registers permit.

The result rejects this direction. Do not combine y64 with further loop changes. The current attack below restores y128 and isolates spill removal.

For each build, first run one Q4_K_M pp8192 core test. Run pp20000 only after pp8192 improves by at least 5%. Use a matching pp8192 trace to inspect VGPR, scratch, and dominant-kernel time for variants that pass or produce an otherwise informative result.

Apply the same process to Q6_K separately after Q4_K. A Q6_K `mmq_y=64` tile would reduce dynamic LDS from about 44.3 KiB to 26.8 KiB and halve accumulator pressure, but its unpacking and scale path differs enough that it should not share a tuning decision with Q4_K.

The first `y64` pp512 model test was rejected: 187.47 versus 191.80 tokens/s, or -2.26%. The gate correctly skipped pp2048. Preserve this result as evidence that duplicated activation-tile and workgroup overhead outweighs the reduced accumulator pressure at this shape. The next isolated diagnostic is `min-blocks-1`, not a longer y64 run.

For subsequent experiments, use pp8192 as the mandatory core gate and pp20000 as the long test. Only variants gaining at least 5% at pp8192 proceed to pp20000.

The y64 pp8192 core result is a stronger rejection: 221.50 versus 244.13 tokens/s, or -9.27%. This confirms that the extra activation-tile loads, workgroups, and synchronization dominate at long prefill. Do not run y64 at pp20000.

## Current attack: remove spills without changing the tile

The next candidate restores the mainline `mmq_y=128`, `mmq_x=64`, four-wave geometry and changes only the gfx906 Q4_K launch-bound minimum resident blocks from two to one. This preserves weight and activation reuse while allowing a larger register allocation.

Compiler metadata validates the intended mechanism for the dominant `mmq_x=64` specialization: the regular kernel now uses 175 VGPRs with zero private segment and zero reported VGPR spills; the edge-check kernel uses 191 VGPRs with zero private segment. The profiled mainline kernel used 128 VGPRs and 44 bytes/thread of scratch.

The tradeoff is explicit: one 256-thread workgroup per CU instead of up to two. This attack wins only if removing scratch traffic and its dependency stalls is more valuable than the lost latency-hiding occupancy. Run the pp8192 core gate first; do not run pp20000 unless it gains at least 5%.

```bash
source scripts/setup-therock-env.sh
RUN_ID=q4k-min-blocks-1 scripts/run-gfx906-q4k-experiment.sh core
RUN_ID=q4k-min-blocks-1 scripts/run-gfx906-q4k-experiment.sh long
```
