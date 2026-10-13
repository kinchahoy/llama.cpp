# gfx906 optimization ideas to test: Q8 first, Q4 and Q6 next

Historical record. Earlier recommendations and task deadlines describe the recorded tree; use the [current vr guide](../../README.md) for operating instructions. Exact pre-cleanup bytes are preserved in `before-cleanup.tar.gz`.

Date: 2026-10-11. Scope: two gfx906 GPUs, ROCm, tensor parallelism, Qwen3.8-27B. This is a research shortlist, not a set of measured improvements. No runtime code was changed and no new benchmarks were run for this review.

## Start here

1. Profile the exact Q8 CLI workload. Test producer-side Q8_1 quantization fusion first if separate quantization launches have meaningful cost.
2. Test grouped activation-cache reuse in the repacked prefill path.
3. For Q4_K/Q6_K, investigate the existing dense FFN fusion admission gap before importing canonical MMVQ changes.
4. Test Q6_K tile geometry only after confirming which tensors still use canonical MMQ.
5. Keep DFlash2 vision correctness as a separate open follow-up. It is not fixed by the upstream merge.

There is no demonstrated drop-in 20-40% improvement over this fork in the reviewed material. Candidate order reflects relevance to our active paths, not a promised speedup.

## Workload and existing reference

Use the [current benchmark workflow](../../README.md#benchmark-workflow). This 256-token CLI run is the trusted quick check; confirm improvements with longer generation, such as `-n 4000`. `llama-bench` is a faster screening tool with less weight for the real workload. Explore batch/ubatch 2048/2048 and 4096/4096 as separate comparisons.

```sh
./build-ilp/bin/llama-cli \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on \
  -f flake.nix -st -b 2048 -ub 2048 -c 150000 \
  --spec-type none --spec-draft-n-max 2 \
  --temp 0.0 --top-k 1 -s 42 -n 256 --simple-io
```

The [October 11 merge record](../../MERGE-GFX906.md#2026-10-11-reconciliation) reports 452.2 PP / 29.7 TG tokens/s for this CLI workload. Its separate synthetic depth-0 benchmark reports 506.89 PP2048 / 30.01 TG256 with the preserved gfx906 GDN launch. These are different workloads. The synthetic results are single measurements without warmup; neither establishes a repeatable improvement from any idea below.

`-c 150000` is capacity, not 150000 populated KV positions. `flake.nix` was 7246 bytes when inspected. Deep-context attention and mask changes need an additional test with real populated context. `--spec-type none` disables drafting; draft-depth and draft-head changes do not accelerate this baseline.

Local source inspected at HEAD `c0dae49eec7002af91f3030026418a8fff9ee5bd`, together with the working tree. Before implementing, record the actual source and binary revisions again.

## Ranked experiments

| ID | Experiment | Quants | Main target | Priority / evidence |
| --- | --- | --- | --- | --- |
| A | RMSNorm or gated-unary producer writes Q8_1 | Q8, Q4_K, Q6_K | TG; narrow verify later | First Q8 experiment; transfer needs wave64 adaptation |
| B | Cache grouped MMQ activations in repacked prefill | Q8, Q4_K, Q6_K | PP | Local extension of activation reuse; unmeasured |
| C | Dense repacked K-quant gate+up+GLU fusion | Q4_K, Q6_K | TG | Concrete admission gap; inspect existing kernels first |
| D | Q6_K occupancy/tile sweep | Q6_K | PP | gfx906-specific idea; active dispatch must be verified |
| E | Selective prefill graph bypass | All | PP / first response | Small experiment; TP ownership may already suppress child graphs |
| F | GDN preparation fusions | All | TG / PP | Incremental; profile operation share first |
| G | VDR and short-K row scheduling | Q4_K, Q6_K, Q8 | TG | Exploratory; canonical/repacked and dense/MoE differ |
| H | Derived masks and selective native quantized KV | All weight quants | Memory / deep PP / deep TG | Separate context and KV-format campaign |
| I | Aligned staging for unaligned Q6 upload widths | Q6_K | Load time | Startup optimization, not steady-state inference |

### A. Fuse activation production with Q8_1 quantization

**Source:** [Stew block 08](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/0008-rdna-boosts-block-08-fused-core-prefill-kernels-and-.patch), plus the norm/gated-unary quantization sections of the [delivery notes](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/README.md). The transferable idea is to populate the quantization cache from the producer while retaining the existing matmul consumer.

**Local opportunity:** [norm.cu](../../../ggml/src/ggml-cuda/norm.cu), `ggml_cuda_op_rms_norm_fused`, writes F32. [repack-common.cuh](../../../ggml/src/ggml-cuda/q8_repack/repack-common.cuh), `repack_quantize_src1_q8_1`, later launches quantization on a cache miss. The cache is already enabled by default; adding another general cache is not the task.

**First implementation:** single-token RMSNorm+weight multiplication also writes the exact row-wise Q8_1 cache layout needed by repacked matvec. Preserve the F32 output for other consumers. Extend to gated-unary producers only after the first case pays.

**Tests:** count eliminated launches; measure producer-plus-consumer time and full CLI TG. Cover wave64 reductions, padding, strided inputs, cache invalidation and captured-buffer lifetime. Q4_K requires the activation-sum term as well as scale; a Q8-only shortcut must not silently feed an affine quant consumer. Compare logits and output against unfused execution. Multi-token MoE consumers need their own layout checks.

### B. Reuse grouped activations in repacked prefill

**Source:** activation reuse in [Stew block 08](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/0008-rdna-boosts-block-08-fused-core-prefill-kernels-and-.patch); the proposed grouped-layout extension comes from local code inspection.

**Local opportunity:** the narrow path in [mul-mat.cu](../../../ggml/src/ggml-cuda/q8_repack/mul-mat.cu) calls the shared cache helper. The wide path allocates grouped MMQ workspace and calls `quantize_mmq_q8_1_cuda` per matmul/chunk. Several projections can consume the same activation.

**First implementation:** reuse an identical grouped quantization between two consumers. Key it by layout, input shape/strides, slice/chunk offset and quantization precision. Row-wise and grouped layouts are not interchangeable. Affine K-quants also need the correct grouped sum layout.

**Tests:** PP2048 and exact CLI PP, cache-hit counts, quantizer time, peak VRAM and workspace-pressure fallback at `-c 150000`. Avoid retaining all large prefill activations just to save small launches. Keep only if end-to-end PP improves repeatably.

### C. Add dense Q4_K/Q6_K FFN fusion where it pays

**Source:** the gate+up+GLU fusion strategy in [Stew block 13](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/0013-rdna-boosts-block-13-fused-MoE-gate-up-GLU-MMQ-mmvq-.patch). Adapting it to dense repacked K-quants is a local proposal, not a measured upstream result.

**Local opportunity:** [repack-common.cu](../../../ggml/src/ggml-cuda/q8_repack/repack-common.cu) already admits Q4_K/Q6_K weights for repacking. `ggml_cuda_repack_mmv_fusion_supported` admits only Q8_0/MXFP4 for dense fusion; the MoE predicate admits additional K-quants. [mul-mat.cu](../../../ggml/src/ggml-cuda/q8_repack/mul-mat.cu) already contains generic fused dispatch arms, so inspect and validate those before writing another kernel. The repack README understates current quant coverage; use source predicates as the authority.

**First implementation:** validate one-token dense Q4_K and Q6_K fused dispatch and epilogues, then admit only supported shapes/types. Check both gate and up layouts and quant types. Preserve bias handling and affine activation sums. Do not just widen the admission predicate without proving the full path.

**Tests:** real dense Q4_K_M and Q6_K GGUFs, mixed tensor types, actual TP shard dimensions, CPU/reference operator comparison, logits/PPL and end-to-end TG. Check register pressure and occupancy: fusion saves launches but doubles accumulator work. The local width helper already records a dense three-token regression; multi-token fusion is a separate experiment.

### D. Tune Q6_K tiles for gfx906 occupancy

**Source:** Milpster's [README](https://github.com/milpster/gfx906-llama-cpp#readme) describes a Q6_K I=64 wave64 tile. Its [profiling/experiment record](https://github.com/milpster/gfx906-llama-cpp/blob/master/bench/FINDINGS.md) also documents failed inner-loop changes. [Stew block 04](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/0004-rdna-boosts-block-04-RDNA4-WMMA-flash-attn-Q6_K-mmq-.patch) is a secondary Q6 prefill lead; its full patch could not be fetched in this review.

**Local opportunity:** [mmq-config-gfx906.cuh](../../../ggml/src/ggml-cuda/mmq-config-gfx906.cuh) specializes Q8/MXFP4 and otherwise falls back to the RDNA2 table. Its Q6_K entries use I=128. But supported repacked Q6 tensors can bypass canonical MMQ entirely.

**First experiment:** log/profile dispatch. For canonical Q6 tensors that matter, sweep I=64 versus 128 with appropriate threads and J widths. For repacked tensors, tune the existing generic GEMM's geometry instead. Match the row-sharded TP grid, not a full-row single-GPU benchmark.

**Tests:** representative FFN, attention and output-head shapes, PP2048, deep PP, spill counts and occupancy. Change host/device selectors and required instances together. Keep Q8 behavior unchanged. Do not enable blanket inner-loop unrolling: Milpster records a severe spill-driven regression.

### E. Bypass changing prefill graph capture while retaining decode replay

**Source:** [Stew block 11](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/0011-rdna-boosts-block-11-skip-CUDA-graphs-for-multi-toke.patch) and its [delivery explanation](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/README.md). The direct patch fetch failed; this idea was reviewed from the delivery notes.

**Local opportunity:** [ggml-cuda.cu](../../../ggml/src/ggml-cuda/ggml-cuda.cu) has warmup and unstable-entry handling but no general prefill exclusion in its compatibility check. However, the [meta backend](../../../ggml/src/ggml-backend-meta.cpp) owns whole-token capture and can disable child graphs. Establish whether prefill capture actually happens in this configuration before changing anything.

**Tests:** instrument capture/update counts, then compare cold first response and warm repeated PP separately. Retain TG replay. Detect tokens from actual matmul operands, including `MUL_MAT_ID`; an expert dimension is not a token count. Do not globally disable graphs and attribute the aggregate difference to this idea.

### F. Fuse GDN preparation operations

**Source:** [Stew blocks 08 and 15](https://github.com/stew675/llama-cpp-rdna-boosts/tree/main/patches) and their [release overview](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/README.md): convolution preparation and beta-sigmoid fusion are relevant patterns. Hyperconnection and QSA work belongs to other architectures.

**Local opportunity:** [gated_delta_net.cu](../../../ggml/src/ggml-cuda/gated_delta_net.cu) already has gfx906 one-column/two-warp decode and a chunked path. Start with the beta producer and convolution input copies around this operation, rather than replacing the recurrent kernel with a WMMA implementation.

**Tests:** identify graph patterns that survive TP splitting; fuse one at a time. Run GDN/KDA operator cases, retained-state snapshots, rollback and real-model output checks. Measure the operation's share of step time first. A large kernel speedup on a small share of TG cannot yield a large total speedup.

### G. Explore VDR and short-K row scheduling for Q4/Q6

**Sources:** [Stew block 10](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/0010-rdna-boosts-block-10-k-quant-boosts-Q4_K-Q5_K-Q6_K-Q.patch), [block 13](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/0013-rdna-boosts-block-13-fused-MoE-gate-up-GLU-MMQ-mmvq-.patch), and [numerical invariants](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/GREEDY-PURITY.md).

The inspected block 10 keeps upstream dense Q8/Q4_K/Q5_K VDR and gates wider expert variants to selected RDNA targets. Historical block names do not establish a current dense Q8 gain. Block 13 restores K-split scheduling for long-K and multi-token rows after item-split regressions.

**Local experiment:** for canonical Q4_0/Q4_K/Q6_K rows that remain hot, sweep independent work per lane, warps and rows per block on wave64. For repacked K-quants, use the corresponding generic kernel parameters. Profile short-K projections separately from FFN long-K. Q4_0, Q4_K and IQ4_NL are different formats, not one interchangeable Q4 path.

**Tests:** one-token and widths 2/3/5/8, several K values around 4096, TP row counts and optional MoE routing. Report changed reduction order explicitly. Compare logits/PPL and speculative acceptance where applicable; deterministic output within one build does not prove decode/verify equivalence. Do not apply RDNA wave32/DPP geometry literally to gfx906.

### H. Reduce deep-context attention memory

**Sources:** [Stew block 15](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/0015-rdna-boosts-block-15-campaign-memory-wins.patch), [delivery notes](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/README.md), and [Milpster's measured native-KV exclusions](https://github.com/milpster/gfx906-llama-cpp#readme).

Derived visibility masks can reduce prefill allocations. Quantized KV can reduce storage and reads, but native in-kernel dequantization can lose to staging. Milpster reports such a loss on its gfx906 lane. Neither memory saving guarantees throughput.

**Local experiment:** first measure mask/conversion allocations in the existing [flash-attention code](../../../ggml/src/ggml-cuda/fattn.cu). Prototype derived masks on the non-matrix tile path only for eligible single-sequence prefill, retaining packed fallback. Separately compare F16 KV, F16-K/Q8-V and Q8-K/Q8-V at identical populated depth.

**Tests:** depth 0, 16k and a representative deep context; PP, TG, peak VRAM and numerical quality. Verify GPU dispatch for every mixed KV type. Cover masked/freed cells, M-RoPE, multiple sequences and position shifts. Keep an unsupported mask form from falling back to CPU. Weight quantization and KV quantization are separate axes.

### I. Avoid slow unaligned Q6 model-upload copies

**Source:** [Stew block 13's aligned staging workaround](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/0013-rdna-boosts-block-13-fused-MoE-gate-up-GLU-MMQ-mmvq-.patch).

**Local opportunity:** `ggml_backend_cuda_buffer_set_tensor_2d` in [ggml-cuda.cu](../../../ggml/src/ggml-cuda/ggml-cuda.cu) directly issues H2D 2D copies. A non-four-byte width can take a poor ROCm path. Q6_K block size makes this worth checking on shard uploads.

**First experiment:** trace actual upload widths and time them. If affected, stage an aligned copy and perform the final unaligned D2D copy. Prove source bounds for the last row; rounding a read width upward must not read beyond the mapped allocation.

**Tests:** load time and upload byte identity, including partial/sharded tensors. Repacked upload may avoid this call. Report this as load-time improvement, not PP/TG.

## Screening the rest of Stew's patch set

This is a relevance screen from the linked patch descriptions and selected source hunks, not a full line-by-line review of all patches.

| Blocks | Decision for our workload |
| --- | --- |
| 00 structural fixes | Check correctness overlap before ports; not an independent speedup claim. |
| 01 adaptive MTP | Keep for a separate speculative experiment. Inert with `--spec-type none`. |
| 02 chunked GDN, 04 WMMA attention/MMB portions | gfx906 has no WMMA/MFMA units. Borrow non-matrix preparation/geometry ideas only. |
| 03 BF16 KV | Quality/compatibility option; F16 and BF16 have the same storage size. Not a bandwidth-saving replacement. |
| 05 CPU decode/verify identity | Useful validation ideas, low performance relevance with all layers on GPUs. |
| 06 system operations | Inspect input-upload/staging ideas only if profiling shows transfer waits. Host-expert slabs/caches target a different fit problem. |
| 07 meta-wrapper skip | Our TP meta backend does real sharding and communication. Do not remove it wholesale. |
| 08 producer fusions | Main source for A/B/F. Existing cache/fusions mean only missing pieces count. |
| 09 compute headroom | Memory/fit idea; allocate only what this workload demonstrably needs. |
| 10 VDR | Keep G as a scoped experiment; current dense kernels and arch gates matter. |
| 11 prefill graph policy | E; check existing graph ownership first. |
| 12 hybrid AllReduce | Our normal HIP path already has small-kernel and large-copy strategies. Compare dispatch thresholds only after measuring communication share; do not replace it with RDNA4-specific spin/fence code. |
| 13 MoE fusion/scheduling | C/G/I contain transferable ideas. Dense Qwen does not benefit from expert caching. |
| 14 qwen4exp/QSA | Flash-Next/QSA/hyperconnection work is a separate model campaign. |
| 15 memory and fusions | F/H; retain eligible patterns and fallback semantics. |

Sources: [patch directory](https://github.com/stew675/llama-cpp-rdna-boosts/tree/main/patches), [project overview](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/README.md). The local [merge guide](../../MERGE-GFX906.md#preserve-the-performance-paths) explains the AllReduce and repack paths that must survive.

## DFlash2 vision fix: still open

**Status from local evidence:** [October 11 DFlash2 follow-up](../../MERGE-GFX906.md#dflash2-vision-follow-up). The upstream output-head sharing correction is included. The vision-position fix is not included, and DFlash2 vision is unvalidated.

**Original source:** [z-lab vision-position change, commit 46d78b7fa5de4a6526c9b950a15d48682285f1d7](https://github.com/z-lab/llama.cpp-fork/pull/4/changes/46d78b7fa5de4a6526c9b950a15d48682285f1d7). That page could not be fetched during this review; the description here comes from our existing merge record and local implementation, not a fresh verification of the remote diff.

The intended correction gives injected feature rows consecutive draft-cache positions and starts the next draft block after the draft cache maximum. Repeated vision M-RoPE positions cannot serve as unique positions in that one-dimensional cache.

Our [DFlash implementation](../../../common/speculative.cpp) also defers feature chunks keyed by target positions, skips pinned image chunks, and aligns draft/rollback operations using target positions. The merge notes flag mixed token/embedding batches as another unresolved case. Porting only the old injection-position hunk does not settle these interactions.

**Next work:** separate target feature row identity from draft-cache positions across eager injection, deferred flush, drafting and rollback; inspect mixed-batch handling and pinned-image semantics; validate an actual screenshot prompt plus a text continuation against non-speculative execution. Include text-only regression and rejected-draft rollback tests. This is correctness work before a vision performance claim.

DFlash output-head mirroring and adaptive depth remain separate performance ideas: [Milpster's explanation](https://github.com/milpster/gfx906-llama-cpp#readme) and [Stew block 01](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/0001-rdna-boosts-block-01-adaptive-MTP-draft-depth.patch). They change the benchmark mode and need draft-model VRAM, prefill cost and acceptance measurements. Do not count their gains toward the no-spec baseline.

## Measurement rules for implementation experiments

- Keep one change per experiment and retain an off switch or clean baseline binary. Record model checksum, source/binary revision, build flags, GPU clocks and relevant environment variables.
- Start with the exact Q8 CLI command above. Compare paired runs in alternating order, with repeated measurements. Record PP, TG and time to first generated token; report load time separately.
- Keep `GGML_CUDA_TP_OVERLAP` unset for the normal-path comparison. Verify that the intended host-staged AllReduce and repacked consumers still execute.
- Use a short operator or kernel benchmark to explain a result, then require a repeatable end-to-end improvement. GPU busy-time percentages alone are not wall-clock attribution across concurrent lanes.
- Add Q4_K_M and Q6_K runs with identical prompt, batch, context and decode settings. Record the real tensor-type mix. Use Q4_0/IQ4 models only as separate cases.
- For layout/fusion changes, run affected backend reference cases and inspect logits. Use PPL/KLD when reduction order changes. A 256-token text hash alone is insufficient evidence of quality or rollback correctness.
- Add deep populated context only for ideas whose mechanism depends on depth. Add speculative widths only for verify-path ideas; keep their results separate from `--spec-type none`.
- Large changes or a new subsystem need design discussion before implementation, per AGENTS.md. Prefer extending existing cache, repack and fusion infrastructure.

## Source provenance and limits

External links above point to live `main`/`master`; re-fetch and pin the exact source commit before porting. Retrieval returned inconsistent snapshots: [Stew release.json](https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/release.json) reported `v16-a55e952b8-r49`, while the delivery README reported r46 and contained dates later than this note's date. The directory HTML was older still. These are discovery references, not a verified coherent release snapshot. Exact patch fetches also failed for blocks 04 and 11 and for Milpster's MMQ tables; those candidates are explicitly description-based.

Milpster's headline results use a different quant, topology and workload from our core test. Its README also says it adopted parts of MX's cache/GDN work. Avoid double-counting those changes. The existing packed-byte subtraction substitutions were found locally, so that advertised optimization is not a fresh Q8 opportunity.

Use [Milpster's current overview](https://github.com/milpster/gfx906-llama-cpp#readme) alongside its [experiment history](https://github.com/milpster/gfx906-llama-cpp/blob/master/bench/FINDINGS.md). The history contains abandoned hypotheses and later corrections; do not promote an early projection over the final measured rejection.
