# Performance evidence

Recorded results on two gfx906 GPUs. PP is prompt processing and TG is token generation, in tokens/s. These are historical measurements, not current regression-test results. Compare only matched workloads and runtime builds.

| Date and workload | Setting | PP / TG | Recorded source identity |
| --- | --- | --- | --- |
| Oct 12, synthetic PP2048/TG128 | Qwen3.8-27B Q8_0, plain | about 682 / 36.8 | Exact commit unknown from the summary; local changes were in use |
| Oct 25, `flake.nix`, 4000-output limit | Qwen3.8-27B Q8_0, MTP2 | 549.7 / 56.8 | Reported build `11776`, commit `766f7c5e0` |
| Same workload | MTP3 | 545.2 / 60.0 | Same reported build |
| Same workload | DFlash7 | 543.7 / 59.9 | Same reported build |
| Oct 25, short CLI, 256-output limit | Qwen3.8-27B Q8_0, plain | 558.9 / 35.2 | Reported build `b11929-fdc8e5b65` |
| Oct 25, matched Flash Next long CLI | UD-IQ3_XXS, plain | 396.9 / 37.3 | Same reported build |
| Same Flash Next workload | MTP2, Q8_0 sidecar | 368.5 / 42.9 | Same reported build |
| Oct 25, Flash Next MTP2, short CLI | Grouped MoE off / on, GPU feedback off | 330.3 / 47.65 -> 372.9 / 50.45 | Local experiment on `fdc8e5b65`; source hashes and patch archived |

The Oct 25 27B runs used batch/ubatch 2048, context capacity 100096, and one run per setting. MTP3 and DFlash7 were effectively tied for TG. Flash Next's matched long runs used batch/ubatch 1024/512, Q8_0 KV, a 4092-token prompt and a 4096-output limit; MTP raised TG by 15% while lowering PP by 7%. Model and quantization differences prevent treating the Flash Next/27B comparison as an isolated architecture result.

Full commands and evidence: [27B CLI](old-ignore/2026-10-25/2026-10-25-qwen3.8-27b-flake-cli.md), [Flash Next MTP](old-ignore/2026-10-25/2026-10-25-flash-next-mtp-short-check.md), and [October profiles](old-ignore/2026-10-12/notes/2026-10-12-tp-profile-findings.md). Reported build commits identify the recorded version; they do not guarantee the absence of local source edits or identify dynamically linked libraries by themselves.

## What these results establish

- Speculation helps some single-request workloads; plain decoding was fastest in the recorded three-request mixed workload. See the [three-slot report](old-ignore/2026-10-12/2026-10-12-three-slots.md).
- Allocating a large context is not testing a fully populated context. The Oct 25 27B runs used about 6K prompt-plus-output tokens despite a 100K capacity setting.
- Target-logprob checks support greedy selection within each run. They do not establish general model-quality equivalence, exact plain/speculative output identity, or all rollback behavior.
- DFlash vision, other models/quants, quantized KV, and maximum populated contexts remain outside these checks.

For a new comparison, retain the source and binary/library identities, local model revision, exact command, environment, input, and actual token counts. Keep the 50/50 split fixed and compare an immediate baseline with the candidate. Ask for explicit OK before performance runs; use the [catch-up helper](how-to-catchup.md) for a bounded preview. Older benchmark records and changed workflows belong in [old-ignore](old-ignore/README.md).

[Grouped-MoE evidence](old-ignore/2026-10-25/flashnext-fast/README.md): two runs per variant, same prompt and output/acceptance, about 5.9% mean TG improvement. PP varied substantially. Canonical IQ2_S/IQ4_NL reference cases and repacked IQ4_NL cases passed; this does not establish general quality equivalence. The accepted integration retains the opt-in grouped path only, not GPU feedback.

The [Oct 29 integration check](old-ignore/2026-10-29/grouped-moe-integration/README.md) passed the build, replay and 18 canonical reference cases; the repacked harness did not finish within the runtime limit. No new performance claim is made for that integration.

## 2026-10-29 official-head check

The supplemental Qwen3.8-27B Q8_0 baseline measured **690.256 PP2048 / 36.802 TG128**. An early, unrepaired 418f port measured Flash Next IQ3_XXS grouped MTP2 at **345.09 / 37.68** on the baseline and **247.31 / 40.22** on the candidate, one run each: PP -28.3%, TG +6.7%, with identical output and draft acceptance. Follow-up source review found the port had changed gfx906 MMQ launch traversal. Restoring gfx906 row-first traversal restored PP behavior; that correction is in the active patch set. The initial negative result is retained as history and does not measure the corrected active tree. A subsequent matched MTP2 pair measured 347.97 / 36.77 on the accepted baseline and 351.30 / 39.51 after restoring traversal, with identical output and acceptance; see the [traversal check](old-ignore/2026-10-29/head-merge-validation/terra-port-fix/README.md). The final IQ2_S change was checked separately with MTP off. The build and 18 canonical grouped reference cases passed. [Commands, identities, raw records and investigation](old-ignore/2026-10-29/head-merge-validation/README.md).

## 2026-10-29 Flash Next quant and CPU placement

With the traversal-fixed official-head candidate, IQ3_XXS with GPU experts measured 377.50 PP / 38.94 TG. IQ4_XS with eight CPU MoE layers measured 197.59 / 24.42 with op-offload on and 204.92 / 25.15 with it off. These are one run each with 8 workers pinned to the verified P-core CPUs 0-7. The small on/off difference does not establish a gain; the IQ3/IQ4 comparison changes both quantization and expert placement. All three 100K-context requests fit. [Commands, placements, buffer reports, CPU telemetry, identities and raw logs](old-ignore/2026-10-29/quant-placement-compare/README.md).

One matched IQ3_XXS layer-split check measured 353.76 PP / 30.99 TG versus the tensor-split control at 377.50 / 38.94. It fit, but was 6.3% lower PP and 20.4% lower TG in this single run. This does not support switching the recorded configuration to layer split. [Full record](old-ignore/2026-10-29/quant-placement-compare/layer-check/README.md).

## 2026-10-29 IQ2_S single-token experiment

A gfx906-only, two-row IQ2_S one-token expert MMVQ path passed 3 focused CPU-reference cases and is included in the active patch set. Its gate is narrow: IQ2_S, expert IDs, one output column, and even row count; odd rows use the existing path. One matched 20-thread, MTP-off Flash Next pair with a doubled prompt (4,143 tokens) and 256 generated tokens measured **396.35 / 37.20 PP/TG** for the traversal-fixed baseline and **398.80 / 38.08** for the candidate. Output matched. The +0.62% PP / +2.37% TG in one pair is not repeatability evidence. This path does not select dense Q8_0 and no Q8_0 dense benefit was measured. [Commands, library identities, CPU telemetry, source diff and logs](old-ignore/2026-10-29/iq2s-single-token-experiment/README.md).

## 2026-10-29 Qwen Next MoE roofline and proposed experiments

Scope: the cached **Qwen3.8 Flash Next UD-IQ3_XXS**, architecture `qwen4exp`, used above, not the standard `qwen3next` model or dense 27B. Use the MTP-off, GPU-expert tensor-split result of **38.94 TG = 25.68 ms/token** as the reference. The estimates below use existing evidence and read-only GGUF header inspection; no new performance run was made.

### Bytes per token, not bytes in the model

The [runtime metadata and placement record](old-ignore/2026-10-29/quant-placement-compare/README.md) gives 48 layers, width 2560, 512 routed experts, 10 selected experts/token, expert width 640 and shared-expert width 640. The quant name does not mean every tensor is three bits. Inspection with the repository's `gguf.GGUFReader` of all three cached shards at revision `c8b5954a88c2775c546b92593eda40ea041d3176` gives:

| Tensor category | Stored bytes, decimal GB | Estimated weight bytes/token, decimal GB |
| --- | ---: | ---: |
| 144 routed gate/up/down tensors | 48.627712 | 0.949760 (multiply by 10/512) |
| Output projection, Q6_K, 2560 x 248320 | 0.521472 | 0.521472 |
| All remaining tensors except the two embedding tables | 3.480005 | 3.480005 (count each once) |
| Input token embedding, Q6_K | 0.521472 | One selected row, about 2100 bytes |
| Per-layer n-gram embedding table, IQ4_NL | 28.800138 | 16 selected rows x 160 elements, about 1440 bytes |
| **Active-weight estimate** | | **4.951237 GB/token** |

Reproduce the tally by summing `ReaderTensor.n_bytes`: names containing `_exps.` get a 10/512 multiplier; exclude `token_embd.weight` and `per_layer_token_embd.weight`; count the rest once. This is a canonical-storage estimate, not measured HBM traffic. Repacking/padding, replicated TP tensors, cache reuse, repeated reads, activations, recurrent state, attention/KV and scratch traffic change physical bytes. Counting all remaining tensors once is an approximation to the forward graph. Model allocation totals cannot substitute for this tally.

As a shape cross-check, routed FFNs perform `48 x 10 x 3 x 2560 x 640 = 2.359296B` weight MACs/token, or 4.719 GFLOP/token counting two operations per MAC. Layer 0 uses IQ2_S gate/up and IQ4_NL down: `10 x (2 x 2560 x 640 x 82/256 + 640 x 2560 x 18/32) = 19.712 MB` of expert weights. Most layers use those types, but two tensors use IQ3_S; the table uses actual tensor sizes rather than assuming uniform types. Shared experts, routers, attention/GDN and hyper-connection projections are in the 3.480005 GB remainder. Routed experts are only **19.2%** of this active-byte estimate; the output projection alone is **10.5%**. The lazy 28.8 GB embedding table is not streamed in full at every token.

### Bandwidth ceiling and the missing time

For balanced two-way TP, the optimistic weight-only time is `max(D0/B0, D1/B1)`, where `Di` is each card's active bytes and `Bi` its sustained bandwidth. It reduces to `D/(2B)` only for equal shards and equal speeds. The [historical dense profile](old-ignore/2026-10-12/notes/2026-10-12-tp-profile-findings.md) measured roughly 650-715 GB/s per card for large Q8 matvecs. Using that empirical range as a hypothetical roof for these mixed-quant weights avoids assuming nameplate bandwidth, but does not establish that small IQ kernels can attain it.

| Sustained bandwidth assumption | Ideal time for 4.951237 GB/token | Weight-only TG ceiling |
| --- | ---: | ---: |
| 650 GB/s per GPU, balanced | 3.81 ms | 263 tokens/s |
| 715 GB/s per GPU, balanced | 3.46 ms | 289 tokens/s |
| Recorded end-to-end 38.94 TG | 25.68 ms | About 193 GB/s useful weight bytes, aggregate |

Thus the recorded run achieves only **13.5-14.8%** of this optimistic streaming ceiling. About **21.9-22.2 ms/token** lies beyond the ideal weight-stream time; this is a diagnostic gap, not a measured breakdown or a promise of a 7x speedup. A formal per-kernel roofline is `min(compute throughput, bandwidth x arithmetic intensity)`. Here `2/b` FLOP/byte for a weight-only matvec omits IQ unpack/table lookup instructions, reductions and activation traffic. Decode can hit an instruction/occupancy ceiling before HBM saturates. End-to-end serial dependencies impose another floor.

The best-supported explanation is that MoE reduces bytes far more than it reduces the serial work needed for each token:

- **Many small operations remain.** Each layer still routes tokens, executes selected experts and a shared FFN, mixes hyper-connections and runs attention or GDN. Narrow expert matrices and ten scattered expert selections are less favorable than a large contiguous dense stream. IQ2_S decode/table work and short reductions may dominate their kernels. This is a hypothesis until a Flash Next trace supplies durations and hardware counters.
- **TP pays latency at every dependency.** The cards have no direct P2P path and use host-staged AllReduce across constrained CPU/PCH links. For single-token vectors, synchronization and arrival skew can matter more than PCIe byte throughput. A busy-wait reduction kernel's duration includes waiting for its peer; summing both GPUs' durations double-counts time. The dense profile demonstrates this mechanism, but its kernel counts and milliseconds must not be assigned to Flash Next. The one layer-split run was slower, so eliminating TP collectives by serializing layers is not an established fix.
- **Most active bytes are outside routed experts.** Even making expert weight streaming free removes only about 0.73 ms at the 650 GB/s/card roof. An expert optimization could save much more if IQ decode is currently slow, but that requires measuring expert kernel time. The two-row IQ2_S result (+2.37% TG in one pair) is consistent with a limited end-to-end gain, not proof of its cause.
- **Verification has a different roofline.** The grouped path in `ggml/src/ggml-cuda/flashnext-moe.cuh` requires 2-8 tokens, so it does not accelerate plain one-token decode. MTP amortizes some kernels but needs draft work and reads the union of selected experts across verification tokens. Do not divide this plain-decode byte estimate by speculative output TG or assume every verification token reuses the same experts.

### Experiments proposed 2026-10-29 (not run)

Keep the exact IQ3 revision, GPU experts, 50/50 tensor split, Q8 KV, 1024/512 batch/ubatch, prompt, generated-token count, CPU affinity, source and resolved-library identities fixed. Use MTP off first and alternate baseline/candidate order with at least three unprofiled runs per setting. Record actual populated context, output hashes, clocks and power. Obtain explicit OK for a bounded runtime before performance runs, as required by the workflow above.

1. **Account for the 25.68 ms/token.** Capture a bounded steady TG window with `rocprofv3 --kernel-trace --memory-copy-trace`, separating prefill/warmup from generation. Attribute the critical path to IQ2_S gate/up, IQ4_NL down, other matvecs/output head, routing, GDN/attention, hyper-connections, collectives, copies and host gaps. Report each GPU separately, kernel calls/token, useful bytes/kernel and union-of-active-interval time; compare unprofiled TG to quantify tracing overhead. Verify HIP graph replay rather than assuming it. If kernels are short with large gaps or collective waits, prioritize dependency/fusion work; if expert kernels dominate with low useful GB/s, prioritize their decode/occupancy. Collect HBM/cache counters only if supported on this gfx906 toolchain; canonical bytes/time alone is not physical bandwidth.

2. **Isolate expert efficiency at the actual TP shapes.** Time the existing IQ2_S one-token path against the prior path, using ten selected experts and the real split dimensions: gate/up start from `[2560,640,512]`, down from `[640,2560,512]` before TP partitioning. Include fused gate/up and the actual repacked IQ4_NL down dispatch, not only a canonical surrogate. Use a rotating expert selection as well as repeated IDs across iterations to expose cache effects; remain within valid routing semantics. Record latency, workgroups, register/LDS use and useful GB/s, then run CPU-reference checks and the matched CLI comparison. The key result is absolute ms/token saved across 48 layers. If this category is only a small share in experiment 1, a broad expert rewrite has a low ceiling regardless of its microbenchmark speedup.

3. **Measure expert reuse during MTP2 verification.** After establishing plain decode, compare MTP2 with `GGML_CUDA_FLASHNEXT_MOE=0/1` using the same sidecar and draft placement. Record accepted output tokens/round, draft time, verify time, selected-expert union size per layer and grouped-kernel time. For three verification tokens, ten routes/token touch between 10 and 30 distinct experts/layer: ideal routed-byte reuse ranges from 3x to 1x before other traffic. Test whether the historical roughly 6% grouped gain survives on the current tree with identical output/acceptance. This distinguishes limited expert overlap from slow grouping/decode or a dominant non-expert critical path; report total round time divided by accepted output tokens, not verification kernel throughput as output TG.

## 2026-10-29 Flash Next TG profile followup

A bounded 64-token capture profiled the cached IQ3_XXS `qwen4exp` model with the IQ2_S candidate HIP library (`fe34c2...`), while the earlier 38.94 TG record used `a4666a...`; these are not a direct build repeat. The matched unprofiled run measured 36.71 TG; profiling measured 28.93 TG (-21.2%) in this single pair, so trace timings are coarse attribution evidence, not application throughput. The trace contains 63 steady graph repeats. Across the final 2.178 s decode window, both GPUs had kernel dispatches active for 1.905 s; 273 ms had no GPU dispatch. This is not a full critical-path breakdown.

| Kernel group | ROCm0 dispatch sum | ROCm1 dispatch sum | Two-GPU interval union | Union per repeat |
| --- | ---: | ---: | ---: | ---: |
| Non-expert Q6_K/type-14 and q8_0-repacked matvecs | 377 ms | 401 ms | 628 ms | 9.97 ms |
| TP AllReduce | 327 ms | 65 ms | 336 ms | 5.33 ms |
| IQ2_S, IQ4_NL and IQ3_S expert matvecs | 199 ms | 215 ms | 295 ms | 4.69 ms |
| Float matvecs | 195 ms | 194 ms | 278 ms | 4.41 ms |
| Routing and MoE reduction | 100 ms | 109 ms | 152 ms | 2.41 ms |
| Position/attention, hyperconnections, GDN/SSM | 142 ms | 153 ms | 274 ms | 4.35 ms |

Per-GPU dispatch sums and category unions can overlap across cards and categories, so they cannot be added to infer critical-path time. The AllReduce dispatch durations are strongly imbalanced; its kernel spins for a peer-arrival flag, consistent with rank skew. The trace does not identify why the peer arrives later. Dividing the roofline's canonical 0.949760 GB expert bytes/token evenly across the GPUs by their expert-kernel intervals gives estimated useful bandwidth of 150 GB/s on ROCm0 and 139 GB/s on ROCm1. These are not HBM measurements; cache residency, repacking and padding, and shard imbalance are not measured. No hardware counters were collected, and host timing gaps cannot be attributed without a HIP API trace. The full trace was too costly during model repacking; the successful kernel+copy capture and limits are recorded in [the profile record](old-ignore/2026-10-29/flashnext-tg-profile/README.md). It supports measuring non-expert projections and TP skew alongside expert kernels; it does not establish a general bottleneck ranking or repeatable speedup.

### Future improvement priorities, 2026-10-29

The first target is the repeated small-operation work around non-expert projections. A further tally of the same trace gives the following ROCm1 dispatch sums, divided by 63 steady graph repeats. These are instrumented kernel times, not predicted end-to-end savings.

| Kernel family | Calls/token | ROCm1 time/token |
| --- | ---: | ---: |
| `quantize_q8_1<32>` | 484 | 2.33 ms |
| `scale_f32` | 303 | 1.77 ms |
| Float matvecs | 288 | 3.08 ms |
| Q6_K and Q8_0 repacked matvecs | 496 | 6.36 ms |

The float total includes the main 264-call specialization (2.90 ms) and other float matvec specializations. The repacked total includes the main 249-call Q6_K and 246-call Q8_0 specializations plus the remaining specialization. Reproduce these totals by grouping kernel names in the saved CSV over the decode window in `summarize.py`; count ROCm1 (`Agent 2`) only.

1. **Remove intermediate scale and quantize passes.** Map repeated producer/consumer chains to their tensor shapes, then fold scaling into a producer or consumer and extend existing norm-plus-quantize fusion where the graph permits it. Reuse existing fusion infrastructure. Removing these passes can reduce memory traffic, kernel execution and dependency costs; the trace does not yet establish which chains are eligible or how much time can be recovered.
2. **Tune tiny float projections.** Identify the shapes behind the 264-call float matvec specialization before changing dispatch. Where projections share an input, evaluate combining them; otherwise test a smaller launch configuration for the small shapes. Keep larger matrices on an appropriate existing path. The roughly 11 us average per call suggests launch granularity and occupancy deserve attention, but does not prove launch overhead is the cause.
3. **Reduce late-rank work before rewriting AllReduce.** ROCm0 spends 5.19 ms/token in AllReduce versus 1.03 ms on ROCm1. The difference is consistent with waiting, not necessarily expensive reduction arithmetic. Trace the work immediately preceding each collective on both ranks and test whether accelerating the late rank removes that wait. Keep 50/50 TP fixed during initial comparisons; a skewed split would be a separate experiment. Do not count faster preceding work and its reduced peer wait as independent savings.

A useful engineering target is a **measured 3-5 ms/token end-to-end reduction across these changes**, not a forecast. Applied to the matched 36.71 TG baseline (27.24 ms/token), that would mean approximately 41-45 TG. The 21% tracing slowdown, overlapping intervals and single short baseline prevent deriving this gain from the kernel sums. Validate each candidate with matched unprofiled repetitions, correctness checks and output comparison before retaining it.

Expert kernels remain a secondary target: their ROCm1 dispatch sum is about 3.42 ms/token in this capture, so an expert-only change has a smaller apparent opportunity than the combined projection and small-operation work. Kernel timing is not itself recoverable end-to-end time. The profile's generic `other` group also remains partly unattributed; these priorities are working hypotheses, not a complete bottleneck ranking.
