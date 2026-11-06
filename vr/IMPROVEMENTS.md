# gfx906 improvements

The aim is a small delta from llama.cpp that keeps useful gfx906 performance. Some support code is backend-generic because tensor parallelism, graph replay, and speculative state must agree across the runtime.

| Improvement | Why it helps | Main source |
| --- | --- | --- |
| Repacked weights and gfx906 MMQ/MMVQ tuning | Match weight layout, tile sizes, and lane work to GCN hardware; reduce decode memory and instruction cost | `ggml/src/ggml-cuda/q8_repack/`, `mmq.cuh`, `mmvq.cu` |
| Narrow one-token IQ2_S expert MMVQ | Use a gfx906-specific two-row path for even-row, one-column expert operations; odd rows keep the existing path | `ggml/src/ggml-cuda/mmvq.cu` |
| Graph-safe AllReduce, duplex copies and overlapped prefill slabs | Extend upstream's host-staged transport to reduce communication and scheduling cost | `ggml/src/ggml-cuda/allreduce.cu`, `allreduce.cuh` |
| Concurrent TP lanes and whole-token graph replay | Reduce host launch gaps and keep both GPUs working | `ggml/src/ggml-backend-meta.cpp`, `ggml-backend.cpp` |
| Fused normalization, activation production, and narrow matvec paths | Avoid launches and intermediate traffic, including small speculative verification batches | `ggml/src/ggml-cuda/ggml-cuda.cu`, `norm.cu`, `q8_repack/` |
| Grouped MoE weight reuse | Decode each expert weight block once for several verification tokens, retaining separate dot products and original routing | `ggml/src/ggml-cuda/flashnext-moe.cuh`, `q8_repack/flashnext-moe.cuh` |
| Speculative prompt replay and repacked draft weights | Reduce draft preparation cost; the single-GPU DFlash draft avoids extra TP communication | `common/speculative.cpp`, `src/llama-context.cpp` |
| Split KV, recurrent snapshots, and per-sequence taps | Keep speculative execution and rollback correct with TP and concurrent requests | `src/llama-graph.cpp`, `llama-memory-recurrent.cpp`, `llama-context.cpp` |

## Coverage by model and quant

Which local paths each model and quant actually uses. "Measured" means a matched run is recorded in [PERFORMANCE.md](PERFORMANCE.md); "path" means the code fires for these tensor types but nobody has measured the gain. Tensor types come from the GGUF headers (2026-11-06).

| Path | 27B Q8_0 | Gemma 4 31B Q8_0 | Gemma 4 31B Q4_K_M, UD-Q4_K_XL | Gemma 4 31B Q4_0, Q4_1 | Flash Next UD quants |
| --- | --- | --- | --- | --- | --- |
| Repacked weights and gfx906 GEMM/mat-vec | all weights; measured: +53% PP, +12% TG vs official | all weights; path | Q4_K/Q5_K/Q6_K weights (IQ4_XS share is not); path | none, plain upstream kernels | non-expert Q6_K/Q8_0 and IQ4_NL down experts; measured +74% PP vs official, no plain TG gain |
| Fused GLU and bias mat-vec | yes | yes; path | no, dense fusion is Q8_0/MXFP4 only | no | MoE fusion for Q8_0, K-quants, IQ4_NL |
| gfx906 MMQ tuning (row-first traversal, J cap) | little, repack takes most matmuls | little | IQ4_XS share only | all PP matmuls; path | IQ gate/up experts in PP |
| TP stack (AllReduce overlap and slabs, parallel lanes, token graph replay) | yes; measured (slabs PP, lanes TG) | yes; path | yes; path | yes; path | yes; measured (lanes TG) |
| Tied output head split | n/a, untied | yes, 262K-vocab head; path | yes; path | yes; path | n/a |
| Shared-expert split | n/a | n/a | n/a | n/a | yes |
| Host n-gram table placement | n/a | n/a | n/a | n/a | yes |
| Narrow IQ2_S expert path, grouped MoE (default on) | n/a | n/a | n/a | n/a | IQ2_S and IQ4_NL experts; measured |
| Cheaper IQ sign apply (S2) | n/a | n/a | n/a | n/a | IQ2_S and IQ3_XXS experts; measured +5.1% TG, +3.1% PP |
| Q5_0 (native) and IQ4_XS (as IQ4_NL) repack | n/a | n/a | IQ4_XS share; path | n/a | Q5_0 down and IQ4_XS gate/up experts of UD-Q3_K_XL/UD-IQ4_XS; measured +2-3.5% TG, +6-9% PP |
| MTP prompt replay, KV-only prefill, backend sampling | yes; measured | upstream Gemma 4 assistant path, not the local MTP work | same | same | yes; measured |
| DFlash | yes; measured | no draft | no | no | no |

Notes:

- Repack fires for whole (non-view) 2D/3D weights of type Q8_0, MXFP4, IQ4_NL, Q5_1, Q5_0 (`ne0 % 32`), and Q4_K/Q5_K/Q6_K/IQ4_XS (`ne0 % 256`); see `ggml_cuda_repack_tensor_supported`. IQ4_XS is stored in the IQ4_NL layout (`ggml_cuda_repack_store_type`). IQ1/IQ2/IQ3 and Q4_0/Q4_1 never repack.
- Gemma 4 31B is dense with a tied output head, so the head split reads half the vocabulary per GPU each token.
- On this fork, prefer a K-quant over Q4_0/Q4_1 for Gemma 4 31B at 4 bits: Q4_0/Q4_1 lose repack and fusion. Not measured yet.
- Flash Next gate/up experts (IQ3_XXS, IQ2_S, IQ4_XS) use the stock IQ kernels plus the narrow IQ2_S path; that is the main Flash Next TG area without a repack layout.

## Merge maintenance index

Use the source map above and this table to decide what needs attention each week. Read detailed evidence only for affected features and their callers. These are preservation rules, not claims that the quick checks cover every case.

| Feature | Preserve across merges | Final checks when affected | Evidence / when to retire local code |
| --- | --- | --- | --- |
| Repack and MMQ/MMVQ | Upload, allocation, layout, dispatch and consumers agree; gfx906 MMQ stays row-first | TP smoke, model generation, matched PP/TG | [Layout](../ggml/src/ggml-cuda/q8_repack/README.md), [rebuild](old-ignore/2026-10-29/rebuild/README.md); retire when upstream covers the layout and measured benefit |
| Narrow IQ2_S expert path | Only even rows and one output column; other shapes retain fallback | Targeted CPU-reference cases and affected MoE workload, with approval | [IQ2_S record](old-ignore/2026-10-29/iq2s-single-token-experiment/README.md); retire if upstream matches it or benefit disappears |
| AllReduce and overlap | HIP host staging by default, graph-safe counters and scheduler hooks; peer transport opt-in; RCCL off on this host | TP smoke at both widths and matched PP/TG | [Rebuild](old-ignore/2026-10-29/rebuild/README.md); remove local extensions only when upstream replaces their behavior |
| TP lanes, token graphs and fusions | Scheduler/readback dependencies, shared-expert reduction and graph update safety; no new per-ubatch host sync | TP smoke, generation and matched PP/TG | [Rebuild](old-ignore/2026-10-29/rebuild/README.md); prefer equivalent upstream paths after validation |
| Grouped MoE reuse | On by default (`GGML_CUDA_FLASHNEXT_MOE=0` disables), eligible narrow IQ2_S/IQ4_NL batches, original routing and fallback | Targeted reference and MoE output checks, with approval; dense Q8 smoke does not cover this | [Grouped-MoE record](old-ignore/2026-10-25/flashnext-fast/README.md); retire if upstream replaces it or benefit disappears |
| Fused hyper-connection mix (F2) | `GGML_OP_DSV4_HC_MIX` (CPU reference, ROCm kernels, meta whitelist, repack exemption for the hc down/up weights); `build_hc_mix` takes it for 1-8 tokens; `GGML_CUDA_HC_MIX_FUSED=0` restores the old graph. A new upstream op must also bump `GGML_OP_COUNT` asserts | `test-backend-ops -o DSV4_HC_MIX`, Flash Next generation; KLD at `-b 8 -ub 8` | [F2/P4 record](PERFORMANCE.md#2026-11-06-f2-p4-toolchain-check-and-acceptance-aware-speculative-runs); retire if upstream fuses hc mix |
| Small-M F32 GEMM (P4) | GCN only, F32, M < 128, N > 16, contiguous K, before the cuBLAS fallback in `ggml_cuda_mul_mat`; `GGML_CUDA_SGEMM_F32=0` disables | `test-backend-ops -o MUL_MAT`, Flash Next PP | [F2/P4 record](PERFORMANCE.md#2026-11-06-f2-p4-toolchain-check-and-acceptance-aware-speculative-runs); retire if rocBLAS gets fast at these shapes |
| Speculative replay and state | Draft repack, split KV, per-sequence taps, rollback snapshots and output stride; rounded ubatch 2048 can need 2049 tap rows | Targeted MTP/DFlash output and rollback checks, with approval; plain generation does not cover this | [Rebuild](old-ignore/2026-10-29/rebuild/README.md), [scripts](scripts/README.md); retire fixes when upstream preserves these invariants |
| Model integration | Host PLE placement, shared/tied output split and MTP mirrors; reconcile loader, graph and memory callers | Generation for affected cached models; report unavailable coverage | [Removed and changed](#2026-10-29-rebuild-removed-and-changed); prefer upstream model implementations and do not restore removed fork-only models |

### Silent merge traps (found on the 4b77dee05 merge, 2026-11-05)

These merged or compiled cleanly but were wrong. Check them first on each merge.

- MMQ MoE now uses the linear `tile_spec` layout from the ids helper (upstream #31047). The kernel J must equal the `J_best` that built `tile_spec`, so J is selected once, in `ggml_cuda_mmq_get_J_best` (it also holds the gfx906 J > 64 cap). Do not add a second J selection in `mul_mat_q_switch_J`.
- The repack and grouped-MoE paths still read per-expert `expert_bounds`. The `ggml_cuda_launch_mm_ids_helper` overload in `mmid.cu` that writes bounds instead of tiles serves them.
- Upstream `build_conv_state_store` writes only `min(n_tokens, K)` rollback slots. Non-ring rollback is therefore bounded by `cell.rs_max`; ring mode keeps `valid_depth`. The fork keeps its `s_write` (delta-net) and `s_write_conv` (qwen4exp) ring branches.
- Upstream `layer_inp_dev` is the DFlash device feature buffer. The fork per-layer persistent taps are `layer_inp_tap`; keep the names apart.
- The upstream DFlash device feature path is opt-in (`LLAMA_DFLASH_DEV=1`), so the fork tap and accum pipeline stays the default. A DFlash draft with its own head does not set `ctx_other`, to keep it single-GPU.
- A new repack type must also be added to the type whitelist in `ggml_backend_meta_device_supports_op` (`ggml-backend-meta.cpp`), or under `-sm tensor` its weights never reach the repack buffer and nothing fails.
- K/V rotations are per-layer constants (`inp->self_k_rot[il]`), not graph inputs. Local MTP KV-only code must index them by layer and use `build_rope`.

Update a row when its dispatch, dependencies or validation needs change. Record new evidence once and link it here. DFlash vision remains unvalidated. The [catch-up guide](how-to-catchup.md#one-final-validation-pass) defines the common final pass and approval limits; feature-specific checks join that pass rather than becoming repeated intermediate suites.

The [repack README](../ggml/src/ggml-cuda/q8_repack/README.md) explains its layout and integration. Preserve upload, allocation, dispatch, and consumers together; canonical kernels cannot read repacked weights. For HIP AllReduce, keep the first `#if !defined(GGML_USE_MUSA)` branch in `allreduce.cu` available to HIP. The normal configuration must not depend on an environment switch.

Upstream already provides the basic HIP host-staged AllReduce. The local delta adds graph-replay device counters, separate H2D/D2H streams, prefill slab overlap, prepare/launch scheduler hooks, and a separate opt-in peer-write/two-shot transport (`tp-allreduce.cu`, `GGML_CUDA_PEER_AR=1`). The unreachable old HIP overlap-only `#elif` has been removed from `allreduce.cu`; do not restore that duplicate implementation. BF16 transport and its threshold already exist in the upstream base; grouped MoE adds no new precision reduction.

Keep the gfx906 MMQ launch traversal aligned with the gfx906 kernel's row-first indexing. The first port of upstream `418f905a4` changed the block-axis order and showed a large PP drop in one matched Flash Next run. Restoring the gfx906 traversal removed that identified porting regression; that correction is in the active patch set. The initial negative result remains archived and does not measure the corrected tree.

Correctness fixes matter as much as speed: graph reservation can round persistent tap storage above the requested ubatch, prompt state must remain per sequence, and chunked recurrent kernels must preserve every requested rollback snapshot. A successful build or fast PP run does not validate those cases.

## Accepted grouped MoE idea

For narrow MoE verification batches on gfx906, group assignments to the same expert and reuse decoded weights across up to eight entries. Expert selection, quantization and each token's output remain unchanged in intent; there is no intentional precision or quality tradeoff. Floating-point execution differences still need correctness checks.

The [recorded experiment](old-ignore/2026-10-25/flashnext-fast/README.md) found about 6% higher Flash Next TG with grouped-only enabled, identical text/acceptance on the tested prompt, and passing CPU-reference cases. This accepts the weight-reuse idea, not a guarantee for every workload. It is on by default since 2026-11-06 (`GGML_CUDA_FLASHNEXT_MOE=0` disables), after byte-identical MTP2/MTP3 output and +6-8% TG at 100K. The separate GPU-feedback path is not included. The wider MMQ fallback remains outside this validation.

The active code also accepts the narrow IQ2_S single-token expert path described above. One matched 20-thread pair measured 38.08 vs 37.20 TG (+2.37%) with identical output, but this is not repeatability evidence. The code only selects for IQ2_S expert operations with one output column and even rows; it does not apply to dense Q8_0, and no Q8_0 benefit was measured. Three focused CPU-reference cases passed. See the [dated single-token report](old-ignore/2026-10-29/iq2s-single-token-experiment/README.md).

## 2026-10-29 rebuild: removed and changed

The fork was rebuilt on upstream `36a0e2ffc` as stages B-D. Each stage matched the old master on 27B Q8_0 and Flash Next PP/TG, and greedy output was identical across B, C and D. Details and numbers are in the [rebuild record](old-ignore/2026-10-29/rebuild/README.md).

Removed:
- Multi-stage TP / pipeline stages (`-tps`, `--tensor-parallel-size`, `-tpsd`). They were unused on two GPUs. To restore, start from mx-llama commit `77ffa6093` ([mixa3607/mx-llama.cpp](https://github.com/mixa3607/mx-llama.cpp)).
- Fork-only DeepSeek V4/V4.1 code: MLA head split, vocabulary-parallel output head, narrow shared-expert buffer rule, HC gate fusion, DSV4 KV cache changes, DFlash DSV4-backbone drafter split, V4.1 hc_shift. These models run on upstream code again. None of them is local, so the loss was not measured.
- MiniMax-M2 under `-sm tensor` (upstream rejects it again) and the MiMo-2 SWA pattern fallback for files that count MTP layers.
- Dead code and debug knobs: scheduler timing and traces, realloc trace, repack fusion stats, unused repack variants, the dead `mmq-config-gfx906.cuh`, the uncalled `tp_custom_ar_allreduce`, the unused ext API for layer accumulation, and fork-only Engram additions.

Kept as generic: shared-expert split (`LLAMA_SHEXP_SPLIT`), tied output split, MTP block mirror, PLE shard, dense DFlash drafter replication, and the DFlash SWA pattern fix.

Fixed:
- B12: unsharded IQ4_NL/MXFP4 getrows gather with `ne00 % 256 != 0` wrote zeros.
- B1: gguf-split split a string literal at a newline.

Environment knobs:
- Peer-write AllReduce is now opt-in with `GGML_CUDA_PEER_AR=1`. Before, it turned on by itself when peer access was reported.
- Removed with the stages: `LLAMA_STAGE_LOCAL_MIRROR`, `GGML_META_SPLIT_CARRY_STAGE`, `GGML_META_XFER_RCCL`, `GGML_META_MOE_XFER_RCCL_THRESHOLD`, `GGML_META_NO_DEDICATED_CPY`, `GGML_META_XFER_ROWS`, `GGML_META_LAYER_SEAM_COST`.
- Removed with DeepSeek V4/V4.1: `LLAMA_MLA_TP` and the fork `LLAMA_DSV41_*` knobs (they were inert). Upstream `LLAMA_DSV41_SPARSE` remains.
- Removed in cleanup (default path unchanged): `GGML_CUDA_TP_OVERLAP`, `GGML_SCHED_TIMING`, `GGML_SCHED_TIMING_INTERVAL`, `GGML_SCHED_TRACE_INPUTS`, `GGML_SCHED_REBIND_EVENT_TRACE`, `GGML_SCHED_SYNC_NONGRAPH`, `GGML_ALLOC_REALLOC_TRACE`, `GGML_CUDA_GATE_DEBUG`, `GGML_CUDA_REPACK_MOE_FUSION_STATS`, `GGML_CUDA_OP_SYNC`, `GGML_RP_NC_FUSION`, `GGML_RP_NC_RPL`, `GGML_RP_NC_23`, `GGML_RP_GENERIC_MMV`, `LLAMA_REUSE_NO_DRAIN`, `LLAMA_NO_FUSED_LID`.
- `GGML_TP_AR_NO_GATE` and `GGML_TP_AR_ONESHOT` are now read only in NCCL builds. They were no-ops without NCCL.

Builds must use the README build-ilp flags (`-mllvm -amdgpu-sched-strategy=max-ilp`, CXX `-O3`). A build without them was 2-3% slower on 27B.

Still open:
- HIP gating of default-on shared-code changes: B8 `calc_nwarps`, B14 BF16 rule, B15 graph key, B16 H2D copy, eager cuBLAS handle, AR slabs, `GGML_SCHED_SHARE_VIEW_INPUTS`. B8 needs an MMVQ launch-path change.
- Move `ggml-cuda.cu` glue into separate files.
- Expose `event_query` through the proc-address table instead of one-line backend edits.

## Worth revisiting

- RCCL after both GPUs have suitable CPU-connected PCIe lanes. Its current exclusion reflects this host's topology.
- TP split and batch sizing after hardware or workload changes. Equal split and batch/ubatch 2048 are comparison defaults, not hardware-independent optima.
- Narrow verification kernels under concurrent speculative load; more than eight rows can leave the optimized path.
- Q4/K-quant performance, populated deep contexts, and quantized KV. They need their own measurements.
- DFlash vision and mixed token/embedding requests, which remain unvalidated.

Earlier failed or slower experiments describe their tested workload and implementation, not permanent limits. Consult the [dated archive](old-ignore/README.md) before repeating work, then recheck assumptions against the current source. Get explicit OK before experiments or longer tests.

## Built on

- [mixa3607/mx-llama.cpp](https://github.com/mixa3607/mx-llama.cpp) - tensor parallelism and speculative runtime work.
- [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) - gfx906 Q8 weight repacking and kernel work.
- [Stastez's host-staged AllReduce work](https://github.com/ggml-org/llama.cpp/pull/27825) - ROCm communication over PCIe.

This fork integrates and extends those ideas on [llama.cpp](https://github.com/ggml-org/llama.cpp); the list is a short lineage note, not a per-line authorship claim.

## Upstream overlap (checked 2026-10-29)

- Host-staged ROCm AllReduce already landed in [PR 27825](https://github.com/ggml-org/llama.cpp/pull/27825), merged Sep 15. It is not gfx906-only. Compare local transport/overlap changes against that implementation before proposing anything new.
- gfx906 MMQ tuning has an existing [discussion, 23881](https://github.com/ggml-org/llama.cpp/discussions/23881). Reported Q8 improvements and Q4 regressions favor small, quant-specific candidates over a blanket architecture default.
- [PR 28198](https://github.com/ggml-org/llama.cpp/pull/28198), merged Sep 3, enables concurrent streams per multi-GPU graph split. It is related groundwork, not the same as this fork's whole-token TP graph implementation.
- The [repack design discussion, 21130](https://github.com/ggml-org/llama.cpp/discussions/21130), identifies testing and RPC/TP integration concerns. Our current GPU repack kernels are gfx906-specific; upstreaming them would need more than a tuning-table change.
- No exact upstream submission for this grouped-MoE implementation was identified in the targeted search. Weight reuse is a general idea, but this implementation and its evidence are gfx906-specific. Absence from search results is not evidence of rejection.

A small per-quant gfx906 tuning change is the clearest first upstream candidate. Larger repack, scheduler and speculative changes need separate discussion and validation. Upstreaming could reduce merge work, but the current fork's measurements do not establish benefits on other GPUs.
