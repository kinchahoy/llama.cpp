# Performance evidence

## Current numbers

Measured source: runtime tree `3875d3850b45825cc12d5aecb22487b7377ad598` (official `36a0e2ffc` plus the [current kit](patches/llama-head-36a0e2ffc/README.md)), which is the runtime content of master commit `f5031e739`. Build: `scripts/configure_stage.sh` flags, TheRock ROCm on two gfx906 (MI60 + MI50).

| Model | Mode | PP2048 | TG | Measured (PDT) | Source |
| --- | --- | ---: | ---: | --- | --- |
| Qwen3.8-27B Q8_0 | plain, llama-bench | 685.3 | 36.7 | 2026-10-29 20:01 | [rebuild table](#2026-10-29-rebuild-on-official-36a0e2ffc) |
| Qwen3.8-27B Q8_0 | MTP2, llama-cli | - | 60.7 | 2026-10-29 20:29 | [speculative runs](#speculative-decoding-on-d-one-run-each) |
| Qwen3.8 Flash Next IQ3_XXS | plain, llama-bench | 527.6 | 39.1 | 2026-10-29 20:01 | [rebuild table](#2026-10-29-rebuild-on-official-36a0e2ffc) |
| Qwen3.8 Flash Next IQ3_XXS | MTP2, llama-cli | - | 49.6 | 2026-10-29 20:29 | [speculative runs](#speculative-decoding-on-d-one-run-each) |

These numbers are out of date once the runtime source changes. Check with:

```sh
git diff --quiet 3875d3850b45825cc12d5aecb22487b7377ad598 HEAD -- . ':!vr' ':!AGENTS.md' ':!CLAUDE.md' ':!README.md' && echo current || echo stale
```

A new toolchain, driver or model revision also makes them stale, even if the check prints `current`. When re-measuring, update the tree id, commit, dates and rows here and add the full run as a new dated record below.

Against official llama.cpp with the same build flags: +53% 27B PP, +12% 27B TG, +74% Flash Next PP, no Flash Next plain TG gain.

## How to test, and what the status words mean

Every idea below has one status:

- **Landed**: in master, with the evidence named.
- **Validated wrong**: a test that could show the effect was run, and the effect was not there. The test is named. Do not retry without a new reason.
- **Not shown**: something was run, but the test could not show the effect (change did not fire, wrong metric, one run inside the noise). Not a reason to drop the idea; the row says what a proper test needs.
- **Open**: not tested.
- **Rejected**: user decision. Do not propose again.

A test can mark an idea validated wrong only if it does all of these:

1. Confirms the change runs: kernel counts in a `rocprofv3 --kernel-trace`, a debug counter, or a changed output where one is expected. The scale-fold attempt below measured a build where the fusion never fired.
2. Measures the metric the change targets. AllReduce slab overlap is a PP change, so a TG run cannot judge it.
3. Beats the noise. Matched builds, same settings, baseline and candidate alternating. 27B TG varies about +-0.3 t/s between runs, so r3 alternating is enough. Flash Next TG ranged 33.3-40.1 t/s across one-repeat baselines on 2026-10-29 and Flash Next PP varies up to about 18 t/s, so Flash Next needs r5 alternating, or a profile that shows the change in kernel time.
4. Compares output: greedy text hash against the baseline. Speculative runs diverge from plain after about 1500-2000 bytes (batched verify flips near-tie tokens), so compare speculative runs to speculative runs.

Keep the 50/50 split for A/B work. Ask for explicit OK before performance runs. Use `rocprofv3` timings only for attribution: profiled TG was 21-45% slower than unprofiled. Scripts: [scripts](scripts/README.md).

## Idea ledger

Ceilings are per-token estimates from profiles, not forecasts. Ids from older records are kept: A = fork audit, N = further code ideas, S = Strata review, L = optimization-lab review, P = Flash Next priorities.

### Landed

| Id | Change | Evidence | Gap in the evidence |
| --- | --- | --- | --- |
| - | AllReduce slab overlap, `GGML_CUDA_AR_SLABS=4` | Oct 12: 27B PP2048 651 (off) -> 682 (4), text identical. TG neutral in the 2026-10-29 A1 sweep | PP not re-measured on the rebuild tree |
| - | Parallel lane dispatch, `GGML_META_PARALLEL_DISPATCH=1` | A1 sweep: off costs about 1.5 t/s TG on both models against neighbouring baselines | One repeat per run, but both models agree and the drop is above 27B noise |
| - | Whole-token HIP graph replay | 2026-10-29 trace: two `hipGraphLaunch` per token, no per-kernel launches, both models | - |
| - | MTP2 with `--spec-draft-backend-sampling` as the speculative default | 27B +68% vs plain, Flash Next +33% (one run each, 2026-10-29) | Best depth: MTP2 beat MTP3 on Oct 12 and 2026-10-29, MTP3 won on Oct 25; all single runs |
| - | gfx906 two-row IQ2_S one-token expert path | +2.37% TG in one matched pair, output matched, 3 CPU-reference cases | One pair; not repeatability evidence |
| - | Grouped MoE for 2-8 token verify, opt-in with `GGML_CUDA_FLASHNEXT_MOE=1` | Oct 25: about 5.9% mean TG on Flash Next MTP2, two runs per variant, same output and acceptance | Off by default; the 2026-10-29 MTP runs and scripts do not enable it. Not re-measured on the rebuild tree |

### Validated wrong

| Id | Idea | Test that refuted it |
| --- | --- | --- |
| N1 | DFlash draft graph rebuilds every call and blocks replay; keep both draft graphs allocated | 2026-10-29 instrumented build: a rebuild costs 0.36 ms, 1.1% of the run, and the draft already replays through ggml-cuda HIP graphs (67 replays, 2 re-captures). The Oct 12 premise predates the single-GPU draft. [Record](#2026-10-29-n1-dflash-draft-replay-check) |
| A1 (27B TG) | Uneven `-ts` for 27B TG | Every split slower on 2026-10-29: 0.52/0.48 about -1 t/s (initial run and two r2 retests), 0.48/0.52 -1.9, 0.56/0.44 -2.5. Oct 12 agrees: 53/47 35.6 and 56/44 34.8 vs 36.8 at 50/50 |
| - | `-sm layer` instead of tensor split on Flash Next | 30.99 vs 38.94 TG, -20%; one run, but far outside the noise |
| N2a | Overlap the two lanes' graph launches | 27B 36.63 -> 36.67, one pair, inside 27B noise; the trace shows the launch stagger is not on the critical path |
| N2b | Stage the remaining small F32 inputs asynchronously (`GGML_SCHED_INPUT_STAGING_F32_MAX=1048576`) | r3 alternating, both models, no change |
| - | Content-keyed meta token graph cache for the draft (Oct 12) | Wrong output: draft acceptance length fell from 2.95 to 1.1-1.6 |
| - | Multi-row planar Q8_1 producer (Oct 12) | Halved quantize launches but made the norm kernel slower by the same amount |
| - | Oct 12 smaller items: MTP fused catch-up (already overlapped with sampling), nc unroll 2, `GGML_RP_NC_FUSION`, MTP `p-min` (TG 55 -> 29-31), DFlash mirrored output head | Each measured in the [Oct 12 work log](old-ignore/2026-10-12/notes/2026-10-12-work-status.md) |

### Not shown (test could not show the effect)

| Id | Idea | What was run | Why it does not decide the idea | Proper test |
| --- | --- | --- | --- | --- |
| P1a | Fold the constant `1/hc` scale into the hyper-connection down mat-vec | Build plus Flash Next r5 TG (-0.5%) | The fusion never fired: identical kernel counts (10145 `scale_f32` on both builds) | First get a lower `scale_f32` count in a trace. The trace bounds this subset at about 0.4 ms/token and all 303 scales at about 1.2 ms (profiled) |
| A2 | Defaults `LLAMA_SHEXP_SPLIT=1` and `GGML_SCHED_SHARE_VIEW_INPUTS=1` | TG128, one repeat each (A1 sweep) | Off was 0.6-1.1 t/s slower on Flash Next, inside its noise; 27B share-view off -0.1. Both knobs also act on PP, which was not run | PP2048 r3 (27B) or r5 (Flash Next), alternating; drop a knob that shows no difference |
| A1 (FN, PP) | `-ts` for Flash Next TG, and for PP on both models | Flash Next TG one repeat: 0.52/0.48 37.80 vs 37.46, 0.48/0.52 36.64 vs 40.08 | Inside Flash Next TG noise; PP not rechecked (Oct 12: 56/44 gave +3% 27B PP for -5% TG) | Flash Next TG r5 alternating; PP2048 r3 at 56/44 for a PP-only profile |
| - | `-t 16` on Flash Next | One sample: 39.44 vs 37.64 in the same run | One sample inside the noise | r5 alternating against `-t 8` |
| N2 | Per-token host gap before each token graph | Profiled idle 1.04 ms/token (27B) and 4.48 ms/token (Flash Next); two fixes (N2a, N2b) refuted | The gap was only measured under the profiler, which slowed TG 31-45% in that trace. Its unprofiled size and cause are unknown; scheduler event waits, launch cost and profiler overhead are guesses | Measure idle time without the profiler (host timestamps around graph compute and the logits readback). Only then consider decode pipelining, a large change (GPU-side sampling only, late stop detection, conflicts with MTP) |
| - | Grouped-MoE gain on the current tree (roofline experiment 3) | Not re-run since Oct 25 | - | MTP2 with `GGML_CUDA_FLASHNEXT_MOE=0/1`, r5 alternating, same acceptance; record the per-layer expert union size. If it holds, consider making it the default |

### Open

| Id | Idea | Model | Basis | Ceiling | Effort |
| --- | --- | --- | --- | --- | --- |
| P1 | Remove intermediate `scale_f32` and `quantize_q8_1` passes as one combined change, reusing the existing norm-plus-quantize fusion | Flash Next (the quantize part helps every quantized model) | About 480 `quantize_q8_1` (2.3 ms) and 303 `scale_f32` (1.8 ms) per token on ROCm1, profiled; each scale about 4.2 us with no gap before it, so only kernel time is recoverable | about 2-3.5 ms/token | M; check kernel counts first |
| P2 | Speed up the tiny float projections: find the shapes behind the 264-call float mat-vec, combine projections that share an input, or use a smaller launch shape | Flash Next | 288 float mat-vecs at about 11 us, 3.1 ms/token | up to about 2 ms | M |
| S3 | Split the hyper-connection down projection along K so it fills 60 CUs | Flash Next | Strata saw 41 blocks at about 100 GB/s; changes sum order, needs a quality check | about 3% (Strata) | M |
| S4 | One-wavefront-per-row small mat-vec with a 64-lane butterfly, 1-8 columns | mostly Flash Next | Candidate for P2; check our block shape first | unknown | S-M |
| P3 | Find the late rank before each AllReduce and speed up its preceding work | both | ROCm0 waits in AR 5.19 vs 1.03 ms/token (Flash Next), 4.1 vs 1.7 (27B, Oct 12) | part of the 5.3 ms AR | M |
| A5 | Expert parallelism: whole experts per GPU instead of 320-row halves | Flash Next | Needs one 640-row expert to run more than 1.25x faster than two halves. [Assessment](#expert-parallelism-a5) | 1-2 ms/token TG; better for PP | L; go/no-go microbenchmark first |
| S1 | Stage the IQ codebook grid and the q8_1 activations in LDS in the grouped expert gate/up kernel | Flash Next | Strata found this kernel compute-bound at 110-130 GB/s; ours reach about 150 GB/s useful | part of the 4.7 ms expert union | M |
| S2 | IQ sign handling as `dp4a(g ^ m, u) - dp4a(m, u)` instead of `__vcmpne4` + `__vsub4` in `vecdotq.cuh`; bit-exact | Flash Next | gfx906 emulates each byte op in about 6 word ops (`vendors/hip.h`); saves about 8 ALU ops per sign group | modest | S; test with `test-backend-ops` and `cmp` |
| S5 | Router top-k in one 64-lane wavefront, no barriers | Flash Next | Strata measured block top-k barrier-bound at 43 us/token; ours is inside the 2.41 ms routing plus MoE reduction | needs our kernel time | S |
| - | Find the about 10 ms host gap per DFlash step (inject/generate gap, sampling and accept, readbacks) | 27B DFlash | Oct 12 profile; not the draft rebuild (N1) | DFlash5 about 70 t/s if closed (Oct 12 estimate) | trace first |
| A6 | Re-profile 27B PP on the current tree | 27B | Oct 12 profile predates duplex copies and slabs | decides A7 vs N3 | S |
| A7 | GCN occupancy tuning (more warps, wider J tiles) in the repacked Q8_0 GEMM | 27B | Upstream discussion 23881: +19% Q8_0 pp512 on GCN with a different kernel | +10-15% PP if GEMM is still the main cost | M; after A6 |
| N3 | Pipeline prefill AllReduce across two half-ubatches so one half's AR overlaps the other's GEMM | 27B, Flash Next PP | Oct 12: 0.78 s of a 3.9 s ubatch in AR copies before overlap; slabs cannot hide the last slab and the reduce | +5-15% PP if A6 shows exposed AR above 10% | M; after A6 |
| L1 | Skip K/V work in `fattn-vec` for wavefronts whose KV rows are all masked; output unchanged | both | +13% decode at 85K context on NVIDIA (optimization lab); nothing at short context | long context only | S; count the masked fraction first |
| A3 | Raise ROCm1's power cap to its 178 W maximum | both | Oct 12 hardware note | small | config |
| A4 | Move ROCm1 to CPU-attached PCIe lanes, then test the peer-write AR (`GGML_CUDA_PEER_AR=1`) and RCCL | both | AR, post-AR gaps and peer wait are 4-7 of 27 ms/token on 27B | largest single lever | hardware |
| - | `mmq-config-gfx906.cuh` is never selected (the `GCN` table wins in `mmq.cuh`) | neither target | 27B uses the repacked GEMM; Flash Next IQ types are not in that table | none for the targets | low priority |

Upstream-only candidate: GCN Q8_0 MMQ rows 256 -> 512 threads (discussion 23881: +19% Q8_0 pp512, -6.9% Q4_K, so Q8_0 only). It helps plain upstream, not this fork's repacked path.

### Rejected

- ngram-mod speculation as a fallback when no MTP head is loaded (2026-10-29). The Oct 12 screen also found ngram-mod alone at plain speed.

## What limits each model

### Qwen3.8-27B Q8_0

About 28.7 GB of weights per token (14.3 GB per GPU). At the measured 650-715 GB/s per card for large Q8 mat-vecs, the weight stream takes about 20-21 ms of the 27.2 ms/token, about 75% of the bandwidth roof. Single-GPU TG of 22.76 t/s gives about 45 t/s for perfect 2-way TP. A software-only result of about 40-44 TG is a plausible target; more most likely needs A4. The rest is AllReduce latency through host memory, rank skew and small operations. [Oct 12 dense profile](old-ignore/2026-10-12/notes/2026-10-12-tp-profile-findings.md).

### Qwen3.8 Flash Next IQ3_XXS (`qwen4exp`)

Flash Next reads about 6x fewer weight bytes per token than 27B but decodes at the same 26-27 ms/token. Its time is mostly fixed per-operation and per-dependency cost, which MoE does not reduce. This is why official llama.cpp matches the fork on Flash Next plain TG: the fork speeds up byte-heavy paths.

| | 27B Q8_0 | Flash Next IQ3_XXS |
| --- | ---: | ---: |
| Weight bytes per token | about 28.7 GB | about 4.95 GB |
| Ideal weight-stream time | about 20-21 ms | about 3.5-3.8 ms |
| Measured time per token | about 27.2 ms | about 25.7 ms |
| Share of the bandwidth roof | about 75% | about 14% |

Active bytes per token (GGUF tally of revision `c8b5954a88c2`: sum of `ReaderTensor.n_bytes`, `_exps.` tensors x 10/512, embedding tables excluded; canonical storage, not measured HBM traffic):

| Tensor category | Stored GB | Active GB/token |
| --- | ---: | ---: |
| 144 routed gate/up/down tensors | 48.63 | 0.95 |
| Output projection, Q6_K | 0.52 | 0.52 |
| Everything else except the two embedding tables | 3.48 | 3.48 |
| Token embedding and the 28.8 GB per-layer n-gram table | 29.32 | a few KB (selected rows) |

Routed experts are only 19% of active bytes; free expert weights would save under 1 ms. Where the time goes per token, from the [profiled trace](#2026-10-29-flash-next-tg-profile) (21% slower than unprofiled, categories overlap, so this is attribution, not a budget):

- Non-expert projections: about 500 Q6_K/Q8_0 mat-vecs (6.4 ms) and about 290 float mat-vecs at about 11 us each (3.1 ms). Each is too small to fill the GPU.
- Glue kernels: about 480 `quantize_q8_1` (2.3 ms), about 300 `scale_f32` (1.8 ms), routing and MoE reduction (2.4 ms).
- AllReduce: about 5.3 ms, with rank skew (ROCm0 5.2 vs ROCm1 1.0 ms of AR dispatch time).
- Attention, hyper-connection mixing and GDN/SSM: about 4.4 ms.
- No dispatch on either GPU: about 4.3 ms under the profiler (see N2).
- Routed experts: about 3.4 ms. Each GPU runs 10 half experts of 320 rows; IQ2_S/IQ3 decode reaches about 150 GB/s useful.

Speculation gains less than on 27B (MTP2 +33% vs +68%): each verify token picks its own experts, so batching reuses fewer weight bytes. The levers are operation count and waiting (P1-P3, S1-S5, A5); a measured 3-5 ms cut would give about 41-45 TG.

### Expert parallelism (A5)

Today the meta backend splits every routed expert across both GPUs: `ffn_{up,gate}_exps` take AXIS_1, `ffn_down_exps` AXIS_0 (`src/llama-model.cpp`), and `MUL_MAT_ID` is split like a plain matmul in `ggml-backend-meta.cpp`. Expert parallelism needs an expert-axis split rule, per-lane expert-id remapping in `MUL_MAT_ID`, and matching changes in the repacked and grouped-MoE kernels, expert scales and `ADD_ID` biases. It removes no communication: routed and shared experts already share one AllReduce.

With 10 uniform picks, the busier GPU averages 6.23 whole experts against today's 10 halves (worth 5 whole). EP wins TG only if a 640-row expert runs more than about 1.25x faster than two 320-row halves; real routing is less uniform. PP is the better case, because load averages out over a ubatch. Go/no-go: on one GPU at the real shapes, time 10 experts x 320 rows against 6-7 experts x 640 rows for IQ2_S gate/up and repacked IQ4_NL down. No expert-parallel prior art was found in ggml-org/llama.cpp or ik_llama.cpp (2026-10-29 search, not exhaustive); open an upstream issue before any PR.

## Dated records

Newest first. Full commands, identities and raw data are in the linked [old-ignore](old-ignore/README.md) folders.

### 2026-10-29 A1 config sweep (TG)

TG128 on the master build, each candidate next to a 1/1 baseline. ROCm0 is taken to be the MI60 (PCI 03:00, the lower bus; HIP orders devices by bus). Most runs are one repeat; the 27B 0.52/0.48 retest is two r2 runs. The sweep script passed `-ts` and `-t` twice, so llama-bench ran both values; the numbers are the candidate rows from the JSON. [Scripts and JSON](old-ignore/2026-10-29/a1-sweep/).

| Setting | 27B Q8_0 t/s (1/1 in same run) | Flash Next t/s (1/1 in same run) |
| --- | --- | --- |
| `-ts 0.52/0.48` | 36.20 (37.42); retest 36.06, 36.16 (37.13, 37.01) | 37.80 (37.46) |
| `-ts 0.48/0.52` | 35.52 (37.39) | 36.64 (40.08) |
| `-ts 0.56/0.44` | 34.92 (37.40) | not run |
| `-t 16` | 36.61 (36.93) | 39.44 (37.64), one sample |
| `-t 8` | 36.86 (37.30) | default |
| `GGML_CUDA_AR_SLABS=0` | 37.39 (baselines 37.16-37.45) | 36.79 (37.52-37.58) |
| `GGML_SCHED_SHARE_VIEW_INPUTS=0` | 37.11 (37.16-37.37) | 36.12 (37.20-37.27) |
| `GGML_META_PARALLEL_DISPATCH=0` | 35.83 (37.37) | 35.82 (37.27-37.99) |
| `LLAMA_SHEXP_SPLIT=0` | n/a | 36.94 (37.52) |

### 2026-10-29 N1 DFlash draft replay check

27B Q8_0, DFlash5, 512 tokens, temporary build counter and HIP-graph logs. The draft rebuilds its graph every call (273 builds, 0 reused) at 0.36 ms each: 98 ms of an 8.5 s run. The draft now runs as a single-device context on ROCm0, not on the meta backend, so meta `tg_lookup` and uids do not apply. The ggml-cuda HIP-graph path replays it anyway: the generate graph (691 nodes) replayed 67 times with 2 re-captures; the inject graph (58 nodes) also replays. The 129 captures and 17400 `hipLaunchKernel` in the trace come from the target's TP lanes. Upper bound of the fix: about 0.7 ms of an 80 ms step, below the noise (DFlash5 57.3 and 60.1 t/s on equal builds). [Notes and instrumentation patch](old-ignore/2026-10-29/n1-dflash/).

### 2026-10-29 per-token host gap (N2)

`rocprofv3` kernel, HIP runtime and memcpy trace of TG32 on master, last 20 tokens. Profiled TG was 25.3 (27B) and 21.4 (Flash Next) against 36.6 and 39.1 unprofiled, so host times are upper bounds. [Record](old-ignore/2026-10-29/tg-host-gap/diagnosis.md), [attribution follow-up](old-ignore/2026-10-29/tg-host-gap/option1.md).

- 27B: both GPUs idle 1.04 ms/token, mostly one 0.45-0.6 ms gap after the end-of-token sync (logits readback, then 4-5 rounds of event wait, small H2D copies and `hipStreamSynchronize`).
- Flash Next: both GPUs idle 4.48 ms/token. About 70 small `hipMemcpyAsync` and about 10 stream syncs before each launch, about 1.7 ms host time.
- llama's per-token `set_input` writes only to pinned host buffers. All pre-launch GPU copies come from the scheduler's split-input path; most already use the fork's async staging, and the rest are synchronous only because of the 4096 B F32 staging cap. Raising the cap (N2b) and overlapping lane launches (N2a) changed nothing.

### 2026-10-29 scale-fold attempt

Folding the `1/hc` SCALE after `hc_attn_down`/`hc_ffn_down` (Q8_0 [10240 x 320], repacked) into the repacked Q8_0 mat-vec store built and matched master output, but a TG32 kernel trace showed identical kernel counts on both builds (116873 on Agent 1, 10145 `scale_f32`), so the fusion never fired. The branch was dropped without finding why. Each `scale_f32` takes about 4.2 us with no dispatch gap before it.

### 2026-10-29 ideas from external reviews

- [Strata](https://github.com/Niko1221/Strata) (MIT, `870aedd9e1eb`): code-only review; its gfx906 results are community MI50 reports at IQ1_M-IQ3_XXS and are not comparable. Gave S1-S5. Lower value: fused gate/up + SwiGLU + q8_1 epilogue (off on gfx906 in Strata), attention query LDS swizzle (+1.2% PP at 64K only), HIP graph node priorities (NVIDIA only). Accuracy-costing features excluded. [Review](old-ignore/2026-10-29/strata-review/README.md).
- [llama-cpp-optimization-lab](https://github.com/Abzolute1/llama-cpp-optimization-lab) (`5408613e`, no license, NVIDIA only, experts in host RAM, own numbers not reproduced): mostly CPU/PCIe expert offload, which does not apply. Gave L1 and the rejected ngram-mod fallback. Its rejection of CPU micro-kernel wins that did not show end to end is a caution for S2 and S4. [Review](old-ignore/2026-10-29/optlab-review/README.md).

### 2026-10-29 rebuild on official 36a0e2ffc

Matched llama-bench, `-sm tensor`, 50/50, FA on. 27B: batch/ubatch 2048, PP2048/TG128, r2. Flash Next: Q8_0 KV, batch/ubatch 1024/512, r2 plus a separate r5. A is the old master on 418f905a4; B-D are rebuild stages. All builds use `-mllvm -amdgpu-sched-strategy=max-ilp` and CXX `-O3` (without them 27B is about 2-3% slower). [Rebuild record](old-ignore/2026-10-29/rebuild/README.md).

| Stage | 27B PP2048 | 27B TG128 | Flash Next PP2048 | Flash Next TG128 | Flash Next r5 PP / TG |
| --- | ---: | ---: | ---: | ---: | --- |
| U: official 36a0e2ffc | 448.60 | 32.77 | 302.99 | 39.66 | 304.33 +- 3.78 / 39.76 |
| A: old master e04bc2e38 | 684.56 | 36.82 | 527.79 | 38.99 | - |
| B: cleanup and fixes | 682.92 / 684.19 | 36.96 / 36.61 | 525.73 / 528.13 | 38.34 / 39.56 | 509.92 +- 17.95 / 39.23 |
| C: no pipeline stages | 685.60 | 36.58 | 520.22 | 36.93 +- 2.78 | 516.43 +- 7.96 / 38.79 |
| D: DSV4/MiniMax/MiMo back to upstream | 685.34 | 36.67 | 527.63 | 39.09 | 520.11 +- 7.34 / 39.26 |

Greedy output (96 tokens) was byte-identical across B, C and D. 27B output matches official byte for byte; Flash Next matches for about 30 tokens and then diverges, consistent with different kernel rounding (repack, fusions), not checked further.

#### Speculative decoding on D (one run each)

`llama-cli`, greedy, `flake.nix` prompt (about 2K tokens), 512 output tokens, context 16384. Flash Next uses the `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` sidecar via `-md`. MTP runs use `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-backend-sampling`. Grouped MoE was not enabled. Script: [mtp_check.sh](scripts/mtp_check.sh).

| Model | Mode | Prompt t/s | Generation t/s | vs plain TG |
| --- | --- | ---: | ---: | ---: |
| 27B Q8_0 | plain | 462.0 | 36.2 | - |
| 27B Q8_0 | MTP2 | 514.0 | 60.7 | +68% |
| 27B Q8_0 | MTP3 | 557.8 | 58.0 | +60% |
| Flash Next IQ3_XXS | plain | 348.0 | 37.2 | - |
| Flash Next IQ3_XXS | MTP2 | 339.1 | 49.6 | +33% |
| Flash Next IQ3_XXS | MTP3 | 349.6 | 44.5 | +20% |

Draft acceptance was not recorded (`--simple-io` prints none). The CLI prompt figure varies more than llama-bench PP.

### 2026-10-29 Flash Next TG profile

64-token capture on the IQ2_S candidate library (`fe34c2...`); matched unprofiled 36.71 TG, profiled 28.93 (-21%). 63 steady graph repeats; in the final 2.178 s window both GPUs had dispatches active for 1.905 s. Per-GPU sums and unions overlap, so they cannot be added into a critical path. The raw trace and `summarize.py` are outside the repository.

| Kernel group | ROCm0 sum | ROCm1 sum | Two-GPU union | Union per repeat |
| --- | ---: | ---: | ---: | ---: |
| Non-expert Q6_K/type-14 and q8_0-repacked mat-vecs | 377 ms | 401 ms | 628 ms | 9.97 ms |
| TP AllReduce | 327 ms | 65 ms | 336 ms | 5.33 ms |
| IQ2_S, IQ4_NL and IQ3_S expert mat-vecs | 199 ms | 215 ms | 295 ms | 4.69 ms |
| Float mat-vecs | 195 ms | 194 ms | 278 ms | 4.41 ms |
| Routing and MoE reduction | 100 ms | 109 ms | 152 ms | 2.41 ms |
| Position/attention, hyper-connections, GDN/SSM | 142 ms | 153 ms | 274 ms | 4.35 ms |

ROCm1 per-token detail: `quantize_q8_1<32>` 484 calls 2.33 ms; `scale_f32` 303 calls 1.77 ms; float mat-vecs 288 calls 3.08 ms (main specialization 264 calls, 2.90 ms); Q6_K and Q8_0 repacked mat-vecs 496 calls 6.36 ms.

### 2026-10-29 earlier checks on the official head

- Official-head port: an early unrepaired 418f port measured Flash Next MTP2 PP -28%; the cause was a changed gfx906 MMQ launch traversal. After restoring row-first traversal: 347.97 / 36.77 baseline vs 351.30 / 39.51, identical output and acceptance. [Traversal check](old-ignore/2026-10-29/head-merge-validation/terra-port-fix/README.md), [validation](old-ignore/2026-10-29/head-merge-validation/README.md).
- Quant and placement: IQ3_XXS with GPU experts 377.50 / 38.94; IQ4_XS with 8 CPU MoE layers 197.59 / 24.42 (op-offload on) and 204.92 / 25.15 (off), one run each, quant and placement changed together. Layer split 353.76 / 30.99. [Record](old-ignore/2026-10-29/quant-placement-compare/README.md).
- IQ2_S single-token path: 396.35 / 37.20 baseline vs 398.80 / 38.08 candidate, one pair, output matched, 4,143-token prompt, 256 generated. [Record](old-ignore/2026-10-29/iq2s-single-token-experiment/README.md).
- Grouped-MoE integration: build, replay and 18 canonical reference cases passed; the repacked harness did not finish. [Record](old-ignore/2026-10-29/grouped-moe-integration/README.md).

### Before 2026-10-29

| Date and workload | Setting | PP / TG | Source identity |
| --- | --- | --- | --- |
| Oct 12, synthetic PP2048/TG128 | Qwen3.8-27B Q8_0, plain | about 682 / 36.8 | local changes, commit unknown |
| Oct 25, `flake.nix`, 4000 outputs | 27B MTP2 | 549.7 / 56.8 | build `11776`, `766f7c5e0` |
| Same workload | 27B MTP3 | 545.2 / 60.0 | same |
| Same workload | 27B DFlash7 | 543.7 / 59.9 | same |
| Oct 25, short CLI, 256 outputs | 27B plain | 558.9 / 35.2 | `b11929-fdc8e5b65` |
| Oct 25, Flash Next long CLI | IQ3_XXS plain | 396.9 / 37.3 | same |
| Same workload | MTP2, Q8_0 sidecar | 368.5 / 42.9 | same |
| Oct 25, Flash Next MTP2, short CLI | grouped MoE off / on | 330.3 / 47.65 -> 372.9 / 50.45 | local experiment on `fdc8e5b65` |

One run per setting. The Oct 25 27B runs used about 6K populated tokens despite a 100K context setting, so they do not test a full context. Speculation helped single-request workloads; plain decoding was fastest in the [three-request mixed workload](old-ignore/2026-10-12/2026-10-12-three-slots.md). Target-logprob checks support greedy selection within a run, not general quality equivalence. DFlash vision, other quants, quantized KV and full contexts are untested. Records: [27B CLI](old-ignore/2026-10-25/2026-10-25-qwen3.8-27b-flake-cli.md), [Flash Next MTP](old-ignore/2026-10-25/2026-10-25-flash-next-mtp-short-check.md), [grouped MoE](old-ignore/2026-10-25/flashnext-fast/README.md), [Oct 12 profiles](old-ignore/2026-10-12/notes/2026-10-12-tp-profile-findings.md), [Oct 12 work log](old-ignore/2026-10-12/notes/2026-10-12-work-status.md).
