# Performance evidence

## Current numbers

Build: [catch-up guide](how-to-catchup.md#one-final-validation-pass) flags, TheRock ROCm on two gfx906 (MI60 + MI50). Settings: [standard](#record-format) unless noted. The source column says which tree each row measured.

| Model | Mode | PP2048 | TG | Measured (PDT) | Source tree | Record |
| --- | --- | ---: | ---: | --- | --- | --- |
| Qwen3.8-27B Q8_0 | plain, llama-bench | 696.9 | 37.9 | 2026-11-05 | `bc0ce9691` | [catch-up](#2026-11-05-catch-up-to-official-4b77dee05) |
| Qwen3.8-27B Q8_0 | MTP2, llama-cli | - | 64.1 | 2026-11-05 | `bc0ce9691` | [catch-up](#2026-11-05-catch-up-to-official-4b77dee05) |
| Qwen3.8-27B Q8_0 | DFlash5, llama-cli | - | 67.4 | 2026-11-05 | `bc0ce9691` | [catch-up](#2026-11-05-catch-up-to-official-4b77dee05) |
| Qwen3.8 Flash Next IQ3_XXS | plain, llama-bench | 527.6 | 39.1 | 2026-10-29 20:01 | `f5031e739`, not re-measured | [rebuild table](#2026-10-29-rebuild-on-official-36a0e2ffc) |
| Qwen3.8 Flash Next IQ3_XXS | MTP2, llama-cli | - | 49.6 | 2026-10-29 20:29 | `f5031e739`, not re-measured | [speculative runs](#speculative-decoding-on-d-one-run-each) |
| Flash Next UD-Q3_K_XL + Q8_0 n-gram table | plain, llama-cli, 100K ctx, F16 KV, `-ub 256` | 409.1 (CLI) | 42.1 | 2026-11-06 | `3e19b6adc` | [big quants](#2026-11-06-flash-next-big-quants-repack-conversion-q8_0-n-gram-table-100k-and-mtp-fit) |
| Flash Next UD-Q3_K_XL + Q8_0 n-gram table | MTP2 (Q4_0 draft on ROCm1), 32K ctx, F16 KV | 390.5 (CLI) | 51.5 | 2026-11-06 | `3e19b6adc` | [big quants](#2026-11-06-flash-next-big-quants-repack-conversion-q8_0-n-gram-table-100k-and-mtp-fit) |

A row is out of date once the runtime source changes after its source tree. Check with (replace the commit with the row's tree):

```sh
git diff --quiet bc0ce9691 HEAD -- . ':!vr' ':!AGENTS.md' ':!CLAUDE.md' ':!README.md' && echo current || echo stale
```

A new toolchain, driver or model revision also makes a row stale, even if the check prints `current`. The Flash Next rows used the IQ3_XXS file from rev `c8b5954a88c2`; unsloth replaced it on 2026-11-04. When re-measuring, update the row and add the full run as a new dated record below.

Against official llama.cpp with the same build flags (2026-10-29, `36a0e2ffc`): +53% 27B PP, +12% 27B TG, +74% Flash Next PP, no Flash Next plain TG gain.

## How to test, and what the status words mean

For weekly merges, use the [single final validation pass](how-to-catchup.md#one-final-validation-pass). Its one-repeat PP/TG pair screens for regressions; it does not validate or reject an optimization idea. The stronger evidence requirements below apply to optimization experiments.

Every idea below has one status:

- **Landed**: in main, with the evidence named.
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

## Record format

Add each run as a dated record (newest first) in this shape. Keep it short; put raw files, including the generated text for quality checks, in `vr/old-ignore/<date>/<name>/` and link them.

Standard settings, so records only name what differs. Both GPUs, all layers offloaded, `-sm tensor -ts 1/1 -fa on`, build flags from the [catch-up guide](how-to-catchup.md#one-final-validation-pass), no env knobs:

- 27B: `-b/-ub 2048/2048 -t 20`, F16 KV. llama-bench PP2048/TG128; llama-cli `flake.nix` prompt (~2K tokens), 512 tokens, greedy, ctx 16384.
- Flash Next: `-b/-ub 1024/512`, F16 KV, same workloads. Never quantize the KV cache (user rule, 2026-11-06); records before 2026-11-06 and the 2026-11-06 8K rows used Q8_0 KV.

```markdown
### YYYY-MM-DD short title

Purpose in one sentence. Kind: regression screen | A/B experiment | profile.

- Source: `<commit>` (what it contains); baseline `<commit or build_commit>`.
- Model: `<quant>` rev `<12 chars>`; draft `<file>` if any.
- Settings: standard 27B | standard Flash Next, plus only what differs (e.g. `-ts 0.52/0.48`, `GGML_CUDA_FLASHNEXT_MOE=1`, ctx 100K).
- Workload: standard, or what differs; repeats and order (e.g. r3 alternating).

| Build | Mode | PP t/s | TG t/s | Change | Accept | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | MTP2 | 514.0 | 60.7 | - | 2.6/3 | sha `ab12...` |
| candidate | MTP2 | 516.2 | 61.0 | +0.5% | 2.6/3 | same |

Verdict: one sentence, and the status word if it judges an idea. Not covered: what was not run. Raw: [link](old-ignore/...).
```

Rules: change is `(candidate / baseline - 1) * 100` on the same row type. Write `-` for not measured, never leave a cell blank. Output is the sha (first 12 hex) of the generated text only, without the CLI banner (it contains the build id), or "same"/"differs after N bytes". Accept is mean accepted tokens per draft step over draft size; `llama-cli --simple-io` does not print it, so write `-`. Update [Current numbers](#current-numbers) only from a record that used the current source.

## Idea ledger

Ceilings are per-token estimates from profiles, not forecasts. Ids from older records are kept: A = fork audit, N = further code ideas, S = Strata review, L = optimization-lab review, P = Flash Next priorities, M = merge follow-ups, D = long-context depth.

Priority: ideas that raise plain PP or TG come first, because every mode gains from them. Work that only helps MTP or DFlash comes after those run out, and then only for large gains.

### Landed

| Id | Change | Evidence | Gap in the evidence |
| --- | --- | --- | --- |
| - | AllReduce slab overlap, `GGML_CUDA_AR_SLABS=4` | Oct 12: 27B PP2048 651 (off) -> 682 (4), text identical. TG neutral in the 2026-10-29 A1 sweep | PP not re-measured on the rebuild tree |
| - | Parallel lane dispatch, `GGML_META_PARALLEL_DISPATCH=1` | A1 sweep: off costs about 1.5 t/s TG on both models against neighbouring baselines | One repeat per run, but both models agree and the drop is above 27B noise |
| - | Whole-token HIP graph replay | 2026-10-29 trace: two `hipGraphLaunch` per token, no per-kernel launches, both models | - |
| - | MTP2 with `--spec-draft-backend-sampling` as the speculative default | 27B +68% vs plain, Flash Next +33% (one run each, 2026-10-29) | Best depth: MTP2 beat MTP3 on Oct 12 and 2026-10-29, MTP3 won on Oct 25; all single runs. On 2026-11-05 DFlash5 beat MTP2 on 27B at a 2K prompt (67.7 vs 63.9) but lost at 32K (39.9 vs 42.7), see [record](#2026-11-05-27b-config-sweep-and-32k-depth) |
| - | gfx906 two-row IQ2_S one-token expert path | +2.37% TG in one matched pair, output matched, 3 CPU-reference cases | One pair; not repeatability evidence |
| - | Repack Q5_0 (native, 22 B/32) and IQ4_XS (into the IQ4_NL layout) on gfx906, `GGML_CUDA_REPACK_CONVERT=0` disables | Flash Next UD-Q3_K_XL @8K: +2.3% TG and about +6% PP; UD-IQ4_XS: +3.5% TG, +9% PP; PPL unchanged; 132 repack-vs-CPU cases and 234 `test-backend-ops` cases pass ([record](#2026-11-06-flash-next-big-quants-repack-conversion-q8_0-n-gram-table-100k-and-mtp-fit)) | A/B measured with Q5_0 still through the Q5_1 layout; native Q5_0 is exact and memory-neutral but not separately timed |
| S2 | Cheaper IQ sign apply on gfx906 (`iq_apply_signs`) in the IQ2_S mat-vec and the IQ2_S/IQ3_XXS MMQ tile loads; bit-exact | Flash Next r5 alternating: +5.1% TG, +3.1% PP, same output ([record](#2026-11-06-s2-cheaper-iq-sign-handling-on-gfx906)) | Measured on the old UD-IQ3_XXS upload only |
| - | Grouped MoE for 2-8 token verify, opt-in with `GGML_CUDA_FLASHNEXT_MOE=1` | Oct 25: about 5.9% mean TG on Flash Next MTP2, two runs per variant, same output and acceptance | Off by default; the 2026-10-29 MTP runs and scripts do not enable it. Not re-measured on the rebuild tree |

### Validated wrong

| Id | Idea | Test that refuted it |
| --- | --- | --- |
| N1 | DFlash draft graph rebuilds every call and blocks replay; keep both draft graphs allocated | 2026-10-29 instrumented build: a rebuild costs 0.36 ms, 1.1% of the run, and the draft already replays through ggml-cuda HIP graphs (67 replays, 2 re-captures). The Oct 12 premise predates the single-GPU draft. [Record](#2026-10-29-n1-dflash-draft-replay-check) |
| A1 (27B TG) | Uneven `-ts` for 27B TG | Every split slower on 2026-10-29: 0.52/0.48 about -1 t/s (initial run and two r2 retests), 0.48/0.52 -1.9, 0.56/0.44 -2.5. Oct 12 agrees: 53/47 35.6 and 56/44 34.8 vs 36.8 at 50/50 |
| - | `-sm layer` instead of tensor split on Flash Next | 30.99 vs 38.94 TG, -20%; one run, but far outside the noise |
| N2a | Overlap the two lanes' graph launches | 27B 36.63 -> 36.67, one pair, inside 27B noise; the trace shows the launch stagger is not on the critical path |
| N2b | Stage the remaining small F32 inputs asynchronously (`GGML_SCHED_INPUT_STAGING_F32_MAX=1048576`) | r3 alternating, both models, no change |
| - | MTP draft with its experts on the CPU (`-cmoed`) to fit Flash Next MTP at long context | 2026-11-06, UD-Q3_K_XL 64K: MTP2 34.9 vs plain 43.8 TG. The CPU sync per draft step costs more than speculation gains. [Record](#2026-11-06-flash-next-big-quants-repack-conversion-q8_0-n-gram-table-100k-and-mtp-fit) |
| - | `-b/-ub 4096` for 27B PP | 2026-11-05 llama-bench PP4096 r3: 606.3 +- 3.2 vs 684.1 +- 1.6 at ub 2048, -11.4%. [Record](#2026-11-05-27b-config-sweep-and-32k-depth) |
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
| S5 | Router top-k in one 64-lane wavefront, no barriers | Flash Next | Strata measured block top-k barrier-bound at 43 us/token; ours is inside the 2.41 ms routing plus MoE reduction | needs our kernel time | S |
| - | Find the about 10 ms host gap per DFlash step (inject/generate gap, sampling and accept, readbacks) | 27B DFlash | Oct 12 profile; not the draft rebuild (N1) | DFlash5 about 70 t/s if closed (Oct 12 estimate) | trace first |
| A6 | Re-profile 27B PP on the current tree | 27B | Oct 12 profile predates duplex copies and slabs | decides A7 vs N3 | S |
| A7 | GCN occupancy tuning (more warps, wider J tiles) in the repacked Q8_0 GEMM | 27B | Upstream discussion 23881: +19% Q8_0 pp512 on GCN with a different kernel | +10-15% PP if GEMM is still the main cost | M; after A6 |
| N3 | Pipeline prefill AllReduce across two half-ubatches so one half's AR overlaps the other's GEMM | 27B, Flash Next PP | Oct 12: 0.78 s of a 3.9 s ubatch in AR copies before overlap; slabs cannot hide the last slab and the reduce | +5-15% PP if A6 shows exposed AR above 10% | M; after A6 |
| L1 | Skip K/V work in `fattn-vec` for wavefronts whose KV rows are all masked; output unchanged | both | +13% decode at 85K context on NVIDIA (optimization lab); nothing at short context | long context only | S; count the masked fraction first |
| D2 | De-duplicate memory so Flash Next MTP fits at long context: keep one copy of the QSA indexer key cache (588 MiB per GPU at 100K, mirrored today) and of the MoE router (0.23 GiB per GPU), and load the unsloth "shared" drafts (1.8 GiB, borrow the target embedding and head) | Flash Next | Total free VRAM is the limit, not per-GPU placement: `-ts` and `-devd` only move memory between the cards, and splitting the draft does not shrink it. Q3_K_XL at F16 KV has about 2.3 GiB free in total at 32K, 1.2 at 64K, 0.05 at 100K; a Q4_0 draft needs about 2.3-2.5. These changes save about 0.8 GiB at 100K | not enough for 64K or 100K MTP on Q3_K_XL; gives the 32K setup some margin. 100K + MTP needs a smaller target (UD-IQ3_XXS: about 5.9 GiB less in total) | M-L; indexer on one GPU with a top-k broadcast |
| E1 | Per-expert placement: keep rarely routed experts of each layer on the CPU, hot ones on the GPU | Flash Next IQ4_XS | `-ot` only places whole tensors (all 512 experts of a projection); per-layer CPU offload costs about 1 ms/token per layer in sync | lets IQ4_XS fit with most tokens never touching the CPU | L; split expert tensors, remap ids |
| D1 | Faster long-context attention on gfx906. With F16 KV and no MFMA/WMMA, `ggml_cuda_get_best_fattn_kernel` picks `VEC` for one-token decode and the generic `TILE` kernel for the 3-token MTP2 and 6-token DFlash5 verify batches. Profile both at 32K, then tune or add a small-batch vec path (several query columns per KV pass) | both, all modes | 27B plain TG 36.8 -> 29.4 from a 2K to a 32K prompt: about +6.8 ms/token, most of it attention over the KV cache. MTP2 drops 33% and DFlash5 41%, so the verify batches look worse than decode. [Record](#2026-11-05-27b-config-sweep-and-32k-depth) | up to the 6.8 ms/token depth cost for plain; more for speculative verify | M; rocprofv3 at 32K first, plain and DFlash5 |
| A4 | Move ROCm1 to CPU-attached PCIe lanes, then test the peer-write AR (`GGML_CUDA_PEER_AR=1`) and RCCL | both | AR, post-AR gaps and peer wait are 4-7 of 27 ms/token on 27B | largest single lever | hardware |
| - | `mmq-config-gfx906.cuh` is never selected (the `GCN` table wins in `mmq.cuh`) | neither target | 27B uses the repacked GEMM; Flash Next IQ types are not in that table | none for the targets | low priority |
| M1 | DFlash decode steps read target features from the device: use the upstream device-row path (`LLAMA_DFLASH_DEV=1`, upstream #31506) for generation and verify chunks, and keep the fork accum replay for prompt chunks | 27B DFlash | Each fork decode step does a full `llama_synchronize(ctx_tgt)`, a D2H tap copy, a host gather into `features_buf` and an H2D inject. The upstream path writes the taps to one device buffer on `dev_output()` (the single-GPU draft's device) and the draft reads rows directly, ordered by events, with no host sync. This targets part of the open about 10 ms host gap per DFlash step. Upstream's path alone bypasses the deferred prompt replay, which measured +22% TTFT when eager | DFlash5 about 70 t/s if the gap closes (Oct 12 estimate); this removes only the feature copy part | S to try, M to combine; first `LLAMA_DFLASH_DEV=1` alone (generation t/s, acceptance, TTFT), then route only non-prompt chunks to it. Needs the draft without its own head, so `ctx_other` is set |
| M2 | Build the repacked MoE GEMM tile list from the upstream `tile_spec` instead of `repack_tile_off<<<1, 1>>>` | Flash Next PP, other repacked MoE (Q8_0, MXFP4, IQ4_NL) | `repack_tile_off` is one GPU thread that walks all experts in order (512 for Flash Next) for each repacked `MUL_MAT_ID` launch with more than one token. Its loads depend on each other, so it is latency-bound and the GEMM waits for it. The ids helper already runs one block per expert and, since #31047, writes the same per-tile `{expert, col_low, col_high}` list when given a tile width (`BN_ID` or `BN_W32`). Using it removes one serial launch per MoE matmul and part of the fork's `expert_bounds` overload in `mmid.cu` | unknown; get the `repack_tile_off` time per ubatch from a PP2048 kernel trace first (fits with A6) | S-M; outputs must match bit for bit, since tile order and contents are the same |

A3 (raise ROCm1's power cap) is already in place: sysfs shows ROCm1 (card2) at 178 W, its maximum and default, and ROCm0 at its 225 W maximum (2026-11-05).

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

### 2026-11-06 Flash Next big quants: repack conversion, Q8_0 n-gram table, 100K and MTP fit

Make UD-Q3_K_XL and UD-IQ4_XS (new upload, rev `d96201194f10`) fast, with 100K context and MTP if they fit. Kind: A/B experiment and fit study.

- Source: branch `s2-iq-signs`: S2, Q5_0/IQ4_XS repack (conversion `ebd76ced1`, then native Q5_0 `3e19b6adc`); build `build-conv-20261106`.
- Model: UD-Q3_K_XL and UD-IQ4_XS; "Q8_0 table" = the unsloth `engram/` Q8_0 n-gram table swapped in as shard 4 (`~/models/flashnext/`). Drafts: unsloth `mtp-...-Q8_0` sidecar, bartowski `mtp-...-Q4_0`.
- Settings: standard Flash Next, except where noted. The 8K and conversion rows used Q8_0 KV (before the F16-only rule); all fit and MTP rows use F16 KV.
- Workload: llama-bench PP2048/TG128 at depth 8192 (r3, or 2 alternating rounds of r2 for the A/B); llama-cli rows are the standard `flake.nix` workload, one run each.

Expert weights per quant (GGUF headers): UD-Q3_K_XL gate/up IQ3_XXS 76% / IQ4_XS 17% / IQ2_S 7%, down IQ4_NL 51% / Q5_0 48%; UD-IQ4_XS gate/up IQ3_XXS 52% / IQ4_XS 44%, down Q5_0 74% / IQ4_NL 24%. Before this change Q5_0 and IQ4_XS ran on the stock kernels.

| Build | Mode | PP t/s | TG t/s | Change | Accept | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| S2 | 27B Q8_0 plain @8K | 617.7 | 36.8 | reference | - | - |
| S2 | UD-IQ3_XXS plain @8K | 524.7 | 38.4 | - | - | - |
| S2 | UD-Q3_K_XL plain @8K | 509.0 | 39.7 | - | - | - |
| S2 | UD-IQ4_XS `-ncmoe 2` @8K | 426.1 | 30.5 | - | - | - |
| conv off | UD-Q3_K_XL @8K | 470.4 (one outlier) | 39.23 | - | - | PPL 1.9170 |
| conv on | UD-Q3_K_XL @8K | 541.0 | 40.12 | +2.3% TG | - | PPL 1.9150 |
| conv on + Q8_0 table | UD-Q3_K_XL @8K | 533.1 | 40.54 | +1% TG | - | PPL 1.9108 |
| conv off | UD-IQ4_XS `-ncmoe 3` @8K | 398.4 | 29.56 | - | - | PPL 1.9236 |
| conv on | UD-IQ4_XS `-ncmoe 3` @8K | 434.4 | 30.58 | +9% PP, +3.5% TG | - | PPL 1.9222 |
| native Q5_0 | UD-Q3_K_XL + Q8_0 table, plain, 100K, `-ub 256` | 409.1 | 42.1 | - | - | - |
| native Q5_0 | same, 64K | 416.4 | 43.8 | - | - | - |
| native Q5_0 | same, 32K | 417.0 | 43.9 | - | - | - |
| native Q5_0 | same, 32K, `-ub 512` | 472.6 | 41.2 | - | - | - |
| native Q5_0 | MTP2, Q4_0 draft on ROCm1, experts on CPU, 64K, `-ts 0.505,0.495` | 382.9 | 34.9 | slower than plain | - | - |
| native Q5_0 | **MTP2, full Q4_0 draft on ROCm1, 32K, `-ts 0.515,0.485`, `-ub 256`, `--spec-draft-n-ubatch 64`** | 390.5 | **51.5** | +17% vs plain 32K | - | - |

Verdict: the conversion is landed (exact for Q5_0, IQ4_XS scale rounded to f16 at about 2^-12; same accuracy as the stock kernels against the CPU reference). UD-IQ4_XS does not fit on the GPUs: 2-3 CPU expert layers cost about 20-25% TG, mostly sync and transfer per CPU layer. MTP with the draft experts on the CPU is validated wrong (slower than plain). The best working setups are plain Q3_K_XL at 100K (42.1 TG) and MTP2 at 32K (51.5 TG, no memory margin; MTP3 and `-ub 512` ran out of memory). The PPL text is too easy to separate the quants; use the unsloth KLD table for quality.

Memory per GPU at 100K, F16 KV, `-ub 256` (allocation log and sysfs peaks): weights 29.9 GiB (0.98 GiB of it mirrored: `ffn_gate_inp`, `hc_*`), main KV 1176 MiB (split by heads), QSA indexer key cache 588 MiB (one key head, so mirrored), compute 364 MiB, recurrent state 56 MiB, AllReduce staging 64 MiB; free 0.02-0.03 GiB. `llama_kv_cache` and compute buffer log lines report per-GPU sizes. A full Q4_0 draft needs about 2.3-2.5 GiB on one GPU; `-sm row` is gone upstream (`af546bb40`). The unsloth "shared" drafts lack `token_embd` and do not load in this tree. Not covered: full-length 32K prompts with MTP2, multi-slot server, acceptance rates. Raw: [flashnext-big-quants](old-ignore/2026-11-06/flashnext-big-quants/).

### 2026-11-06 S2: cheaper IQ sign handling on gfx906

Replace the emulated byte ops `__vcmpne4` + `__vsub4` with `sign_mask4` and `iq_apply_signs` (`(g ^ m) + (m & 0x01010101)`) in the IQ2_S mat-vec and the IQ2_S and IQ3_XXS MMQ tile loads. Kind: A/B experiment.

- Source: branch `s2-iq-signs` on `02980eb3d`; baseline `bc0ce9691` (build `build-catchup-20261105-213622`).
- Model: Flash Next UD-IQ3_XXS rev `c8b5954a88c2` (the pre-2026-11-04 upload).
- Settings: standard Flash Next.
- Workload: llama-bench PP2048/TG128, r5 alternating; one 96-token greedy `llama-completion` per build.

| Build | Mode | PP t/s | TG t/s | Change | Accept | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | plain | 541.45 +- 1.84 | 41.17 +- 0.11 | - | - | sha `d9c2f5346efc` |
| candidate | plain | 557.97 +- 1.09 | 43.27 +- 0.15 | +3.05% / +5.11% | - | same |

Verdict: landed. The ranges do not overlap, and output is byte-identical. Correctness: an exhaustive host check of all IQ2_S grid words x 256 sign bytes and IQ3_XXS cases found 0 mismatches (grid bytes are 0x01..0x3e, so the +1 never carries), and `test-backend-ops` passed all 115 IQ2_S/IQ3_XXS `MUL_MAT`/`MUL_MAT_ID` cases on ROCm0 (n = 1..64). Not covered: the new Flash Next upload, MTP, grouped MoE. The merge build's Flash Next numbers here (541 / 41.2) are above the 2026-10-29 rows (527.6 / 39.1), but that is not a matched comparison. Raw: [s2-iq-signs](old-ignore/2026-11-06/s2-iq-signs/).

### 2026-11-05 27B config sweep and 32K depth

Find the best 27B speculative mode, test a larger ubatch, and check speed at a 32K prompt. Kind: A/B experiment.

- Source: `bc0ce9691` (merge of official `4b77dee05`), build `build-catchup-20261105-213622`.
- Model: 27B Q8_0 rev `4ca720788d1e`; DFlash draft `z-lab/Qwen3.8-27B-DFlash2` Q8_0 rev `2d9571f8ce46`.
- Settings: standard 27B. 32K runs use ctx 40960. The ubatch test is llama-bench PP4096 with `-b 4096 -ub 2048,4096`.
- Workload: speculative runs are standard, one run each. The 32K prompt is the first 116,000 bytes of `src/llama-context.cpp` at `bc0ce9691` plus a request to explain it (about 32K tokens at 3.66 bytes/token). ubatch r3.

| Build | Mode | PP t/s | TG t/s | Change | Accept | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| candidate | plain, 2K | 587.4 | 36.8 | - | - | - |
| candidate | MTP2, 2K (two runs) | 584.5 / 582.1 | 64.1 / 63.9 | - | - | `dd228997b103` both |
| candidate | MTP3, 2K | 579.0 | 61.5 | -3.8% vs MTP2 | - | `39f8c45d0e87` |
| candidate | DFlash5, 2K (two runs) | 574.4 / 570.6 | 67.4 / 67.7 | +5.9% vs MTP2 | - | `9f2493efdd9b` both |
| candidate | DFlash7, 2K | 570.7 | 62.5 | -7.7% vs DFlash5 | - | same as DFlash5 |
| candidate | plain, 32K | 541.2 | 29.4 | -20% vs 2K | - | `b94840dcade5` |
| candidate | MTP2, 32K | 512.3 | 42.7 | -33% vs 2K | - | `c8ae52851843` |
| candidate | DFlash5, 32K | 512.0 | 39.9 | -41% vs 2K | - | same as plain |
| candidate | PP4096, ub 2048 / 4096 | 684.1 / 606.3 | - | -11.4% | - | - |

Verdict: DFlash5 is the fastest 27B mode at a 2K prompt and MTP2 at 32K; the crossover was not measured. Deeper drafts (MTP3, DFlash7) are slower. ubatch 4096 is validated wrong for 27B PP. Plain TG loses 20% at 32K, which motivates D1. Repeats at 2K varied by 0.2-0.3 t/s TG; noise at 32K was not measured. DFlash output matches plain greedy at 32K. Not covered: Flash Next, acceptance rates (`--simple-io` prints none), the crossover depth. Raw, including generated text: [config-sweep](old-ignore/2026-11-05/config-sweep/), [depth32k](old-ignore/2026-11-05/depth32k/).

### 2026-11-05 catch-up to official 4b77dee05

Check that the merge keeps 27B speed and output. Kind: regression screen.

- Source: `bc0ce9691` (merge of official `4b77dee05`; build made before the doc-only commits); baseline `build-ilp`, `build_commit e04bc2e38` (old master, stage A of the rebuild), run with `LD_LIBRARY_PATH=build-ilp/bin`.
- Model: 27B Q8_0 rev `4ca720788d1e`; DFlash draft `z-lab/Qwen3.8-27B-DFlash2` Q8_0 rev `2d9571f8ce46`.
- Settings: standard 27B. MTP2 adds `--spec-draft-backend-sampling`.
- Workload: standard, one run each, baseline then candidate per mode.

| Build | Mode | PP t/s | TG t/s | Change | Accept | Output |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | plain, llama-bench | 697.55 | 37.75 | - | - | - |
| candidate | plain, llama-bench | 696.88 | 37.90 | -0.1% / +0.4% | - | - |
| baseline | MTP2 | 598.0 | 64.0 | - | - | sha `dd228997b103` |
| candidate | MTP2 | 584.5 | 64.1 | -2.3% / +0.2% | - | same |
| baseline | DFlash5 | 584.0 | 66.8 | - | - | sha `9f2493efdd9b` |
| candidate | DFlash5 | 574.4 | 67.4 | -1.6% / +0.9% | - | same |

Verdict: no regression; speculative output is byte-identical between builds. The CLI prompt rate varies more than llama-bench PP (see below), so the -2% is not a signal. TP smoke checks passed at n=1 and n=2048; a 32-token `llama-completion` generation was coherent. Not covered: Flash Next (MoE PP after the MMQ tile layout change #31047), grouped MoE, multi-slot server. Raw, including generated text: [catchup-4b77dee05](old-ignore/2026-11-05/catchup-4b77dee05/).

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
