# Two-GPU TP profile: where PP and TG time goes

Historical record. Earlier recommendations and task deadlines describe the recorded tree; use the [current vr guide](../../README.md) for operating instructions. Exact pre-cleanup bytes are preserved in `before-cleanup.tar.gz`.

Date: 2026-10-12. Source: HEAD `8d73dc74e` working tree, binary `build-ilp` (reports build `5d6e2c412`). Model: `unsloth/Qwen3.8-27B-GGUF:Q8_0`. All runs: `-ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1`, `GGML_CUDA_TP_OVERLAP` unset unless stated. Raw outputs and analysis scripts: [results/2026-10-12-tp-profile](../../results/2026-10-12-tp-profile). Speculative decoding is out of scope.

## Summary

- The two GPUs are not equal. ROCm1 is the slower card and paces every TP step. An uneven split `-ts 56/44` gives +8% PP for free; TG is unchanged.
- PP: the repacked Q8 GEMM already reaches about 56% of the MI60 int8 dot-product peak. The main loss is AllReduce PCIe traffic that is not overlapped with compute (about 0.78 s of a 3.9 s ubatch).
- TG: the repacked matvec streams weights at about 715 GB/s and takes about 20 of 33 ms per token. Most of the rest is about 1400 small kernels per token, AllReduce latency and the host sampling round trip.

## Machine

| Device | Card | CUs | Power cap | Host link |
| --- | --- | ---: | --- | --- |
| ROCm0 (`03:00.0`) | Radeon Instinct MI60, VBIOS 113-D1630600-107 | 64 | 210 W (max 225 W) | CPU root port, PCIe Gen3 x8 |
| ROCm1 (`87:00.0`) | MI50 32 GB with Radeon Pro VII VBIOS 113-D1640200-043 (has DP outputs) | 60 | 170 W (max 178 W) | PCH root port, PCIe Gen4 x4, behind DMI |

Both GPU-side links report x16, but the upstream root ports are x8 Gen3 and x4 Gen4. The host-staged AllReduce is therefore limited to roughly 7 GB/s per direction per GPU. There is no direct P2P path between the cards.

The ROCm1 identification is from VBIOS and display outputs; confirm the CU count with `rocminfo` if it matters.

## Benchmarks

`llama-bench -b 2048 -ub 2048 -p 2048 -n 128 -r 2` unless stated.

| Config | PP | TG128 |
| --- | ---: | ---: |
| TP `-ts 50/50` (default) | 546.7 | 30.22 |
| TP `-ts 53/47` | 579.9 | 30.42 |
| TP `-ts 56/44` | **591.3** | 30.17 |
| TP `-ts 58/42` | 570.9 | - |
| TP `-ts 60/40` | 561.0 | - |
| TP `-b/-ub 4096`, `-ts 50/50`, PP4096 | 526.8 | - |
| TP `-b/-ub 4096`, `-ts 56/44`, PP4096 | 563.4 | - |
| TP `GGML_CUDA_TP_OVERLAP=1`, `-ts 50/50` | 543.7 | - |
| TP `GGML_CUDA_TP_OVERLAP=1`, `-ts 56/44` | 587.4 | - |
| Single GPU ROCm0, `-sm none`, `-b/-ub 512`, PP512 | 437.2 | 22.76 |

- Single-GPU TG of 22.76 t/s is about 660 GB/s of weight streaming. Perfect 2-way TP would reach about 45 t/s; measured TP is 30.2.
- `GGML_CUDA_TP_OVERLAP=1` gives no gain and printed no overlap log line in this run. It is not known whether it engaged.
- ubatch 4096 is slower than 2048 for this test.
- These are llama-bench numbers. Confirm `-ts 56/44` with the [CLI workflow](../../README.md#benchmark-workflow) before adopting it.

## PP profile (one 2048-token ubatch, `-ts 50/50`)

`rocprofv3 --kernel-trace --memory-copy-trace` on `llama-bench -p 2048 -n 0 -r 1` (518 t/s under the profiler). Window: the last ubatch, 3.88 s.

| | ROCm0 | ROCm1 |
| --- | ---: | ---: |
| `mmq_gemm_repacked` (493 calls) | 1698 ms | 2018 ms |
| `gated_delta_net_chunked_cuda` (48) | 207 ms | 243 ms |
| `concat_non_cont` (48) | 71 ms | 75 ms |
| `quantize_mmq_q8_1` (493) | 66 ms | 67 ms |
| all kernels | 2259 ms | 2625 ms |
| AllReduce D2H copies | 378 ms | 378 ms |
| AllReduce H2D copies | 397 ms | 383 ms |
| kernel or copy active | 3035 ms | 3386 ms |

- ROCm1 runs the same GEMM work 19% slower than ROCm0. That is why the uneven split helps.
- GEMM rate: about 33 int8 TOPS on ROCm0, about 56% of the dp4a peak. Little is left in the GEMM itself.
- The copies are mostly serialized with compute on ROCm1. Hiding them is the largest PP opportunity, with a ceiling of about +20%.
- Each AllReduce moves 2048 x 5120 x 2 bytes (BF16 wire) per direction; 127 per ubatch is about 2.7 GB each way per GPU.

## TG profile (per token, `-ts 50/50`)

`rocprofv3` on `llama-bench -p 0 -n 64 -r 1` (29.2 t/s under the profiler, 34.3 ms per token in the steady window).

| Item | Per token | Notes |
| --- | ---: | --- |
| `mul_mat_vec_q8_0_repacked` (432 calls) | 19.3 ms | About 715 GB/s average |
| of which fused gate+up (64 calls) | 8.5 ms | About 712 GB/s for 2 x 8704 x 5120 Q8_0 per GPU |
| of which 2-workgroup launches (96 calls) | 0.65 ms | 6.6 us each; tiny projections, probably GDN alpha/beta |
| `ggml_cuda_ar_kernel` (128 calls) | 4.1 ms ROCm0 / 1.7 ms ROCm1 | About 13 us intrinsic; ROCm0 also waits for ROCm1 |
| `rms_norm_f32` (305 calls) | 1.9 ms | 129 of them are 1024-thread full-row norms at 8.6 us |
| `quantize_q8_1` (257 calls) | 1.1 ms | |
| LM head `mul_mat_vec_q` (1 call) | 0.95 ms | About 700 GB/s, not repacked |
| `k_bin_bcast` (176 calls) | 0.75 ms | Mostly residual adds after AllReduce |
| other small kernels | about 3 ms | gated unary, GDN, flash attention, get_rows, rope, conv |
| idle: AllReduce -> next kernel | 1.0-1.5 ms | 127 gaps of 8-12 us |
| idle: LM head -> next token upload | about 1.2 ms | Host logits read and sampling |

- About 1870 kernels per token per GPU; GPUs are idle only about 3 ms per token, so HIP graphs already hide launch overhead. The cost is the minimum duration of each small kernel and the dependency gaps.
- The small AllReduce already runs on the compute stream, so the gap after it is not a cross-stream wait.
- llama-bench TG does not include sampling. The sampling round trip appears only in CLI/server runs.

## Plan

Order follows expected gain per effort. Each item keeps an off switch and needs op tests plus CLI output comparison.

| ID | Change | Target | Estimate | Applies to |
| --- | --- | --- | --- | --- |
| T0 | `-ts 56/44` | PP | +8% (measured) | Any TP model on this pair |
| T2 | Backend sampling (`-bs`) for greedy decode | TG | about 1.2 ms per token | Any model |
| T1 | Fuse residual add, RMS norm and Q8_1 quantize into the AllReduce epilogue | TG | 2.5-3.5 ms per token (+8-11%) | TP models with add -> norm after the reduction |
| T4 | Merge tiny 2-workgroup matvecs | TG | about 0.3-0.5 ms per token | Qwen3.5/3.8 GDN layers |
| T3 | Tune fused gate+up repacked matvec toward 850 GB/s | TG | up to about 1.5-2 ms per token | Dense Q8_0 repacked FFN |
| P2 | Overlap AllReduce transfers with compute (token slabs) | PP | up to about +20% | Any TP model |
| P3 | Remove `concat_non_cont` in GDN prefill; review chunked GDN | PP | up to about 2-6% | Qwen3.5/3.8 GDN layers |

Lower priority from the [2026-10-11 shortlist](2026-10-11-gfx906-optimization-ideas-q8-q4-q6.md): A is covered by T1; B has a small ceiling (`quantize_mmq` is 1.7% of PP); E does not match the profile (about 2500 kernels per ubatch); C, D and G target Q4/Q6 rather than this Q8 path.

Hardware: raising ROCm1's cap to its 178 W maximum is a one-line root sysfs write. Restoring a stock MI50 VBIOS may help more but removes the display outputs.

## Reproduce

```sh
rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d prof_tg -o tg -- \
  build-ilp/bin/llama-bench -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 -p 0 -n 64 -r 1
python3 vr/results/2026-10-12-tp-profile/kernel_summary.py prof_tg/tg_kernel_trace.csv
python3 vr/results/2026-10-12-tp-profile/gap_summary.py prof_tg/tg_kernel_trace.csv
```

For PP use `-b 2048 -ub 2048 -p 2048 -n 0 -r 1` and `pp_summary.py`.

## Results of the follow-up changes (2026-10-12, uncommitted)

Baseline: `../mx-base/build-ilp` (HEAD `8d73dc74e`). New: `build-ilp` with the changes below. `-ts 50/50`, `GGML_CUDA_TP_OVERLAP` unset.

| Run | Baseline | New | Change |
| --- | ---: | ---: | ---: |
| llama-bench PP2048 (r3) | 551.8 | 649.2 | +18% |
| llama-bench TG128 (r3) | 30.16 | 35.81 (36.8 with 8-row default) | +19-22% |
| CLI 256 tokens, PP | 460.8 | 539.5 | +17% |
| CLI 256 tokens, TG | 29.9 | 35.2 (rows 16) | +18% |

CLI generated text is byte-identical to the baseline. `test-backend-ops` passes for CONCAT (with 3 new transposed cases), RMS_NORM, ADD and MUL_MAT.

| Change | Files | Switch (default on) | Measured effect |
| --- | --- | --- | --- |
| Host-staged AR kernel takes token/slot from a device counter, no host sync; split prepare/launch_rank so the meta whole-token graph captures without P2P | `allreduce.cu/.cuh`, `ggml-cuda.cu` | none | TG 30.2 -> 34.8 |
| Fused residual add + RMS norm + mul, also fills the Q8_1 activation cache for the next mat-vec | `norm.cu/.cuh`, `ggml-cuda.cu` | `GGML_CUDA_FUSE_ADD_RMS_NORM`, `GGML_CUDA_Q8_1_PRODUCER` | part of TG 34.8 -> 36.2 |
| Q8_0 repacked mat-vec rows per workgroup, default 8 (was 16) | `q8_repack/mul-mat.cu` | `GGML_RP_MMV_ROWS=16/8/4` | TG 36.2 -> 36.8 |
| Copy-engine AR: H2D on its own stream, overlaps D2H | `allreduce.cu` | `GGML_CUDA_AR_DUPLEX` | PP 599 -> 647 |
| Tiled transpose for dim-0 concat with transposed src1 (GDN conv input) | `concat.cu`, test | `GGML_CUDA_CONCAT_TRANSPOSED` | part of PP gain |

Notes:

- The uneven split no longer pays after these changes: `-ts 56/44` gives PP 647 / TG 34.6 versus 649 / 35.8 at 50/50.
- `GGML_CUDA_TP_OVERLAP=1` had no effect because its HIP implementation is in an `#elif defined(GGML_USE_HIP)` branch of `allreduce.cu` that HIP never compiles (HIP takes the first `#if !defined(GGML_USE_MUSA)` branch).
- `-bs` backend sampling is refused under `-sm tensor` (`llama_context::set_sampler`).
- Not done: merging the GDN alpha/beta mat-vecs (non-adjacent nodes, at most about 1%); slab overlap of AR with the producing GEMM in prefill; producer-side Q8_1 for the SwiGLU output; the about 1-2 ms per-token host gap before each token graph launch.

Other models: the AR token-graph and duplex changes apply to any 2-GPU TP model. Fusion A needs the pre-norm add -> RMS_NORM -> MUL pattern (Muse Glimmer: yes; Gemma 4: no, it uses post-norms that the existing RMS_NORM+MUL+ADD fusion already covers). The rows knob applies to Q8_0 weights through the repack path (Muse Glimmer Q8_0: yes; the local Gemma 4 31B Q4_1 is not repacked). The concat change only matters for GDN models (Qwen3.5/3.8 family).

### Slab-overlapped prefill AllReduce

The meta backend marks the node each AllReduce reads (`ggml_backend_ar_slab_mark`). `ggml_cuda_mul_mat` computes that mat-mul in token slabs and records an event per slab; `ggml_cuda_ar_allreduce_slabs` converts and copies each slab to the host as soon as its event fires, so the transfer overlaps the following slabs' GEMM. Switch: `GGML_CUDA_AR_SLABS` (default 4; 0 or 1 disables). Only multi-token (>= 512) F32 mat-mul outputs on the BF16 copy-engine path use it; TG is unchanged.

| `GGML_CUDA_AR_SLABS` | 0 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| llama-bench PP2048 (r2) | 651.1 | 670.7 | 682.0 | 680.1 |

CLI 256 tokens: 558.2 PP / 36.0 TG (was 539.5 / 35.2), generated text identical to the baseline. The +4.7% is below the +8-12% estimate; a PP trace would show how much transfer is still exposed (last slab, attention/GDN output projections whose GEMM is shorter than their transfer, and the BF16 conversion competing with the GEMM).

### Tensor split after the slab overlap

llama-bench r2, PP2048 / TG128: `50/50` 683.6 / 36.8, `53/47` 702.2 / 35.6, `56/44` 704.1 / 34.8. The PP gain from giving the faster MI60 more rows is back (+3%) now that the AllReduce is mostly hidden, but TG loses because TG is bandwidth bound and both cards have the same HBM bandwidth. The split is fixed at load time, so choose per workload.

## DFlash2 quick look (2026-10-12, current build)

CLI, `-n 256`, `--spec-draft-n-max 7`: 423.2 PP / 39.3 TG (plain decode about 36). Mean acceptance length 3.86 (66 steps for 256 tokens); acceptance by position 0.80, 0.64, 0.47, 0.33, 0.26, 0.21, 0.15.

Per step about 98 ms: target verify of 8 tokens about 57-62 ms, draft generation about 27 ms (`dur g` 1778 ms / 66), draft prepare about 17 ms (`dur p` 1186 ms / 70).

Target batch cost (llama-bench `-p N -n 0`): 1 token 31.6 ms, 2: 34.1, 3: 38.5, 4: 40.5, 5: 44.9, 8: 57.1 ms. Each extra token costs about 3.6 ms; a bandwidth-bound mat-vec should add well under 1 ms per token. The narrow-batch (2-8 token) repacked path is the main loss and also limits MTP.

The draft runs tensor-parallel across both GPUs because it shares the target's output head: `-devd ROCm0` aborts with `pre-allocated tensor (output.weight) in a buffer (Meta(ROCm0,ROCm1)) that cannot run the operation`. A tiny model split over two GPUs pays an AllReduce per layer.

Levers, in order: (1) a narrow-batch mat-vec that reads each weight once for all 2-8 columns at near single-token cost; (2) draft on one GPU with a mirrored output head (about 1.35 GB Q8_0) or captured in a token graph; (3) `--spec-draft-n-max` 4-5, since positions 5-7 accept at 0.26 or less. If verify(8) dropped to about 35 ms and draft+prepare to about 10 ms, the same acceptance would give about 85 t/s.

## Final boss: speculative decoding push (2026-10-12, 01:50-03:30)

Workload: `llama-cli -f flake.nix -n 4000`, greedy (`--temp 0 --top-k 1 -s 42`), `-sm tensor -dev ROCm0,ROCm1 -fa on -b 2048 -ub 2048 -c 32768`. Target: 85 t/s TG with CLI PP >= 543.

| Arm (n 4000) | PP t/s | TG t/s | Mean acc len | Text vs plain |
| --- | ---: | ---: | ---: | --- |
| plain, start of session (cold GPUs) | 564.2 | 35.6 | - | reference |
| dflash7, start | 420.9 | 40.1 | n/a | diverges at char 1436 |
| dflash5, start | 420.6 | 45.4 | n/a | diverges at char 1436 |
| mtp3, start | 510.6 | 48.9 | n/a | diverges at char 1436 |
| plain, end (run right after mtp2) | 558.3 | 31.7 | - | reference |
| **mtp2 + `--spec-draft-backend-sampling`, end** | **545.5** | **56.7** | 2.60 | diverges at char 2548 of 13906 |
| mtp2 + `--spec-draft-backend-sampling`, confirm run 03:11 | 553.2 | 60.3 | 2.60 | same |
| plain right after it | 556.3 | 32.3 | - | reference |
| mtp2, end (earlier runs) | 546.5-542.9 | 57.7-56.1 | 2.60 | same |
| mtp3, end | 548.7-557.1 | 53.8-55.5 | 2.86 | diverges at char 1232 |
| dflash5, end | 428.7 | 54.5 | 3.33 | diverges at char 1232 |
| dflash7, end | 425.3 | 51.6 | 3.59 | - |

Best: MTP depth 2, 56.7-60.3 t/s TG (1.8x plain run back to back), PP 545-553 (on the limit; run-to-run PP noise is about +-1.5%). 85 t/s was not reached. Plain TG at 4000 tokens fell from 35.6 to 32 over the session; llama-bench pp1 did not change (28.2-28.5 ms), and the GPUs idle at 71-72 C, so this looks thermal. Greedy text of every speculative arm differs from plain at some near-tie (char 1232-2548): batched verify uses other kernels (narrow mat-vec, multi-row FA/GDN) than 1-token decode, so the logits are not bit-identical. Spec arms agree with each other up to the same points.

Why 85 is out of reach here: one weight pass costs 28 ms (1 token) and verify(3) about 36 ms. MTP2 step is about 45 ms: verify(3) 36, MTP catch-up 1.8, two drafts 1.6 each (bandwidth bound: MTP block + head), about 5 ms host gaps. L = 2.6 gives 57.7 t/s. 85 needs step/L <= 11.8 ms; with verify(k+1) >= 35 ms that needs L >= 3.5 at near-zero draft cost. DFlash reaches L 3.3-3.6 but its draft still costs about 17 ms per step (two TP draft decodes that cannot be captured because inject and generate alternate graph shapes, plus host gaps of about 10 ms).

What each change gave:

| Change | Switch (off) | Effect |
| --- | --- | --- |
| Planar activations for the 2-8 column repacked mat-vec (`rp_quantize_q8_1_planar`, q8_1 cache variant 100) | `GGML_RP_NC_PLANAR=0` | pp4 40.6 -> 37.2 ms |
| 16-lane rows (32-lane for mid-size) for 4-8 columns, `__launch_bounds__` on the nc kernel | `GGML_RP_NC_L16=0` | pp8 57.0 -> 45.7 ms, pp6 48.7 -> 40.8 ms |
| Lazy draft prompt replay: MTP/DFlash replay moves from `begin()` to the first `draft()`/`process()` | `LLAMA_SPEC_LAZY_PREFILL=0` | mtp3 CLI PP 512 -> 549 (first token is sampled before the replay; total work unchanged, about 0.2 s moves into generation) |
| Repacked MTP blocks and output head (`il >= n_layer` used the plain buft list) | `LLAMA_MTP_REPACK=0` | mtp3 n512 TG 52.9 -> 54.2 |
| Repacked sidecar draft weights | `LLAMA_DRAFT_REPACK=0` | dflash5 draft gen 20.6 -> 17 ms/step, TG 42.9 -> 45.9 (n512) |
| No lane broadcast for top-k over `result_output` | `GGML_META_TOPK_BCAST_LOGITS=1` | needed for a fully mirrored DFlash2 draft graph; no TG change alone |
| Per-lane HIP graphs stay on until a token graph is about to be captured | `GGML_META_INNER_GRAPHS_EAGER_OFF=1` | DFlash draft launch time per call 4.0 -> 2.8 ms |
| `--spec-draft-backend-sampling` | flag | about +0.6 t/s for mtp2 |

Tried, not kept: RPL 4 rows per lane (`GGML_RP_NC_RPL=4`, spilled at the 64 VGPR default, no gain after launch bounds); dense up/gate fusion for 2-8 columns (`GGML_RP_NC_FUSION=1`, slower at 6-8 columns); 2-3 columns through the 4-8 shapes (`GGML_RP_NC_23=1`, no change); mirrored output head for DFlash (`LLAMA_DFLASH_MIRROR_OUTPUT=1`, TG 45.9 -> 43.1, the target head read doubles); `--spec-draft-p-min` 0.5/0.6 (TG 30.7/28.6, variable verify widths); MTP depth 1 (47.7) and 4 (47.2 n512).

Prefill costs left: MTP adds about 2-3% to CLI PP (the unmasked pre-norm hidden extraction during prefill); the lazy replay itself is about 0.2 s (KV-only MTP decode of 2086 rows, about 45 ms fixed overhead per chunk call). DFlash prefill is still 425-429 t/s: the tap extraction during the target prefill, not the replay, is the cost.

Next step: DFlash draft cost. Make the draft graph capturable (keep one graph per shape for inject and generate, or pad inject to a fixed width) and drop the host gaps; with draft+prepare at about 5 ms, dflash5 (L 3.33) would give about 70 t/s. Then fix DFlash prefill (tap extraction).

## Round 2: DFlash prefill and step cost (2026-10-12, 11:58-13:00)

Same workload and flags as above. Screening used `-n 64`/`-n 256`; the `-n 4000` table was run once.

| Arm (n 4000) | PP t/s | TG t/s | Mean acc len | Text vs plain |
| --- | ---: | ---: | ---: | --- |
| plain | 568.0 | 34.8 | - | reference |
| **mtp2 + `--spec-draft-backend-sampling` (best, unchanged)** | **546.0** | **55.1** | 2.60 | diverges at char 2548 of 13906 |
| dflash n-max 4 | 535.2 | 50.9 | 3.13 | diverges at char 1232 |
| dflash n-max 5 | 533.5 | 51.4 | 3.33 | diverges at char 1232 |
| dflash n-max 6 | 534.4 | 49.9 | 3.46 | diverges at char 1232 |

At `-n 64`/`-n 256` DFlash PP is now 539-548 (was 428-436, plain 570-575). At `-n 4000` it measured 533-535, so it is on the 543 limit within run-to-run noise but not reliably above it. TG did not improve; MTP2 is still the best arm. 85 t/s was not reached.

### Why DFlash prefill was slow

- The position-indexed tap accum (deferred draft prompt encode) never engaged. Qwen3.8 (`qwen35` arch) uses M-RoPE, so `ubatch.n_pos == 4`, and `extract_layer_inputs` required `n_pos == 1`. Every prompt chunk took the eager path: a target synchronize, a host gather from `embd_layer_inp`, and a 2044-row draft inject (396 ms) inside the prompt time.
- The server splits the prompt as 42 + 7 + 2044 + 4 tokens (checkpoint breaks). Chunks below 16 rows were always eager, and an eager chunk first replays all deferred chunks, so the 4-token tail alone pulled the full replay into the prompt.

### Changes (all default on)

| Change | Switch (off) | Effect |
| --- | --- | --- |
| Tap accum accepts M-RoPE text rows (first position section) | `LLAMA_TAP_ACCUM_MROPE=0` | Deferral works; prompt gather 130 -> 21 ms, prompt tgtsync 206 -> 0 ms |
| Accum written for every ubatch size (gate 16 -> 1) | `LLAMA_TAP_ACCUM_MIN=16` | Needed for the short prompt tails |
| DFlash defers short chunks while in the prompt (after a >= 16-row chunk, until `begin()`) | `LLAMA_DFLASH_DEFER_TAIL=0` | The 4-token tail no longer forces the replay; the replay (about 0.67 s) runs at the first draft |
| Skip the second D2H tap copy when the chunk lands in the accum as one run | `LLAMA_TAP_EMBD_SKIP=0` | Halves tap D2H for prompt chunks |

Together: df5 `-n 64` PP 433.8 -> 547.8 with the same acceptance (L 3.71).

### DFlash step cost (not fixed)

rocprofv3 over the generation phase (n-max 5, about 63 ms per step under the profiler): GPU busy 67%. Per step on ROCm0: verify(6) about 45 ms busy, draft inject 37 kernels (0.7 ms), draft generate 313 kernels (about 6 ms), and about 10 ms of host gaps spread over 7-8 points (sampling and accept 2.3 ms, inject -> generate 3.2 ms, output readback and input copies about 1 ms each). The draft graph is rebuilt on every call because inject and generate alternate shapes, so it gets a new uid and the meta token graph never replays it.

Tried and removed: a content-keyed meta token-graph cache (hash of nodes, shapes, data pointers and op params instead of the uid). It replayed the draft but acceptance fell from 2.95 to 1.1-1.6, even when only the first subgraph was captured (`GGML_META_TG_LIMIT=1`) and with the cache cleared when tmp buffers grow. MTP was unaffected (L 2.32 either way). The cause was not found.

Other n256 screens: `GGML_META_INNER_GRAPHS_EAGER_OFF=1` for DFlash 37.6 -> 38.4 TG (noise level); MTP2 with `-ts 52,48` PP 557.7 vs 549.4, TG 50.7 vs 50.6 (noise level).

### Why 85 t/s is out of reach here

85 t/s needs a step of at most L x 11.8 ms. Verify alone is about 41 ms for 6 tokens and 36 ms for 3 tokens, so even a free draft needs L >= 3.5 (n-max 5 gives 3.33 at 4000 tokens; n-max 7 gives 3.59 with verify(8) about 46 ms). With the draft and its gaps at about 17 ms per step, DFlash lands at 50-51 t/s and MTP2 at 55-60.

Next step: cut verify(k) itself (about 23 ms of 2-8 column mat-vec per step, 5.6 ms of AllReduce on ROCm0 waiting for the slower ROCm1, many small set_rows/copy kernels). After that, give the DFlash draft one stable graph shape (for example a fused inject+generate decode), so the meta token graph can capture it under its normal uid.

## Round 3: verify cost and MTP step (2026-10-12, 12:33-13:00)

Screening at `-n 256`; one `-n 4000` run of the best config.

| Arm | PP t/s | TG t/s | L | Text vs plain |
| --- | ---: | ---: | ---: | --- |
| mtp2 + `--spec-draft-backend-sampling`, n 4000 (current build) | 558.3 | 61.6 | 2.60 | diverges at char 2548 (same as before) |
| same config, round 2 sweep | 546.0 | 55.1 | 2.60 | same |

The 55.1 -> 61.6 gap is mostly run-to-run variance (round 1 saw 56.7-60.3 for this config). The one kept change only touches 6-8 column verifies. 85 t/s was not reached.

### Where verify time goes (rocprofv3, llama-bench `-p 1/3/6`, GPU busy per evaluation on ROCm1)

| | p1 | p3 | p6 |
| --- | ---: | ---: | ---: |
| all kernels | 29.7 ms | 33.8 ms | 39.6 ms |
| mat-vec | 19.9 | 22.3 | 25.9 |
| AllReduce kernel | 1.7 (ROCm0: 4.0) | 2.5 (5.7) | - |
| Q8_1 quantize | 0.6 | 1.2 | 1.1 |

Mat-vec is already close to the weight-bandwidth floor: 3 columns cost 1.12x and 6 columns 1.3x of one column. The 1-column pass is about 13 GB per GPU in 19.9 ms (about 650 GB/s on the MI50), so verify cannot get much below one decode plus a few ms. ROCm0 waits about 2.3 ms per token in the AllReduce for the slower ROCm1. MTP2 step under rocprof: verify(3) 35.4 ms busy, MTP catch-up plus two drafts about 5 ms, host gaps about 3.5 ms (44 ms total).

### Changes

| Change | Switch (off) | Effect |
| --- | --- | --- |
| 2-8 column repacked mat-vec stages planar activations in an LDS tile per workgroup | `GGML_RP_NC_LDS=0` | 6-column kernel 75.4 -> 73.1 us (-3%); DFlash n-max 5 n256 TG 37.8 -> 38.4 |

### Tried and reverted

- `#pragma unroll 2` on the nc mat-vec loop: 6-column mat-vec 25.1 -> 28.9 ms per evaluation (slower).
- Planar Q8_1 output from the fused add+rms_norm for 2-8 rows (so the mat-vec skips its quantize): quantize launches halved, but the norm kernel (one block per row, 3-6 blocks) got slower by the same 0.6 ms; no net gain.
- MTP fused catch-up: stage the verify rows in `process()` and decode them together with the first draft row in `draft()`, so each step runs one MTP decode less. This needed a fix: the MTP graph sets `t_h_nextn` before the `inp_out_ids` crop, so masked extraction read the first row instead of the output row. Without the fix L fell to 2.0. With it L was 2.30 (vs 2.32) but TG was 51.0 vs 51.8: the separate catch-up decode was already queued behind the verify while the host sampled, so it cost nothing.

### Not done

- DFlash draft as one replayable graph: no new attempt. Round 2 showed that replaying a rebuilt draft graph gives wrong drafts even with only its first subgraph captured, and the cause is still unknown. Padding alone cannot fix it, because inject and generate are two different graphs.
- DFlash PP at n 4000 (533-535) and the n-max re-tune: not repeated; DFlash stays below MTP2 on TG.

Next step: the remaining levers are small (about 1-2% each): the AllReduce wait on ROCm0 (split the work by GPU speed only for the mat-vecs), GDN small-op fusions (alpha add+softplus+mul, scale), and the 108 copies and 83 fills per token. Reaching 85 t/s with MTP2 (L 2.6) needs a 30.6 ms step. That is about one plain decode, so it is not reachable on this hardware without higher acceptance.

### N-gram drafting (2026-10-12, after round 3)

`-n 512`, `--spec-draft-backend-sampling`: MTP2 555.2 PP / 54.4 TG; `ngram-mod,draft-mtp` 559.6 / 54.7; `ngram-simple,draft-mtp` 553.1 / 48.7; `ngram-mod` alone 567.1 / 35.6 (same as plain, so almost no n-gram hits). The final-boss output is reasoning text that rarely copies `flake.nix`, so lookup drafting does not help.
