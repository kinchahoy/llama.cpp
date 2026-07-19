# gfx906 Q4_K prefill — architecture revamp notes & session handoff

Date: 2026-07-02. Status: analysis captured mid-work before system suspend.
Author context: written for the next engineer/session. Not yet validated by any build.

> UPDATE 2026-07-02 (measured): the pp8192 kernel-time split and per-kernel
> counters have now been RUN (rocprofv3). Raw data + reading:
> `vr/bench-results/gfx906-pp8192-kernel-profile-2026-07-02/`. Headlines that
> supersede parts of this doc:
> - Quantized matmul is 80% of prefill (Q4_K 54%, Q6_K 20%, Q5_K 6%).
>   Flash Attention is 4%, DeltaNet 8%. So section 5.5 (FA as the pp8192
>   Amdahl ceiling) is FALSE at this depth; the matmul focus (5.1) is correct.
> - The bottleneck is register-pressure-limited OCCUPANCY, not LDS. Measured:
>   Q4_K 128 VGPR -> 2 waves/SIMD, Q6_K 140 VGPR -> 1 wave/SIMD; VALU busy
>   only 60%/44%; LDS bank conflicts ~0.1%; memory ~idle. Section 3's
>   "hide LDS waits" framing is wrong: LDS is not the constraint. The lever
>   is REDUCING VGPRs to raise waves/SIMD.
> - Q6_K (20%, most occupancy-starved) is the largest under-worked target
>   after Q4_K.

## 1. What was delivered this session (done)

- New patch: `vr/patches/gfx906-q4k-mmvq-paired-fusion.patch` (applies clean via
  `git apply --check`, verified). Adds a paired value+gate Q4_K MMVQ helper that
  loads the shared Q8_1 activation once; math is bit-for-bit identical to two
  `vec_dot_q4_K_q8_1` calls. Q4_K TG only, exact-gfx906 gated. **Hypothesis, not
  guaranteed** — README warns the compiler may already dedupe the activation
  load; revert if the TG screen is neutral.
- Tester handover: `vr/patches/gfx906-q4k-mmvq-paired-fusion.HANDOVER.md`.
- Source tree left CLEAN (patch not applied in-tree; it is a candidate-only
  experiment per the vr/ workflow).

## 2. The 400 t/s question — honest framing

- Dual-GPU Q4_K_M pp8192 is ALREADY ~390 t/s (RESULTS.md). "400" is ~2.5% away
  there and needs no revamp.
- Single-GPU Q4_K pp8192 = 189 t/s. Hitting 400 single-GPU is ~2.1x. That is the
  hard target this document addresses.

## 3. Roofline / bottleneck diagnosis (the load-bearing math)

- pp end-to-end FLOP/token ≈ 2·N_params. 189 t/s on ~27B ≈ 10.2 eff TFLOP/s
  end-to-end. Operator Q4_K n=512 ≈ 15.9 TFLOPS (RESULTS baseline, MI50-0).
- gfx906 DP4A roofline (MI50-0, 225W, ~1.4–1.5 GHz sustained):
  3840 lanes × 4 INT8 MAC × ~1.45e9 ≈ 22 TMAC/s ≈ **~44 INT8 TOP/s**.
- So the Q4_K operator runs at ~36% of the DP4A roofline; end-to-end ~23%.
- 400 t/s single-GPU = ~21.6 eff TFLOP/s end-to-end = the Q4_K operator must
  reach ~30–35 TFLOPS (≥2x, ~70–80% of roofline) AND attention/other ops must
  not become the new ceiling.
- CRITICAL DIAGNOSIS (now measured, see UPDATE at top): Q4_K MMQ is
  **register-pressure-limited occupancy-bound**, not VALU-saturated and not LDS
  bound. Counters: VALU busy 60%, 100% lane utilization, memory ~idle, LDS bank
  conflicts ~0.1%, and 128 VGPR/lane capping it at 2 waves/SIMD (of 10). The
  VALU idles ~40% waiting on instruction/load latency that too few resident
  waves can hide. This explains the rejected experiments: min_blocks=1 (-34%),
  y64 tile (-9%), 2-accumulator (-2.4%) all ADD live VGPR state and lower
  occupancy further. The revamp must **cut VGPRs to raise waves/SIMD**; the
  earlier "hide LDS waits" framing was optimizing a non-bottleneck.

## 4. Model facts (confirmed from GGUF metadata)

- arch `qwen35`, block_count 65, embedding 5120, ffn 17408 (dense FFN size),
  heads 24 / kv 4 (GQA), context 262144, 866 tensors (MTP variant → extra
  multi-token-prediction/nextn tensors likely).
- OPEN ITEM: confirm dense vs MoE. `strings | grep exps` timed out on the ~big
  blob. Run `gguf_dump.py` / `llama-gguf` on the file to check for
  `*_exps.*` tensors and `qwen35.expert_count`. If MoE, prefill is
  `MUL_MAT_ID`-dominated and the repack in §5.1 must cover the expert path too;
  the roofline argument is unchanged.

## 5. Proposed architecture revamp — ranked by expected leverage

### 5.1 Offline weight repack into a GCN/DP4A-native Q4_K layout  (HIGHEST)
Today MMQ unpacks 4-bit nibbles and decodes 6-bit scales/mins **inside the hot
loop**, competing with DP4A for VALU and VGPRs and forcing LDS shuffles (the
stride-9 patch only mitigates the symptom). Introduce a gfx906-specific
**repacked Q4_K weight type** (analogous to ggml CPU `_r4/_r8` repacks, and the
natural conclusion of the accepted precompute patch):
- At model load, convert each Q4_K block into (a) pre-shuffled int8-ready quants
  laid out so a wave's 32 weights are contiguous and **bank-conflict-free by
  construction**, and (b) a separate compact array already in final
  `dm*scale` / `dm*min` form.
- The prefill kernel becomes ~pure DP4A + one FMA: no in-loop unpack, fewer
  VGPRs → **higher occupancy** (exactly what this machine needs), and the LDS
  conflict problem is removed structurally instead of by stride tuning.
- Expected: the single biggest mover; plausibly 189 → ~280–320 t/s single-GPU.
- Cost/risk: new tensor type + load-time conversion + dedicated MMQ kernel;
  keep the generic path for non-gfx906; validate MUL_MAT 1103/1103 and a repack
  round-trip correctness test.

### 5.2 Software-pipelined / double-buffered LDS staging  (pairs with 5.1)
GCN has no async global→LDS copy. Current loader does load → barrier → compute.
Restructure to prefetch tile N+1 while DP4A-ing tile N (double LDS/register
buffer). This hides memory latency in software, which then lets you **lower
occupancy and raise per-thread reuse** — breaking the occupancy-vs-reuse
deadlock that killed the y64 experiment. Best done together with 5.1.

### 5.3 Ground-up wave64 MMQ tiling
Upstream MMQ tiling is warp32 + tensor-core shaped and retrofitted to wave64.
A native wave64 redesign (64-wide K reduction, LDS sized to 64KB/CU and
gfx906's 32-bank LDS, 256 VGPR/lane budget) can raise DP4A duty cycle. Highest
effort, most invasive; do after 5.1/5.2 prove the direction.

### 5.4 Split-K for CU-starved tails  (low effort, modest)
Wide prefill usually has enough parallelism, but the last tile wave can
underfill 60 CUs. Split-K + partial-sum reduction recovers that tail. Cheap,
complements the above; not a primary mover for PP.

### 5.5 Flash Attention tile tuning (REFUTED as a prefill priority)
Superseded by measurement: at pp8192, flash_attn_tile is only 4% of kernel
time (DeltaNet 8%). FA is NOT the Amdahl ceiling here. Deprioritize for
prefill; it would only matter at much longer contexts (32K+) where FA grows
O(n^2). The pp8192 kernel-time split that this section asked for has been run;
see the UPDATE at the top and the bench-results profile.

## 5B. Making Q8_0 ultra fast (separate problem from Q4_K)

Q8_0 = 32 int8 + 1 fp16 scale per block = ~1.06 bytes/weight, ~2x the bytes of
Q4_K. Two consequences that split the problem cleanly:

- Q8_0 needs NO nibble unpack (weights already int8), so its MMQ inner loop is
  nearly pure DP4A + scale — it is NOT VALU/unpack bound like Q4_K.
- Q8_0 moves 2x the bytes, so it is **LDS/bandwidth bound in prefill and hard
  bandwidth bound in decode.** Baseline (MI50-0): Q8_0 MMQ prefill n=512 = 8.52
  TFLOPS (WORSE than Q4_K's 15.9 — pure traffic cost), decode n=1 = 1.16 TFLOPS.

### Prefill — where the real headroom is
1. **Tune the rocBLAS dispatch threshold (highest EV, cheapest).** Current gate
   (`ggml_cuda_should_use_mmq`, mmq.cu): Q8_0 uses MMQ for `ne11 <= 256`, else
   the non-MMQ path (dequant Q8_0→F16 then rocBLAS hgemm — VERIFY this is the
   path, not int8 GEMM). That non-MMQ path already gave +27–57% at wide PP
   because Vega F16 hgemm (~2x FP32, well-tuned Tensile) beats the traffic-bound
   Q8_0 MMQ. Since Q8_0 MMQ is so weak (8.52 TFLOPS), rocBLAS likely wins far
   below 256 — sweep the threshold down (192/128/96/64) with pp256/pp384/pp512
   A/B and pick the crossover. Pure tuning, no kernel work.
2. **Fused dequant-into-GEMM to kill the scratch roundtrip.** The non-MMQ path
   dequantizes Q8_0→F16 into a scratch buffer, then hgemm reads it back — extra
   HBM traffic on a bandwidth-bound op. A prologue that feeds hgemm from Q8_0
   directly (or a custom F16-accumulate GEMM reading Q8_0) removes that
   roundtrip. Medium effort, medium win, prefill-only.
3. **For the ne11<=256 range that stays on MMQ:** double-buffered LDS staging
   (§5.2) and heavier register-blocking of the fat int8 weight tile (more DP4A
   reuse per LDS load). Q8_0 needs no unpack, so the ONLY lever here is
   traffic/latency hiding. Medium.

NOTE: native int8 rocBLAS/hipBLASLt GEMM does NOT cleanly apply — Q8_0 has a
scale per 32 elements along K, so a single int8 GEMM over K=14336 can't express
the 448 block scales without splitting into 32-wide panels (kills GEMM). That
block-scaled structure is exactly why MMQ exists; don't chase plain int8 GEMM.

### Decode — essentially capped, be honest
Q8_0 TG is hard bandwidth bound: ~27B × 1.06 B = ~28.7 GB/token-pass; at ~700
GB/s effective that's ~24 tok/s theoretical, and RESULTS shows ~19 t/s dual d0
— already ~80% of a 700 GB/s roofline. This is why the VDR-4 experiment was
neutral (+0.19%): TG is not compute bound. The only Q8_0 TG levers are:
- **Raise EFFECTIVE bandwidth**: profile memory-efficiency/coalescing counters
  on the Q8_0 MMVQ loader; if HBM transactions aren't fully used, fixing access
  patterns raises effective BW. This is the only kernel-level TG win.
- Memory clock/power state — real but OUT OF SCOPE (settings, not kernel).
- Fewer bytes = use Q6_K/Q4_K — a model choice, out of scope.
Verdict: "ultra fast Q8_0" means winning PREFILL (levers 1–3). Decode is near
its bandwidth ceiling; do not expect a large TG gain from kernel work.

## 6. Honest verdict

400 t/s single-GPU is a stretch. 5.1 + 5.2 realistically target ~280–330 t/s
(operator toward ~60–70% of roofline). Approaching 400 additionally requires
5.5 (so attention doesn't dominate at depth 8192) plus near-roofline matmul.
Present 400 as the aggressive end of the range; ~320–360 is the likely landing.
The repack (5.1) is a different *class* of change (offline, structural) than the
in-loop layout micro-tweaks that were rejected — those rejections do not apply
to it.

## 7. Next-session TODO

1. DONE: confirmed dense (hybrid attn+DeltaNet) via gguf; DeltaNet is 8% of
   prefill so the roofline argument is unchanged.
2. DONE: pp8192 kernel-time split (see UPDATE at top). Matmul 80%, FA 4%.
3. IN PROGRESS: the Q4_K `min_blocks=3` occupancy candidate is integrated in
   the current bigbang patch but still needs a controlled A/B measurement.
   Rebuild it after the upstream merge; see
   `vr/bench-results/gfx906-pp8192-kernel-profile-2026-07-02/FINDINGS.md` and
   the worktree pointer there.
4. If (3) confirms occupancy-bound, prototype 5.1 repack (mechanism = VGPR
   reduction) across Q4_K/Q5_K/Q6_K; MUL_MAT 1103/1103 + repack round-trip test;
   A/B pp512/pp2048/pp8192.
5. Validate the integrated Q4_K and Q8_0 paired-fusion TG candidates with a
   graph that exercises value-plus-gate fusion and controlled model A/B runs.

### Per-type launch bounds mechanism (confirmed 2026-07-02)

`mul_mat_q` in `mmq.cuh` is one template, but each quant type is instantiated in
its OWN translation unit (`template-instances/mmq-instance-q4_k.cu`, `-q6_k.cu`,
...). The gfx906 gate in `ggml-hip/CMakeLists.txt` sets COMPILE_DEFINITIONS per
instance file, so the `#if defined(...MIN_BLOCKS_1)` in the launch_bounds block
is scoped to just that type. Confirmed by counters: Q4_K compiles under the GCN
default min_blocks=2 (128 VGPR, 2 waves/SIMD); Q6_K under min_blocks=1 (140 VGPR,
1 wave/SIMD). So per-type occupancy tuning is a one-line define on the specific
instance file plus a matching `#elif` in `mmq.cuh` - no constexpr, no coupling.
The experiment in (3) adds `GGML_CUDA_MMQ_Q4K_GFX906_MIN_BLOCKS_3` to
`mmq-instance-q4_k.cu` to force Q4_K to <=85 VGPR / 3 waves/SIMD.
