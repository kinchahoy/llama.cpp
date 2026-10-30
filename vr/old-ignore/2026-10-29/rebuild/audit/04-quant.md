# Audit 04: quantized matmul (q8_repack, MMQ, MMVQ, quantize, dequant)

Scope: `ggml/src/ggml-cuda/q8_repack/` (all files), `mmq.cu`, `mmq.cuh`, `mmq-config-gfx906.cuh`, `mmq-vec-dot.cuh`, `mmq-load-tiles.cuh`, `mmvq.cu`, `vecdotq.cuh`, `dequantize.cuh`, `quantize.cu`, `quantize.cuh`. Related but outside the list: `ggml-cuda/flashnext-moe.cuh` (included by mmvq.cu), `mmq-config-gcn.cuh` (upstream), `ggml-cuda.cu` (fusion gates, q8_1 cache).

Method: static reading only (no build, no run). Dead-code checks by `grep -rw` over `ggml/src/ggml-cuda` at HEAD working tree (HEAD == our commit). Base comparisons via `git show 418f905a4:<path>`.

Diff size (+/-, vs 418f905a4): q8_repack new 4862 lines (README 209, buffer.cu 703, mul-mat.cu 698, mul-mat-id.cu 641, repack-common.cu 437, repack-common.cuh 656, repack-kernels.cuh 1344, repack.cuh 115, flashnext 59); mmq.cu +288/-82; mmq.cuh +55/-5; mmvq.cu +54/-11; mmq-load-tiles.cuh +11/-2; mmq-vec-dot.cuh +14; mmq-config-gfx906.cuh +28; vecdotq.cuh +19; dequantize.cuh +22/-2; quantize.cu +17/-5; quantize.cuh +6/-4; flashnext-moe.cuh (parent) +190.

---

## 0. Headline findings

1. **The gfx906 MMQ config table is dead code.** `ggml_cuda_mmq_get_config_gfx906` (mmq-config-gfx906.cuh) is never selected. Host: `mmq.cuh:235-236` returns the GCN table first because `GGML_CUDA_CC_IS_GCN(0x1000906)` is true (`common.cuh:98`). The gfx906 branch at `mmq.cuh:253-254` comes after it and is unreachable. Device: `mmq.cuh:276-277` is `#ifdef GCN`, which `vendors/hip.h:186-196` defines for `__gfx906__`. The `#elif defined(__gfx906__)` at `mmq.cuh:288-289` is unreachable. The effective Q8_0 table on gfx906 is upstream `mmq-config-gcn.cuh:74-92`.
2. **Upstream already has most of the fork's intended Q8_0 gfx906 table.** The only differing rows are `mmq-config-gcn.cuh:75, 76, 83` (nthreads 256 where the fork intends 512). Discussion 23881 (fetched) reports that changing only the GCN Q8_0 nthreads 256 to 512 gave +19% Q8_0 pp512 and -6.9% Q4_K_M pp512 on MI50 (rumas77). This is the smallest per-quant tuning-table candidate for upstream. The fork's vr notes attribute gains to `mmq-config-gfx906.cuh`, which is not active, so those attributions need re-measurement.
3. **`mmvq.cu` `calc_nwarps` change is not gfx906-gated.** The Q8_0 5-column case returns 2 warps for the whole GCN table. That table covers GCN5 (gfx900/906/909/90c) and CDNA (gfx908+) through `hip.h:213-214` and `mmvq.cu:113-114, 134-135`. The comment says gfx906 only.
4. **Three copies of the MMQ J-tile selection.** `mmq.cu:135-162` (base helper, still used at line 729), inline copy at `mmq.cu:250-277` (sets only padding and nthreads; `args.J_best` is written at `mmq.cu:442, 583` and never read), and inline copy at `mmq.cuh:1560-1593` (this one selects the kernel J, recomputed per launch on all GPUs). In a chunked dense launch the padding (from outer J) and the kernel tile (from inner J, with chunk ncols) can disagree. Medium confidence.
5. **Generic (non-gfx906) functional changes in the MMQ/MMVQ path:** q8_1 activation cache, default ON, all backends (`ggml-cuda.cu:2740-2800`, used by `mmvq.cu:1580`, `mmq.cu:~300`, repack); MMQ workspace chunking under memory pressure (`mmq.cu:~330-600`); MoE id chunking.
6. **Duplicate MoE grouped-GEMM ("flashnext") implementations** behind one env var: parent `ggml-cuda/flashnext-moe.cuh` (canonical IQ2_S/IQ4_NL) and `q8_repack/flashnext-moe.cuh` (repacked IQ4_NL only). Both opt-in, `GGML_CUDA_FLASHNEXT_MOE=1`.
7. **q8_repack has dead code and undocumented or negative-result env knobs** (section 3).

---

## 1. Quant types covered by q8_repack

Supported (repack-common.cu:12-35; buffer.cu:417-461 set_tensor/kernels; dispatch in mul-mat.cu and mul-mat-id.cu):

| Type | Dense GEMM | Dense MMV | MoE GEMM | MoE MMV | Fused dense | Fused MoE |
|---|---|---|---|---|---|---|
| Q8_0 | yes | yes | yes | yes | yes | yes |
| MXFP4 | yes | yes | yes | yes | yes | yes |
| IQ4_NL | yes | yes | yes | yes | **unreachable** (see 3.1) | yes (env `GGML_CUDA_REPACK_KQUANT_MOE_FUSION`, default on) |
| Q5_1 | yes | yes | yes | yes | no | no |
| Q5_K | yes | yes | yes | yes | **unreachable** | yes (K-quant env) |
| Q4_K | yes | yes | yes | yes | **unreachable** | yes (K-quant env) |
| Q6_K | yes | yes | yes | yes | **unreachable** | yes (K-quant env) |

README is stale: its intro (line 3) and "Known gaps" (last bullet) say only Q8_0 and MXFP4. Repack is gated to `cc == GGML_CUDA_CC_VEGA20` at `buffer.cu:685-687`, matching the README.

## 2. Gating of the MMQ/MMVQ changes

| Change | Location | Gate | Affects other GPUs? |
|---|---|---|---|
| MMQ row-first traversal (device swap) | mmq.cuh:1057-1064 (device, `GGML_USE_HIP && __gfx906__`); host mmq.cuh:1487-1489 (`cc == VEGA20`) | gfx906 only, host and device consistent | no. Keep (vr IMPROVEMENTS.md:20). |
| gfx906 MMQ config | mmq-config-gfx906.cuh, mmq.cuh:253-254, 288-289 | none effective (dead) | no (dead) |
| J cap >64 unless tiles >= nsm | mmq.cuh:1578-1585, mmq.cu:264-268 | `cc == VEGA20` | no |
| Loader threads_per_row 8 | mmq-load-tiles.cuh:520-525 | `GGML_USE_HIP && GCN5` (gfx900/906/909/90c) | no on NVIDIA/RDNA; affects other GCN5 parts |
| Loader loop rewrite (same result for 32) | mmq-load-tiles.cuh:546-552 | none (generic `#else` branch: non-MFMA/WMMA) | equivalent algebraically (k steps 0,32 map to bxi[0], bxi[4]); no behavior change |
| vec_dot `#pragma unroll 2` | mmq-vec-dot.cuh:123-134 | `__gfx906__` | no |
| iq3_xxs sign-mask dp4a | vecdotq.cuh:1235-1240, 1256-1283 | `GGML_USE_HIP && __gfx906__` | no. Math checked: `(bits<<8)-bits` gives 0xFF per set byte; the spread multiply has no carries for 4 bits |
| quantize_q8_1 block 32 | quantize.cu:563-580, quantize.cuh:9-13 | runtime `cc == VEGA20` inside `GGML_USE_HIP` | no |
| IQ2_S two-row MMVQ | mmvq.cu:1222-1235 (dispatch), 643-646 (`two_rows` template), 1069-1081 (switch) | `cc == VEGA20 && has_ids && nrows % 2 == 0`, ncols_dst == 1 | no. Launch dims consistent (both nwarps 2) |
| **calc_nwarps Q8_0 case 5 = 2** | mmvq.cu:478-483 | **none** (table GCN) | **yes: GCN5 and CDNA (MI100+)** |
| flashnext include and call | mmvq.cu:1474, 1617-1622 | `GGML_USE_HIP`, then cc==VEGA20 and env (inside callee) | no (env opt-in) |
| MMQ tp_overlap slab block | mmq.cu:296-354 | `GGML_USE_HIP && cc==VEGA20 && Q8_0`, hard-coded shape (ne00 8704, ne01 5120, ne11 2048, etc.), `ctx.tp_overlap.enabled` (env `GGML_CUDA_TP_OVERLAP=1`, default off) | no |
| q8_1 activation cache | ggml-cuda.cu:2740-2800; mmvq.cu:1580; mmq.cu:~300; repack | **none**, env `GGML_CUDA_Q8_1_CACHE`, default ON | **yes, all backends** |
| MMQ workspace and MoE id chunking | mmq.cu:~330-600 | none (only under allocation pressure) | yes (generic) |
| dequantize `_n` variants (n_sub tail) | dequantize.cuh:404-471; used by getrows.cu:481,495 (qwen4exp) | none | yes, generic; no change for full super-blocks |

## 3. Feature table

Class key (per brief): A essential gfx906 compat, B perf, C functional extension, D superseded, E redundant/dead/debug, F unclear.

| Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|
| 3.1 q8_repack dense fused branches for IQ4_NL/Q5_K/Q4_K/Q6_K | mul-mat.cu:673-692 | ~20 / 0 | E (unreachable) | none | Q8_0/MXFP4 admission | Remove branches; fix stale comment repack-common.cu:68-71 (claims MXFP4 must not be fused; code does) | high |
| 3.2 Repacked weight layout + buffer + upload | buffer.cu, repack-common.cu/.cuh, repack.cuh | ~1800 | B/C (core of the fork) | none (upstream CPU repack only) | gfx906 (cc==VEGA20) | keep | high |
| 3.3 Host repack reference vs device repack kernels | repack-common.cu:95-310 (host), buffer.cu:127-375 (device) | ~450 duplicated logic | F/E-risk | none | both live: host sync path buffer.cu:106, device async path buffer.cu:594 | keep, but add a cross-check test; divergence risk across 7 layouts | med |
| 3.4 Dense narrow-batch GEMV (`mul_mat_vec_repacked_nc`) with env geometry knobs | mul-mat.cu:241-378, repack-kernels.cuh:972-1260 | ~140 | B | none | gfx906 | keep default path; drop env variants (3.6) | med-high |
| 3.5 MoE fused and unfused routing, tile prefix-sum | mul-mat-id.cu, repack-kernels.cuh:1261-1340 | ~640 | B/C | none | gfx906 | keep | med |
| 3.6 Env experiments with negative or no recorded win | see below | ~150 | E | none | env | remove (list below) | med-high |
| 3.7 Dense fused nc_glu (`ggml_cuda_mul_mat_vec_repacked_nc_glu`) | mul-mat.cu:533-~600, repack-kernels.cuh:1244-1260, repack-common.cu:47-52 (`GGML_RP_NC_FUSION`) | ~85 | E (env-only) | none | env default off; unreachable by default (ggml-cuda.cu:5723, 5934 admit dense fusion only at ne[1]==1) | remove; archived QUICKSTART: "no gain"; code comment: "measured slower" | high |
| 3.8 `repack_qs_bytes` | repack-common.cuh:84-88 | 5 / 0 | E (0 callers) | none | none | delete | high |
| 3.9 `rp_scale_from_slot` | repack-common.cuh:129-137 | 9 / 0 | E (0 callers) | none | none | delete | high |
| 3.10 `GGML_RP_GENERIC_MMV` A/B hook (undocumented) | mul-mat.cu:389-400, mul-mat-id.cu:145-152 | ~20 / 0 | E (debug) | none | env | remove hook, keep generic kernel call only if still used elsewhere | med-high |
| 3.11 flashnext IQ4_NL repacked grouped MoE | q8_repack/flashnext-moe.cuh (59), mul-mat-id.cu:77-82 | 60 / 0 | C/E (opt-in) | duplicates 3.12 | env `GGML_CUDA_FLASHNEXT_MOE=1` | pick one implementation (3.12 covers IQ2_S too); remove this one or document | med |
| 3.12 flashnext canonical grouped MoE (IQ2_S, IQ4_NL) | flashnext-moe.cuh (parent, 190), mmvq.cu:1474, 1617-1622 | 190 / 0 | C (opt-in, vr accepted but opt-in) | none | env default off; cc==VEGA20 | keep (it is the one that covers IQ2_S) | med |
| 3.13 IQ2_S two-row single-token MMVQ | mmvq.cu:643-646, 1069-1081, 1222-1235 | ~30 / 0 | B (+2.37% one matched pair, vr IMPROVEMENTS.md:30) | none | gfx906 + MoE + even rows | keep, low confidence on gain; needs repeat | med |
| 3.14 gfx906 MMQ config (mmq-config-gfx906.cuh) | file + mmq.cuh:229, 253-254, 288-289 | 28 + 4 / 0 | E (DEAD, see headline 1) | upstream gcn table already has Q8_0 at 512 for most J | none effective | delete (no behavior change) or fix ordering and re-measure | high |
| 3.15 J-tile selection, 3 copies | mmq.cu:135-162, 250-277; mmq.cuh:1560-1593; arg mmq.cuh:1453 | ~60 duplicated | E/F | base helper mmq.cu:135 | gfx906 cap inside the copies | consolidate to one selector; pass J_best; delete inline copy in switch | med |
| 3.16 MMQ row-first traversal | mmq.cuh:1057-1064, 1487-1489 | ~10 | A (per vr notes) | none | gfx906 | keep | high |
| 3.17 gfx906 loader threads_per_row 8 | mmq-load-tiles.cuh:520-531, 546-552 | 11 | B (vr: ~+50% pp512 vs 32) | none | GCN5 | keep; the loop rewrite could be reverted to the base form under `#if` to shrink generic diff | med |
| 3.18 vec_dot `#pragma unroll 2` (Q8_0 only) | mmq-vec-dot.cuh:123-134 | 14 | B (vr: 22% fewer v_add) | none | `__gfx906__` | keep | med |
| 3.19 Q8_0 calc_nwarps 5-col = 2 | mmvq.cu:478-483 | 6 | B, but ungated | none | GCN table (includes CDNA) | gate to gfx906 (needs a separate table id, since host and device must agree) or accept and document | med-high |
| 3.20 quantize_q8_1 block 32 on gfx906 | quantize.cu:563-580, quantize.cuh:9-13 | ~20 | B (vr PERFORMANCE.md:118 `quantize_q8_1<32>`) | none | runtime VEGA20 | keep | med |
| 3.21 iq3_xxs sign-mask dp4a | vecdotq.cuh:1235-1283 | 19 | B | none | gfx906 | keep (math verified) | med |
| 3.22 dequantize `_n` tail variants | dequantize.cuh:404-471 | 22 | C (qwen4exp PLE table, generic) | getrows base has `dequantize_kq_n` pattern | none | keep, separate upstream PR candidate | med |
| 3.23 q8_1 activation cache | ggml-cuda.cu:2740-2800 (+ call sites) | ~80 | C (generic, default on) | none | none; env `GGML_CUDA_Q8_1_CACHE` | keep, but add a content-hazard review (headline 5) and an A/B test with env=0 | med |
| 3.24 MMQ workspace and id chunking | mmq.cu:~330-600 | ~120 | C (generic, pressure path) | none | none | keep; flag as generic (needs owner review) | med |
| 3.25 MMQ tp_overlap slab block | mmq.cu:296-354 | ~60 | E (default off, hard-coded shape; archived QUICKSTART: "does nothing useful on HIP") | none | HIP+VEGA20+Q8_0+shape+env | remove from mmq.cu after coordinating with allreduce owner | med |
| 3.26 `fprintf` in default case | mmq.cuh:1643 | 0 | E (trivial) | base printed args.J_best | none | keep (now prints local J) | high |
| 3.27 Stale README and comments | q8_repack/README.md lines 3, last bullet; repack.cuh:1; repack-common.cu:68-71 | doc | E | - | - | update docs | high |

### 3.6 Env knobs in the repack path (all read with static caching)

| Env | Default | Status | Recommendation |
|---|---|---|---|
| `GGML_RP_MMV_ROWS` | 8 | documented (archived QUICKSTART:88), mul-mat.cu:15-26 | keep |
| `GGML_RP_NC_PLANAR` | 1 | documented, mul-mat.cu:47-53 | keep |
| `GGML_RP_NC_L16` | 1 | documented ("verify about 10 ms slower at 8"), mul-mat.cu:291-295 | keep |
| `GGML_RP_NC_LDS` | 1 | documented (about 3% slower at 6 columns), mul-mat.cu:305-310 | keep or fix default, low priority |
| `GGML_RP_NC_RPL` | off | archived "no gain", mul-mat.cu:297-298, 346-348 | remove |
| `GGML_RP_NC_23` | off | archived "no gain", mul-mat.cu:300-305 and `switch (nc23 && ne11 <= 3 ...)` | remove |
| `GGML_RP_NC_FUSION` | off | archived "no gain"; code says slower; repack-common.cu:47-52 | remove (with 3.7) |
| `GGML_RP_GENERIC_MMV` | unset | undocumented A/B hook, mul-mat.cu:391, mul-mat-id.cu:146 | remove |
| `GGML_CUDA_REPACK_KQUANT_MOE_FUSION` | 1 | documented in code, repack-common.cu:81-90 | keep |
| `GGML_CUDA_FLASHNEXT_MOE` | 0 | opt-in; read in 2 places (flashnext-moe.cuh:165, q8_repack/flashnext-moe.cuh:55) | keep one reader (see 3.11/3.12) |
| `GGML_CUDA_TP_OVERLAP` | 0 | opt-in, mmq.cu:296-354 (and allreduce) | remove mmq side (3.25) |
| `GGML_CUDA_Q8_1_CACHE` | 1 | generic, ggml-cuda.cu:2740 | keep, add hazard review |

---

## 4. Duplication and structure

- **repack-common.cu/.cuh vs repack-kernels.cuh:** no copied kernel logic. `rp_warp_reduce_sum` has one DPP and one fallback definition behind a guard (repack-common.cuh:230-251). Host layout math lives in repack-common.cuh (`repack_qs_stride`, `repack_gcn_nbytes`), used by both host and device.
- **Host vs device repack of the same 7 layouts** (repack-common.cu:95-310 vs buffer.cu:127-375). These are two implementations by design (sync path vs async staged path). Divergence risk is the concern, not dead code.
- **J selection copies** (3.15).
- **Two flashnext grouped kernels** (3.11, 3.12).

## 5. Key correctness/structural risks

- **J-tile padding mismatch (medium).** `mmq.cu:283-285` sizes `src1_q8_1_padding` from outer `J_best`; the kernel tile is picked again inside `mul_mat_q_switch_J` with `args.ncols_opt = iter_ne11` (chunk size). If the chunk's best J exceeds outer J, the kernel can read past the padded workspace. Ties go to the smaller J, so this is probably rare but not excluded.
- **q8_1 cache content hazard (medium).** Keys are src1 pointer, data, strides, and shape (`ggml-cuda.cu:2771-2779`), with no content version. Validity relies on the comment argument that ggml-alloc keeps buffers live. That does not cover an in-place write to src1 between two consumers in one graph. Needs an A/B run with `GGML_CUDA_Q8_1_CACHE=0`.
- **calc_nwarps applies to CDNA (medium-high).** Changes results and performance on MI100/MI200/MI300 for 5-column Q8_0 MMVQ, not only gfx906.

## 6. Upstream discussion 23881 (fetched) and smallest per-quant change

From the discussion (no maintainer replies): JackBinary tested Q8_0 only; nwarps 4 to 8 gave about +30% pp512 on MI60. rumas77 found that gfx906 falls to the RDNA2 table in their build. A GCN config changing only nthreads 256 to 512 (nwarps 8) gave Q8_0 pp512 611.8 to 728.5 (+19.1%), and Q4_K_M 637.9 to 594.2 (-6.9%). Q5/Q6 and MXFP4 were not tested. No FAILs in test-backend-ops at nwarps 4 or 8, apart from one intermittent q5_1 case at nwarps 8 reported by JackBinary.

**Smallest upstreamable tuning-table change:** in `ggml/src/ggml-cuda/mmq-config-gcn.cuh`, change the Q8_0 nthreads from 256 to 512 on three rows:
- line 75: `CASE(GGML_TYPE_Q8_0, 256, 2, 128, 16, ..., false, true)` -> 512
- line 76: `CASE(GGML_TYPE_Q8_0, 256, 2, 128, 32, ..., false, true)` -> 512
- line 83: `CASE(GGML_TYPE_Q8_0, 256, 2, 128, 16, ..., false, false)` -> 512

This is exactly the fork's intended gfx906 Q8_0 nthreads on the rows upstream already lists. Q8_0-only keeps Q4_K untouched. Caveat: our fork's measured gains were not from this table (it was dead), so measure on gfx906 before proposing. The larger lever in this slice is the loader change (3.17), which is not a table change.

## 7. Quick wins (file:line, why safe)

1. **Dead gfx906 config**: delete `ggml/src/ggml-cuda/mmq-config-gfx906.cuh` and the include at `mmq.cuh:229`, plus the unreachable branches at `mmq.cuh:253-254` and `mmq.cuh:288-289`. No behavior change: the branch is shadowed on host (`mmq.cuh:235`) and device (`mmq.cuh:276`). Confidence high. Verify first that the fork's measured numbers are not expected to change.
2. **Delete dead helpers**: `q8_repack/repack-common.cuh:84-88` (`repack_qs_bytes`) and `:129-137` (`rp_scale_from_slot`). Zero callers (grep -rw). High.
3. **Remove unreachable dense fused branches**: `q8_repack/mul-mat.cu:673-692` (IQ4_NL, Q5_K, Q4_K, Q6_K). Admission is Q8_0/MXFP4 only (`repack-common.cu:72-76`, used at `ggml-cuda.cu:5717, 5933`). Also fix the stale comment at `repack-common.cu:68-71`. High.
4. **Remove `GGML_RP_NC_FUSION` path**: `repack-common.cu:47-52` (nc_fusion term), `mul-mat.cu:533-~600` (`ggml_cuda_mul_mat_vec_repacked_nc_glu`), `repack-kernels.cuh:1244-1260`. Unreachable by default (`ggml-cuda.cu:5723, 5934`). Medium-high.
5. **Remove negative-result env variants**: `GGML_RP_NC_RPL` (`mul-mat.cu:297-298`, rpl4 branch in `RP_NC_CASE_L`), `GGML_RP_NC_23` (`mul-mat.cu:300-305` and the `nc23` switch). Archived QUICKSTART:111 says no gain. Medium-high.
6. **Remove undocumented hook**: `GGML_RP_GENERIC_MMV` at `mul-mat.cu:389-400` and `mul-mat-id.cu:145-152`. Medium-high.
7. **Consolidate J selection**: keep `mmq.cu:250-277` as the single selector, pass `J_best` through `mmq_args` (already a field, `mmq.cuh:1453`), delete the inline loop at `mmq.cuh:1560-1593`. Behavior change only in the chunked dense case (which is the padding fix). Medium; needs a targeted check.
8. **Gate or document `calc_nwarps` case 5**: `mmvq.cu:478-483`. Either gate it (needs a table id distinct from GCN) or document it as applying to CDNA. Medium-high.
9. **Remove mmq tp_overlap block**: `mmq.cu:296-354` (default off, shape hard-coded, archived "does nothing useful on HIP"). Needs the allreduce owner's OK, since `ggml-cuda.cu:1712` sets the flag. Medium.
10. **Pick one flashnext repacked implementation**: `q8_repack/flashnext-moe.cuh` (59 LOC) plus `mul-mat-id.cu:77-82`, or the parent file. Medium.
11. **Doc fixes**: `q8_repack/README.md` line 3 and "Known gaps" (7 types, not 2); `repack.cuh:1` stray space. High.
