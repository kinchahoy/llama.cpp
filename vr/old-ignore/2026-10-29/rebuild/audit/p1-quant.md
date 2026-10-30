# Pass 1 report: quant (Q28-Q33, Q37)

Tree: /home/raistlin/infer/mx-rebuild. Not built, not run. Symbol checks by grep (excluding vr/) show no remaining users of anything removed.

| Item | Status | Change |
|---|---|---|
| Q28 | done | `rm ggml/src/ggml-cuda/mmq-config-gfx906.cuh` (28 lines). mmq.cuh: include removed, host `cc == VEGA20` branch removed (dead behind GCN check), device `#elif defined(__gfx906__)` removed (dead behind `#ifdef GCN`). mmq.cuh -6 lines. |
| Q29 | done | repack-common.cuh: `repack_qs_bytes` (-5 lines), `rp_scale_from_slot` with its comment (-12 lines). Zero callers. |
| Q30 | done | mul-mat.cu: dense fused IQ4_NL/Q5_K/Q4_K/Q6_K branches removed (-20 lines), MXFP4 and Q8_0 kept. repack-common.cu: stale comment above `ggml_cuda_repack_mmv_fusion_supported` replaced (it said MXFP4 must not be fused; code admits it). |
| Q31 | done | GGML_RP_NC_FUSION read and its comment removed from repack-common.cu (`ggml_cuda_repack_mmv_fusion_width_ok` now allows 1 token dense, `has_ids ? MOE : 1`). `ggml_cuda_mul_mat_vec_repacked_nc_glu` (mul-mat.cu, ~66 lines) and the `dst->ne[1] > 1` dispatch in `ggml_cuda_mul_mat_vec_repacked_fused` removed. `mul_mat_vec_repacked_nc_glu` kernel removed from repack-kernels.cuh (-10 lines). |
| Q32 | done | mul-mat.cu: GGML_RP_NC_RPL (`rpl4` + `else if (rpl4)` branch) and GGML_RP_NC_23 (`nc23`, switch now `switch (ne11)`, case labels 102/103 removed) deleted. Default path unchanged. |
| Q33 | done | GGML_RP_GENERIC_MMV removed from mul-mat.cu (Q8_0 case in `ggml_cuda_mul_mat_repacked_slice`, including its now-unused `grid`) and mul-mat-id.cu (Q8_0 n_tokens==1 case). |
| Q37 | done | q8_repack/README.md: intro lists the 7 repacked types; "Known gaps" last bullet rewritten (dense fused up/gate only Q8_0 and MXFP4). repack.cuh line 1: stray space before period removed. |
| Q34, Q35, Q36 | skipped | Per instruction. |

Approximate totals (vs. index): mul-mat.cu -117, repack-common.cuh -17, repack-kernels.cuh -10, repack-common.cu -11, mul-mat-id.cu -8, mmq.cuh -6, mmq-config-gfx906.cuh -28 (deleted), README -2 net.

Left in place on purpose:
- `ggml_cuda_repack_mmv_fusion_width_ok` keeps its 3-argument signature. Callers in ggml-cuda.cu (lines ~5694, ~5900) and mul-mat-id.cu:481 are outside my files. `wt` is now `GGML_UNUSED`. Dropping it needs a ggml-cuda.cu edit (other owner).
- `mul_mat_vec_repacked_nc_impl` still has its HAS_GLU template parameter. `true` is still used by `mul_mat_vec_rp` and `rp_q8_0_mmv_launch<true>`, so it is not dead. Only the nc_glu wrapper is gone.
- ggml-cuda.cu fusion gates are unchanged. They now see the dense width limit of 1 token by default, same as before.
- Gfx906 row-first MMQ traversal (mmq.cuh device and host) and the IQ2_S two-row MMVQ path are untouched.
- mmvq.cu: no edits (Q35 is skipped and no other item applies).
