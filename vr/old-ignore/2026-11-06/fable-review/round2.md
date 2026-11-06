# Fable review, round 2 (2026-11-06, verbatim)

Input: outcomes of round 1 (F1, F2, Q1, N3, P4, T1), load-excluded profiles in `build-fnspeed-20261106/validation/fable-prof/SUMMARY.txt`, and the 5-prompt speculative numbers. See the [F2/P4 record](../../../PERFORMANCE.md#2026-11-06-f2-p4-toolchain-check-and-acceptance-aware-speculative-runs).

## Round 2

### 1. F2 / P4 as implemented
- F2: T=1 at 27 us is at the floor (3 launches x ~4.2 us + 7 MB of Q8_0 hc weights at ~700 GB/s = 10 us). T=4 at 47 us means the down/up kernels serialize or re-stream per token. Make them weight-stationary: load one Q8_0 block once, dp4a it against all NT activations held in registers/LDS, so T=4 lands near 30 us. Worth ~17 us x 96 calls = 1.6 ms per MTP3 step (~3%). Second small item: `dsv4_hc_post` writes the stream that `dsv4_hc_mix_prep` reads next; one post+prep kernel removes 96 launches/token (~0.4 ms, 2%). Int8 is the right choice (same math as MMVQ on Q8_0 weights; KLD at floor).
- P4: the router at ~9 TFLOPS in rocBLAS is fine; nothing left there.

### 2. Next ideas, ranked by gain per effort

**FN-B1. Fused MoE routing block (TG and MTP, S-M effort, ~7%).** Profile fn-tg128: `topk_moe_cuda<512>` runs as one 32-thread block per token at 27.8 us, 48/GPU/token = 1.3 ms of the 21.9 ms token; MTP3 shows the same 28.8 us plus `flashnext_expert_groups` 20.7 us, two `k_get_rows`, `scale_f32` and `moe_weighted_reduction` glue: ~7 launches per layer. One wave64 kernel per token (sigmoid/softmax, top-10 by butterfly argmax, weight normalization, ids + weights + group table out) replaces them. Keep upstream's float op order so the output is bit-exact. Expect ~1.5 ms/GPU/token and ~2 ms per MTP3 step. Test: kernel count per token drops by ~250; byte-identical output; `spec_accept.sh`.

**FN-A. Weight-stationary grouped expert GEMM for PP (L effort, +20% PP at ub 512, more at ub 256).** fn-pp2048 (ub 512): stock `mul_mat_q` IQ2_S/IQ3_XXS gate/up = 194 calls x 2.04 ms = 26.4%; each call streams ~160 MB of expert weights at ~78 GB/s (4 TOPS), because 512 tokens x 10 / 512 experts = 10 tokens per expert and the MMQ tile is mostly padding. Repacked IQ4_NL down (`w32`, 0.88 ms) is 9.5 TOPS. Design: extend `flashnext_moe_grouped` (ggml/src/ggml-cuda/flashnext-moe.cuh, today T<=8 union) to take per-expert token lists from `mm_ids_helper`, hold 8-16 q8_1 tokens per chunk in LDS, decode each weight once (S1 grid in LDS, S2 signs) and dp4a across the chunk. At >=300 GB/s weight stream, gate+up go 4.1 -> ~1.1 ms per layer and down 0.9 -> 0.5: ~160 ms of the ~860 ms ubatch. Same integer dots; float accumulation order differs (KLD check). Go/no-go first: microbench one layer's real ids at ub 256/512/2048 against `mul_mat_q`. This also lifts the 100K CLI PP, which runs at ub 256 (5 tokens per expert).

**AR LL protocol (both models, M effort, 3-4%).** `ggml_cuda_ar_kernel` is 26.9 us on FN for 5 KB (92/GPU/token = 2.5 ms, part skew) and 16.9 us on 27B (129/token = 2.2 ms). Today: payload write, `__threadfence_system`, flag write, peer polls flag, peer reads payload: 2-3 PCIe round trips. NCCL-LL style 8 B data + 8 B sequence-word packets let the peer poll the data itself, one trip. Expect ~10-12 us when not skew-bound: FN -1 ms/token, 27B -0.7 ms. Bit-exact.

### 3. 27B (~10%)
TG is 70% in `mul_mat_vec_q8_0_repacked` at ~780 GB/s/GPU (78% of HBM peak), so only two routes reach 10%:
- **MTP2 step (S effort, ~7-8% with AR LL):** `k_set_rows` 193/tok at 8.8 us = 1.7 ms/step (4.5%) are the K/V cache writes for 3 tokens; fuse them into the rope (K) and the V copy so each layer loses 2 launches. Plus AR LL above.
- **Mat-vec ceiling (M, uncertain, <=7%):** before touching `mul_mat_vec_q8_0_repacked<8,8,...>` (436 calls x 45 us), time a pure streaming-read kernel over the same bytes; if it also tops at ~800 GB/s, drop this. If it reaches ~900, try 16 B loads with 2-deep prefetch and more rows per block.
- **Depth (D1, M-L, only at 32K+):** TG -22% and MTP2 -32% at 32K means attention runs far below bandwidth; profile `flash_attn_tile` at 32K, and if per-layer time implies <300 GB/s of K/V, add a gfx906 decode path for ncols<=8 with KV split over >=240 blocks and a fixed-order combine. Also helps Flash Next at 100K (12 attention layers, ~2 ms/token).

Order: FN-B1, F2 NT tuning, 27B set_rows fuse, AR LL, FN-A, 27B mat-vec check, D1.
