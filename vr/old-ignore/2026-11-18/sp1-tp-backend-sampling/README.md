# sp1-tp-backend-sampling: backend sampling on split logits (DROPPED)

Idea SP1 (PERFORMANCE.md): under `-sm tensor`, `llama_context::set_sampler` refuses backend sampling, so 27B MTP2 samples about 248K logits per row on the CPU. Try: lift the refusal (`LLAMA_TP_BACKEND_SAMPLING=1`, [patch](tp-backend-sampling-switch.patch)) and let the meta backend gather the vocab-split rows for TOP_K / ARGMAX / SOFT_MAX.

Result (2026-11-18, merged tree 992b4580f + switch, 27B Q8_0 MTP2, `spec_accept.sh`): the server aborts during load, in the first warm-up decode (`common_context_can_seq_rm`), so no speed was measured.

1. First abort: `ggml-cuda.cu:1761 GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0)` in the internal AllReduce. A sampler node reduced over the split vocab gives a tiny PARTIAL tensor (a few floats), below the 16-byte unit of the host-staged AllReduce. Fixed in 6f21fe163: that case now returns false like the other unsupported inputs (kept, it is safe).
2. Second abort: `ggml-backend-meta.cpp:3055 fatal error` in `get_tensor_async` ([log](abort-at-load.txt)). It handles AXIS_0/1/2 and MIRRORED, not PARTIAL, so the unreduced sampler tensor cannot be read back.

The switch was removed from the code; the AllReduce fallback stays.

Hints for the next attempt:
- Give the meta backend a reduction for small PARTIAL tensors when the AllReduce declines (copy both lanes to host and add, or pad to 16 bytes), or keep sampler sums out of PARTIAL by gathering the logits row first (MIRRORED) and running the whole chain on one lane.
- Better design (ledger SP1): split-aware top-k / argmax, each lane takes its vocab half, then merge 2k candidates; exact for greedy and top-k. A full-distribution sampler (top-p 1.0) needs a distributed softmax: per-lane max and sum, exchanged as two floats per row.
- First measure the CPU sampling share on 27B MTP2 (`perf record -g` on the server thread); the ledger estimate is 1.5-3 ms per 40 ms step.
