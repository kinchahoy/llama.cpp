# Option 1 (deferred input staging): attribution and result (2026-10-29)

Method: temporary fprintf in ggml_backend_tensor_set and sched compute_splits (reverted, git tree clean), llama-bench -p 0 -n 8 -r 1, last decode tokens.

## Attribution per decode token
- llama set_input (inp_tokens, inp_pos, rs_s_copy, s_copy_main, leaf_1): all write to ROCm_Host (pinned) buffers, 4-5 tiny memcpy, ZERO GPU copies, no meta set_tensor. So path (a) is not a source.
- All GPU-bound copies come from the scheduler split-input copies (split 1, meta backend), 2 splits per token (split 0 = CPU embedding gather).
  - Staged async path (already fork code: sched input staging, F32<=4096 B, I32<=16 KiB, I64<=64 B, F16 big; non-graph F32 from CPU <=32 KiB): meta set_tensor_async per lane, no host sync. 27B: ~7 tensors (inp_mixed_slots 8 B, s_copy_main 4, inp_pos 16, k/v_idxs 8+8, kq_mask 512, out_ids 4, model.input_embed 20480 non-graph) x 2 lanes = ~14 copies (matches trace).
  - Unstaged sync path (memcpy + stream sync per lane via ggml_backend_tensor_copy -> meta buffer set_tensor):
    - 27B: inp_mixed_embd 20480 B F32 (1 tensor, 2 lane copies, 2 syncs).
    - Flash Next: inp_mixed_embd 10240 B, attn_inp_k_rot 262144 B, attn_inp_v_rot 16384 B, one 16 B "(reshaped)" tensor: 4 tensors x 2 lanes = 8 copies + 8 syncs. Remaining FN copies (~30+) are staged async ones (leaf_8..11, k/v idxs, inp_pos, masks, etc.).
  - Reason unstaged: F32 staging cap 4096 B in ggml_backend_sched_input_staging_type_max_size.

## Experiment (no code change, env only, baseline binary mx-final/build-rb, TG128 r3, alternating)
GGML_SCHED_INPUT_STAGING_F32_MAX=1048576 stages all of the above:
- 27B: BASE 37.09, 37.13; STG 37.11, 37.13 (delta ~0)
- FN:  BASE 38.43, 38.66; STG 38.45, 38.12 (delta -0.1, noise)

## Conclusion
Not implemented. The meta set_tensor path is not the cost, and removing the remaining sync copies (the only candidates) gives no TG gain. The per-token syncs/copies seen before the launch are mostly scheduler/event overhead already async, and the remaining gap is not on the host critical path unprofiled. No code changed; GGML_META_INPUT_STAGING knob not added (nothing to gate). Proposal if pursued: only the knob GGML_SCHED_INPUT_STAGING_F32_MAX exists already; a larger default is not justified by data.
