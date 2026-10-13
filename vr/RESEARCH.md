# Remaining work

Updated 2026-10-12. This is a shortlist, not authorization to run experiments or download models. Keep tensor split at 50/50 until the final optimization pass.

## Correctness

- DFlash vision remains unvalidated. Target M-RoPE positions are not unique draft-cache row identities; examine eager/deferred injection, rollback, pinned image chunks and mixed embedding/token batches together. Preserve text regression coverage. See the [merge record](MERGE-GFX906.md#dflash2-vision-follow-up).
- One large-context server run spent about 46 seconds saving the RAM prompt cache. Saving is synchronous; allocation, checkpoint copying and readback have not been timed separately.
- Validate genuinely populated deep contexts and quantized KV separately from allocation limits.

## Performance

- Measure verification batches above eight rows under three-slot load before extending narrow matvec dispatch or changing scheduling.
- Profile remaining plain decode costs: small operations, host gaps and draft preparation. Keep accepted/rejected changes tied to end-to-end results.
- Q4_K/Q6_K research remains unmeasured for this target. Inspect actual canonical/repacked dispatch before tuning tiles or adding dense FFN fusion. A smaller quant requires a separate quality and model-selection decision.
- Investigate grouped MMQ activation reuse, GDN preparation fusion, eligible prefill graph policy, and deep-context mask/KV memory savings only where profiling supports them.
- Producer fusion, slab-overlapped AllReduce, narrow planar/LDS activation work and single-GPU DFlash replay already landed; count only additional improvements.

The [October 11 research record](old/2026-10-12/2026-10-11-gfx906-optimization-ideas-q8-q4-q6.md) preserves detailed source references, design constraints and rejected candidates. Its unfinished-work labels describe that historical tree. External patch links are discovery references; pin and inspect a coherent source revision before porting.
