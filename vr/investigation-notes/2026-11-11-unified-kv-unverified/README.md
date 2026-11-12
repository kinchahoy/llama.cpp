# Four-slot unified KV smoke checks

2026-11-11. Runtime: build-play-b1; executable and library hashes in runtime-sha256.json. Model paths and snapshot revisions are in each command.json. Tests used the existing server/models.ini settings with explicit four slots, unified KV, F16 KV, metrics and the context overrides below. No sampling parameters were supplied or changed. Servers were run sequentially and stopped after each check.

Four concurrent /completion requests used flake.nix plus an explanation request (2066 prompt tokens each), explicit slot ids 0-3, n_predict=300, ignore_eos=true and cache_prompt=false. Each successful run returned 300 tokens per slot with accepted drafts, no truncation, and healthy status afterward. This checks short populated contexts within a large allocated pool; it does not validate full-pool use, output equivalence, slot reuse or sustained load.

| Preset | Shared context requested | Result | Peak free VRAM GPU0 / GPU1, MiB |
| --- | ---: | --- | --- |
| Qwen27B MTP2 | 250000 | Four requests completed | 1959 / 1966 |
| Qwen27B DFlash5 | 240000 | GPU0 OOM loading private draft head | - |
| Qwen27B DFlash5 | 212992 | GPU0 OOM reserving draft compute buffer | - |
| Qwen27B DFlash5 | 180224 | Four requests completed, tight GPU0 margin | 395 / 7355 |
| Flash Next MTP3 | 100352 | ROCm1 OOM allocating draft KV | - |
| Flash Next MTP3 | 32768 | Four requests completed; later investigation confirms checkpoint fallback | 3398 / 971 |

Qwen MTP reports its rounded slot limit as 250112. The unified pool is shared, not multiplied by four. DFlash uses the default single-GPU draft on ROCm0. Target batch/ubatch remain 2048/2048 for Qwen and 1024/512 for Flash Next.

Qwen logs disable the optional fused "Flash Attention over K/V rows" operation after device probing. This does not disable the separately configured Flash Attention. The fallback attends over the whole unified cache with a dense mask, so it matters for large-pool memory and performance.

Flash Next reports "qwen4exp recurrent rollback currently requires n_seq_max=1; clamping n_rs_seq to 0" and "refusing rollback seq 0". Follow-up source inspection and the verbose 64K run confirm the refusal is the startup capability probe: common_context_can_seq_rm selects FULL, and the server logs "speculative decoding will use checkpoints". Rejected drafts restore checkpoints and replay accepted tokens. These warnings alone are not correctness failures. Full output equivalence and sustained load remain untested; checkpoint replay can explain the low multi-slot throughput.

Raw successful responses, logs, commands and summaries: unified-four/ for Qwen MTP and the intermediate allocation failures; unified-four-dflash180k/ for smaller-pool DFlash and Flash Next. unified-four-attempt1/ retains the first DFlash OOM and two harness argument failures; those MTP argument failures are not inference results. Reproduction scripts are check-unified.py and check-unified-smaller.py; run with the existing TheRock environment and normal fork optimization defaults.

Deployment presets were not changed by these checks. No inference process was left running.

## Larger Qwen pools and Flash Next placement follow-up

Qwen MTP2 passed at shared pool 262144 with 1340/1348 MiB free at peak, then 274432 with 744/730 MiB free, using batch/ubatch 2048/2048. Pool 350000 failed allocating draft KV at that ubatch. Changing only ubatch to 512 allowed the same 350000 pool to complete all four 300-token requests with 3406/3365 MiB free. Server logs cap each slot at the native 262144 tokens. The shared allocation is padded to 350208. This is a capacity smoke check, not a full-pool or matched throughput benchmark.

Flash Next at 100352 with the sidecar on ROCm1 still failed after moving target split to 54/46 (draft compute allocation). Moving the sidecar to ROCm0 and target split to 46/54 also failed at 100352 (draft KV allocation). The latter placement completed at 65536 but left only 10/2552 MiB free at peak, making it unsuitable as a robust deployment setting. Its verbose log confirms the checkpoint fallback described above. At 100352, sidecar on ROCm0, target split 44/56 and ubatch 128 still failed reserving the draft compute buffer (268.85 MiB). A final 43/57 attempt is recorded in unified-four-fn-draft0-100k-ts43/.

## Constraint correction

The user requires Qwen batch/ubatch 2048/2048 to preserve PP. The 350000/ubatch512 experiment does not meet that requirement and must not be used as a deployment recommendation. No matched PP baseline was run: concurrent /completion timing is not comparable to the recorded single-request or llama-bench PP results. At unchanged Qwen 2048/2048, 274432 is the largest successful pool tested here; 350000 failed. No deployment configuration was modified. Stop microbatch-reduction experiments; the final Flash Next 43/57 attempt had already failed allocation when this instruction arrived.
