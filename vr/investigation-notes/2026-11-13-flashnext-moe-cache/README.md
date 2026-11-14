# Flash Next with the upstream MoE cache (2026-11-13)

Upstream (merged at 9d25e0014, v1.0.0) can keep MoE experts in host memory (`-ncmoe`, `-cmoe`, `-ot`) and hold the most used ones in a GPU cache (`--moe-cache-mib`, `--moe-cache-cpu-share`). See `docs/moe-cache.md`. The cache serves batches up to 32 tokens (decode, MTP verify). Larger batches upload the experts they use.

## Probe

Build `build-catchup-20261113-200420`, Flash Next UD-IQ3_XXS, `llama-completion`, `-ts 0.52,0.48 -fa on -c 8192 -ub 512`, greedy, 256 tokens, no MTP. One run each. Logs: `<build>/validation/moe-probe/`.

| Placement | TG t/s | vs all GPU |
|---|---|---|
| all experts on GPU | 39.39 | - |
| `-ncmoe 12 -lm none` (about 12 GiB of experts on host, pinned) | 28.90 | -27% |
| `-ncmoe 12 -lm none --moe-cache-mib 4096` (about 8 GiB VRAM freed) | 33.40 | -15% |

Output stays coherent. The repack MoE fusion still engages. PP with host experts is not measured.

## Model facts

48 layers, 512 experts, top 10, expert FFN 640. Routed experts are 46.8 GiB (about 1 GiB per layer, about 2 MiB per expert per layer). The per-layer-input table (50.7 GiB) is already on host. Host RAM is 91 GB, so only part of the experts can be on host. Both GPUs use PCIe 4.0 x16.

## Ideas, not tested

- Context: about 8 GiB freed for 15% TG could take the 90K preset toward the 262K model limit. `-ncmoe` frees VRAM on ROCm0 only (first layers), so shift `-ts` toward ROCm0 to spread the KV.
- Smaller offload: `-ncmoe 4..6` with a 2-3 GiB cache should cost less TG. Measure it.
- Quality: UD-IQ4_XS does not fit fully on GPU. With about 10 GiB of experts on host plus a cache, it may run at about 0.85x the IQ3_XXS TG.
- Layer choice: `-ot` can put the experts of selected layers on host, instead of the first N. Use the `LLAMA_MOE_CACHE_TRACE` routing trace to find layers whose routing caches well.
- Tuning: run `scripts/moe-cache-tune.py -- <flashnext server command>` (6 server starts, a few minutes) to pick the cache size and CPU share.
- PP risk: each ubatch uploads the used experts of every host layer (about 1 GiB per layer). Expect a large PP loss at ubatch 512. Larger ubatches spread this cost.
- MTP: a 4-token verify step touches about 4x as many experts per layer, so the hit rate can fall. Measure it with the serving preset.
