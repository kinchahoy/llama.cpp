# Flash Next expert placement with the upstream MoE cache (2026-11-13)

Upstream (merged at 9d25e0014, v1.0.0) can keep MoE experts in host memory (`-ncmoe`, `-cmoe`, `-ot`) and hold the most used ones in a GPU cache (`--moe-cache-mib`, `--moe-cache-cpu-share`). See `docs/moe-cache.md`. The cache serves batches up to 32 tokens (decode, MTP verify). Larger batches copy cached experts on the device and upload the rest. Only GPUs that hold host experts get cache budget.

Build `build-catchup-20261113-200420`. One run per row, no MTP, `-fa on`, `-ts 0.52/0.48` unless noted, host experts pinned (`-lm none`). Logs: `<build>/validation/moe-probe/`.

## Hardware

ROCm0 = MI60 (64 CUs, bus 03, CPU PCIe lanes). ROCm1 = MI50 (60 CUs, bus 87, chipset lanes shared over DMI). The MI60 is 10-20% faster and has the better host link. 91 GB RAM.

## Model sizes (GiB)

| Quant | routed experts | other GPU weights | per layer experts |
|---|---|---|---|
| UD-IQ3_XXS | 46.8 | 4.2 | 0.72-1.27 |
| UD-Q3_K_XL | 52.3 | 4.2 | 0.72-1.37 |
| UD-IQ4_XS | 56.2 | 4.2 | 0.72-1.37 |

The per-layer-input table (50.7 GiB) is on host for all of them. KV costs about 30 KB per token (f16, split evenly over both GPUs): 90K = 2.6 GiB, 200K = 5.9 GiB, 262K = 7.5 GiB. q8_0 halves it. ROCm1 also holds the larger compute buffer (694 vs 286 MiB at 8K) and the MTP draft, so it fills first.

## Measurements: IQ3_XXS, llama-bench PP2048 / TG128

| Config | ub | PP | TG |
|---|---|---|---|
| all GPU | 512 | 753 | 39.8 |
| all GPU | 2048 | 652 | 40.4 |
| `-ncmoe 2` (MI60 layers) | 512 | 600 (-20%) | 36.3 (-9%) |
| `-ncmoe 4` (MI60 layers) | 512 | 490 (-35%) | 33.1 (-17%) |
| `-ncmoe 4` | 2048 | 575 (-12% vs ub 2048) | 33.7 |
| layers 44-47 on host (MI50 layers) | 512 | 443 (-41%) | 33.7 |
| all GPU, `-ts 0.58/0.42` | 512 | 721 (-4%) | 40.1 |
| f16 KV at depth 16K | 512 | 414 | 36.6 |
| q8_0 KV at depth 16K | 512 | 412 | 34.2 (-7%) |

Each host layer costs about 10% PP at ub 512 and about 3% at ub 2048: the used experts are uploaded at about 10.5 GiB/s, not overlapped with compute. Without the cache, TG costs about 4.5% per host layer (the CPU computes those experts).

## Measurements: MoE cache, llama-completion, 256 tokens (short prompt) or 2464-token prompt + 128 tokens

| Config | short TG | long PP | long TG |
|---|---|---|---|
| IQ3_XXS all GPU | 39.4 | 456 | 38.95 |
| IQ3_XXS `-ncmoe 4`, no cache | 34.2 | 329 | 19.1 |
| IQ3_XXS `-ncmoe 4`, 1 GiB cache | 33.6 | - | - |
| IQ3_XXS `-ncmoe 4`, 2 GiB cache | 39.0 | 331 | 34.8 |
| IQ3_XXS `-ncmoe 4`, 2 GiB, cpu share 0.5 | 32.3 | - | - |
| IQ3_XXS `-ncmoe 12`, 4 GiB cache | 33.4 | - | - |
| IQ4_XS `-ncmoe 9`, no cache (llama-bench, ub 512 / 2048) | 30.7 | 311 / 492 (PP2048) | - |
| IQ4_XS `-ncmoe 12`, 4 GiB cache | 32.1 | 198 | 27.6 |

The cache does not help prefill. A cache below about half of the host experts thrashes (1 GiB is worse than none). CPU share hurts on this rig. The hit rate depends on the text: the short prompt nearly recovers the all-GPU speed, the document prompt keeps an 11% loss.

## Strategy

- 200K context: use q8_0 KV first. 200K at q8_0 needs about 1.46 GiB per GPU, about 0.17 GiB per GPU more than today's 90K f16 pool, at no PP cost and about 7% TG at depth. Free the rest on ROCm1 by moving a little split to the MI60 (`-ts 0.54/0.46`, about 1-2% PP). Only if that is not enough, put 2 MI60 layers on host with a 1 GiB cache. Getting 200K with f16 KV by offloading needs about 3.3 GiB more, so 4-6 host layers and 20-35% less PP. Not worth it.
- 3-4 host layers on the current IQ3_XXS: always take MI60 layers (`-ncmoe N`, CPU-lane link). Use a cache of at least half the host experts (2 GiB for 4 layers). Expect about -10% PP per layer at ub 384-512 (use ub 1024-2048 when offloading), TG -1% to -11%. Net saving is only about 0.45 GiB per layer, so this is a last resort, not a context strategy.
- IQ4_XS: about 12 host layers plus a 4 GiB cache to fit the MTP preset. Expect TG -18% to -29% and PP about -57% at ub 512 (less at ub 2048). Use it only if the quality gain is worth it; Q3_K_XL (+5.5 GiB, about 6 host layers) sits halfway.

## Ideas to test (code)

1. Overlap uploads with compute in prefill: upload the next host layer's experts on a copy stream while the current layer runs. At ub 2048 a layer computes for about 65 ms and needs about 90 ms of upload, so most of the 3% per layer could go.
2. Upload speed: 10.5 GiB/s is about half of PCIe 4.0 x16. Check the copy size and pinning in `llama_moe_cache::copy_experts` and the scheduler copy path. 2x upload speed halves the PP cost.
3. Layer choice by routing locality: record `LLAMA_MOE_CACHE_TRACE` on real chats and put the layers that cache best on host with `-ot` (within the MI60's layer range).
4. Move the MTP draft to ROCm0 once offload gives ROCm0 real headroom. Earlier this did not help because tensor split moves in coarse steps.
5. Fixed hot set: pin each layer's hot experts from the trace instead of LRU, so MTP verify (about 4x the experts per step) and 2 users do not evict each other.
6. Re-check the upstream GCN MMQ retune per quant type (old table: `git show b79696f1f^:ggml/src/ggml-cuda/mmq-config-gcn.cuh`). It gave Flash Next +8.9% PP at ub 512, but ub 2048 is now slower than ub 512.
7. Try `LLAMA_MTP_DEV_ROWS=1` (upstream device rows for the MTP draft) on the serving preset.

None of these were tested with MTP or in the server yet.
