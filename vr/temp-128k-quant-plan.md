# Temporary 128K quant plan

Date: 2026-11-02. Planning notes only; no new downloads or inference runs performed. Estimates are not measured allocation or throughput results.

## Host and constraints

- Two 32 GiB gfx906 GPUs (MI60 + MI50), 91 GiB reported host RAM (96 GB installed), Core Ultra 7 265K, fast NVMe.
- Keep approximately 500 MB free on each GPU after model, context and compute allocations. Do not reserve an arbitrary additional 8 GiB.
- Target one sequence at 131072 tokens. Context capacity includes prompt and generated tokens.
- Use the standard Hugging Face cache for all model downloads. Do not pass `--local-dir` or introduce a separate model directory.
- At inspection, the checkout filesystem had about 1.8 TB available. Check the actual HF cache filesystem before downloading.
- `common/fit.cpp` rejects `SPLIT_MODE_TENSOR`; `--fit-target 500` cannot automate placement in this mode. Use `--fit off` and tune manually.
- Keep the existing 50/50 tensor split and host-staged AllReduce. Read [PERFORMANCE.md](PERFORMANCE.md) for measured evidence and its source identities. Ask for explicit OK before performance runs.

## DeepSeek V4.1 Flash 1017

Source: <https://unsloth.ai/docs/models/deepseek-v4.1-flash> and <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json>.

| Quant | Download GB | SSD Engram GB | Non-Engram GiB |
| --- | ---: | ---: | ---: |
| UD-IQ1_S | 233.0 | 110.6 | 114.0 |
| UD-IQ1_M | 250.4 | 110.6 | 130.2 |
| UD-IQ2_XXS | 300.5 | 135.2 | 154.0 |

Recommendation: try IQ1_M with its default IQ4_NL Engram on SSD; IQ1_S is the fallback for lower CPU weight load and more page-cache space. IQ2_XXS consumes essentially all combined memory before runtime allocations.

With 155 GiB combined physical capacity and approximately 1 GiB total GPU spare, IQ1_M leaves about 23.8 GiB for OS, context, compute, repacking and Engram cache. CPU and GPU memory are separate pools; this arithmetic alone does not prove a placement fits.

The local DSV4.1 cache implementation stores compressed KV and index keys together, K only. At 128K, three source layers have 65536 rows and one has 131072 rows, each 640 values wide:

`(3 * 65536 + 131072) * 640 * 2 = 400 MiB` for F16, or 212.5 MiB for Q8_0. This is logical compressed-cache storage, excluding sliding-window cache, compressor state, device mirrors/replication and compute buffers. Budget conservatively for possible device replication until startup logs confirm physical allocation.

Initial CLI placement (not an optimum): keep the first 28 layers' routed experts on CPU. Decrease `--n-cpu-moe` to move more experts onto GPU, checking actual allocations and the requested spare on both devices. Increase it if allocation fails. The value 28 has not been derived from a tensor allocation tally.

```sh
source /home/raistlin/amd-clean/env/therock.sh

./build-ilp/bin/llama-cli \
  -hf unsloth/DeepSeek-V4.1-Flash-1017-GGUF:UD-IQ1_M --offline \
  -dev ROCm0,ROCm1 -sm tensor -ts 1,1 \
  -ngl all --n-cpu-moe 28 --fit off \
  --load-mode mmap --lazy-mode on \
  --override-tensor 'per_layer_token_embd.weight=CPU' \
  -c 131072 -b 512 -ub 128 \
  -fa on -ctk f16 -ctv f16 -t 20 -tb 20 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --chat-template-kwargs '{"reasoning_effort":75}'
```

No DSpark drafter in the first configuration. Successful execution and speed on these gfx906 devices remain unverified.

## Qwen3.8 Flash Next: prioritize speed

Source: <https://unsloth.ai/docs/models/qwen3.8-next>. Local GGUF tally used cached revision `766911a6b7369840a91dbcd95f9f997acaab6cd6` for IQ3_XXS and IQ4_XS. A newer revision may differ.

| Quant | Total weights GiB | PLE GiB | Non-PLE GiB | Room after non-PLE weights and 1 GiB total GPU spare |
| --- | ---: | ---: | ---: | ---: |
| UD-IQ3_XXS | 76.32 | 26.82 | 49.50 | 13.50 GiB |
| UD-Q3_K_XL | ~83.82 | ~26.82 | ~57.00 | ~6.00 GiB |
| UD-IQ4_XS | 87.23 | 26.82 | 60.42 | 2.58 GiB |

IQ3 and IQ4 values are sums of GGUF tensor bytes. Q3 uses the published 90 GB total and assumes the same PLE table; inspect its tensors before treating that split as established. Tables exclude runtime repacking overhead.

Cached metadata: 48 layers, attention every fourth layer (12 attention layers), two KV heads, K/V head dimensions 256, indexer dimension 128 with two stored rows per token. At 128K:

| Logical cache | Q8_0 GiB | F16 GiB |
| --- | ---: | ---: |
| Attention K + V | 1.59 | 3.00 |
| Indexer | 0.40 | 0.75 |
| Total | 1.99 | 3.75 |

Recurrent state, compute buffers, device replication, repacking and drafter allocations are additional. PLE stays on CPU, backed by mmap; 91 GiB host RAM can accommodate this 26.82 GiB table comfortably when other models are unloaded.

- Speed-first baseline: IQ3_XXS with MTP2. Existing short-context measurements: about 39.1 t/s plain and 49.6 t/s MTP2. Neither predicts populated-128K speed.
- Better-quality trial: Q3_K_XL with all experts on GPU, initially without MTP. About 4 GiB remains after estimated Q8 cache and spare; actual compute/repacking allocations determine whether it fits.
- IQ4_XS leaves only about 0.59 GiB after logical Q8 cache and spare. CPU expert offload is likely required. Existing IQ4_XS with eight CPU MoE layers measured 24-25 t/s versus about 39 t/s for GPU-resident IQ3_XXS; quant and placement changed together, so this is not a clean quant-only comparison.
- Flash Next profiling indicates small operations and dependencies dominate plain decode. A larger quant is a quality experiment, not a demonstrated speed improvement.

Initial Q3 launch:

```sh
source /home/raistlin/amd-clean/env/therock.sh

./build-ilp/bin/llama-cli \
  -hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q3_K_XL --offline \
  -dev ROCm0,ROCm1 -sm tensor -ts 1,1 \
  -ngl all --fit off \
  --load-mode mmap --lazy-mode on \
  --override-tensor 'per_layer_token_embd.weight=CPU' \
  -c 131072 -b 1024 -ub 512 \
  -fa on -ctk q8_0 -ctv q8_0 -t 20 -tb 20 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
  --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

For the speed baseline, select `UD-IQ3_XXS` and use the existing Q8_0 MTP sidecar with `-md`, plus `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling`. See [mtp_check.sh](scripts/mtp_check.sh) for the previously tested sidecar path and settings. Check both GPUs' remaining memory at 128K before assuming MTP fits. Grouped MoE remains an opt-in experiment, not part of the baseline.

## HF login, disk check and download

Use the same shell for these commands. No token needs to be placed in command arguments. Xet high-performance mode and four file workers are starting settings, not a measured download optimum.

```sh
hf auth whoami

python - <<'PY'
from huggingface_hub.constants import HF_HUB_CACHE
from pathlib import Path
import shutil

p = Path(HF_HUB_CACHE)
while not p.exists():
    p = p.parent
print(f"HF cache: {HF_HUB_CACHE}")
print(f"Available: {shutil.disk_usage(p).free / 1024**3:.1f} GiB")
PY

hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
  --include 'UD-IQ1_M/*.gguf' --dry-run

hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --include 'UD-Q3_K_XL/*.gguf' --dry-run
```

For fresh downloads, allow 300 GiB free for DeepSeek or 100 GiB for Qwen, including download/cache overhead. If downloading both, allow 400 GiB. Dry-run reports the currently missing files; cached files can reduce the needed space. Confirm space on the cache filesystem, then run the desired command:

```sh
HF_XET_HIGH_PERFORMANCE=1 \
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
  --include 'UD-IQ1_M/*.gguf' --max-workers 4

HF_XET_HIGH_PERFORMANCE=1 \
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --include 'UD-Q3_K_XL/*.gguf' --max-workers 4
```

Rerun the same command after interruption; do not use `--force-download`. The initial plan does not pin remote revisions; record the downloaded snapshot identity before comparing results. Keep all shards in the HF cache; `-hf ... --offline` selects the cached quant for launch.
