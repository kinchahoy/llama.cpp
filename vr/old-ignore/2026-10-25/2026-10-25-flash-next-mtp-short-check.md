> Historical record. Commands and recommendations may be obsolete; see [the current guide](../../README.md).

# Flash Next MTP diagnosis and performance on gfx906

Measured 2026-10-25 on two 32 GiB MI60 cards (gfx906), TheRock ROCm 10.2.0, using `build-ilp/bin/llama-cli` build `b11929-fdc8e5b65` at source commit `fdc8e5b65cd6120b2f89e509d5f354bda4a3a30b`. One run per setting. Prompt: repository `flake.nix` (2,046 raw tokens, SHA-256 `fe8d514fa094db2d26b29e4819d5585658533e0f552815620bd1bd54fdb1e475`); output limit 256; context 100096; batch/ubatch 2048; greedy sampling; flash attention; tensor parallel on ROCm0 and ROCm1. Flash Next used Q8_0 KV and UD-IQ3_XXS weights.

| Short check: model and setting | PP (tok/s) | TG (tok/s) |
| --- | ---: | ---: |
| Qwen3.8-27B Q8_0, no MTP | 558.9 | 35.2 |
| Qwen3.8 Flash Next UD-IQ3_XXS, no MTP | 454.9 | 38.4 |

On this short check, Flash Next without MTP had 19% lower prompt processing and 9% higher generation throughput than the 27B Q8_0 model without MTP. The quantizations differ, so this is a model-and-quant comparison rather than an isolated architecture comparison.

The first MTP attempts omitted the required draft sidecar. The Flash Next IQ3_XXS target GGUF has no MTP layer, so `--spec-type draft-mtp` alone tries to create an MTP context from a model that does not contain one. The publisher's [MTP instructions](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md) use a separate head with `-md`.

For this build, use the self-contained `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` sidecar. The Qwen4Exp loader requires token embeddings in the draft GGUF; the smaller `shared-Q8_0` sidecar omits them and is not the tested choice here. At 100K, the Q8_0 sidecar with batch/ubatch 2048/2048 failed while allocating the MTP KV cache, short by a 195.5 MiB allocation on ROCm0. Reducing batch/ubatch to 1024/512 allowed the same context to fit.

| Matched long check, IQ3_XXS target | PP (tok/s) | TG (tok/s) |
| --- | ---: | ---: |
| No MTP | 396.9 | 37.3 |
| MTP2, self-contained Q8_0 sidecar | 368.5 | 42.9 |

These single runs used the same 4,092-token prompt, 4,096-token output limit, context 100096, Q8_0 KV, batch/ubatch 1024/512, greedy sampling and two-GPU tensor parallelism. MTP2 raised TG by 15% and lowered PP by 7%. During MTP2 decoding the GPUs had about 0.6-0.7 GiB VRAM free each, so this 100K setup has little headroom.

For Qwen3.8-27B Q8_0, the separate 4,000-output CLI runs on build 11776 measured MTP2 at 549.7 PP / 56.8 TG and MTP3 at 545.2 / 60.0. See the [27B Q8_0 report](2026-10-25-qwen3.8-27b-flake-cli.md); those results used batch/ubatch 2048/2048 and a different build.

## Reproduction commands

Run from the repository root after sourcing the local ROCm environment. For the Flash Next command, replace the `-m` value below with the cached UD-IQ3_XXS shard path.

```sh
source /home/raistlin/amd-clean/env/therock.sh
LLAMA=/home/raistlin/infer/mx-llama.cpp/build-ilp/bin/llama-cli
QWEN=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
FLASH=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
COMMON=(-ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -b 2048 -ub 2048 -c 100096 \
  -ctk q8_0 -ctv q8_0 --temp 0 --top-k 1 --seed 42 \
  -f /home/raistlin/infer/mx-llama.cpp/flake.nix -n 256 --no-display-prompt --simple-io)

"$LLAMA" -m "$QWEN" --offline "${COMMON[@]}"
"$LLAMA" -m "$FLASH" --offline "${COMMON[@]}"
```

These are short single-run checks, not repeated measurements. The first MTP2/MTP3 attempts omitted the draft sidecar and are not compatibility or performance results.

## Matched long comparison commands

```sh
source /home/raistlin/amd-clean/env/therock.sh
LLAMA=/home/raistlin/infer/mx-llama.cpp/build-ilp/bin/llama-cli
FLASH=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
MTP=/home/raistlin/.cache/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --include 'MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf' \
  --local-dir /home/raistlin/.cache/models/qwen38-flash-next-mtp
PROMPT=/tmp/qwen38-flashnext-flake2x.nix
cat flake.nix flake.nix > "$PROMPT"
LONG=(-m "$FLASH" --offline -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on \
  -b 1024 -ub 512 -c 100096 -ctk q8_0 -ctv q8_0 --temp 0 --top-k 1 --seed 42 \
  -f "$PROMPT" -n 4096 --no-display-prompt --simple-io)

"$LLAMA" "${LONG[@]}"
"$LLAMA" "${LONG[@]}" -md "$MTP" \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling
```

The prompt was made with `cat flake.nix flake.nix`; its SHA-256 was `d121f276660ecead20225aea5e164b8c3a105273c6453833ac17178320241fa1`.
