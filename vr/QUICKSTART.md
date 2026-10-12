# Quickstart: build and run on two gfx906 GPUs

For the MI60 + MI50 box with Qwen3.8-27B Q8_0. Measured numbers and details: [TP profile findings](2026-10-12-tp-profile-findings.md), [roofline](2026-10-12-roofline.md).

## Build

```sh
cmake -S . -B build-ilp \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF \
  -DLLAMA_OPENSSL=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument"
cmake --build build-ilp --config Release -j16
```

`GGML_HIP_GRAPHS=ON` is required: the fast decode path replays one graph per token. Keep RCCL off; the built-in host-staged AllReduce is faster on this box (no P2P between the cards).

## Run

CLI:

```sh
./build-ilp/bin/llama-cli \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on \
  -b 2048 -ub 2048 -c 150000 \
  --temp 0.0 --top-k 1 -s 42
```

Fastest generation (speculative, MTP depth 2; final boss `-f flake.nix -n 4000`: PP 546-558, TG 55-62 vs 32-35.6 plain):

```sh
./build-ilp/bin/llama-cli \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on \
  -b 2048 -ub 2048 -c 32768 \
  --temp 0.0 --top-k 1 -s 42 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling
```

Greedy text is not bit-identical to plain decode: the batched verify kernels round differently, so a near-tie flips after some hundred tokens.

Server: the same model and GPU flags with `llama-server` and `--host` / `--port`.

What matters:

| Flag | Use | Why |
| --- | --- | --- |
| `-sm tensor -dev ROCm0,ROCm1` | always | Tensor parallel over both GPUs |
| `-fa on` | always | Flash attention |
| `-b 2048 -ub 2048` | default choice | Best measured PP; 4096 was slower |
| `-ts` | 50/50 for chat; `53/47` for prompt-heavy work | 53/47: PP 702 vs 684, TG 35.6 vs 36.8 (one split for both phases) |
| `-c` | as needed | Capacity only; it does not slow short prompts |
| `-bs` | do not use | Backend sampling is refused with `-sm tensor` |
| `--spec-type draft-mtp --spec-draft-n-max 2` | for faster generation | MTP; depth 2 beat 3 at 4000 tokens (56.7 vs 53.8-55.5), 1 and 4 are slower |
| `--spec-draft-p-min` | do not use | Variable verify widths: TG drops to about 30 |
| `--spec-type draft-dflash` | not now | n4000 n-max 5: PP 533, TG 51.4 (MTP2: 546 / 55.1); PP is now on the limit, TG is lower |

## Environment

All optimizations are on by default. Do not set `GGML_CUDA_TP_OVERLAP` (it does nothing useful on HIP).

These switches exist only to turn things off for A/B tests:

| Variable | Default | Effect when changed |
| --- | --- | --- |
| `GGML_CUDA_AR_SLABS` | 4 | 0: no prefill AllReduce/compute overlap |
| `GGML_CUDA_AR_DUPLEX` | 1 | 0: prefill AllReduce copies run one direction at a time |
| `GGML_RP_MMV_ROWS` | 8 | 16 or 4: rows per workgroup of the Q8_0 mat-vec |
| `GGML_CUDA_FUSE_ADD_RMS_NORM` | 1 | 0: no fused residual add + norm |
| `GGML_CUDA_Q8_1_PRODUCER` | 1 | 0: the fused norm stops pre-quantizing the next mat-vec input |
| `GGML_CUDA_CONCAT_TRANSPOSED` | 1 | 0: generic concat for the GDN conv input |
| `GGML_META_TOKEN_GRAPH` | 1 | 0: no whole-token graph (much slower decode) |
| `GGML_RP_NC_PLANAR` | 1 | 0: 2-8 column mat-vec reads block_q8_1 rows instead of planar activations |
| `GGML_RP_NC_L16` | 1 | 0: 64 lanes per row for 4-8 columns (verify about 10 ms slower at 8) |
| `GGML_RP_NC_LDS` | 1 | 0: 2-8 column mat-vec reads activations from L1/L2 instead of an LDS tile (about 3% slower at 6 columns) |
| `LLAMA_MTP_REPACK` | 1 | 0: MTP blocks and output head keep canonical weights |
| `LLAMA_DRAFT_REPACK` | 1 | 0: sidecar draft (DFlash) loads canonical weights |
| `LLAMA_SPEC_LAZY_PREFILL` | 1 | 0: draft prompt replay runs before the first token (lower CLI PP) |
| `GGML_META_INNER_GRAPHS_EAGER_OFF` | 0 | 1: per-lane graphs off at the first token-graph attempt |
| `GGML_META_TOPK_BCAST_LOGITS` | 0 | 1: broadcast top-k over logits across lanes |
| `LLAMA_DFLASH_MIRROR_OUTPUT` | 0 | 1: mirror the head for a DFlash draft (slower) |
| `LLAMA_DFLASH_DRAFT_TP` | 0 | 1: keep the DFlash draft tensor-parallel instead of on the first GPU |
| `LLAMA_DFLASH_PRIVATE_HEAD` | 1 | 0: no private head copy (a single-GPU draft then fails) |
| `LLAMA_DFLASH_ASYNC_REPLAY` | 1 | 0: draft prompt replay on the calling thread (lower PP) |
| `LLAMA_DSPARK_DRAFT_WINDOW` | 4096 | 0: replay the whole prompt into the draft |
| `LLAMA_DSPARK_FLUSH_ROWS` | 2048 | rows between lag-one prompt replays |
| `LLAMA_TAP_ACCUM_MROPE` | 1 | 0: tap accum only for n_pos == 1 (DFlash prompt goes eager again on Qwen3.8, PP about 430) |
| `LLAMA_TAP_ACCUM_MIN` | 1 | 16: no accum rows for short ubatches (short prompt tails go eager) |
| `LLAMA_DFLASH_DEFER_TAIL` | 1 | 0: short prompt chunks take the eager DFlash path and force the replay |
| `LLAMA_TAP_EMBD_SKIP` | 1 | 0: prompt taps are copied to host twice |
| `GGML_RP_NC_FUSION`, `GGML_RP_NC_RPL`, `GGML_RP_NC_23` | off | experiments: narrow up/gate fusion, 4 rows per lane, 2-3 column shapes (no gain) |

## Check it is working

```sh
GGML_META_PROFILE=1 ./build-ilp/bin/llama-bench -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 -b 2048 -ub 2048 -p 2048 -n 128 -r 2
```

- The `[meta-prof] replayed tokens=...` line must appear. If it is missing, decode is not using the token graph.
- Expected on 2026-10-12: about 682 PP2048 and about 36.8 TG128. The CLI command above with `-f flake.nix -n 256` gave about 558 PP / 36 TG.
- Optional power headroom on ROCm1 (MI50 with Pro VII VBIOS, capped at 170 W of 178 W): `echo 178000000 | sudo tee /sys/class/drm/card2/device/hwmon/hwmon*/power1_cap`. Check the card number first; not measured.
