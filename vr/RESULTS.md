# Results

Campaign started: 2026-08-12

Last updated: 2026-09-10

Hardware: two AMD MI60 (gfx906) over PCIe

Base source: `b8efb9510` (build 10275)

Candidate source: `b8efb9510-dirty`; standalone implementation patch SHA-256 `080ecd80b35237237b48f9db89f4577b588f6ddd7411cca4cfed89e8c1ac3531`.

## 2026-09-10 mx and llama.cpp head merge

Branch: `mx-upstream-2026-09-09`

Inputs: mx head `0c81bd502`, llama.cpp head `434ddbbc0`, and PR 27825 head `ee5b990d1`. The measurements were made from the resolved but not yet committed llama.cpp-head merge. Release targets `llama-cli`, `llama-bench`, and `vr-test-gfx906-tp` built successfully with HIP for gfx906 and RCCL disabled.

Canonical protocol: Qwen3.8 27B Q8_0, two MI60 GPUs, tensor split, Flash Attention, batch and ubatch 4096, actual cached depth 16384, PP2048, TG256, one repetition, and no warmup. See the exact command in [README.md](README.md).

| Arm | PP t/s | TG t/s | Finding |
| --- | ---: | ---: | --- |
| Merged mx plus llama.cpp heads | 324.4369 | 18.3208 | Baseline; PP target met |
| PR 27825 ROCm host-staged AllReduce | 325.7131 | 18.7649 | Best merged arm; keep |
| PR 27825 rebuilt, resumed-session validation | 322.2327 | 18.4168 | PP target reconfirmed; TG remains below target |
| Clean llama.cpp head, `../llamahead/build-norccl` | 196.9662 | 20.1327 | Confirms higher pure-head TG at real 16K depth |
| Clean llama.cpp head, batch/ubatch 2048/512 | 179.5703 | 19.8510 | Non-apples-to-apples versus merged 4096/4096; modest defaults do not improve TG |
| PR 27825 with `LLAMA_SHEXP_SPLIT=0` | 314.0392 | 17.6764 | Reject; shared-expert split helps PP and TG |
| PR 27825 with `GGML_RP_GENERIC_MMV=1` | 315.6193 | 18.4499 | Reject; generic MMV loses to the repacked path |
| Private exact BF16 overlap, batch/ubatch 2048 | 325.0877 | 18.5718 | Reject on this head; PR 27825 is simpler and faster |

PR 27825 improved TG by 2.42% over the merged baseline while PP increased by 0.39%. The optimized branch is 65.36% faster than clean head in PP but 6.80% slower in TG. The immediate optimization problem is therefore the gfx906 TG path rather than prompt processing or a larger global batch.

Resumed-session check on 2026-09-10: HEAD is `c0df68229`, MERGE_HEAD is `434ddbbc0`, and the index has no unresolved conflicts. The four unstaged files from the handoff and untracked `vr/build/` are present. All three requested targets rebuilt successfully with RCCL OFF. The modest clean-head run used the canonical command with `../llamahead/build-norccl/bin/llama-bench` and only batch/ubatch changed to 2048/512; clean-head source is unchanged at `434ddbbc0`. PP and TG sample times were 11405002441 ns and 12896066509 ns, respectively. This is one repetition, not a variance estimate.

Retained AllReduce focused checks passed with `HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=internal`, overlap unset: `--n 1 --check` took 38351 us and `--n 2048 --check` took 81601 us. Both logged successful pipeline initialization. The harness checks 16 output values against 17408; this is a limited execution check, not model-quality validation. Its `wire=f32` label describes the old overlap control, not the generic pipeline's default BF16 round-trip (`GGML_CUDA_AR_BF16_THRESHOLD=1`).

The clean head result does not reproduce 22 TG at 16K depth, but it supports the earlier observation: the shallower interactive run can reasonably be about 22 TG while the long-context canonical result is 20.13 TG. Use 20.13 TG as the first like-for-like target and retain 22 TG as the application goal.

### 2026-09-10 afternoon TG isolation

Correction: source inspection and the TG32 kernel trace show that the historical arms labeled PR 27825 did not exercise its AllReduce kernels with overlap off. `ggml_backend_cuda_comm_allreduce_internal()` still returned false on HIP unless the old exact overlap was active, so these arms used the meta-backend fallback. Pipeline initialization in the focused checks did not prove pipeline execution. The BF16 default described above applies only when the pipeline actually runs. Treat the earlier attribution of the throughput difference to PR 27825 as unproven.

The earlier no-repack process lost its session handle without a captured result. It was rerun with persistent output under `/tmp/mx-tg-20260910/`. All arms below use the same merged binary, canonical 4096/4096 settings, and real 16K depth. No source changes were made for these experiments.

| Arm | PP t/s | TG t/s | Raw output |
| --- | ---: | ---: | --- |
| Repacking disabled (`-nr 1`) | 130.4371 | 16.9108 | `no-repack.jsonl` |
| Repacking enabled, serial dispatch (`GGML_META_PARALLEL_DISPATCH=0`) | 317.0269 | 18.7604 | `serial.jsonl` |
| Fresh default control | 325.4868 | 18.6829 | `control.jsonl` |
| HIP overlap-only guard removed; generic AllReduce active | 394.1701 | 25.8895 | `ar-enabled.jsonl` |

Against the fresh control, disabling repacking loses about 59.9% PP and 9.49% TG. Serial dispatch gains only 0.42% TG while losing 2.60% PP; do not promote it from this single sample. Keep repacking and parallel dispatch. The no-repack result does not support blaming repacked weight layout alone for the clean-head TG advantage.

The TG32 trace is in `/tmp/mx-tg-20260910/profile/`. Tracing stretched decode to 123.14 seconds (0.2599 TG), so its absolute timings are not representative. It contains repacked single-token kernels and host/device copies, but no `ggml_cuda_ar` kernel, which prompted inspection of the stale HIP guard. Removing only that guard allows the existing generic pipeline to execute.

Retain the 11-line guard removal: it raises PP by 21.10% and TG by 38.57% against the fresh fallback control, meeting the 300+ PP / 22-25 TG target at 16K. Both requested samples completed with no warmup and one repetition; sample times are 5195727007 ns for PP2048 and 9888164938 ns for TG256. The old overlap controls remain unset, with repacking and parallel dispatch at their defaults. No 64K run was performed.

The CLI, bench, and focused test rebuilt successfully. After removing the guard, both `--n 1 --check` and `--n 2048 --check` passed. The n=1 trace explicitly records four `ggml_cuda_ar_kernel<float, __hip_bfloat16>` launches in `ar-check/n1_kernel_stats.csv`, proving execution rather than just pipeline initialization. The n=2048 unprofiled check took 67643 us. Logs are `ar-check.log` and `ar-check-2048.log`. These all-ones checks cover limited execution correctness, not model quality; active transport defaults to BF16, unlike the earlier fallback arms. No perplexity or model-output equivalence claim is made.

Whole-token capture requires custom peer-access AllReduce through `comm_ar_prepare`; it cannot capture the retained host-staged path. Disabling parallel dispatch also disables this capture attempt. These are one-shot screening measurements, not estimates of run-to-run variance.

### 64K winner gate

The PR 27825 winner was run once with the identical command except for `-d 65536`.

| Depth | PP t/s | TG t/s | PP change from 16K | TG change from 16K |
| ---: | ---: | ---: | ---: | ---: |
| 16384 | 325.7131 | 18.7649 | - | - |
| 65536 | 216.0927 | 16.2924 | -33.65% | -13.17% |

The 64K cache and benchmark fit on both 32 GB MI60 GPUs. This is historical evidence only: the user canceled further 64K testing on 2026-09-10 and set the active target to 300+ PP and 22-25 TG at depth 16384.

## 2026-08-14 head merge and Qwen3.8 validation

The private patch merged cleanly after pulling mx head `46b95d97e` (build 10279). `git apply --check --reverse vr/patches/q8-tp-overlap.patch` passes on the merged tree. The standalone patch remains unchanged, with SHA-256 `080ecd80b35237237b48f9db89f4577b588f6ddd7411cca4cfed89e8c1ac3531`.

The production build uses Release, HIP for gfx906, RCCL enabled at build time, ccache disabled, and `LLAMA_BUILD_VR_EXPERIMENTS=ON`. The UI asset fetch had no network access, so the build embedded the existing local UI assets. The server, CLI, bench, and focused test targets all built successfully.

Qwen3.8 model: `unsloth/Qwen3.8-27B-GGUF`, local `Qwen3.8-27B-Q8_0.gguf`, 29036089344 tensor bytes, model SHA-256 `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`. Direct GGUF inspection found 506 Q8_0 and 360 F32 tensors. All 65 `ffn_down.weight` tensors have the eligible Q8_0 shape, and the GGUF contains the integrated `nextn` MTP tensors.

| Check | Result |
| --- | --- |
| Focused `n=1` fallback | Pass, 34570 us |
| Focused `n=2048` BF16 overlap | Pass, 43938 us; optimized path explicitly logged |
| Qwen3.8 PP2048 at depth 16384 | 319.478 t/s |
| Qwen3.8 TG1024 at depth 16384 | 18.4011 t/s |
| Score `TG + PP/10` | 50.3489 |
| Estimated 16K+1K time | 106.93 s |

The paired benchmark used tensor split, Flash Attention, batch/ubatch 2048/2048, direct I/O, one repetition, no warmup, and the five established runtime variables. Raw transcript: `~/infer/qwen38-gfx906-results/2026-08-14-head-46b95d97e/paired-q8.typescript`, SHA-256 `014b366647618d66af30b6340f0840bf93b0d10ce13bc05c8488ad6e1a49d41d`.

Qwen3.8 MTP used the preserved 5072-byte synthetic engineering prompt, greedy sampling, seed 42, 1024 generated tokens, and one run per arm. Depth 2 measured 243.9 prompt t/s and 29.7 generation t/s. Depth 4 measured 243.7 prompt t/s and 28.8 generation t/s. Keep `--spec-draft-n-max 2` as the evidence-based quickstart setting for this prompt; MTP acceptance and the winning depth remain content-dependent.

MTP transcripts and outputs are in `~/infer/qwen38-gfx906-results/2026-08-14-head-46b95d97e/`. The exact input remains `~/infer/qwen36-gfx906-results/2026-08-12-quant-selection/qwen36-synthetic-mtp-prompt.txt`, SHA-256 `e1140a074f250751be8d3b22c85d91e8b1a1236f11a3e0baffa52ecae229b4f7`.

## Correctness

All focused checks passed. The expected output is 17408 for a Q8_0 matrix and F32 activation filled with ones.

| Path | n | Time us | Result |
| --- | ---: | ---: | --- |
| Initial broad HIP transport, overlap off | 1 | 26751 | Pass |
| Initial broad HIP transport, overlap off | 2048 | 68423 | Pass |
| F32 two-slab overlap, pre-gate | 2048 | 52684 | Pass |
| BF16-wire two-slab overlap, final gate | 2048 | 42105 | Pass |

The GPUs reported no direct peer access. The overlap tests used the HIP copy-staged path. The focused timing is diagnostic, not the acceptance benchmark.

## Full-model benchmark

Model: `unsloth/Qwen3.6-27B-MTP-GGUF:Q8_0`

Protocol: tensor split, flash attention, context 16384, PP2048, TG1024, batch and ubatch 2048, one measured repetition, no warmup, three-second cooldown.

Established production reference: PP 282.976 t/s, TG 19.8199 t/s, weighted score 48.1175.

| Arm | Internal HIP transport | Overlap | Wire | PP t/s | TG t/s | Score | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| C0 production binary | previous behavior | off | F32 | 282.976 | 19.8199 | 48.1175 | Reference |
| C1 candidate | matching nodes only | off | F32 | 283.038 | 18.4718 | 46.7756 | Same-build control |
| C2 candidate | matching nodes only | on | F32 | 276.340 | 18.0294 | 45.6634 | Reject |
| C3 candidate, final | matching nodes only | on | BF16 | 318.778 | 19.6266 | 51.5044 | Keep as opt-in |

Score is `TG + PP/10`, matching the current 10 PP t/s = 1 TG t/s preference.

C3 improved PP by 12.65% against production and 12.63% against the same-build control. Its score was 3.3869 above production. The measured 16K+1K wall time was 103.57 seconds versus 109.56 seconds for production, a 5.47% reduction. The final version skips internal validation when no overlap node is active; TG returned to 19.6266 t/s, close to the 19.8199 production reference.

Conclusion: keep C3 as an opt-in Q8_0 PP2048 optimization. Reject F32 overlap and broad HIP transport. BF16 wire changes reduction precision; the focused test proves execution and synchronization correctness, not model-quality equivalence.

Raw logs are in `~/infer/qwen36-gfx906-results/2026-08-12-mx-vr-port/`.

Final benchmark command:

```sh
GGML_CUDA_ALLREDUCE=internal \
GGML_ENABLE_CUSTOM_AR=1 \
HSA_FORCE_FINE_GRAIN_PCIE=1 \
GGML_CUDA_TP_OVERLAP=1 \
GGML_CUDA_TP_OVERLAP_BF16=1 \
HIP_VISIBLE_DEVICES=0,1 \
/tmp/mx-tp-overlap-build/bin/llama-bench \
  -m ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q8_0.gguf \
  -ngl 99 -fa on -sm tensor -dev ROCm0/ROCm1 -lm dio \
  -b 2048 -ub 2048 -p 0 -n 0 -pg 2048,0 -pg 0,1024 \
  -d 16384 -r 1 -o jsonl --progress --no-warmup
```

## Rejected broad HIP transport

The first integration let the new copy-staged transport handle every HIP collective. It produced PP 262.514 t/s and TG 14.0310 t/s, score 40.2824. This is substantially below the reference, especially for TG. The implementation was narrowed before the overlap arms: non-matching collectives now return `false` and retain mx's existing meta-backend path. Raw log: `~/infer/qwen36-gfx906-results/2026-08-12-mx-vr-port/c1-transport.typescript`.

## UD-Q8_K_XL comparison

UD-Q8_K_XL contains eligible Q8_0 down tensors despite its mixed-quant package name. The same-build overlap-off control measured 281.024 PP t/s; enabling BF16 overlap increased it to 314.852 PP t/s, a 12.04% gain.

| Model and path | Size GiB | PP t/s | TG t/s | Score | 16K+1K seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0, BF16 overlap | 27.04 | 318.778 | 19.6266 | 51.5044 | 103.57 |
| UD-Q8_K_XL, BF16 overlap | 33.31 | 314.852 | 15.7750 | 47.2601 | 116.95 |

UD-Q8_K_XL nearly matches optimized Q8_0 PP, trailing by 1.23%, but ordinary TG is 19.63% slower. It is 6.27 GiB larger and takes 12.92% longer for the weighted 16K+1K scenario. Keep Q8_0 as the speed recommendation; use UD-Q8_K_XL only when its higher-fidelity tensor mix justifies the TG and memory cost.

Raw logs: `ud-q8-k-xl-final-build.typescript` and `ud-q8-k-xl-final-build-overlap-off.typescript` in the artifact directory above.

### What UD-Q8_K_XL changes

Direct GGUF inspection shows that UD-Q8_K_XL is a selective higher-precision mix, not a uniform Q8_K tensor format.

| Model | Q8_0 | F16 | F32 | Tensor bytes |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 506 | 0 | 360 | 27.04 GiB |
| UD-Q8_K_XL | 291 | 119 | 456 | 33.31 GiB |

UD promotes 215 Q8_0 tensors: 119 to F16 and 96 small recurrent alpha/beta tensors to F32. The largest added bytes are the F16 embedding and LM head (+2.22 GiB combined), all 48 recurrent gate and output pairs (+2.64 GiB), and five FFN down/gate/up triplets (+1.17 GiB). Three attention Q/K pairs are also F16. The promoted FFN layers are 50, 51, 59, 62, and 63; attention Q/K is promoted in layers 51, 59, and 63.

Sixty of 65 FFN down tensors remain Q8_0 and match the exact PP overlap path. At PP2048, each weight matrix is reused across 2048 columns and gfx906 has efficient F16 matrix arithmetic, so the added weight bytes are amortized. At TG1, there is almost no weight reuse: the LM head, recurrent gate/output matrices, and FFNs are streamed for every token. This makes TG mostly weight-bandwidth-bound and explains why a 23.18% larger model is 19.63% slower in measured TG despite nearly equal PP.
