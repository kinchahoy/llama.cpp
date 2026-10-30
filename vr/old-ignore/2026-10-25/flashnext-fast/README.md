> Historical experiment. Grouped MoE was accepted for integration on 2026-10-29; GPU feedback remains experimental and is not included in the accepted patch set. These measurements predate that integration. See [current improvements](../../../IMPROVEMENTS.md).

# Flash Next GPU feedback and grouped verification experiment

2026-10-25. Isolated worktree: `/home/raistlin/infer/mx-llama.cpp/.worktrees/flashnext-fast`. No commits. The original working tree was preserved.

## Implementation

- `src/llama-mtp-device.cpp` and `.h`: persistent GPU hidden state and token feedback, GPU top-10 sampling, pinned scalar results, and a single-sequence Qwen4Exp MTP entry point. The common speculative driver uses it only with `LLAMA_QWEN4_MTP_DEVICE=1`. It requires a single-GPU draft; the measured commands use `-devd ROCm1`.
- `ggml/src/ggml-cuda/flashnext-moe.cuh`: groups canonical IQ2_S/IQ4_NL assignments by expert and decodes weights once for up to eight entries at a time. It preserves supported SwiGLU fusion and falls back for other fusion combinations.
- `ggml/src/ggml-cuda/q8_repack/flashnext-moe.cuh`: grouped repacked IQ4_NL down projection using the existing expert routing map and repack traits. The two expert paths are enabled by `GGML_CUDA_FLASHNEXT_MOE=1` on gfx906.
- Thin runtime, graph and dispatch hooks connect these files. Q8_0 kernels are unchanged. Both experiments are disabled by default.

The device path keeps hidden state and tokens on GPU but waits at each decode boundary before reusable memory metadata is changed. Fully asynchronous back-to-back decoding lost second-position acceptance; the synchronization restores the control result. This is not a fully asynchronous graph chain.

## Standard CLI comparison

Two 32 GiB gfx906 GPUs. Same `flake.nix` bytes, greedy sampling, seed 42, 256 output limit, flash attention, 50/50 tensor parallel target, context capacity 100096 and Q8_0 target KV. Raw prompt SHA-256: `fe8d514fa094db2d26b29e4819d5585658533e0f552815620bd1bd54fdb1e475`. The CLI reported 2097 framed prompt tokens and 256 generated tokens. This is not a populated 100K-context benchmark.

Flash Next target: cached Unsloth UD-IQ3_XXS revision `c8b5954a88c2775c546b92593eda40ea041d3176`, self-contained Q8_0 MTP sidecar, MTP2, batch/ubatch 1024/512. Each variant used a fresh process and the default warmup policy. Run order was control, grouped, device, both, then the reverse order.

| Flash Next MTP2 setting | Mean PP | Mean TG | TG range | Runs |
| --- | ---: | ---: | ---: | ---: |
| Single-GPU draft, both experiments off | 330.3 | 47.65 | 47.3-48.0 | 2 |
| Single-GPU draft, grouped only | 372.9 | 50.45 | 49.1-51.8 | 2 |
| Single-GPU draft, device feedback only | 376.4 | 48.35 | 48.3-48.4 | 2 |
| Single-GPU draft, both enabled | 326.6 | 49.55 | 47.3-51.8 | 2 |
| Tensor-parallel draft control, experiments off | 379.2 | 47.30 | 47.3-47.3 | 1 |

Grouped-only improved average TG by about 6% over the matched single-GPU control. Device feedback alone showed a smaller gain, and the combined setting had more variation. The fastest measured setting is grouped-only; these short runs do not establish a reliable additional benefit from enabling device feedback. PP varied substantially and should not be interpreted as an isolated prefill improvement.

All eight single-GPU Flash Next runs and the TP draft control produced identical generated text. Each accepted 157 of 194 draft proposals (80.928%); mean output length per round was 2.62, with acceptance by draft depth 0.876 and 0.742.

## 27B regression check

Cached Qwen3.8-27B Q8_0 revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`, same prompt/context/KV/sampling/output settings, batch/ubatch 2048/2048, built-in MTP2. Flag order was off/on/on/off.

| Qwen3.8-27B setting | Mean PP | Mean TG | TG range | Runs |
| --- | ---: | ---: | ---: | ---: |
| Both flags off | 575.8 | 52.60 | 52.3-52.9 | 2 |
| Both flags on | 577.0 | 53.30 | 53.3-53.3 | 2 |
| Plain decode, both flags on | 585.0 | 35.60 | 35.6-35.6 | 1 |

No measured 27B regression. All four MTP2 runs produced identical generated text. Flags-on averaged 53.3 TG versus 52.6 flags-off; this small difference is not evidence that the new paths accelerate the dense model. The existing original binary also completed checks at 34.7 TG plain and 48.7 TG MTP2; the same-binary flag comparison above is the stronger isolation of this experiment.

## Validation

- HIP Release build passed for `llama-cli`, `llama-bench` and `test-backend-ops` with the existing TheRock environment and ILP compiler setting.
- Optional CMake target `vr-test-flashnext-moe` builds through `LLAMA_BUILD_VR_EXPERIMENTS=ON`.
- 18 canonical CPU-reference cases passed: IQ2_S/IQ4_NL; two, three and four verification tokens; broadcast/per-expert activations; strided weights; offset IDs; duplicate routes.
- 24 repacked IQ4_NL CPU-reference cases passed: 17/2560 rows; two, three and eight tokens; broadcast/per-expert activations; shared/disjoint expert selections across tokens. Maximum NMSE was below 1.1e-14. Repacked routing uses unique expert IDs within each token, as the model router supplies.
- All 14 final CLI runs succeeded. Generated-text hashes and full commands are in `final-runs.json`; raw stdout/stderr and per-run JSON are alongside it.
- `git diff --check` passed. `experiment.patch` was checked against the preserved original working-tree snapshot.

An initial wider strided IQ2_S test entered the MMQ fallback and faulted. The final canonical cases are limited to widths that exercise the changed MMVQ path. This experiment does not validate or repair the MMQ fallback. The full upstream suite, other GPU architectures, sampled verification, vision, concurrent requests, nonzero confidence thresholds and populated maximum contexts were not tested.

## Reproduction

Build and run from the isolated worktree after sourcing `/home/raistlin/amd-clean/env/therock.sh`. The build configuration uses HIP graphs, gfx906, Release, RCCL off, `-mllvm -amdgpu-sched-strategy=max-ilp` and `-O3 -Wno-unused-command-line-argument`.

```sh
cmake -S . -B build-fast -DLLAMA_BUILD_VR_EXPERIMENTS=ON
cmake --build build-fast --target llama-cli llama-bench test-backend-ops vr-test-flashnext-moe -j 8
GGML_CUDA_FLASHNEXT_MOE=1 build-fast/bin/test-backend-ops test -b ROCm0 -o MUL_MAT_ID -p 'type_a=(iq2_s|iq4_nl).*n_mats=16,n_used=10'
GGML_CUDA_FLASHNEXT_MOE=1 build-fast/bin/vr-test-flashnext-moe
python3 vr/results/flashnext-fast/run-checks.py final
```

For the fastest measured configuration, use this worktree's CLI with the recorded Flash Next command, add `-devd ROCm1`, set `GGML_CUDA_FLASHNEXT_MOE=1`, and leave `LLAMA_QWEN4_MTP_DEVICE=0`. Set the latter to `1` to exercise GPU feedback too.

## Artifacts

- `experiment.patch`: only the experimental code/test changes relative to the original working tree, excluding inherited changes.
- `baseline-snapshot.patch`: original tracked working-tree changes relative to `fdc8e5b65cd6120b2f89e509d5f354bda4a3a30b`, excluding the large VR patch-kit directory.
- `source-manifest.json`: source and binary SHA-256 identities.
- `final-runs.json`: the 14 final records and generated-text hashes.
- `canonical-cpu-reference.log` and `repacked-cpu-reference.log`: passing reference checks.
