# Flash Next quant and CPU placement comparison, 2026-10-29

Three user-approved single runs used traversal-fixed candidate binary SHA256 `9384bc4b695b9e2d92817366b6975f8b0cb1b003a943af4570af3200483013f1`, `libllama.so` SHA256 `ce19763a448693004f5813c273502fa4af65717b82759249906f9f318b90870d`, and the TheRock Release/HIP build. All completed successfully with a 100096 context, Q8 KV, 1024/512 batch/ubatch, 128 generated tokens, 2097 framed prompt tokens, greedy seed 42, and the `flake.nix` prompt. MTP/speculation was disabled. Each process used `-t 8 -tb 8` and `taskset -c 0-7`; these are the eight verified performance-core CPUs. CPUs 8-19 are efficiency cores. The 20-core machine was not tested with all 20 workers.

| Target and placement | PP tok/s | TG tok/s | Wall time | CPU | Max RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ3_XXS, experts on GPU | 377.50 | 38.94 | 30.49 s | 98% | 5,472,544 KiB |
| IQ4_XS, first 8 MoE layers on CPU, op-offload on | 197.59 | 24.42 | 43.68 s | 299% | 15,798,316 KiB |
| IQ4_XS, first 8 MoE layers on CPU, op-offload off | 204.92 | 25.15 | 42.49 s | 303% | 15,797,884 KiB |

The two IQ4_XS runs differ only by `--no-op-offload`. The off setting was 3.7% higher PP and 3.0% higher TG in these single runs; that small difference does not establish a benefit. The IQ3_XXS vs IQ4_XS comparison also changes quantization and MoE placement, so it cannot isolate GPU quant-kernel cost from CPU expert cost. No quality or cross-quant output claim is made.

All three allocations and requests fit. Runtime metadata identifies `general.name=Qwen3.8 Flash Next`, architecture `qwen4exp`, type A3B, 176.94B parameters. Revision paths and shard sizes are recorded in [model-files.txt](model-files.txt). The prompt SHA is in [prompt.sha256](prompt.sha256). Exact argv and environment are in [commands.txt](commands.txt) and [environment.txt](environment.txt); candidate/build and CPU topology details are in [preflight.txt](preflight.txt). [Binary hashes](binaries.sha256), all 44 hashes of resolved libraries ([ldd](ldd.txt), [libraries.sha256)), [status and time/CPU/RSS records](status.txt), and each run's [raw stdout, stderr, status, and `time -v`](.) are retained.

The level-4 runtime logs report the following buffer and memory figures. The `Meta(ROCm0,ROCm1)` label is kept as logged; values are not reinterpreted per GPU.

| Run | Meta model buffer | Repacked model buffer | ROCm_Host buffer | CPU_Mapped buffer | Final Meta breakdown: total / free / self(model + context + compute) / unaccounted | Host breakdown |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| IQ3_XXS GPU experts | 12,721.78 MiB | 13,369.11 MiB | 497.31 MiB | 27,465.95 MiB | 65,504 / 8,133 / 27,733 (26,090 + 991 + 651) / 29,636 MiB | 28,123 MiB (27,963 + 0 + 160) |
| IQ4_XS CPU MoE 8, op-offload on | 14,066.78 MiB | 12,590.85 MiB | 10,706.64 MiB | 27,465.95 MiB | 65,504 / 7,824 / 28,197 (26,657 + 991 + 549) / 29,482 MiB | 38,317 MiB (38,172 + 0 + 144) |
| IQ4_XS CPU MoE 8, op-offload off | 14,066.78 MiB | 12,590.85 MiB | 10,706.64 MiB | 27,465.95 MiB | 65,504 / 7,806 / 28,197 (26,657 + 991 + 549) / 29,500 MiB | 38,317 MiB (38,172 + 0 + 144) |

The per-layer token embedding tensor is logged as lazy-read, size 27,465 MiB. For the CPU-MoE runs the runtime reports 10,706.64 MiB in `ROCm_Host`; this is an observed buffer, not a performance attribution. The command line records op-offload selection; no explicit runtime line echoed the flag state.

No further runs or tuning were performed. The 116.66 seconds of measured process time is within the authorized 180-second total.

## Layer-split follow-up

One matched IQ3_XXS run with `-sm layer` measured **353.76 PP / 30.99 TG**, versus **377.50 / 38.94** for the tensor-split control. It completed and fit the same context, but was 6.3% lower PP and 20.4% lower TG in this single run. See the [layer-check record](layer-check/README.md) for commands, logs, per-device memory and output hashes. No further split-mode runs were made.
