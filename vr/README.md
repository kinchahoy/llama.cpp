# gfx906: current guide

For the two 32 GiB MI60/MI50 GPUs and Qwen3.8-27B Q8_0. Updated 2026-10-12.

| Document | Purpose |
| --- | --- |
| [Quickstart](QUICKSTART.md) | Build, CLI commands, environment switches |
| [Three-slot validation](2026-10-12-three-slots.md) | Server capacity, concurrency fixes, measured limits |
| [Performance](PERFORMANCE.md) | Latest results, workload differences and measurement method |
| [Research](RESEARCH.md) | Remaining experiments and unresolved correctness work |
| [Merge guide](MERGE-GFX906.md) | Preserve the runtime overlay when updating upstream |
| [Active patch kit](patches/llama-head-08ffba21e/README.md) | Apply, regenerate and verify the five runtime patches |
| [Historical records](old/README.md) | Earlier measurements, designs and handoffs; recent edits preserved |

Use a fixed 50/50 tensor split for comparisons and capacity checks. Tune the split only after choosing the other settings. Keep `GGML_CUDA_TP_OVERLAP` unset; the normal HIP path uses host-staged AllReduce without P2P.

The active C++ harness is `CMakeLists.txt` plus `test-gfx906-tp.cpp`, enabled with `-DLLAMA_BUILD_VR_EXPERIMENTS=ON`. Raw measurements remain under [results](results/).

## Benchmark workflow

Compare identical prompt bytes, model revision, context capacity, batch sizes, sampling, generation limit, environment and warmup policy. The quick CLI check is the [quickstart command](QUICKSTART.md) with `-f flake.nix -st -n 256 --simple-io`; use longer generation only when authorized. `llama-bench` is useful for screening and merge regression checks; confirm performance claims with representative CLI or concurrent server requests.

Record source and binary identity, prompt hash, exact command, actual prompt/output counts, PP, TG, first-token latency and draft acceptance. Alternate baseline/candidate order and repeat small differences. Context allocation does not establish performance at a fully populated context. Greedy text equality alone does not validate rollback or numerical correctness.
