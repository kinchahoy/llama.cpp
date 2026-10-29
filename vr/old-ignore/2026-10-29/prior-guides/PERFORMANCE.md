> Historical record. Commands and recommendations may be obsolete; see [the current guide](../../../README.md).

# Current performance evidence

Updated 2026-10-25. Target: Qwen3.8-27B Q8_0, two gfx906 GPUs, HIP graphs, host-staged AllReduce, batch/ubatch 2048. These are recorded observations; workload and run order affect the numbers.

## One active request

| Workload | Plain TG | MTP2 TG | DFlash5 TG |
| --- | ---: | ---: | ---: |
| Five diverse prompts, 512 outputs, mean | 35.3 | 52.3 | 55.5 |
| `flake.nix`, 3000 outputs, server logprobs | 32.5 | 52.9 | 52.0 |

The later DFlash implementation uses a single-GPU draft, private repacked head and asynchronous prompt replay with a 4096-row draft window. DFlash5 won three of five diverse prompts; MTP2 won the longest document prompt. Choose using representative requests. Earlier findings that describe a TP draft or recommend MTP universally predate these changes.

Plain synthetic PP2048/TG128 reached about 682/36.8 after token graph capture, producer fusion and slab-overlapped AllReduce. The earlier 649 PP measurement predates slab overlap. Single-request speculative CLI PP is about 546-606 on the recorded long prompts; synthetic PP and CLI PP are different workloads.

Full tables, prompt lengths, acceptance, race fix and numerical observations: [later performance and correctness evidence](../../2026-10-12/notes/2026-10-12-roofline.md#dflash2-faster-than-mtp2-2026-10-12-later). Earlier implementation trials and reverted experiments: [profile record](../../2026-10-12/notes/2026-10-12-tp-profile-findings.md).

## Three simultaneous requests

Plain decoding was faster and allowed more context in the mixed long/short prompt check. Single-request speculative gains do not predict concurrent throughput. Use the [three-slot report](../../2026-10-12/2026-10-12-three-slots.md) for capacity, wall times and fixed-build validation.

The optimized narrow repacked matvec handles up to eight rows. Three active MTP2 slots can verify nine rows; DFlash5 can verify eighteen. Leaving that path is a plausible slowdown source; its contribution has not been measured separately.

## Correctness limits

Recorded single-request checks found zero non-argmax tokens in returned target logprobs, including 3000-output runs. Plain and speculative outputs can diverge at near ties because verification uses different batch widths and rounding. These checks support greedy selection against each run's reported logits; they do not prove all numerical differences come from rounding or establish general quality equivalence.

The three-slot review found separate DFlash tap-allocation and shared prompt-state bugs. Their fixes and regression evidence belong in the three-slot report. Vision, other target models/quants, quantized KV and fully populated maximum contexts remain unvalidated by these runs.

## 2026-10-25 CLI update

The current one-run Qwen3.8-27B Q8_0 results at context 100096 are recorded in [the flake.nix CLI report](../../2026-10-25/2026-10-25-qwen3.8-27b-flake-cli.md). MTP3 had the highest measured TG (60.0 tok/s); MTP2 had the highest PP (549.7 tok/s). These single-run CLI results are not directly comparable to the older multi-prompt averages above.


## 2026-10-25 Flash Next quant and context check

UD-IQ3_XXS ran at 552.3 PP / 36.7 TG with a 4,092-token file prompt, a 4,096-token output limit, Q8 KV and context set to 100096 on the two gfx906 cards. A shorter 2,046-token prompt run measured 434.7 / 38.0. The GGUF advertises 262144 context; a one-token context-allocation check passed at that setting, but a fully populated 262K conversation was not tested. UD-IQ4_XS could not reserve prefill buffers at 100K with all model layers on the GPUs, even at batch/ubatch 256 and Q4 KV. Full settings and limits: [Flash Next gfx906 report](../../2026-10-25/2026-10-25-qwen3.8-flash-next-gfx906.md).

## 2026-10-25 Flash Next MTP2 check

The short `flake.nix` CLI check measured Qwen3.8-27B Q8_0 at 558.9 PP / 35.2 TG and Flash Next UD-IQ3_XXS without MTP at 454.9 / 38.4. For a matched long comparison at 100K context, batch/ubatch 1024/512 and a 4,092-token prompt with a 4,096-token output limit, Flash Next measured 396.9 / 37.3 without MTP and 368.5 / 42.9 with MTP2. MTP2 raised TG by 15% and lowered PP by 7% in these single runs. It needs the separate self-contained Q8_0 MTP sidecar via `-md`; the target GGUF alone has no MTP layer. The 27B Q8_0 long CLI results on the previous build were 549.7 / 56.8 for MTP2 and 545.2 / 60.0 for MTP3 at batch/ubatch 2048/2048. See the [MTP diagnosis, matched comparison and commands](../../2026-10-25/2026-10-25-flash-next-mtp-short-check.md).
