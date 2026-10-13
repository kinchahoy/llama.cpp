# Current performance evidence

Updated 2026-10-12. Target: Qwen3.8-27B Q8_0, two gfx906 GPUs, HIP graphs, host-staged AllReduce, batch/ubatch 2048. These are recorded observations; workload and run order affect the numbers.

## One active request

| Workload | Plain TG | MTP2 TG | DFlash5 TG |
| --- | ---: | ---: | ---: |
| Five diverse prompts, 512 outputs, mean | 35.3 | 52.3 | 55.5 |
| `flake.nix`, 3000 outputs, server logprobs | 32.5 | 52.9 | 52.0 |

The later DFlash implementation uses a single-GPU draft, private repacked head and asynchronous prompt replay with a 4096-row draft window. DFlash5 won three of five diverse prompts; MTP2 won the longest document prompt. Choose using representative requests. Earlier findings that describe a TP draft or recommend MTP universally predate these changes.

Plain synthetic PP2048/TG128 reached about 682/36.8 after token graph capture, producer fusion and slab-overlapped AllReduce. The earlier 649 PP measurement predates slab overlap. Single-request speculative CLI PP is about 546-606 on the recorded long prompts; synthetic PP and CLI PP are different workloads.

Full tables, prompt lengths, acceptance, race fix and numerical observations: [later performance and correctness evidence](old/2026-10-12/2026-10-12-roofline.md#dflash2-faster-than-mtp2-2026-10-12-later). Earlier implementation trials and reverted experiments: [profile record](old/2026-10-12/2026-10-12-tp-profile-findings.md).

## Three simultaneous requests

Plain decoding was faster and allowed more context in the mixed long/short prompt check. Single-request speculative gains do not predict concurrent throughput. Use the [three-slot report](2026-10-12-three-slots.md) for capacity, wall times and fixed-build validation.

The optimized narrow repacked matvec handles up to eight rows. Three active MTP2 slots can verify nine rows; DFlash5 can verify eighteen. Leaving that path is a plausible slowdown source; its contribution has not been measured separately.

## Correctness limits

Recorded single-request checks found zero non-argmax tokens in returned target logprobs, including 3000-output runs. Plain and speculative outputs can diverge at near ties because verification uses different batch widths and rounding. These checks support greedy selection against each run's reported logits; they do not prove all numerical differences come from rounding or establish general quality equivalence.

The three-slot review found separate DFlash tap-allocation and shared prompt-state bugs. Their fixes and regression evidence belong in the three-slot report. Vision, other target models/quants, quantized KV and fully populated maximum contexts remain unvalidated by these runs.
