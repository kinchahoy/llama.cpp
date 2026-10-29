# Performance evidence

Recorded results on two gfx906 GPUs. PP is prompt processing and TG is token generation, in tokens/s. These are historical measurements, not current regression-test results. Compare only matched workloads and runtime builds.

| Date and workload | Setting | PP / TG | Recorded source identity |
| --- | --- | --- | --- |
| Oct 12, synthetic PP2048/TG128 | Qwen3.8-27B Q8_0, plain | about 682 / 36.8 | Exact commit unknown from the summary; local changes were in use |
| Oct 25, `flake.nix`, 4000-output limit | Qwen3.8-27B Q8_0, MTP2 | 549.7 / 56.8 | Reported build `11776`, commit `766f7c5e0` |
| Same workload | MTP3 | 545.2 / 60.0 | Same reported build |
| Same workload | DFlash7 | 543.7 / 59.9 | Same reported build |
| Oct 25, short CLI, 256-output limit | Qwen3.8-27B Q8_0, plain | 558.9 / 35.2 | Reported build `b11929-fdc8e5b65` |
| Oct 25, matched Flash Next long CLI | UD-IQ3_XXS, plain | 396.9 / 37.3 | Same reported build |
| Same Flash Next workload | MTP2, Q8_0 sidecar | 368.5 / 42.9 | Same reported build |
| Oct 25, Flash Next MTP2, short CLI | Grouped MoE off / on, GPU feedback off | 330.3 / 47.65 -> 372.9 / 50.45 | Local experiment on `fdc8e5b65`; source hashes and patch archived |

The Oct 25 27B runs used batch/ubatch 2048, context capacity 100096, and one run per setting. MTP3 and DFlash7 were effectively tied for TG. Flash Next's matched long runs used batch/ubatch 1024/512, Q8_0 KV, a 4092-token prompt and a 4096-output limit; MTP raised TG by 15% while lowering PP by 7%. Model and quantization differences prevent treating the Flash Next/27B comparison as an isolated architecture result.

Full commands and evidence: [27B CLI](old-ignore/2026-10-25/2026-10-25-qwen3.8-27b-flake-cli.md), [Flash Next MTP](old-ignore/2026-10-25/2026-10-25-flash-next-mtp-short-check.md), and [October profiles](old-ignore/2026-10-12/notes/2026-10-12-tp-profile-findings.md). Reported build commits identify the recorded version; they do not guarantee the absence of local source edits or identify dynamically linked libraries by themselves.

## What these results establish

- Speculation helps some single-request workloads; plain decoding was fastest in the recorded three-request mixed workload. See the [three-slot report](old-ignore/2026-10-12/2026-10-12-three-slots.md).
- Allocating a large context is not testing a fully populated context. The Oct 25 27B runs used about 6K prompt-plus-output tokens despite a 100K capacity setting.
- Target-logprob checks support greedy selection within each run. They do not establish general model-quality equivalence, exact plain/speculative output identity, or all rollback behavior.
- DFlash vision, other models/quants, quantized KV, and maximum populated contexts remain outside these checks.

For a new comparison, retain the source and binary/library identities, local model revision, exact command, environment, input, and actual token counts. Keep the 50/50 split fixed and compare an immediate baseline with the candidate. Ask for explicit OK before performance runs; use the [catch-up helper](how-to-catchup.md) for a bounded preview. Older benchmark records and changed workflows belong in [old-ignore](old-ignore/README.md).

[Grouped-MoE evidence](old-ignore/2026-10-25/flashnext-fast/README.md): two runs per variant, same prompt and output/acceptance, about 5.9% mean TG improvement. PP varied substantially. Canonical IQ2_S/IQ4_NL reference cases and repacked IQ4_NL cases passed; this does not establish general quality equivalence. The accepted integration retains the opt-in grouped path only, not GPU feedback.

The [Oct 29 integration check](old-ignore/2026-10-29/grouped-moe-integration/README.md) passed the build, replay and 18 canonical reference cases; the repacked harness did not finish within the runtime limit. No new performance claim is made for that integration.

## 2026-10-29 official-head check

The supplemental Qwen3.8-27B Q8_0 baseline measured **690.256 PP2048 / 36.802 TG128**. An early, unrepaired 418f port measured Flash Next IQ3_XXS grouped MTP2 at **345.09 / 37.68** on the baseline and **247.31 / 40.22** on the candidate, one run each: PP -28.3%, TG +6.7%, with identical output and draft acceptance. Follow-up source review found the port had changed gfx906 MMQ launch traversal. Restoring gfx906 row-first traversal restored PP behavior; that correction is in the active patch set. The initial negative result is retained as history and does not measure the corrected active tree. A subsequent matched MTP2 pair measured 347.97 / 36.77 on the accepted baseline and 351.30 / 39.51 after restoring traversal, with identical output and acceptance; see the [traversal check](old-ignore/2026-10-29/head-merge-validation/terra-port-fix/README.md). The final IQ2_S change was checked separately with MTP off. The build and 18 canonical grouped reference cases passed. [Commands, identities, raw records and investigation](old-ignore/2026-10-29/head-merge-validation/README.md).

## 2026-10-29 Flash Next quant and CPU placement

With the traversal-fixed official-head candidate, IQ3_XXS with GPU experts measured 377.50 PP / 38.94 TG. IQ4_XS with eight CPU MoE layers measured 197.59 / 24.42 with op-offload on and 204.92 / 25.15 with it off. These are one run each with 8 workers pinned to the verified P-core CPUs 0-7. The small on/off difference does not establish a gain; the IQ3/IQ4 comparison changes both quantization and expert placement. All three 100K-context requests fit. [Commands, placements, buffer reports, CPU telemetry, identities and raw logs](old-ignore/2026-10-29/quant-placement-compare/README.md).

One matched IQ3_XXS layer-split check measured 353.76 PP / 30.99 TG versus the tensor-split control at 377.50 / 38.94. It fit, but was 6.3% lower PP and 20.4% lower TG in this single run. This does not support switching the recorded configuration to layer split. [Full record](old-ignore/2026-10-29/quant-placement-compare/layer-check/README.md).

## 2026-10-29 IQ2_S single-token experiment

A gfx906-only, two-row IQ2_S one-token expert MMVQ path passed 3 focused CPU-reference cases and is included in the active patch set. Its gate is narrow: IQ2_S, expert IDs, one output column, and even row count; odd rows use the existing path. One matched 20-thread, MTP-off Flash Next pair with a doubled prompt (4,143 tokens) and 256 generated tokens measured **396.35 / 37.20 PP/TG** for the traversal-fixed baseline and **398.80 / 38.08** for the candidate. Output matched. The +0.62% PP / +2.37% TG in one pair is not repeatability evidence. This path does not select dense Q8_0 and no Q8_0 dense benefit was measured. [Commands, library identities, CPU telemetry, source diff and logs](old-ignore/2026-10-29/iq2s-single-token-experiment/README.md).
