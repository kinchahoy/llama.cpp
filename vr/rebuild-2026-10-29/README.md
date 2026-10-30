# Fork rebuild on upstream 36a0e2ffc (2026-10-29)

The fork rebuilt as clean upstream `36a0e2ffc` plus a smaller overlay. Each patch is cumulative and applies on its own to `36a0e2ffc` (`git apply --index`). Progress notes are in [../2026-10-29-rebuild-status.md](../2026-10-29-rebuild-status.md).

| Stage | Patch | Added / removed lines vs upstream | Files |
| --- | --- | ---: | ---: |
| A | current master `e04bc2e38` (on `418f905a4`) | ~18.9k / 1.2k | 123 |
| B | `1-cleanup.patch`: dead code, debug knobs, B12 getrows fix, B1 gguf-split fix, peer AR opt-in (`GGML_CUDA_PEER_AR=1`) | 17655 / 1213 | 121 |
| C | `2-no-stages.patch`: B without multi-stage TP (`-tps`); restore from mx-llama `77ffa6093` if needed | 15735 / 1177 | 118 |
| D | `3-no-dsv4-minimax-mimo.patch`: C with DeepSeek V4/V4.1, MiniMax-M2 and MiMo extensions back to upstream | 14894 / 924 | 109 |

## Results

Two gfx906, `-sm tensor`, `scripts/bench_stage.sh` (r2) and `scripts/fn_reps.sh` (Flash Next r5). All builds use `scripts/configure_stage.sh`, which matches the build-ilp flags (`-mllvm -amdgpu-sched-strategy=max-ilp`, CXX `-O3`). A build without these flags is about 2-3% slower on 27B.

| Stage | 27B Q8_0 PP2048 | 27B TG128 | Flash Next PP2048 | Flash Next TG128 | FN r5 PP / TG |
| --- | ---: | ---: | ---: | ---: | --- |
| U (official 36a0e2ffc) | 448.60 | 32.77 | 302.99 | 39.66 | 304.3 +- 3.8 / 39.76 |
| A | 684.56 | 36.82 | 527.79 | 38.99 | - |
| B | 682.92 / 684.19 | 36.96 / 36.61 | 525.73 / 528.13 | 38.34 / 39.56 | 509.9 +- 18.0 / 39.23 |
| C | 685.60 | 36.58 | 520.22 | 36.93 +- 2.78 | 516.4 +- 8.0 / 38.79 |
| D | 685.34 | 36.67 | 527.63 | 39.09 | 520.1 +- 7.3 / 39.26 |

Greedy output (`scripts/output_check.sh`, 96 tokens) is byte-identical across B, C and D for both models (`out/`).

Speculative decoding on D, one `llama-cli` run each (`scripts/mtp_check.sh`, flake.nix prompt, 512 outputs): 27B plain 36.2, MTP2 60.7, MTP3 58.0 TG; Flash Next plain 37.2, MTP2 49.6, MTP3 44.5 TG (Q8_0 MTP sidecar). MTP text diverges from plain after about 1.6-2.0 KB of output. Details: [PERFORMANCE.md](../PERFORMANCE.md#speculative-decoding-on-d-one-run-each).

## What D drops

DeepSeek V4/V4.1, MiniMax-M2 and MiMo run on upstream code again. Lost fork-only behavior (none of these models is local, so not measured here):
- DSV4 MLA head split (`LLAMA_MLA_TP`), vocabulary-parallel DSV4 output head, narrow shared-expert buffer rule, HC gate fusion kernel, DSV4 KV cache changes, DFlash DSV4-backbone drafter split and V4.1 hc_shift.
- MiniMax-M2 under `-sm tensor` (upstream rejects it again).
- MiMo-2 SWA pattern fallback for files that count MTP layers.

Kept as generic: shared-expert split (`LLAMA_SHEXP_SPLIT`), tied output split, MTP block mirror, PLE shard, dense DFlash drafter replication, the DFlash SWA pattern fix.

## Not done yet

- HIP gating of default-on shared-code changes (B8 calc_nwarps, B14 BF16 rule, B15 graph key, B16 H2D copy, eager cuBLAS handle, AR slabs, `GGML_SCHED_SHARE_VIEW_INPUTS`). No effect on this host; B8 needs an MMVQ launch-path change.
- Moving ggml-cuda.cu glue into separate files; event_query through the proc-address table.
- Remaining DSV4 mentions in shared files (meta backend split rules, comments) are kept: they are generic TP rules.

`audit/` holds the static audit (INVENTORY.md, CONVERGENCE-PLAN.md, per-area reports) and the per-pass agent reports.
