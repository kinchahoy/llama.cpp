# Historical experiments and workflows

These are records, not current operating instructions. Start at [the current guide](../README.md). Old commands, hardware assumptions and permissions may no longer apply. Results describe their tested workload, often a single run; failed ideas can be worth revisiting.

Dates below are campaign dates where known. `2026-10-29/prior-guides/` is the date of archival, not a claim about when each document was written. Undated files retain unknown provenance. The cleanup started from repository checkpoint `84932540a0b8fbcd67128b8615da434763c185ec`.

| Campaign | Evidence | Source identity and caveats |
| --- | --- | --- |
| Aug 12-13 | [Q8 overlap ledger](2026-08-13/results/2026-08-13-gfx906-q8.csv), [method](2026-08-13/results/README.md) | Baseline `b8efb9510`; candidates include explicitly dirty builds. See [legacy results](undated/legacy/RESULTS.md). |
| Sep 10 | [AllReduce/repack/harness comparisons](2026-09-10/benchmark_results) | JSONL reports `9b06d8c7a`, `fd04166db`, `0abd145ec`. The Gemma AllReduce file is empty. |
| Sep 10 | [Legacy AllReduce investigation](undated/legacy/README.md) | Uncommitted merge inputs MX `0c81bd502`, official `434ddbbc0`, PR head `ee5b990d1`. Some earlier PR-labelled arms used fallback; the record corrects that attribution. |
| Oct 1 | [First port kit](2026-10-01/patches/llama-head-0bc845-to-mx-38b4/README.md), [corrected port](2026-10-01/patches/llama-head-e358d-to-mx-20261001/README.md) | First target `38b4da7ba`; later corrected source `8a6351ddf` includes MTP planning/GDN snapshot fixes. |
| Oct 11 | [Head-merge runs](2026-10-11/head-merge) | Before `5d6e2c412`; merged candidates are uncommitted integration with official `08ffba21e`. Their binary metadata still reports the old commit. |
| Oct 12 | [TP profiling](2026-10-12/tp-profile), [notes](2026-10-12/notes) | Baseline `8d73dc74e`; optimization candidates include uncommitted changes. |
| Oct 12 | [Three-slot report](2026-10-12/2026-10-12-three-slots.md), [raw evidence](2026-10-12/three-slots), [port kit](2026-10-12/patches/llama-head-08ffba21e/README.md) | Fixes local on `a39e959d9`; target tree `bf7b95184082c982eeaa9384a3b24a14a268d3e8`. Preserve the tap-reservation and per-sequence prompt-state findings. |
| Oct 19 | [Superseded kit](2026-10-19/patches/llama-head-797de63b2/README.md) | Official base `797de63b2`; target tree `08913b10dffa50666300700608215ca0b62ba11f`, not a source commit. |
| Oct 25 | [27B CLI](2026-10-25/2026-10-25-qwen3.8-27b-flake-cli.md) | Recorded build `766f7c5e0`. |
| Oct 25 | [Flash Next quant/context](2026-10-25/2026-10-25-qwen3.8-flash-next-gfx906.md) | Official base `344be0a60` plus local overlay; exact measured local tree unspecified. |
| Oct 25 | [Flash Next MTP](2026-10-25/2026-10-25-flash-next-mtp-short-check.md) | Recorded source `fdc8e5b65`. |
| Oct 25 | [Pre-grouped patch kit](2026-10-25/patches/llama-head-344be0a60/README.md) | Historical overlay on official `344be0a60`; superseded by the accepted grouped-MoE kit. |
| Oct 25 | [Flash Next grouped MoE/GPU-feedback experiment](2026-10-25/flashnext-fast/README.md) | Local changes on `fdc8e5b65`, with source hashes. Grouped-only accepted Oct 29; GPU feedback remains separate. |
| Oct 29 | [Grouped-MoE integration](2026-10-29/grouped-moe-integration/README.md) | Build/replay and 18 canonical cases passed; repacked runtime check timed out. |
| Oct 29 cleanup | [Branch/worktree record](2026-10-29/cleanup.md) | Historical names normalized; unfinished work preserved. |
| Oct 29 archival | [Previous guides](2026-10-29/prior-guides) | Superseded workflows preserved for reference. |
| Unknown | [Legacy notes/patches](undated/legacy), [loose local outputs](undated/local-outputs) | No run date or exact source inferred from filesystem timestamps. |

Raw response/probability files, patch payloads and the earlier cleanup tarball were retained. Generated archived CMake/build output and empty root `perf.data` were dropped. Historical scripts and archived exporters may assume their original paths; they are not maintained helpers. Patch payload checksums remain applicable from each kit directory.
