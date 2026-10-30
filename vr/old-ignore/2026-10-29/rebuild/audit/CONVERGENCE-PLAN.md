# Fork convergence plan (audit only, 2026-10-29)

Base: upstream `418f905a4`; HEAD `e04bc2e38`. upstream/master is 1 unrelated SYCL commit ahead, so the whole diff (~18.9k+/1.2k-, 123 files) is ours. Static audit only: nothing was built, run or benchmarked. Detail: `INVENTORY.md` (271 features, 88 quick wins, 33 findings, ~110 env vars) and the per-area reports `01-*.md` ... `10-*.md` in this directory.

## Executive summary

- **Nothing is superseded by upstream (class D is empty).** The fork sits on a 1-day-old base, so upstream has not caught up with anything. The divergence is real fork code, not stale copies.
- **Safe to eliminate: about 3.0-3.5k added lines (~17-19%).** No default behavior changes:
  - pipeline stages ~2.0k (decided: cut, with a pointer to mx-llama);
  - default-off debug/trace ~0.7k;
  - dead code and dead experiment knobs ~0.5k.
- **Optional: another ~1.0-1.3k.** This is the DeepSeek V4/V4.1, MiniMax and MiMo specific code. Its only recorded wins are from mx-llama's 8-10 GPU rigs.
- **Must stay (~11k):**
  - Q8 repack (3.75k);
  - 2-GPU TP core: lanes, token graph, AR overlap (2.5k);
  - speculative runtime (1.8k), context/taps (1.5k), recurrent snapshot/KV (1.0k);
  - Qwen model paths (0.9k), kernels/fusions (0.6k), gfx906 MMQ traversal and IQ2_S path;
  - peer-write AR (1.4k, kept for a future topology fix).
- **Main maintenance risks:**
  - Several default-on changes in shared CUDA/scheduler code also alter NVIDIA/CDNA behavior. They should be gated to HIP or gfx906.
  - The public `llama_model_params` struct got a field inserted mid-struct (an ABI break). Removing the stages fixes it.
  - About 700 lines of `ggml-cuda.cu` glue could live in separate files, which would cut merge conflicts.

## Confirmed bugs (verified by reading the code myself)

| ID | Where | Problem | Fix |
| --- | --- | --- | --- |
| B12 | `ggml-cuda/getrows.cu:267` | Unsharded IQ4_NL/MXFP4 gather with `ne00 % 256 != 0` writes zeros. The comment says `n_rows == 0` disables the mask; the kernel does not implement that. Hits Flash Next's 160-wide PLE table whenever it is not tensor-sharded (single GPU, `-sm layer`). The Oct 29 layer-split check recorded speed only, not output. | `if (n_rows > 0 && (i01 < 0 \|\| i01 >= n_rows))`; add a `test-backend-ops` GET_ROWS case with ne00=160 |
| B1 | `tools/gguf-split/gguf-split.cpp:565` | String literal split by a raw newline: target does not compile (normal builds skip it) | join the line |
| B10 | `ggml-cuda.cu:1410-1433`, `tp-allreduce.cu` | If ROCm ever reports peer access, the one-shot peer AR silently takes every F32 AR, switches the wire from BF16 to F32 and disables whole-token graph capture | make the peer AR opt-in (env or CMake), keep the code |
| B23 | `include/llama.h:332` | `tensor_parallel_size` inserted mid-struct (ABI break) | goes away with the stage removal |
| B24/25 | `deepseek4.cpp`, `llama-kv-cache-dsv4.cpp:1491` | `dsv41_n_kv_source` is never assigned: all V4.1 tier branches and the `LLAMA_DSV41_*` knobs are dead | populate it or delete it (see decisions) |

Unverified, needs a check or test: B3 (`ggml_backend_event_query` returns true when the backend has no query), B4/B5 (two-buffer layout cache ignores buffer ids), B6 (Q8_1 cache has no content version), B7 (MMQ J padding vs chunk J), B19/B20 (layer-tap accum branch skips the host copy and the extract_all assert). Full list: `INVENTORY.md` section 4.

## Deviation inventory (by theme)

| Theme | ~LOC added | Class | Upstream equiv | Recommendation | Confidence |
| --- | ---: | --- | --- | --- | --- |
| Q8 repack (`q8_repack/`) | 3751 | B | none (discussion 21130) | keep; prune dead knobs Q29-Q33, Q36 | high |
| 2-GPU TP core (meta lanes, token graph, AR slabs/duplex, counters) | 2460 | B | partial (base meta backend, PR 27825, PR 28198) | keep; A/B the unmeasured defaults (AR_SLABS, SHARE_VIEW_INPUTS) | high |
| Pipeline stages (`-tps`, stage partition/seams/transfers) | 1990 | C, unused on 2 GPUs | none | **cut**, pointer to mx-llama `77ffa6093` | high (unused); medium (entangled) |
| Speculative runtime (DFlash/MTP replay, lazy prefill, single-GPU draft) | 1790 | B/C | base has DFlash/MTP | keep; upstream the fixes (B33 etc.) separately | high |
| Context runtime (taps, accum, pre-norm, ring) | 1532 | A/B/C | none | keep; remove dead ext API (Q61, Q62) | high |
| Peer-write AR (`tp-allreduce.cu/.cuh`) | 1379 | C, inactive (no P2P) | none | **keep** (user); gate opt-in; delete the uncalled function (Q38) | high |
| KV / recurrent memory | 966 | A | none | keep; upstream generic fixes (B28) | med-high |
| Qwen models (qwen35/35moe/4exp, delta-net) | 937 | B/C | partial | keep | high |
| DeepSeek V4/V4.1 + DSV4 KV | 781 (+~300 in shared files) | B/C | base has DSV4 | decision; delete the dead V4.1 plumbing either way | high |
| Debug / trace / profile knobs | 780 | E | n/a | remove (default off, no vr evidence) | high |
| Kernels / fusions (GDN chunk, norm, softcap, concat, argsort, top-k) | 630 | A/B | none for these | keep; argsort >1024 cols is an upstream bugfix candidate | high |
| MMQ / MMVQ | 532 | A/B | base GCN table | keep traversal + IQ2_S; delete dead gfx906 config (Q28); gate `calc_nwarps` (B8) | high |
| Flash Next grouped MoE | 250 | B, opt-in | none | keep opt-in; keep one implementation (Q36) | med |
| Build / misc / docs | 241 | - | - | drop the CANN stub (Q86); keep vr CMake option | med |
| MiniMax / MiMo / other | 179 | C | - | decision (non-target) | med |

## High-confidence removals (behavior-neutral)

Group R1, dead code. Compile check is enough.
- Q1 (`ggml-backend-meta.cpp:162-163`), Q8 (`:1743-1746`), Q17-Q19 (`common.cuh`), Q28 (`mmq-config-gfx906.cuh` and its `mmq.cuh` branches: dead because GCN is matched first; verified)
- Q29, Q30 (`q8_repack`), Q38-Q40 (allreduce), Q45 (`top-k.cu`), Q51-Q54 (spec/server)
- Q61 (`llama-context.cpp` p_end accum wait/ready), Q70 (`ggml_mul_mat_aux` duplicates `llama_mul_mat_hadamard`; verified identical)
- Q74, Q75, Q83, Q84; and the Engram additions that the base does not have, Q71-Q73. Engram itself exists upstream, so only remove our additions.

Est. ~350-450 LOC.

Group R2, default-off debug/trace. Defaults unchanged.
- Q2 chunk trace, Q5 TG_LIMIT, Q10 sched timing (~190), Q11 rebind trace, Q12 realloc trace
- Q23 repack MoE stats + OP_SYNC, Q46 gate debug, Q47 getrows check, Q52 dead timers
- Q78 debug prints, including an unconditional `fprintf("[shexp-split]")`
- Q3 meta profiler (~110) and Q4 dbg_part/run/xfer (~60) only if you do not use them

Est. ~450-700 LOC.

## Conditional removals (need a test or a decision)

| Item | Validation needed |
| --- | --- |
| Q24 `GGML_CUDA_TP_OVERLAP` (~100, README already says keep unset; latch bug B9) | build + short TP run |
| Q31-Q33 `GGML_RP_NC_FUSION`, `NC_RPL`, `NC_23`, `GENERIC_MMV` (archived "no gain") | build + repack reference cases |
| Q48, Q58, Q59, Q63-Q65 experiment env knobs | build; Q63 restores an unconditional drain (safer) |
| Q21 dead SCALE/UNARY/SCALE/DSV4 fuse site | graph trace on DSV4 or delete together with DSV4 |
| Q34 three J-tile selectors -> one (also B7) | MMQ op tests + PP A/B |
| Pipeline-stage removal (~2.0k) | build all targets; TP correctness (CLI output identical); matched PP/TG A/B on 27B and Flash Next |

## Essential fork features (keep, and why)

- Q8 repack + gfx906 MMQ row-first traversal + IQ2_S two-row MMVQ: the measured PP/TG gains on gfx906. Upstream has no repack path.
- TP core: device-counter AR, duplex copies, whole-token graph replay, ADD+RMS+Q8_1 fusion. Oct 12 follow-up: PP +18%, TG +19-22% vs the earlier base.
- GDN chunked kernel with rollback snapshots: upstream has only a TODO, and rollback correctness depends on it.
- Speculative runtime, per-sequence taps, recurrent snapshot ring: required for MTP/DFlash correctness under TP and concurrency.
- Peer-write AR: kept by decision for a future P2P topology.

## Architectural improvements (less coupling, same behavior)

1. **Gate shared-code behavior changes to HIP/gfx906.** Today these are default-on for NVIDIA and CDNA too:
   - AR slab overlap, custom AR dispatch, eager cuBLAS handle, cuBLAS chunking;
   - `copy_from_host` H2D (B16), graph key change (B15), BF16-less F32 rule (B14);
   - `calc_nwarps` case 5 (B8), `GGML_SCHED_SHARE_VIEW_INPUTS`.

   This cuts semantic divergence for every other backend at no cost to us.
2. **Move glue out of `ggml-cuda.cu`** (~700 lines) into its own files, with one-line hooks left in `ggml-cuda.cu`:
   - repack fusion block -> `q8_repack/`;
   - custom AR + staging -> a comm-ext file;
   - Q8_1 cache -> its own file.
3. **Avoid the 15 one-line backend edits** (`event_query = NULL`) by exposing `event_query` through the reg proc-address table, the way `comm_*` is. That removes 15 files from the diff. Needs a build check with `-Wextra`.
4. **Unify the q8_repack README** with reality: 7 types, not 2.

## Upstream candidates (each removes fork delta if merged)

- argsort >1024 columns on HIP/non-CUB
- recurrent-memory fixes: `apply()` returns the `find_slot` result, rollback clamp, `seq_keep` stale `rs_idx` (B28)
- MTP `pending_h` reset and non-finite guard, DFlash per-seq row mapping and null anchor, DSpark mask fallback (B33, report 07)
- GCN Q8_0 MMQ rows 256 -> 512 threads (discussion 23881)

Each would be a separate, human-authored PR with its own discussion first.

## Prioritized execution plan

| Step | Change | Confidence | Est. LOC removed | Perf risk | Validation |
| --- | --- | --- | ---: | --- | --- |
| 1 | Fix B12 (getrows) and B1 (gguf-split) | high | ~0 | none | op test + build all targets |
| 2 | R1 dead code | high | 350-450 | none | build all targets |
| 3 | R2 debug/trace | high | 450-700 | none | build |
| 4 | Peer AR opt-in gate (B10) | high | ~0 | none on this host | build + startup log |
| 5 | Conditional knob removals | med-high | 250-350 | low | build + short TP run |
| 6 | V4.1 dead plumbing (and DSV4/MiniMax if cut) | high dead / decision | 100-1300 | none for targets | build |
| 7 | HIP-gate the shared behavior changes | med | ~0 (smaller semantic delta) | none on gfx906 | build CUDA + HIP |
| 8 | Pipeline-stage removal (own commit series) | med | ~2000 | medium (TP path) | matched A/B with repeats, output identity |
| 9 | Move ggml-cuda.cu glue into files; event_query via proc address | med | ~15 files out of the diff | low | build all |

Steps 1-4 are the only ones that fit "high confidence, low risk" for Phase 7. They need your approval to start, plus a build, which I have not run.
