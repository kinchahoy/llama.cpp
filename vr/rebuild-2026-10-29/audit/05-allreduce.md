# Audit 05: multi-GPU AllReduce

Scope: `ggml/src/ggml-cuda/allreduce.cu` (1220 lines, base 977), `allreduce.cuh` (37), `tp-allreduce.cu` (1037, new), `tp-allreduce.cuh` (212, new). Dispatch glue read in `ggml/src/ggml-cuda/ggml-cuda.cu` and `ggml/src/ggml-backend-meta.cpp` where it decides which path runs.

Method: read-only. Nothing built, run, or benchmarked. The P2P answer comes from code tracing plus existing vr/ notes. It is not confirmed on hardware.

Diff stat vs base 418f905a4 for these four files: allreduce.cu +312/-69 (381 touched lines), allreduce.cuh +8, tp-allreduce.cu/.cuh new (+1037/+212).

---

## 1. Critical question: is tp-allreduce.cu reached in the target config?

Target: HIP build, `GGML_HIP_RCCL=OFF`, `GGML_CUDA_ALLREDUCE` unset, `GGML_CUDA_TP_OVERLAP` unset, 2x gfx906, no direct P2P between cards.

### 1a. Build-time gates

- `GGML_USE_NCCL` is defined only by `GGML_HIP_RCCL=ON` (`ggml/src/ggml-hip/CMakeLists.txt:112-113`). The `GGML_CUDA_NCCL` block in `ggml/src/ggml-cuda/CMakeLists.txt:184-189` sits inside the CUDA-toolkit branch. So in the target, `s_have_nccl == false` and `GGML_USE_NCCL` is undefined.
- `tp-allreduce.cu` is compiled for HIP. It is picked up by the `GLOB "../ggml-cuda/*.cu"` at `ggml/src/ggml-hip/CMakeLists.txt:63`.

### 1b. Runtime trace (ordered by call)

1. Comm init, `ggml-cuda.cu:1773-1786`. `ggml_backend_cuda_comm_init` runs once per stage. For n=2 it calls `tp_custom_ar_init` unless `GGML_ENABLE_CUSTOM_AR=0` (default: on). It then sets `use_custom_ar = custom_ar.initialized` (line 1784).
2. `tp_custom_ar_init`, `tp-allreduce.cu:608-643`. For every ordered pair it calls `cudaDeviceCanAccessPeer`, which maps to `hipDeviceCanAccessPeer` (`vendors/hip.h:52`).
   - Any pair returns 0: it logs `TP custom AR: peer access not available between GPU %d and %d`, then `TP custom AR: disabled because full peer access is not available`, and returns with `initialized == false`. No allocations are made.
   - All pairs return 1: `cudaDeviceEnablePeerAccess` runs. Any error other than AlreadyEnabled hits `CUDA_CHECK`, which aborts. On success the signal buffers are allocated and `initialized = true` (line 705).
3. If `initialized == false`, `use_custom_ar == false`. `ggml_backend_cuda_comm_allreduce_custom_prepare` (`ggml-cuda.cu:1366-1369`) returns false on every call, so the custom path is never reached.
4. `ggml_backend_cuda_comm_allreduce_prepare` (`ggml-cuda.cu:1485-1510`) and `ggml_backend_cuda_comm_allreduce_tensor` (`1814-1823`) fall through to the host-staged pipeline (`ggml_cuda_ar_prepare` / `ggml_cuda_ar_allreduce` in `allreduce.cu`). The pipeline is created unconditionally by `comm_init_internal` (`ggml-cuda.cu:1704`).

**Conclusion, target config:** if P2P is reported unavailable (step 2, first branch), `tp-allreduce.cu` is dead. Peer-write broadcast and two-shot are dead in every reachable case. Repo evidence says the cards have no direct P2P path: `vr/PERFORMANCE.md:83`, `vr/old-ignore/2026-10-12/notes/2026-10-12-tp-profile-findings.md:22`, and `vr/old-ignore/undated/legacy/RESULTS.md:110` ("The GPUs reported no direct peer access"). That evidence is not a runtime check on the current ROCm.

### 1c. The branch that is NOT dead, and why it matters

`hipDeviceCanAccessPeer` may report 1 for a pair even when peer DMA is poor. If it does, and `cudaDeviceEnablePeerAccess` succeeds, then:

- `initialized = true`, and `broadcast_ok = is_gfx9_xgmi || HSA_FORCE_FINE_GRAIN_PCIE` (`tp-allreduce.cu:681-704`). gfx906 is not XGMI, and no repo script sets `HSA_FORCE_FINE_GRAIN_PCIE`. So `broadcast_ok = false`.
- The NCCL gate at `ggml-cuda.cu:1417` only applies when `s_have_nccl`. In the target it is skipped. The size gate at ~1421-1432 is also skipped. So the **one-shot** custom kernel takes **every contiguous F32 tensor, any size**.
- Custom runs before the internal path (`ggml-cuda.cu:1819-1822`). So F32 all-reduces leave the host-staged BF16-wire pipeline, lose the BF16 wire default (`GGML_CUDA_AR_BF16_THRESHOLD=1`), lose the copy-engine/duplex/slab overlap, and switch to F32 wire. The numerics change.
- One-shot in the fused path returns -1 from `allreduce_prepare` (`ggml-cuda.cu:1497-1499`), and `ggml-backend-meta.cpp:5197-5199` treats `nr < 0` as `ok = false`. So whole-token graph capture fails and TG falls back to per-subgraph dispatch. The 2026-10-12 TG 30.2 -> 34.8 gain was measured on the host-staged prepare path (`vr/old-ignore/2026-10-12/notes/2026-10-12-tp-profile-findings.md:135`), so this would be a regression.

So the custom path is the one that matters. If it goes live in the target, the outcome is worse on both TG and numerics. The decisive check is the startup log, which is not run here:

- Dead path (expected): `TP custom AR: peer access not available between GPU ...` and `TP custom AR: disabled because full peer access is not available`. No `TP custom AllReduce: initialized`.
- Live path: `TP custom AllReduce: initialized for 2 GPUs, path = one-shot (PCIe without HSA_FORCE_FINE_GRAIN_PCIE=1 ...)`.

The stale comment at `ggml-cuda.cu:1218-1220` and `tp-allreduce.cuh:12-15` says peer-write falls back to NCCL. That holds only for NCCL builds. In the target there is no NCCL fallback. The one-shot path simply runs.

### 1d. Summary answer

| Kernel | Reachable in target? | Condition |
|---|---|---|
| tp-allreduce one-shot (`k_cross_device_reduce_1stage`) | No, if canAccessPeer=0 (expected) | Needs canAccessPeer=1 on all pairs. Otherwise yes, with the F32-wire shadowing and graph-capture loss above. |
| tp-allreduce broadcast (`k_broadcast_reduce`) | No | Needs `broadcast_ok` (XGMI or `HSA_FORCE_FINE_GRAIN_PCIE=1`) and canAccessPeer=1. No repo script sets FGP. The old 2026-08 benchmark (`RESULTS.md:139-140`) did. |
| tp-allreduce two-shot (`k_twoshot_f32`) | No | Needs `broadcast_ok`, and `n % (4*nranks) == 0`. For nranks=2 it also needs `ne >= 262144` (or `GGML_TP_AR_TWOSHOT=1`). |
| Host-staged pipeline (`allreduce.cu`) | Yes, this is the target path | Default when custom is not initialized. |

Recommendation: do not delete the peer-write kernels without user approval. That is a large change, and AGENTS.md asks for confirmation first. The safe, small actions are in the quick-wins list.

---

## 2. HIP branch and head-merge checks (AGENTS.md / vr/IMPROVEMENTS.md)

- `allreduce.cu:3`: `#if !defined(GGML_USE_MUSA)` opens the first branch. HIP does not define `GGML_USE_MUSA` (not defined in `ggml-hip/CMakeLists.txt` or `vendors/hip.h`). HIP therefore enters this branch. **Confirmed.**
- HIP-specific device code lives inside that branch: `s_sleep` at `allreduce.cu:186-188`.
- The `#else // defined(GGML_USE_MUSA)` stub is at `allreduce.cu:1204-1217`. It is the only other branch. It also declares `ggml_cuda_ar_prepare` and `ggml_cuda_ar_launch_rank`, so the header signatures stay consistent.
- `GGML_CUDA_TP_OVERLAP` is not read anywhere in `allreduce.cu`. It is read in `ggml-cuda.cu:1708` (HIP only; sets `tp_overlap.enabled`, which affects the mmq path), and in `ggml-cuda.cu:3989` (disables CUDA graphs for one exact Q8 shape). `ggml_cuda_ar_pipeline_init` is called at `ggml-cuda.cu:1704` regardless of that variable. **The normal config does not need `GGML_CUDA_TP_OVERLAP=1`.** Confirmed.
- The "unreachable HIP overlap-only `#elif`" that vr/IMPROVEMENTS.md says was removed is not present in base either (`git show 418f905a4:...allreduce.cu` has no `#elif defined`). Nothing remains to restore.

---

## 3. Every getenv() in the slice, with default

Helper `ggml_cuda_ar_env_u64` (`allreduce.cu:375-384`) is used for the first four.

| Variable | Read at | Default | Effect | Base has it? |
|---|---|---|---|---|
| `GGML_CUDA_AR_COPY_THRESHOLD` | allreduce.cu:450 | 1048576 (1 MiB) | Size at which copy-engine path replaces chunked kernel | Yes |
| `GGML_CUDA_AR_COPY_CHUNK_BYTES` | allreduce.cu:453 | 0 = heuristic clamp(nbytes/4, 512K, 2M); non-zero floored at 256K | Copy-engine chunk size | Yes |
| `GGML_CUDA_AR_BF16_THRESHOLD` | allreduce.cu:462 | 1 (F32 always BF16 wire) | BF16 wire for F32 >= threshold bytes. Precision-changing. | Yes |
| `GGML_CUDA_AR_DUPLEX` | allreduce.cu:481 | 1 (separate H2D stream) | Separate H2D stream; 0 shares the stream | **No, ours** |
| `HSA_FORCE_FINE_GRAIN_PCIE` | tp-allreduce.cu:692 | unset (off) | Sets `broadcast_ok` on gfx906 PCIe | No, ours |
| `GGML_TP_AR_KTIMING` | tp-allreduce.cu:719 (destroy), :944 (prepare) | unset (off) | Prints device timing; prepare uploads flag. Destroy prints for ANY set value, including `0`. | No, ours |
| `GGML_TP_AR_TWOSHOT` | tp-allreduce.cu:794 | -1 = auto | Force two-shot on/off | No, ours |
| `GGML_TP_AR_TWOSHOT_MIN_NE` | tp-allreduce.cu:815 | -1 = auto (1 for nranks 5/8/10, or 4 with `prefer_small_twoshot`; else 8192 for nranks>=4, 262144 for 2) | Two-shot crossover in elements | No, ours |
| `GGML_TP_AR_BLOCKS` | tp-allreduce.cu:932 | kDefaultBlocks: 16 HIP / 36 CUDA | Kernel grid cap | No, ours |
| `GGML_TP_AR_NO_GATE` | ggml-cuda.cu:1410 | off (gate on) | Disables size gate. **No effect in non-NCCL builds** (gate only runs when `s_have_nccl`). | No, ours |
| `GGML_TP_AR_ONESHOT` | ggml-cuda.cu:1414 | off | Forces custom path past the NCCL fallback. **No effect in non-NCCL builds.** | No, ours |
| `GGML_ENABLE_CUSTOM_AR` | ggml-cuda.cu:1779 | on (only `0` disables) | Creates the custom AR context | No, ours |
| `GGML_CUDA_ALLREDUCE` | ggml-cuda.cu:1787 | Linux: `nccl`, which falls back to `internal` without NCCL. Non-Linux: `internal`. | Selects comm backend | Yes |
| `GGML_CUDA_TP_OVERLAP` | ggml-cuda.cu:1708, :3989 | unset | HIP mmq overlap state and one-shape graph exclusion. Not used by allreduce.cu. | No, ours |
| `GGML_CUDA_AR_SLABS` | ggml-cuda.cu:2953 | 4 (on). 0 or 1 disables. Clamped to MAX_SLABS. | Prefill token-slab overlap of F32 mul_mat feeding an AR | No, ours |

---

## 4. Feature table

LOC are approximate, from hunk counts and section line ranges. Base ends at 977 lines for allreduce.cu. Upstream equivalents are from the base commit.

| Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|
| Device-side token counter in chunked kernel; no host slot acquire on chunked path; `launch_chunk` helper | allreduce.cu:114-224 (kernel), 929-961, 1138-1155 | +60 / -35 | B (graph replay; measured TG 30.2 -> 34.8, notes 2026-10-12) | Base uses host `call_count` token (base:905-933) | Always on, HIP and CUDA | Keep | High |
| Separate H2D stream (`h2d_streams`, `GGML_CUDA_AR_DUPLEX`) | allreduce.cu:337-339, 480-487, 744-747, 899-907, 644-647 | +50 / -7 | B (measured PP 599 -> 647, notes 2026-10-12) | Base has one stream | Env default 1 | Keep | High |
| Prefill slab overlap: `ggml_cuda_ar_allreduce_slabs`, `dev_src`/`src_read_done`, dispatch block | allreduce.cu:827-927, 1006-1024, 333-335, 580-586 | +175 / 0 | B (prefill overlap) | None | Default ON (`GGML_CUDA_AR_SLABS=4`); needs copy-engine + BF16 wire + F32 mul_mat >= 512 rows. Producer side in ggml-cuda.cu:2951-3000+, state at common.cuh:1500. | Keep, but **confirm default** (see note A) | Medium |
| Prepare/launch split (`ggml_cuda_ar_prepare`, `ggml_cuda_ar_launch_rank`, `plan`, MUSA stubs) | allreduce.cu:364-372, 1161-1202, 1214-1217 | +57 / 0 | B/C (whole-token graph capture) | None | Always on | Keep | High |
| Copy-engine cross-stream handshakes (`host_large_read_done`, `dev_tmp_kernel_done`, `*_valid` flags) | allreduce.cu:342-354, 705-788 | +25 / -8 | A/B (needed for single-buffered dev_tmp) | Partially in base | Always on | Keep | High |
| Custom AR dispatch glue (CustomARPlan, `comm_*` custom hooks, env checks) | ggml-cuda.cu:1228-1252, 1366-1520, 1779-1784 | ~+200 / -10 | C/E (live only with P2P) | None (`git grep custom_ar` at base: no match) | GGML_ENABLE_CUSTOM_AR; see section 1 | Keep glue; trim dead knobs (quick win 6) | High |
| tp-allreduce header and types | tp-allreduce.cuh (all) | +212 | C | None | Compiled for CUDA/HIP; live only with P2P | Keep unless peer-write is abandoned (ask user) | High |
| Barrier primitives (MUSA, CUDA, HIP variants) | tp-allreduce.cu:15-174 | +160 | C | None | Live only with P2P | Keep / ask user | High |
| One-shot kernel `k_cross_device_reduce_1stage` | tp-allreduce.cu:176-258 | +83 | C/F (**reachable only if canAccessPeer=1**) | None | Custom path, one-shot (`!broadcast_ok`) | Keep; see section 1c | High |
| Broadcast kernel `k_broadcast_reduce` + device KTIMING | tp-allreduce.cu:261-425 | +165 | C (dead in target) | None | Needs `broadcast_ok` (FGP=1 or XGMI). Target: dead. | Keep / ask user. KTIMING part is E (quick win 5). | High |
| Two-shot kernel `k_twoshot_f32` | tp-allreduce.cu:428-602 | +175 | C (dead in target) | None | Needs `broadcast_ok`, nranks 2 needs ne >= 262144 | Keep / ask user | High |
| Init, peer check, `broadcast_ok` | tp-allreduce.cu:608-716 | +110 | A/C (gates everything) | None | Peer check (616-643), FGP env (692) | Keep | High |
| Destroy and KTIMING readout | tp-allreduce.cu:718-769 | +52 | E (debug) | None | `GGML_TP_AR_KTIMING` | Trim KTIMING (quick win 5) | Medium |
| Prepare (staging alloc, path select, RankData upload, grid) | tp-allreduce.cu:771-973 | +203 | C | None | Staging alloc on any reachable path | Keep, minus unused tuning in target | Medium |
| Launch per rank | tp-allreduce.cu:975-1019 | +45 | C | None | Always with prepare | Keep | High |
| `tp_custom_ar_allreduce` (non-split wrapper) | tp-allreduce.cu:1021-1035; tp-allreduce.cuh:155-165 | +26 | **E: no callers** | None | None | **Remove** (quick win 1) | High |
| Unused include `<type_traits>` in allreduce.cu | allreduce.cu:12 | +1 | E | n/a | none | **Remove** (quick win 2) | High |
| Unused `slot_info.token`; `acquire_slot` returns it unused | allreduce.cu:386-404 | ~+6 | E | n/a | none | Simplify (quick win 3) | Medium |
| Removed MUSA stub explanation comment | allreduce.cu:1204 (base had 4-line comment at 960-963) | -4 | F (doc loss) | Base had it | none | Restore a 1-line comment if wanted (low priority) | Medium |

---

## 5. Per-feature notes

**A. Slab overlap default.** `ggml_cuda_ar_slab_count()` defaults to 4 (`ggml-cuda.cu:2953`), so slab overlap is on by default. The only archived measurement is `vr/old-ignore/undated/legacy/RESULTS.md` (2026-08-12). It shows F32 slab overlap rejected (C2, PP 276 vs 283), and BF16-wire slab overlap recorded as "Keep as opt-in" (C3, PP +12.6%, TG 19.6 vs 19.8). `vr/README.md:55` and `vr/IMPROVEMENTS.md` say most local optimizations are default-on. The code and the recorded decision disagree. The owner should decide and record which default is intended. The code does not change here.

Note on cost: `dev_src` (32 MiB per GPU, `allreduce.cu:580-585`) is allocated unconditionally at init, even if slabs are never used. Base allocates `host_large` and `dev_tmp` at the same size. That is roughly +32 MiB device memory per GPU versus base.

**B. Device counter token (`allreduce.cu:140-148`).** Each block takes `counter[bid] + 1` as its token and writes it back after its spin (`counter[bid] = token` at `allreduce.cu:195`). Both ranks issue the same chunked sequence, so the counters agree. Tokens are device-only, which is what makes replay under a captured graph valid. `slot = token % 2`. The copy-engine path still uses `call_count` (`acquire_slot`, `allreduce.cu:391-404`) for its own slot, and it uses events rather than arrival ints, so the two paths do not share state. Not a defect, but the two slot-selection schemes are easy to confuse.

**C. Prepare/launch.** `ggml_cuda_ar_prepare` returns -1 for copy-engine and for `nbytes > buf_bytes` (`allreduce.cu:1174-1179`). Those calls fall back to the non-fused path. That is the intended behavior.

**D. One-shot prepare side effects.** `tp_custom_ar_prepare` issues the staging `cudaMemcpyAsync` and event records (`tp-allreduce.cu:886-904`) before the fused caller returns -1 (`ggml-cuda.cu:1497-1499`). That is why the comment at `ggml-cuda.cu:1486-1491` exists. It is only safe if the captured work is discarded when capture fails (`ggml-backend-meta.cpp:5197-5199`). I did not verify that discard path. Low-medium risk; only reachable if the custom path is live.

**E. Dead knobs in target.** `GGML_TP_AR_NO_GATE` and `GGML_TP_AR_ONESHOT` are read inside a block that only has effect with `s_have_nccl` (`ggml-cuda.cu:1403-1433`). Setting them does nothing in a RCCL-off HIP build. The same is true of `GGML_TP_AR_TWOSHOT*` for nranks=2 unless broadcast is live.

**F. Stale comments.** `ggml-cuda.cu:1218-1220` ("Without a peer-write coherent fabric it stays on the NCCL path by itself") and `tp-allreduce.cuh:10-15` (gate "routes large messages to NCCL") are wrong for non-NCCL builds.

**G. KTIMING.** Device-side instrumentation exists only in `k_broadcast_reduce` (`tp-allreduce.cu:283-300`, `412-425`), with globals at `:284-285`. It is runtime-gated and defaults off, but the device code and two host readers add noise. Keep it only if broadcast profiling is still wanted.

**H. Peer-check failure path.** If `cudaDeviceEnablePeerAccess` fails with an error other than AlreadyEnabled, `tp-allreduce.cu:626-628` calls `CUDA_CHECK`, which aborts. That can happen when `canAccessPeer` says 1 but the enable call fails. Not a target-config issue if canAccessPeer=0, but it is a hard failure mode worth knowing about.

**I. Silent numerics switch.** Section 1c: if the custom path goes live in the target, all F32 all-reduces change from BF16 wire to F32 wire. That changes the output, and the 2026-08 notes already flagged BF16 vs F32 precision as a real tradeoff (`RESULTS.md`, "BF16 wire changes reduction precision"). The routing order (`ggml-cuda.cu:1819-1822`) makes this change silent.

---

## 6. Quick wins (high confidence, low risk)

1. **Remove unused `tp_custom_ar_allreduce`.** `tp-allreduce.cu:1021-1035` and its declaration `tp-allreduce.cuh:155-165`. Zero callers (`git grep HEAD`: only the definition, the declaration, and comments). Update the references in the comments at `tp-allreduce.cuh:10` and `:50`. About -26 LOC. Confidence high.
2. **Remove unused `#include <type_traits>`** at `allreduce.cu:12`. No `std::is_*` or `enable_if` in the file. -1 LOC. Confidence high.
3. **Drop the unused `token` from `ggml_cuda_ar_slot_info`** and make `ggml_cuda_ar_acquire_slot` return `int`. Call sites: `allreduce.cu:689`, `:867`. About -6 LOC. Confidence medium-high. Compile-only check needed.
4. **Make the GGML_TP_AR_NO_GATE / GGML_TP_AR_ONESHOT reads NCCL-only.** Wrap `ggml-cuda.cu:1407-1416` in `#ifdef GGML_USE_NCCL`. Behavior is unchanged in both builds, and the dead reads disappear from the RCCL-off build. Confidence high.
5. **Trim KTIMING** (`tp-allreduce.cu:283-300`, `:412-425`, `:284-285`, `:719-735`, `:943-956`) if broadcast profiling is no longer used. Medium confidence; ask first, because the vr/ notes use `[ar-ktime]` output.
6. **Lazily allocate `dev_src`** (`allreduce.cu:580-585`, and the matching free at `:619-622`) only when `copy_bytes` is used with slabs (`GGML_CUDA_AR_SLABS >= 2`). Saves 32 MiB per GPU when slabs are off. Behavior-preserving. Medium confidence, needs a run to validate.
7. **Fix stale comments:** `ggml-cuda.cu:1218-1220`, `tp-allreduce.cuh:10-15`. Cosmetic.

Not a quick win, needs user approval (AGENTS.md: ask before large changes): removing the peer-write kernels and their prepare logic (`tp-allreduce.cu:176-602`, ~430 LOC). They are dead in the stated config, but they are the only P2P path and the 2026-08 benchmark used them.

---

## 7. Limitations

- No build, run, or benchmark, per BRIEF.
- P2P availability on this host was not measured. The dead-path conclusion rests on code plus the vr/ notes. The startup log lines in section 1c decide it.
- The kernel bodies of `k_cross_device_reduce_1stage`, `k_broadcast_reduce`, `k_twoshot_f32`, and the barrier primitives were read for structure and dispatch, not audited line by line. Their correctness is outside this brief.
- The slab producer (`ggml-cuda.cu:2951-3000+`) and the meta-backend capture path were traced only as far as needed for gating.
