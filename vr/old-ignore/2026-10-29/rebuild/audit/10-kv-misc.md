# Audit 10: KV cache, memory, and misc files

Base: 418f905a4. HEAD: e04bc2e38. All line numbers refer to HEAD unless marked base. Static review only: nothing was built or run.

## Part 1: KV cache and memory

### 1. Blocker

- **BLOCKER (compile error): `tools/gguf-split/gguf-split.cpp:565-566`.** A `fprintf` format string is split across two lines by a raw newline inside the string literal (`"... may be truncated` then a newline then `", __func__, ..."`). A C++ string literal cannot span lines, so this does not compile. The base had `\n` escaped on one line. Fix: restore the single-line `\n` form. High confidence (verified with `cat -A`).

### 2. Feature table (KV, memory, and misc)

| Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|
| Shared-layer KV replicas (per-device copy of a reused layer's compressed KV, written in-graph) | llama-kv-cache.cpp (ctor ~270-320, update ~895, shift ~2147, clear ~1969), llama-kv-cache.h (~336-352) | +150 / -5 | A/C | Base DSv4 "mirror" layers (llama-kv-cache-dsv4.cpp base 1322-1356, state refill base 435-449). Ours replaces mirror tensors with generic replicas. | Runtime: offload && `LLAMA_KV_REPLICA != 0` (default on). Reached only when a `reuse` callback is set. Callers: DSv4 (dsv4.cpp ~1436-1460) and Mistral-Large-4 (llama-model.cpp:3453). Not HIP-only. | Keep for V4.1 tensor split. Document that it also affects Mistral-Large-4. | med |
| Replica-aware graph helpers (`get_k_view`, `get_v_view`, `cpy_k_into`, `cpy_v_into`, `*_replica` accessors) | llama-kv-cache.cpp (~1330-1500), llama-kv-cache.h, llama-kv-cache-dsv4.cpp/.h (`get_kv`, `get_k_replica`, `cpy_k_replica`) | +100 / -20 | A (dependency of above) | None | Always compiled, no behavior change when no replicas | Keep (refactor needed by replicas) | high |
| KV ctx tensor budget 2u to 4u | llama-kv-cache.cpp:116 | +1 / -1 | A (dependency) | None | Always | Keep. Cheap, needed for replica tensors. | high |
| DSv4 comp-plan block padding (`overlap` replaces `pad_blocks`, fixed dummy-write count per stream) | llama-kv-cache-dsv4.cpp:632-690 | +55 / -30 | B (graph reuse) | None | DSv4 only, always on | Keep, but see dead parameter below. Needs a graph-reuse perf check before any upstream port. | med |
| DSv4 prefill snapshot gate (`LLAMA_DSV4_NO_PREFILL_SNAPSHOT`, default ON) | llama-kv-cache-dsv4.cpp:722-750 (fill), 966-980 (reserve) | +40 / -10 | B (perf) with correctness risk | None | DSv4 with rollback. Env default on (set 0 to revert). | Keep only with a rollback-correctness test. See note 4. | med |
| DSv4 V4.1 lightning-indexer cache now populated | llama-kv-cache-dsv4.cpp:1426-1436, 1484-1488 | +4 / -6 | F, likely E (possible VRAM regression) | None | V4.1 (`deepseek41`) only | Verify. Probably revert `kv_lid` to empty for V4.1. See note 5. | med |
| DSv4 `get_can_checkpoint() == false` | llama-kv-cache-dsv4.cpp:1672-1676, .h:124 | +6 | C/F | Hook absent from base (`llama-memory.h:141` in HEAD; used at tools/server/server-context.cpp:75) | DSv4 always | Keep. The comment gives the reason. Verify server prompt-cache behavior. | med |
| Recurrent snapshot ring (plane head, valid depth, row-scattered snapshot writes) | llama-memory-recurrent.cpp (ring_enabled 36-50, advance_planes 559, plan_snapshot_writes ~580, s_copy ~1560-1585, state_write plane_for ~970-990), .h (~76-105, 213-215) | +230 / -40 | C/B | Base plane-indexed rollback (`rs_idx`, `set_rs_idx`). Ring is a superset. | Gated to Qwen3Next, Qwen35, Qwen35MoE, Qwen4Exp (line 46) with `n_rs_seq > 0 && n_seq_max == 1`. Not HIP-only. | Keep. vr says "every delta-net model shares it", but code gates 4 archs. Fix the docs or the gate. | med |
| Ring failure invalidation (`pass_touched`, `pass_staged`, `ring_valid`, `finish_rs_pass` ~534) and `finish_compute` plumbing (hybrid, hybrid-iswa, `llama-context.cpp:1985/1993`) | llama-memory-recurrent.cpp:534-557, 1502-1507; llama-memory-hybrid(.h/.cpp), hybrid-iswa(.h/.cpp) (+5 each) | +90 / -5 | A/C (safety) | None in base (`finish_compute` absent at 418f905a4) | Ring mode, plus hybrid forwarding (always) | Keep. The hook is new infra in `src/llama-memory.h` (outside this slice). | med |
| `apply()` now returns `find_slot()` result (recurrent) | llama-memory-recurrent.cpp:1499 (base ignored result) | +1 / -3 | Generic bugfix candidate | Base ignores it | ALL recurrent and hybrid models, not only ring | Upstream candidate. Failed slot search now fails decode, not silent. Verify no path where `find_slot` fails benignly. | med |
| Rollback guard: `rs_target` duplicate check, `max_rollback` clamp, warn + refuse | llama-memory-recurrent.cpp:210-266 | +40 / -15 | Generic correctness candidate | Base silently returns false | ALL recurrent models | Upstream candidate. Prevents double-arming a rollback. | med |
| `seq_keep` resets other sequences' ring and rollback index | llama-memory-recurrent.cpp:351-360 | +6 | Generic correctness candidate | Base keeps stale `rs_idx` of dropped seqs | ALL recurrent models | Upstream candidate. Base can apply a stale rollback to a later sequence reusing that id. Medium confidence from code reading. | med |
| State read: `reset_ring(seq_id)` replaces `set_rs_idx(seq_id, 0)` | llama-memory-recurrent.cpp:1043-1080 | +3 / -4 | B/C | Base sets rs_idx only if `n_rs_seq != 0` | ALL | Keep. Trailing blank line before `}` at ~1074 is cosmetic. | high |
| Ring snapshot scatter in graph (`plan_snapshot_writes`, `s_copy` plane mapping) | llama-memory-recurrent.cpp:563-600, 1555-1585 | (counted above) | C | Base `s_copy` uses `rs_idx * size` | Ring mode | Keep with the ring. Must match `llama-graph.cpp:422-528`, which is out of this slice. | med |
| DSv4 `dsv4_state_read_k_cache` uses `sync_replicas` instead of mirror refill | llama-kv-cache-dsv4.cpp:381-433 | +5 / -20 | A (dependency) | Base mirror refill | Replica mode | Keep. The removed `n_rows_ref == 0` early return is harmless (`sync` checks `r1 <= r0`). | high |
| Dead DSv4 lambdas `is_comp_owner`, `is_state_owner` | llama-kv-cache-dsv4.cpp:1417-1426 | 0 / -10 | E (dead code) | Replaced by new filter code | n/a | Delete. Each has 1 occurrence at HEAD (the definition). `comp_read` must stay: `get_comp_read_layer` uses it (line 1927). | high |
| Dead DSv4 `pad_blocks` parameter | llama-kv-cache-dsv4.cpp:453, 842, 854, 2301/2304/2307 | 0 / -8 | E (dead param) | n/a | n/a | Delete the parameter and its three call-site args. No body reads it at HEAD. | high |
| Dead `unified_compressed` constant (always false) | llama-kv-cache-dsv4.cpp:1450 | 0 / 0 | E (pre-existing) | Base | n/a | Leave, not fork-specific. | high |
| `dsv4_build_comp_plan` reserve/fill gate mismatch (`n_seq_tokens >= 64` vs per-seq `n_tok >= 64` and `n_out <= 1`) | llama-kv-cache-dsv4.cpp:973 vs 735-750 | - | F (risk) | - | DSv4 with rollback | Verify. See note 4. | low |

### 3. Per-feature notes (KV and memory)

1. **Replicas.** A V4.1 layer that reuses another layer's compressed KV on a different GPU gets a local copy. The copy is written inside the graph (`cpy_k_replica`), so it stays bit-identical with the owner, and it is copied out of graph on state read and stream copy (`sync_replicas`). This replaces the base DSv4 mirror tensors (base llama-kv-cache-dsv4.cpp:1322-1356). `LLAMA_KV_REPLICA=0` falls back to reading the owner's device (no replica tensors). The code is in generic `llama_kv_cache` and only runs when a `reuse` callback exists. Mistral-Large-4 uses `reuse` too (llama-model.cpp:3453), so it is not DSv4-only. No vr doc records a measurement for `LLAMA_KV_REPLICA`. Only the kit patch `vr/patches/llama-head-418f905a4/patches/0003-llama-model-runtime.patch` (`has_replicas`) mentions it.

2. **Recurrent ring.** Snapshots go into a ring of physical planes per sequence. A pending rollback is an index into the ring (`rs_idx`), and `valid_depth` bounds how far back a sequence can rewind. `ring_enabled` is set only for four Qwen archs and `n_seq_max == 1` (line 46). It is off for Kimi-K3, DSv4, GLM5-Next, Nemotron-H, and LFM2 even though they support rollback (`llm_arch_supports_rs_rollback`, llama-arch.cpp:1340). vr/old-ignore/2026-10-29/prior-guides/FEATURES.md:104-120 says "Backend-generic ... every delta-net model shares it." That contradicts the code gate. FEATURES.md:104-120 also says rollback is "clamped off above one sequence and below a minimum micro-batch with a warning." The ring code has no micro-batch clamp. The clamp is in llama-context.cpp:362-366, which is outside this slice.

3. **Failure invalidation.** When a graph fails, `finish_rs_pass(false)` marks every touched sequence invalid and sets `valid_depth = 0`, so a half-written plane is never exposed. Decode then fails until the sequence is cleared (find_slot line 679). Good design. The `apply()` return change (line 1499) affects all recurrent models.

4. **DSv4 prefill snapshot gate.** This skips snapshot and restore planning for ubatches where every sequence has at least 64 tokens, at most one output, and no pending rollback. The stated basis is that rollback only targets the current verify window. The "2.4x" figure appears only in a code comment (and in the old-ignore kit patch). No vr doc records a result for `LLAMA_DSV4_NO_PREFILL_SNAPSHOT`. The reserve path (line 973) uses `max(1, n_seq_tokens) >= 64` with no output check. That can under-reserve if a real ubatch has at least 64 tokens and logits for more than one token, since full planning then produces more entries than were reserved. Verify with a real rollback test before upstreaming.

5. **V4.1 indexer cache (possible regression).** `kv_lid` previously used `filter_lid = !is_v41 && filter_csa`, which is empty for V4.1 (base). The new code passes `filter_csa` unchanged (line 1484-1488), and the V4.1 path now selects ratio-2 source layers. `deepseek41.cpp` computes the indexer from the CSA rows ([kv | index K], per the comment at line 1362) and does not call `get_lid()`. So for V4.1 the lightning-indexer cache is probably allocated but never read or written by the graph. That costs VRAM. Confirm by checking whether `kv_lid` has writes for V4.1, then restore the V4.1 exclusion.

6. **Checkpoint gate.** `get_can_checkpoint() == false` for DSv4 is a behavior change. Base has no such hook. The server uses it at server-context.cpp:75. The comment says the compressed attention state cannot be restored from a partial checkpoint. That is plausible, but it disables prompt checkpoints for DSv4. Keep, and say so in the report.

### 4. Upstream candidates (generic correctness, not gfx906-specific)

- `llama-memory-recurrent.cpp:1499`: `apply()` returns `find_slot()` result (base ignored it).
- `llama-memory-recurrent.cpp:210-266`: rollback duplicate detection, `max_rollback` clamp, warning when refusing.
- `llama-memory-recurrent.cpp:351-360`: `seq_keep` resets dropped sequences' rollback index.
- `llama-memory-recurrent.cpp:1043-1080`: `state_read` resets rollback index unconditionally.

These are small and generic. The ring itself and the replica code are not upstream candidates.

## Part 2: Misc files

### 5. Feature table (misc)

| Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|
| gguf-split chunked copy and merge (64 MB chunks) | tools/gguf-split/gguf-split.cpp:337-375, 549-580 | +40 / -25 | B/C (robustness) | None | Always on | Keep the idea. **Fix the compile error at 565-566** before anything else. Restore the truncation check in `copy_file_to_file` (line 369 has no `if (!f_in)` check now). Delete the orphan comment at line 340. | high (for the compile error) |
| llama-bench `-tps` and `-nr`, `tensor_parallel_size` in output | tools/llama-bench/llama-bench.cpp (+55) | +55 / -2 | C (fork option) | Base llama-bench has no `-tps`. `-nr` exists in common/arg.cpp:2435. | Only when user passes `-tps`. Default `{0}`, markdown shows it only when non-default. | Keep for fork benchmarks. Upstream would need its own review. | high |
| test-backend-ops additions (GDN chunked snapshots, IQ2_S/IQ4_NL `mul_mat_id` with broadcast, transposed concat, top-k k=32/40/64) | tests/test-backend-ops.cpp (+51) | +51 / -2 | C (tests) | None | Test only | Keep. Good coverage for the GDN snapshot fix in vr. | high |
| Root CMake `LLAMA_BUILD_VR_EXPERIMENTS` option | CMakeLists.txt:139, 250-252; vr/CMakeLists.txt | +5 | C (fork-only) | None | Option default OFF | Keep in fork. It is only the option plus `add_subdirectory(vr)`, which builds `vr-test-gfx906-tp` and `vr-test-flashnext-moe`. Upstream PR must exclude it. | high |
| CANN CI stub job `cann-disabled` | .github/workflows/build-cann.yml:41-46 | +6 | E (noise) / F | Base has every job commented out | `if: false` | See note 7. Remove or drop from the upstream diff. | med |
| README intro line | README.md:1 | +2 | C (fork docs) | None | n/a | Keep in fork only. Do not upstream. | high |
| AGENTS.md "Head merge check" section | AGENTS.md:252-256 | +5 | C (fork docs) | None | n/a | Keep. Its claims match vr/IMPROVEMENTS.md:16 and `ggml-cuda.cu:1708`. | high |
| CLAUDE.md now `@AGENTS.md` | CLAUDE.md:1 | 1 / 1 | F (neutral) | Base had text | n/a | Keep. | high |

### 6. Answers to the specific questions

- **Why the CANN workflow changed.** The base workflow has every job commented out (`jobs:` contains only comments, lines 40-end). A workflow with no active job is an invalid workflow file, and GitHub reports it on every push. The fork added a disabled stub job (`if: ${{ false }}`) so the file parses. This is an inference from the file contents. No vr doc or commit message explains it. The cleanest fork fix is to delete the workflow file, or leave the base file alone. Do not include the stub in an upstream PR.

- **Is the root CMake change just the vr experiments option?** Yes. It is the `LLAMA_BUILD_VR_EXPERIMENTS` option (default OFF) plus `add_subdirectory(vr)` gated on it. `vr/CMakeLists.txt` builds `vr-test-gfx906-tp` and `vr-test-flashnext-moe` and links `ggml`. No other root CMake changes.

- **Are the gguf-split changes needed?** The chunking is a memory-robustness fix. The reason given in the code comment (line 563-564) is a DSv4.1 Engram table of 52 GB each that OOM-killed the tool at 54 GB resident. No vr doc records this OOM, so the claim is unverified. Normal-sized models are unaffected functionally. The change is needed only for very large single tensors. Also, the split path's `read_buf.resize(n_bytes)` removal is correct. The file does not compile as it stands (see blocker).

### 7. Quick wins (safe, high confidence)

1. **Fix compile error**: `tools/gguf-split/gguf-split.cpp:565-566`. Join the format string into one line with `\n`.
2. **Delete dead lambdas**: `src/llama-kv-cache-dsv4.cpp:1417-1426` (`is_comp_owner`, `is_state_owner`). Keep `comp_read`.
3. **Delete dead parameter `pad_blocks`**: `src/llama-kv-cache-dsv4.cpp:453, 842, 854`, and the args at 2301, 2304, 2307.
4. **Delete orphan comment** `tools/gguf-split/gguf-split.cpp:340`.
5. **Restore truncation check** in `copy_file_to_file` (`tools/gguf-split/gguf-split.cpp:367-370`), matching the merge path.
6. **Fix the CANN stub**: delete `.github/workflows/build-cann.yml` (fork-only), or drop the stub job from the upstream diff.
7. **Shorten ring comments** (`llama-memory-recurrent.cpp:38-45` and the blank line at ~1074) to the repo style in AGENTS.md.

### 8. Priority follow-ups (need a test, not a cleanup)

- V4.1 `kv_lid` allocation (note 5). Possible VRAM regression.
- Prefill snapshot gate and reserve mismatch (note 4). Possible rollback bug.
- Docs say "every delta-net model shares the ring" (FEATURES.md:104-120). Code gates four archs.
