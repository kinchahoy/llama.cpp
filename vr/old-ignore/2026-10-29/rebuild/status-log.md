# 2026-10-29 rebuild status log

Note: section times after 19:45 were written as estimates and run ahead of the clock. Result file timestamps are authoritative: D llama-bench 20:01-20:02, D MTP runs 20:29-20:30.

Goal: rebuild the fork as clean upstream (36a0e2ffc) + only valuable features, as patch files in vr/. Three passes, perf after each: A = current master, B = pass 1 (cleanup + fixes), C = pass 2 (pipeline stages removed), D = pass 3 (DSV4/MiniMax/MiMo extensions removed, HIP gating).

- Worktree: /home/raistlin/infer/mx-rebuild, branch rebuild/gfx906 (uncommitted, staged overlay + edits). master untouched.
- Audit and plan: session scratchpad audit/ (INVENTORY.md, CONVERGENCE-PLAN.md); copied to vr/ at the end.
- Bench script: one llama-bench run each for 27B Q8_0 (b/ub 2048, PP2048/TG128, r2) and Flash Next IQ3_XXS (b/ub 1024/512, q8_0 KV, PP2048/TG128, r2).

## 18:01
- Pass 1: 4/8 Haiku agents done (context, core, spec/misc, AR/kernels incl. getrows B12 fix and GGML_CUDA_PEER_AR opt-in). Meta, CUDA, quant, model still running.
- Stage A bench running on build-ilp.

## 18:01
- Stage A (master build-ilp): 27B PP2048 684.56 +- 0.31, TG128 36.82 +- 0.16; Flash Next PP2048 527.79 +- 2.01, TG128 38.99 +- 0.47. Raw: scratchpad bench/A-*.json.

## 18:04
- Pass 1 edits done (8 agents): 18932 -> 17654 added lines vs upstream. Snapshot: scratchpad pass1.patch. One compile fix by hand (top-k.cu struct init after dead-field removal).
- Pass 1 build (B) running in /home/raistlin/infer/mx-rebuild/build-rb.
- Pass 2 (stage removal) running in a second worktree /home/raistlin/infer/mx-rebuild-p2 (detached, seeded from pass1.patch), 3 agents.

## 19:45 (recovered after session crash ~18:09)
- Stage B (pass 1, build-rb): 27B PP2048 682.92 +- 4.13, TG128 36.96 +- 0.02; Flash Next PP2048 525.73 +- 0.44, TG128 38.34 +- 0.81. Perf neutral vs A.
- Old scratchpad (552f07ca...) copied to new session scratchpad 709421cf.../scratchpad.
- Pass 2 tree: no remaining refs to removed stage symbols. Only core/cuda agent report survived; meta.cpp edits finished 18:09. Removed leftover tp-allreduce prefer_small_twoshot by hand. Next: build C in mx-rebuild-p2/build-rb.

## 19:55
- Build C (mx-rebuild-p2/build-rb, same flags as B) OK, all targets incl. tests. Fixed one new unused-var warning in meta ctx ctor. Patch snapshot: scratchpad pass2-full.patch (17654 -> ~15740 added lines).
- Waiting for approval to run stage C bench + output-identity check.

## 20:15
- Stage C (pass 2): 27B PP2048 669.23 +- 0.96, TG128 35.70 +- 0.02; Flash Next PP2048 514.19 +- 5.19, TG128 40.69 +- 0.38. 27B TG -3.4% vs B: regression, investigating before pass 3.
- Pass 3 worktree /home/raistlin/infer/mx-rebuild-p3 (detached 36a0e2ffc + pass2-full.patch); model files reverted so far (deepseek4/41, kv-cache-dsv4, minimax-m2, mimo2, dflash DSV4 hunks).

## 20:50
- C regression root cause: my hand-configured C build lacked CMAKE_HIP_FLAGS=-mllvm -amdgpu-sched-strategy=max-ilp and CXX -O3. B re-run (B2) 684.19 / 36.61, 528.13 / 39.56 matches B. New scratchpad configure_stage.sh copies B flags exactly; C rebuilt and re-benching as C3.
- Pass 3 (mx-rebuild-p3) edits: reverted to upstream deepseek4/41.cpp, kv-cache-dsv4.*, minimax-m2.cpp, mimo2.cpp, softcap.cu/.cuh, models.h DSV4 struct, dflash DSV4 graph hunks (kept SWA pattern fix); llama-model.cpp: upstream is_dsv4 split block restored, removed MLA_TP / design-A replicate / DFlash DSV4 split / MiniMax qk-norm rules / DSV4 narrow_shexp; arch: dead V4.1 KV keys, replicates_attention, MiniMax-M2 back in unsupported list, ATTN_OUT_A back to MUL_MAT; loader RESHAPE probe removed; ggml-cuda HC gate fusion removed. Kept generic: shexp split, tied output, MTP mirror, PLE shard, DFlash dense replicate.
- HIP gating (B8/B14/B15/B16, eager cuBLAS, AR slabs, SHARE_VIEW_INPUTS) deferred to its own step: B8 calc_nwarps is host+device constexpr, gating needs launch-path change.

## 21:20
- C3 (pass 2, correct flags): 27B PP2048 685.60 +- 1.32, TG128 36.58 +- 0.21; Flash Next PP2048 520.22 +- 1.14, TG128 36.93 +- 2.78 (noisy). 27B matches B.
- Build D OK (needed: drop DSV4 comp-context template instantiations in llama-graph.cpp). Patch: scratchpad pass3-full.patch, 14894+/924- over 109 files (C: 15735+/1177-, 118 files).
- Running: D bench, greedy output check B/C/D (scratchpad out/), Flash Next r5 for B/C/D.

## 21:50
- D (pass 3): 27B PP2048 685.34 +- 1.52, TG128 36.67 +- 0.22; Flash Next PP2048 527.63 +- 0.50, TG128 39.09 +- 0.15. Same as B.
- Flash Next r5: B 509.92 +- 17.95 / 39.23 +- 0.13; C 516.43 +- 7.96 / 38.79 +- 0.97; D 520.11 +- 7.34 / 39.26 +- 0.26. All within noise.
- Greedy output (llama-completion, 96 tokens, -sm tensor): byte-identical across B, C, D for both 27B and Flash Next. Outputs in scratchpad out/.

## 22:05
- Passes 1-3 done. Artifacts in vr/rebuild-2026-10-29/ (README, cumulative patches 1-3 applying to 36a0e2ffc, bench JSON, scripts, output samples, audit). Patch 1 refreshed from the built tree (snapshot lacked the top-k fix).
- Worktrees: mx-rebuild (B), mx-rebuild-p2 (C), mx-rebuild-p3 (D, index staged by git add). master untouched.
- Open: HIP gating step, ggml-cuda.cu glue split, decision on landing D.

## 22:45
- U (official 36a0e2ffc, same flags): 27B 448.60 / 32.77, Flash Next 302.99 / 39.66, FN r5 304.33 / 39.76. D vs U: +53% 27B PP, +12% 27B TG, +74% FN PP, FN plain TG no gain. 27B output identical to U; FN diverges after ~30 tokens.
- New kit vr/patches/llama-head-36a0e2ffc/ (5 patches, verify.py 0 mismatches, refs refs/port-snapshots/gfx906-36a0e2ffc-rebuild-20261029-{target,mid,staged-full}). Docs updated (README, how-to-catchup, IMPROVEMENTS, PERFORMANCE incl. N1-N3).

## 23:10
- MTP on D, one llama-cli run each (flake.nix, 512 out): 27B plain 36.2 / MTP2 60.7 / MTP3 58.0 TG; Flash Next plain 37.2 / MTP2 49.6 / MTP3 44.5 TG. MTP text diverges from plain after 1590-2023 bytes. Added to PERFORMANCE.md and rebuild README.
