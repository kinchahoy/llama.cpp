# Audit brief (read fully before starting)

Repo: /home/raistlin/infer/mx-llama.cpp. Fork of llama.cpp targeting AMD gfx906 (MI50/MI60), dual-GPU tensor parallel (-sm tensor), HIP build, RCCL off, host-staged AllReduce.
- Upstream base commit: 418f905a4. Our fork = one squashed commit on top (HEAD). upstream/master is only 1 unrelated SYCL commit ahead, so `git diff 418f905a4 HEAD -- <files>` is exactly OUR customization.
- Upstream base ALREADY has: tensor-split meta backend (ggml-backend-meta.cpp), host-staged HIP AllReduce, DFlash, MTP (draft-mtp), speculative decoding. Our changes mostly EXTEND these.
- Fork docs (read-only evidence): vr/IMPROVEMENTS.md, vr/PERFORMANCE.md, vr/README.md, vr/old-ignore/** (dated experiment notes; grep them for feature names / env vars).

## Rules
- READ ONLY on the repo. Do not edit repo files, do not run git commands that change state (no checkout/commit/stash/reset).
- Do NOT build, run binaries, benchmarks or tests.
- See base version: `git show 418f905a4:<path>`; search base: `git grep -n <pat> 418f905a4 -- <paths>`.
- Write your full report to the output file given in your task. Return to caller ONLY a <=250 word summary.

## For each LOGICAL change (group hunks by purpose, not by file) report:
- Name, files + functions, approx added/removed LOC
- Purpose (what it does, why)
- Gating: gfx906/HIP-only? env var (name, default on/off)? always-on for all backends? Does it touch shared generic code paths used by other backends/architectures?
- Upstream equivalent in base? (search for it). Is ours a superset, duplicate, or replacement?
- Classification: A essential gfx906 compat / B perf optimization / C functional extension / D superseded by upstream / E potentially redundant (dead code, unused env var, debug leftovers, duplicate) / F unclear
- Removal impact, confidence (high/med/low), evidence (file:line, vr doc refs)

Specifically hunt for HIGH-CONFIDENCE low-risk cleanups:
- pure cosmetic hunks (whitespace, reformatting, renamed vars, reordered code, comment-only changes) that just add diff noise
- debug/logging/printf/env-var knobs that are never needed in the default config, or experimental paths default-off with no recorded win in vr/
- dead code: functions/defines added but never called (verify with git grep at HEAD)
- duplicated upstream logic (copied function where a small conditional would do)
- generic-code changes that could be moved behind a backend-specific hook or `#ifdef GGML_USE_HIP`

## Report format (markdown)
1. Table: | Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
2. Per-feature notes (brief, with evidence)
3. "Quick wins" list: exact file:line ranges + why it is safe
