# Pass 1 brief: behavior-neutral cleanup + fixes

Work tree: /home/raistlin/infer/mx-rebuild
It is clean upstream llama.cpp (36a0e2ffc) with our fork overlay applied (uncommitted, staged). Line numbers in INVENTORY.md are from the original repo HEAD and match this tree exactly (same file contents) before anyone edits.

Reference: /tmp/claude-1000/-home-raistlin-infer-mx-llama-cpp/552f07ca-9f17-4f87-894a-a8efaa505805/scratchpad/audit/INVENTORY.md section 3 (quick wins Q1..Q88) and section 4 (bugs B1..B33). Per-area detail in 01-*.md .. 10-*.md in the same dir.

## Hard rules
- Edit ONLY files in /home/raistlin/infer/mx-rebuild that your task lists as yours. Other agents edit other files at the same time.
- Never touch /home/raistlin/infer/mx-llama.cpp (the original repo) or any vr/ directory.
- No git commands that change state (no commit, checkout, stash, reset, add) unless your task says so explicitly.
- Do not build or run anything. A single build happens after all agents finish.
- Default runtime behavior must NOT change, except for the fixes your task names. Removing an env knob is fine only if its default path stays exactly as it is today (keep the default branch, delete the non-default branch and the getenv).
- Before deleting a symbol, `grep -rn` it across /home/raistlin/infer/mx-rebuild (excluding vr/) to confirm no remaining users. If a symbol is used by a file you do not own, keep it and note it.
- Code style: match surrounding code; ASCII only; short comments; do not add new comments explaining removals.
- Keep pipeline-stage (multi-stage TP, -tps) code for now; a later pass removes it.

## Report
Write a short markdown report to the output path in your task: each Q/B item -> done / skipped (why), files+approx lines removed, and any symbol you left because another file uses it. Return a <=120 word summary.
