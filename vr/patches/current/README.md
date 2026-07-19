# current patch

`gfx906-current-bigbang.patch` targets `571d0d540` but is stale relative to
the reduced working tree. It still contains rejected and unproven code and is
not the accepted private candidate.

First isolate the combined Q4_K/Q6_K PP regression, freeze the active compile
gates, remove disabled experiment code, and regenerate this artifact. See
`../../README.md` for the handoff and
`../../WORKTREE-BENCHMARKS.md` for the wall-clock-efficient procedure.
