# current

This directory contains the only current runnable source patch:

```text
gfx906-current-bigbang.patch
```

It is the speculative all-in-one patch for the next MI50/MI60 bigbang
benchmark run. It targets upstream commit `571d0d540`; an exact-gfx906 build
succeeded and the complete ROCm0 `MUL_MAT` gate passed 1134 of 1134 cases.
It also applies cleanly to a fresh archive of that commit. Model benchmarks
and direct fused value-plus-gate validation remain pending.
