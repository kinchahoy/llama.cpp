# Unverified unified-KV investigation

Exploratory notes and raw artifacts, kept separate from the official PERFORMANCE.md, TOKEN-BUDGET.md and old-ignore records.

Start with verification/RESULTS.md for the performance comparison, correctness checks, output similarity, build audit and limitations. The root README.md contains earlier capacity smoke checks and subsequent corrections; its claims are superseded by the verification report where they differ. In particular, the 350K Qwen result reduced microbatch to 512 and does not meet the user's 2048/2048 requirement. With 2048/2048 unchanged, the largest successful Qwen shared pool tested was 274432; 350000 failed allocation. Deployment configuration was not changed.

Raw logs, commands, response JSON, scripts and runtime hashes are retained for review. Invalid single-GPU benchmark data is explicitly labeled INVALID and excluded from the results. All tests have stopped.
