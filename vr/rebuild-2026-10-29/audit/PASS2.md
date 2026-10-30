# Pass 2 brief: remove multi-stage TP (pipeline stages, -tps)

Work tree: /home/raistlin/infer/mx-rebuild-p2 (clean upstream 36a0e2ffc + fork overlay + pass-1 edits, uncommitted). Upstream reference: `git -C /home/raistlin/infer/mx-rebuild-p2 show upstream/master:<path>`.

Feature being removed: "multi-stage TP". With `-sm tensor`, the fork lets `-tps T < n_devs` split GPUs into n_stages = n_devs/T pipeline stages, each a TP group, with stage-to-stage transfers. Our machine has 2 GPUs and always runs ONE TP group over all devices (tps == n_devs, n_stages == 1). Goal: delete every code path that exists only for n_stages > 1, so the tree behaves EXACTLY like today's n_stages == 1 path. The feature stays available upstream of us in mx-llama.cpp commit 77ffa6093 (https://github.com/mixa3607/mx-llama.cpp).

## Contract shared by all pass-2 agents (do your part, assume others do theirs)
- `ggml_backend_meta_device(...)` returns to the upstream signature (no `tps` argument): `ggml_backend_dev_t ggml_backend_meta_device(ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud);`
- Removed entirely (definitions, exports, declarations, lookups): `ggml_backend_meta_n_stages`, `ggml_backend_meta_graph_begin`, `ggml_backend_comm_set_pipeline_stages`, `ggml_backend_comm_sendrecv_tensor`, `ggml_backend_comm_set_staging_depth`, `ggml_backend_cpy_tensor_async_dedicated_queue`, `ggml_backend_cpy_tensor_async_dedicated_drain`, plus their typedefs.
- Removed: `tensor_parallel_size` from `llama_model_params` (include/llama.h), from common params (`common.h` both places), the `-tps`/`--tensor-parallel-size` and `-tpsd` CLI args, llama-bench's `-tps` column/param, and the speculative draft copy of it.
- The scheduler n_copies logic that took max(meta n_stages) goes back to upstream behavior.
- Keep everything that the single TP group uses: lanes, parallel dispatch, whole-token graph capture/replay, AllReduce prepare/launch hooks, AR slabs, comm_init/comm_free/allreduce, split-state rules, repack bufts, TOPK broadcast, event proxy, GATHER closure.

## Hard rules
- Edit ONLY your listed files. Other agents edit the others concurrently.
- Never touch /home/raistlin/infer/mx-llama.cpp or any vr/ directory.
- No git state changes. No builds or runs.
- When a branch tests `n_stages > 1` (or `tps < n_devs`, or a stage index != 0), delete the multi-stage branch and keep the single-stage one. When code computes per-stage data, collapse it to the single stage (stage 0, all lanes). Do not restructure or "improve" single-stage code beyond what removal requires.
- Delete helpers, fields, env knobs and comments that become unused (verify with grep across the tree, excluding vr/). Env knobs that only matter for stages: remove.
- ASCII only, short comments, match surrounding style.

## Report
Write a short report (file path in your task): what you removed, approx lines, anything you were unsure about or left in place and why. Return a <=120 word summary.
