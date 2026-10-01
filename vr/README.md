# Local gfx906 harness

`CMakeLists.txt` and `test-gfx906-tp.cpp` are the active tensor-parallel test harness. Enable them with `-DLLAMA_BUILD_VR_EXPERIMENTS=ON` when configuring the root build.

The current resolved runtime patches and benchmark command are in [ports/llama-head-e358d-to-mx-20261001](../ports/llama-head-e358d-to-mx-20261001/README.md). Follow [MERGE-GFX906.md](MERGE-GFX906.md) for typical conflict resolutions that preserve the MI50/MI60 performance paths. The [older pinned kit](../ports/llama-head-0bc845-to-mx-38b4/README.md) remains available. Historical experiments, discussion, results, rejected patches, and generated build files are in [OLD-IGNORE](OLD-IGNORE/ARCHIVE.md).

For future llama.cpp versions, start with the [injection commands](MERGE-GFX906.md#inject-into-a-future-upstream-head). The guide includes conflict/resume instructions, a source map of the performance paths, commands to export the next kit, and a [copyable agent handoff](MERGE-GFX906.md#copyable-agent-handoff). Apply all five patch groups and finish source reconciliation before the one final PP/TG run.
