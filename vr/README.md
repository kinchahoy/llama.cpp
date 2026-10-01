# Local gfx906 harness

`CMakeLists.txt` and `test-gfx906-tp.cpp` are the active tensor-parallel test harness. Enable them with `-DLLAMA_BUILD_VR_EXPERIMENTS=ON` when configuring the root build.

The reproducible runtime patch kit, application instructions, and current benchmark command are in [ports/llama-head-0bc845-to-mx-38b4](../ports/llama-head-0bc845-to-mx-38b4/README.md). Historical experiments, discussion, results, rejected patches, and generated build files are in [OLD-IGNORE](OLD-IGNORE/ARCHIVE.md).
