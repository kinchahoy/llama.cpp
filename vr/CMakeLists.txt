add_executable(vr-test-gfx906-tp test-gfx906-tp.cpp)
target_link_libraries(vr-test-gfx906-tp PRIVATE ggml)

add_executable(vr-test-flashnext-moe test-flashnext-moe.cpp)
target_link_libraries(vr-test-flashnext-moe PRIVATE ggml)
