#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int64_t K_FULL = 17408;
static constexpr int64_t M      = 5120;

static ggml_backend_meta_split_state get_split_state(const ggml_tensor * tensor, void *) {
    if (strcmp(tensor->name, "weight") == 0 || strcmp(tensor->name, "activation") == 0) {
        GGML_ASSERT(tensor->ne[0] % 2 == 0);
        ggml_backend_meta_split_state state = {
            GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1
        };
        state.ne[0] = tensor->ne[0] / 2;
        state.ne[1] = tensor->ne[0] / 2;
        return state;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static void initialize_check_data(ggml_tensor * weight, ggml_tensor * activation) {
    const size_t block_size = ggml_type_size(GGML_TYPE_Q8_0);
    const size_t block_ne   = ggml_blck_size(GGML_TYPE_Q8_0);
    GGML_ASSERT(block_size == sizeof(ggml_fp16_t) + block_ne);

    std::vector<uint8_t> weight_data(ggml_nbytes(weight));
    const ggml_fp16_t one_f16 = ggml_fp32_to_fp16(1.0f);
    for (size_t offset = 0; offset < weight_data.size(); offset += block_size) {
        memcpy(weight_data.data() + offset, &one_f16, sizeof(one_f16));
        memset(weight_data.data() + offset + sizeof(one_f16), 1, block_ne);
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());

    std::vector<float> activation_data(ggml_nelements(activation), 1.0f);
    ggml_backend_tensor_set(activation, activation_data.data(), 0, activation_data.size() * sizeof(float));
}

int main(int argc, char ** argv) {
    int64_t n = 0;
    bool check = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--check") == 0) {
            check = true;
        } else {
            fprintf(stderr, "usage: %s --n 1|2048 [--check]\n", argv[0]);
            return 2;
        }
    }
    if (n != 1 && n != 2048) {
        fprintf(stderr, "--n must be 1 or 2048\n");
        return 2;
    }
    ggml_backend_load_all();

    ggml_backend_dev_t devices[2] = {};
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        if (strcmp(name, "ROCm0") == 0) {
            devices[0] = dev;
        } else if (strcmp(name, "ROCm1") == 0) {
            devices[1] = dev;
        }
    }
    if (devices[0] == nullptr || devices[1] == nullptr) {
        fprintf(stderr, "ROCm0 and ROCm1 are required\n");
        return 1;
    }

    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devices, 2, 2, get_split_state, nullptr);
    ggml_backend_ptr backend(ggml_backend_dev_init(meta_dev, nullptr));
    if (!backend) {
        fprintf(stderr, "failed to initialize the two-ROCm meta backend\n");
        return 1;
    }

    ggml_init_params params_static = {
        ggml_tensor_overhead() * 4,
        nullptr,
        true,
    };
    ggml_context_ptr ctx_static(ggml_init(params_static));
    if (!ctx_static) {
        fprintf(stderr, "failed to initialize static GGML context\n");
        return 1;
    }

    ggml_tensor * weight     = ggml_new_tensor_2d(ctx_static.get(), GGML_TYPE_Q8_0, K_FULL, M);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx_static.get(), GGML_TYPE_F32, K_FULL, n);
    ggml_tensor * residual   = ggml_new_tensor_2d(ctx_static.get(), GGML_TYPE_F32, M, n);
    ggml_set_name(weight, "weight");
    ggml_set_name(activation, "activation");
    ggml_set_name(residual, "residual");

    ggml_backend_buffer_ptr buffer_static(ggml_backend_alloc_ctx_tensors(ctx_static.get(), backend.get()));
    if (!buffer_static) {
        fprintf(stderr, "failed to allocate static meta-backend tensors\n");
        return 1;
    }
    ggml_backend_buffer_set_usage(buffer_static.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(buffer_static.get(), 0);
    if (check) {
        initialize_check_data(weight, activation);
    }

    const size_t graph_size = 8;
    ggml_init_params params_compute = {
        ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(graph_size, false),
        nullptr,
        true,
    };
    ggml_context_ptr ctx_compute(ggml_init(params_compute));
    if (!ctx_compute) {
        fprintf(stderr, "failed to initialize compute GGML context\n");
        return 1;
    }

    ggml_tensor * down = ggml_mul_mat(ctx_compute.get(), weight, activation);
    ggml_set_name(down, "down_partial");
    ggml_tensor * out = ggml_add(ctx_compute.get(), down, residual);
    ggml_set_name(out, "out");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx_compute.get(), graph_size, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_ptr backend_cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!backend_cpu) {
        fprintf(stderr, "failed to initialize CPU scheduler fallback\n");
        return 1;
    }

    ggml_backend_t backends[] = {backend.get(), backend_cpu.get()};
    ggml_backend_buffer_type_t bufts[] = {
        ggml_backend_get_default_buffer_type(backends[0]),
        ggml_backend_get_default_buffer_type(backends[1]),
    };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, graph_size, false, true));
    if (!sched || !ggml_backend_sched_alloc_graph(sched.get(), graph)) {
        fprintf(stderr, "failed to allocate scheduler graph\n");
        return 1;
    }
    if (out->buffer == nullptr || ggml_backend_buffer_get_usage(out->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        fprintf(stderr, "output was not allocated in a compute buffer\n");
        return 1;
    }

    auto compute = [&]() {
        const ggml_status status = ggml_backend_sched_graph_compute(sched.get(), graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed with status %d\n", (int) status);
            return false;
        }
        ggml_backend_sched_synchronize(sched.get());
        return true;
    };

    if (!compute()) {
        return 1;
    }

    const int64_t start_us = ggml_time_us();
    if (!compute()) {
        return 1;
    }
    const int64_t time_us = ggml_time_us() - start_us;

    if (check) {
        float values[16] = {};
        ggml_backend_tensor_get(out, values, 0, sizeof(values));
        for (float value : values) {
            if (!std::isfinite(value) || std::abs(value - float(K_FULL)) > 2.0f) {
                fprintf(stderr, "incorrect output: got %.6f, expected %" PRId64 "\n", value, K_FULL);
                return 1;
            }
        }
    }

    const char * allreduce = getenv("GGML_CUDA_ALLREDUCE");
    const char * overlap = getenv("GGML_CUDA_TP_OVERLAP");
    const char * overlap_bf16 = getenv("GGML_CUDA_TP_OVERLAP_BF16");
    printf("gfx906_tp_down n=%" PRId64 " time_us=%" PRId64 " result_bytes=%zu allreduce=%s overlap=%s wire=%s output_usage=compute check=%s\n",
        n, time_us, size_t(M * n * sizeof(float)), allreduce ? allreduce : "default", overlap ? overlap : "off",
        overlap_bf16 ? "bf16" : "f32", check ? "pass" : "not-run");
    return 0;
}
