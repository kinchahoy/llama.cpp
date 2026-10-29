#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static std::vector<float> evaluate(ggml_backend_t backend, ggml_backend_buffer_type_t weight_type,
        int rows, int tokens, int channels, bool shared_routes) {
    constexpr int k = 640, experts = 16, used = 10;
    ggml_init_params params = {16*ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true};
    ggml_context_ptr weights(ggml_init(params));
    ggml_context_ptr ctx(ggml_init(params));
    auto * w = ggml_new_tensor_3d(weights.get(), GGML_TYPE_IQ4_NL, k, rows, experts);
    auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, channels, tokens);
    auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, used, tokens);
    auto * y = ggml_mul_mat_id(ctx.get(), w, x, ids);
    auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_ptr weight_buffer(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), weight_type));
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(weight_buffer && buffer);
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> raw(ggml_nelements(w));
    for (auto & value : raw) { value = uniform(rng); }
    std::vector<uint8_t> quantized(ggml_nbytes(w));
    ggml_quantize_chunk(w->type, raw.data(), quantized.data(), 0, rows*experts, k, nullptr);
    ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size());
    raw.resize(ggml_nelements(x));
    for (auto & value : raw) { value = uniform(rng); }
    ggml_backend_tensor_set(x, raw.data(), 0, ggml_nbytes(x));
    std::vector<int32_t> routing(used*tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int slot = 0; slot < used; ++slot) {
            routing[t*used + slot] = shared_routes ? slot : (slot + t*3) % experts;
        }
    }
    ggml_backend_tensor_set(ids, routing.data(), 0, ggml_nbytes(ids));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    std::vector<float> result(ggml_nelements(y));
    ggml_backend_tensor_get(y, result.data(), 0, ggml_nbytes(y));
    return result;
}

int main() {
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("ROCm0");
    if (!dev) { return 2; }
    auto extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts"));
    if (!extra || !extra(dev) || !extra(dev)[0] || !strstr(ggml_backend_buft_name(extra(dev)[0]), "Repack")) {
        return 2;
    }
    ggml_backend_ptr gpu(ggml_backend_dev_init(dev, nullptr));
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_ptr cpu(ggml_backend_dev_init(cpu_dev, nullptr));
    GGML_ASSERT(gpu && cpu);
    for (int rows : {17, 2560}) {
        for (int tokens : {2, 3, 8}) {
            for (int channels : {1, 10}) {
                for (bool shared_routes : {false, true}) {
                    const auto ref = evaluate(cpu.get(), ggml_backend_dev_buffer_type(cpu_dev), rows, tokens, channels, shared_routes);
                    const auto got = evaluate(gpu.get(), extra(dev)[0], rows, tokens, channels, shared_routes);
                    double error = 0.0, norm = 0.0;
                    for (size_t i = 0; i < ref.size(); ++i) {
                        const double delta = double(ref[i]) - got[i];
                        error += delta*delta;
                        norm += double(ref[i])*ref[i];
                    }
                    const double nmse = error / norm;
                    printf("repacked IQ4_NL rows=%d tokens=%d channels=%d shared_routes=%d nmse=%.6g\n", rows, tokens, channels, int(shared_routes), nmse);
                    if (!std::isfinite(nmse) || nmse > 5e-4) { return 1; }
                }
            }
        }
    }
    return 0;
}
