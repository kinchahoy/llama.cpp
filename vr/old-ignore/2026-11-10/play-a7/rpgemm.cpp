// usage: rpgemm dev outfile perf(0/1) ; reads GGML_RP_GEMM_VAR in the library
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
struct shp { int64_t m, k, n; };
int main(int argc, char ** argv) {
    int dev_idx = atoi(argv[1]); const char * prefix = argv[2]; int perf = atoi(argv[3]);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    int gi = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) { if (gi++ == dev_idx) { dev = d; break; } }
    }
    if (!dev) { printf("no dev\n"); return 2; }
    printf("dev %s\n", ggml_backend_dev_name(dev));
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t rbuft = get_extra ? get_extra(dev)[0] : nullptr;
    if (!rbuft) { printf("no repack buft\n"); return 2; }
    ggml_backend_buffer_type_t dbuft = ggml_backend_dev_buffer_type(dev);
    std::vector<shp> shapes;
    if (perf) {
        for (int64_t m : {17408, 8704, 5120}) shapes.push_back({m, 5120, 2048});
    } else {
        for (int64_t m : {5120, 8704, 17408, 5152}) for (int64_t n : {128, 129, 191, 512, 2048}) shapes.push_back({m, 5120, n});
    }
    FILE * fo = perf ? nullptr : fopen(prefix, "wb");
    std::mt19937 rng(1);
    int bad = 0;
    for (auto s : shapes) {
        std::vector<float> w(s.m * s.k), x(s.k * s.n);
        std::normal_distribution<float> nd(0, 1);
        for (auto & v : w) v = nd(rng);
        for (auto & v : x) v = nd(rng);
        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q8_0, s.k) * s.m);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), wq.data(), 0, s.m, s.k, nullptr);
        std::vector<float> outs[2];
        double tms = 0;
        for (int rep = 0; rep < 2; rep++) {  // 0 = repack, 1 = stock reference
            ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, s.k, s.m);
            ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.n);
            ggml_tensor * c = ggml_mul_mat(ctx, a, b);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, c);
            ggml_backend_buffer_type_t bt = rep == 0 ? rbuft : dbuft;
            ggml_backend_buffer_t buf_a = ggml_backend_buft_alloc_buffer(bt, ggml_backend_buft_get_alloc_size(bt, a) + 4096);
            ggml_tallocr ta = ggml_tallocr_new(buf_a);
            ggml_tallocr_alloc(&ta, a);
            ggml_backend_buffer_t buf_rest = ggml_backend_alloc_ctx_tensors_from_buft(ctx, dbuft);
            ggml_backend_tensor_set(a, wq.data(), 0, wq.size());
            ggml_backend_tensor_set(b, x.data(), 0, x.size() * 4);
            ggml_backend_graph_compute(be, gf);
            outs[rep].resize(s.m * s.n);
            ggml_backend_tensor_get(c, outs[rep].data(), 0, outs[rep].size() * 4);
            if (perf && rep == 0) {
                for (int i = 0; i < 3; i++) ggml_backend_graph_compute(be, gf);
                ggml_backend_synchronize(be);
                auto t0 = std::chrono::steady_clock::now();
                const int N = 20;
                for (int i = 0; i < N; i++) ggml_backend_graph_compute(be, gf);
                ggml_backend_synchronize(be);
                tms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N;
            }
            ggml_backend_buffer_free(buf_a); ggml_backend_buffer_free(buf_rest); ggml_free(ctx);
            if (perf) break;
        }
        if (perf) {
            printf("m=%lld k=%lld n=%lld  %.3f ms  %.1f GOPS\n", (long long) s.m, (long long) s.k, (long long) s.n, tms, 2.0 * s.m * s.k * s.n / tms / 1e6);
        } else {
            double num = 0, den = 0;
            for (size_t i = 0; i < outs[0].size(); i++) { double d = outs[0][i] - outs[1][i]; num += d * d; den += (double) outs[1][i] * outs[1][i]; }
            double nmse = num / den;
            printf("m=%lld n=%lld nmse_vs_stock=%.3e %s\n", (long long) s.m, (long long) s.n, nmse, nmse < 1e-6 ? "ok" : "BAD");
            if (nmse >= 1e-6) bad++;
            fwrite(outs[0].data(), 4, outs[0].size(), fo);
        }
    }
    if (fo) fclose(fo);
    return bad;
}
