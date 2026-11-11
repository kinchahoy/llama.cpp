// Q3 go/no-go: streaming read vs repacked Q8_0 mat-vec at the 27B per-GPU decode shapes.
#include "ggml-cuda/q8_repack/repack-kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)

struct mshape { const char * name; int ne0, ne1, count; bool fused; };

// one 27B token on one GPU of the 1/1 tensor split
static std::vector<mshape> token_shapes(int nl) {
    // nl: number of layers to model (scaled 64 = 48 GDN + 16 attn)
    std::vector<mshape> s;
    int n_attn = nl / 4, n_gdn = nl - n_attn;
    s.push_back({"ffn_gate_up(fused)", 5120, 8704, nl, true});
    s.push_back({"ffn_down", 8704, 5120, nl, false});
    s.push_back({"gdn_qkv", 5120, 5120, n_gdn, false});
    s.push_back({"gdn_gate", 5120, 3072, n_gdn, false});
    s.push_back({"gdn_alpha", 5120, 24, n_gdn, false});
    s.push_back({"gdn_beta", 5120, 24, n_gdn, false});
    s.push_back({"ssm_out", 3072, 5120, n_gdn, false});
    s.push_back({"attn_q", 5120, 6144, n_attn, false});
    s.push_back({"attn_k", 5120, 512, n_attn, false});
    s.push_back({"attn_v", 5120, 512, n_attn, false});
    s.push_back({"attn_out", 3072, 5120, n_attn, false});
    return s;
}

static size_t mbytes(int ne0, int ne1) { return repack_gcn_nbytes(GGML_TYPE_Q8_0, ne0, ne1); }

// Pure stream: each block reads a contiguous chunk with U independent uint4 loads per thread.
template <int U>
__global__ void __launch_bounds__(256) stream_read(const uint4 * __restrict__ p, size_t n16, uint32_t * __restrict__ out) {
    uint32_t acc = 0;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (U - 1) * stride < n16; i += U * stride) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = p[i + u * stride];
#pragma unroll
        for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    for (; i < n16; i += stride) { uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678u) out[0] = acc;
}

// Each wave reads its own contiguous chunk of 64*16*U*NIT bytes, U loads in flight.
template <int U>
__global__ void __launch_bounds__(256) stream_chunk(const uint4 * __restrict__ p, size_t n16, size_t per_wave16, uint32_t * __restrict__ out) {
    uint32_t acc = 0;
    const size_t wave = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) / 64;
    const int lane = threadIdx.x % 64;
    size_t i = wave * per_wave16 + lane;
    const size_t end = min(n16, (wave + 1) * per_wave16);
    for (; i + (U - 1) * 64 < end; i += U * 64) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = p[i + u * 64];
#pragma unroll
        for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    for (; i < end; i += 64) { uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678u) out[0] = acc;
}

// Same shape as the real kernel (one wave per row, ROWS rows per block) but no math.
template <int ROWS>
__global__ void __launch_bounds__(ROWS * 64) stream_rows(const uint8_t * __restrict__ w, uint32_t ne0, uint32_t ne1, uint32_t * __restrict__ out) {
    const uint32_t qs_str = repack_qs_stride(ne0);
    const int row = blockIdx.x * ROWS + threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const uint32_t wrow = min((uint32_t) row, ne1 - 1);
    const uint4 * wr = reinterpret_cast<const uint4 *>(w + (size_t) wrow * qs_str);
    const uint16_t * dr = reinterpret_cast<const uint16_t *>(w + (size_t) ne1 * qs_str) + (size_t) wrow * (ne0 / 32);
    const uint32_t n_half = ne0 / 16;
    uint32_t acc = 0;
    for (uint32_t hb = lane; hb < n_half; hb += 64) {
        uint4 v = wr[hb];
        acc ^= v.x ^ v.y ^ v.z ^ v.w ^ dr[hb >> 1];
    }
    if (acc == 0x12345678u) out[0] = acc;
}


// S1: one wave per row, all loads of the row in flight.
template <int ROWS, int MAXL, bool D = true>
__global__ void __launch_bounds__(ROWS * 64) stream_rows_all(const uint8_t * __restrict__ w, uint32_t ne0, uint32_t ne1, uint32_t * __restrict__ out, uint32_t qs_str_o = 0) {
    const uint32_t qs_str = qs_str_o ? qs_str_o : repack_qs_stride(ne0);
    const int row = blockIdx.x * ROWS + threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const uint32_t wrow = min((uint32_t) row, ne1 - 1);
    const uint4 * wr = reinterpret_cast<const uint4 *>(w + (size_t) wrow * qs_str);
    const uint16_t * dr = reinterpret_cast<const uint16_t *>(w + (size_t) ne1 * qs_str) + (size_t) wrow * (ne0 / 32);
    const uint32_t n_half = ne0 / 16;
    uint4 v[MAXL]; uint16_t d[MAXL];
#pragma unroll
    for (int j = 0; j < MAXL; ++j) { uint32_t hb = lane + 64 * j; if (hb < n_half) { v[j] = wr[hb]; d[j] = D ? dr[hb >> 1] : 0; } else { v[j] = make_uint4(0,0,0,0); d[j] = 0; } }
    uint32_t acc = 0;
#pragma unroll
    for (int j = 0; j < MAXL; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w ^ d[j];
    if (acc == 0x12345678u) out[0] = acc;
}

// S2: block of ROWS rows read as one contiguous region; wave w at step s reads piece s*ROWS+w.
template <int ROWS>
__global__ void __launch_bounds__(ROWS * 64) stream_rows_il(const uint8_t * __restrict__ w, uint32_t ne0, uint32_t ne1, uint32_t * __restrict__ out) {
    const uint32_t qs_str = repack_qs_stride(ne0);
    const int wv = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const uint32_t r0 = blockIdx.x * ROWS;
    const uint32_t nr = min((uint32_t) ROWS, ne1 - r0);
    const uint4 * base = reinterpret_cast<const uint4 *>(w + (size_t) r0 * qs_str);
    const uint32_t n16 = nr * qs_str / 16;
    uint32_t acc = 0;
    for (uint32_t p = wv * 64 + lane; p < n16; p += ROWS * 64) { uint4 v = base[p]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    const uint16_t * dr = reinterpret_cast<const uint16_t *>(w + (size_t) ne1 * qs_str) + (size_t) r0 * (ne0 / 32);
    for (uint32_t p = threadIdx.x; p < nr * (ne0 / 32); p += ROWS * 64) acc ^= dr[p];
    if (acc == 0x12345678u) out[0] = acc;
}


// Candidate: dense LANES=64, WPR=1. U loads in flight per lane, predicated last batch, same per-lane sum order.
template <int ROWS, int U, bool HAS_FUSION>
static __global__ void __launch_bounds__(ROWS * 64) mmv_cand(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint8_t * __restrict__ wbase_gate, const ggml_glu_op glu_op) {
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t qs_str   = repack_qs_stride(ne0);
    const int lane = threadIdx.x % 64;
    const int row  = blockIdx.x * ROWS + threadIdx.x / 64;
    const uint32_t wrow = min((uint32_t) row, ne1 - 1);
    const uint4    * w_row = reinterpret_cast<const uint4 *>(wbase + (size_t) wrow * qs_str);
    const uint16_t * d_row = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * qs_str) + (size_t) wrow * n_blocks;
    const uint16_t rmask = (row < (int) ne1) ? (uint16_t) 0xffffu : (uint16_t) 0x0000u;
    const bool use_gate = HAS_FUSION && wbase_gate != nullptr;
    const uint4    * w_row_g = nullptr;
    const uint16_t * d_row_g = nullptr;
    if constexpr (HAS_FUSION) {
        if (use_gate) {
            w_row_g = reinterpret_cast<const uint4 *>(wbase_gate + (size_t) wrow * qs_str);
            d_row_g = reinterpret_cast<const uint16_t *>(wbase_gate + (size_t) ne1 * qs_str) + (size_t) wrow * n_blocks;
        }
    }
    const uint32_t n_half = n_blocks * 2;
    float acc = 0.0f, acc_gate = 0.0f;
    auto step = [&](const uint32_t base, const bool check) {
        uint4 wv[U]; uint16_t db[U]; int xs[U][4]; float dxv[U];
        [[maybe_unused]] uint4 wg[U]; [[maybe_unused]] uint16_t dg[U];
#pragma unroll
        for (int j = 0; j < U; ++j) {
            const uint32_t hb = base + j * 64 + lane;
            if (!check || hb < n_half) {
                const block_q8_1 * xb = xq + (hb >> 1);
                dxv[j] = __low2float(xb->ds);
                const int * xq32 = reinterpret_cast<const int *>(xb->qs) + (hb & 1) * 4;
                xs[j][0] = xq32[0]; xs[j][1] = xq32[1]; xs[j][2] = xq32[2]; xs[j][3] = xq32[3];
                wv[j] = w_row[hb];
                db[j] = d_row[hb >> 1] & rmask;
                if constexpr (HAS_FUSION) {
                    if (use_gate) { wg[j] = w_row_g[hb]; dg[j] = d_row_g[hb >> 1] & rmask; }
                }
            }
        }
#pragma unroll
        for (int j = 0; j < U; ++j) {
            const uint32_t hb = base + j * 64 + lane;
            if (!check || hb < n_half) {
                const float dw = __half2float(*reinterpret_cast<const __half *>(&db[j]));
                acc += dw * dxv[j] * (float) mmvq_dp4a_u4(wv[j], xs[j][0], xs[j][1], xs[j][2], xs[j][3]);
                if constexpr (HAS_FUSION) {
                    if (use_gate) {
                        const float dwg = __half2float(*reinterpret_cast<const __half *>(&dg[j]));
                        acc_gate += dwg * dxv[j] * (float) mmvq_dp4a_u4(wg[j], xs[j][0], xs[j][1], xs[j][2], xs[j][3]);
                    }
                }
            }
        }
    };
    uint32_t base = 0;
    for (; base + U * 64 <= n_half; base += U * 64) {
        step(base, false);
    }
    if (base < n_half) {
        step(base, true);
    }
    const float a = rp_warp_reduce_sum<64>(acc);
    if constexpr (HAS_FUSION) {
        const float g = rp_warp_reduce_sum<64>(acc_gate);
        if (lane == 0 && row < (int) ne1) {
            y[row] = rp_mmv_fusion_epilogue(a, g, nullptr, nullptr, (uint32_t) row, glu_op, use_gate);
        }
    } else {
        if (lane == 0 && row < (int) ne1) {
            y[row] = a;
        }
    }
}

int main(int argc, char ** argv) {
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    int nl = 16; // layers per pass (about 3.4 GB), well above the 4 MB L2
    int iters = 5;
    CK(hipSetDevice(dev));
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, dev));
    printf("device %d %s CUs %d\n", dev, prop.gcnArchName, prop.multiProcessorCount);

    auto shapes = token_shapes(nl);
    struct inst { int si; uint8_t * w; uint8_t * wg; };
    std::vector<inst> insts;
    size_t total = 0;
    // interleave per layer like the real graph
    for (int l = 0; l < nl; ++l) {
        for (int si = 0; si < (int) shapes.size(); ++si) {
            const mshape & m = shapes[si];
            bool attn = (l % 4) == 3;
            bool gdn_only = std::string(m.name).rfind("gdn", 0) == 0 || std::string(m.name) == "ssm_out";
            bool attn_only = std::string(m.name).rfind("attn", 0) == 0;
            if ((gdn_only && attn) || (attn_only && !attn)) continue;
            inst it{si, nullptr, nullptr};
            size_t nb = mbytes(m.ne0, m.ne1);
            CK(hipMalloc(&it.w, nb)); total += nb;
            if (m.fused) { CK(hipMalloc(&it.wg, nb)); total += nb; }
            insts.push_back(it);
        }
    }
    // fill weights with small random bytes (scales become finite f16)
    {
        std::vector<uint8_t> h(64u << 20);
        for (auto & b : h) b = (uint8_t) (rand() & 0x3f);
        for (auto & it : insts) {
            for (uint8_t * p : {it.w, it.wg}) {
                if (!p) continue;
                size_t nb = mbytes(shapes[it.si].ne0, shapes[it.si].ne1);
                for (size_t o = 0; o < nb; o += h.size()) CK(hipMemcpy(p + o, h.data(), std::min(h.size(), nb - o), hipMemcpyHostToDevice));
            }
        }
    }
    printf("pass bytes %.3f GB over %zu mat-vecs (%d layers)\n", total / 1e9, insts.size(), nl);

    block_q8_1 * xq; CK(hipMalloc(&xq, 9216 / 32 * sizeof(block_q8_1)));
    {
        std::vector<block_q8_1> hx(9216 / 32);
        for (auto & b : hx) {
            for (int k = 0; k < 32; ++k) b.qs[k] = (int8_t) ((rand() % 255) - 127);
            b.ds = __halves2half2(__float2half((rand() % 2000 - 1000) * 1e-4f), __float2half(0.0f));
        }
        CK(hipMemcpy(xq, hx.data(), hx.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
    }
    float * y; CK(hipMalloc(&y, 32768 * sizeof(float)));
    uint32_t * sink; CK(hipMalloc(&sink, 64));
    hipStream_t st; CK(hipStreamCreate(&st));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));

    auto timeit = [&](const char * label, auto && body) {
        body(); CK(hipStreamSynchronize(st));
        CK(hipEventRecord(e0, st));
        for (int i = 0; i < iters; ++i) body();
        CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        ms /= iters;
        printf("%-34s %8.3f ms  %7.1f GB/s\n", label, ms, total / (ms * 1e6));
        return ms;
    };

    // one big flat buffer for a peak number
    {
        size_t nb = 2ull << 30; uint4 * p; CK(hipMalloc(&p, nb)); CK(hipMemset(p, 1, nb));
        for (int grid_mul : {4, 8, 16, 32}) {
            hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
            stream_read<4><<<prop.multiProcessorCount * grid_mul, 256, 0, st>>>(p, nb / 16, sink);
            CK(hipEventRecord(a, st));
            for (int i = 0; i < 10; ++i) stream_read<4><<<prop.multiProcessorCount * grid_mul, 256, 0, st>>>(p, nb / 16, sink);
            CK(hipEventRecord(b, st)); CK(hipEventSynchronize(b));
            float ms; CK(hipEventElapsedTime(&ms, a, b));
            printf("flat 2GiB stream U4 grid %2dxCU      %7.1f GB/s\n", grid_mul, 10.0 * nb / (ms * 1e6));
        }
        for (int kb : {1, 2, 4, 8, 16}) {
            size_t pw = (size_t) kb * 1024 / 16; if (pw < 64 * 8) { /* U8 needs 8 KB per wave */ }
            int g = (int) ((nb / 16 + pw * 4 - 1) / (pw * 4));
            auto run = [&] { stream_chunk<4><<<g, 256, 0, st>>>(p, nb / 16, pw, sink); };
            hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
            run(); CK(hipEventRecord(a, st));
            for (int i = 0; i < 10; ++i) run();
            CK(hipEventRecord(b, st)); CK(hipEventSynchronize(b));
            float ms; CK(hipEventElapsedTime(&ms, a, b));
            printf("flat 2GiB chunk U4 %3d KB/wave       %7.1f GB/s\n", kb, 10.0 * nb / (ms * 1e6));
        }
        {
            uint4 * q; CK(hipMalloc(&q, nb / 2));
            hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
            CK(hipMemcpyAsync(q, p, nb / 2, hipMemcpyDeviceToDevice, st)); CK(hipEventRecord(a, st));
            for (int i = 0; i < 10; ++i) CK(hipMemcpyAsync(q, p, nb / 2, hipMemcpyDeviceToDevice, st));
            CK(hipEventRecord(b, st)); CK(hipEventSynchronize(b));
            float ms; CK(hipEventElapsedTime(&ms, a, b));
            printf("D2D memcpy 1GiB (read+write)        %7.1f GB/s\n", 10.0 * nb / (ms * 1e6));
            CK(hipFree(q));
        }
        CK(hipFree(p));
    }

    for (int grid_mul : {4, 8, 16}) {
        char lbl[64]; snprintf(lbl, sizeof lbl, "per-matrix stream U4 grid %dxCU", grid_mul);
        timeit(lbl, [&] {
            for (auto & it : insts) {
                const mshape & m = shapes[it.si];
                size_t n16 = mbytes(m.ne0, m.ne1) / 16;
                int g = (int) std::min<size_t>(prop.multiProcessorCount * grid_mul, (n16 + 1023) / 1024);
                stream_read<4><<<g, 256, 0, st>>>((const uint4 *) it.w, n16, sink);
                if (it.wg) stream_read<4><<<g, 256, 0, st>>>((const uint4 *) it.wg, n16, sink);
            }
        });
    }
    timeit("row-shaped stream (8 rows/block)", [&] {
        for (auto & it : insts) {
            const mshape & m = shapes[it.si];
            stream_rows<8><<<(m.ne1 + 7) / 8, 512, 0, st>>>(it.w, m.ne0, m.ne1, sink);
            if (it.wg) stream_rows<8><<<(m.ne1 + 7) / 8, 512, 0, st>>>(it.wg, m.ne0, m.ne1, sink);
        }
    });

    for (int kb : {2, 4, 8}) {
        char lbl[64]; snprintf(lbl, sizeof lbl, "per-matrix chunk U4 %d KB/wave", kb);
        timeit(lbl, [&] {
            for (auto & it : insts) {
                const mshape & m = shapes[it.si];
                size_t n16 = mbytes(m.ne0, m.ne1) / 16;
                size_t pw = (size_t) kb * 1024 / 16;
                int g = (int) ((n16 + pw * 4 - 1) / (pw * 4));
                stream_chunk<4><<<g, 256, 0, st>>>((const uint4 *) it.w, n16, pw, sink);
                if (it.wg) stream_chunk<4><<<g, 256, 0, st>>>((const uint4 *) it.wg, n16, pw, sink);
            }
        });
    }
    auto realw = [&](auto rows_tag, auto nw_tag) {
        constexpr int R = decltype(rows_tag)::value;
        constexpr int NW = decltype(nw_tag)::value;
        for (auto & it : insts) {
            const mshape & m = shapes[it.si];
            if (m.fused) {
                mul_mat_vec_q8_0_repacked<R, NW, false, 64, true><<<(m.ne1 + R - 1) / R, NW * 64, 0, st>>>(
                    it.w, xq, y, m.ne0, m.ne1, nullptr, nullptr, nullptr, 0, 1, 0, 0, 0, it.wg, nullptr, nullptr, GGML_GLU_OP_SWIGLU);
            } else {
                mul_mat_vec_q8_0_repacked<R, NW, false, 64, false><<<(m.ne1 + R - 1) / R, NW * 64, 0, st>>>(
                    it.w, xq, y, m.ne0, m.ne1, nullptr, nullptr, nullptr, 0, 1, 0, 0, 0, nullptr, nullptr, nullptr, GGML_GLU_OP_SWIGLU);
            }
        }
    };
    using std::integral_constant;
    auto cand = [&](auto rows_tag, auto u_tag) {
        constexpr int R = decltype(rows_tag)::value;
        constexpr int U = decltype(u_tag)::value;
        for (auto & it : insts) {
            const mshape & m = shapes[it.si];
            if (m.fused) mmv_cand<R, U, true><<<(m.ne1 + R - 1) / R, R * 64, 0, st>>>(it.w, xq, y, m.ne0, m.ne1, it.wg, GGML_GLU_OP_SWIGLU);
            else         mmv_cand<R, U, false><<<(m.ne1 + R - 1) / R, R * 64, 0, st>>>(it.w, xq, y, m.ne0, m.ne1, nullptr, GGML_GLU_OP_SWIGLU);
        }
    };
    // bit-identity check vs real kernel, every shape
    {
        float * y2; CK(hipMalloc(&y2, 32768 * sizeof(float)));
        std::vector<float> h1(32768), h2(32768);
        int bad = 0;
        for (int si = 0; si < (int) shapes.size(); ++si) {
            for (auto & it : insts) if (it.si == si) {
                const mshape & m = shapes[si];
                CK(hipMemset(y, 0, 32768 * 4)); CK(hipMemset(y2, 0, 32768 * 4));
                if (m.fused) {
                    mul_mat_vec_q8_0_repacked<8, 8, false, 64, true><<<(m.ne1 + 7) / 8, 512, 0, st>>>(it.w, xq, y, m.ne0, m.ne1, nullptr, nullptr, nullptr, 0, 1, 0, 0, 0, it.wg, nullptr, nullptr, GGML_GLU_OP_SWIGLU);
                    mmv_cand<8, 4, true><<<(m.ne1 + 7) / 8, 512, 0, st>>>(it.w, xq, y2, m.ne0, m.ne1, it.wg, GGML_GLU_OP_SWIGLU);
                } else {
                    mul_mat_vec_q8_0_repacked<8, 8, false, 64, false><<<(m.ne1 + 7) / 8, 512, 0, st>>>(it.w, xq, y, m.ne0, m.ne1, nullptr, nullptr, nullptr, 0, 1, 0, 0, 0, nullptr, nullptr, nullptr, GGML_GLU_OP_SWIGLU);
                    mmv_cand<8, 4, false><<<(m.ne1 + 7) / 8, 512, 0, st>>>(it.w, xq, y2, m.ne0, m.ne1, nullptr, GGML_GLU_OP_SWIGLU);
                }
                CK(hipMemcpy(h1.data(), y, m.ne1 * 4, hipMemcpyDeviceToHost));
                CK(hipMemcpy(h2.data(), y2, m.ne1 * 4, hipMemcpyDeviceToHost));
                if (memcmp(h1.data(), h2.data(), m.ne1 * 4) != 0) { bad++; printf("MISMATCH %s y[0]=%g vs %g\n", m.name, h1[0], h2[0]); }
                break;
            }
        }
        printf("bit-identity check: %s (sample y=%g)\n", bad ? "FAIL" : "pass", h1[1]);
    }
    timeit("cand R8 U2", [&] { cand(integral_constant<int, 8>{}, integral_constant<int, 2>{}); });
    timeit("cand R8 U3", [&] { cand(integral_constant<int, 8>{}, integral_constant<int, 3>{}); });
    timeit("cand R8 U4", [&] { cand(integral_constant<int, 8>{}, integral_constant<int, 4>{}); });
    timeit("cand R8 U5", [&] { cand(integral_constant<int, 8>{}, integral_constant<int, 5>{}); });
    timeit("cand R4 U4", [&] { cand(integral_constant<int, 4>{}, integral_constant<int, 4>{}); });
    timeit("cand R4 U5", [&] { cand(integral_constant<int, 4>{}, integral_constant<int, 5>{}); });
    timeit("cand R16 U4", [&] { cand(integral_constant<int, 16>{}, integral_constant<int, 4>{}); });
    timeit("real WPR2 <8,16>", [&] { realw(integral_constant<int, 8>{}, integral_constant<int, 16>{}); });
    timeit("real WPR2 <4,8>", [&] { realw(integral_constant<int, 4>{}, integral_constant<int, 8>{}); });
    timeit("real WPR4 <4,16>", [&] { realw(integral_constant<int, 4>{}, integral_constant<int, 16>{}); });
    timeit("real WPR4 <2,8>", [&] { realw(integral_constant<int, 2>{}, integral_constant<int, 8>{}); });
    timeit("real WPR8 <2,16>", [&] { realw(integral_constant<int, 2>{}, integral_constant<int, 16>{}); });
    auto real = [&](auto rows_tag) {
        constexpr int R = decltype(rows_tag)::value;
        for (auto & it : insts) {
            const mshape & m = shapes[it.si];
            if (m.fused) {
                mul_mat_vec_q8_0_repacked<R, R, false, 64, true><<<(m.ne1 + R - 1) / R, R * 64, 0, st>>>(
                    it.w, xq, y, m.ne0, m.ne1, nullptr, nullptr, nullptr, 0, 1, 0, 0, 0, it.wg, nullptr, nullptr, GGML_GLU_OP_SWIGLU);
            } else {
                mul_mat_vec_q8_0_repacked<R, R, false, 64, false><<<(m.ne1 + R - 1) / R, R * 64, 0, st>>>(
                    it.w, xq, y, m.ne0, m.ne1, nullptr, nullptr, nullptr, 0, 1, 0, 0, 0, nullptr, nullptr, nullptr, GGML_GLU_OP_SWIGLU);
            }
        }
    };
    timeit("real repacked mat-vec ROWS=8", [&] { real(std::integral_constant<int, 8>{}); });
    timeit("real repacked mat-vec ROWS=16", [&] { real(std::integral_constant<int, 16>{}); });
    timeit("real repacked mat-vec ROWS=4", [&] { real(std::integral_constant<int, 4>{}); });

    // big single matrices: rate without launch tails
    for (int ne1b : {124160, 8704}) {
        const int ne0b = 5120;
        size_t nb = mbytes(ne0b, ne1b); uint8_t * wb; CK(hipMalloc(&wb, nb + (size_t) ne1b * 1100)); CK(hipMemset(wb, 0x11, nb + (size_t) ne1b * 1100));
        float * yb; CK(hipMalloc(&yb, (size_t) ne1b * sizeof(float)));
        auto t = [&](const char * lbl, auto && f) {
            f(); CK(hipStreamSynchronize(st));
            CK(hipEventRecord(e0, st)); for (int i = 0; i < 10; ++i) f(); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); ms /= 10;
            printf("big %dx%d %-30s %8.1f us %7.1f GB/s\n", ne0b, ne1b, lbl, ms * 1e3, nb / (ms * 1e6));
        };
        t("real ROWS=8", [&] { mul_mat_vec_q8_0_repacked<8, 8, false, 64, false><<<(ne1b + 7) / 8, 512, 0, st>>>(wb, xq, yb, ne0b, ne1b, nullptr, nullptr, nullptr, 0, 1, 0, 0, 0, nullptr, nullptr, nullptr, GGML_GLU_OP_SWIGLU); });
        for (int kb : {1, 2, 4}) {
            char l[32]; snprintf(l, sizeof l, "chunk U4 %d KB/wave", kb);
            size_t n16 = nb / 16, pw = (size_t) kb * 1024 / 16; int g = (int) ((n16 + pw * 4 - 1) / (pw * 4));
            t(l, [&] { stream_chunk<4><<<g, 256, 0, st>>>((const uint4 *) wb, n16, pw, sink); });
        }
        t("row-shaped stream 8 rows", [&] { stream_rows<8><<<(ne1b + 7) / 8, 512, 0, st>>>(wb, ne0b, ne1b, sink); });
        t("S1 rows all-in-flight 8", [&] { stream_rows_all<8, 5><<<(ne1b + 7) / 8, 512, 0, st>>>(wb, ne0b, ne1b, sink); });
        t("S1 rows all-in-flight 4", [&] { stream_rows_all<4, 5><<<(ne1b + 3) / 4, 256, 0, st>>>(wb, ne0b, ne1b, sink); });
        t("S1 no scales 8", [&] { stream_rows_all<8, 5, false><<<(ne1b + 7) / 8, 512, 0, st>>>(wb, ne0b, ne1b, sink); });
        for (uint32_t strd : {5120u, 5136u, 5184u, 5248u, 6144u}) { char l[40]; snprintf(l, sizeof l, "S1 stride %u", strd);
            if ((size_t) strd * ne1b + (size_t) ne1b * 320 > nb + (size_t) ne1b * 1024) continue;
            t(l, [&] { stream_rows_all<8, 5><<<(ne1b + 7) / 8, 512, 0, st>>>(wb, ne0b, ne1b, sink, strd); }); }
        t("S2 interleaved 8", [&] { stream_rows_il<8><<<(ne1b + 7) / 8, 512, 0, st>>>(wb, ne0b, ne1b, sink); });
        t("S2 interleaved 4", [&] { stream_rows_il<4><<<(ne1b + 3) / 4, 256, 0, st>>>(wb, ne0b, ne1b, sink); });
        t("S2 interleaved 16", [&] { stream_rows_il<16><<<(ne1b + 15) / 16, 1024, 0, st>>>(wb, ne0b, ne1b, sink); });
        CK(hipFree(wb)); CK(hipFree(yb));
    }
    for (int kb : {1}) {
        char lbl[64]; snprintf(lbl, sizeof lbl, "per-matrix chunk U4 %d KB/wave", kb);
        timeit(lbl, [&] {
            for (auto & it : insts) {
                const mshape & m = shapes[it.si];
                size_t n16 = mbytes(m.ne0, m.ne1) / 16;
                size_t pw = (size_t) kb * 1024 / 16;
                int g = (int) ((n16 + pw * 4 - 1) / (pw * 4));
                stream_chunk<4><<<g, 256, 0, st>>>((const uint4 *) it.w, n16, pw, sink);
                if (it.wg) stream_chunk<4><<<g, 256, 0, st>>>((const uint4 *) it.wg, n16, pw, sink);
            }
        });
    }
    // per-class breakdown, real kernel ROWS=8
    for (int si = 0; si < (int) shapes.size(); ++si) {
        size_t cb = 0; int cnt = 0;
        for (auto & it : insts) if (it.si == si) { cb += mbytes(shapes[si].ne0, shapes[si].ne1) * (it.wg ? 2 : 1); cnt++; }
        if (!cnt) continue;
        auto body = [&] {
            for (auto & it : insts) if (it.si == si) {
                const mshape & m = shapes[si];
                if (m.fused) mul_mat_vec_q8_0_repacked<8, 8, false, 64, true><<<(m.ne1 + 7) / 8, 512, 0, st>>>(it.w, xq, y, m.ne0, m.ne1, nullptr, nullptr, nullptr, 0, 1, 0, 0, 0, it.wg, nullptr, nullptr, GGML_GLU_OP_SWIGLU);
                else mul_mat_vec_q8_0_repacked<8, 8, false, 64, false><<<(m.ne1 + 7) / 8, 512, 0, st>>>(it.w, xq, y, m.ne0, m.ne1, nullptr, nullptr, nullptr, 0, 1, 0, 0, 0, nullptr, nullptr, nullptr, GGML_GLU_OP_SWIGLU);
            }
        };
        auto sbody = [&] {
            for (auto & it : insts) if (it.si == si) {
                size_t n16 = mbytes(shapes[si].ne0, shapes[si].ne1) / 16;
                int g = (int) std::min<size_t>(prop.multiProcessorCount * 8, (n16 + 1023) / 1024);
                stream_read<4><<<g, 256, 0, st>>>((const uint4 *) it.w, n16, sink);
                if (it.wg) stream_read<4><<<g, 256, 0, st>>>((const uint4 *) it.wg, n16, sink);
            }
        };
        body(); sbody(); CK(hipStreamSynchronize(st));
        float ms, ms2;
        CK(hipEventRecord(e0, st)); for (int i = 0; i < iters; ++i) body(); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1));
        CK(hipEventRecord(e0, st)); for (int i = 0; i < iters; ++i) sbody(); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms2, e0, e1));
        ms /= iters * cnt; ms2 /= iters * cnt;
        printf("  %-20s %5dx%-6d n=%3d  real %7.2f us %6.1f GB/s | stream %7.2f us %6.1f GB/s\n", shapes[si].name, shapes[si].ne0, shapes[si].ne1, cnt,
               ms * 1e3, cb / cnt / (ms * 1e6), ms2 * 1e3, cb / cnt / (ms2 * 1e6));
    }
    return 0;
}
