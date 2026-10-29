#pragma once

#include <cstdlib>
#include <cstring>

#if defined(GGML_USE_HIP)

static constexpr int FLASHNEXT_GROUP_ENTRIES = 80;

static __global__ void flashnext_expert_groups(const int32_t * ids, int * groups, int slots, int tokens, int stride) {
    const int entry = threadIdx.x;
    const int entries = slots*tokens;
    if (entry >= entries) {
        return;
    }
    const int expert = ids[(entry / slots)*stride + entry % slots];
    int * group = groups + entry*(FLASHNEXT_GROUP_ENTRIES + 1);
    group[0] = 0;
    if (expert < 0) {
        return;
    }
    for (int i = 0; i < entry; ++i) {
        if (ids[(i / slots)*stride + i % slots] == expert) {
            return;
        }
    }
    for (int i = entry; i < entries; ++i) {
        if (ids[(i / slots)*stride + i % slots] == expert) {
            group[++group[0]] = i;
        }
    }
}

struct flashnext_iq2_s {
    int values[8];
    int ls0, ls1;
    float d;

    __device__ flashnext_iq2_s(const void * weights, int block, int iqs) {
        const auto & w = static_cast<const block_iq2_s *>(weights)[block];
        const int qs_packed = get_int_b2(w.qs, iqs/2);
        const auto * qs = reinterpret_cast<const uint8_t *>(&qs_packed);
        const int signs_packed = get_int_b2(w.qs, QK_K/32 + iqs/2);
        const auto * signs = reinterpret_cast<const uint8_t *>(&signs_packed);
        const int qh = w.qh[iqs/2];
        ls0 = w.scales[iqs/2] & 15;
        ls1 = w.scales[iqs/2] >> 4;
        d = __half2float(w.d);
#pragma unroll
        for (int l = 0; l < 8; l += 2) {
            const auto * grid = reinterpret_cast<const int *>(iq2s_grid + (qs[l/2] | ((qh << (8-l)) & 0x300)));
            const int s0 = __vcmpne4(((signs[l/2] & 0x03) << 7) | ((signs[l/2] & 0x0c) << 21), 0);
            const int s1 = __vcmpne4(((signs[l/2] & 0x30) << 3) | ((signs[l/2] & 0xc0) << 17), 0);
            values[l] = __vsub4(grid[0] ^ s0, s0);
            values[l+1] = __vsub4(grid[1] ^ s1, s1);
        }
    }

    __device__ float dot(const block_q8_1 * x, int iqs) const {
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int u = get_int_b4(x[iqs/2].qs, l);
            if (l < 4) {
                s0 = ggml_cuda_dp4a(values[l], u, s0);
            } else {
                s1 = ggml_cuda_dp4a(values[l], u, s1);
            }
        }
        const int sum = (s0*ls0 + s1*ls1 + (s0+s1)/2)/4;
        return (d*__low2float(x[iqs/2].ds))*sum;
    }
};

struct flashnext_iq4_nl {
    int2 values[2];
    float d;

    __device__ flashnext_iq4_nl(const void * weights, int block, int iqs) {
        const auto & w = static_cast<const block_iq4_nl *>(weights)[block];
        d = __half2float(w.d);
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            values[l] = get_int_from_table_16(get_int_b2(w.qs, iqs+l), kvalues_iq4nl);
        }
    }

    __device__ float dot(const block_q8_1 * x, int iqs) const {
        const auto * q = reinterpret_cast<const int *>(x->qs) + iqs;
        int sum = 0;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            sum = ggml_cuda_dp4a(values[l].x, q[l], sum);
            sum = ggml_cuda_dp4a(values[l].y, q[l+4], sum);
        }
        return (d*__low2float(x->ds))*sum;
    }
};

template<ggml_type type, bool gated>
static __global__ void flashnext_moe_grouped(
        const void * weights, const void * gate, const block_q8_1 * x, const int32_t * ids, const int * groups,
        float * dst, int k, int rows, int slots, int ids_stride, int channels_y,
        int row_stride, int expert_stride, int x_token_stride, int x_channel_stride, int dst_token_stride, int dst_channel_stride) {
    constexpr int warp = ggml_cuda_get_physical_warp_size();
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    using decoded = std::conditional_t<type == GGML_TYPE_IQ2_S, flashnext_iq2_s, flashnext_iq4_nl>;
    const int * group = groups + blockIdx.y*(FLASHNEXT_GROUP_ENTRIES + 1);
    const int count = group[0];
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (count == 0 || row >= rows) {
        return;
    }
    const int entry0 = group[1];
    const int expert = ids[(entry0 / slots)*ids_stride + entry0 % slots];
    const int offset = expert*expert_stride + row*row_stride;
    const int lane = threadIdx.x;
    const int iqs = 2*(lane % (qi/2));
    for (int start = 0; start < count; start += 8) {
        const int n = min(8, count-start);
        float sums[8] = {}, gates[8] = {};
        for (int b = lane / (qi/2); b < k/qk; b += 2*warp/qi) {
            const decoded w(weights, offset+b, iqs);
            if constexpr (gated) {
                const decoded g(gate, offset+b, iqs);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (j >= n) { continue; }
                    const int e = group[start+j+1];
                    const auto * act = x + (e / slots)*x_token_stride + ((e % slots) % channels_y)*x_channel_stride + b*(qk/QK8_1);
                    sums[j] += w.dot(act, iqs);
                    gates[j] += g.dot(act, iqs);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (j >= n) { continue; }
                    const int e = group[start+j+1];
                    const auto * act = x + (e / slots)*x_token_stride + ((e % slots) % channels_y)*x_channel_stride + b*(qk/QK8_1);
                    sums[j] += w.dot(act, iqs);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j >= n) { continue; }
            float value = warp_reduce_sum<warp>(sums[j]);
            if constexpr (gated) {
                value *= ggml_cuda_op_silu_single(warp_reduce_sum<warp>(gates[j]));
            }
            if (lane == 0) {
                const int e = group[start+j+1];
                dst[(e / slots)*dst_token_stride + (e % slots)*dst_channel_stride + row] = value;
            }
        }
    }
}

static bool flashnext_moe_launch(ggml_backend_cuda_context & ctx, ggml_type type,
        const void * weights, const block_q8_1 * x, const int32_t * ids, ggml_cuda_mm_fusion_args_device fusion, float * dst,
        int k, int rows, int tokens, int slots, int channels_y, int row_stride, int expert_stride,
        int x_token_stride, int x_channel_stride, int dst_token_stride, int dst_channel_stride, int ids_stride) {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CUDA_FLASHNEXT_MOE");
        return env && std::strcmp(env, "1") == 0;
    }();
    if (!enabled || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_VEGA20 || !ids ||
            (type != GGML_TYPE_IQ2_S && type != GGML_TYPE_IQ4_NL) || tokens < 2 || tokens > 8 || slots < 1 || slots > 10 ||
            fusion.shared_up || fusion.x_bias || fusion.gate_bias || fusion.x_scale || fusion.gate_scale ||
            (fusion.gate && fusion.glu_op != GGML_GLU_OP_SWIGLU)) {
        return false;
    }
    ggml_cuda_pool_alloc<int> groups(ctx.pool(), tokens*slots*(FLASHNEXT_GROUP_ENTRIES + 1));
    flashnext_expert_groups<<<1, 128, 0, ctx.stream()>>>(ids, groups.get(), slots, tokens, ids_stride);
    const dim3 grid((rows+3)/4, tokens*slots);
    const dim3 block(ggml_cuda_info().devices[ctx.device].warp_size, 4);
#define FLASHNEXT_LAUNCH(T, G) flashnext_moe_grouped<T, G><<<grid, block, 0, ctx.stream()>>>(weights, fusion.gate, x, ids, groups.get(), dst, k, rows, slots, ids_stride, channels_y, row_stride, expert_stride, x_token_stride, x_channel_stride, dst_token_stride, dst_channel_stride)
    if (type == GGML_TYPE_IQ2_S) {
        if (fusion.gate) { FLASHNEXT_LAUNCH(GGML_TYPE_IQ2_S, true); }
        else             { FLASHNEXT_LAUNCH(GGML_TYPE_IQ2_S, false); }
    } else {
        if (fusion.gate) { FLASHNEXT_LAUNCH(GGML_TYPE_IQ4_NL, true); }
        else             { FLASHNEXT_LAUNCH(GGML_TYPE_IQ4_NL, false); }
    }
#undef FLASHNEXT_LAUNCH
    return true;
}

#endif
