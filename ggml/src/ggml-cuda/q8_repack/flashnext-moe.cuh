#pragma once

static __global__ void flashnext_iq4_repacked_grouped(
        const uint8_t * weights, const block_q8_1 * x, float * dst,
        uint32_t k, uint32_t rows, const int32_t * src_ids, const int32_t * dst_ids,
        const int32_t * bounds, uint32_t experts, size_t expert_stride, uint32_t xs, uint32_t ys) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    constexpr int lanes = 16;
    constexpr int rows_per_block = 8;
    const uint32_t a = blockIdx.y;
    const uint32_t expert = expert_bounds_search(bounds, experts, a);
    if ((a - bounds[expert]) % 8 != 0) {
        return;
    }
    const int count = min(8u, bounds[expert+1] - a);
    const uint32_t row = blockIdx.x*rows_per_block + threadIdx.x/lanes;
    if (row >= rows) {
        return;
    }
    const int lane = threadIdx.x % lanes;
    const rp_traits<GGML_TYPE_IQ4_NL>::geom geom(k, rows);
    const uint8_t * w = weights + expert*expert_stride;
    float sums[8] = {};
    for (uint32_t sb = lane; sb < k/32; sb += lanes) {
        uint4 lo, hi;
        uint16_t ds;
        rp_traits<GGML_TYPE_IQ4_NL>::load_w(w, geom, row, sb, lo, hi, ds);
        const float dw = rp_traits<GGML_TYPE_IQ4_NL>::scale(ds);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j >= count) { continue; }
            const auto * act = x + (size_t) src_ids[a+j]*xs + sb;
            const auto * q = reinterpret_cast<const int *>(act->qs);
            int dot = mmvq_dp4a_u4(lo, q[0], q[1], q[2], q[3]);
            dot += mmvq_dp4a_u4(hi, q[4], q[5], q[6], q[7]);
            sums[j] += dw*__low2float(act->ds)*float(dot);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j >= count) { continue; }
        const float value = rp_warp_reduce_sum<lanes>(sums[j]);
        if (lane == 0) {
            dst[(size_t) dst_ids[a+j]*ys + row] = value;
        }
    }
#else
    GGML_UNUSED_VARS(weights, x, dst, k, rows, src_ids, dst_ids, bounds, experts, expert_stride, xs, ys);
    NO_DEVICE_CODE;
#endif
}

static bool flashnext_repacked_moe_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_FLASHNEXT_MOE");
        return env && strcmp(env, "1") == 0;
    }();
    return enabled;
}
