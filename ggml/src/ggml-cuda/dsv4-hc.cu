#include "common.cuh"
#include "dsv4-hc.cuh"
#include "q8_repack/repack.cuh"
#include "q8_repack/repack-common.cuh"
#include "../ggml-backend-impl.h"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

// gated: post = gate_scale_out*sigmoid(gate_scale_in*post), the fused SCALE -> SIGMOID -> SCALE chain of qwen4exp
template <bool has_comb, bool gated>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float gate_scale_in,
        float gate_scale_out) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float p = post[idst*sp0 + it*sp1];
    if constexpr (gated) {
        p = gate_scale_out * (1.0f / (1.0f + expf(-(gate_scale_in * p))));
    }

    float sum = x[i0*sx0 + it*sx1] * p;
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

// hc_mix keeps F32 math with a fixed summation order: strided thread partials, shuffle tree, warp order
// weights are F32, canonical Q8_0 or repacked Q8_0 (qs plane with padded row stride, then f16 scale plane)
// the weight format is a template parameter so the loads of a block stay branch free and issue early
enum { HC_W_F32 = 0, HC_W_Q8 = 1, HC_W_Q8_RP = 2 };

struct dsv4_hc_w {
    const char * base;
    int64_t      len;   // values per row
    int64_t      ne1;   // rows
    size_t       nb;    // row bytes, F32 and canonical Q8_0
    int          mode;
};

static dsv4_hc_w dsv4_hc_make_w(const ggml_tensor * w) {
    int mode = HC_W_F32;
    if (w->type == GGML_TYPE_Q8_0) {
        mode = w->buffer && ggml_backend_buft_is_cuda_repack(w->buffer->buft) ? HC_W_Q8_RP : HC_W_Q8;
    }
    return { (const char *) w->data, w->ne[0], w->ne[1], w->nb[1], mode };
}

// Q8_0 weights are consumed like mul_mat does: the activation is quantized to int8 per 32 values and dotted with dp4a
// the 4 lanes that hold one 32 value block (consecutive octets) agree on the scale through shuffles
static __device__ __forceinline__ void dsv4_hc_quant8(const float (&z)[8], float & dx, int (&qw)[2]) {
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        amax = fmaxf(amax, fabsf(z[k]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 4));
    dx = amax/127.0f;
    const float inv = amax > 0.0f ? 127.0f/amax : 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int w = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            w |= (__float2int_rn(z[4*i + k]*inv) & 0xff) << (8*k);
        }
        qw[i] = w;
    }
}

// d[t] = dw*dx[t]*dot(int8 weights of octet o in row, qx[t])
template <int MODE, int NT>
static __device__ __forceinline__ void dsv4_hc_dotq(
        const dsv4_hc_w & w, int64_t row, int64_t o, const float (&dx)[NT], const int (&qx)[NT][2], float (&out)[NT]) {
    float dw;
    int wq[2];
    if constexpr (MODE == HC_W_Q8_RP) {
        const int64_t qs_str = repack_qs_stride(w.len);
        const half * sc = (const half *) (w.base + w.ne1*qs_str);
        dw = __half2float(sc[row*(w.len/32) + (o >> 2)]);
        const uint2 u = *(const uint2 *) (w.base + row*qs_str + o*8);
        wq[0] = u.x;
        wq[1] = u.y;
    } else {
        const char * blk = w.base + row*w.nb + (o >> 2)*sizeof(block_q8_0);
        dw = __half2float(*(const half *) blk);
        const uint16_t * qs = (const uint16_t *) (blk + sizeof(half)) + (o & 3)*4;
        wq[0] = qs[0] | (qs[1] << 16);
        wq[1] = qs[2] | (qs[3] << 16);
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        out[t] = dw*dx[t]*(float) ggml_cuda_dp4a(wq[1], qx[t][1], ggml_cuda_dp4a(wq[0], qx[t][0], 0));
    }
}

// out[t] = dot(F32 weights of octet o in row, z[t])
template <int NT>
static __device__ __forceinline__ void dsv4_hc_dotf(
        const dsv4_hc_w & w, int64_t row, int64_t o, const float (&z)[NT][8], float (&out)[NT]) {
    const float4 a = ((const float4 *) (w.base + row*w.nb))[2*o + 0];
    const float4 b = ((const float4 *) (w.base + row*w.nb))[2*o + 1];
    const float v[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            s += v[k]*z[t][k];
        }
        out[t] = s;
    }
}

// stage 0: one block per (token, stream). z = x*w_norm, kept as F32 and as int8 per 32 values (octet j = h*n8 + o)
// rms scale is applied later: the dot is linear in it
template <int BLK>
static __global__ void __launch_bounds__(BLK) dsv4_hc_mix_prep_f32(
        const float * x,
        const float * w_norm,
        float * zf,
        int2  * zq,
        float * zd,
        float * rinv_out,
        int64_t n_embd,
        float eps) {
    constexpr int WS = ggml_cuda_get_physical_warp_size();
    __shared__ float red[BLK/WS];

    const int tid = threadIdx.x;
    const int h   = blockIdx.x % DSV4_HC;
    const int64_t t = blockIdx.x / DSV4_HC;
    const int n8 = n_embd/8;

    float sq = 0.0f;
    for (int o0 = 0; o0 < n8; o0 += BLK) {
        const int or_ = o0 + tid;
        const bool act = or_ < n8;
        const int64_t j = h*n8 + (act ? or_ : 0);
        const float4 a  = ((const float4 *) x)[(t*DSV4_HC*n8 + j)*2 + 0];
        const float4 b  = ((const float4 *) x)[(t*DSV4_HC*n8 + j)*2 + 1];
        const float4 wa = ((const float4 *) w_norm)[j*2 + 0];
        const float4 wb = ((const float4 *) w_norm)[j*2 + 1];
        const float xv[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
        const float wv[8] = { wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w };
        float z[8];
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            s += xv[k]*xv[k];
            z[k] = xv[k]*wv[k];
        }
        float dx;
        int qw[2];
        dsv4_hc_quant8(z, dx, qw);
        if (act) {
            sq += s;
            ((float4 *) zf)[(t*DSV4_HC*n8 + j)*2 + 0] = make_float4(z[0], z[1], z[2], z[3]);
            ((float4 *) zf)[(t*DSV4_HC*n8 + j)*2 + 1] = make_float4(z[4], z[5], z[6], z[7]);
            zq[t*DSV4_HC*n8 + j] = make_int2(qw[0], qw[1]);
            if ((or_ & 3) == 0) {
                zd[(t*DSV4_HC*n8 + j) >> 2] = dx;
            }
        }
    }

    sq = warp_reduce_sum<WS>(sq);
    if (tid % WS == 0) {
        red[tid/WS] = sq;
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int w = 0; w < BLK/WS; ++w) {
            sum += red[w];
        }
        rinv_out[t*DSV4_HC + h] = rsqrtf(sum/(float) n_embd + eps);
    }
}

// stage 1: one block per weight row, one warp per stream: lo = silu(w_down*xn/hc) -> scratch, inject = w_inject*xn/hc -> dst
// lane l owns octets l + WS*i of its stream row, HS = n8/WS octets per lane
template <int MODE, int NT, int HS>
static __device__ __forceinline__ void dsv4_hc_mix_down_row(
        const float * zf,
        const int2  * zq,
        const float * zd,
        const float * rinv,
        const dsv4_hc_w & w,
        const int64_t row,
        const bool is_inject,
        float * dst,
        float * lo_out,
        const int64_t n_embd,
        const int64_t n_lo,
        float (* red)[ggml_cuda_get_physical_warp_size()*(NT + 1)],
        float * sums) {
    constexpr int WS = ggml_cuda_get_physical_warp_size();

    const int lane = threadIdx.x % WS;
    const int h    = threadIdx.x / WS;
    const int64_t n8 = n_embd/8;

    float bin[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        bin[t] = 0.0f;
    }

#pragma unroll
    for (int i = 0; i < HS; ++i) {
        const int64_t j = (int64_t) h*n8 + i*WS + lane;
        float d[NT];
        if constexpr (MODE == HC_W_F32) {
            float z[NT][8];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4 a = ((const float4 *) zf)[((int64_t) t*DSV4_HC*n8 + j)*2 + 0];
                const float4 b = ((const float4 *) zf)[((int64_t) t*DSV4_HC*n8 + j)*2 + 1];
                z[t][0] = a.x; z[t][1] = a.y; z[t][2] = a.z; z[t][3] = a.w;
                z[t][4] = b.x; z[t][5] = b.y; z[t][6] = b.z; z[t][7] = b.w;
            }
            dsv4_hc_dotf<NT>(w, row, j, z, d);
        } else {
            float dx[NT];
            int   qx[NT][2];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int2 q = zq[(int64_t) t*DSV4_HC*n8 + j];
                qx[t][0] = q.x;
                qx[t][1] = q.y;
                dx[t]    = zd[((int64_t) t*DSV4_HC*n8 + j) >> 2];
            }
            dsv4_hc_dotq<MODE, NT>(w, row, j, dx, qx, d);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            bin[t] += d[t];
        }
    }

    // fixed order lane sum through LDS: lane t < NT adds column t of all lanes in lane order
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        red[h][lane*(NT + 1) + t] = bin[t];
    }
#if !defined(GGML_USE_HIP)
    __syncwarp();
#endif // !defined(GGML_USE_HIP)
    if (lane < NT) {
        float c = 0.0f;
        for (int l = 0; l < WS; ++l) {
            c += red[h][l*(NT + 1) + lane];
        }
        sums[h*NT + lane] = c;
    }
    __syncthreads();

    if (threadIdx.x < NT) {
        const int t = threadIdx.x;
        float dd = 0.0f;
#pragma unroll
        for (int hh = 0; hh < DSV4_HC; ++hh) {
            dd += sums[hh*NT + t]*rinv[t*DSV4_HC + hh];
        }
        dd *= 1.0f/(float) DSV4_HC;
        if (!is_inject) {
            lo_out[t*n_lo + row] = dd/(1.0f + expf(-dd));
        } else {
            dst[n_embd*NT + t*DSV4_HC + row] = dd;
        }
    }
}

// blocks [0, n_lo) take the down rows, the next 4 take the inject rows
template <int NT, int HS>
static __global__ void __launch_bounds__(DSV4_HC*ggml_cuda_get_physical_warp_size()) dsv4_hc_mix_down_f32(
        const float * zf,
        const int2  * zq,
        const float * zd,
        const float * rinv,
        const dsv4_hc_w w_down,
        const dsv4_hc_w w_inject,
        float * dst,
        float * lo_out,
        int64_t n_embd,
        int64_t n_lo) {
    constexpr int WS = ggml_cuda_get_physical_warp_size();
    __shared__ float red[DSV4_HC][WS*(NT + 1)];
    __shared__ float sums[DSV4_HC*NT];

    const int64_t row = blockIdx.x;
    if (row >= n_lo) {
        dsv4_hc_mix_down_row<HC_W_F32, NT, HS>(zf, zq, zd, rinv, w_inject, row - n_lo, true, dst, lo_out, n_embd, n_lo, red, sums);
    } else if (w_down.mode == HC_W_Q8_RP) {
        dsv4_hc_mix_down_row<HC_W_Q8_RP, NT, HS>(zf, zq, zd, rinv, w_down, row, false, dst, lo_out, n_embd, n_lo, red, sums);
    } else if (w_down.mode == HC_W_Q8) {
        dsv4_hc_mix_down_row<HC_W_Q8, NT, HS>(zf, zq, zd, rinv, w_down, row, false, dst, lo_out, n_embd, n_lo, red, sums);
    } else {
        dsv4_hc_mix_down_row<HC_W_F32, NT, HS>(zf, zq, zd, rinv, w_down, row, false, dst, lo_out, n_embd, n_lo, red, sums);
    }
}

// stage 2: one embd index per warp. A group of 16 lanes owns one stream row of w_up (octets sub, sub+16, ...).
// gate = w_up*lo, then the gated mean over the streams in stream order
constexpr int DSV4_HC_MIX_UP_MAXIT = 4;
constexpr int DSV4_HC_MIX_UP_BLK   = 1024;
constexpr int DSV4_HC_MIX_MAX_LO   = 512;

template <int MODE, int NT>
static __global__ void __launch_bounds__(DSV4_HC_MIX_UP_BLK) dsv4_hc_mix_up_f32(
        const float * x,
        const float * w_norm,
        const dsv4_hc_w w_up,
        const float * lo,
        const float * rinv,
        float * dst,
        int64_t n_embd,
        int64_t n_lo) {
    constexpr int WS = ggml_cuda_get_physical_warp_size();
    constexpr int G  = WS/16;

    const int n8 = n_lo/8;

    // Q8_0: quantize lo once per block, all warps read it from LDS
    __shared__ int   lo_q[MODE == HC_W_F32 ? 1 : NT*DSV4_HC_MIX_MAX_LO/4];
    __shared__ float lo_d[MODE == HC_W_F32 ? 1 : NT*DSV4_HC_MIX_MAX_LO/32];
    if constexpr (MODE != HC_W_F32) {
        for (int i0 = 0; i0 < NT*n8; i0 += DSV4_HC_MIX_UP_BLK) {
            const int i = i0 + threadIdx.x;
            const bool act = i < NT*n8;
            const int ii = act ? i : 0;
            const float4 a = ((const float4 *) lo)[ii*2 + 0];
            const float4 b = ((const float4 *) lo)[ii*2 + 1];
            const float zv[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
            float dx;
            int qw[2];
            dsv4_hc_quant8(zv, dx, qw);
            if (act) {
                lo_q[ii*2 + 0] = qw[0];
                lo_q[ii*2 + 1] = qw[1];
                if ((ii & 3) == 0) {
                    lo_d[ii >> 2] = dx;
                }
            }
        }
        __syncthreads();
    }

    const int lane = threadIdx.x % WS;
    const int grp  = lane / 16;
    const int sub  = lane % 16;
    const int64_t e = (int64_t) blockIdx.x*(blockDim.x/WS) + threadIdx.x/WS;
    if (e >= n_embd) {
        return;
    }

    float term[DSV4_HC/G][NT];
#pragma unroll
    for (int hp = 0; hp < DSV4_HC/G; ++hp) {
        const int h = hp*G + grp;
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            acc[t] = 0.0f;
        }
#pragma unroll
        for (int it = 0; it < DSV4_HC_MIX_UP_MAXIT; ++it) {
            const int or_  = sub + 16*it;
            const bool act = or_ < n8;
            if (!act) {
                continue;
            }
            const int o = or_;
            float d[NT];
            if constexpr (MODE == HC_W_F32) {
                float z[NT][8];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const float4 a = ((const float4 *) (lo + t*n_lo))[o*2 + 0];
                    const float4 b = ((const float4 *) (lo + t*n_lo))[o*2 + 1];
                    z[t][0] = a.x; z[t][1] = a.y; z[t][2] = a.z; z[t][3] = a.w;
                    z[t][4] = b.x; z[t][5] = b.y; z[t][6] = b.z; z[t][7] = b.w;
                }
                dsv4_hc_dotf<NT>(w_up, h*n_embd + e, o, z, d);
            } else {
                float dx[NT];
                int   qx[NT][2];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    qx[t][0] = lo_q[(t*n8 + o)*2 + 0];
                    qx[t][1] = lo_q[(t*n8 + o)*2 + 1];
                    dx[t]    = lo_d[(t*n8 + o) >> 2];
                }
                dsv4_hc_dotq<MODE, NT>(w_up, h*n_embd + e, o, dx, qx, d);
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[t] += d[t];
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float g  = warp_reduce_sum<16>(acc[t]);
            const float xn = x[(int64_t) t*DSV4_HC*n_embd + h*n_embd + e]*rinv[t*DSV4_HC + h]*w_norm[h*n_embd + e];
            term[hp][t] = xn*(1.0f/(1.0f + expf(-g)));
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float tt[DSV4_HC];
#pragma unroll
        for (int hp = 0; hp < DSV4_HC/G; ++hp) {
#pragma unroll
            for (int gi = 0; gi < G; ++gi) {
                tt[hp*G + gi] = __shfl_sync(0xffffffff, term[hp][t], gi*16, WS);
            }
        }
        if (lane == 0) {
            float sum = 0.0f;
#pragma unroll
            for (int h = 0; h < DSV4_HC; ++h) {
                sum += tt[h];
            }
            dst[t*n_embd + e] = sum*(1.0f/(float) DSV4_HC);
        }
    }
}

template <int NT, int HS>
static void dsv4_hc_mix_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * w_norm   = dst->src[1];
    const ggml_tensor * w_down   = dst->src[2];
    const ggml_tensor * w_inject = dst->src[3];
    const ggml_tensor * w_up     = dst->src[4];

    const int64_t n_embd = x->ne[0];
    const int64_t n_lo   = w_down->ne[1];
    const int64_t n_rows = n_lo + DSV4_HC;
    const float   eps    = ggml_get_op_params_f32(dst, 0);

    const int64_t n8 = n_embd/8;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), NT*(DSV4_HC*n_embd + 2*DSV4_HC*n8 + DSV4_HC*n8/4 + DSV4_HC + n_lo));
    float * zf   = scratch.get();
    int2  * zq   = (int2 *) (zf + NT*DSV4_HC*n_embd);
    float * zd   = (float *) (zq + NT*DSV4_HC*n8);
    float * rinv = zd + NT*DSV4_HC*n8/4;
    float * lo   = rinv + NT*DSV4_HC;

    dsv4_hc_mix_prep_f32<256><<<NT*DSV4_HC, 256, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) w_norm->data, zf, zq, zd, rinv, n_embd, eps);
    CUDA_CHECK(cudaGetLastError());

    dsv4_hc_mix_down_f32<NT, HS><<<n_rows, DSV4_HC*ggml_cuda_info().devices[ctx.device].warp_size, 0, ctx.stream()>>>(
            zf, zq, zd, rinv, dsv4_hc_make_w(w_down), dsv4_hc_make_w(w_inject),
            (float *) dst->data, lo, n_embd, n_lo);
    CUDA_CHECK(cudaGetLastError());

    const int wpb = DSV4_HC_MIX_UP_BLK/ggml_cuda_info().devices[ctx.device].warp_size;
    const dsv4_hc_w wu = dsv4_hc_make_w(w_up);
    auto up = wu.mode == HC_W_Q8_RP ? dsv4_hc_mix_up_f32<HC_W_Q8_RP, NT>
            : wu.mode == HC_W_Q8    ? dsv4_hc_mix_up_f32<HC_W_Q8, NT>
                                    : dsv4_hc_mix_up_f32<HC_W_F32, NT>;
    up<<<(n_embd + wpb - 1)/wpb, DSV4_HC_MIX_UP_BLK, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) w_norm->data, wu,
            lo, rinv, (float *) dst->data, n_embd, n_lo);
    CUDA_CHECK(cudaGetLastError());
}

template <int HS>
static void dsv4_hc_mix_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[2]) {
        case 1: dsv4_hc_mix_launch<1, HS>(ctx, dst); break;
        case 2: dsv4_hc_mix_launch<2, HS>(ctx, dst); break;
        case 3: dsv4_hc_mix_launch<3, HS>(ctx, dst); break;
        case 4: dsv4_hc_mix_launch<4, HS>(ctx, dst); break;
        case 5: dsv4_hc_mix_launch<5, HS>(ctx, dst); break;
        case 6: dsv4_hc_mix_launch<6, HS>(ctx, dst); break;
        case 7: dsv4_hc_mix_launch<7, HS>(ctx, dst); break;
        case 8: dsv4_hc_mix_launch<8, HS>(ctx, dst); break;
        default: GGML_ABORT("unsupported n_tokens");
    }
}

void ggml_cuda_op_dsv4_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // HS = octets per stream row / warp size
    const int ws = ggml_cuda_info().devices[ctx.device].warp_size;
    switch (dst->src[0]->ne[0]/(8*ws)) {
        case 1: dsv4_hc_mix_dispatch<1>(ctx, dst); break;
        case 2: dsv4_hc_mix_dispatch<2>(ctx, dst); break;
        case 3: dsv4_hc_mix_dispatch<3>(ctx, dst); break;
        case 4: dsv4_hc_mix_dispatch<4>(ctx, dst); break;
        case 5: dsv4_hc_mix_dispatch<5>(ctx, dst); break;
        case 6: dsv4_hc_mix_dispatch<6>(ctx, dst); break;
        case 7: dsv4_hc_mix_dispatch<7>(ctx, dst); break;
        case 8: dsv4_hc_mix_dispatch<8>(ctx, dst); break;
        default: GGML_ABORT("unsupported n_embd");
    }
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * comb     = dst->src[3];

    // when fused, read the raw post weights from before the SCALE -> SIGMOID -> SCALE chain
    const bool          gated = gate_scale_in != nullptr;
    const ggml_tensor * post  = gated ? gate_scale_in->src[0] : dst->src[2];

    const float gate_in  = gated ? ggml_get_op_params_f32(gate_scale_in,  0) : 1.0f;
    const float gate_out = gated ? ggml_get_op_params_f32(gate_scale_out, 0) : 1.0f;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? (gated ? dsv4_hc_post_f32<true,  true> : dsv4_hc_post_f32<true,  false>)
                       : (gated ? dsv4_hc_post_f32<false, true> : dsv4_hc_post_f32<false, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            gate_in, gate_out);
}
