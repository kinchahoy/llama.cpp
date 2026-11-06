#include "sgemm-f32.cuh"

#include <cstring>

// Small tiled F32 GEMM for skinny src0 (HC inject, ssm alpha/beta: M < 128): dst[m, n] = sum_k src0[m, k] * src1[n, k]
// Both inputs are contiguous along k. Each thread makes a 4x4 output tile.
// Split-K gives enough waves for 512x512 outputs. rocBLAS is slow for these shapes.
// GCN only. The sum order differs from rocBLAS; Flash Next KLD stays at its noise floor. GGML_CUDA_SGEMM_F32=0 disables.

#define SGEMM_BK 16
#define SGEMM_PAD 4

template <int BM, int BN>
static __global__ void __launch_bounds__((BM/4)*(BN/4)) sgemm_f32_kernel(
        const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ D,
        const int M, const int N, const int K, const int k_per_split, const int64_t lda, const int64_t ldb, const int64_t ldd) {
    constexpr int NT = (BM/4)*(BN/4);
    __shared__ float As[SGEMM_BK][BM + SGEMM_PAD];
    __shared__ float Bs[SGEMM_BK][BN + SGEMM_PAD];

    const int tid = threadIdx.x;
    const int tm = tid % (BM/4);
    const int tn = tid / (BM/4);
    const int m0 = blockIdx.x*BM;
    const int n0 = blockIdx.y*BN;

    const int k_begin = blockIdx.z*k_per_split;
    const int k_end = min(K, k_begin + k_per_split);
    D += (int64_t) blockIdx.z*N*ldd;

    float acc[4][4] = {};

    for (int k0 = k_begin; k0 < k_end; k0 += SGEMM_BK) {
        for (int i = tid; i < BM*(SGEMM_BK/4); i += NT) {
            const int r = i / (SGEMM_BK/4);
            const int c = (i % (SGEMM_BK/4))*4;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (m0 + r < M) {
                v = *(const float4 *) (A + (m0 + r)*lda + k0 + c);
            }
            As[c + 0][r] = v.x; As[c + 1][r] = v.y; As[c + 2][r] = v.z; As[c + 3][r] = v.w;
        }
        for (int i = tid; i < BN*(SGEMM_BK/4); i += NT) {
            const int r = i / (SGEMM_BK/4);
            const int c = (i % (SGEMM_BK/4))*4;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (n0 + r < N) {
                v = *(const float4 *) (B + (n0 + r)*ldb + k0 + c);
            }
            Bs[c + 0][r] = v.x; Bs[c + 1][r] = v.y; Bs[c + 2][r] = v.z; Bs[c + 3][r] = v.w;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < SGEMM_BK; ++k) {
            const float4 a = *(const float4 *) &As[k][tm*4];
            const float4 b = *(const float4 *) &Bs[k][tn*4];
            const float av[4] = {a.x, a.y, a.z, a.w};
            const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[j][i] = fmaf(av[i], bv[j], acc[j][i]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n = n0 + tn*4 + j;
        if (n >= N) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = m0 + tm*4 + i;
            if (m < M) {
                D[n*ldd + m] = acc[j][i];
            }
        }
    }
}

static __global__ void sgemm_f32_reduce(const float * __restrict__ P, float * __restrict__ D, const int M, const int N, const int split, const int64_t ldd) {
    const int m = blockIdx.x*blockDim.x + threadIdx.x;
    const int n = blockIdx.y;
    if (m >= M) {
        return;
    }
    float sum = 0.0f;
    for (int z = 0; z < split; ++z) {
        sum += P[((int64_t) z*N + n)*ldd + m];
    }
    D[n*ldd + m] = sum;
}

bool ggml_cuda_should_use_sgemm_f32(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_SGEMM_F32");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled && GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)
        && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1
        && src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float)
        && src0->ne[0] % SGEMM_BK == 0
        && src0->nb[1] % 16 == 0 && src1->nb[1] % 16 == 0
        && (uintptr_t) src0->data % 16 == 0 && (uintptr_t) src1->data % 16 == 0
        && ggml_is_contiguous(dst)
        && src0->ne[0] == src1->ne[0]
        && src1->ne[1] > 16 && src1->ne[1] <= 65535 && src0->ne[1] < 128
        && (strstr(src0->name, "hc_") || strstr(src0->name, "ssm_alpha") || strstr(src0->name, "ssm_beta"));
}

template <int BM, int BN>
static void sgemm_f32_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const int split) {
    const int M = src0->ne[1];
    const int N = src1->ne[1];
    const int K = src0->ne[0];
    const int64_t ldd = dst->nb[1]/sizeof(float);
    const int k_per_split = (K/SGEMM_BK + split - 1)/split*SGEMM_BK;
    const int n_split = (K + k_per_split - 1)/k_per_split;
    const dim3 grid((M + BM - 1)/BM, (N + BN - 1)/BN, n_split);

    ggml_cuda_pool_alloc<float> partial(ctx.pool());
    float * out = (float *) dst->data;
    if (n_split > 1) {
        partial.alloc((size_t) n_split*N*ldd);
        out = partial.get();
    }
    sgemm_f32_kernel<BM, BN><<<grid, (BM/4)*(BN/4), 0, ctx.stream()>>>(
        (const float *) src0->data, (const float *) src1->data, out,
        M, N, K, k_per_split, src0->nb[1]/sizeof(float), src1->nb[1]/sizeof(float), ldd);
    CUDA_CHECK(cudaGetLastError());
    if (n_split > 1) {
        sgemm_f32_reduce<<<dim3((M + 127)/128, N), 128, 0, ctx.stream()>>>(partial.get(), (float *) dst->data, M, N, n_split, ldd);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_sgemm_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    sgemm_f32_launch<32, 32>(ctx, src0, src1, dst, 8);
}
