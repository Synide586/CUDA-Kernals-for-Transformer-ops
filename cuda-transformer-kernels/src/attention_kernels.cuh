#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "reduce.cuh"

constexpr int ATTN_D = 64;
constexpr int ATTN_TILE = 64;

__device__ __forceinline__ float attn_rescale(float m_old, float m_new) {
  return (m_old == -CUDART_INF_F) ? 0.f : expf(m_old - m_new);
}

// unfused path: masked softmax over the materialised N x N score matrix
__global__ void attn_softmax_inplace(float* S, int N, bool causal) {
  float* row = S + (size_t)blockIdx.x * N;
  const int q = blockIdx.x % N;
  const int limit = causal ? q + 1 : N;
  float m = -CUDART_INF_F;
  for (int c = threadIdx.x; c < limit; c += blockDim.x) m = fmaxf(m, row[c]);
  m = block_reduce<true>(m);
  float s = 0.f;
  for (int c = threadIdx.x; c < limit; c += blockDim.x) s += expf(row[c] - m);
  s = block_reduce<false>(s);
  const float inv = 1.f / s;
  for (int c = threadIdx.x; c < N; c += blockDim.x)
    row[c] = (c < limit) ? expf(row[c] - m) * inv : 0.f;
}

// fused v1: one thread per query row, K/V tiles in shared memory
template <int D, int BC>
__global__ void __launch_bounds__(64)
    flash_attn_v1(const float* __restrict__ Q, const float* __restrict__ K,
                  const float* __restrict__ V, float* __restrict__ O, int N,
                  float scale, bool causal) {
  __shared__ float Ks[BC][D];
  __shared__ float Vs[BC][D];

  const size_t base = (size_t)blockIdx.y * N * D;
  const float* Qb = Q + base;
  const float* Kb = K + base;
  const float* Vb = V + base;
  float* Ob = O + base;
  const int q = blockIdx.x * blockDim.x + threadIdx.x;

  float qr[D], o[D];
#pragma unroll
  for (int d = 0; d < D; ++d) {
    qr[d] = Qb[(size_t)q * D + d] * scale;
    o[d] = 0.f;
  }
  float m = -CUDART_INF_F, l = 0.f;

  const int kv_end = causal ? min(N, (int)((blockIdx.x + 1) * blockDim.x)) : N;

  for (int kv0 = 0; kv0 < kv_end; kv0 += BC) {
    for (int idx = threadIdx.x; idx < BC * D; idx += blockDim.x) {
      const int c = idx / D, d = idx % D;
      Ks[c][d] = Kb[(size_t)(kv0 + c) * D + d];
      Vs[c][d] = Vb[(size_t)(kv0 + c) * D + d];
    }
    __syncthreads();

    for (int j = 0; j < BC; ++j) {
      if (causal && kv0 + j > q) break;
      float s = 0.f;
#pragma unroll
      for (int d = 0; d < D; ++d) s += qr[d] * Ks[j][d];
      // new running max: rescale everything accumulated so far
      if (s > m) {
        const float c = attn_rescale(m, s);
        l *= c;
#pragma unroll
        for (int d = 0; d < D; ++d) o[d] *= c;
        m = s;
      }
      const float p = expf(s - m);
      l += p;
#pragma unroll
      for (int d = 0; d < D; ++d) o[d] += p * Vs[j][d];
    }
    __syncthreads();
  }

  const float inv = 1.f / l;
#pragma unroll
  for (int d = 0; d < D; ++d) Ob[(size_t)q * D + d] = o[d] * inv;
}

// fused v2: 64x64 tiles, 4x4 register blocking, 16 lanes share a query row
template <int D, int BR, int BC>
__global__ void __launch_bounds__(256)
    flash_attn_v2(const float* __restrict__ Q, const float* __restrict__ K,
                  const float* __restrict__ V, float* __restrict__ O, int N,
                  float scale, bool causal) {
  static_assert(D == 64 && BR == 64 && BC == 64, "thread mapping assumes 64 x 64 tiles, D = 64");
  // +1 padding keeps transposed column reads on different shared-memory banks
  constexpr int QP = BR + 1;
  constexpr int KP = BC + 1;
  extern __shared__ float smem[];
  float* QsT = smem;
  float* KsT = QsT + D * QP;
  // P reuses the K tile space once scores are computed
  float* Ps = KsT;
  float* Vs = KsT + D * KP;

  const int tid = threadIdx.x;
  const int ty = tid / 16, tx = tid % 16;
  const int q0 = blockIdx.x * BR;
  const size_t base = (size_t)blockIdx.y * N * D;
  const float* Qb = Q + base;
  const float* Kb = K + base;
  const float* Vb = V + base;
  float* Ob = O + base;

  for (int idx = tid; idx < BR * D; idx += 256) {
    const int r = idx / D, k = idx % D;
    QsT[k * QP + r] = Qb[(size_t)(q0 + r) * D + k] * scale;
  }

  float acc[4][4] = {};
  float m[4], l[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) { m[i] = -CUDART_INF_F; l[i] = 0.f; }

  // causal: key tiles fully above the diagonal are skipped
  const int kv_end = causal ? min(N, q0 + BR) : N;

  for (int kv0 = 0; kv0 < kv_end; kv0 += BC) {
    for (int idx = tid; idx < BC * D; idx += 256) {
      const int c = idx / D, k = idx % D;
      const float kval = Kb[(size_t)(kv0 + c) * D + k];
      KsT[k * KP + c] = kval;
      Vs[c * D + k] = Vb[(size_t)(kv0 + c) * D + k];
    }
    __syncthreads();

    float s[4][4] = {};
#pragma unroll 8
    for (int k = 0; k < D; ++k) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = QsT[k * QP + ty * 4 + i];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = KsT[k * KP + tx + 16 * j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) s[i][j] += a[i] * b[j];
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int qi = q0 + ty * 4 + i;
      float mt = -CUDART_INF_F;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (causal && kv0 + tx + 16 * j > qi) s[i][j] = -CUDART_INF_F;
        mt = fmaxf(mt, s[i][j]);
      }
#pragma unroll
      for (int off = 8; off > 0; off >>= 1)
        mt = fmaxf(mt, __shfl_xor_sync(FULL_MASK, mt, off));
      const float m_new = fmaxf(m[i], mt);
      const float c = attn_rescale(m[i], m_new);
      float rs = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float p = (s[i][j] == -CUDART_INF_F) ? 0.f : expf(s[i][j] - m_new);
        Ps[(ty * 4 + i) * KP + tx + 16 * j] = p;
        rs += p;
      }
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) rs += __shfl_xor_sync(FULL_MASK, rs, off);
      l[i] = l[i] * c + rs;
      m[i] = m_new;
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] *= c;
    }
    __syncthreads();

#pragma unroll 8
    for (int c = 0; c < BC; ++c) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = Ps[(ty * 4 + i) * KP + c];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Vs[c * D + tx + 16 * j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float inv = 1.f / l[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) Ob[(size_t)(q0 + ty * 4 + i) * D + tx + 16 * j] = acc[i][j] * inv;
  }
}

inline void check_attn_shape(int N) {
  if (N % ATTN_TILE != 0) {
    fprintf(stderr, "attention kernels need N %% %d == 0 (got %d)\n", ATTN_TILE, N);
    exit(EXIT_FAILURE);
  }
}

inline void launch_flash_v1(const float* Q, const float* K, const float* V, float* O, int BH,
                            int N, bool causal) {
  check_attn_shape(N);
  dim3 grid(N / ATTN_TILE, BH);
  flash_attn_v1<ATTN_D, ATTN_TILE><<<grid, ATTN_TILE>>>(Q, K, V, O, N, 1.0f / sqrtf((float)ATTN_D), causal);
}

inline size_t flash_v2_smem_bytes() {
  return (size_t)(ATTN_D * (ATTN_TILE + 1) * 2 + ATTN_TILE * ATTN_D) * sizeof(float);
}

inline void launch_flash_v2(const float* Q, const float* K, const float* V, float* O, int BH,
                            int N, bool causal, cudaStream_t stream = 0) {
  check_attn_shape(N);
  static bool configured = false;
  const size_t smem = flash_v2_smem_bytes();
  // more than 48 KB of dynamic shared memory requires an explicit opt-in
  if (!configured) {
    cudaError_t e = cudaFuncSetAttribute(flash_attn_v2<ATTN_D, ATTN_TILE, ATTN_TILE>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) {
      fprintf(stderr, "cudaFuncSetAttribute: %s\n", cudaGetErrorString(e));
      exit(EXIT_FAILURE);
    }
    configured = true;
  }
  dim3 grid(N / ATTN_TILE, BH);
  flash_attn_v2<ATTN_D, ATTN_TILE, ATTN_TILE><<<grid, 256, smem, stream>>>(Q, K, V, O, N,
                                                                  1.0f / sqrtf((float)ATTN_D), causal);
}
