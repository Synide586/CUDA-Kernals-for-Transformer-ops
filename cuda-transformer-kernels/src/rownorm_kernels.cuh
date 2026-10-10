#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "reduce.cuh"

#ifndef CEIL_DIV
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#endif

constexpr int ROW_BLOCK_THREADS = 256;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int MAX_WARP_COLS = 1024;

// one thread per row: uncoalesced and too few threads
__global__ void softmax_naive(const float* __restrict__ x, float* __restrict__ y,
                              int rows, int cols) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* xr = x + (size_t)r * cols;
  float* yr = y + (size_t)r * cols;
  float m = -CUDART_INF_F;
  for (int c = 0; c < cols; ++c) m = fmaxf(m, xr[c]);
  float s = 0.f;
  for (int c = 0; c < cols; ++c) s += expf(xr[c] - m);
  const float inv = 1.f / s;
  for (int c = 0; c < cols; ++c) yr[c] = expf(xr[c] - m) * inv;
}

// one block per row: coalesced, but x is read three times
__global__ void softmax_block(const float* __restrict__ x, float* __restrict__ y,
                              int rows, int cols) {
  const float* xr = x + (size_t)blockIdx.x * cols;
  float* yr = y + (size_t)blockIdx.x * cols;
  float m = -CUDART_INF_F;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, xr[c]);
  m = block_reduce<true>(m);
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s += expf(xr[c] - m);
  s = block_reduce<false>(s);
  const float inv = 1.f / s;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = expf(xr[c] - m) * inv;
}

// exp(m_old - m_new), with 0 for a lane that has not seen any value yet
__device__ __forceinline__ float rescale(float m_old, float m_new) {
  return (m_old == -CUDART_INF_F) ? 0.f : expf(m_old - m_new);
}

// online softmax: running max and sum in a single pass
__global__ void softmax_warp_online(const float* __restrict__ x, float* __restrict__ y,
                                    int rows, int cols) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + (size_t)row * cols;
  float* yr = y + (size_t)row * cols;

  float m = -CUDART_INF_F, d = 0.f;
  for (int c = lane; c < cols; c += 32) {
    const float v = xr[c];
    const float m_new = fmaxf(m, v);
    d = d * rescale(m, m_new) + expf(v - m_new);
    m = m_new;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float m2 = __shfl_xor_sync(FULL_MASK, m, off);
    const float d2 = __shfl_xor_sync(FULL_MASK, d, off);
    const float m_new = fmaxf(m, m2);
    d = d * rescale(m, m_new) + d2 * rescale(m2, m_new);
    m = m_new;
  }
  const float inv = 1.f / d;
  for (int c = lane; c < cols; c += 32) yr[c] = expf(xr[c] - m) * inv;
}

// row cached in registers: x read once, y written once
template <int VPT>
__global__ void softmax_warp_reg(const float* __restrict__ x, float* __restrict__ y,
                                 int rows, int cols) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + (size_t)row * cols;
  float* yr = y + (size_t)row * cols;

  float v[VPT];
  float m = -CUDART_INF_F;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int c = lane + i * 32;
    v[i] = (c < cols) ? xr[c] : -CUDART_INF_F;
    m = fmaxf(m, v[i]);
  }
  m = warp_max(m);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int c = lane + i * 32;
    v[i] = (c < cols) ? expf(v[i] - m) : 0.f;
    s += v[i];
  }
  s = warp_sum(s);
  const float inv = 1.f / s;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int c = lane + i * 32;
    if (c < cols) yr[c] = v[i] * inv;
  }
}

__global__ void layernorm_naive(const float* __restrict__ x, const float* __restrict__ gamma,
                                const float* __restrict__ beta, float* __restrict__ y,
                                int rows, int cols, float eps) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* xr = x + (size_t)r * cols;
  float* yr = y + (size_t)r * cols;
  float sum = 0.f;
  for (int c = 0; c < cols; ++c) sum += xr[c];
  const float mean = sum / cols;
  float sq = 0.f;
  for (int c = 0; c < cols; ++c) {
    const float d = xr[c] - mean;
    sq += d * d;
  }
  const float rstd = rsqrtf(sq / cols + eps);
  for (int c = 0; c < cols; ++c) yr[c] = (xr[c] - mean) * rstd * gamma[c] + beta[c];
}

__global__ void layernorm_block(const float* __restrict__ x, const float* __restrict__ gamma,
                                const float* __restrict__ beta, float* __restrict__ y,
                                int rows, int cols, float eps) {
  const float* xr = x + (size_t)blockIdx.x * cols;
  float* yr = y + (size_t)blockIdx.x * cols;
  float sum = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += xr[c];
  const float mean = block_reduce<false>(sum) / cols;
  float sq = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float d = xr[c] - mean;
    sq += d * d;
  }
  const float rstd = rsqrtf(block_reduce<false>(sq) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    yr[c] = (xr[c] - mean) * rstd * gamma[c] + beta[c];
}

// row cached in registers; two-pass variance reads registers, not DRAM
template <int VPT>
__global__ void layernorm_warp_reg(const float* __restrict__ x, const float* __restrict__ gamma,
                                   const float* __restrict__ beta, float* __restrict__ y,
                                   int rows, int cols, float eps) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const float* xr = x + (size_t)row * cols;
  float* yr = y + (size_t)row * cols;

  float v[VPT];
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int c = lane + i * 32;
    v[i] = (c < cols) ? xr[c] : 0.f;
    sum += v[i];
  }
  const float mean = warp_sum(sum) / cols;
  float sq = 0.f;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int c = lane + i * 32;
    const float d = (c < cols) ? v[i] - mean : 0.f;
    sq += d * d;
  }
  const float rstd = rsqrtf(warp_sum(sq) / cols + eps);
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const int c = lane + i * 32;
    if (c < cols) yr[c] = (v[i] - mean) * rstd * gamma[c] + beta[c];
  }
}

inline bool warp_reg_supported(int cols) { return cols <= MAX_WARP_COLS; }

inline int pick_vpt(int cols) {
  int vpt = 1;
  while (vpt * 32 < cols) vpt *= 2;
  return vpt;
}

#define DISPATCH_VPT(cols, LAUNCH)                                            \
  switch (pick_vpt(cols)) {                                                   \
    case 1:  { constexpr int VPT = 1;  LAUNCH; } break;                       \
    case 2:  { constexpr int VPT = 2;  LAUNCH; } break;                       \
    case 4:  { constexpr int VPT = 4;  LAUNCH; } break;                       \
    case 8:  { constexpr int VPT = 8;  LAUNCH; } break;                       \
    case 16: { constexpr int VPT = 16; LAUNCH; } break;                       \
    case 32: { constexpr int VPT = 32; LAUNCH; } break;                       \
    default: fprintf(stderr, "cols=%d too large for warp_reg\n", cols); exit(1); \
  }

inline void launch_softmax_naive(const float* x, float* y, int rows, int cols) {
  softmax_naive<<<CEIL_DIV(rows, 256), 256>>>(x, y, rows, cols);
}
inline void launch_softmax_block(const float* x, float* y, int rows, int cols,
                                 cudaStream_t stream = 0) {
  softmax_block<<<rows, ROW_BLOCK_THREADS, 0, stream>>>(x, y, rows, cols);
}
inline void launch_softmax_warp_online(const float* x, float* y, int rows, int cols) {
  softmax_warp_online<<<CEIL_DIV(rows, WARPS_PER_BLOCK), WARPS_PER_BLOCK * 32>>>(x, y, rows, cols);
}
inline void launch_softmax_warp_reg(const float* x, float* y, int rows, int cols,
                                    cudaStream_t stream = 0) {
  const int grid = CEIL_DIV(rows, WARPS_PER_BLOCK), block = WARPS_PER_BLOCK * 32;
  DISPATCH_VPT(cols, (softmax_warp_reg<VPT><<<grid, block, 0, stream>>>(x, y, rows, cols)));
}

inline void launch_layernorm_naive(const float* x, const float* g, const float* b, float* y,
                                   int rows, int cols, float eps) {
  layernorm_naive<<<CEIL_DIV(rows, 256), 256>>>(x, g, b, y, rows, cols, eps);
}
inline void launch_layernorm_block(const float* x, const float* g, const float* b, float* y,
                                   int rows, int cols, float eps, cudaStream_t stream = 0) {
  layernorm_block<<<rows, ROW_BLOCK_THREADS, 0, stream>>>(x, g, b, y, rows, cols, eps);
}
inline void launch_layernorm_warp_reg(const float* x, const float* g, const float* b, float* y,
                                      int rows, int cols, float eps, cudaStream_t stream = 0) {
  const int grid = CEIL_DIV(rows, WARPS_PER_BLOCK), block = WARPS_PER_BLOCK * 32;
  DISPATCH_VPT(cols, (layernorm_warp_reg<VPT><<<grid, block, 0, stream>>>(x, g, b, y, rows, cols, eps)));
}
