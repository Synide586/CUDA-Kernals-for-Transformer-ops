#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

#define FULL_MASK 0xffffffffu

// butterfly reduction: after 5 shuffles every lane holds the result
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(FULL_MASK, v, off);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, off));
  return v;
}

template <bool IS_MAX>
__device__ float block_reduce(float v) {
  __shared__ float partial[32];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;
  const float identity = IS_MAX ? -CUDART_INF_F : 0.f;

  v = IS_MAX ? warp_max(v) : warp_sum(v);
  if (lane == 0) partial[wid] = v;
  __syncthreads();

  if (wid == 0) {
    v = (lane < num_warps) ? partial[lane] : identity;
    v = IS_MAX ? warp_max(v) : warp_sum(v);
    if (lane == 0) partial[0] = v;
  }
  __syncthreads();
  const float result = partial[0];
  __syncthreads();
  return result;
}
