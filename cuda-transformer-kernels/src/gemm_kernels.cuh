#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// threadIdx.x walks down a column: a warp reads A at stride K (uncoalesced)
__global__ void sgemm_naive_uncoalesced(int M, int N, int K,
                                        const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// threadIdx.x walks along a row: a warp reads 32 consecutive floats of B
__global__ void sgemm_naive_coalesced(int M, int N, int K,
                                      const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// each 32x32 tile of A and B is loaded once into shared memory and reused 32 times
template <int TILE>
__global__ void sgemm_smem_tiled(int M, int N, int K,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  float acc = 0.f;
  for (int t = 0; t < K; t += TILE) {
    As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.f;
    Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// each thread computes an 8x8 block of C: 64 FMAs per 16 shared-memory loads
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_2d_blocktiling(int M, int N, int K, const float* __restrict__ A,
                         const float* __restrict__ B, float* __restrict__ C) {
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);
  static_assert(NUM_THREADS % BK == 0, "A-tile load mapping");
  static_assert(NUM_THREADS % BN == 0, "B-tile load mapping");

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;

  const int thread_row = threadIdx.x / (BN / TN);
  const int thread_col = threadIdx.x % (BN / TN);

  A += block_row * BM * K;
  B += block_col * BN;
  C += block_row * BM * N + block_col * BN;

  const int a_row = threadIdx.x / BK, a_col = threadIdx.x % BK;
  constexpr int A_STRIDE = NUM_THREADS / BK;
  const int b_row = threadIdx.x / BN, b_col = threadIdx.x % BN;
  constexpr int B_STRIDE = NUM_THREADS / BN;

  float acc[TM * TN] = {0.f};
  float reg_a[TM];
  float reg_b[TN];

  for (int bk = 0; bk < K; bk += BK) {
#pragma unroll
    for (int off = 0; off < BM; off += A_STRIDE)
      As[(a_row + off) * BK + a_col] = A[(a_row + off) * K + a_col];
#pragma unroll
    for (int off = 0; off < BK; off += B_STRIDE)
      Bs[(b_row + off) * BN + b_col] = B[(b_row + off) * N + b_col];
    __syncthreads();

    A += BK;
    B += BK * N;

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) reg_a[i] = As[(thread_row * TM + i) * BK + k];
#pragma unroll
      for (int j = 0; j < TN; ++j) reg_b[j] = Bs[k * BN + thread_col * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i * TN + j] += reg_a[i] * reg_b[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      C[(thread_row * TM + i) * N + thread_col * TN + j] = acc[i * TN + j];
}

inline void launch_naive_uncoalesced(int M, int N, int K, const float* A,
                                     const float* B, float* C) {
  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_naive_uncoalesced<<<grid, block>>>(M, N, K, A, B, C);
}

inline void launch_naive_coalesced(int M, int N, int K, const float* A,
                                   const float* B, float* C) {
  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  sgemm_naive_coalesced<<<grid, block>>>(M, N, K, A, B, C);
}

inline void launch_smem_tiled(int M, int N, int K, const float* A,
                              const float* B, float* C) {
  constexpr int TILE = 32;
  dim3 block(TILE, TILE);
  dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
  sgemm_smem_tiled<TILE><<<grid, block>>>(M, N, K, A, B, C);
}

inline void launch_2d_blocktiling(int M, int N, int K, const float* A,
                                  const float* B, float* C, cudaStream_t stream = 0) {
  constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  if (M % BM || N % BN || K % BK) {
    fprintf(stderr, "2d_blocktiling needs M,N %% 128 == 0 and K %% 8 == 0\n");
    exit(EXIT_FAILURE);
  }
  dim3 block((BM * BN) / (TM * TN));
  dim3 grid(N / BN, M / BM);
  sgemm_2d_blocktiling<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(M, N, K, A, B, C);
}
