#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_), \
              __FILE__, __LINE__);                                            \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t st_ = (call);                                              \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__,       \
              __LINE__);                                                      \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

inline void fill_random(std::vector<float>& v, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = dist(gen);
}

inline float max_rel_error(const std::vector<float>& out,
                           const std::vector<float>& ref) {
  float max_diff = 0.f, max_ref = 1e-6f;
  for (size_t i = 0; i < ref.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(out[i] - ref[i]));
    max_ref = std::max(max_ref, std::fabs(ref[i]));
  }
  return max_diff / max_ref;
}

template <typename Launch>
float time_kernel_ms(Launch launch, float target_ms = 300.f, int min_reps = 3,
                     int max_reps = 100) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float probe_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&probe_ms, start, stop));

  int reps = (int)(target_ms / std::max(probe_ms, 1e-3f));
  reps = std::min(std::max(reps, min_reps), max_reps);

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < reps; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float total_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms / reps;
}

inline double gemm_gflops(int M, int N, int K, float ms) {
  return 2.0 * M * N * K / (ms * 1e6);
}

inline int fp32_cores_per_sm(int major, int minor) {
  if (major == 6) return minor == 0 ? 64 : 128;
  if (major == 7) return 64;
  if (major == 8) return minor == 0 ? 64 : 128;
  if (major >= 9) return 128;
  return 64;
}

struct DeviceInfo {
  char name[256];
  int sms, cc_major, cc_minor;
  double peak_gflops;
  double peak_bw_gbs;
};

inline DeviceInfo query_device(int dev = 0) {
  DeviceInfo d{};
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  snprintf(d.name, sizeof(d.name), "%s", p.name);
  d.sms = p.multiProcessorCount;
  d.cc_major = p.major;
  d.cc_minor = p.minor;
  int clk_khz = 0, mem_clk_khz = 0, bus_bits = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
  d.peak_gflops =
      2.0 * d.sms * fp32_cores_per_sm(p.major, p.minor) * (clk_khz * 1e3) / 1e9;
  d.peak_bw_gbs = 2.0 * (mem_clk_khz * 1e3) * (bus_bits / 8.0) / 1e9;
  return d;
}
