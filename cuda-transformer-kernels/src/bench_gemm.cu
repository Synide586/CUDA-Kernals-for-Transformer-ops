#include <cstring>
#include <functional>
#include <string>

#include "common.cuh"
#include "gemm_kernels.cuh"

using GemmFn = std::function<void(int, int, int, const float*, const float*, float*)>;

struct Kernel {
  std::string name;
  GemmFn fn;
};

static void cpu_gemm(int M, int N, int K, const std::vector<float>& A,
                     const std::vector<float>& B, std::vector<float>& C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += (double)A[i * K + k] * B[k * N + j];
      C[i * N + j] = (float)acc;
    }
}

static std::vector<int> parse_sizes(const char* s) {
  std::vector<int> out;
  std::string str(s);
  size_t pos = 0;
  while (pos < str.size()) {
    size_t next = str.find(',', pos);
    if (next == std::string::npos) next = str.size();
    out.push_back(std::stoi(str.substr(pos, next - pos)));
    pos = next + 1;
  }
  return out;
}

int main(int argc, char** argv) {
  std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
  std::string csv_path, profile_kernel;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--sizes") && i + 1 < argc) sizes = parse_sizes(argv[++i]);
    else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
    else if (!strcmp(argv[i], "--profile") && i + 1 < argc) profile_kernel = argv[++i];
    else {
      fprintf(stderr, "usage: %s [--sizes 256,1024] [--csv file] [--profile kernel]\n", argv[0]);
      return 1;
    }
  }

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // cuBLAS is column-major: row-major C = A*B is computed as C^T = B^T * A^T
  auto cublas_sgemm = [&](int M, int N, int K, const float* A, const float* B, float* C) {
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             B, N, A, K, &beta, C, N));
  };

  std::vector<Kernel> kernels = {
      {"naive_uncoalesced", launch_naive_uncoalesced},
      {"naive_coalesced", launch_naive_coalesced},
      {"smem_tiled", launch_smem_tiled},
      {"2d_blocktiling", [](int M, int N, int K, const float* A, const float* B, float* C) { launch_2d_blocktiling(M, N, K, A, B, C); }},
      {"2d_blocktiling_occ", [](int M, int N, int K, const float* A, const float* B, float* C) { launch_2d_blocktiling_occ(M, N, K, A, B, C); }},
      {"2d_blocktiling_v2", [](int M, int N, int K, const float* A, const float* B, float* C) { launch_2d_blocktiling_v2(M, N, K, A, B, C); }},
      {"cublas", cublas_sgemm},
  };

  if (!profile_kernel.empty()) {
    int n = sizes.back();
    size_t bytes = (size_t)n * n * sizeof(float);
    std::vector<float> hA((size_t)n * n), hB((size_t)n * n);
    fill_random(hA, 1);
    fill_random(hB, 2);
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    bool found = false;
    for (auto& k : kernels)
      if (k.name == profile_kernel) {
        k.fn(n, n, n, dA, dB, dC);
        found = true;
      }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!found) fprintf(stderr, "unknown kernel '%s'\n", profile_kernel.c_str());
    else printf("profiled %s at n=%d\n", profile_kernel.c_str(), n);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(handle);
    return found ? 0 : 1;
  }

  DeviceInfo dev = query_device();
  printf("Device: %s | %d SMs | sm_%d%d\n", dev.name, dev.sms, dev.cc_major, dev.cc_minor);
  printf("Theoretical peak: %.0f GFLOPS FP32 (boost clock), %.0f GB/s DRAM\n\n",
         dev.peak_gflops, dev.peak_bw_gbs);

  FILE* csv = nullptr;
  if (!csv_path.empty()) {
    csv = fopen(csv_path.c_str(), "w");
    if (!csv) { perror("csv"); return 1; }
    fprintf(csv, "kernel,size,ms,gflops,pct_cublas,pct_peak,rel_err,status\n");
  }

  printf("%-18s %6s %10s %10s %9s %8s %10s %6s\n", "kernel", "size", "ms",
         "GFLOPS", "%cuBLAS", "%peak", "rel_err", "check");
  printf("-----------------------------------------------------------------------------------\n");

  bool all_pass = true;
  for (int n : sizes) {
    const int M = n, N = n, K = n;
    size_t bytes = (size_t)n * n * sizeof(float);
    std::vector<float> hA((size_t)n * n), hB((size_t)n * n);
    std::vector<float> hRef((size_t)n * n), hOut((size_t)n * n);
    fill_random(hA, 1);
    fill_random(hB, 2);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    cublas_sgemm(M, N, K, dA, dB, dC);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC, bytes, cudaMemcpyDeviceToHost));
    if (n <= 512) {
      std::vector<float> hCpu((size_t)n * n);
      cpu_gemm(M, N, K, hA, hB, hCpu);
      float e = max_rel_error(hRef, hCpu);
      printf("[ref check] cuBLAS vs CPU at n=%d: rel_err=%.2e %s\n", n, e,
             e < 1e-4f ? "OK" : "MISMATCH");
    }

    float cublas_ms = time_kernel_ms([&] { cublas_sgemm(M, N, K, dA, dB, dC); });

    for (auto& k : kernels) {
      CUDA_CHECK(cudaMemset(dC, 0, bytes));
      k.fn(M, N, K, dA, dB, dC);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(hOut.data(), dC, bytes, cudaMemcpyDeviceToHost));
      float err = max_rel_error(hOut, hRef);
      bool pass = err < 1e-4f;
      all_pass &= pass;

      float ms = (k.name == "cublas") ? cublas_ms
                                      : time_kernel_ms([&] { k.fn(M, N, K, dA, dB, dC); });
      double gf = gemm_gflops(M, N, K, ms);
      double pct_cublas = 100.0 * cublas_ms / ms;
      double pct_peak = 100.0 * gf / dev.peak_gflops;

      printf("%-18s %6d %10.3f %10.1f %8.1f%% %7.1f%% %10.2e %6s\n", k.name.c_str(), n,
             ms, gf, pct_cublas, pct_peak, err, pass ? "PASS" : "FAIL");
      if (csv)
        fprintf(csv, "%s,%d,%.4f,%.2f,%.2f,%.2f,%.3e,%s\n", k.name.c_str(), n, ms, gf,
                pct_cublas, pct_peak, err, pass ? "PASS" : "FAIL");
    }
    printf("\n");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
  }

  if (csv) { fclose(csv); printf("Wrote %s\n", csv_path.c_str()); }
  cublasDestroy(handle);
  printf(all_pass ? "All correctness checks PASSED\n" : "Some checks FAILED\n");
  return all_pass ? 0 : 1;
}
