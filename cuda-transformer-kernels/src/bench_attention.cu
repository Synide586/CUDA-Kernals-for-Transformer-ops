#include <cstring>
#include <functional>
#include <string>

#include "attention_kernels.cuh"
#include "common.cuh"

using AttnFn = std::function<bool(const float*, const float*, const float*, float*, int, int, bool)>;

struct AttnKernel {
  std::string name;
  AttnFn fn;
};

static std::vector<int> parse_list(const char* s) {
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

static void attention_row_ref(const std::vector<float>& Q, const std::vector<float>& K,
                              const std::vector<float>& V, int N, int bh, int q, bool causal,
                              double* out) {
  const int D = ATTN_D;
  const size_t base = (size_t)bh * N * D;
  const int limit = causal ? q + 1 : N;
  const double scale = 1.0 / std::sqrt((double)D);
  std::vector<double> s(limit);
  double m = -1e300;
  for (int j = 0; j < limit; ++j) {
    double acc = 0.0;
    for (int d = 0; d < D; ++d) acc += (double)Q[base + (size_t)q * D + d] * K[base + (size_t)j * D + d];
    s[j] = acc * scale;
    m = std::max(m, s[j]);
  }
  double l = 0.0;
  for (int j = 0; j < limit; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
  for (int d = 0; d < D; ++d) {
    double acc = 0.0;
    for (int j = 0; j < limit; ++j) acc += s[j] * V[base + (size_t)j * D + d];
    out[d] = acc / l;
  }
}

int main(int argc, char** argv) {
  int BH = 16;
  std::vector<int> seqs = {512, 1024, 2048, 4096, 8192, 16384};
  std::string csv_path, profile_kernel;
  bool profile_causal = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--bh") && i + 1 < argc) BH = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--seqs") && i + 1 < argc) seqs = parse_list(argv[++i]);
    else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
    else if (!strcmp(argv[i], "--profile") && i + 1 < argc) profile_kernel = argv[++i];
    else if (!strcmp(argv[i], "--causal")) profile_causal = true;
    else {
      fprintf(stderr, "usage: %s [--bh 16] [--seqs 1024,4096] [--csv file] [--profile kernel [--causal]]\n", argv[0]);
      return 1;
    }
  }
  const int D = ATTN_D;

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  float* S = nullptr;
  size_t S_bytes = 0;
  // unfused: cuBLAS QK^T -> masked softmax in place -> cuBLAS PV (row-major via column-major)
  auto unfused = [&](const float* Q, const float* K, const float* V, float* O, int bh, int N,
                     bool causal) -> bool {
    const size_t need = (size_t)bh * N * N * sizeof(float);
    if (need != S_bytes) {
      if (S) { cudaFree(S); S = nullptr; S_bytes = 0; }
      if (cudaMalloc(&S, need) != cudaSuccess) {
        cudaGetLastError();
        return false;
      }
      S_bytes = need;
    }
    const float scale = 1.0f / sqrtf((float)D), one = 1.f, zero = 0.f;
    const long long sQK = (long long)N * D, sS = (long long)N * N;
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, D, &scale,
                                           K, D, sQK, Q, D, sQK, &zero, S, N, sS, bh));
    attn_softmax_inplace<<<bh * N, 256>>>(S, N, causal);
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, N, N, &one,
                                           V, D, sQK, S, N, sS, &zero, O, D, sQK, bh));
    return true;
  };
  auto release_S = [&] { if (S) { cudaFree(S); S = nullptr; S_bytes = 0; } };

  std::vector<AttnKernel> kernels = {
      {"unfused_cublas", unfused},
      {"flash_v1", [](auto q, auto k, auto v, auto o, int bh, int n, bool c) { launch_flash_v1(q, k, v, o, bh, n, c); return true; }},
      {"flash_v2", [](auto q, auto k, auto v, auto o, int bh, int n, bool c) { launch_flash_v2(q, k, v, o, bh, n, c); return true; }},
  };

  if (!profile_kernel.empty()) {
    const int N = seqs.back();
    const size_t n = (size_t)BH * N * D, bytes = n * sizeof(float);
    std::vector<float> h(n);
    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, bytes)); CUDA_CHECK(cudaMalloc(&dK, bytes));
    CUDA_CHECK(cudaMalloc(&dV, bytes)); CUDA_CHECK(cudaMalloc(&dO, bytes));
    fill_random(h, 11); CUDA_CHECK(cudaMemcpy(dQ, h.data(), bytes, cudaMemcpyHostToDevice));
    fill_random(h, 12); CUDA_CHECK(cudaMemcpy(dK, h.data(), bytes, cudaMemcpyHostToDevice));
    fill_random(h, 13); CUDA_CHECK(cudaMemcpy(dV, h.data(), bytes, cudaMemcpyHostToDevice));
    bool found = false, ok = true;
    for (auto& k : kernels)
      if (k.name == profile_kernel) { ok = k.fn(dQ, dK, dV, dO, BH, N, profile_causal); found = true; }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!found) fprintf(stderr, "unknown kernel '%s'\n", profile_kernel.c_str());
    else if (!ok) fprintf(stderr, "%s: out of memory at N=%d\n", profile_kernel.c_str(), N);
    else printf("profiled %s at BH=%d N=%d causal=%d\n", profile_kernel.c_str(), BH, N, profile_causal);
    release_S();
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cublasDestroy(handle);
    return (found && ok) ? 0 : 1;
  }

  DeviceInfo dev = query_device();
  size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  printf("Device: %s | %d SMs | %.1f GB memory | peak %.0f GFLOPS FP32\n", dev.name, dev.sms,
         total_b / 1e9, dev.peak_gflops);
  printf("BH = %d, D = %d, FP32. FLOPs = 4*BH*N^2*D (halved for causal: useful work only)\n\n", BH, D);

  FILE* csv = nullptr;
  if (!csv_path.empty()) {
    csv = fopen(csv_path.c_str(), "w");
    if (!csv) { perror("csv"); return 1; }
    fprintf(csv, "kernel,causal,bh,n,ms,gflops,extra_mb,rel_err,status\n");
  }

  printf("%-15s %6s %6s %10s %9s %10s %10s %6s\n", "kernel", "causal", "N", "ms", "GFLOPS",
         "extra_MB", "rel_err", "check");
  printf("-----------------------------------------------------------------------------\n");

  const int SAMPLES = 128;
  bool all_pass = true;
  for (int N : seqs) {
    const size_t n = (size_t)BH * N * D, bytes = n * sizeof(float);
    std::vector<float> hQ(n), hK(n), hV(n), hO(n);
    fill_random(hQ, 11);
    fill_random(hK, 12);
    fill_random(hV, 13);
    for (auto& x : hQ) x *= 2.f;
    for (auto& x : hK) x *= 2.f;

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, bytes)); CUDA_CHECK(cudaMalloc(&dK, bytes));
    CUDA_CHECK(cudaMalloc(&dV, bytes)); CUDA_CHECK(cudaMalloc(&dO, bytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), bytes, cudaMemcpyHostToDevice));

    for (bool causal : {false, true}) {
      std::vector<std::pair<int, int>> rows;
      std::mt19937 gen(N * 2 + causal);
      for (int s = 0; s < SAMPLES; ++s) rows.push_back({(int)(gen() % BH), (int)(gen() % N)});
      rows.push_back({0, 0});
      rows.push_back({BH - 1, N - 1});
      std::vector<double> ref(rows.size() * D);
      for (size_t r = 0; r < rows.size(); ++r)
        attention_row_ref(hQ, hK, hV, N, rows[r].first, rows[r].second, causal, &ref[r * D]);

      const double flops = 4.0 * BH * (double)N * N * D * (causal ? 0.5 : 1.0);

      for (auto& k : kernels) {
        CUDA_CHECK(cudaMemset(dO, 0, bytes));
        const bool ran = k.fn(dQ, dK, dV, dO, BH, N, causal);
        const double extra_mb = (k.name == "unfused_cublas") ? (double)BH * N * N * 4 / 1e6 : 0.0;
        if (!ran) {
          printf("%-15s %6d %6d %10s %9s %10.0f %10s %6s\n", k.name.c_str(), causal, N, "-", "-",
                 extra_mb, "-", "OOM");
          if (csv) fprintf(csv, "%s,%d,%d,%d,,,%.1f,,OOM\n", k.name.c_str(), causal, BH, N, extra_mb);
          continue;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hO.data(), dO, bytes, cudaMemcpyDeviceToHost));

        double max_diff = 0.0, max_ref = 1e-12;
        for (size_t r = 0; r < rows.size(); ++r)
          for (int d = 0; d < D; ++d) {
            const double got = hO[((size_t)rows[r].first * N + rows[r].second) * D + d];
            max_diff = std::max(max_diff, std::fabs(got - ref[r * D + d]));
            max_ref = std::max(max_ref, std::fabs(ref[r * D + d]));
          }
        const double err = max_diff / max_ref;
        const bool pass = err < 1e-4;
        all_pass &= pass;

        const float ms = time_kernel_ms([&] { k.fn(dQ, dK, dV, dO, BH, N, causal); }, 300.f, 3, 50);
        const double gflops = flops / (ms * 1e6);
        printf("%-15s %6d %6d %10.3f %9.1f %10.0f %10.2e %6s\n", k.name.c_str(), causal, N, ms,
               gflops, extra_mb, err, pass ? "PASS" : "FAIL");
        if (csv)
          fprintf(csv, "%s,%d,%d,%d,%.4f,%.2f,%.1f,%.3e,%s\n", k.name.c_str(), causal, BH, N, ms,
                  gflops, extra_mb, err, pass ? "PASS" : "FAIL");
      }
      printf("\n");
    }
    release_S();
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
  }

  if (csv) { fclose(csv); printf("Wrote %s\n", csv_path.c_str()); }
  cublasDestroy(handle);
  printf(all_pass ? "All correctness checks PASSED\n" : "Some checks FAILED\n");
  return all_pass ? 0 : 1;
}
