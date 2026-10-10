#include <cstring>
#include <functional>
#include <string>

#include "common.cuh"
#include "rownorm_kernels.cuh"

static const float EPS = 1e-5f;

static void softmax_ref(const std::vector<float>& x, std::vector<float>& y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = &x[(size_t)r * cols];
    double m = -1e300, s = 0.0;
    for (int c = 0; c < cols; ++c) m = std::max(m, (double)xr[c]);
    for (int c = 0; c < cols; ++c) s += std::exp((double)xr[c] - m);
    for (int c = 0; c < cols; ++c) y[(size_t)r * cols + c] = (float)(std::exp((double)xr[c] - m) / s);
  }
}

static void layernorm_ref(const std::vector<float>& x, const std::vector<float>& g,
                          const std::vector<float>& b, std::vector<float>& y, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = &x[(size_t)r * cols];
    double mean = 0.0, var = 0.0;
    for (int c = 0; c < cols; ++c) mean += xr[c];
    mean /= cols;
    for (int c = 0; c < cols; ++c) var += (xr[c] - mean) * (xr[c] - mean);
    var /= cols;
    const double rstd = 1.0 / std::sqrt(var + EPS);
    for (int c = 0; c < cols; ++c)
      y[(size_t)r * cols + c] = (float)((xr[c] - mean) * rstd * g[c] + b[c]);
  }
}

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

struct RowKernel {
  std::string op, name;
  std::function<void(const float*, const float*, const float*, float*, int, int)> fn;
  bool needs_small_cols;
};

int main(int argc, char** argv) {
  int rows = 8192;
  std::vector<int> cols_list = {128, 256, 512, 768, 1024, 2048, 4096};
  std::string csv_path, profile_kernel;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cols") && i + 1 < argc) cols_list = parse_list(argv[++i]);
    else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
    else if (!strcmp(argv[i], "--profile") && i + 1 < argc) profile_kernel = argv[++i];
    else {
      fprintf(stderr, "usage: %s [--rows R] [--cols 768,1024] [--csv file] [--profile kernel]\n", argv[0]);
      return 1;
    }
  }

  std::vector<RowKernel> kernels = {
      {"softmax", "softmax_naive", [](auto x, auto, auto, auto y, int r, int c) { launch_softmax_naive(x, y, r, c); }, false},
      {"softmax", "softmax_block", [](auto x, auto, auto, auto y, int r, int c) { launch_softmax_block(x, y, r, c); }, false},
      {"softmax", "softmax_warp_online", [](auto x, auto, auto, auto y, int r, int c) { launch_softmax_warp_online(x, y, r, c); }, false},
      {"softmax", "softmax_warp_reg", [](auto x, auto, auto, auto y, int r, int c) { launch_softmax_warp_reg(x, y, r, c); }, true},
      {"layernorm", "layernorm_naive", [](auto x, auto g, auto b, auto y, int r, int c) { launch_layernorm_naive(x, g, b, y, r, c, EPS); }, false},
      {"layernorm", "layernorm_block", [](auto x, auto g, auto b, auto y, int r, int c) { launch_layernorm_block(x, g, b, y, r, c, EPS); }, false},
      {"layernorm", "layernorm_warp_reg", [](auto x, auto g, auto b, auto y, int r, int c) { launch_layernorm_warp_reg(x, g, b, y, r, c, EPS); }, true},
  };

  if (!profile_kernel.empty()) {
    const int cols = cols_list.back();
    const size_t n = (size_t)rows * cols, bytes = n * sizeof(float);
    std::vector<float> hx(n), hg(cols, 1.f), hb(cols, 0.f);
    fill_random(hx, 7);
    float *dx, *dy, *dg, *db;
    CUDA_CHECK(cudaMalloc(&dx, bytes)); CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMalloc(&dg, cols * sizeof(float))); CUDA_CHECK(cudaMalloc(&db, cols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg, hg.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    bool found = false;
    for (auto& k : kernels)
      if (k.name == profile_kernel) {
        if (k.needs_small_cols && !warp_reg_supported(cols)) {
          fprintf(stderr, "%s supports cols <= %d\n", k.name.c_str(), MAX_WARP_COLS);
          return 1;
        }
        k.fn(dx, dg, db, dy, rows, cols);
        found = true;
      }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!found) fprintf(stderr, "unknown kernel '%s'\n", profile_kernel.c_str());
    else printf("profiled %s at %d x %d\n", profile_kernel.c_str(), rows, cols);
    cudaFree(dx); cudaFree(dy); cudaFree(dg); cudaFree(db);
    return found ? 0 : 1;
  }

  DeviceInfo dev = query_device();
  printf("Device: %s | %d SMs | theoretical DRAM bandwidth %.0f GB/s\n", dev.name, dev.sms, dev.peak_bw_gbs);
  printf("rows = %d, FP32. Effective bandwidth = (read x + write y) / time\n\n", rows);

  FILE* csv = nullptr;
  if (!csv_path.empty()) {
    csv = fopen(csv_path.c_str(), "w");
    if (!csv) { perror("csv"); return 1; }
    fprintf(csv, "op,kernel,rows,cols,ms,gbs,pct_peak,pct_memcpy,rel_err,status\n");
  }

  printf("%-21s %6s %9s %8s %7s %8s %10s %6s\n", "kernel", "cols", "ms", "GB/s", "%peak", "%memcpy", "rel_err", "check");
  printf("-------------------------------------------------------------------------------\n");

  bool all_pass = true;
  for (int cols : cols_list) {
    const size_t n = (size_t)rows * cols, bytes = n * sizeof(float);
    const double traffic = 2.0 * bytes;

    std::vector<float> hx(n), hy(n), hsm(n), hln(n), hg(cols), hb(cols);
    fill_random(hx, 7);
    for (auto& v : hx) v *= 8.f;
    fill_random(hg, 8);
    fill_random(hb, 9);
    for (auto& v : hg) v = 1.f + 0.1f * v;
    softmax_ref(hx, hsm, rows, cols);
    layernorm_ref(hx, hg, hb, hln, rows, cols);

    float *dx, *dy, *dg, *db;
    CUDA_CHECK(cudaMalloc(&dx, bytes)); CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMalloc(&dg, cols * sizeof(float))); CUDA_CHECK(cudaMalloc(&db, cols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg, hg.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    // device-to-device memcpy of the same size = practical bandwidth ceiling
    const float copy_ms = time_kernel_ms([&] {
      CUDA_CHECK(cudaMemcpyAsync(dy, dx, bytes, cudaMemcpyDeviceToDevice));
    });
    const double copy_gbs = traffic / (copy_ms * 1e6);
    printf("%-21s %6d %9.3f %8.1f %6.1f%% %7.1f%% %10s %6s\n", "memcpy (ceiling)", cols, copy_ms,
           copy_gbs, 100.0 * copy_gbs / dev.peak_bw_gbs, 100.0, "-", "-");
    if (csv)
      fprintf(csv, "copy,memcpy,%d,%d,%.4f,%.2f,%.2f,100.00,0,PASS\n", rows, cols, copy_ms, copy_gbs,
              100.0 * copy_gbs / dev.peak_bw_gbs);

    for (auto& k : kernels) {
      if (k.needs_small_cols && !warp_reg_supported(cols)) {
        printf("%-21s %6d %9s %8s %7s %8s %10s %6s\n", k.name.c_str(), cols, "-", "-", "-", "-", "-", "n/a");
        continue;
      }
      CUDA_CHECK(cudaMemset(dy, 0, bytes));
      k.fn(dx, dg, db, dy, rows, cols);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost));
      const auto& ref = (k.op == "softmax") ? hsm : hln;
      const float err = max_rel_error(hy, ref);
      const bool pass = err < 1e-4f;
      all_pass &= pass;

      const float ms = time_kernel_ms([&] { k.fn(dx, dg, db, dy, rows, cols); });
      const double gbs = traffic / (ms * 1e6);
      printf("%-21s %6d %9.3f %8.1f %6.1f%% %7.1f%% %10.2e %6s\n", k.name.c_str(), cols, ms, gbs,
             100.0 * gbs / dev.peak_bw_gbs, 100.0 * gbs / copy_gbs, err, pass ? "PASS" : "FAIL");
      if (csv)
        fprintf(csv, "%s,%s,%d,%d,%.4f,%.2f,%.2f,%.2f,%.3e,%s\n", k.op.c_str(), k.name.c_str(), rows,
                cols, ms, gbs, 100.0 * gbs / dev.peak_bw_gbs, 100.0 * gbs / copy_gbs, err,
                pass ? "PASS" : "FAIL");
    }
    printf("\n");
    cudaFree(dx); cudaFree(dy); cudaFree(dg); cudaFree(db);
  }

  if (csv) { fclose(csv); printf("Wrote %s\n", csv_path.c_str()); }
  printf(all_pass ? "All correctness checks PASSED\n" : "Some checks FAILED\n");
  return all_pass ? 0 : 1;
}
