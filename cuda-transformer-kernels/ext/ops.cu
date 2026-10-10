#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <climits>
#include <cstdint>

#include "attention_kernels.cuh"
#include "gemm_kernels.cuh"
#include "rownorm_kernels.cuh"

#define CHECK_INPUT(x)                                                       \
  TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");                   \
  TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32");       \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

// crossover measured in Part 2: warp kernel up to 512 columns, block kernel above
constexpr int WARP_REG_MAX_COLS = 512;

torch::Tensor softmax(torch::Tensor x) {
  CHECK_INPUT(x);
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  auto y = torch::empty_like(x);
  if (x.numel() == 0) return y;
  const int64_t cols = x.size(-1);
  const int64_t rows = x.numel() / cols;
  TORCH_CHECK(rows <= INT_MAX && cols <= INT_MAX, "tensor too large");
  auto stream = at::cuda::getCurrentCUDAStream();
  if (cols <= WARP_REG_MAX_COLS)
    launch_softmax_warp_reg(x.data_ptr<float>(), y.data_ptr<float>(), (int)rows, (int)cols, stream);
  else
    launch_softmax_block(x.data_ptr<float>(), y.data_ptr<float>(), (int)rows, (int)cols, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor layer_norm(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
  const int64_t cols = x.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == cols, "weight must have shape [", cols, "]");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == cols, "bias must have shape [", cols, "]");
  TORCH_CHECK(weight.device() == x.device() && bias.device() == x.device(), "all tensors must be on one device");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  auto y = torch::empty_like(x);
  if (x.numel() == 0) return y;
  const int64_t rows = x.numel() / cols;
  TORCH_CHECK(rows <= INT_MAX && cols <= INT_MAX, "tensor too large");
  auto stream = at::cuda::getCurrentCUDAStream();
  if (cols <= WARP_REG_MAX_COLS)
    launch_layernorm_warp_reg(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                              y.data_ptr<float>(), (int)rows, (int)cols, (float)eps, stream);
  else
    launch_layernorm_block(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                           y.data_ptr<float>(), (int)rows, (int)cols, (float)eps, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2-D");
  TORCH_CHECK(a.size(1) == b.size(0), "inner dimensions must match: ", a.sizes(), " x ", b.sizes());
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
  const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 8 == 0,
              "this kernel needs M and N divisible by 128 and K divisible by 8 (got M=", M,
              ", N=", N, ", K=", K, ")");
  TORCH_CHECK(M <= INT_MAX && N <= INT_MAX && K <= INT_MAX, "matrix too large");
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  auto c = torch::empty({M, N}, a.options());
  if (M == 0 || N == 0) return c;
  if (K == 0) return c.zero_();
  // v2 uses 128-bit loads, which need 16-byte-aligned pointers (a sliced tensor may not be)
  const bool aligned = reinterpret_cast<uintptr_t>(a.data_ptr<float>()) % 16 == 0 &&
                       reinterpret_cast<uintptr_t>(b.data_ptr<float>()) % 16 == 0;
  if (aligned)
    launch_2d_blocktiling_v2((int)M, (int)N, (int)K, a.data_ptr<float>(), b.data_ptr<float>(),
                             c.data_ptr<float>(), at::cuda::getCurrentCUDAStream());
  else
    launch_2d_blocktiling((int)M, (int)N, (int)K, a.data_ptr<float>(), b.data_ptr<float>(),
                          c.data_ptr<float>(), at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

torch::Tensor flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  TORCH_CHECK(q.dim() >= 2, "q must be [..., N, D]");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "q, k and v must have the same shape (self-attention)");
  TORCH_CHECK(k.device() == q.device() && v.device() == q.device(), "all tensors must be on one device");
  const int64_t N = q.size(-2), D = q.size(-1);
  TORCH_CHECK(D == ATTN_D, "head dimension must be ", ATTN_D, " (got ", D, ")");
  TORCH_CHECK(N % ATTN_TILE == 0, "sequence length must be divisible by ", ATTN_TILE, " (got ", N, ")");
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  auto o = torch::empty_like(q);
  if (q.numel() == 0) return o;
  const int64_t BH = q.numel() / (N * D);
  TORCH_CHECK(BH <= 65535, "batch x heads must be <= 65535 (grid y-dimension limit)");
  TORCH_CHECK(N <= INT_MAX, "sequence too long");
  launch_flash_v2(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), o.data_ptr<float>(),
                  (int)BH, (int)N, causal, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Hand-written CUDA kernels for transformer operations (FP32, forward only)";
  m.def("softmax", &softmax, "Row-wise softmax over the last dimension", py::arg("x"));
  m.def("layer_norm", &layer_norm, "LayerNorm over the last dimension", py::arg("x"),
        py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
  m.def("matmul", &matmul, "C = A @ B with the register-blocked SGEMM kernel", py::arg("a"),
        py::arg("b"));
  m.def("flash_attention", &flash_attention, "Fused attention forward (head dim 64)", py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("causal") = false);
}
