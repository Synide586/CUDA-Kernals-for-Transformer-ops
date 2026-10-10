# CUDA Transformer Kernels

Hand-written CUDA kernels for the main operations inside a transformer layer: matrix multiplication, softmax, LayerNorm and attention. Each kernel is built step by step, checked for correctness, benchmarked against cuBLAS and PyTorch, and profiled with Nsight Compute to explain where the time goes. The kernels are also packaged as a PyTorch extension with a test suite.

All results: NVIDIA Tesla T4 (Google Colab), FP32.

## Highlights

- **Matrix multiplication:** 3.38 TFLOPS, **79% of cuBLAS** and 54x faster than the naive kernel.
- **Softmax and LayerNorm:** **93-99% of the GPU's practical memory bandwidth**; LayerNorm up to 3.6x faster than PyTorch at small hidden sizes.
- **Fused attention:** runs at sequence length 16,384 where the standard approach **runs out of memory** (it needs a 17 GB matrix); 1.3-2.6x faster than unfused cuBLAS attention and **within about 11% of PyTorch's built-in attention**.
- **PyTorch extension:** all four kernels callable from Python; **40 tests passing**.

## 1. Matrix multiplication (SGEMM)

Four versions, each fixing the main bottleneck of the previous one.

| Kernel | Idea | GFLOPS (N = 4096) | % of cuBLAS |
|---|---|---|---|
| naive_uncoalesced | one thread per output; scattered memory reads | 62 | 1.4% |
| naive_coalesced | neighbouring threads read neighbouring addresses | 592 | 13.8% |
| smem_tiled | 32x32 tiles loaded into shared memory and reused | 900 | 21.0% |
| 2d_blocktiling | each thread computes an 8x8 block in registers | **3,380** | **78.8%** |
| cuBLAS | NVIDIA's library | 4,288 | 100% |

![GEMM throughput](results/gemm_gflops.png)

What I learned:

- **Memory access pattern matters most.** Changing only which thread reads which element gave a 9.5x speedup.
- **Fixing one bottleneck exposes the next.** After tiling, the profiler showed main memory almost idle (12%) but shared memory at 90%, so shared memory became the limit. Keeping data in registers (8x8 per thread) then gave another 3.8x.
- **Big tiles need big problems.** At N = 256 the fastest kernel launches only 4 blocks for 40 processors and is the slowest.

## 2. Softmax and LayerNorm

These operations do very little math per element, so speed is limited by memory bandwidth. The goal is to read the input once and write the output once. Results are compared with a plain GPU memory copy, the practical speed limit (about 238 GB/s on the T4).

| Kernel | Idea | GB/s (8192 x 1024) |
|---|---|---|
| naive | one thread per row | 39 |
| block | one block of 256 threads per row | 234 |
| warp_online | one warp per row, max and sum in a single pass | 163 |
| warp_reg | one warp per row, row kept in registers | 223 (best up to 512 columns) |
| PyTorch softmax | | 224 |

![Softmax bandwidth](results/softmax_bw.png)
![LayerNorm bandwidth](results/layernorm_bw.png)

What I learned:

- **Warp shuffles** let the 32 threads of a warp combine values directly through registers, with no shared memory or synchronisation.
- **The best kernel depends on the row size.** The warp kernel wins up to 512 columns; the block kernel wins above that. The PyTorch extension uses this measured crossover to pick a kernel.
- **Fewer memory passes is not always faster.** The single-pass online softmax was slower because each thread's loop is a chain of dependent operations, so too few memory requests are in flight.
- LayerNorm was faster than PyTorch at every size in the C++ benchmark (3.6x at 128 columns). PyTorch's softmax was faster for very wide rows (4096 columns), where our kernel re-reads the row.

## 3. Fused attention

Standard attention builds an N x N score matrix in GPU memory, which grows with the square of sequence length. The fused kernels (FlashAttention idea) process keys in tiles and use an **online softmax**: each row keeps a running maximum and sum, and rescales its partial result when a larger value appears. The N x N matrix is never stored.

Time in ms (lower is better), 16 heads, head size 64:

| N | Causal | Unfused (cuBLAS) | flash_v1 | flash_v2 | PyTorch SDPA |
|---|---|---|---|---|---|
| 4096 | no | 32.7 | 38.1 | 24.9 | 22.6 |
| 8192 | no | 165.9 | 151.8 | 101.9 | 91.7 |
| 16384 | no | out of memory | 607.8 | 407.8 | 362.4 |
| 4096 | yes | 28.7 | 18.4 | 12.7 | 12.0 |
| 8192 | yes | 130.0 | 68.4 | 49.7 | 50.5 |
| 16384 | yes | out of memory | 289.0 | 203.3 | 194.1 |

![Attention, causal](results/attention_causal.png)

- **flash_v1:** one thread per query row. Simple, but slow.
- **flash_v2:** 256 threads share a 64x64 tile; each computes a 4x4 block (the register trick from Part 1), and row maximums and sums use warp shuffles (from Part 2). With a causal mask it skips tiles above the diagonal.

What I learned:

- **Fusion removes the memory wall.** At N = 16384 the unfused version needs 17 GB and fails; the fused kernels need no extra memory.
- **The profiler confirms it:** flash_v2 uses only 0.7% of main-memory bandwidth. The limit is now on-chip shared memory, the same limit seen in Part 1.
- **Causal masking is where fusion wins most:** 2.6x faster than unfused at N = 8192, because unfused computes the whole matrix and throws half away.

## 4. PyTorch extension

```python
import tkernels
y = tkernels.softmax(x)
y = tkernels.layer_norm(x, weight, bias)
c = tkernels.matmul(a, b)
o = tkernels.flash_attention(q, k, v, causal=True)
```

- Checks inputs (GPU, float32, contiguous, supported shape) and raises clear Python errors.
- Runs on PyTorch's current CUDA stream, so it works correctly alongside other PyTorch code.
- 40 pytest tests: correctness against PyTorch, edge cases, invalid inputs, non-default streams.

Called from Python (ratio above 1.00x means ours is faster):

| Operation | Shape | Ours (ms) | PyTorch (ms) | PyTorch / ours |
|---|---|---|---|---|
| softmax | 8192 x 512 | 0.145 | 0.144 | 1.00x |
| softmax | 8192 x 4096 | 1.858 | 1.189 | 0.64x |
| layer_norm | 8192 x 768 | 0.234 | 0.210 | 0.90x |
| matmul | 4096 x 4096 | 39.9 | 31.9 | 0.80x |
| attention | 2 x 8 x 4096 x 64 | 24.9 | 21.9 | 0.88x |
| attention, causal | 2 x 8 x 4096 x 64 | 12.8 | 11.6 | 0.91x |
| attention layer (LN + projections + attention) | 2 x 2048 x 512 | 6.68 | 5.38 | 0.81x |

The full attention layer built from these kernels matches PyTorch to within 2e-6. For very short kernels (around 0.2 ms), call overhead and measurement setup matter, so differences under about 10% between the C++ and Python benchmarks should not be over-interpreted.

## How to run (Google Colab, T4 GPU)

```bash
!git clone <repo-url>
%cd cuda-transformer-kernels
!make                                  # build the C++ benchmarks
!make run && make plot                 # Part 1: GEMM
!make run-rownorm && make torch-baseline && make plot-rownorm      # Part 2
!make run-attention && make torch-attention && make plot-attention # Part 3
!make ext-test && make ext-bench       # Part 4
```

Profiling example:

```bash
!ncu --section SpeedOfLight ./bin/bench_gemm --profile 2d_blocktiling --sizes 4096
```

For other GPUs, set the architecture, e.g. `make ARCH=sm_80` for an A100.

## Project structure

```
src/        CUDA kernels and C++ benchmark programs
ext/        PyTorch bindings
tkernels/   Python package (builds the extension on first import)
tests/      pytest tests
scripts/    PyTorch baselines and plotting
results/    measured results (CSV) and plots
```

## Limitations and next steps

FP32 only, forward pass only, head size 64. Next steps: vectorised `float4` loads, double buffering, and FP16 tensor-core versions of the GEMM and attention kernels.
