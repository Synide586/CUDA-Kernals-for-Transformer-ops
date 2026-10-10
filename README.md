# CUDA-Kernals-for-Transformer-ops
Wrote CUDA kernels for GEMM, softmax, LayerNorm and fused attention as a PyTorch extension; tiled GEMM reaches X% of cuBLAS on T4; fused attention cuts HBM traffic by Y× vs. unfused PyTorch baseline (Nsight Compute).
