import os

from torch.utils.cpp_extension import load

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# compiled on first import (about 1-2 minutes), cached afterwards
_ext = load(
    name="tkernels_ext",
    sources=[os.path.join(_ROOT, "ext", "ops.cu")],
    extra_include_paths=[os.path.join(_ROOT, "src")],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    verbose=os.environ.get("TKERNELS_VERBOSE", "0") == "1",
)

def softmax(x):
    return _ext.softmax(x)

def layer_norm(x, weight, bias, eps=1e-5):
    return _ext.layer_norm(x, weight, bias, eps)

def matmul(a, b):
    return _ext.matmul(a, b)

def flash_attention(q, k, v, causal=False):
    return _ext.flash_attention(q, k, v, causal)

__all__ = ["softmax", "layer_norm", "matmul", "flash_attention"]
