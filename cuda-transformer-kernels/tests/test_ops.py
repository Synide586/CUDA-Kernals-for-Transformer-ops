import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

import tkernels

DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False

def rand(*shape, scale=1.0, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.rand(*shape, device=DEV, generator=g) * 2 - 1) * scale

@pytest.mark.parametrize("cols", [1, 7, 32, 100, 512, 513, 768, 1000, 2048, 4096])
def test_softmax_matches_torch(cols):
    x = rand(3, 5, cols, scale=8.0)
    torch.testing.assert_close(tkernels.softmax(x), torch.softmax(x, dim=-1), rtol=1e-5, atol=1e-6)

def test_softmax_large_values_are_stable():
    x = rand(64, 256, scale=1.0) + 1000.0
    y = tkernels.softmax(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.sum(-1), torch.ones(64, device=DEV), rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("cols", [1, 31, 64, 300, 512, 768, 1024, 3000])
def test_layer_norm_matches_torch(cols):
    x = rand(4, 6, cols, scale=3.0, seed=1) + 2.0
    w = rand(cols, seed=2) + 1.0
    b = rand(cols, seed=3)
    torch.testing.assert_close(
        tkernels.layer_norm(x, w, b, 1e-5), F.layer_norm(x, (cols,), w, b, 1e-5), rtol=1e-4, atol=1e-5
    )

@pytest.mark.parametrize("m,k,n", [(128, 8, 128), (128, 256, 384), (512, 64, 256), (1024, 1024, 1024)])
def test_matmul_matches_torch(m, k, n):
    a, b = rand(m, k, seed=4), rand(k, n, seed=5)
    torch.testing.assert_close(tkernels.matmul(a, b), a @ b, rtol=1e-4, atol=1e-4)

def test_matmul_rejects_unsupported_shape():
    with pytest.raises(RuntimeError, match="divisible by 128"):
        tkernels.matmul(rand(100, 64), rand(64, 128))

def attention_reference(q, k, v, causal):
    q, k, v = q.double(), k.double(), v.double()
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        n = s.shape[-1]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=DEV).triu(1), float("-inf"))
    return (torch.softmax(s, dim=-1) @ v).float()

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 64, 64), (2, 4, 128, 64), (1, 3, 512, 64), (2, 64, 64)])
def test_flash_attention_matches_reference(shape, causal):
    q, k, v = rand(*shape, scale=2.0, seed=6), rand(*shape, scale=2.0, seed=7), rand(*shape, seed=8)
    out = tkernels.flash_attention(q, k, v, causal)
    torch.testing.assert_close(out, attention_reference(q, k, v, causal), rtol=1e-4, atol=1e-5)

def test_flash_attention_matches_sdpa():
    q, k, v = (rand(2, 8, 256, 64, seed=s) for s in (9, 10, 11))
    torch.testing.assert_close(
        tkernels.flash_attention(q, k, v, True),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
        rtol=1e-4,
        atol=1e-5,
    )

def test_flash_attention_rejects_bad_head_dim():
    x = rand(1, 1, 64, 32)
    with pytest.raises(RuntimeError, match="head dimension"):
        tkernels.flash_attention(x, x, x)

def test_flash_attention_rejects_bad_seq_len():
    x = rand(1, 1, 100, 64)
    with pytest.raises(RuntimeError, match="divisible by 64"):
        tkernels.flash_attention(x, x, x)

def test_rejects_cpu_tensor():
    with pytest.raises(RuntimeError, match="CUDA tensor"):
        tkernels.softmax(torch.randn(4, 4))

def test_rejects_wrong_dtype():
    with pytest.raises(RuntimeError, match="float32"):
        tkernels.softmax(rand(4, 4).half())

def test_rejects_non_contiguous():
    with pytest.raises(RuntimeError, match="contiguous"):
        tkernels.softmax(rand(64, 32).t())

def test_empty_tensor():
    assert tkernels.softmax(torch.empty(0, 16, device=DEV)).shape == (0, 16)

def test_runs_on_current_stream():
    x = rand(1024, 768, scale=4.0, seed=12)
    q = rand(2, 4, 512, 64, seed=13)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y = tkernels.softmax(x * 2)
        o = tkernels.flash_attention(q, q, q, True)
    s.synchronize()
    torch.testing.assert_close(y, torch.softmax(x * 2, -1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(o, attention_reference(q, q, q, True), rtol=1e-4, atol=1e-5)
