import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkernels

torch.backends.cuda.matmul.allow_tf32 = False

def time_ms(fn, target_ms=300.0):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    start.record(); fn(); stop.record(); torch.cuda.synchronize()
    reps = int(min(max(target_ms / max(start.elapsed_time(stop), 1e-3), 3), 100))
    start.record()
    for _ in range(reps):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / reps

def sublayer(x, ln_w, ln_b, w_qkv, w_o, heads, use_custom):
    B, N, C = x.shape
    D = C // heads
    x2 = x.reshape(B * N, C)
    if use_custom:
        h = tkernels.layer_norm(x2, ln_w, ln_b)
        qkv = tkernels.matmul(h, w_qkv)
    else:
        h = F.layer_norm(x2, (C,), ln_w, ln_b)
        qkv = h @ w_qkv
    qkv = qkv.view(B, N, 3, heads, D).permute(2, 0, 3, 1, 4)
    q, k, v = (t.contiguous() for t in qkv)
    if use_custom:
        o = tkernels.flash_attention(q, k, v, causal=True)
    else:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    o = o.transpose(1, 2).reshape(B * N, C)
    out = tkernels.matmul(o, w_o) if use_custom else o @ w_o
    return x + out.view(B, N, C)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/ext_bench.csv")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    print(f"PyTorch {torch.__version__} on {torch.cuda.get_device_name()}\n")

    x_sm1 = torch.randn(8192, 512, device=dev)
    x_sm2 = torch.randn(8192, 4096, device=dev)
    x_ln = torch.randn(8192, 768, device=dev)
    w_ln, b_ln = torch.randn(768, device=dev), torch.randn(768, device=dev)
    a_mm, b_mm = torch.randn(4096, 4096, device=dev), torch.randn(4096, 4096, device=dev)
    q, k, v = (torch.randn(2, 8, 4096, 64, device=dev) for _ in range(3))

    cases = [
        ("softmax", "8192x512", lambda: tkernels.softmax(x_sm1), lambda: torch.softmax(x_sm1, -1)),
        ("softmax", "8192x4096", lambda: tkernels.softmax(x_sm2), lambda: torch.softmax(x_sm2, -1)),
        ("layer_norm", "8192x768", lambda: tkernels.layer_norm(x_ln, w_ln, b_ln),
         lambda: F.layer_norm(x_ln, (768,), w_ln, b_ln)),
        ("matmul", "4096^3", lambda: tkernels.matmul(a_mm, b_mm), lambda: a_mm @ b_mm),
        ("attention", "2x8x4096x64", lambda: tkernels.flash_attention(q, k, v, False),
         lambda: F.scaled_dot_product_attention(q, k, v)),
        ("attention causal", "2x8x4096x64", lambda: tkernels.flash_attention(q, k, v, True),
         lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
    ]

    rows = []
    print(f"{'op':<18} {'shape':<13} {'tkernels ms':>12} {'torch ms':>10} {'torch/ours':>11} {'max |diff|':>11}")
    for op, shape, ours, theirs in cases:
        diff = (ours() - theirs()).abs().max().item()
        t_ours, t_torch = time_ms(ours), time_ms(theirs)
        print(f"{op:<18} {shape:<13} {t_ours:>12.3f} {t_torch:>10.3f} {t_torch / t_ours:>10.2f}x {diff:>11.2e}")
        rows.append([op, shape, f"{t_ours:.4f}", f"{t_torch:.4f}", f"{t_torch / t_ours:.3f}", f"{diff:.3e}"])

    B, N, C, H = 2, 2048, 512, 8
    x = torch.randn(B, N, C, device=dev)
    ln_w, ln_b = torch.ones(C, device=dev), torch.zeros(C, device=dev)
    w_qkv = torch.randn(C, 3 * C, device=dev) / C**0.5
    w_o = torch.randn(C, C, device=dev) / C**0.5
    ref = sublayer(x, ln_w, ln_b, w_qkv, w_o, H, use_custom=False)
    got = sublayer(x, ln_w, ln_b, w_qkv, w_o, H, use_custom=True)
    diff = (ref - got).abs().max().item()
    t_custom = time_ms(lambda: sublayer(x, ln_w, ln_b, w_qkv, w_o, H, True))
    t_torch = time_ms(lambda: sublayer(x, ln_w, ln_b, w_qkv, w_o, H, False))
    print(f"\nCausal attention sublayer, x = [{B}, {N}, {C}], {H} heads:")
    print(f"  tkernels {t_custom:.3f} ms | PyTorch {t_torch:.3f} ms | "
          f"torch/ours {t_torch / t_custom:.2f}x | max |diff| {diff:.2e}")
    rows.append(["attention sublayer", f"{B}x{N}x{C}", f"{t_custom:.4f}", f"{t_torch:.4f}",
                 f"{t_torch / t_custom:.3f}", f"{diff:.3e}"])

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "shape", "tkernels_ms", "torch_ms", "torch_over_ours", "max_abs_diff"])
        w.writerows(rows)
    print(f"Wrote {args.csv}")

if __name__ == "__main__":
    main()
