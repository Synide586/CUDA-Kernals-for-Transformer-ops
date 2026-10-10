import argparse
import csv
import math

import torch
import torch.nn.functional as F

SEQS = [512, 1024, 2048, 4096, 8192, 16384]
D = 64

def time_ms(fn, target_ms=300.0):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    start.record(); fn(); stop.record(); torch.cuda.synchronize()
    reps = int(min(max(target_ms / max(start.elapsed_time(stop), 1e-3), 3), 50))
    start.record()
    for _ in range(reps):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / reps

def math_attention(q, k, v, causal):
    s = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    if causal:
        n = s.shape[-1]
        mask = torch.ones(n, n, dtype=torch.bool, device=s.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bh", type=int, default=16)
    ap.add_argument("--csv", default="results/torch_attention.csv")
    args = ap.parse_args()
    B, H = 2, args.bh // 2

    print(f"PyTorch {torch.__version__} on {torch.cuda.get_device_name()} | BH = {args.bh}, D = {D}, FP32")
    print(f"{'kernel':<12} {'causal':>6} {'N':>6} {'ms':>10} {'GFLOPS':>9}")
    out = []
    for n in SEQS:
        q, k, v = (torch.randn(B, H, n, D, device="cuda") for _ in range(3))
        for causal in (False, True):
            flops = 4.0 * args.bh * n * n * D * (0.5 if causal else 1.0)
            for name, fn in [
                ("torch_math", lambda: math_attention(q, k, v, causal)),
                ("torch_sdpa", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)),
            ]:
                try:
                    ms = time_ms(fn)
                    gf = flops / (ms * 1e6)
                    print(f"{name:<12} {int(causal):>6} {n:>6} {ms:>10.3f} {gf:>9.1f}")
                    out.append([name, int(causal), args.bh, n, f"{ms:.4f}", f"{gf:.2f}", "OK"])
                except torch.OutOfMemoryError:
                    print(f"{name:<12} {int(causal):>6} {n:>6} {'-':>10} {'-':>9}  OOM")
                    out.append([name, int(causal), args.bh, n, "", "", "OOM"])
                torch.cuda.empty_cache()
        del q, k, v
        torch.cuda.empty_cache()

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "causal", "bh", "n", "ms", "gflops", "status"])
        w.writerows(out)
    print(f"Wrote {args.csv}")

if __name__ == "__main__":
    main()
