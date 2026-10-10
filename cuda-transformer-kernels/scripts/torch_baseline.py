import argparse
import csv

import torch
import torch.nn.functional as F

COLS = [128, 256, 512, 768, 1024, 2048, 4096]

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--csv", default="results/torch_rownorm.csv")
    args = ap.parse_args()

    print(f"PyTorch {torch.__version__} on {torch.cuda.get_device_name()} | rows = {args.rows}")
    print(f"{'kernel':<18} {'cols':>6} {'ms':>9} {'GB/s':>8}")
    rows_out = []
    for cols in COLS:
        x = torch.rand(args.rows, cols, device="cuda") * 16 - 8
        g = torch.ones(cols, device="cuda")
        b = torch.zeros(cols, device="cuda")
        traffic = 2.0 * x.numel() * 4
        for name, fn in [
            ("torch_softmax", lambda: torch.softmax(x, dim=-1)),
            ("torch_layernorm", lambda: F.layer_norm(x, (cols,), g, b, 1e-5)),
        ]:
            ms = time_ms(fn)
            gbs = traffic / (ms * 1e6)
            print(f"{name:<18} {cols:>6} {ms:>9.3f} {gbs:>8.1f}")
            op = "softmax" if "softmax" in name else "layernorm"
            rows_out.append([op, name, args.rows, cols, f"{ms:.4f}", f"{gbs:.2f}"])

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "kernel", "rows", "cols", "ms", "gbs"])
        w.writerows(rows_out)
    print(f"Wrote {args.csv}")

if __name__ == "__main__":
    main()
