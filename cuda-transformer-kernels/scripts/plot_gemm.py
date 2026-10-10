import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["naive_uncoalesced", "naive_coalesced", "smem_tiled", "2d_blocktiling", "2d_blocktiling_occ",
         "2d_blocktiling_v2", "cublas"]

def load(path):
    data = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["kernel"]].append(
                (int(row["size"]), float(row["gflops"]), float(row["pct_cublas"]))
            )
    for k in data:
        data[k].sort()
    return data

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/gemm.csv"
    out_dir = os.path.dirname(path) or "."
    data = load(path)
    kernels = [k for k in ORDER if k in data]

    plt.figure(figsize=(8, 5))
    for k in kernels:
        sizes, gflops, _ = zip(*data[k])
        style = "--" if k == "cublas" else "-"
        plt.plot(sizes, gflops, style, marker="o", label=k)
    plt.xscale("log", base=2)
    plt.xlabel("Matrix size N (M = N = K)")
    plt.ylabel("GFLOPS (FP32)")
    plt.title("SGEMM throughput by kernel")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(out_dir, "gemm_gflops.png")
    plt.savefig(p1, dpi=150)

    largest = max(s for s, _, _ in data["cublas"])
    vals = []
    for k in kernels:
        pct = [p for s, _, p in data[k] if s == largest]
        vals.append(pct[0] if pct else 0.0)
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(kernels, vals, color=["#bbb", "#999", "#6a9", "#2a7", "#4a8", "#185", "#357"][: len(kernels)])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")
    plt.ylabel("% of cuBLAS throughput")
    plt.title(f"SGEMM at N = {largest}")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, max(vals) * 1.15)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "gemm_pct_cublas.png")
    plt.savefig(p2, dpi=150)
    print(f"Saved {p1} and {p2}")

if __name__ == "__main__":
    main()
