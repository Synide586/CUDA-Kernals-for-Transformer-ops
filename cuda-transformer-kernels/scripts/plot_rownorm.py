import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path, data):
    with open(path) as f:
        for row in csv.DictReader(f):
            data[(row["op"], row["kernel"])].append((int(row["cols"]), float(row["gbs"])))

def main():
    ours = sys.argv[1] if len(sys.argv) > 1 else "results/rownorm.csv"
    torch_csv = sys.argv[2] if len(sys.argv) > 2 else "results/torch_rownorm.csv"
    data = defaultdict(list)
    load(ours, data)
    if os.path.exists(torch_csv):
        load(torch_csv, data)
    out_dir = os.path.dirname(ours) or "."

    for op in ["softmax", "layernorm"]:
        plt.figure(figsize=(8, 5))
        for (o, k), pts in sorted(data.items()):
            if o not in (op, "copy"):
                continue
            pts.sort()
            cols, gbs = zip(*pts)
            style = ":" if o == "copy" else ("--" if k.startswith("torch") else "-")
            label = "memcpy (practical ceiling)" if o == "copy" else k
            plt.plot(cols, gbs, style, marker="o", label=label)
        plt.xscale("log", base=2)
        plt.xlabel("Hidden size (cols)")
        plt.ylabel("Effective bandwidth (GB/s)")
        plt.title(f"Row-wise {op}: effective DRAM bandwidth")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(out_dir, f"{op}_bw.png")
        plt.savefig(path, dpi=150)
        print(f"Saved {path}")

if __name__ == "__main__":
    main()
