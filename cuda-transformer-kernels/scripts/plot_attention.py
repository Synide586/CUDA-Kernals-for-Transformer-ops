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
            if row["status"] in ("OOM", "FAIL") or not row["ms"]:
                continue
            data[(row["kernel"], int(row["causal"]))].append((int(row["n"]), float(row["ms"])))

def main():
    ours = sys.argv[1] if len(sys.argv) > 1 else "results/attention.csv"
    theirs = sys.argv[2] if len(sys.argv) > 2 else "results/torch_attention.csv"
    data = defaultdict(list)
    load(ours, data)
    if os.path.exists(theirs):
        load(theirs, data)
    out_dir = os.path.dirname(ours) or "."

    for causal, tag in [(0, "noncausal"), (1, "causal")]:
        plt.figure(figsize=(8, 5))
        for (k, c), pts in sorted(data.items()):
            if c != causal:
                continue
            pts.sort()
            n, ms = zip(*pts)
            plt.plot(n, ms, "--" if k.startswith("torch") else "-", marker="o", label=k)
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("Sequence length N")
        plt.ylabel("Time (ms, log scale)")
        plt.title(f"Attention forward, FP32, {'causal' if causal else 'non-causal'} (missing point = OOM)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(out_dir, f"attention_{tag}.png")
        plt.savefig(path, dpi=150)
        print(f"Saved {path}")

if __name__ == "__main__":
    main()
