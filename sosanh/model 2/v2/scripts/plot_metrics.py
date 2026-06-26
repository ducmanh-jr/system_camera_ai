"""Plot loss/CER/seq_acc curves from the metrics.jsonl file emitted by TrainerV2.

Usage:
    python v2/scripts/plot_metrics.py --run runs/crnn_v2

Generates:
    runs/crnn_v2/curves.png — 2×2 grid (loss/lr/cer/seq_acc)
    runs/crnn_v2/confusion.png — top character confusion heatmap (if data present)
"""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_metrics(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot(run_dir: str):
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    rows = _read_metrics(metrics_path)
    if not rows:
        print("No metrics found in", metrics_path)
        return
    epochs = [r["epoch"] for r in rows]
    cers = [r["cer"] for r in rows]
    wers = [r["wer"] for r in rows]
    accs = [r["seq_acc"] for r in rows]
    char_accs = [r["char_acc"] for r in rows]
    lrs = [r["lr"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(epochs, cers, label="CER", color="tab:red")
    axes[0, 0].plot(epochs, wers, label="WER", color="tab:purple")
    axes[0, 0].set_title("CER / WER vs Epoch"); axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, accs, label="seq_acc", color="tab:green")
    axes[0, 1].plot(epochs, char_accs, label="char_acc", color="tab:olive")
    axes[0, 1].set_title("Accuracy"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, lrs, color="tab:blue")
    axes[1, 0].set_title("Learning Rate"); axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    # confusion bar
    confusion = Counter()
    for r in rows[-1:]:
        for (g, p), c in r.get("top_confusion", []):
            confusion[(g, p)] += c
    if confusion:
        items = confusion.most_common(15)
        labels = [f"{g}→{p}" for (g, p), _ in items]
        counts = [c for _, c in items]
        axes[1, 1].barh(labels[::-1], counts[::-1], color="tab:orange")
        axes[1, 1].set_title("Top character confusions (last epoch)")
        axes[1, 1].grid(True, alpha=0.3, axis="x")
    else:
        axes[1, 1].axis("off")

    fig.suptitle(f"VN-ALPR training curves: {os.path.basename(run_dir)}", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(run_dir, "curves.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print("Saved", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    plot(args.run)


if __name__ == "__main__":
    main()
