"""Visualize baseline results from baseline_results.json.

Produces three figures saved alongside the JSON:
  1. baseline_accuracy.png  — test accuracy vs ρ per optimizer family
  2. baseline_gen_gap.png   — generalization gap (test_loss - train_loss) per optimizer/ρ
  3. baseline_summary.png   — best-ρ accuracy + gen-gap side by side

Usage:
    uv run python experiments/plot_baseline.py \
        --results results/results/baseline/resnet18/baseline_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── colour / marker map ──────────────────────────────────────────────────────
OPT_STYLE: dict[str, dict] = {
    "sam":  {"color": "#2196F3", "marker": "o", "label": "SAM"},
    "msam": {"color": "#4CAF50", "marker": "s", "label": "M-SAM"},
    "asam": {"color": "#FF9800", "marker": "^", "label": "ASAM"},
    "sgd":  {"color": "#9E9E9E", "marker": "D", "label": "SGD"},
}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _summary(entry: dict) -> dict:
    return entry["summary"]


# ── Figure 1: test accuracy vs ρ ─────────────────────────────────────────────

def plot_accuracy(data: list[dict], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # SGD — horizontal dashed reference
    sgd = next(s for s in data if _summary(s)["optimizer"] == "sgd")
    sgd_acc = _summary(sgd)["test_acc_mean"]
    ax.axhline(sgd_acc, color=OPT_STYLE["sgd"]["color"], linestyle="--",
               linewidth=1.5, label=f"SGD  ({sgd_acc:.4f})", zorder=1)

    # SAM & MSAM — line plots over ρ
    for opt in ("sam", "msam"):
        rows = sorted(
            [s for s in data if _summary(s)["optimizer"] == opt],
            key=lambda s: _summary(s)["rho"],
        )
        rhos = [_summary(r)["rho"] for r in rows]
        accs = [_summary(r)["test_acc_mean"] for r in rows]
        style = OPT_STYLE[opt]
        ax.plot(rhos, accs, color=style["color"], marker=style["marker"],
                linewidth=2, markersize=6, label=style["label"], zorder=3)

    # ASAM — single point (different ρ scale — annotate separately)
    asam = next(s for s in data if _summary(s)["optimizer"] == "asam")
    asam_rho = _summary(asam)["rho"]
    asam_acc = _summary(asam)["test_acc_mean"]
    style = OPT_STYLE["asam"]
    ax.scatter([asam_rho], [asam_acc], color=style["color"], marker=style["marker"],
               s=80, zorder=4, label=f"{style['label']}  (ρ={asam_rho})")
    ax.annotate(f"ρ={asam_rho}", xy=(asam_rho, asam_acc),
                xytext=(asam_rho - 0.07, asam_acc - 0.0015),
                fontsize=8, color=style["color"])

    ax.set_xlabel("Perturbation radius ρ", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("Test Accuracy vs Perturbation Radius (ResNet-18 / CIFAR-10)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    path = os.path.join(out_dir, "baseline_accuracy.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Figure 2: generalization gap vs optimizer/ρ ──────────────────────────────

def plot_gen_gap(data: list[dict], out_dir: str) -> None:
    # Separate SAM/MSAM (shared ρ axis) from ASAM/SGD (single values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: SAM & MSAM line plot
    ax = axes[0]
    for opt in ("sam", "msam"):
        rows = sorted(
            [s for s in data if _summary(s)["optimizer"] == opt],
            key=lambda s: _summary(s)["rho"],
        )
        rhos = [_summary(r)["rho"] for r in rows]
        gaps = [_summary(r)["divergence_rate_mean"] for r in rows]
        style = OPT_STYLE[opt]
        ax.plot(rhos, gaps, color=style["color"], marker=style["marker"],
                linewidth=2, markersize=6, label=style["label"])

    sgd = next(s for s in data if _summary(s)["optimizer"] == "sgd")
    ax.axhline(_summary(sgd)["divergence_rate_mean"],
               color=OPT_STYLE["sgd"]["color"], linestyle="--",
               linewidth=1.5, label="SGD")

    ax.set_xlabel("Perturbation radius ρ", fontsize=11)
    ax.set_ylabel("Generalization gap (test_loss − train_loss)", fontsize=11)
    ax.set_title("Generalization Gap vs ρ", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: bar chart of best-ρ per optimizer
    ax2 = axes[1]
    best: list[tuple[str, float]] = []
    for opt in ("sgd", "sam", "msam", "asam"):
        rows = [s for s in data if _summary(s)["optimizer"] == opt]
        best_entry = min(rows, key=lambda s: _summary(s)["divergence_rate_mean"])
        label = OPT_STYLE[opt]["label"]
        gap = _summary(best_entry)["divergence_rate_mean"]
        rho = _summary(best_entry)["rho"]
        best.append((f"{label}\n(ρ={rho})", gap, opt))

    labels = [b[0] for b in best]
    gaps = [b[1] for b in best]
    colors = [OPT_STYLE[b[2]]["color"] for b in best]
    bars = ax2.bar(labels, gaps, color=colors, edgecolor="white", linewidth=0.8)
    for bar, gap in zip(bars, gaps):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{gap:.4f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Generalization gap", fontsize=11)
    ax2.set_title("Best ρ per Optimizer", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "baseline_gen_gap.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Figure 3: summary — best accuracy + gen gap side by side ─────────────────

def plot_summary(data: list[dict], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax_idx, (metric_key, ylabel, title) in enumerate([
        ("test_acc_mean", "Test accuracy", "Best Test Accuracy per Optimizer"),
        ("divergence_rate_mean", "Generalization gap", "Smallest Gen. Gap per Optimizer"),
    ]):
        ax = axes[ax_idx]
        best: list[tuple[str, float, str]] = []
        for opt in ("sgd", "sam", "msam", "asam"):
            rows = [s for s in data if _summary(s)["optimizer"] == opt]
            if metric_key == "test_acc_mean":
                best_entry = max(rows, key=lambda s: _summary(s)[metric_key])
            else:
                best_entry = min(rows, key=lambda s: _summary(s)[metric_key])
            label = OPT_STYLE[opt]["label"]
            val = _summary(best_entry)[metric_key]
            rho = _summary(best_entry)["rho"]
            best.append((f"{label}\n(ρ={rho})", val, opt))

        labels = [b[0] for b in best]
        vals = [b[1] for b in best]
        colors = [OPT_STYLE[b[2]]["color"] for b in best]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.0003 if metric_key == "test_acc_mean" else 0.001),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("ResNet-18 / CIFAR-10 — Baseline Sweep Summary", fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "baseline_summary.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(results_path: str) -> None:
    data = load_results(results_path)
    out_dir = os.path.dirname(os.path.abspath(results_path))

    print(f"\nLoaded {len(data)} entries from {results_path}")
    print("\n── Accuracy table ──")
    print(f"{'Optimizer':8s}  {'rho':6s}  {'test_acc':10s}  {'gen_gap':10s}")
    print("-" * 42)
    for entry in data:
        s = _summary(entry)
        print(f"{s['optimizer']:8s}  {s['rho']:<6.3f}  {s['test_acc_mean']:.4f}      {s['divergence_rate_mean']:.4f}")

    plot_accuracy(data, out_dir)
    plot_gen_gap(data, out_dir)
    plot_summary(data, out_dir)

    print("\nAll figures saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="results/results/baseline/resnet18/baseline_results.json",
        help="Path to baseline_results.json",
    )
    args = parser.parse_args()
    main(args.results)
