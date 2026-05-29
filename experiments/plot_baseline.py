"""Visualize baseline results from baseline_results.json.

Produces three interactive HTML figures saved alongside the JSON:
  1. baseline_accuracy.html  — test accuracy vs ρ per optimizer family
  2. baseline_gen_gap.html   — generalization gap per optimizer/ρ
  3. baseline_summary.html   — best-ρ accuracy + gen-gap side by side

Usage:
    uv run python experiments/plot_baseline.py \
        --results results/experiments/baseline/resnet18/baseline_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── colour / symbol map ──────────────────────────────────────────────────────
OPT_STYLE: dict[str, dict] = {
    "sgd":  {"color": "#9E9E9E", "symbol": "diamond",  "label": "SGD"},
    "sam":  {"color": "#2196F3", "symbol": "circle",   "label": "SAM"},
    "asam": {"color": "#FF9800", "symbol": "triangle-up", "label": "ASAM"},
    "msam": {"color": "#4CAF50", "symbol": "square",   "label": "M-SAM"},
}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _summary(entry: dict) -> dict:
    return entry["summary"]


# ── Figure 1: test accuracy vs ρ ─────────────────────────────────────────────

def plot_accuracy(data: list[dict], out_dir: str) -> None:
    fig = go.Figure()

    # SGD — horizontal dashed reference
    sgd = next(s for s in data if _summary(s)["optimizer"] == "sgd")
    sgd_acc = _summary(sgd)["test_acc_mean"]
    fig.add_hline(
        y=sgd_acc,
        line_dash="dash",
        line_color=OPT_STYLE["sgd"]["color"],
        annotation_text=f"SGD ({sgd_acc:.4f})",
        annotation_position="top right",
    )

    # SAM, ASAM & MSAM — line plots over ρ
    for opt in ("sam", "asam", "msam"):
        rows = sorted(
            [s for s in data if _summary(s)["optimizer"] == opt],
            key=lambda s: _summary(s)["rho"],
        )
        rhos = [_summary(r)["rho"] for r in rows]
        accs = [_summary(r)["test_acc_mean"] for r in rows]
        style = OPT_STYLE[opt]
        fig.add_trace(go.Scatter(
            x=rhos, y=accs, mode="lines+markers",
            name=style["label"],
            line=dict(color=style["color"]),
            marker=dict(color=style["color"], size=8, symbol=style["symbol"]),
        ))

    fig.update_layout(
        title="Test Accuracy vs Perturbation Radius (ResNet-18 / CIFAR-10)",
        xaxis_title="Perturbation radius ρ",
        yaxis_title="Test accuracy",
        yaxis_tickformat=".4f",
        template="plotly_white",
        legend=dict(x=0.01, y=0.01),
        font=dict(size=16),
        title_font=dict(size=18),
    )
    path = os.path.join(out_dir, "baseline_accuracy.html")
    fig.write_html(path)
    print(f"Saved → {path}")
    png_path = os.path.join(out_dir, "baseline_accuracy.png")
    fig.write_image(png_path, width=900, height=550, scale=2)
    print(f"Saved → {png_path}")


# ── Figure 2: generalization gap vs optimizer/ρ ──────────────────────────────

def plot_gen_gap(data: list[dict], out_dir: str) -> None:
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=["Gen. Gap vs ρ", "Best ρ per Optimizer"],
    )

    # Left: SAM, ASAM & MSAM line plots
    for opt in ("sam", "asam", "msam"):
        rows = sorted(
            [s for s in data if _summary(s)["optimizer"] == opt],
            key=lambda s: _summary(s)["rho"],
        )
        rhos = [_summary(r)["rho"] for r in rows]
        gaps = [_summary(r)["divergence_rate_mean"] for r in rows]
        style = OPT_STYLE[opt]
        fig.add_trace(go.Scatter(
            x=rhos, y=gaps, mode="lines+markers",
            name=style["label"],
            line=dict(color=style["color"]),
            marker=dict(color=style["color"], size=7, symbol=style["symbol"]),
        ), row=1, col=1)

    sgd = next(s for s in data if _summary(s)["optimizer"] == "sgd")
    sgd_gap = _summary(sgd)["divergence_rate_mean"]
    fig.add_hline(
        y=sgd_gap, line_dash="dash", line_color=OPT_STYLE["sgd"]["color"],
        annotation_text=f"SGD ({sgd_gap:.4f})",
        annotation_position="top right",
        row=1, col=1,
    )

    # Right: best-ρ bar chart
    best_labels, best_gaps, best_colors = [], [], []
    for opt in ("sgd", "sam", "msam", "asam"):
        rows_ = [s for s in data if _summary(s)["optimizer"] == opt]
        best_entry = min(rows_, key=lambda s: _summary(s)["divergence_rate_mean"])
        label = OPT_STYLE[opt]["label"]
        gap = _summary(best_entry)["divergence_rate_mean"]
        rho = _summary(best_entry)["rho"]
        best_labels.append(f"{label}<br>(ρ={rho})")
        best_gaps.append(gap)
        best_colors.append(OPT_STYLE[opt]["color"])

    fig.add_trace(go.Bar(
        x=best_labels, y=best_gaps,
        marker_color=best_colors,
        text=[f"{g:.4f}" for g in best_gaps],
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title="Generalization Gap Analysis (ResNet-18 / CIFAR-10)",
        template="plotly_white",
        yaxis_title="Generalization gap",
        font=dict(size=16),
        title_font=dict(size=18),
    )
    path = os.path.join(out_dir, "baseline_gen_gap.html")
    fig.write_html(path)
    print(f"Saved → {path}")
    png_path = os.path.join(out_dir, "baseline_gen_gap.png")
    fig.write_image(png_path, width=1200, height=550, scale=2)
    print(f"Saved → {png_path}")


# ── Figure 3: summary — best accuracy + generalization gap side by side ─────────────────

def plot_summary(data: list[dict], out_dir: str) -> None:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Best Test Accuracy per Optimizer", "Smallest Generalization Gap per Optimizer"],
    )

    for col_idx, (metric_key, maximize) in enumerate([
        ("test_acc_mean", True),
        ("divergence_rate_mean", False),
    ], start=1):
        labels, vals, colors = [], [], []
        for opt in ("sgd", "sam", "msam", "asam"):
            rows_ = [s for s in data if _summary(s)["optimizer"] == opt]
            best_entry = (max if maximize else min)(rows_, key=lambda s: _summary(s)[metric_key])
            label = OPT_STYLE[opt]["label"]
            val = _summary(best_entry)[metric_key]
            rho = _summary(best_entry)["rho"]
            labels.append(f"{label}<br>(ρ={rho})")
            vals.append(val)
            colors.append(OPT_STYLE[opt]["color"])

        fig.add_trace(go.Bar(
            x=labels, y=vals,
            marker_color=colors,
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_layout(
        title="ResNet-18 / CIFAR-10 — Baseline Sweep Summary",
        template="plotly_white",
        font=dict(size=16),
        title_font=dict(size=18),
    )
    path = os.path.join(out_dir, "baseline_summary.html")
    fig.write_html(path)
    print(f"Saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(results_path: str) -> None:
    data = load_results(results_path)
    out_dir = os.path.dirname(results_path) or "."

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
        default="results/experiments/baseline/resnet18/baseline_results.json",
        help="Path to baseline_results.json",
    )
    args = parser.parse_args()
    main(args.results)
