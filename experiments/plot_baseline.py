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

# Plotly Colormap for optimizers
OPT_STYLE: dict[str, dict] = {
    "sgd":  {"color": "#7B8794", "symbol": "diamond",     "label": "SGD"},
    "sam":  {"color": "#4878CF", "symbol": "circle",      "label": "SAM"},
    "asam": {"color": "#C8703A", "symbol": "triangle-up", "label": "ASAM"},
    "msam": {"color": "#3D8C6E", "symbol": "square",      "label": "M-SAM"},
}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def summary(entry: dict) -> dict:
    return entry["summary"]


def plot_accuracy_vs_rho(data: list[dict], out_dir: str) -> None:
    """Plot test accuracy vs rho for all optimizers, with SGD as a horizontal dashed reference line."""
    fig = go.Figure()

    # Collect all rho values from SAM-family optimizers to set the x-span of the SGD reference
    all_rhos = []
    for opt in ("sam", "asam", "msam"):
        all_rhos += [summary(s)["rho"] for s in data if summary(s)["optimizer"] == opt]
    x_min = min(all_rhos) - 0.05 if all_rhos else 0.0
    x_max = max(all_rhos) + 0.05 if all_rhos else 1.0

    # Plot SGD as a Scatter trace (not add_hline) so its y-value is included in autorange
    sgd = next(s for s in data if summary(s)["optimizer"] == "sgd")
    sgd_acc = summary(sgd)["test_acc_mean"]
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[sgd_acc, sgd_acc],
        mode="lines",
        name=f"SGD ({sgd_acc:.4f})",
        line=dict(color=OPT_STYLE["sgd"]["color"], dash="dash"),
    ))

    # Line plot Test accuracy for SAM, ASAM & MSAM
    for opt in ("sam", "asam", "msam"):
        rows = sorted(
            [s for s in data if summary(s)["optimizer"] == opt],
            key=lambda s: summary(s)["rho"],
        )
        rhos = [summary(r)["rho"] for r in rows]
        accs = [summary(r)["test_acc_mean"] for r in rows]
        style = OPT_STYLE[opt]
        fig.add_trace(go.Scatter(
            x=rhos, y=accs, mode="lines+markers",
            name=style["label"],
            line=dict(color=style["color"]),
            marker=dict(color=style["color"], size=8, symbol=style["symbol"]),
        ))

    fig.update_layout(
        title="Test Accuracy vs Perturbation Radius (ResNet-18 / CIFAR-10)",
        xaxis_title="Perturbation radius rho",
        yaxis_title="Test accuracy",
        yaxis=dict(range=[0.95, 0.97], tickformat=".2f"),
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top", font=dict(size=12)),
        font=dict(size=16),
        title_font=dict(size=18),
        width=800,
        height=500,
    )
    path = os.path.join(out_dir, "baseline_accuracy.html")
    fig.write_html(path)
    print(f"Saved → {path}")
    png_path = os.path.join(out_dir, "baseline_accuracy.png")
    fig.write_image(png_path, width=800, height=500, scale=2)
    print(f"Saved → {png_path}")


def plot_gen_gap_vs_rho(data: list[dict], out_dir: str) -> None:
    """Plot generalization gap vs rho for all optimizers, with SGD as a horizontal dashed reference line."""
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.4, 0.6],
        subplot_titles=["Gen. Gap vs rho", "Best rho per Optimizer"],
    )

    # Line plot for generalization gap over rho, with SGD as horizontal dashed reference
    for opt in ("sam", "asam", "msam"):
        rows = sorted(
            [s for s in data if summary(s)["optimizer"] == opt],
            key=lambda s: summary(s)["rho"],
        )
        if not rows:
            continue
        rhos = [summary(r)["rho"] for r in rows]
        gaps = [summary(r)["divergence_rate_mean"] for r in rows]
        style = OPT_STYLE[opt]
        fig.add_trace(go.Scatter(
            x=rhos, y=gaps, mode="lines+markers",
            name=style["label"],
            line=dict(color=style["color"]),
            marker=dict(color=style["color"], size=7, symbol=style["symbol"]),
        ), row=1, col=1)

    sgd = next(s for s in data if summary(s)["optimizer"] == "sgd")
    sgd_gap = summary(sgd)["divergence_rate_mean"]
    all_gap_rhos = []
    for opt in ("sam", "asam", "msam"):
        all_gap_rhos += [summary(s)["rho"] for s in data if summary(s)["optimizer"] == opt]
    gx_min = min(all_gap_rhos) - 0.05 if all_gap_rhos else 0.0
    gx_max = max(all_gap_rhos) + 0.05 if all_gap_rhos else 1.0
    fig.add_trace(go.Scatter(
        x=[gx_min, gx_max], y=[sgd_gap, sgd_gap],
        mode="lines",
        name=f"SGD ({sgd_gap:.4f})",
        line=dict(color=OPT_STYLE["sgd"]["color"], dash="dash"),
        showlegend=True,
    ), row=1, col=1)

    # Bar chart for best-rho per optimizer
    best_labels, best_gaps, best_colors = [], [], []
    for opt in ("sgd", "sam", "msam", "asam"):
        rows_ = [s for s in data if summary(s)["optimizer"] == opt]
        if not rows_:
            continue
        best_entry = min(rows_, key=lambda s: summary(s)["divergence_rate_mean"])
        label = OPT_STYLE[opt]["label"]
        gap = summary(best_entry)["divergence_rate_mean"]
        rho = summary(best_entry)["rho"]
        best_labels.append(f"{label}<br>(rho={rho})")
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
        yaxis_title="Generalization gap",
        font=dict(size=16),
        title_font=dict(size=18),
        width=800,
        height=500,
    )
    path = os.path.join(out_dir, "baseline_gen_gap.html")
    fig.write_html(path)
    print(f"Saved → {path}")
    png_path = os.path.join(out_dir, "baseline_gen_gap.png")
    fig.write_image(png_path, width=800, height=500, scale=2)
    print(f"Saved → {png_path}")


def plot_summary_vs_rho(data: list[dict], out_dir: str) -> None:
    """Bar charts comparing best-rho test accuracy and gen-gap per optimizer."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Avg Test Accuracy per Optimizer", "Avg Generalization Gap per Optimizer"],
    )

    for col_idx, metric_key in enumerate([
        "test_acc_mean",
        "divergence_rate_mean",
    ], start=1):
        labels, vals, colors = [], [], []
        for opt in ("sgd", "sam", "msam", "asam"):
            rows_ = [s for s in data if summary(s)["optimizer"] == opt]
            if not rows_:
                continue
            val = sum(summary(r)[metric_key] for r in rows_) / len(rows_)
            labels.append(OPT_STYLE[opt]["label"])
            vals.append(val)
            colors.append(OPT_STYLE[opt]["color"])

        fig.add_trace(go.Bar(
            x=labels, y=vals,
            marker_color=colors,
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
            showlegend=True,
        ), row=1, col=col_idx)

    fig.update_layout(
        title="ResNet-18 / CIFAR-10 — Baseline Sweep Summary",
        font=dict(size=16),
        title_font=dict(size=18),
        width=800,
        height=500,
    )
    path = os.path.join(out_dir, "baseline_summary.html")
    fig.write_html(path)
    print(f"Saved → {path}")
    png_path = os.path.join(out_dir, "baseline_summary.png")
    fig.write_image(png_path, width=800, height=500, scale=2)
    print(f"Saved → {png_path}")

def main(results_path: str, out_dir: str | None = None) -> None:
    """Produces three interactive HTML figures saved alongside the JSON:
        1. baseline_accuracy.html  — test accuracy vs rho per optimizer family
        2. baseline_gen_gap.html   — generalization gap per optimizer/rho
        3. baseline_summary.html   — best-rho accuracy + gen-gap side by side
    """
    data = load_results(results_path)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(results_path) or ".", "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoaded {len(data)} entries from {results_path}")
    print("\n── Accuracy table ──")
    print(f"{'Optimizer':8s}  {'rho':6s}  {'test_acc':10s}  {'gen_gap':10s}")
    print("-" * 42)
    for entry in data:
        s = summary(entry)
        print(f"{s['optimizer']:8s}  {s['rho']:<6.3f}  {s['test_acc_mean']:.4f}      {s['divergence_rate_mean']:.4f}")

    plot_accuracy_vs_rho(data, out_dir)
    plot_gen_gap_vs_rho(data, out_dir)
    plot_summary_vs_rho(data, out_dir)

    print("\nPlots saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="results/resnet18/experiments/baseline/resnet18/baseline_results.json",
        help="Path to baseline_results.json",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for output plots (default: <results_dir>/plots/)",
    )
    args = parser.parse_args()
    main(args.results, args.out_dir)
