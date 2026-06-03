"""Visualize reparametrisation-invariance results from reparam_results.json.

Raw entries are flat dicts:
    {
        "model": str,
        "optimizer": str,       # e.g. "msam"
        "rho": float,
        "alpha": float,
        "test_acc_mean": float,
        "test_acc_sem": float,
        "per_seed": [
            {"train_acc": ..., "test_acc": ..., "train_loss": ...,
             "test_loss": ..., "alpha": ..., "rho": ..., "seed": ..., ...}
        ]
    }

Before plotting, results are aggregated across all seeds *and* all ρ values for
each (optimizer, alpha) pair.  Each aggregated entry pools every per_seed run
from every ρ, then recomputes mean / SEM / gen-gap from that pool.

This collapses the ρ dimension so all plots show a single curve per optimizer
vs α — the cleanest view of reparametrisation invariance.

Produces four figures:

  1. reparam_accuracy.html  — test accuracy vs α, one line per optimizer.
  2. reparam_gen_gap.html   — generalisation gap (test_loss − train_loss) vs α,
                               one line per optimizer.

Usage:
    uv run python experiments/plot_reparam.py --results results/resnet18/experiments/reparam/resnet18/reparam_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── colour / symbol map ──────────────────────────────────────────────────────
OPT_STYLE: dict[str, dict] = {
    "sgd":  {"color": "#7B8794", "symbol": "diamond",     "label": "SGD"},
    "sam":  {"color": "#4878CF", "symbol": "circle",      "label": "SAM"},
    "asam": {"color": "#C8703A", "symbol": "triangle-up", "label": "ASAM"},
    "msam": {"color": "#3D8C6E", "symbol": "square",      "label": "M-SAM"},
}

# Canonical display order
OPT_ORDER = ("sgd", "sam", "asam", "msam")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _sorted_alphas(agg: list[dict]) -> list[float]:
    return sorted({e["alpha"] for e in agg})


def _infer_model_label(data: list[dict]) -> str:
    models = {e.get("model", "") for e in data}
    m = next(iter(models), "")
    return {"resnet18": "ResNet-18", "vit_b_32": "ViT-B/32"}.get(m, m)


def aggregate_by_alpha(data: list[dict]) -> list[dict]:
    """Pool per_seed entries across every ρ for each (optimizer, alpha) pair.

    Returns one entry per (optimizer, alpha) with:
      - test_acc_mean / test_acc_sem  — from pooled test_acc values
      - gen_gap_mean                  — mean of (test_loss − train_loss) in the pool
      - per_seed                      — concatenated list of all per_seed dicts
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    model_map: dict[tuple, str] = {}

    for entry in data:
        key = (entry["optimizer"], entry["alpha"])
        buckets[key].extend(entry.get("per_seed", []))
        model_map[key] = entry.get("model", "")

    result = []
    for (opt, alpha), seeds in sorted(buckets.items()):
        test_accs = [s["test_acc"] for s in seeds if "test_acc" in s]
        train_accs = [s["train_acc"] for s in seeds if "train_acc" in s]
        gaps = [
            s["test_loss"] - s["train_loss"]
            for s in seeds
            if "test_loss" in s and "train_loss" in s
            and not (math.isnan(s["test_loss"]) or math.isnan(s["train_loss"]))
        ]
        result.append({
            "model": model_map[(opt, alpha)],
            "optimizer": opt,
            "alpha": alpha,
            "test_acc_mean": float(np.mean(test_accs)) if test_accs else float("nan"),
            "test_acc_sem": float(np.std(test_accs, ddof=1) / math.sqrt(len(test_accs)))
                            if len(test_accs) > 1 else 0.0,
            "gen_gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
            "per_seed": seeds,
        })
    return result


def _lookup_agg(agg: list[dict], optimizer: str, alpha: float) -> dict | None:
    for e in agg:
        if e["optimizer"] == optimizer and e["alpha"] == alpha:
            return e
    return None


# ── Figure 1: test accuracy vs α ─────────────────────────────────────────────

def plot_accuracy(agg: list[dict], out_dir: str, model_label: str) -> None:
    """Grouped bar chart: x = α, one bar group per optimizer, error bars = SEM."""
    alphas = _sorted_alphas(agg)
    x_labels = [str(a) for a in alphas]

    fig = go.Figure()

    for opt in OPT_ORDER:
        style = OPT_STYLE.get(opt, {"color": "#000", "symbol": "circle", "label": opt})
        accs, sems = [], []
        for alpha in alphas:
            entry = _lookup_agg(agg, opt, alpha)
            accs.append(entry["test_acc_mean"] if entry else None)
            sems.append(entry["test_acc_sem"] if entry else None)

        if all(a is None for a in accs):
            continue

        fig.add_trace(go.Bar(
            name=style["label"],
            x=x_labels,
            y=accs,
            error_y=dict(type="data", array=sems, visible=True),
            marker_color=style["color"],
        ))

    fig.update_layout(
        title=f"Test Accuracy vs Reparametrisation Scale α  ({model_label} / CIFAR-10)",
        xaxis_title="α",
        yaxis_title="Test accuracy",
        yaxis_tickformat=".4f",
        yaxis_rangemode="tozero",
        barmode="group",
        bargap=0.35,
        bargroupgap=0.08,
        template="plotly_white",
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
        font=dict(size=15),
        title_font=dict(size=17),
        width=800,
        height=500,
    )
    fig.update_yaxes(range=[0.94, 0.96])  # accuracy is always in [0, 1]

    path = os.path.join(out_dir, "reparam_accuracy.html")
    fig.write_html(path)
    print(f"Saved → {path}")
    png_path = os.path.join(out_dir, "reparam_accuracy.png")
    fig.write_image(png_path, width=max(900, 200 * len(alphas) + 200), height=550, scale=2)
    print(f"Saved → {png_path}")


# ── Figure 2: generalization gap vs α ───────────────────────────────────────

def plot_gen_gap(agg: list[dict], out_dir: str, model_label: str) -> None:
    """Grouped bar chart: x = α, one bar group per optimizer."""
    alphas = _sorted_alphas(agg)
    x_labels = [str(a) for a in alphas]

    fig = go.Figure()

    for opt in OPT_ORDER:
        style = OPT_STYLE.get(opt, {"color": "#000", "symbol": "circle", "label": opt})
        gaps = []
        for alpha in alphas:
            entry = _lookup_agg(agg, opt, alpha)
            gaps.append(entry["gen_gap_mean"] if entry else None)

        if all(g is None or (g is not None and math.isnan(g)) for g in gaps):
            continue

        fig.add_trace(go.Bar(
            name=style["label"],
            x=x_labels,
            y=gaps,
            marker_color=style["color"],
        ))

    fig.update_layout(
        title=f"Generalisation Gap vs α  ({model_label} / CIFAR-10)",
        xaxis_title="α",
        yaxis_title="Generalisation gap (test loss − train loss)",
        yaxis_tickformat=".4f",
        yaxis_rangemode="tozero",
        barmode="group",
        bargap=0.35,
        bargroupgap=0.08,
        template="plotly_white",
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
        font=dict(size=15),
        title_font=dict(size=17),
        width=800,
        height=500,
    )
    fig.update_yaxes(range=[0.1, 0.2])  # accuracy is always in [0, 1]

    path = os.path.join(out_dir, "reparam_gen_gap.html")
    fig.write_html(path)
    print(f"Saved → {path}")
    png_path = os.path.join(out_dir, "reparam_gen_gap.png")
    fig.write_image(png_path, width=max(900, 200 * len(alphas) + 200), height=550, scale=2)
    print(f"Saved → {png_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(results_path: str, out_dir: str | None = None) -> None:
    data = load_results(results_path)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(results_path) or ".", "plots")
    os.makedirs(out_dir, exist_ok=True)

    model_label = _infer_model_label(data)
    rhos = sorted({e["rho"] for e in data})

    print(f"\nLoaded {len(data)} entries from {results_path}")
    print(f"Model: {model_label}  |  ρ values: {rhos}")
    print("Aggregating across all seeds and ρ values…")

    agg = aggregate_by_alpha(data)
    alphas = _sorted_alphas(agg)
    print(f"Aggregated to {len(agg)} entries  |  α values: {alphas}\n")

    print(f"{'Optimizer':8s}  {'alpha':6s}  {'test_acc_mean':13s}  {'test_acc_sem':12s}  {'gen_gap':8s}  {'n_seeds':7s}")
    print("-" * 62)
    for entry in sorted(agg, key=lambda e: (e["optimizer"], e["alpha"])):
        n = len(entry["per_seed"])
        print(
            f"{entry['optimizer']:8s}  {entry['alpha']:<6.2f}"
            f"  {entry['test_acc_mean']:.4f}         "
            f"  {entry['test_acc_sem']:.4f}        "
            f"  {entry['gen_gap_mean']:.4f}    "
            f"  {n}"
        )

    print("\n── Reparametrisation variance (Var of test_acc_mean over α) ──")
    print(f"{'Optimizer':8s}  {'variance':12s}")
    print("-" * 22)
    for opt in OPT_ORDER:
        means = [
            _lookup_agg(agg, opt, a)["test_acc_mean"]
            for a in alphas
            if _lookup_agg(agg, opt, a) is not None
        ]
        print(f"{opt:8s}  {np.var(means):.6f}")

    plot_accuracy(agg, out_dir, model_label)
    plot_gen_gap(agg, out_dir, model_label)

    print("\nAll figures saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        required=True,
        help="Path to reparam_results.json",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for output plots (default: <results_dir>/plots/)",
    )
    args = parser.parse_args()
    main(args.results, args.out_dir)
