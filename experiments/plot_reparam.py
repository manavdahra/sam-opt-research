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

# Plotly Colormap for optimizers
OPT_STYLE: dict[str, dict] = {
    "sgd":  {"color": "#7B8794", "symbol": "diamond",     "label": "SGD"},
    "sam":  {"color": "#4878CF", "symbol": "circle",      "label": "SAM"},
    "asam": {"color": "#C8703A", "symbol": "triangle-up", "label": "ASAM"},
    "msam": {"color": "#3D8C6E", "symbol": "square",      "label": "M-SAM"},
}

# Display order for optimizers in legends and summaries
OPT_ORDER = ("sgd", "sam", "asam", "msam")


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def sorted_alphas(agg: list[dict]) -> list[float]:
    return sorted({e["alpha"] for e in agg})


def infer_model_label(data: list[dict]) -> str:
    """Infer a human-friendly model label from the 'model' field in the data entries."""
    models = {e.get("model", "") for e in data}
    m = next(iter(models), "")
    return {"resnet18": "ResNet-18", "vit_b_32": "ViT-B/32"}.get(m, m)


def aggregate_by_opt_alpha(data: list[dict]) -> list[dict]:
    """Pool per_seed entries across every rho for each (optimizer, alpha) pair.

    Returns one entry per (optimizer, alpha) with:
      - test_acc_mean / test_acc_sem  — from pooled test_acc values
      - gen_gap_mean                  — mean of (test_loss - train_loss) in the pool
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
        gaps = []
        for s in seeds:
            if "test_loss" in s and "train_loss" in s:
                test_loss = s["test_loss"]
                train_loss = s["train_loss"]
                if not (math.isnan(test_loss) or math.isnan(train_loss)):
                    gaps.append(test_loss - train_loss)

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


def lookup_agg(agg: list[dict], optimizer: str, alpha: float) -> dict | None:
    for e in agg:
        if e["optimizer"] == optimizer and e["alpha"] == alpha:
            return e
    return None


def plot_accuracy_vs_alpha(agg: list[dict], out_dir: str, model_label: str) -> None:
    """Bar chart for Test accuracy vs alpha, each bar is grouped per optimizer, error bars are SEM."""
    alphas = sorted_alphas(agg)
    x_labels = [str(a) for a in alphas]

    fig = go.Figure()

    for opt in OPT_ORDER:
        style = OPT_STYLE.get(opt, {"color": "#000", "symbol": "circle", "label": opt})
        accs, sems = [], []
        min_acc, max_acc = float("inf"), float("-inf")
        for alpha in alphas:
            entry = lookup_agg(agg, opt, alpha)
            accs.append(entry["test_acc_mean"] if entry else None)
            sems.append(entry["test_acc_sem"] if entry else None)
            if entry:
                min_acc = min(min_acc, entry["test_acc_mean"])
                max_acc = max(max_acc, entry["test_acc_mean"])

        if all(a is None for a in accs):
            # skip optimizers with no data
            continue

        fig.add_trace(go.Bar(
            name=style["label"],
            x=x_labels,
            y=accs,
            error_y=dict(type="data", array=sems, visible=True),
            marker_color=style["color"],
        ))

    fig.update_layout(
        title=f"Test Accuracy vs Reparametrisation Scale alpha  ({model_label} / CIFAR-10)",
        xaxis_title="alpha (reparametrisation scale)",
        yaxis_title="Test accuracy",
        yaxis_tickformat=".4f",
        yaxis_rangemode="tozero",
        barmode="group",
        bargap=0.35,
        bargroupgap=0.08,
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
        font=dict(size=15),
        title_font=dict(size=17),
        width=800,
        height=500,
    )
    fig.update_yaxes(range=[min_acc - 0.01, max_acc + 0.01]) # Zoom in on the accuracy range for better visibility

    path = os.path.join(out_dir, "reparam_accuracy.html")
    fig.write_html(path)
    print(f"Saved at {path}")
    png_path = os.path.join(out_dir, "reparam_accuracy.png")
    fig.write_image(png_path, width=800, height=500, scale=2)
    print(f"Saved at {png_path}")


def plot_gen_gap_vs_alpha(agg: list[dict], out_dir: str, model_label: str) -> None:
    """Bar chart for Generalization gap vs alpha, one bar group per optimizer."""
    alphas = sorted_alphas(agg)
    x_labels = [str(a) for a in alphas]

    fig = go.Figure()

    for opt in OPT_ORDER:
        style = OPT_STYLE.get(opt, {"color": "#000", "symbol": "circle", "label": opt})
        gaps = []
        for alpha in alphas:
            entry = lookup_agg(agg, opt, alpha)
            gaps.append(entry["gen_gap_mean"] if entry else None)

        if all(g is None or (g is not None and math.isnan(g)) for g in gaps):
            # skip optimizers with no data
            continue

        fig.add_trace(go.Bar(
            name=style["label"],
            x=x_labels,
            y=gaps,
            marker_color=style["color"],
        ))

    fig.update_layout(
        title=f"Generalisation Gap vs alpha  ({model_label} / CIFAR-10)",
        xaxis_title="alpha (reparametrisation scale)",
        yaxis_title="Generalisation gap (test loss - train loss)",
        yaxis_tickformat=".4f",
        yaxis_rangemode="tozero",
        barmode="group",
        bargap=0.35,
        bargroupgap=0.08,
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top"),
        font=dict(size=15),
        title_font=dict(size=17),
        width=800,
        height=500,
    )
    fig.update_yaxes(range=[0.1, 0.2])  # Zoom in on the generalisation gap for better visibility

    path = os.path.join(out_dir, "reparam_gen_gap.html")
    fig.write_html(path)
    print(f"Saved at {path}")
    png_path = os.path.join(out_dir, "reparam_gen_gap.png")
    fig.write_image(png_path, width=800, height=500, scale=2)
    print(f"Saved at {png_path}")


def main(results_path: str, out_dir: str | None = None) -> None:
    data = load_results(results_path)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(results_path) or ".", "plots")
    os.makedirs(out_dir, exist_ok=True)

    model_label = infer_model_label(data)
    rhos = sorted({e["rho"] for e in data})

    print(f"\nLoaded {len(data)} entries from {results_path}")
    print(f"Model: {model_label}  |  rho values: {rhos}")
    print("Aggregating across all seeds and rho values for each (optimizer, alpha) pair...")

    agg = aggregate_by_opt_alpha(data)
    alphas = sorted_alphas(agg)
    print(f"Aggregated to {len(agg)} entries  |  alpha values: {alphas}\n")

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

    plot_accuracy_vs_alpha(agg, out_dir, model_label)
    plot_gen_gap_vs_alpha(agg, out_dir, model_label)

    print("\nPlots saved.")


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
