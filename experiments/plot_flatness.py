import argparse
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import plotly.graph_objects as go

from experiments.utils import OPT_STYLE, load_results, save_figure, resolve_out_dir


def plot_sharpness_bars(sharpness: dict[str, float], out_dir: str) -> None:
    """Bar chart of tr(H)/d for every checkpoint key."""
    keys = list(sharpness.keys())
    vals = [sharpness[k] for k in keys]
    opt_keys = [k.split("_rho")[0] for k in keys]
    colors = [OPT_STYLE.get(o, {}).get("color", "#607D8B") for o in opt_keys]
    labels = [k.replace("_rho", " rho=") for k in keys]

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=colors,
        text=[f"{v:.6f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="Hutchinson Sharpness Estimate — All Checkpoints",
        xaxis_title="Checkpoint",
        yaxis_title="tr(H) / d  (sharpness)",
        template="plotly_white",
        font=dict(size=16),
        title_font=dict(size=18),
        width=800,
        height=500,
    )
    save_figure(fig, out_dir, "sharpness_bars")


def plot_sharpness_vs_rho(sharpness: dict[str, float], out_dir: str) -> None:
    """Line plot of sharpness vs rho, one trace per optimizer family."""
    by_opt: dict[str, list] = defaultdict(list)
    for key, val in sharpness.items():
        opt = key.split("_rho")[0]
        rho = float(key.split("rho")[1].split("_")[0])
        by_opt[opt].append((rho, val))

    fig = go.Figure()
    for opt in ("sam", "msam", "asam", "sgd"):
        if opt not in by_opt:
            continue
        pts = sorted(by_opt[opt])
        rhos = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        style = OPT_STYLE[opt]
        mode = "markers" if len(rhos) == 1 else "lines+markers"
        fig.add_trace(go.Scatter(
            x=rhos, y=vals, mode=mode,
            name=style["label"],
            line=dict(color=style["color"]),
            marker=dict(color=style["color"], size=8, symbol=style["symbol"]),
        ))

    fig.update_layout(
        title="Sharpness vs rho (ResNet-18 / CIFAR-10)",
        xaxis_title="Perturbation radius rho",
        yaxis_title="tr(H) / d  (sharpness)",
        template="plotly_white",
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top", font=dict(size=12)),
        font=dict(size=16),
        title_font=dict(size=18),
        width=800,
        height=500,
    )
    save_figure(fig, out_dir, "sharpness_vs_rho")


def plot_all(sharpness: dict[str, float], out_dir: str) -> None:
    """Produce all sharpness plots for the given results dict."""
    plot_sharpness_bars(sharpness, out_dir)
    plot_sharpness_vs_rho(sharpness, out_dir)


def main(results_path: str, out_dir: str | None = None) -> None:
    sharpness = load_results(results_path)
    out_dir = resolve_out_dir(results_path, out_dir)

    print(f"\nLoaded {len(sharpness)} entries from {results_path}")
    print("\n── Sharpness table ──")
    print(f"{'Key':40s}  {'tr(H)/d':>12s}")
    print("-" * 55)
    for key, val in sharpness.items():
        print(f"{key:40s}  {val:12.6f}")

    plot_all(sharpness, out_dir)
    print("\nAll figures saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sharpness results from sharpness_all.json.")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to sharpness_all.json produced by run_flatness.py",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for output plots (default: <results_dir>/plots/)",
    )
    args = parser.parse_args()
    main(args.results, args.out_dir)
