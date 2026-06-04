import argparse
import json
from pathlib import Path
import os
import sys
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RESULTS_DIR = Path(_ROOT) / "results"

METRICS_PATHS = {
    "resnet18": {
        "baseline": RESULTS_DIR / "resnet18/experiments/baseline/resnet18/baseline_results.json",
        "reparam":  RESULTS_DIR / "resnet18/experiments/reparam/resnet18/reparam_results.json",
    },
    "vit_b_32": {
        "baseline": RESULTS_DIR / "vit_b_32/experiments/baseline/vit_b_32/baseline_results.json",
        "reparam":  RESULTS_DIR / "vit_b_32/experiments/reparam/vit_b_32/reparam_results.json",
    },
}

def get_metrics(model: str, optimizer: str, rho: float, seed: int, alpha: float = None) -> dict:
    is_baseline = alpha is None
    data = json.loads(METRICS_PATHS[model]["baseline" if is_baseline else "reparam"].read_text())

    for entry in data:
        summary = entry["summary"] if is_baseline else entry
        if summary["optimizer"] != optimizer or not math.isclose(summary["rho"], rho):
            continue
        if not is_baseline and not math.isclose(summary["alpha"], alpha):
            continue
        for s in entry["per_seed"]:
            if s["seed"] != seed:
                continue
            return dict(
                history=s["history"],
                train_loss=s["train_loss"],
                test_loss=s["test_loss"],
                divergence_rate=s.get("divergence_rate"),
                elapsed_sec=s["elapsed_sec"],
            )

    print(f"Metrics not found: model={model}, optimizer={optimizer}, rho={rho}, seed={seed}, alpha={alpha}")
    return None


def discover_configs(model: str, optimizer: str) -> dict[str, list]:
    """Return all rho, seed, and alpha values available for a given model and optimizer.

    Returns a dict with keys:
        "baseline": list of (rho, seed) tuples from the baseline results.
        "reparam":  list of (rho, alpha, seed) tuples from the reparam results,
                    or an empty list if no reparam results exist.
    """
    baseline_path = METRICS_PATHS[model]["baseline"]
    reparam_path  = METRICS_PATHS[model]["reparam"]

    baseline_configs: list[tuple[float, int]] = []
    if baseline_path.exists():
        for entry in json.loads(baseline_path.read_text()):
            s = entry["summary"]
            if s["optimizer"] != optimizer:
                continue
            for seed_result in entry["per_seed"]:
                baseline_configs.append((s["rho"], seed_result["seed"]))

    reparam_configs: list[tuple[float, float, int]] = []
    if reparam_path.exists():
        for entry in json.loads(reparam_path.read_text()):
            if entry["optimizer"] != optimizer:
                continue
            for seed_result in entry["per_seed"]:
                reparam_configs.append((entry["rho"], seed_result["seed"], entry["alpha"]))
    
    # Sort by rho, then alpha, then seed for consistent ordering.
    baseline_configs.sort(key=lambda x: (x[0], x[1]))
    reparam_configs.sort(key=lambda x: (x[0], x[1], x[2]))

    return {"baseline": baseline_configs, "reparam": reparam_configs}


OPT_STYLE: dict[str, dict] = {
    "sgd":  {"color": "#7B8794", "label": "SGD"},
    "sam":  {"color": "#4878CF", "label": "SAM"},
    "asam": {"color": "#C8703A", "label": "ASAM"},
    "msam": {"color": "#3D8C6E", "label": "M-SAM"},
}


def main(model, optimizer, out_dir, ncols: int = 3):
    """Collect every (rho, seed) baseline config and its matching reparam runs,
    then render all curves into a single HTML file as a subplot grid."""
    configs = discover_configs(model, optimizer)
    baseline_configs = configs["baseline"]   # [(rho, seed), ...]
    reparam_configs  = configs["reparam"]    # [(rho, seed, alpha), ...]

    # Build one panel per baseline (rho, seed); within each panel overlay all
    # matching reparam alpha values as well.
    panels: list[dict] = []
    for rho, seed in baseline_configs:
        matching_alphas = [alpha for r, s, alpha in reparam_configs if math.isclose(r, rho) and s == seed]
        panels.append({"rho": rho, "seed": seed, "alphas": matching_alphas})

    n = len(panels)
    if n == 0:
        print(f"No configs found for model={model}, optimizer={optimizer}")
        return

    nrows = math.ceil(n / ncols)
    subtitles = [
        f"{OPT_STYLE.get(optimizer, {}).get('label', optimizer.upper())} rho={p['rho']} seed={p['seed']}"
        for p in panels
    ]

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subtitles,
        shared_yaxes=False,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for idx, panel in enumerate(panels):
        row, col = idx // ncols + 1, idx % ncols + 1
        rho, seed, alphas = panel["rho"], panel["seed"], panel["alphas"]
        show_legend = idx == 0

        # Baseline traces
        baseline_data = get_metrics(model, optimizer, rho, seed)
        if baseline_data is not None:
            df = pd.DataFrame(baseline_data["history"])
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df["train_loss"], mode="lines",
                name="Baseline — Train",
                line=dict(color="#78d57e", dash="dash", width=1),
                showlegend=show_legend, legendgroup="bl_train",
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df["test_loss"], mode="lines",
                name="Baseline — Test",
                line=dict(color="#78d57e", width=1),
                showlegend=show_legend, legendgroup="bl_test",
            ), row=row, col=col)

        # Reparam traces — one colour per alpha
        alpha_colors = ["#da655d", "#9b59b6", "#e67e22", "#1abc9c"]
        for a_idx, alpha in enumerate(alphas):
            reparam_data = get_metrics(model, optimizer, rho, seed, alpha)
            if reparam_data is None:
                continue
            df_r = pd.DataFrame(reparam_data["history"])
            color = alpha_colors[a_idx % len(alpha_colors)]
            fig.add_trace(go.Scatter(
                x=df_r["epoch"], y=df_r["train_loss"], mode="lines",
                name=f"Reparam alpha={alpha} — Train",
                line=dict(color=color, dash="dash", width=1),
                showlegend=show_legend, legendgroup=f"rp{a_idx}_train",
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=df_r["epoch"], y=df_r["test_loss"], mode="lines",
                name=f"Reparam alpha={alpha} — Test",
                line=dict(color=color, width=1),
                showlegend=show_legend, legendgroup=f"rp{a_idx}_test",
            ), row=row, col=col)

    fig.update_layout(
        title=f"Loss Curves — {model} | {optimizer.upper()}",
        font=dict(size=12),
        title_font=dict(size=18),
        height=320 * nrows,
        width=320 * ncols,
        legend=dict(orientation="v", xanchor="left", x=1.02, yanchor="middle", y=0.5, font=dict(size=11)),
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")

    os.makedirs(out_dir, exist_ok=True)
    fname = f"loss_curves_{model}_{optimizer}.html"
    path = os.path.join(out_dir, fname)
    fig.write_html(path)
    print(f"Saved at {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot all loss curves to HTML.")
    parser.add_argument("--model", default="resnet18", choices=list(METRICS_PATHS.keys()))
    parser.add_argument("--optimizer", default="sam", choices=list(OPT_STYLE.keys()))
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: results/<model>/experiments/plots/)")
    parser.add_argument("--ncols", type=int, default=3,
                        help="Number of subplot columns (default: 3)")
    args = parser.parse_args()

    _out = args.out_dir or str(RESULTS_DIR / args.model / "experiments" / "plots")
    main(args.model, args.optimizer, _out, ncols=args.ncols)
