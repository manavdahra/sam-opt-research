"""Plot convergence-rate comparison (Priority 2).

Three-tier convergence analysis at accuracy thresholds τ ∈ {90%, 94%, 95%}:

  Panel 1 — Algorithmic:  epochs to reach τ
  Panel 2 — Computational: total GFLOPs to reach τ  (SAM-family = 2× SGD)
  Panel 3 — Wall-clock:   cumulative seconds to reach τ (requires elapsed_sec
                           in history; skipped if data not available)

FLOPs are estimated as:
    forward_flops_per_epoch  = 2 × MACs_per_sample × samples_per_epoch
    backward_flops_per_epoch = 2 × forward_flops_per_epoch
    SGD epoch FLOPs          = forward + backward  = 3 × forward
    SAM epoch FLOPs          = 2 × (forward + backward) = 6 × forward
    (MACs estimated via fvcore if installed, otherwise a ResNet-18 constant)

Usage:
    uv run python experiments/plot_convergence.py \\
        --results results/experiments/baseline/resnet18/baseline_results.json \
        --out-dir results/experiments/baseline/resnet18
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

from src.analysis.metrics import (
    epochs_to_threshold,
    flops_to_threshold,
    wallclock_to_threshold,
)

# ── constants ────────────────────────────────────────────────────────────────

THRESHOLDS = [0.90, 0.94, 0.95]
THRESHOLD_LABELS = ["90%", "94%", "95%"]

OPT_STYLE: dict[str, dict] = {
    "sam":  {"color": "#2196F3", "label": "SAM"},
    "msam": {"color": "#4CAF50", "label": "M-SAM"},
    "asam": {"color": "#FF9800", "label": "ASAM"},
    "sgd":  {"color": "#9E9E9E", "label": "SGD"},
}

# SAM-family runs 2 forward+backward passes per batch; SGD runs 1.
SAM_FAMILY = {"sam", "asam", "msam"}

# ── FLOPs estimation ─────────────────────────────────────────────────────────

def _estimate_macs_per_sample(cfg: dict) -> float:
    """Compute MACs per sample using fvcore FlopCountAnalysis."""
    import torch
    from fvcore.nn import FlopCountAnalysis

    model_name = cfg.get("model", "resnet18")
    if model_name == "resnet18":
        from src.models.resnet18 import get_resnet18
        model = get_resnet18(num_classes=10).eval()
        input_size = cfg.get("resize") or 32
        dummy = torch.zeros(1, 3, input_size, input_size)
    elif model_name == "vit_b_32":
        from src.models.vit import get_vit_b_32
        model = get_vit_b_32(num_classes=10, pretrained=False).eval()
        input_size = cfg.get("resize") or 224
        dummy = torch.zeros(1, 3, input_size, input_size)
    else:
        raise ValueError(f"No FLOPs profile for model: {model_name!r}")

    flops = FlopCountAnalysis(model, dummy)
    flops.unsupported_ops_warnings(False)
    return float(flops.total())


def compute_flops_per_epoch(opt_name: str, macs_per_sample: float, cfg: dict) -> float:
    """Compute total FLOPs for one training epoch.

    Convention (matching standard ML FLOPs counting):
      - 1 MAC = 2 FLOPs
      - forward pass = 2 × MACs
      - backward pass ≈ 2 × forward
      - SGD epoch  = (forward + backward) × n_samples = 6 × MACs × n_samples
      - SAM epoch  = 2 × (forward + backward) × n_samples = 12 × MACs × n_samples
    """
    n_train = 50_000  # CIFAR-10
    # forward = 2 * MACs; backward ≈ 2 * forward = 4 * MACs; total = 6 * MACs
    sgd_epoch_flops = 6.0 * macs_per_sample * n_train
    if opt_name in SAM_FAMILY:
        return 2.0 * sgd_epoch_flops
    return sgd_epoch_flops


# ── data loading ─────────────────────────────────────────────────────────────

def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def best_rho_entries(data: list[dict]) -> dict[str, dict]:
    """Return the entry with highest test_acc_mean per optimizer."""
    best: dict[str, dict] = {}
    for entry in data:
        s = entry["summary"]
        opt = s["optimizer"]
        if opt not in best or s["test_acc_mean"] > best[opt]["summary"]["test_acc_mean"]:
            best[opt] = entry
    return best


def _acc_curves(entry: dict) -> list[list[float]]:
    """Return per-seed test_acc curves from per_seed history lists."""
    curves = []
    for seed_result in entry["per_seed"]:
        history = seed_result.get("history")
        if history:
            curves.append([row["test_acc"] for row in history])
    return curves


def _time_curves(entry: dict) -> list[list[float]]:
    """Return per-seed elapsed_sec curves from per_seed history lists."""
    curves = []
    for seed_result in entry["per_seed"]:
        history = seed_result.get("history")
        if history and "elapsed_sec" in history[0]:
            curves.append([row["elapsed_sec"] for row in history])
    return curves


# ── main plot ─────────────────────────────────────────────────────────────────

def build_figure(
    best: dict[str, dict],
    macs_per_sample: float,
    cfg: dict,
) -> go.Figure:
    opt_order = [o for o in ("sgd", "sam", "asam", "msam") if o in best]
    has_wallclock = any(_time_curves(best[o]) for o in opt_order)
    n_panels = 3 if has_wallclock else 2
    panel_titles = ["Epochs to threshold", "GFLOPs to threshold"]
    if has_wallclock:
        panel_titles.append("Wall-clock (s) to threshold")

    fig = make_subplots(
        rows=1, cols=n_panels,
        subplot_titles=panel_titles,
        horizontal_spacing=0.10,
    )

    for opt in opt_order:
        entry = best[opt]
        style = OPT_STYLE.get(opt, {"color": "#000000", "label": opt.upper()})
        flops_epoch = compute_flops_per_epoch(opt, macs_per_sample, cfg)
        acc_curves = _acc_curves(entry)
        time_curves = _time_curves(entry)

        epochs_vals: list[float | None] = []
        flops_vals: list[float | None] = []
        wallclock_vals: list[float | None] = []

        for tau in THRESHOLDS:
            if acc_curves:
                # Average across seeds
                ep_list = [epochs_to_threshold(c, tau) for c in acc_curves]
                ep_valid = [e for e in ep_list if e is not None]
                ep_mean = sum(ep_valid) / len(ep_valid) if ep_valid else None
            else:
                ep_mean = None

            epochs_vals.append(ep_mean)
            flops_vals.append(flops_to_threshold(flops_epoch / 1e9, ep_mean))  # GFLOPs

            if time_curves and ep_mean is not None:
                wc_list = [wallclock_to_threshold(tc, int(round(ep_mean))) for tc in time_curves]
                wc_valid = [w for w in wc_list if w is not None]
                wc_mean = sum(wc_valid) / len(wc_valid) if wc_valid else None
            else:
                wc_mean = None
            wallclock_vals.append(wc_mean)

        show_legend = True
        for col, vals in enumerate([epochs_vals, flops_vals, wallclock_vals[:n_panels - 1 if n_panels == 2 else 3]], start=1):
            if col > n_panels:
                break
            # Replace None with 0 for display (bar won't render for None otherwise)
            y = [v if v is not None else 0.0 for v in vals]
            text = [f"{v:.1f}" if v is not None else "N/A" for v in vals]
            fig.add_trace(
                go.Bar(
                    name=style["label"],
                    x=THRESHOLD_LABELS,
                    y=y,
                    text=text,
                    textposition="outside",
                    marker_color=style["color"],
                    showlegend=show_legend,
                    legendgroup=opt,
                ),
                row=1, col=col,
            )
            show_legend = False  # only show once in legend

    fig.update_layout(
        title="Convergence-rate comparison (best-ρ per optimizer)",
        barmode="group",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Accuracy threshold τ")
    fig.update_yaxes(title_text="Epochs", row=1, col=1)
    fig.update_yaxes(title_text="GFLOPs", row=1, col=2)
    if has_wallclock:
        fig.update_yaxes(title_text="Seconds", row=1, col=3)

    return fig


def main(results_path: str, out_dir: str, config_path: str | None) -> None:
    data = load_results(results_path)
    best = best_rho_entries(data)

    # Try to load config for FLOPs estimation
    cfg: dict = {}
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        # Infer config path next to results
        candidate = os.path.join(
            _ROOT, "configs",
            os.path.basename(os.path.dirname(results_path)) + "_baseline.yaml",
        )
        if os.path.exists(candidate):
            import yaml
            with open(candidate) as f:
                cfg = yaml.safe_load(f)

    macs_per_sample = _estimate_macs_per_sample(cfg)
    print(f"MACs per sample: {macs_per_sample / 1e6:.0f} M")

    # Warn if histories are missing (pre-existing results without history)
    acc_available = any(_acc_curves(e) for e in best.values())
    if not acc_available:
        print(
            "WARNING: No per-epoch history found in results JSON.\n"
            "  Convergence plots require re-running experiments with the updated\n"
            "  run_baseline.py (which now persists per-epoch history).\n"
            "  Exiting without producing output."
        )
        return

    fig = build_figure(best, macs_per_sample, cfg)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "convergence.html")
    fig.write_html(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", required=True, help="Path to baseline_results.json"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (defaults to same directory as results JSON)",
    )
    parser.add_argument(
        "--config", default=None, help="Path to YAML config (for FLOPs estimation)"
    )
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.dirname(args.results) or "."
    main(args.results, out_dir, args.config)
