"""Experiment 3 — Sharpness analysis (Hutchinson trace estimate).

Loads a trained model checkpoint (or a directory of checkpoints) and computes
the Hutchinson trace estimate tr(H)/d as a sharpness proxy.

For 2D loss landscape visualisation see experiments/plot_landscape.py.

Single-checkpoint mode:
    python experiments/run_flatness.py --config configs/resnet18_baseline.yaml \
        --checkpoint results/runs/<run-id>/checkpoint.pt

Batch mode (all checkpoints in a directory):
    python experiments/run_flatness.py --config configs/resnet18_baseline.yaml \
        --ckpt-dir results/experiments/baseline/resnet18/checkpoints \
        --out-dir  results/experiments/flatness/resnet18
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go

from src.data.cifar10 import get_cifar10_loaders
from src.analysis.hutchinson import hutchinson_trace
from experiments.utils import get_device, set_seed, build_model, save_results

# Filename pattern: <opt>_rho<rho>_seed<seed>.pt
_CKPT_RE = re.compile(r"^(?P<opt>[a-z]+)_rho(?P<rho>[0-9.]+)_seed(?P<seed>\d+)\.pt$")
# Reparam filename pattern: <opt>_rho<rho>_alpha<alpha>_seed<seed>.pt
_REPARAM_CKPT_RE = re.compile(r"^(?P<opt>[a-z]+)_rho(?P<rho>[0-9.]+)_alpha(?P<alpha>[0-9.]+)_seed(?P<seed>\d+)\.pt$")

OPT_STYLE: dict[str, dict] = {
    "sgd":  {"color": "#9E9E9E", "linestyle": "--"},
    "sam":  {"color": "#2196F3", "linestyle": "-"},
    "asam": {"color": "#FF9800", "linestyle": "-"},
    "msam": {"color": "#4CAF50", "linestyle": "-"},
}


def _load_checkpoint(path: str, cfg: dict, device: torch.device) -> nn.Module:
    model = build_model(cfg, device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _compute_sharpness(
    model: nn.Module,
    loss_fn: nn.Module,
    test_loader,
    device: torch.device,
    n_samples: int = 20,
    max_batch: int = 64,
) -> float:
    """Returns the Hutchinson trace estimate tr(H)/d."""
    trace = hutchinson_trace(model, loss_fn, test_loader, device, n_samples=n_samples, max_batch=max_batch)
    print(f"  tr(H)/d = {trace:.6f}")
    return trace


# ── single-checkpoint mode ───────────────────────────────────────────────────

def main(config_path: str, checkpoint: str | None, seed: int, experiment: str = "baseline") -> None:
    """Entry point for main.py CLI dispatch.

    If *checkpoint* is provided, runs single-checkpoint mode.
    Otherwise runs batch mode over the checkpoints saved by run_baseline.py
    or run_reparam.py depending on *experiment* ("baseline" or "reparam").
    """
    with open(config_path) as _f:
        _cfg = yaml.safe_load(_f)
    out_dir = os.path.join(
        _cfg.get("experiments_dir", "./results/experiments"),
        "flatness",
        experiment,
        _cfg["model"],
    )
    if checkpoint is not None:
        main_single(config_path, checkpoint, seed, out_dir)
    else:
        ckpt_dir = os.path.join(
            _cfg.get("experiments_dir", "./results/experiments"),
            experiment, _cfg["model"], "checkpoints",
        )
        main_batch(config_path, ckpt_dir, out_dir, seed, experiment=experiment)


def main_single(config_path: str, checkpoint: str, seed: int, out_dir: str, n_samples: int = 20, max_batch: int = 64) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    set_seed(seed)
    _, test_loader = get_cifar10_loaders(
        data_dir=cfg["data_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        resize=cfg.get("resize"),
        max_samples=cfg.get("max_samples"),
    )
    loss_fn = nn.CrossEntropyLoss()

    name = os.path.basename(checkpoint)
    print(f"\nAnalysing: {name}")
    model = _load_checkpoint(checkpoint, cfg, device)
    trace = _compute_sharpness(model, loss_fn, test_loader, device, n_samples=n_samples, max_batch=max_batch)

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    save_results(os.path.join(out_dir, "sharpness.json"), {name: trace})
    print(f"Data saved to {out_dir}/")


# ── batch mode ───────────────────────────────────────────────────────────────

def main_batch(config_path: str, ckpt_dir: str, out_dir: str, seed: int, n_samples: int = 20, max_batch: int = 64, experiment: str = "baseline") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    os.makedirs(out_dir, exist_ok=True)

    # Discover and sort checkpoints
    ckpt_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    is_reparam = experiment == "reparam"
    entries: list[dict] = []
    for fname in ckpt_files:
        m = (_REPARAM_CKPT_RE if is_reparam else _CKPT_RE).match(fname)
        if not m:
            print(f"Skipping unrecognised file: {fname}")
            continue
        entry: dict = {
            "fname": fname,
            "path": os.path.join(ckpt_dir, fname),
            "opt": m.group("opt"),
            "rho": float(m.group("rho")),
            "seed": int(m.group("seed")),
        }
        if is_reparam:
            entry["alpha"] = float(m.group("alpha"))
        entries.append(entry)

    # Accumulate per-seed results grouped by config key (opt/rho[/alpha])
    # so we can average across seeds rather than silently overwriting them.
    from collections import defaultdict
    traces_by_key: dict[str, list[float]] = defaultdict(list)

    for i, e in enumerate(entries, 1):
        key = f"{e['opt']}_rho{e['rho']}"
        if "alpha" in e:
            key += f"_alpha{e['alpha']}"
        set_seed(e["seed"])
        _, test_loader = get_cifar10_loaders(
            data_dir=cfg["data_dir"],
            batch_size=cfg["batch_size"],
            num_workers=cfg.get("num_workers", 4),
            resize=cfg.get("resize"),
            max_samples=cfg.get("max_samples"),
        )
        print(f"\n[{i}/{len(entries)}] {key} (seed={e['seed']})")
        model = _load_checkpoint(e["path"], cfg, device)
        trace = _compute_sharpness(model, loss_fn, test_loader, device, n_samples=n_samples, max_batch=max_batch)
        traces_by_key[key].append(trace)
        del model, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Incremental save of per-seed-averaged sharpness seen so far
        sharpness_partial = {k: float(np.mean(v)) for k, v in traces_by_key.items()}
        save_results(os.path.join(out_dir, "sharpness_all.json"), sharpness_partial)

    # Average across seeds
    sharpness_all: dict[str, float] = {k: float(np.mean(v)) for k, v in traces_by_key.items()}

    save_results(os.path.join(out_dir, "sharpness_all.json"), sharpness_all)

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    _plot_sharpness_bars(sharpness_all, plots_dir)

    print(f"\nData saved to {out_dir}/  |  Plots saved to {plots_dir}/")


def _opt_from_key(key: str) -> str:
    return key.split("_rho")[0]


def _plot_sharpness_bars(sharpness: dict[str, float], out_dir: str) -> None:
    keys = list(sharpness.keys())
    vals = [sharpness[k] for k in keys]
    colors = [OPT_STYLE.get(_opt_from_key(k), {}).get("color", "#607D8B") for k in keys]
    labels = [k.replace("_rho", " ρ=") for k in keys]

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
    )
    path = os.path.join(out_dir, "sharpness_bars.html")
    fig.write_html(path)
    print(f"Saved → {path}")

    # Also: sharpness vs ρ per optimizer family (line plot)
    _plot_sharpness_vs_rho(sharpness, out_dir)


def _plot_sharpness_vs_rho(sharpness: dict[str, float], out_dir: str) -> None:
    from collections import defaultdict
    by_opt: dict[str, list] = defaultdict(list)
    for key, trace in sharpness.items():
        opt = _opt_from_key(key)
        rho_str = key.split("rho")[1].split("_")[0]
        rho = float(rho_str)
        by_opt[opt].append((rho, trace))

    fig = go.Figure()
    for opt in ("sam", "msam", "asam", "sgd"):
        if opt not in by_opt:
            continue
        pts = sorted(by_opt[opt])
        rhos = [p[0] for p in pts]
        traces_ = [p[1] for p in pts]
        style = OPT_STYLE.get(opt, {})
        mode = "markers" if len(rhos) == 1 else "lines+markers"
        fig.add_trace(go.Scatter(
            x=rhos, y=traces_, mode=mode,
            name=opt.upper(),
            line=dict(color=style.get("color", "gray")),
            marker=dict(color=style.get("color", "gray"), size=8),
        ))

    fig.update_layout(
        title="Sharpness vs ρ (ResNet-18 / CIFAR-10)",
        xaxis_title="Perturbation radius ρ",
        yaxis_title="tr(H) / d  (sharpness)",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    path = os.path.join(out_dir, "sharpness_vs_rho.html")
    fig.write_html(path)
    print(f"Saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    # Single-checkpoint mode
    parser.add_argument("--checkpoint", default=None, help="Path to a single .pt checkpoint")
    # Batch mode
    parser.add_argument("--ckpt-dir", default=None, help="Directory of .pt checkpoints (batch mode)")
    parser.add_argument("--out-dir", default=None, help="Output directory (batch mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Rademacher samples for Hutchinson trace (default: 20)")
    parser.add_argument("--max-batch", type=int, default=64,
                        help="Images per Hessian estimate (default: 64)")
    parser.add_argument("--experiment", default=None, choices=["baseline", "reparam"],
                        help="Checkpoint naming scheme: 'baseline' or 'reparam' (auto-detected from --ckpt-dir path if omitted)")
    args = parser.parse_args()

    if args.ckpt_dir:
        with open(args.config) as _f:
            _cfg = yaml.safe_load(_f)
        _out = args.out_dir or os.path.join(
            _cfg.get("experiments_dir",
                     _cfg.get("flatness_results_dir",
                              _cfg.get("results_dir", "./results/experiments").replace("baseline", "flatness"))),
            "flatness" if "experiments_dir" in _cfg else "",
            _cfg["model"],
        ).replace("//", "/")
        # Auto-detect experiment type from path when not explicitly provided
        _experiment = args.experiment or ("reparam" if "reparam" in args.ckpt_dir else "baseline")
        main_batch(args.config, args.ckpt_dir, _out, args.seed, args.n_samples, args.max_batch, experiment=_experiment)
    elif args.checkpoint:
        with open(args.config) as _f:
            _cfg = yaml.safe_load(_f)
        _out = args.out_dir or os.path.join(
            _cfg.get("experiments_dir",
                     _cfg.get("flatness_results_dir",
                              _cfg.get("results_dir", "./results/experiments").replace("baseline", "flatness"))),
            "flatness" if "experiments_dir" in _cfg else "",
            _cfg["model"],
        ).replace("//", "/")
        main_single(args.config, args.checkpoint, args.seed, _out, args.n_samples, args.max_batch)
    else:
        parser.error("Provide either --checkpoint or --ckpt-dir.")
