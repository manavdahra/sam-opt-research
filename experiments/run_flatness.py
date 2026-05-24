"""Experiment 3 — Flatness / sharpness analysis.

Loads a trained model checkpoint (or trains a fresh one if no checkpoint is
provided) and computes:
  1. Hutchinson trace estimate (tr(H)/d) — sharpness proxy.
  2. 1D loss landscape along a random filter-normalized direction.

Single-checkpoint mode:
    python experiments/run_flatness.py --config configs/resnet18_baseline.yaml \
        --checkpoint results/results/baseline/resnet18/checkpoints/sam_rho0.05_seed42.pt

Batch mode (all checkpoints in a directory):
    python experiments/run_flatness.py --config configs/resnet18_baseline.yaml \
        --ckpt-dir results/results/baseline/resnet18/checkpoints \
        --out-dir  results/results/flatness/resnet18
"""
from __future__ import annotations

import argparse
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
from plotly.subplots import make_subplots

from src.data.cifar10 import get_cifar10_loaders
from src.analysis.hutchinson import hutchinson_trace
from src.analysis.landscape import loss_landscape_2d, plot_loss_landscape_2d
from experiments.utils import get_device, set_seed, build_model, save_results

# Filename pattern: <opt>_rho<rho>_seed<seed>.pt
_CKPT_RE = re.compile(r"^(?P<opt>[a-z]+)_rho(?P<rho>[0-9.]+)_seed(?P<seed>\d+)\.pt$")

OPT_STYLE: dict[str, dict] = {
    "sam":  {"color": "#2196F3", "linestyle": "-"},
    "msam": {"color": "#4CAF50", "linestyle": "-"},
    "asam": {"color": "#FF9800", "linestyle": "-"},
    "sgd":  {"color": "#9E9E9E", "linestyle": "--"},
}


def _load_checkpoint(path: str, cfg: dict, device: torch.device) -> nn.Module:
    model = build_model(cfg, device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _analyse_one(
    name: str,
    model: nn.Module,
    loss_fn: nn.Module,
    test_loader,
    device: torch.device,
    n_samples: int = 20,
    max_batch: int = 64,
) -> tuple[float, list, list, list]:
    """Returns (trace, alphas_list, betas_list, losses_2d_list)."""
    trace = hutchinson_trace(model, loss_fn, test_loader, device, n_samples=n_samples, max_batch=max_batch)
    print(f"  tr(H)/d = {trace:.6f}")
    alphas, betas, losses = loss_landscape_2d(model, loss_fn, test_loader, device, steps=31, range_=1.0)
    return trace, alphas.tolist(), betas.tolist(), losses.tolist()


# ── single-checkpoint mode ───────────────────────────────────────────────────

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
    )
    loss_fn = nn.CrossEntropyLoss()

    name = os.path.basename(checkpoint)
    print(f"\nAnalysing: {name}")
    model = _load_checkpoint(checkpoint, cfg, device)
    trace, alphas, betas, losses = _analyse_one(name, model, loss_fn, test_loader, device, n_samples=n_samples, max_batch=max_batch)

    os.makedirs(out_dir, exist_ok=True)
    save_results(os.path.join(out_dir, "sharpness.json"), {name: trace})
    save_results(os.path.join(out_dir, "landscape.json"), {name: (alphas, betas, losses)})
    plot_loss_landscape_2d(
        np.array(alphas), np.array(betas), np.array(losses),
        title=f"2D Loss Landscape — {name}",
        save_path=os.path.join(out_dir, "landscape.html"),
    )
    print(f"Outputs saved to {out_dir}/")


# ── batch mode ───────────────────────────────────────────────────────────────

def main_batch(config_path: str, ckpt_dir: str, out_dir: str, seed: int, n_samples: int = 20, max_batch: int = 64) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    set_seed(seed)
    _, test_loader = get_cifar10_loaders(
        data_dir=cfg["data_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        resize=cfg.get("resize"),
    )
    loss_fn = nn.CrossEntropyLoss()
    os.makedirs(out_dir, exist_ok=True)

    # Discover and sort checkpoints
    ckpt_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    entries: list[dict] = []
    for fname in ckpt_files:
        m = _CKPT_RE.match(fname)
        if not m:
            print(f"Skipping unrecognised file: {fname}")
            continue
        entries.append({
            "fname": fname,
            "path": os.path.join(ckpt_dir, fname),
            "opt": m.group("opt"),
            "rho": float(m.group("rho")),
            "seed": int(m.group("seed")),
        })

    sharpness_all: dict[str, float] = {}
    landscape_all: dict[str, tuple] = {}

    for i, e in enumerate(entries, 1):
        key = f"{e['opt']}_rho{e['rho']}"
        print(f"\n[{i}/{len(entries)}] {key} (seed={e['seed']})")
        model = _load_checkpoint(e["path"], cfg, device)
        trace, alphas, betas, losses = _analyse_one(key, model, loss_fn, test_loader, device, n_samples=n_samples, max_batch=max_batch)
        sharpness_all[key] = trace
        landscape_all[key] = (alphas, betas, losses)
        # Incremental save after each checkpoint
        save_results(os.path.join(out_dir, "sharpness_all.json"), sharpness_all)

    save_results(os.path.join(out_dir, "landscape_all.json"), landscape_all)

    # ── Plot 1: sharpness bar chart ──────────────────────────────────────────
    _plot_sharpness_bars(sharpness_all, out_dir)

    # ── Plot 2: landscape comparison (one line per checkpoint) ───────────────
    _plot_landscape_comparison(landscape_all, out_dir)

    # ── Plot 3: landscape comparison — best ρ per optimizer ──────────────────
    _plot_landscape_best(landscape_all, out_dir)

    print(f"\nAll flatness outputs saved to {out_dir}/")


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
        rho = float(key.split("rho")[1])
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


def _plot_landscape_comparison(landscape: dict[str, tuple], out_dir: str) -> None:
    Z_CLIP = 5.0
    keys = sorted(landscape.keys())
    n = len(keys)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    titles = [k.replace("_rho", " ρ=") for k in keys]

    specs = [[{"type": "scene"} for _ in range(ncols)] for _ in range(nrows)]
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=titles,
        specs=specs,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )
    scene_updates = {}
    for idx, key in enumerate(keys):
        alphas, betas, losses = landscape[key]
        Z = np.clip(np.array(losses), 0, Z_CLIP)
        row, col = idx // ncols + 1, idx % ncols + 1
        fig.add_trace(go.Surface(
            z=Z.tolist(),
            x=alphas if isinstance(alphas, list) else alphas.tolist(),
            y=betas if isinstance(betas, list) else betas.tolist(),
            colorscale="RdBu_r",
            showscale=(idx == 0),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        ), row=row, col=col)
        scene_key = "scene" if idx == 0 else f"scene{idx + 1}"
        scene_updates[scene_key] = dict(
            xaxis_title="α", yaxis_title="β", zaxis_title="Loss",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        )

    fig.update_layout(
        title="3D Loss Landscape — All Checkpoints",
        template="plotly_white",
        height=380 * nrows,
        **scene_updates,
    )
    path = os.path.join(out_dir, "landscape_all.html")
    fig.write_html(path)
    print(f"Saved → {path}")


def _plot_landscape_best(landscape: dict[str, tuple], out_dir: str) -> None:
    """2×2 contour grid — best ρ per optimizer (lowest centre loss)."""
    from collections import defaultdict
    Z_CLIP = 5.0
    by_opt: dict[str, list] = defaultdict(list)
    for key, (alphas, betas, losses) in landscape.items():
        opt = _opt_from_key(key)
        rho = float(key.split("rho")[1])
        losses_arr = np.array(losses)
        mid = losses_arr.shape[0] // 2
        centre_loss = losses_arr[mid, mid]
        by_opt[opt].append((centre_loss, rho, alphas, betas, losses))

    opt_order = [o for o in ("sgd", "sam", "msam", "asam") if o in by_opt]
    ncols = 2
    nrows = (len(opt_order) + ncols - 1) // ncols
    titles = []
    best_data = []
    for opt in opt_order:
        best = min(by_opt[opt], key=lambda x: x[0])
        _, rho, alphas, betas, losses = best
        titles.append(f"{opt.upper()} ρ={rho}" if opt != "sgd" else "SGD")
        best_data.append((alphas, betas, losses))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        specs=[[{"type": "scene"} for _ in range(ncols)] for _ in range(nrows)],
        subplot_titles=titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )
    for idx, (alphas, betas, losses) in enumerate(best_data):
        Z = np.clip(np.array(losses), 0, Z_CLIP).T
        row, col = idx // ncols + 1, idx % ncols + 1
        fig.add_trace(go.Surface(
            z=Z.tolist(),
            x=alphas if isinstance(alphas, list) else alphas.tolist(),
            y=betas if isinstance(betas, list) else betas.tolist(),
            colorscale="RdBu_r",
            showscale=(idx == 0),
            colorbar=dict(title="Loss", x=1.02),
        ), row=row, col=col)

    # Apply consistent camera angle to all scene axes
    scene_cfg = dict(
        xaxis_title="α", yaxis_title="β", zaxis_title="Loss",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    )
    scene_updates = {}
    for idx in range(len(best_data)):
        key = "scene" if idx == 0 else f"scene{idx + 1}"
        scene_updates[key] = scene_cfg

    fig.update_layout(
        title="2D Loss Landscape — Best ρ per Optimizer",
        template="plotly_white",
        height=520 * nrows,
        **scene_updates,
    )
    path = os.path.join(out_dir, "landscape_best.html")
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
    args = parser.parse_args()

    if args.ckpt_dir:
        with open(args.config) as _f:
            _cfg = yaml.safe_load(_f)
        _out = args.out_dir or os.path.join(
            _cfg.get("results_dir", "./results/baseline").replace("baseline", "flatness"),
            _cfg["model"],
        )
        main_batch(args.config, args.ckpt_dir, _out, args.seed, args.n_samples, args.max_batch)
    elif args.checkpoint:
        with open(args.config) as _f:
            _cfg = yaml.safe_load(_f)
        _out = args.out_dir or os.path.join(
            _cfg.get("results_dir", "./results/baseline").replace("baseline", "flatness"),
            _cfg["model"],
        )
        main_single(args.config, args.checkpoint, args.seed, _out, args.n_samples, args.max_batch)
    else:
        parser.error("Provide either --checkpoint or --ckpt-dir.")
