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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.cifar10 import get_cifar10_loaders
from src.analysis.hutchinson import hutchinson_trace
from src.analysis.landscape import loss_landscape_1d, plot_loss_landscape
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
) -> tuple[float, list, list]:
    """Returns (trace, alphas_list, losses_list)."""
    trace = hutchinson_trace(model, loss_fn, test_loader, device, n_samples=n_samples)
    print(f"  tr(H)/d = {trace:.6f}")
    alphas, losses = loss_landscape_1d(model, loss_fn, test_loader, device, steps=51, range_=1.0)
    return trace, alphas.tolist(), losses.tolist()


# ── single-checkpoint mode ───────────────────────────────────────────────────

def main_single(config_path: str, checkpoint: str, seed: int, out_dir: str, n_samples: int = 20) -> None:
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
    trace, alphas, losses = _analyse_one(name, model, loss_fn, test_loader, device, n_samples=n_samples)

    os.makedirs(out_dir, exist_ok=True)
    save_results(os.path.join(out_dir, "sharpness.json"), {name: trace})
    save_results(os.path.join(out_dir, "landscape.json"), {name: (alphas, losses)})
    plot_loss_landscape(
        {name: (np.array(alphas), np.array(losses))},
        save_path=os.path.join(out_dir, "landscape.png"),
    )
    print(f"Outputs saved to {out_dir}/")


# ── batch mode ───────────────────────────────────────────────────────────────

def main_batch(config_path: str, ckpt_dir: str, out_dir: str, seed: int, n_samples: int = 20) -> None:
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
        trace, alphas, losses = _analyse_one(key, model, loss_fn, test_loader, device, n_samples=n_samples)
        sharpness_all[key] = trace
        landscape_all[key] = (alphas, losses)
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

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.7), 4.5))
    bars = ax.bar(range(len(keys)), vals, color=colors, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_rho", "\nρ=") for k in keys], fontsize=8)
    ax.set_ylabel("tr(H) / d  (sharpness)", fontsize=11)
    ax.set_title("Hutchinson Sharpness Estimate — All Checkpoints", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, "sharpness_bars.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
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

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt in ("sam", "msam", "asam", "sgd"):
        if opt not in by_opt:
            continue
        pts = sorted(by_opt[opt])
        rhos = [p[0] for p in pts]
        traces = [p[1] for p in pts]
        style = OPT_STYLE.get(opt, {})
        if len(rhos) == 1:
            ax.scatter(rhos, traces, color=style.get("color", "gray"),
                       s=80, zorder=4, label=opt.upper())
        else:
            ax.plot(rhos, traces, color=style.get("color", "gray"),
                    linestyle=style.get("linestyle", "-"),
                    marker="o", linewidth=2, markersize=6, label=opt.upper())

    ax.set_xlabel("Perturbation radius ρ", fontsize=12)
    ax.set_ylabel("tr(H) / d  (sharpness)", fontsize=12)
    ax.set_title("Sharpness vs ρ (ResNet-18 / CIFAR-10)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "sharpness_vs_rho.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def _plot_landscape_comparison(landscape: dict[str, tuple], out_dir: str) -> None:
    histories = {k: (np.array(v[0]), np.array(v[1])) for k, v in landscape.items()}
    # Colour by optimizer family
    styled: dict[str, tuple] = {}
    for k, (alphas, losses) in histories.items():
        styled[k] = (alphas, losses)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, (alphas, losses) in sorted(styled.items()):
        opt = _opt_from_key(key)
        style = OPT_STYLE.get(opt, {})
        ax.plot(alphas, losses, color=style.get("color", "gray"),
                linestyle=style.get("linestyle", "-"), alpha=0.7,
                linewidth=1.5, label=key.replace("_rho", " ρ="))
    ax.set_xlabel("Perturbation α", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("1D Loss Landscape — All Checkpoints", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "landscape_all.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def _plot_landscape_best(landscape: dict[str, tuple], out_dir: str) -> None:
    """One line per optimizer family using the sharpest-minimum (flattest) ρ."""
    from collections import defaultdict
    by_opt: dict[str, list] = defaultdict(list)
    for key, (alphas, losses) in landscape.items():
        opt = _opt_from_key(key)
        rho = float(key.split("rho")[1])
        # Use centre loss (index 25 = α=0) as sharpness proxy; pick flattest
        centre_loss = np.array(losses)[len(losses) // 2]
        by_opt[opt].append((centre_loss, rho, key, alphas, losses))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt in ("sgd", "sam", "msam", "asam"):
        if opt not in by_opt:
            continue
        # Pick the flattest minimum = smallest centre loss (most regularised)
        best = min(by_opt[opt], key=lambda x: x[0])
        _, rho, key, alphas, losses = best
        style = OPT_STYLE.get(opt, {})
        label = f"{opt.upper()} ρ={rho}" if opt != "sgd" else "SGD"
        ax.plot(np.array(alphas), np.array(losses),
                color=style.get("color", "gray"),
                linestyle=style.get("linestyle", "-"),
                linewidth=2, label=label)

    ax.set_xlabel("Perturbation α", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("1D Loss Landscape — Best ρ per Optimizer", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "landscape_best.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
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
    args = parser.parse_args()

    if args.ckpt_dir:
        with open(args.config) as _f:
            _cfg = yaml.safe_load(_f)
        _out = args.out_dir or os.path.join(
            _cfg.get("results_dir", "./results/baseline").replace("baseline", "flatness"),
            _cfg["model"],
        )
        main_batch(args.config, args.ckpt_dir, _out, args.seed, args.n_samples)
    elif args.checkpoint:
        with open(args.config) as _f:
            _cfg = yaml.safe_load(_f)
        _out = args.out_dir or os.path.join(
            _cfg.get("results_dir", "./results/baseline").replace("baseline", "flatness"),
            _cfg["model"],
        )
        main_single(args.config, args.checkpoint, args.seed, _out, args.n_samples)
    else:
        parser.error("Provide either --checkpoint or --ckpt-dir.")
