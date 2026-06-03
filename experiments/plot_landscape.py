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

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.cifar10 import get_cifar10_loaders
from src.analysis.landscape import loss_landscape_2d, plot_loss_landscape_2d
from experiments.utils import get_device, set_seed, build_model, save_results

# Baseline filename pattern: <opt>_rho<rho>_seed<seed>.pt to look for in results directory 
_CKPT_RE = re.compile(r"^(?P<opt>[a-z]+)_rho(?P<rho>[0-9.]+)_seed(?P<seed>\d+)\.pt$")
# Reparam filename pattern: <opt>_rho<rho>_alpha<alpha>_seed<seed>.pt to look for in results directory
_REPARAM_CKPT_RE = re.compile(r"^(?P<opt>[a-z]+)_rho(?P<rho>[0-9.]+)_alpha(?P<alpha>[0-9.]+)_seed(?P<seed>\d+)\.pt$")

# Plotly Colormap for optimizers
OPT_STYLE: dict[str, dict] = {
    "sgd":  {"color": "#9E9E9E", "linestyle": "--"},
    "sam":  {"color": "#2196F3", "linestyle": "-"},
    "asam": {"color": "#FF9800", "linestyle": "-"},
    "msam": {"color": "#4CAF50", "linestyle": "-"},
}


def load_checkpoint(path: str, cfg: dict, device: torch.device) -> nn.Module:
    """Load a model checkpoint from the given path and return the model on the specified device."""
    model = build_model(cfg, device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval() # Set to eval mode since we're only doing forward passes for loss evaluation
    return model


def compute_landscape(
    model: nn.Module,
    loss_fn: nn.Module,
    test_loader,
    device: torch.device,
    steps: int = 31,
    range_: float = 2.5,
) -> tuple[list, list, list]:
    """Returns (alphas, betas, losses) as plain Python lists.
    To understand the format and semantics of these outputs, see the docstring of loss_landscape_2d() in src/analysis/landscape.py.
    """
    alphas, betas, losses = loss_landscape_2d(
        model, loss_fn, test_loader, device, steps=steps, range_=range_
    )
    return alphas.tolist(), betas.tolist(), losses.tolist()


def opt_from_key(key: str) -> str:
    """Extract the optimizer name from a config key."""
    return key.split("_rho")[0]


# ── single-checkpoint mode ────────────────────────────────────────────────────

def main_single(
    config_path: str,
    checkpoint: str,
    seed: int,
    out_dir: str,
    steps: int = 31,
    range_: float = 2.5,
) -> None:
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
    print(f"\nComputing landscape: {name}")
    model = load_checkpoint(checkpoint, cfg, device)
    alphas, betas, losses = compute_landscape(
        model=model, 
        loss_fn=loss_fn, 
        test_loader=test_loader, 
        device=device, 
        steps=steps, 
        range_=range_,
    )

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    save_results(os.path.join(out_dir, "landscape.json"), {name: (alphas, betas, losses)})
    
    plot_loss_landscape_2d(
        np.array(alphas), np.array(betas), np.array(losses),
        title=f"2D Loss Landscape — {name}",
        save_path=os.path.join(plots_dir, "landscape.html"),
    )
    print(f"Data saved to {out_dir}/  |  Plots saved to {plots_dir}/")


def main_batch(
    config_path: str,
    ckpt_dir: str,
    out_dir: str,
    seed: int,
    steps: int = 31,
    range_: float = 2.5,
    experiment: str = "baseline",
) -> None:
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
        
        entry = {
            "fname": fname,
            "path": os.path.join(ckpt_dir, fname),
            "opt": m.group("opt"),
            "rho": float(m.group("rho")),
            "seed": int(m.group("seed")),
        }
        
        if is_reparam:
            entry["alpha"] = float(m.group("alpha"))
        entries.append(entry)

    # Accumulate per-seed landscapes grouped by config key (opt/rho[/alpha])
    landscapes_by_key = defaultdict(list)
    alphas_ref, betas_ref = {}, {}

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
        model = load_checkpoint(e["path"], cfg, device)
        alphas, betas, losses = compute_landscape(model, loss_fn, test_loader, device, steps=steps, range_=range_)
        landscapes_by_key[key].append(np.array(losses))
        alphas_ref[key] = alphas
        betas_ref[key]  = betas
        # Move model off GPU before deletion so CUDA memory is freed immediately,
        # regardless of Python GC timing.
        model.cpu()
        del model, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average the values across seeds
    landscape_all: dict[str, tuple] = {
        k: (alphas_ref[k], betas_ref[k], np.mean(landscapes_by_key[k], axis=0).tolist())
        for k in landscapes_by_key
    }

    save_results(os.path.join(out_dir, "landscape_all.json"), landscape_all)

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_landscape_comparison(landscape_all, plots_dir)

    plot_landscape_best(landscape_all, plots_dir)

    print(f"\nData saved to {out_dir}/  |  Plots saved to {plots_dir}/")


def plot_landscape_comparison(landscape: dict[str, tuple], out_dir: str) -> None:
    """4 x 4 surface grid — all checkpoints side by side for visual comparison of landscape shapes across optimizer families and rho values.
    This is useful for visually inspecting how the loss landscape changes with different optimizers and rho values, and for identifying patterns such as sharper minima for certain optimizers or rho values. However, the large number of subplots can make it difficult to discern fine details in each landscape, especially if there are many checkpoints. For a clearer view of the differences between optimizers, see the separate 2 x 2 plot of best-rho landscapes in plot_landscape_best().
    """
    
    Z_CLIP = 5.0
    keys = sorted(landscape.keys())
    n = len(keys)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    titles = [k.replace("_rho", " rho=") for k in keys]

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
        
        Z = np.clip(np.array(losses), 0, Z_CLIP).T
        row, col = idx // ncols + 1, idx % ncols + 1
        # Add contour lines to better visualise the sharpness and flatness of minima in the landscape
        contour_cfg = dict(
            show=True, 
            usecolormap=True, 
            project_z=True,
        )

        fig.add_trace(go.Surface(
            z=Z.tolist(),
            x=alphas if isinstance(alphas, list) else alphas.tolist(),
            y=betas if isinstance(betas, list) else betas.tolist(),
            colorscale="RdBu_r",
            showscale=(idx == 0),
            contours=dict(z=contour_cfg),
        ), row=row, col=col)
        scene_key = "scene" if idx == 0 else f"scene{idx + 1}"

        camera_eye = dict(x=1.5, y=1.5, z=1.0) # Diagonal view to better show landscape features than the default front-on view
        scene_updates[scene_key] = dict(
            xaxis_title="alpha", 
            yaxis_title="beta", 
            zaxis_title="Loss",
            camera=dict(eye=camera_eye),
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


def plot_landscape_best(landscape: dict[str, tuple], out_dir: str) -> None:
    """2 x 2 surface grid — best rho per optimizer (lowest centre loss)."""
    Z_CLIP = 5.0 # Clip loss values above this threshold to avoid extreme outliers dominating the color scale and obscuring landscape features in other regions.
    by_opt: dict[str, list] = defaultdict(list)
    for key, (alphas, betas, losses) in landscape.items():
        opt = opt_from_key(key)
        rho = float(key.split("rho")[1].split("_")[0])
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
        titles.append(f"{opt.upper()} rho={rho}" if opt != "sgd" else "SGD")
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

    camera_eye = dict(x=1.5, y=1.5, z=1.0) # Diagonal view to better show landscape features than the default front-on view
    scene_cfg = dict(
        xaxis_title="alpha", 
        yaxis_title="beta", 
        zaxis_title="Loss",
        camera=dict(eye=camera_eye),
    )
    scene_updates = {}
    for idx in range(len(best_data)):
        key = "scene" if idx == 0 else f"scene{idx + 1}"
        scene_updates[key] = scene_cfg

    fig.update_layout(
        title="2D Loss Landscape — Best rho per Optimizer",
        template="plotly_white",
        height=520 * nrows,
        **scene_updates,
    )
    path = os.path.join(out_dir, "landscape_best.html")
    fig.write_html(path)
    print(f"Saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2D loss landscape for trained checkpoints.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    # Single-checkpoint mode
    parser.add_argument("--checkpoint", default=None, help="Path to a single .pt checkpoint")
    # Batch mode
    parser.add_argument("--ckpt-dir", default=None, help="Directory of .pt checkpoints (batch mode)")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=31,
                        help="Grid resolution along each axis (default: 31)")
    parser.add_argument("--range", dest="range_", type=float, default=2.5,
                        help="Perturbation range in filter-normalised units (default: 2.5)")
    parser.add_argument("--experiment", default=None, choices=["baseline", "reparam"],
                        help="Checkpoint naming scheme: 'baseline' or 'reparam' (auto-detected from --ckpt-dir path if omitted)")
    args = parser.parse_args()

    with open(args.config) as _f:
        _cfg = yaml.safe_load(_f)

    def _default_out(subdir: str) -> str:
        base = _cfg.get(
            "experiments_dir",
            _cfg.get("results_dir", "./results/experiments"),
        )
        return os.path.join(base, subdir, _cfg["model"])

    if args.ckpt_dir:
        _out = args.out_dir or _default_out("landscape")
        _experiment = args.experiment or ("reparam" if "reparam" in args.ckpt_dir else "baseline")
        main_batch(args.config, args.ckpt_dir, _out, args.seed, args.steps, args.range_, experiment=_experiment)
    elif args.checkpoint:
        _out = args.out_dir or _default_out("landscape")
        main_single(args.config, args.checkpoint, args.seed, _out, args.steps, args.range_)
    else:
        parser.error("Provide either --checkpoint or --ckpt-dir.")
