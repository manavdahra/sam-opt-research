import argparse
import os
import sys
import yaml
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn

from src.analysis.hutchinson import hutchinson_trace
from experiments.utils import (
    get_device, set_seed, build_model, save_results,
    load_checkpoint, build_data_loaders, discover_checkpoints, free_gpu_resources,
)
from experiments.plot_flatness import plot_all as plot_sharpness_all


def compute_sharpness(
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


def main(config_path: str, checkpoint: str | None, seed: int, experiment: str = "baseline") -> None:
    """Entry point for main.py CLI dispatch.

    If checkpoint is provided, runs single-checkpoint mode.
    Otherwise runs batch mode over the checkpoints saved by run_baseline.py
    or run_reparam.py depending on *experiment* ("baseline" or "reparam").
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    out_dir = os.path.join(
        cfg.get("experiments_dir", "./results/experiments"),
        "flatness",
        experiment,
        cfg["model"],
    )
    if checkpoint is not None:
        main_single(config_path, checkpoint, seed, out_dir)
    else:
        ckpt_dir = os.path.join(
            cfg.get("experiments_dir", "./results/experiments"),
            experiment, cfg["model"], "checkpoints",
        )
        main_batch(config_path, ckpt_dir, out_dir, seed, experiment=experiment)


def main_single(
        config_path: str, 
        checkpoint: str, 
        seed: int, 
        out_dir: str, 
        n_samples: int = 20, 
        max_batch: int = 64,
    ) -> None:
    """Compute and save the sharpness estimate for a single checkpoint when specified."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    set_seed(seed)
    _, test_loader = build_data_loaders(cfg)
    loss_fn = nn.CrossEntropyLoss()

    name = os.path.basename(checkpoint)
    print(f"\nAnalysing: {name}")
    model = load_checkpoint(checkpoint, cfg, device)
    trace = compute_sharpness(model, loss_fn, test_loader, device, n_samples=n_samples, max_batch=max_batch)

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    save_results(os.path.join(out_dir, "sharpness.json"), {name: trace})
    print(f"Data saved to {out_dir}/")


def main_batch(
        config_path: str, 
        ckpt_dir: str, 
        out_dir: str, 
        seed: int, 
        n_samples: int = 20, 
        max_batch: int = 64, 
        experiment: str = "baseline"
    ) -> None:
    """Compute and save sharpness estimates for all checkpoints in the specified directory, then produce summary plots."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    os.makedirs(out_dir, exist_ok=True)

    # Discover and sort checkpoints
    is_reparam = experiment == "reparam"
    entries = discover_checkpoints(ckpt_dir, is_reparam=is_reparam)

    # Accumulate per-seed results grouped by config key (opt/rho[/alpha])
    # so we can average across seeds rather than silently overwriting them.
    traces_by_key: dict[str, list[float]] = defaultdict(list)

    for i, e in enumerate(entries, 1):
        key = f"{e['opt']}_rho{e['rho']}"
        if "alpha" in e:
            key += f"_alpha{e['alpha']}"
        set_seed(e["seed"])
        _, test_loader = build_data_loaders(cfg)
        print(f"\n[{i}/{len(entries)}] {key} (seed={e['seed']})")
        model = load_checkpoint(e["path"], cfg, device)
        trace = compute_sharpness(model, loss_fn, test_loader, device, n_samples=n_samples, max_batch=max_batch)
        traces_by_key[key].append(trace)
        del model, test_loader
        free_gpu_resources()

        # Incremental save of per-seed-averaged sharpness seen so far
        sharpness_partial = {k: float(np.mean(v)) for k, v in traces_by_key.items()}
        save_results(os.path.join(out_dir, "sharpness_all.json"), sharpness_partial)

    # Average across seeds
    sharpness_all: dict[str, float] = {k: float(np.mean(v)) for k, v in traces_by_key.items()}

    save_results(os.path.join(out_dir, "sharpness_all.json"), sharpness_all)

    plots_dir = os.path.join(out_dir, "plots")
    plot_sharpness_all(sharpness_all, plots_dir)

    print(f"\nData saved to {out_dir}/  |  Plots saved to {plots_dir}/")


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
