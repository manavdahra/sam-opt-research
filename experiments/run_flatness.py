import argparse
import gc
import os
import re
import sys
import yaml
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn

from src.data.cifar10 import get_cifar10_loaders
from src.analysis.hutchinson import hutchinson_trace
from experiments.utils import get_device, set_seed, build_model, save_results
from experiments.plot_flatness import plot_all as plot_sharpness_all

# Baseline filename pattern: <opt>_rho<rho>_seed<seed>.pt to look for in results directory 
_CKPT_RE = re.compile(r"^(?P<opt>[a-z]+)_rho(?P<rho>[0-9.]+)_seed(?P<seed>\d+)\.pt$")
# Reparam filename pattern: <opt>_rho<rho>_alpha<alpha>_seed<seed>.pt to look for in results directory
_REPARAM_CKPT_RE = re.compile(r"^(?P<opt>[a-z]+)_rho(?P<rho>[0-9.]+)_alpha(?P<alpha>[0-9.]+)_seed(?P<seed>\d+)\.pt$")

def load_checkpoint(path: str, cfg: dict, device: torch.device) -> nn.Module:
    """Load a model checkpoint from the given path and return the model in eval mode."""
    model = build_model(cfg, device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval() # Set to eval mode since we're only doing inference for sharpness estimation
    return model


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
    ckpt_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    is_reparam = experiment == "reparam"
    
    entries = []
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

    # Accumulate per-seed results grouped by config key (opt/rho[/alpha])
    # so we can average across seeds rather than silently overwriting them.
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
        model = load_checkpoint(e["path"], cfg, device)
        trace = compute_sharpness(model, loss_fn, test_loader, device, n_samples=n_samples, max_batch=max_batch)
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
