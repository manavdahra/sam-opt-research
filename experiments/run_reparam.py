"""Experiment 2 — Reparametrisation invariance.

Applies a function-preserving scale transform (alpha) to the model's initial
weights before training, then measures variance of final test accuracy across
alpha values. A reparametrisation-invariant optimizer should show low variance.

Usage:
    python experiments/run_reparam.py --config configs/resnet18_reparam.yaml
    python experiments/run_reparam.py --config configs/vit_reparam.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import yaml
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch.nn as nn

from src.data.cifar10 import get_cifar10_loaders
from src.training.trainer import train
from src.analysis.metrics import aggregate_seeds
from experiments.utils import get_device, set_seed, build_model, build_optimizer, save_results


def _apply_reparam(model: nn.Module, model_name: str, alpha: float) -> None:
    if model_name == "resnet18":
        from src.models.resnet18 import apply_relu_reparam
        apply_relu_reparam(model, alpha)
    elif model_name == "vit_b_32":
        from src.models.vit import apply_mlp_reparam, measure_reparam_deviation
        import torch
        # Log the pre-training deviation for ViT (GELU is approximate)
        sample = torch.randn(2, 3, 224, 224, device=next(model.parameters()).device)
        dev = measure_reparam_deviation(model, sample, alpha)
        print(f"  [ViT reparam] alpha={alpha:.2f}  output_deviation={dev:.6f}")
        apply_mlp_reparam(model, alpha)


def run_single(cfg: dict, opt_type: str, rho: float, alpha: float, seed: int) -> dict:
    device = get_device()
    set_seed(seed)

    resize = cfg.get("resize")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=cfg["data_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        resize=resize,
    )

    model = build_model(cfg, device)
    _apply_reparam(model, cfg["model"], alpha)

    optimizer, scheduler = build_optimizer(opt_type, model.parameters(), cfg, rho)
    loss_fn = nn.CrossEntropyLoss()

    history = train(
        model,
        optimizer,
        train_loader,
        test_loader,
        loss_fn,
        device,
        epochs=cfg["epochs"],
        scheduler=scheduler,
        verbose=False,
    )

    final = history[-1]
    final["seed"] = seed
    final["optimizer"] = opt_type
    final["alpha"] = alpha
    final["rho"] = rho
    return final


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["results_dir"]
    model_name = cfg["model"]
    seeds = cfg["seeds"]
    alpha_values = cfg["alpha_values"]
    opt_names = cfg["optimizers"]
    rho = cfg["rho"]
    asam_rho = cfg.get("asam_rho", 0.5)

    all_results = []

    for opt_name in opt_names:
        opt_rho = asam_rho if opt_name == "asam" else rho
        # Add asam_eta to cfg for build_optimizer
        if opt_name == "asam":
            cfg["asam_eta"] = cfg.get("asam_eta", 0.01)

        alpha_accs: dict[float, list[float]] = {}

        for alpha in alpha_values:
            per_seed = []
            for seed in seeds:
                print(f"\n[{model_name}] opt={opt_name} alpha={alpha} seed={seed}")
                result = run_single(cfg, opt_name, opt_rho, alpha, seed)
                per_seed.append(result)
            accs = [r["test_acc"] for r in per_seed]
            alpha_accs[alpha] = accs
            all_results.append({
                "model": model_name,
                "optimizer": opt_name,
                "alpha": alpha,
                "test_acc_mean": float(np.mean(accs)),
                "test_acc_sem": float(np.std(accs, ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else 0.0,
                "per_seed": per_seed,
            })

        # Variance of final test acc across alpha values (mean over seeds, then var over alphas)
        mean_accs = [float(np.mean(alpha_accs[a])) for a in alpha_values]
        var_across_alpha = float(np.var(mean_accs))
        print(f"\n[{model_name}] opt={opt_name}  reparam_variance={var_across_alpha:.6f}")

    out_path = os.path.join(results_dir, model_name, "reparam_results.json")
    save_results(out_path, all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
