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
from src.analysis.reparam import apply_reparam
from experiments.utils import get_device, set_seed, build_model, build_optimizer, save_results


def _apply_reparam(model: nn.Module, model_name: str, alpha: float) -> None:
    bound = apply_reparam(model, model_name, alpha)
    if bound is not None:
        # ViT: log the analytic Taylor deviation bound alongside results
        print(f"  [ViT reparam] alpha={alpha:.2f}  taylor_bound(x=1)={bound:.6f}")


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

    final = history[-1].copy()
    final["seed"] = seed
    final["optimizer"] = opt_type
    final["alpha"] = alpha
    final["rho"] = rho
    return final


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    experiments_dir = cfg["experiments_dir"]
    model_name = cfg["model"]
    seeds = cfg["seeds"]
    alpha_values = cfg["alpha_values"]
    opt_cfgs = cfg["optimizers"]

    all_results = []
    out_path = os.path.join(experiments_dir, "reparam", model_name, "reparam_results.json")

    # Load already-completed results to support resuming after restart
    _done: set[tuple] = set()
    if os.path.exists(out_path):
        import json
        with open(out_path) as _f:
            all_results = json.load(_f)
        for _r in all_results:
            for _ps in _r.get("per_seed", []):
                _done.add((_r["optimizer"], _r["rho"], _r["alpha"], _ps["seed"]))
        print(f"Resuming: {len(_done)} (opt, rho, alpha, seed) combos already done.")

    for opt_name, opt_cfg in opt_cfgs.items():
        opt_type = opt_cfg["type"]
        rho_sweep = opt_cfg.get("rho_sweep", [0.0])
        if opt_type == "asam":
            cfg["asam_eta"] = opt_cfg.get("eta", 0.01)

        for rho in rho_sweep:
            alpha_accs: dict[float, list[float]] = {}

            for alpha in alpha_values:
                per_seed = []
                for seed in seeds:
                    if (opt_name, rho, alpha, seed) in _done:
                        print(f"\n[{model_name}] opt={opt_name} rho={rho} alpha={alpha} seed={seed} — skipping (done)")
                        continue
                    print(f"\n[{model_name}] opt={opt_name} rho={rho} alpha={alpha} seed={seed}")
                    result = run_single(cfg, opt_type, rho, alpha, seed)
                    per_seed.append(result)
                accs = [r["test_acc"] for r in per_seed]
                alpha_accs[alpha] = accs
                all_results.append({
                    "model": model_name,
                    "optimizer": opt_name,
                    "rho": rho,
                    "alpha": alpha,
                    "test_acc_mean": float(np.mean(accs)),
                    "test_acc_sem": float(np.std(accs, ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else 0.0,
                    "per_seed": per_seed,
                })

            # Variance of final test acc across alpha values (mean over seeds, then var over alphas)
            mean_accs = [float(np.mean(alpha_accs[a])) for a in alpha_values]
            var_across_alpha = float(np.var(mean_accs))
            print(f"\n[{model_name}] opt={opt_name} rho={rho}  reparam_variance={var_across_alpha:.6f}")

    out_path = os.path.join(experiments_dir, "reparam", model_name, "reparam_results.json")
    save_results(out_path, all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
