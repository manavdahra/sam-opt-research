"""Experiment 1 — Baseline accuracy/loss comparison.

Usage:
    python experiments/run_baseline.py --config configs/resnet18_baseline.yaml
    python experiments/run_baseline.py --config configs/vit_baseline.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from src.data.cifar10 import get_cifar10_loaders
from src.training.trainer import train
from src.analysis.metrics import aggregate_seeds, divergence_rate
from experiments.utils import get_device, set_seed, build_model, build_optimizer, save_results


def run_single(cfg: dict, opt_type: str, rho: float, seed: int) -> dict:
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
        verbose=True,
    )

    ckpt_dir = os.path.join(cfg["results_dir"], cfg["model"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{opt_type}_rho{rho}_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

    final = history[-1]
    final["divergence_rate"] = divergence_rate(final["train_loss"], final["test_loss"])
    final["seed"] = seed
    final["optimizer"] = opt_type
    final["rho"] = rho
    final["checkpoint"] = ckpt_path
    final["history"] = history
    return final


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["results_dir"]
    model_name = cfg["model"]
    seeds = cfg["seeds"]
    opt_cfgs = cfg["optimizers"]

    all_results = []

    for opt_name, opt_cfg in opt_cfgs.items():
        opt_type = opt_cfg["type"]
        rho_sweep = opt_cfg.get("rho_sweep", [0.0])

        for rho in rho_sweep:
            per_seed = []
            for seed in seeds:
                print(f"\n[{model_name}] opt={opt_name} rho={rho} seed={seed}")
                result = run_single(cfg, opt_type, rho, seed)
                per_seed.append(result)

            agg = aggregate_seeds(
                [{k: v for k, v in r.items() if k not in ("history", "seed")} for r in per_seed]
            )
            agg["optimizer"] = opt_name
            agg["rho"] = rho
            agg["model"] = model_name
            all_results.append({"summary": agg, "per_seed": per_seed})

    out_path = os.path.join(results_dir, model_name, "baseline_results.json")
    save_results(out_path, all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
