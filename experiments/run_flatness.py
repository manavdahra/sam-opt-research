"""Experiment 3 — Flatness / sharpness analysis.

Loads a trained model checkpoint (or trains a fresh one if no checkpoint is
provided) and computes:
  1. Hutchinson trace estimate (tr(H)/d) — sharpness proxy.
  2. 1D loss landscape along a random filter-normalized direction.

Usage:
    python experiments/run_flatness.py --config configs/resnet18_baseline.yaml \
        --checkpoint results/baseline/resnet18/sam_rho0.05_seed0.pt
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
from src.analysis.hutchinson import hutchinson_trace
from src.analysis.landscape import loss_landscape_1d, plot_loss_landscape
from experiments.utils import get_device, set_seed, build_model, save_results


def main(config_path: str, checkpoint: str | None, seed: int) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    set_seed(seed)

    resize = cfg.get("resize")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=cfg["data_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        resize=resize,
    )
    loss_fn = nn.CrossEntropyLoss()

    results_dir = cfg.get("results_dir", "./results/flatness").replace(
        "baseline", "flatness"
    )
    model_name = cfg["model"]

    if checkpoint:
        model = build_model(cfg, device)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        checkpoints = {os.path.basename(checkpoint): model}
    else:
        # Quick single-epoch smoke-test mode: just build a random model
        print("No checkpoint provided — using randomly initialised model for demonstration.")
        model = build_model(cfg, device)
        checkpoints = {"random_init": model}

    sharpness_results = {}
    landscape_histories = {}

    for name, mdl in checkpoints.items():
        print(f"\nAnalysing: {name}")
        mdl.eval()

        # Hutchinson sharpness
        trace = hutchinson_trace(mdl, loss_fn, test_loader, device, n_samples=100)
        sharpness_results[name] = trace
        print(f"  tr(H)/d = {trace:.6f}")

        # 1D landscape
        alphas, losses = loss_landscape_1d(
            mdl, loss_fn, test_loader, device, steps=51, range_=1.0
        )
        landscape_histories[name] = (alphas.tolist(), losses.tolist())

    # Save scalar sharpness
    out_json = os.path.join(results_dir, model_name, "sharpness.json")
    save_results(out_json, sharpness_results)

    # Save landscape data
    out_landscape = os.path.join(results_dir, model_name, "landscape.json")
    save_results(out_landscape, landscape_histories)

    # Save landscape plot
    import numpy as np
    fig = plot_loss_landscape(
        {k: (np.array(v[0]), np.array(v[1])) for k, v in landscape_histories.items()},
        save_path=os.path.join(results_dir, model_name, "landscape.png"),
    )
    print(f"Landscape plot saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Path to model .pt checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.seed)
