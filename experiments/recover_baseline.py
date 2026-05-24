"""Recover baseline_results.json from saved checkpoints.

Because `run_baseline.py` writes the results JSON only once at the end, an
interrupted run (e.g. Vast.ai preemption) truncates the file even after all
checkpoints have been saved successfully.

This script:
1. Scans the checkpoint directory for all `<opt>_rho<rho>_seed<seed>.pt` files.
2. Loads each checkpoint and evaluates it on the CIFAR-10 test set to recover
   test_loss and test_acc.
3. Also evaluates on the full training set to recover train_loss and train_acc
   (and therefore the divergence rate).
4. Reconstructs a valid baseline_results.json with the final-epoch metrics.

NOTE: per-epoch training history cannot be recovered — only final metrics are
written. The `history` field is omitted for recovered runs.

Usage:
    python experiments/recover_baseline.py --config configs/resnet18_baseline.yaml
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import yaml
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.cifar10 import get_cifar10_loaders
from src.training.trainer import evaluate
from src.analysis.metrics import divergence_rate, aggregate_seeds
from experiments.utils import get_device, build_model, save_results


_CKPT_PATTERN = re.compile(r"^(?P<opt>[a-z]+)_rho(?P<rho>[0-9.]+)_seed(?P<seed>\d+)\.pt$")


def eval_checkpoint(ckpt_path: str, cfg: dict, device: torch.device) -> dict:
    model = build_model(cfg, device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    resize = cfg.get("resize")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=cfg["data_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        resize=resize,
    )
    loss_fn = nn.CrossEntropyLoss()

    print(f"  Evaluating test set ...", end=" ", flush=True)
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    print(f"test_acc={test_metrics['test_acc']:.4f}")

    print(f"  Evaluating train set ...", end=" ", flush=True)
    _train_raw = evaluate(model, train_loader, loss_fn, device)
    train_metrics = {"train_loss": _train_raw["test_loss"], "train_acc": _train_raw["test_acc"]}
    print(f"train_acc={train_metrics['train_acc']:.4f}")

    return {**test_metrics, **train_metrics}


def main(config_path: str, ckpt_dir_override: str | None = None, out_override: str | None = None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    ckpt_dir = ckpt_dir_override or os.path.join(cfg["results_dir"], cfg["model"], "checkpoints")
    out_path = out_override or os.path.join(cfg["results_dir"], cfg["model"], "baseline_results.json")

    if not os.path.isdir(ckpt_dir):
        print(f"Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)

    # Group checkpoints by (opt, rho)
    groups: dict[tuple[str, float], list[dict]] = {}
    for fname in sorted(os.listdir(ckpt_dir)):
        m = _CKPT_PATTERN.match(fname)
        if not m:
            continue
        opt = m.group("opt")
        rho = float(m.group("rho"))
        seed = int(m.group("seed"))
        key = (opt, rho)
        groups.setdefault(key, []).append({"seed": seed, "path": os.path.join(ckpt_dir, fname)})

    if not groups:
        print("No checkpoints found matching pattern <opt>_rho<rho>_seed<seed>.pt")
        sys.exit(1)

    all_results = []

    for (opt, rho), entries in sorted(groups.items()):
        print(f"\n[{cfg['model']}] opt={opt} rho={rho}")
        per_seed = []
        for entry in sorted(entries, key=lambda e: e["seed"]):
            seed = entry["seed"]
            ckpt_path = entry["path"]
            print(f"  seed={seed}  checkpoint={ckpt_path}")
            metrics = eval_checkpoint(ckpt_path, cfg, device)
            dr = divergence_rate(metrics["train_loss"], metrics["test_loss"])
            per_seed.append({
                "epoch": cfg["epochs"],
                "train_loss": metrics["train_loss"],
                "train_acc": metrics["train_acc"],
                "test_loss": metrics["test_loss"],
                "test_acc": metrics["test_acc"],
                "divergence_rate": dr,
                "seed": seed,
                "optimizer": opt,
                "rho": rho,
                "checkpoint": ckpt_path,
                # history is not recoverable from checkpoints alone
            })

        _non_numeric = {"seed", "optimizer", "checkpoint"}
        agg = aggregate_seeds(
            [{k: v for k, v in r.items() if k not in _non_numeric} for r in per_seed]
        )
        agg["optimizer"] = opt
        agg["rho"] = rho
        agg["model"] = cfg["model"]

        all_results.append({"summary": agg, "per_seed": per_seed})
        # Write incrementally after each (opt, rho) group so partial results
        # are not lost if the script is interrupted.
        save_results(out_path, all_results)
        print(f"  -> Saved {len(all_results)} result(s) to {out_path}")

    print(f"\nRecovery complete. {len(all_results)} optimizer/rho combinations written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recover baseline_results.json from checkpoints")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/resnet18_baseline.yaml)")
    parser.add_argument("--ckpt-dir", default=None, help="Override checkpoint directory path")
    parser.add_argument("--out", default=None, help="Override output JSON path")
    args = parser.parse_args()
    main(args.config, ckpt_dir_override=args.ckpt_dir, out_override=args.out)
