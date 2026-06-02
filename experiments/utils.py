"""Shared utilities for experiment runner scripts."""
from __future__ import annotations

import datetime
import os
import sys
import json
import random

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Make sure the project root is importable when scripts are run directly
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.optimizers.sam import SAM
from src.optimizers.asam import ASAM
from src.optimizers.msam import MSAM


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    model_name = cfg["model"]
    if model_name == "resnet18":
        from src.models.resnet18 import get_resnet18
        model = get_resnet18(num_classes=10)
    # ViT experiments are disabled pending resolution of GELU reparam approximation issues.
    elif model_name == "vit_b_32":
        from src.models.vit import get_vit_b_32
        model = get_vit_b_32(num_classes=10, pretrained=cfg.get("pretrained", True))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def build_optimizer(
    opt_type: str,
    params,
    cfg: dict,
    rho: float,
) -> tuple:
    """Build (optimizer, scheduler).

    Returns the optimizer and a CosineAnnealingLR scheduler.
    """
    base_kwargs = dict(
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    epochs = cfg["epochs"]

    if opt_type == "sgd":
        optimizer = torch.optim.SGD(params, **base_kwargs)
    elif opt_type == "sam":
        optimizer = SAM(params, torch.optim.SGD, rho=rho, **base_kwargs)
    elif opt_type == "asam":
        optimizer = ASAM(
            params,
            torch.optim.SGD,
            rho=rho,
            eta=cfg.get("asam_eta", 0.01),
            **base_kwargs,
        )
    elif opt_type == "msam":
        optimizer = MSAM(params, torch.optim.SGD, rho=rho, **base_kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    # Attach the scheduler to the underlying SGD for SAM-family optimizers so
    # that _step_count is incremented by base_optimizer.step() and PyTorch
    # does not emit a spurious "scheduler called before optimizer" warning.
    # SAM shares param_groups with base_optimizer, so LR changes are identical.
    sched_target = getattr(optimizer, "base_optimizer", optimizer)
    scheduler = CosineAnnealingLR(sched_target, T_max=epochs)
    return optimizer, scheduler


def save_results(path: str, data: dict | list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved → {path}")


# ── Run-directory helpers ────────────────────────────────────────────────────

def make_run_id(model: str, opt_type: str, rho: float, seed: int, suffix: str = "") -> str:
    """Return a unique, descriptive run identifier.

    Format: YYYYMMDD-HHMMSS-<model>-<opt>-rho<rho>-seed<seed>[-<suffix>]
    Example: 20260527-143012-resnet18-sam-rho0.05-seed42
    """
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"{ts}-{model}-{opt_type}-rho{rho}-seed{seed}"
    return f"{base}-{suffix}" if suffix else base


def write_run_dir(
    runs_dir: str,
    run_id: str,
    config: dict,
    history: list[dict],
    model_state: dict,
) -> str:
    """Write per-run artefacts into runs/<run-id>/.

    Creates:
      - config.yaml   — exact config used for this run
      - metrics.json  — per-epoch train/val metrics + elapsed_sec
      - checkpoint.pt — final model weights

    Returns the absolute path of the run directory.
    """
    import yaml as _yaml

    run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        _yaml.dump(dict(config), f)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model_state, os.path.join(run_dir, "checkpoint.pt"))

    return run_dir


def update_index(results_root: str, run_id: str, meta: dict) -> None:
    """Append a run entry to results/index.json (atomic read-modify-write).

    The index is the single source of truth for querying runs, e.g.:
        "all SAM runs on resnet18 with rho=0.05"
    """
    index_path = os.path.join(results_root, "index.json")
    index: dict = {"runs": {}}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
    index["runs"][run_id] = meta
    tmp = index_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, index_path)
