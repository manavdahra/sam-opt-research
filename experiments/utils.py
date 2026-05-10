"""Shared utilities for experiment runner scripts."""
from __future__ import annotations

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
    # elif model_name == "vit_b_32":
    #     from src.models.vit import get_vit_b_32
    #     model = get_vit_b_32(num_classes=10, pretrained=cfg.get("pretrained", True))
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

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler


def save_results(path: str, data: dict | list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved → {path}")
