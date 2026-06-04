"""Shared utilities for experiment runner scripts."""

import datetime
import gc
import os
import re
import sys
import json
import random
import yaml

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
from src.models.resnet18 import get_resnet18
from src.models.vit import get_vit_b_32
from src.data.cifar10 import get_cifar10_loaders


# Canonical Plotly colour/symbol/label mapping for all optimizers
OPT_STYLE: dict[str, dict] = {
    "sgd":  {"color": "#7B8794", "symbol": "diamond",     "label": "SGD"},
    "sam":  {"color": "#4878CF", "symbol": "circle",      "label": "SAM"},
    "asam": {"color": "#C8703A", "symbol": "triangle-up", "label": "ASAM"},
    "msam": {"color": "#3D8C6E", "symbol": "square",      "label": "M-SAM"},
}

# Checkpoint filename patterns used by run_baseline / run_reparam
_CKPT_RE = re.compile(r"^(?P<opt>[a-z]+)_rho(?P<rho>[0-9.]+)_seed(?P<seed>\d+)\.pt$")
_REPARAM_CKPT_RE = re.compile(
    r"^(?P<opt>[a-z]+)_rho(?P<rho>[0-9.]+)_alpha(?P<alpha>[0-9.]+)_seed(?P<seed>\d+)\.pt$"
)


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
        model = get_resnet18(num_classes=10)
    elif model_name == "vit_b_32":
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
    print(f"Saved at {path}")


def load_results(path: str):
    """Load and return the contents of a JSON results file."""
    with open(path) as f:
        return json.load(f)


def load_checkpoint(path: str, cfg: dict, device: torch.device) -> nn.Module:
    """Load a model checkpoint and return the model in eval mode."""
    model = build_model(cfg, device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def build_data_loaders(cfg: dict):
    """Build CIFAR-10 train and test data loaders from a config dict."""
    return get_cifar10_loaders(
        data_dir=cfg["data_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 4),
        resize=cfg.get("resize"),
        max_samples=cfg.get("max_samples"),
    )


def save_figure(fig, out_dir: str, stem: str, width: int = 800, height: int = 500) -> None:
    """Save a Plotly figure as both HTML and PNG under out_dir."""
    path = os.path.join(out_dir, f"{stem}.html")
    fig.write_html(path)
    print(f"Saved at {path}")
    png_path = os.path.join(out_dir, f"{stem}.png")
    fig.write_image(png_path, width=width, height=height, scale=2)
    print(f"Saved at {png_path}")


def discover_checkpoints(ckpt_dir: str, is_reparam: bool = False) -> list[dict]:
    """Return parsed checkpoint entries from ckpt_dir.

    Each entry dict contains: fname, path, opt, rho, seed.
    Reparam entries additionally contain: alpha.
    Files not matching the expected naming pattern are skipped.
    """
    ckpt_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    pattern = _REPARAM_CKPT_RE if is_reparam else _CKPT_RE
    entries: list[dict] = []
    for fname in ckpt_files:
        m = pattern.match(fname)
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
    return entries


def resolve_out_dir(results_path: str, out_dir: str | None) -> str:
    """Return out_dir, defaulting to a 'plots/' subdirectory beside results_path."""
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(results_path) or ".", "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def free_gpu_resources() -> None:
    """Run garbage collection and clear the CUDA memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(dict(config), f)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model_state, os.path.join(run_dir, "checkpoint.pt"))

    return run_dir


def update_index(results_root: str, run_id: str, meta: dict) -> None:
    """Append a run entry to results/index.json.

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
