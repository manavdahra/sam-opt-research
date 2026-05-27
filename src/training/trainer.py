from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _bn_set_mode(model: nn.Module, training: bool) -> None:
    """Set all BatchNorm layers to train/eval without touching other layers."""
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            m.train(training)


def _is_sam_family(optimizer: Optimizer) -> bool:
    """Duck-type check: optimizer exposes first_step / second_step."""
    return hasattr(optimizer, "first_step") and hasattr(optimizer, "second_step")


def train_one_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    scheduler: LRScheduler | None = None,
) -> dict[str, float]:
    """Run one training epoch.

    For SAM-family optimizers the loss is evaluated twice per batch:
      1. forward + backward → first_step (perturb)
      2. forward + backward → second_step (restore + update)

    Returns:
        dict with keys "train_loss", "train_acc", and "elapsed_sec".
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    t0 = time.perf_counter()
    for inputs, targets in tqdm(loader, leave=False, desc="train"):
        inputs, targets = inputs.to(device), targets.to(device)

        if _is_sam_family(optimizer):
            # --- First forward/backward: compute perturbation ---
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # --- Second forward/backward: update at perturbed point ---
            # Freeze BN running stats so the perturbed activations (at θ+ε)
            # don't corrupt the running mean/variance used at inference time.
            # BN still normalises using the current batch statistics — only
            # the EMA update of running_mean/running_var is suppressed.
            _bn_set_mode(model, training=False)
            
            # Use separate variables so 'outputs' and 'loss' below
            # reflect the original-weights values (for accurate metrics).
            outputs_perturbed = model(inputs)
            loss_perturbed = loss_fn(outputs_perturbed, targets)
            loss_perturbed.backward()
            optimizer.second_step(zero_grad=True)
            _bn_set_mode(model, training=True)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total_samples += targets.size(0)

    if scheduler is not None:
        scheduler.step()

    return {
        "train_loss": total_loss / total_samples,
        "train_acc": total_correct / total_samples,
        "elapsed_sec": time.perf_counter() - t0,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on the given loader.

    Returns:
        dict with keys "test_loss" and "test_acc".
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        total_loss += loss.item() * targets.size(0)
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total_samples += targets.size(0)

    return {
        "test_loss": total_loss / total_samples,
        "test_acc": total_correct / total_samples,
    }


def train(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    scheduler: LRScheduler | None = None,
    verbose: bool = True,
    compile_model: bool = True,
) -> list[dict[str, float]]:
    """Full training loop for `epochs` epochs.

    Args:
        compile_model: If True, apply torch.compile() before training.
            Adds ~30s warmup on first batch but speeds up subsequent epochs.

    Returns:
        List of per-epoch metric dicts, each containing:
        train_loss, train_acc, test_loss, test_acc, epoch.
    """
    if compile_model:
        try:
            # torch.compile with the inductor backend generates invalid Metal
            # shader code on MPS (known upstream bug). Skip on MPS.
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                torch.set_float32_matmul_precision("high")  # enable TF32 on Ampere+
                model = torch.compile(model)
        except Exception:
            pass  # torch.compile not supported (e.g. Windows), silently skip

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, optimizer, train_loader, loss_fn, device, scheduler
        )
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        row = {"epoch": epoch, **train_metrics, **test_metrics}
        history.append(row)
        if verbose:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={row['train_loss']:.4f} "
                f"train_acc={row['train_acc']:.4f} | "
                f"test_loss={row['test_loss']:.4f} "
                f"test_acc={row['test_acc']:.4f}"
            )
    return history
