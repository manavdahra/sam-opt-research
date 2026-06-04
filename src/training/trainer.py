import contextlib
import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _bn_disable_running_stats(model: nn.Module) -> None:
    """Suppress EMA updates to BN running_mean/running_var during the SAM second
    forward pass by setting momentum=0.  The model stays in train mode so batch
    statistics are still used for normalisation — only the running-stat write-back
    is skipped.  Call _bn_restore_running_stats() afterwards.
    """
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            m._bak_momentum = m.momentum
            m.momentum = 0.0


def _bn_restore_running_stats(model: nn.Module) -> None:
    """Restore the BN momentum values saved by _bn_disable_running_stats()."""
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            if hasattr(m, '_bak_momentum'):
                m.momentum = m._bak_momentum
                del m._bak_momentum


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
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Run one training epoch.

    For SAM-family optimizers the loss is evaluated twice per batch:
      1. forward + backward at first_step (perturb)
      2. forward + backward at second_step (restore + update)

    Args:
        amp_dtype: If set (e.g. torch.bfloat16), wraps forward passes with
            torch.amp.autocast for mixed-precision training. BF16 is faster.

    Returns:
        dict with keys "train_loss", "train_acc", and "elapsed_sec".
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else contextlib.nullcontext()
    )

    t0 = time.perf_counter()
    for inputs, targets in tqdm(loader, leave=False, desc="train"):
        inputs, targets = inputs.to(device), targets.to(device)

        if _is_sam_family(optimizer):
            # --- First forward/backward: compute perturbation ---
            with autocast_ctx:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # --- Second forward/backward: update at perturbed point ---
            # Keep model in train mode so BN normalises with batch statistics
            # (essential for numerical stability).  Set momentum=0 so the
            # second pass does not corrupt running_mean/running_var with
            # activations computed at the perturbed weights (θ+ε).
            _bn_disable_running_stats(model)
            with autocast_ctx:
                outputs_perturbed = model(inputs)
                loss_perturbed = loss_fn(outputs_perturbed, targets)
            loss_perturbed.backward()
            optimizer.second_step(zero_grad=True)
            _bn_restore_running_stats(model)
        else:
            optimizer.zero_grad()
            with autocast_ctx:
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
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Evaluate model on the given loader.

    Returns:
        dict with keys "test_loss" and "test_acc".
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else contextlib.nullcontext()
    )

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with autocast_ctx:
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
    # Auto-enable BF16 AMP on CUDA devices that support it (compute capability >= 8.0).
    # BF16 has the same dynamic range as FP32 so no GradScaler is needed.
    amp_dtype: torch.dtype | None = None
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16

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
            model, optimizer, train_loader, loss_fn, device, scheduler, amp_dtype
        )
        test_metrics = evaluate(model, test_loader, loss_fn, device, amp_dtype)
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
