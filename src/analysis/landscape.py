from __future__ import annotations

import copy

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def _filter_normalized_direction(model: nn.Module) -> list[torch.Tensor]:
    """Generate a random direction in parameter space with filter normalization.

    Each filter (2D+ weight tensor, or 1D bias) is normalized to have the same
    Frobenius norm as the corresponding weight.
    """
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        if p.dim() >= 2:
            # Per-filter normalization: normalize each filter slice
            for i in range(p.size(0)):
                norm_d = d[i].norm()
                norm_p = p[i].norm()
                if norm_d > 1e-10:
                    d[i] = d[i] / norm_d * norm_p
        else:
            d = d / (d.norm() + 1e-10) * p.norm()
        direction.append(d)
    return direction


@torch.no_grad()
def _perturb_model(model: nn.Module, direction: list[torch.Tensor], step: float) -> None:
    for p, d in zip(model.parameters(), direction):
        p.add_(d * step)


def loss_landscape_1d(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int = 51,
    range_: float = 1.0,
    direction: list[torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the loss along a random 1D direction through the current θ*.

    Args:
        model: Trained model.
        loss_fn: Loss function.
        loader: DataLoader (a single batch is used for speed).
        device: Computation device.
        steps: Number of interpolation steps (odd number recommended).
        range_: Perturbation range in each direction.
        direction: Pre-computed direction (optional). If None, a new
            filter-normalized random direction is sampled.

    Returns:
        (alphas, losses): 1D arrays of shape (steps,).
    """
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)

    if direction is None:
        direction = _filter_normalized_direction(model)

    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()
    dir_device = [d.to(device) for d in direction]

    alphas = np.linspace(-range_, range_, steps)
    losses = []
    for alpha in alphas:
        _perturb_model(model_copy, dir_device, float(alpha))
        with torch.no_grad():
            outputs = model_copy(inputs)
            loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        _perturb_model(model_copy, dir_device, -float(alpha))  # restore

    return alphas, np.array(losses)


def plot_loss_landscape(
    histories: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot 1D loss landscapes for multiple optimizers on a single figure.

    Args:
        histories: {optimizer_name: (alphas, losses)} from loss_landscape_1d().
        save_path: If given, save the figure to this path.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (alphas, losses) in histories.items():
        ax.plot(alphas, losses, label=name)
    ax.set_xlabel("Perturbation α")
    ax.set_ylabel("Loss")
    ax.set_title("1D Loss Landscape")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
