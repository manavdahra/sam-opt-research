from __future__ import annotations

import copy

import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from torch.utils.data import DataLoader


def _filter_normalized_direction(model: nn.Module) -> list[torch.Tensor]:
    """Generate a random direction in parameter space with filter normalization.

    Only 2D+ weight tensors are perturbed. 1D parameters (BatchNorm scale/shift
    and biases) are zeroed out — perturbing them alongside fixed BN running
    statistics produces erratic loss spikes that obscure the true landscape.
    """
    direction = []
    for p in model.parameters():
        if p.dim() >= 2:
            d = torch.randn_like(p)
            # Per-filter normalization: normalize each filter slice
            for i in range(p.size(0)):
                norm_d = d[i].norm()
                norm_p = p[i].norm()
                if norm_d > 1e-10:
                    d[i] = d[i] / norm_d * norm_p
            direction.append(d)
        else:
            # Keep BN scale/shift and biases fixed
            direction.append(torch.zeros_like(p))
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
    y_clip: float | None = 5.0,
) -> go.Figure:
    """Plot 1D loss landscapes for multiple optimizers on a single figure.

    Args:
        histories: {optimizer_name: (alphas, losses)} from loss_landscape_1d().
        save_path: If given, save as HTML (replaces .png suffix if present).
        y_clip: Clip y-axis to this value to keep the basin visible.
            The extreme perturbations (α≈±1) can reach very high losses that
            squash the interesting region near the minimum. Default 5.0.

    Returns:
        plotly Figure.
    """
    fig = go.Figure()
    for name, (alphas, losses) in histories.items():
        y = np.clip(losses, 0, y_clip) if y_clip is not None else losses
        fig.add_trace(go.Scatter(
            x=alphas.tolist(), y=y.tolist(),
            mode="lines", name=name,
        ))
    fig.update_layout(
        title="1D Loss Landscape",
        xaxis_title="Perturbation α",
        yaxis_title="Loss",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    if save_path is not None:
        html_path = save_path.replace(".png", ".html") if save_path.endswith(".png") else save_path
        fig.write_html(html_path)
    return fig


def loss_landscape_2d(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int = 31,
    range_: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 2D loss landscape around θ* along two orthogonal filter-normalized directions.

    Loss is evaluated on a (steps × steps) grid:
        L(θ* + α·δ₁ + β·δ₂)  for  α, β ∈ [-range_, +range_].

    The two directions are Gram-Schmidt orthogonalised so the grid spans an
    unbiased 2D cross-section of the loss surface.

    Returns:
        alphas: 1D array of shape (steps,)       — first direction coefficients.
        betas:  1D array of shape (steps,)       — second direction coefficients.
        losses: 2D array of shape (steps, steps) — losses[i, j] = L(α_i, β_j).
    """
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)

    dir1 = _filter_normalized_direction(model)
    dir2 = _filter_normalized_direction(model)

    # Gram-Schmidt: remove the dir1 component from dir2
    dot = sum((d1 * d2).sum().item() for d1, d2 in zip(dir1, dir2))
    norm1_sq = sum((d1 ** 2).sum().item() for d1 in dir1)
    coeff = dot / (norm1_sq + 1e-12)
    dir2 = [d2 - coeff * d1 for d2, d1 in zip(dir2, dir1)]

    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()
    dir1_dev = [d.to(device) for d in dir1]
    dir2_dev = [d.to(device) for d in dir2]
    # Save original parameters — restore cleanly at every grid point
    originals = [p.data.clone() for p in model_copy.parameters()]

    alphas = np.linspace(-range_, range_, steps)
    betas = np.linspace(-range_, range_, steps)
    losses = np.zeros((steps, steps))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            for p, orig, d1, d2 in zip(model_copy.parameters(), originals, dir1_dev, dir2_dev):
                p.data.copy_(orig + d1 * float(alpha) + d2 * float(beta))
            with torch.no_grad():
                outputs = model_copy(inputs)
                loss = loss_fn(outputs, targets)
            losses[i, j] = loss.item()

    return alphas, betas, losses


def plot_loss_landscape_2d(
    alphas: np.ndarray,
    betas: np.ndarray,
    losses: np.ndarray,
    title: str = "2D Loss Landscape",
    save_path: str | None = None,
    z_clip: float | None = None,
) -> go.Figure:
    """Plot a single 2D loss landscape as a filled contour figure.

    Args:
        alphas: 1D array — first direction coefficients (x-axis).
        betas:  1D array — second direction coefficients (y-axis).
        losses: 2D array of shape (len(alphas), len(betas)), losses[i, j] = L(α_i, β_j).
        title:  Figure title.
        save_path: If given, save as HTML (replaces .png suffix if present).
        z_clip: Clip loss values above this threshold (None = no clipping).
    """
    # plotly Surface: z[i][j] = value at (x[j], y[i]) → transpose losses
    Z = np.clip(losses, 0, z_clip).T if z_clip is not None else losses.T
    fig = go.Figure(data=go.Surface(
        z=Z.tolist(),
        x=alphas.tolist(),
        y=betas.tolist(),
        colorscale="RdBu_r",
        colorbar=dict(title="Loss"),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
        ),
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="α (direction 1)",
            yaxis_title="β (direction 2)",
            zaxis_title="Loss",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        template="plotly_white",
    )
    if save_path is not None:
        html_path = save_path.replace(".png", ".html") if save_path.endswith(".png") else save_path
        fig.write_html(html_path)
    return fig
