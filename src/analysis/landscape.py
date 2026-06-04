import gc

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
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                d = torch.randn_like(p)
                # Per-filter normalization: normalize each filter slice
                for i in range(p.size(0)):
                    norm_d = d[i].norm()
                    norm_p = p[i].norm()
                    if norm_d > 1e-10:
                        d[i].mul_(norm_p / norm_d)
                direction.append(d)
            else:
                # Keep BN scale/shift and biases fixed
                direction.append(torch.zeros_like(p))
    return direction


def loss_landscape_2d(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int = 31,
    range_: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute the 2D loss landscape around θ* along two orthogonal filter-normalized directions.

    Loss is evaluated on a (steps x steps) grid:
    $$
        L(\theta^* + \alpha \cdot \delta_1 + \beta \cdot \delta_2)  for  \alpha, \beta \in [-1.0, +1.0]
    $$
        where $\delta_1$ and $\delta_2$ are random filter-normalized directions in parameter space,

    The two directions are Gram-Schmidt orthogonalised so the grid spans an
    unbiased 2D cross-section of the loss surface.

    Returns:
        alphas: 1D array of shape (steps,)       — first direction coefficients.
        betas:  1D array of shape (steps,)       — second direction coefficients.
        losses: 2D array of shape (steps, steps) — losses[i, j] = L($\alpha_i$, $\beta_j$).
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

    # Re-apply filter normalization to dir2 after orthogonalization so that
    # both directions have comparable per-filter scales (matching model weights).
    dir2_renorm = []
    with torch.no_grad():
        for d2, p in zip(dir2, model.parameters()):
            if p.dim() >= 2:
                d = d2.clone()
                for i in range(p.size(0)):
                    norm_d = d[i].norm()
                    norm_p = p[i].norm()
                    if norm_d > 1e-10:
                        d[i].mul_(norm_p / norm_d)
                dir2_renorm.append(d)
            else:
                dir2_renorm.append(d2)
    # Release the Gram-Schmidt dir2 intermediates (2D-param tensors no longer needed)
    del dir2
    dir2 = dir2_renorm
    del dir2_renorm

    # Save original parameters — perturb model in-place, restore after grid sweep
    # (avoids a full deepcopy of the model on GPU)
    originals = [p.data.clone() for p in model.parameters()]
    was_training = model.training
    model.eval()

    alphas = np.linspace(-range_, range_, steps)
    betas = np.linspace(-range_, range_, steps)
    losses = np.zeros((steps, steps))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            with torch.no_grad():
                for p, orig, d1, d2 in zip(model.parameters(), originals, dir1, dir2):
                    p.data.copy_(orig + d1 * float(alpha) + d2 * float(beta))
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            losses[i, j] = loss.item()

    # Restore original parameters and training mode
    with torch.no_grad():
        for p, orig in zip(model.parameters(), originals):
            p.data.copy_(orig)
    model.train(was_training)

    # Explicitly free all GPU tensors before returning
    del originals, dir1, dir2, inputs, targets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return alphas, betas, losses


def plot_loss_landscape_2d(
    alphas: np.ndarray,
    betas: np.ndarray,
    losses: np.ndarray,
    title: str = "2D Loss Landscape",
    save_path: str | None = None,
    z_clip: float | None = 5.0,
) -> go.Figure:
    r"""Plot a single 2D loss landscape as a filled contour figure.

    Args:
        alphas: 1D array — first direction coefficients (x-axis).
        betas:  1D array — second direction coefficients (y-axis).
        losses: 2D array of shape (len(alphas), len(betas)), losses[i, j] = L($\alpha_i$, $\beta_j$).
        title:  Figure title.
        save_path: If given, save as HTML (replaces .png suffix if present).
        z_clip: Clip loss values above this threshold (None = no clipping).
    """
    
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
