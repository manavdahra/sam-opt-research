from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def hutchinson_trace(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 20,
    max_batch: int = 64,
) -> float:
    """Estimate tr(H) / d via the Hutchinson stochastic estimator.

    Uses Rademacher random vectors z ~ {±1}^d. Each estimate is:
        z^T H z  =  z · ∇(∇L · z)
    computed via two backward passes (Pearlmutter trick).

    Args:
        model: Trained model (in eval mode).
        loss_fn: Loss function (e.g. nn.CrossEntropyLoss()).
        loader: DataLoader used to compute the loss (a single batch suffices).
        device: Computation device.
        n_samples: Number of Rademacher vectors to average over.
        max_batch: Cap on the number of images used per Hessian estimate.
            Smaller values are faster; 64 is a good default.

    Returns:
        Scalar estimate of tr(H) / d  (sharpness proxy).
    """
    model.eval()
    # Use a single batch for speed, capped to max_batch images
    inputs, targets = next(iter(loader))
    inputs = inputs[:max_batch].to(device)
    targets = targets[:max_batch].to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)

    trace_estimates: list[float] = []
    for _ in range(n_samples):
        # Sample Rademacher vector
        zs = [torch.randint_like(p, 0, 2).float() * 2.0 - 1.0 for p in params]

        # First backward: compute ∇L
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        grads = torch.autograd.grad(loss, params, create_graph=True)

        # Dot product ∇L · z
        grad_dot_z = sum((g * z).sum() for g, z in zip(grads, zs))

        # Second backward: compute ∇(∇L · z) = Hz
        hz = torch.autograd.grad(grad_dot_z, params, retain_graph=False)

        # Estimate: z^T H z / d
        ztHz = sum((z * h).sum().item() for z, h in zip(zs, hz))
        trace_estimates.append(ztHz / d)

        # Free graph
        del grads, grad_dot_z, hz

    return float(torch.tensor(trace_estimates).mean().item())
