"""Reparametrisation — consolidated API.

Re-exports the model-specific functions and provides a unified
``apply_reparam`` dispatcher used by experiment scripts.

ResNet-18 (exact):
    ReLU is 1-homogeneous → scaling BN1/conv1 output by α and compensating
    conv2's input by 1/α preserves the network function exactly.

ViT-B/32 (approximate, Taylor-bounded):
    GELU is *not* scale-homogeneous.  The Linear→GELU→Linear reparam applies
    the same linear1×α / linear2×(1/α) transform, but incurs a per-activation
    deviation ≈ 0.4255·x²·(α−1) from the GELU quadratic term.
    ``apply_mlp_reparam_taylor`` applies the transform and returns the analytic
    bound; use ``measure_reparam_deviation`` for empirical confirmation.
"""
from __future__ import annotations

import torch.nn as nn

from src.models.resnet18 import apply_relu_reparam, verify_reparam_resnet
from src.models.vit import (
    apply_mlp_reparam_taylor,
    measure_reparam_deviation,
)

__all__ = [
    "apply_reparam",
    "apply_relu_reparam",
    "verify_reparam_resnet",
    "apply_mlp_reparam_taylor",
    "measure_reparam_deviation",
]


def apply_reparam(model: nn.Module, model_name: str, alpha: float) -> float | None:
    """Dispatch the correct reparametrisation function for a given model.

    For ResNet-18 the transform is exact (ReLU homogeneity) and ``None`` is
    returned.  For ViT-B/32 the Taylor-bounded approximate transform is applied
    and the analytic deviation bound (per unit activation at x=1) is returned
    so callers can log it alongside results.

    Args:
        model: Model instance (output of ``get_resnet18`` or ``get_vit_b_32``).
        model_name: ``"resnet18"`` or ``"vit_b_32"``.
        alpha: Scale factor.  1.0 is a no-op.

    Returns:
        ``None`` for ResNet-18; analytic Taylor bound (float) for ViT-B/32.

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name == "resnet18":
        apply_relu_reparam(model, alpha)
        return None
    elif model_name == "vit_b_32":
        bound = apply_mlp_reparam_taylor(model, alpha)
        return bound
    else:
        raise ValueError(f"Unknown model for reparametrisation: {model_name!r}")
