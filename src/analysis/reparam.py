"""Reparametrisation — consolidated API.

Re-exports the model-specific functions and provides a unified
``apply_reparam`` dispatcher used by experiment scripts.

ResNet-18 (exact):
    ReLU is 1-homogeneous → scaling BN1/conv1 output by alpha and compensating
    conv2's input by 1/alpha preserves the network function exactly.

ViT-B/32 (exact):
    Scales ln_2 affine parameters (gamma, beta) by alpha and mlp.linear1.weight by 1/alpha.
    The transform crosses only the linear LayerNorm → Linear1 boundary;
    GELU is never touched, so no homogeneity assumption is required.
"""
from __future__ import annotations

import torch.nn as nn

from src.models.resnet18 import apply_relu_reparam, verify_reparam_resnet
from src.models.vit import (
    apply_layernorm_reparam,
)

__all__ = [
    "apply_reparam",
    "apply_relu_reparam",
    "verify_reparam_resnet",
    "apply_layernorm_reparam",
]


def apply_reparam(model: nn.Module, model_name: str, alpha: float) -> None:
    """Dispatch the correct reparametrisation function for a given model.

    Both ResNet-18 and ViT-B/32 transforms are now exact and function-preserving:

    - ResNet-18: scales BN1 affine params by alpha and conv2.weight by 1/alpha.
      Exact because ReLU is 1-homogeneous.
    - ViT-B/32: scales ln_2 affine params (gamma, beta) by alpha and mlp.linear1.weight
      by 1/alpha.  Exact because the transform crosses only a linear boundary
      (LayerNorm → Linear); GELU is never touched.

    Args:
        model: Model instance (output of ``get_resnet18`` or ``get_vit_b_32``).
        model_name: ``"resnet18"`` or ``"vit_b_32"``.
        alpha: Scale factor.  1.0 is a no-op.

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name == "resnet18":
        apply_relu_reparam(model, alpha)
        return None
    elif model_name == "vit_b_32":
        apply_layernorm_reparam(model, alpha)
        return None
    else:
        raise ValueError(f"Unknown model for reparametrisation: {model_name!r}")
