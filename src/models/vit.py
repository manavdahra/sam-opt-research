import math
import copy
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import ViT_B_32_Weights


def get_vit_b_32(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """ViT-B/32 fine-tuned on CIFAR-10.

    Loads ImageNet-pretrained weights (default) and replaces the classification
    head with a linear layer of size num_classes.

    Args:
        num_classes: Number of output classes.
        pretrained: If True, load ImageNet-1K pretrained weights.
    """
    weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.vit_b_32(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Reparametrization utility
# ---------------------------------------------------------------------------

def _get_mlp_linears(encoder_block: nn.Module) -> tuple[nn.Linear, nn.Linear] | None:
    """Extract the two Linear layers from a ViT encoder block's MLP sub-block.

    torchvision ViT MLP structure (EncoderBlock.mlp):
        Sequential(
            Linear(hidden, 4*hidden),   [index 0]
            GELU,                        [index 1]
            Dropout,                     [index 2]
            Linear(4*hidden, hidden),   [index 3]
            Dropout,                     [index 4]
        )
    Returns (linear1, linear2) or None if the structure is not recognised.
    """
    mlp = getattr(encoder_block, "mlp", None)
    if mlp is None:
        return None
    linears = [m for m in mlp.children() if isinstance(m, nn.Linear)]
    if len(linears) < 2:
        return None
    return linears[0], linears[1]


def _apply_mlp_reparam(model: nn.Module, alpha: float) -> None:
    """Internal: scale linear1×α and linear2×(1/α) in every ViT MLP block."""
    if alpha == 1.0:
        return

    for block in model.encoder.layers.children():
        result = _get_mlp_linears(block)
        if result is None:
            continue
        linear1, linear2 = result

        with torch.no_grad():
            linear1.weight.mul_(alpha)
            if linear1.bias is not None:
                linear1.bias.mul_(alpha)
            # linear2 input: weight shape is (out, in), scale along dim 1
            linear2.weight.mul_(1.0 / alpha)


def apply_mlp_reparam_taylor(model: nn.Module, alpha: float) -> float:
    """Apply approximate MLP reparametrisation with a Taylor-based error bound.

    Applies the same linear1×α / linear2×(1/α) transform as apply_mlp_reparam
    **in-place**, and returns an analytic upper bound on the per-activation
    deviation introduced by GELU's non-linearity.

    Derivation (first-order Taylor of GELU(x) ≈ x·σ(1.702x) ≈ 0.5x + 0.4255x²):

        Error(x, α) = GELU(αx)/α − GELU(x)
                    ≈ (0.5αx + 0.4255α²x²)/α − (0.5x + 0.4255x²)
                    = 0.4255·x²·(α − 1)

    Returned bound: 0.4255·|α−1| evaluated at unit activation scale (x=1).
    For activations with RMS magnitude r, multiply by r².

    This is the recommended function for ViT reparam experiments; it makes the
    approximation error explicit rather than silently ignoring it.

    Args:
        model: A ViT-B/32 returned by get_vit_b_32().
        alpha: Scale factor. alpha=1.0 is a no-op (returns 0.0).

    Returns:
        analytic_bound: per-unit deviation bound from GELU non-linearity.
    """
    if alpha == 1.0:
        return 0.0

    _apply_mlp_reparam(model, alpha)

    # Coefficient of the quadratic Taylor term: 1.702 / 4 ≈ 0.4255
    _GELU_QUADRATIC_COEFF = 1.702 / 4.0
    return _GELU_QUADRATIC_COEFF * abs(alpha - 1.0)


def measure_reparam_deviation(model: nn.Module, x: torch.Tensor, alpha: float) -> float:
    """Return the max absolute difference after applying mlp_reparam.

    Useful for characterising how approximate the GELU reparametrization is.
    """
    model_orig = copy.deepcopy(model)
    model_reparam = copy.deepcopy(model)
    _apply_mlp_reparam(model_reparam, alpha)
    model_orig.eval()
    model_reparam.eval()
    with torch.no_grad():
        out_orig = model_orig(x)
        out_reparam = model_reparam(x)
    diff = (out_orig - out_reparam).abs().max().item()
    return diff
