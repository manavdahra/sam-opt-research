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
            Linear(hidden, 4*hidden),    [index 0]
            GELU,                        [index 1]
            Dropout,                     [index 2]
            Linear(4*hidden, hidden),    [index 3]
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
    """Internal: scale linear1 x $\alpha$ and linear2 x $(1/\alpha)$ in every ViT MLP block."""
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


class _TaylorGELU(nn.Module):
    """First-order Taylor approximation of GELU around x=0: f(x) = 0.5 * x.

    GELU'(0) = Phi(0) = 0.5, so the linearisation is f(x) ≈ 0.5x.
    This IS positively homogeneous: f(alpha*x) = alpha*f(x), which makes the
    linear1 x alpha / linear2 x (1/alpha) weight scaling exactly function-
    preserving under this activation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x


def _replace_gelu_with_taylor(model: nn.Module) -> None:
    """Replace every GELU in the encoder MLP blocks with _TaylorGELU in-place."""
    for block in model.encoder.layers.children():
        mlp = getattr(block, "mlp", None)
        if mlp is None:
            continue
        for i, child in enumerate(mlp):
            if isinstance(child, nn.GELU):
                mlp[i] = _TaylorGELU()


def apply_mlp_reparam_taylor(model: nn.Module, alpha: float) -> float:
    r"""Apply function-preserving MLP reparametrisation via Taylor-linearised GELU.

    Two things happen in-place:

    1. Every GELU activation in the encoder MLP blocks is replaced by its
       first-order Taylor approximation around x=0, i.e. f(x) = 0.5x.
       This activation IS positively homogeneous (f(αx) = αf(x)), so the
       weight scaling below becomes exactly function-preserving.

    2. For each MLP block: linear1 weights/bias are scaled by α and linear2
       weights are scaled by 1/α, leaving the block's output unchanged.

    The returned bound quantifies the per-activation error introduced by
    replacing GELU with its linear approximation (independent of α):

    .. math::
        \epsilon(x) = |\text{GELU}(x) - 0.5x|
                    \approx \tfrac{1.702}{4} x^2  \quad \text{(quadratic residual)}

    The returned scalar is evaluated at unit activation scale (|x|=1):
    0.4255 * |alpha - 1| is zero when alpha=1 (no-op) and grows linearly.

    Args:
        model: A ViT-B/32 returned by get_vit_b_32().
        alpha: Scale factor. alpha=1.0 is a no-op (returns 0.0).

    Returns:
        analytic_bound: GELU linearisation error at unit activation scale.
    """
    if alpha == 1.0:
        return 0.0

    _replace_gelu_with_taylor(model)
    _apply_mlp_reparam(model, alpha)

    # Coefficient of the quadratic GELU residual: 1.702 / 4 ≈ 0.4255
    _GELU_QUADRATIC_COEFF = 1.702 / 4.0
    return _GELU_QUADRATIC_COEFF * abs(alpha - 1.0)


def measure_reparam_deviation(model: nn.Module, x: torch.Tensor, alpha: float) -> float:
    """Return the max absolute output difference between the original GELU model
    and the TaylorGELU + weight-scaled reparametrised model.

    Useful for characterising how much the GELU→TaylorGELU substitution (plus
    weight scaling) shifts the model output as alpha moves away from 1.
    Returns 0.0 when alpha=1.0 (no reparametrisation applied).
    """
    if alpha == 1.0:
        return 0.0
    model_orig = copy.deepcopy(model)
    model_reparam = copy.deepcopy(model)
    _replace_gelu_with_taylor(model_reparam)
    _apply_mlp_reparam(model_reparam, alpha)
    model_orig.eval()
    model_reparam.eval()
    with torch.no_grad():
        out_orig = model_orig(x)
        out_reparam = model_reparam(x)
    diff = (out_orig - out_reparam).abs().max().item()
    return diff
