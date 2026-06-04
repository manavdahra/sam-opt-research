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


def apply_layernorm_reparam(model: nn.Module, alpha: float) -> None:
    r"""Apply an exact, function-preserving reparametrisation to ViT-B/32.

    For each encoder block the transform is:

        ln_2.weight  (gamma) *= alpha
        ln_2.bias    (beta) *= alpha
        mlp.linear1.weight *= 1/alpha

    This preserves the network function exactly because the transform only crosses
    a linear boundary (LayerNorm at Linear); we avoid touching GELU which is non-homogeneous.

    Args:
        model: A ViT-B/32 returned by get_vit_b_32().
        alpha: Scale factor.  1.0 is a no-op.
    """
    if alpha == 1.0:
        return

    for block in model.encoder.layers.children():
        ln_2: nn.LayerNorm = getattr(block, "ln_2", None)
        if ln_2 is None:
            continue
        result = _get_mlp_linears(block)
        if result is None:
            continue
        linear1, _ = result

        with torch.no_grad():
            if ln_2.weight is not None:
                ln_2.weight.mul_(alpha)
            if ln_2.bias is not None:
                ln_2.bias.mul_(alpha)
            linear1.weight.mul_(1.0 / alpha)

