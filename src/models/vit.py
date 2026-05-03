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


def apply_mlp_reparam(model: nn.Module, alpha: float) -> None:
    """Apply an approximate scale reparametrization to each ViT MLP block.

    Scales the first Linear layer (weight + bias) by alpha and compensates the
    second Linear layer's input weights by 1/alpha. This is **exact** only for
    ReLU-like activations. For GELU the transform is approximate; the deviation
    grows with |alpha - 1|.

    The transform is applied **in-place**.

    Args:
        model: A ViT-B/32 returned by get_vit_b_32().
        alpha: Scale factor. alpha=1.0 is a no-op.
    """
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


def measure_reparam_deviation(model: nn.Module, x: torch.Tensor, alpha: float) -> float:
    """Return the max absolute difference after applying mlp_reparam.

    Useful for characterising how approximate the GELU reparametrization is.
    """
    model_orig = copy.deepcopy(model)
    model_reparam = copy.deepcopy(model)
    apply_mlp_reparam(model_reparam, alpha)
    model_orig.eval()
    model_reparam.eval()
    with torch.no_grad():
        out_orig = model_orig(x)
        out_reparam = model_reparam(x)
    diff = (out_orig - out_reparam).abs().max().item()
    return diff
