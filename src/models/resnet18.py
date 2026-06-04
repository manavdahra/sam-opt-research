import torch
import torch.nn as nn
import torchvision.models as tvm


def get_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for CIFAR-10.

    Modifications vs. the ImageNet variant:
    - First conv: 7x7 stride-2 at 3x3 stride-1
    - Removes the initial MaxPool layer
    - Final FC: 512 at num_classes
    """
    model = tvm.resnet18(weights=None, num_classes=num_classes)
    # Adapt stem for 32x32 inputs
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def apply_relu_reparam(model: nn.Module, alpha: float) -> None:
    r"""Apply a function-preserving scale reparametrization to ResNet-18.

    For each BasicBlock, ReLU is 1-homogeneous so scaling conv1's output by $\alpha$
    and compensating with $1/\alpha$ on conv2's input channels exactly preserves the
    network function.

    The transform is applied **in-place**.

    Args:
        model: A ResNet-18 returned by get_resnet18().
        alpha: Scale factor. $\alpha=1.0$ is a no-op.
    """
    if alpha == 1.0:
        return

    for module in model.modules():
        if not isinstance(module, tvm.resnet.BasicBlock):
            continue

        bn1: nn.BatchNorm2d = module.bn1
        conv2: nn.Conv2d = module.conv2

        # Scale BN1 affine parameters so the BN output is multiplied by alpha.
        # This correctly propagates the scale through BN (which would otherwise
        # absorb any scaling applied directly to conv1).
        with torch.no_grad():
            if bn1.weight is not None:
                bn1.weight.mul_(alpha)
            if bn1.bias is not None:
                bn1.bias.mul_(alpha)

        # Compensate conv2 input (in_channels dimension, dim 1)
        with torch.no_grad():
            conv2.weight.mul_(1.0 / alpha)


def verify_reparam_resnet(model: nn.Module, x: torch.Tensor, alpha: float) -> float:
    """Verify function-preservation of relu_reparam.

    Returns the max absolute difference between the original and reparametrized
    model outputs for input x.
    """
    import copy
    model_orig = copy.deepcopy(model)
    model_reparam = copy.deepcopy(model)
    apply_relu_reparam(model_reparam, alpha)
    model_orig.eval()
    model_reparam.eval()
    with torch.no_grad():
        out_orig = model_orig(x)
        out_reparam = model_reparam(x)
    diff = (out_orig - out_reparam).abs().max().item()
    return diff
