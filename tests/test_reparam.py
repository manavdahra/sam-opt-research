import torch
import pytest
from src.models.resnet18 import get_resnet18, apply_relu_reparam, verify_reparam_resnet


@pytest.fixture
def model():
    m = get_resnet18(num_classes=10)
    m.eval()
    return m


@pytest.fixture
def dummy_input():
    torch.manual_seed(0)
    return torch.randn(4, 3, 32, 32)


def test_reparam_preserves_function_alpha2(model, dummy_input):
    diff = verify_reparam_resnet(model, dummy_input, alpha=2.0)
    assert diff < 1e-4, f"Max output diff after reparam (alpha=2.0): {diff}"


def test_reparam_preserves_function_alpha0_5(model, dummy_input):
    diff = verify_reparam_resnet(model, dummy_input, alpha=0.5)
    assert diff < 1e-4, f"Max output diff after reparam (alpha=0.5): {diff}"


def test_reparam_preserves_function_alpha10(model, dummy_input):
    diff = verify_reparam_resnet(model, dummy_input, alpha=10.0)
    assert diff < 1e-4, f"Max output diff after reparam (alpha=10.0): {diff}"


def test_reparam_noop_alpha1(model, dummy_input):
    """alpha=1.0 must be a strict no-op (weights unchanged)."""
    import copy
    original_state = copy.deepcopy(model.state_dict())
    apply_relu_reparam(model, alpha=1.0)
    for key in original_state:
        assert torch.equal(model.state_dict()[key], original_state[key]), (
            f"Weight '{key}' changed despite alpha=1.0"
        )


def test_reparam_scales_weights(model):
    """Verify that BN1 and conv2 weights are actually scaled (not silently skipped)."""
    import copy
    import torchvision.models.resnet as tvr

    model_reparam = copy.deepcopy(model)
    alpha = 3.0
    apply_relu_reparam(model_reparam, alpha=alpha)

    for (name, orig_block), (_, reparam_block) in zip(
        [(n, m) for n, m in model.named_modules() if isinstance(m, tvr.BasicBlock)],
        [(n, m) for n, m in model_reparam.named_modules() if isinstance(m, tvr.BasicBlock)],
    ):
        # BN1 gamma should be scaled by alpha
        assert torch.allclose(reparam_block.bn1.weight, orig_block.bn1.weight * alpha), (
            f"Block {name}: bn1.weight not scaled by alpha"
        )
        # conv2 weights should be scaled by 1/alpha
        assert torch.allclose(reparam_block.conv2.weight, orig_block.conv2.weight / alpha), (
            f"Block {name}: conv2.weight not scaled by 1/alpha"
        )
