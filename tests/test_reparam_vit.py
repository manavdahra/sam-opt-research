"""Tests for ViT-B/32 MLP reparametrisation and the analysis.reparam facade."""
import pytest
import torch
import torch.nn as nn

from src.models.vit import (
    get_vit_b_32,
    apply_layernorm_reparam,
)
from src.analysis.reparam import apply_reparam


@pytest.fixture(scope="module")
def vit_model():
    m = get_vit_b_32(num_classes=10, pretrained=False)
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_input():
    torch.manual_seed(0)
    return torch.randn(2, 3, 224, 224)


def test_facade_resnet18_returns_none():
    from src.models.resnet18 import get_resnet18
    model = get_resnet18(num_classes=10)
    result = apply_reparam(model, "resnet18", alpha=2.0)
    assert result is None


def test_facade_vit_returns_none(vit_model):
    import copy
    result = apply_reparam(copy.deepcopy(vit_model), "vit_b_32", alpha=2.0)
    assert result is None


def test_facade_unknown_model_raises(vit_model):
    with pytest.raises(ValueError, match="Unknown model"):
        apply_reparam(vit_model, "unknown_arch", alpha=2.0)


def test_layernorm_reparam_noop_for_alpha1(vit_model):
    import copy
    model = copy.deepcopy(vit_model)
    # Capture original ln_2 weights
    orig_weights = [
        block.ln_2.weight.clone()
        for block in model.encoder.layers.children()
    ]
    apply_layernorm_reparam(model, alpha=1.0)
    for block, orig_w in zip(model.encoder.layers.children(), orig_weights):
        assert torch.equal(block.ln_2.weight, orig_w)


def test_layernorm_reparam_scales_ln2_and_linear1(vit_model):
    """ln_2.weight/bias must be scaled by alpha; linear1.weight by 1/alpha."""
    import copy
    alpha = 3.0
    model_orig = copy.deepcopy(vit_model)
    model_reparam = copy.deepcopy(vit_model)
    apply_layernorm_reparam(model_reparam, alpha=alpha)

    for orig_block, reparam_block in zip(
        model_orig.encoder.layers.children(),
        model_reparam.encoder.layers.children(),
    ):
        assert torch.allclose(reparam_block.ln_2.weight, orig_block.ln_2.weight * alpha)
        assert torch.allclose(reparam_block.ln_2.bias, orig_block.ln_2.bias * alpha)
        orig_l1 = [m for m in orig_block.mlp.children() if isinstance(m, nn.Linear)][0]
        reparam_l1 = [m for m in reparam_block.mlp.children() if isinstance(m, nn.Linear)][0]
        assert torch.allclose(reparam_l1.weight, orig_l1.weight / alpha)
        # linear1.bias must be unchanged
        assert torch.allclose(reparam_l1.bias, orig_l1.bias)


@pytest.mark.parametrize("alpha", [0.5, 2.0, 5.0, 10.0])
def test_layernorm_reparam_is_exactly_function_preserving(vit_model, dummy_input, alpha):
    """The LayerNorm-based reparam must not change model output at all."""
    import copy
    model_orig = copy.deepcopy(vit_model)
    model_reparam = copy.deepcopy(vit_model)
    apply_layernorm_reparam(model_reparam, alpha=alpha)

    model_orig.eval()
    model_reparam.eval()
    with torch.no_grad():
        out_orig = model_orig(dummy_input)
        out_reparam = model_reparam(dummy_input)
    assert torch.allclose(out_orig, out_reparam, atol=1e-4), (
        f"alpha={alpha}: max diff={(out_orig - out_reparam).abs().max().item():.6f}"
    )
