"""Tests for ViT-B/32 MLP reparametrisation and the analysis.reparam facade."""
import pytest
import torch
import torch.nn as nn

from src.models.vit import (
    _TaylorGELU,
    _replace_gelu_with_taylor,
    get_vit_b_32,
    apply_mlp_reparam_taylor,
    measure_reparam_deviation,
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


# ---------------------------------------------------------------------------
# apply_mlp_reparam_taylor — returns analytic bound
# ---------------------------------------------------------------------------

def test_taylor_reparam_returns_zero_for_alpha1(vit_model):
    import copy
    bound = apply_mlp_reparam_taylor(copy.deepcopy(vit_model), alpha=1.0)
    assert bound == 0.0


@pytest.mark.parametrize("alpha", [0.5, 2.0, 10.0])
def test_taylor_bound_formula(vit_model, alpha):
    """Returned bound should equal 0.4255 * |alpha - 1|."""
    import copy
    bound = apply_mlp_reparam_taylor(copy.deepcopy(vit_model), alpha=alpha)
    expected = (1.702 / 4.0) * abs(alpha - 1.0)
    assert abs(bound - expected) < 1e-9, f"bound={bound}, expected={expected}"


@pytest.mark.parametrize("alpha", [0.5, 2.0])
def test_taylor_bound_grows_with_alpha_distance(vit_model, alpha):
    """Bound should grow as alpha moves further from 1."""
    import copy
    bound_near = apply_mlp_reparam_taylor(copy.deepcopy(vit_model), alpha=1.1)
    bound_far = apply_mlp_reparam_taylor(copy.deepcopy(vit_model), alpha=alpha)
    assert bound_far > bound_near


def test_taylor_reparam_applies_weights(vit_model):
    """Taylor variant must actually scale the weights (delegates to apply_mlp_reparam)."""
    import copy
    alpha = 3.0
    model_reparam = copy.deepcopy(vit_model)
    apply_mlp_reparam_taylor(model_reparam, alpha=alpha)

    for (orig_block, reparam_block) in zip(
        vit_model.encoder.layers.children(),
        model_reparam.encoder.layers.children(),
    ):
        orig_linears = [m for m in orig_block.mlp.children() if isinstance(m, nn.Linear)]
        reparam_linears = [m for m in reparam_block.mlp.children() if isinstance(m, nn.Linear)]
        if len(orig_linears) < 2:
            continue
        assert torch.allclose(reparam_linears[0].weight, orig_linears[0].weight * alpha)
        assert torch.allclose(reparam_linears[1].weight, orig_linears[1].weight / alpha)


def test_taylor_reparam_replaces_gelu(vit_model):
    """All GELU activations in MLP blocks must be replaced with _TaylorGELU."""
    import copy
    model = copy.deepcopy(vit_model)
    apply_mlp_reparam_taylor(model, alpha=3.0)

    for block in model.encoder.layers.children():
        mlp = getattr(block, "mlp", None)
        if mlp is None:
            continue
        for child in mlp:
            assert not isinstance(child, nn.GELU), "nn.GELU still present after reparam"
            if isinstance(child, _TaylorGELU):
                break
        else:
            pytest.fail("No _TaylorGELU found in MLP block after reparam")


@pytest.mark.parametrize("alpha", [0.5, 2.0, 5.0])
def test_taylor_reparam_is_function_preserving(vit_model, dummy_input, alpha):
    """Weight scaling must preserve output when the activation is TaylorGELU.

    Both the baseline and reparametrised model must use TaylorGELU — comparing
    against the original GELU model is wrong because the activation itself changes.
    """
    import copy
    # Baseline: TaylorGELU, unscaled weights
    model_base = copy.deepcopy(vit_model)
    _replace_gelu_with_taylor(model_base)
    # Reparametrised: TaylorGELU, scaled weights
    model_reparam = copy.deepcopy(vit_model)
    apply_mlp_reparam_taylor(model_reparam, alpha=alpha)

    with torch.no_grad():
        out_base = model_base(dummy_input)
        out_reparam = model_reparam(dummy_input)
    assert torch.allclose(out_base, out_reparam, atol=1e-4), (
        f"alpha={alpha}: max diff={(out_base - out_reparam).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# measure_reparam_deviation — empirical output diff
# ---------------------------------------------------------------------------

def test_measure_deviation_noop(vit_model, dummy_input):
    dev = measure_reparam_deviation(vit_model, dummy_input, alpha=1.0)
    assert dev == 0.0


def test_measure_deviation_grows_with_alpha(vit_model, dummy_input):
    dev_small = measure_reparam_deviation(vit_model, dummy_input, alpha=1.5)
    dev_large = measure_reparam_deviation(vit_model, dummy_input, alpha=5.0)
    assert dev_large > dev_small


# ---------------------------------------------------------------------------
# apply_reparam facade
# ---------------------------------------------------------------------------

def test_facade_resnet18_returns_none():
    from src.models.resnet18 import get_resnet18
    model = get_resnet18(num_classes=10)
    result = apply_reparam(model, "resnet18", alpha=2.0)
    assert result is None


def test_facade_vit_returns_float(vit_model):
    import copy
    result = apply_reparam(copy.deepcopy(vit_model), "vit_b_32", alpha=2.0)
    assert isinstance(result, float)
    assert result > 0.0


def test_facade_unknown_model_raises(vit_model):
    with pytest.raises(ValueError, match="Unknown model"):
        apply_reparam(vit_model, "unknown_arch", alpha=2.0)
