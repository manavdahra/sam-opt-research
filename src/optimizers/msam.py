import torch
from torch.optim import Optimizer

from .sam import SAM


class MSAM(Optimizer):
    """M-SAM: Riemannian SAM (Jacobsen & Arvanitidis, 2025).

    The perturbation is the SAM perturbation rescaled by the inverse of the
    Riemannian metric factor:

        δ*_MSAM = δ*_SAM / sqrt(1 + ‖∇ℓ(θ)‖²₂)

    This makes the perturbation invariant under reparametrization of the loss
    landscape (metric-aware). Same two-step API as SAM.
    """

    def __init__(
        self,
        params,
        base_optimizer: type[Optimizer],
        rho: float = 0.05,
        **kwargs,
    ) -> None:
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        # Riemannian rescaling factor
        metric_scale = 1.0 / (1.0 + grad_norm**2).sqrt()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12) * metric_scale
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError(
            "MSAM does not support a single step(). Use first_step() / second_step()."
        )

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.stack(norms).norm(p=2)

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
