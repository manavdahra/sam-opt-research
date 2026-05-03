import torch
from torch.optim import Optimizer


class ASAM(Optimizer):
    """Adaptive Sharpness-Aware Minimization (Kwon et al., 2021).

    Adaptive perturbation:
        ε = ρ · T_w² ∇L / ‖T_w ∇L‖₂
    where T_w = diag(|w|), making the perturbation scale-invariant.

    Same two-step API as SAM.
    """

    def __init__(
        self,
        params,
        base_optimizer: type[Optimizer],
        rho: float = 0.5,
        eta: float = 0.01,
        **kwargs,
    ) -> None:
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        defaults = dict(rho=rho, eta=eta, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        # Compute T_w scaled gradient: T_w * g = |w| * g  (element-wise)
        # Then norm: ‖T_w * g‖₂
        t_grad_norm = self._t_grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (t_grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # ε = ρ · |w|² · g / ‖|w| · g‖₂
                t_w = p.abs() + group["eta"]
                e_w = (t_w * t_w * p.grad) * scale
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
            "ASAM does not support a single step(). Use first_step() / second_step()."
        )

    def _t_grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            eta = group["eta"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                t_w = p.abs() + eta
                norms.append((t_w * p.grad).norm(p=2).to(shared_device))
        return torch.stack(norms).norm(p=2)

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
