import torch
from torch.optim import Optimizer

class ASAM(Optimizer):
    r"""ASAM optimizer implementation.
    Reference:
        Kwon et al., "Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning
        https://arxiv.org/abs/2102.06171
    ASAM modifies the SAM perturbation to be scale-invariant:
    .. math::
        \epsilon = \rho \cdot \frac{T_w^2 \nabla L}{\|T_w \nabla L\|_2} \quad \text{where} \quad T_w = \text{diag}(|w|)

    This makes the perturbation invariant to the scale of the weights, which is important for modern architectures with normalization layers (e.g. ResNets, ViTs) where weight norms can vary widely across layers.
    ASAM has the same two-step API as SAM (first_step() to perturb, second_step() to update), and can be used with any base optimizer (e.g. SGD, Adam).
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
        r"""Compute and apply ASAM perturbation.
         ASAM perturbation is:
         .. math::
            \epsilon = \rho \cdot \frac{T_w^2 \nabla L}{\|T_w \nabla L\|_2} \quad \text{where} \quad T_w = \text{diag}(|w|)
            This makes the perturbation invariant to the scale of the weights, which is important for modern architectures with normalization layers (e.g. ResNets, ViTs) where weight norms can vary widely across layers.
        Args:
            zero_grad: If True, set gradients to zero after the step (usually True).
        """
        t_grad_norm = self._t_grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (t_grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                
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
                if "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
                del self.state[p]["e_w"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError(
            "ASAM does not support a single step(). Use first_step() / second_step()."
        )

    def _t_grad_norm(self) -> torch.Tensor:
        r"""Compute the ASAM gradient norm: \|T_w \nabla L\|_2 where T_w = diag(|w| + eta).
        This is the key difference from SAM, which uses the unscaled gradient norm.
        """
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
