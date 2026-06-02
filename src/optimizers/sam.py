import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    r"""Sharpness-Aware Minimization (Foret et al., 2021).

    Wraps a base SGD optimizer. Exposes a two-step API:
        first_step()  — perturb parameters to the local maximum of the loss
        second_step() — restore parameters and apply the base optimizer update
    
    .. math::
        \delta^*_\text{SAM} = \arg\max_{\|\delta\|_2 \leq \rho} L(\theta + \delta)
        
        \theta \leftarrow \theta - \eta \nabla L(\theta + \delta^*_\text{SAM})

    .. code-block:: python
        optimizer = SAM(model.parameters(), base_optimizer=torch.optim.SGD,
                        rho=0.05, lr=0.1, momentum=0.9, weight_decay=5e-4)
        # inner loop
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        loss = criterion(model(x), y)
        loss.backward()
        optimizer.second_step(zero_grad=True)
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
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # θ ← θ + ε
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])  # restore θ
                del self.state[p]["e_w"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError(
            "SAM does not support a single step(). Use first_step() / second_step()."
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
