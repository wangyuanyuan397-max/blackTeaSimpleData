"""AdamNorm optimizer with AdaNorm-style gradient correction."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer


class AdamNorm(Optimizer):
    """Adam with AdaNorm gradient-norm correction.

    The second-moment estimate uses the raw gradient, while the first-moment
    estimate uses the AdaNorm-corrected gradient. Weight decay is decoupled in
    the AdamW style to match the existing baseline optimizer semantics.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] | list[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        gamma: float = 0.95,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if len(betas) != 2:
            raise ValueError(f"Invalid beta pair: {betas}")
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"Invalid gamma value: {gamma}")

        defaults = {
            "lr": float(lr),
            "betas": (beta1, beta2),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "gamma": float(gamma),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            gamma = group["gamma"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamNorm does not support sparse gradients")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )
                    state["grad_norm_ema"] = torch.zeros(
                        (),
                        dtype=grad.dtype,
                        device=grad.device,
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad_norm_ema = state["grad_norm_ema"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    parameter.mul_(1.0 - lr * weight_decay)

                grad_norm = torch.linalg.vector_norm(grad)
                grad_norm_ema.mul_(gamma).add_(grad_norm, alpha=1.0 - gamma)

                corrected_grad = grad
                if bool(grad_norm_ema > grad_norm):
                    corrected_grad = grad * (grad_norm_ema / (grad_norm + eps))

                exp_avg.mul_(beta1).add_(corrected_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                parameter.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
