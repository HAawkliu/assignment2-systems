from __future__ import annotations

from collections.abc import Iterable

import torch


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip global gradient norm to at most ``max_l2_norm``.

    Assignment requirement:
    - Use epsilon = 1e-6 in the scaling denominator for numerical stability.
    - Modify parameter gradients in place.
    """
    if max_l2_norm < 0:
        raise ValueError(f"max_l2_norm must be non-negative, got {max_l2_norm}.")

    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    grad_norms = [torch.linalg.vector_norm(g.detach(), ord=2) for g in grads]
    total_norm = torch.linalg.vector_norm(torch.stack(grad_norms), ord=2)

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for grad in grads:
            grad.mul_(clip_coef.to(device=grad.device, dtype=grad.dtype))


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Adapter-facing wrapper for section 4.5."""
    clip_gradients(parameters=parameters, max_l2_norm=max_l2_norm)
