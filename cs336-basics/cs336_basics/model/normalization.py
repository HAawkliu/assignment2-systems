from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension."""

    def __init__(
        self,
        d_model: int,
        *,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0)

    def forward(self, in_features: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(in_features * in_features, dim=-1, keepdim=True) + self.eps)
        return (in_features / rms) * self.weight


def run_rmsnorm_from_weights(*, d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    norm = RMSNorm(d_model=d_model, eps=eps, device=weights.device, dtype=weights.dtype)
    with torch.no_grad():
        norm.weight.copy_(weights)
    return norm(in_features)
