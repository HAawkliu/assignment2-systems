from __future__ import annotations

import torch
from torch import Tensor, nn

from .layers import Linear


def silu(in_features: Tensor) -> Tensor:
    return in_features * torch.sigmoid(in_features)


class SwiGLU(nn.Module):
    """Position-wise SwiGLU feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_in=d_model, d_out=d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = Linear(d_in=d_ff, d_out=d_model, bias=False, device=device, dtype=dtype)
        self.w3 = Linear(d_in=d_model, d_out=d_ff, bias=False, device=device, dtype=dtype)

    def forward(self, in_features: Tensor) -> Tensor:
        return self.w2(silu(self.w1(in_features)) * self.w3(in_features))


def run_swiglu_from_weights(
    *,
    d_model: int,
    d_ff: int,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    with torch.no_grad():
        swiglu.w1.weight.copy_(w1_weight)
        swiglu.w2.weight.copy_(w2_weight)
        swiglu.w3.weight.copy_(w3_weight)
    return swiglu(in_features)


def run_silu(in_features: Tensor) -> Tensor:
    return silu(in_features)
