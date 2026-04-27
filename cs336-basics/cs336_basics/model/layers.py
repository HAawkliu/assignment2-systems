from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Linear(nn.Module):
    """Bias-optional linear layer with CS336 initialization."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(d_out, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = (2.0 / (self.d_in + self.d_out)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, in_features: Tensor) -> Tensor:
        out = torch.matmul(in_features, self.weight.transpose(0, 1))
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(nn.Module):
    """Token embedding layer with CS336 initialization."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return F.embedding(token_ids, self.weight)


def run_linear_from_weights(*, d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    layer = Linear(d_in=d_in, d_out=d_out, bias=False, device=weights.device, dtype=weights.dtype)
    with torch.no_grad():
        layer.weight.copy_(weights)
    return layer(in_features)


def run_embedding_from_weights(*, vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    layer = Embedding(vocab_size=vocab_size, d_model=d_model, device=weights.device, dtype=weights.dtype)
    with torch.no_grad():
        layer.weight.copy_(weights)
    return layer(token_ids)
