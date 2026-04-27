from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .layers import Linear


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE expects even d_k, got {d_k}.")
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        seq_len = x.shape[-2]
        if token_positions.shape[-1] != seq_len:
            raise ValueError("token_positions last dimension must match sequence length.")

        token_positions = token_positions.to(device=x.device)
        while token_positions.ndim < x.ndim - 1:
            token_positions = token_positions.unsqueeze(-2)

        half_d = self.d_k // 2
        dim_ids = torch.arange(0, self.d_k, 2, device=x.device, dtype=x.dtype)
        inv_freq = self.theta ** (-dim_ids / self.d_k)
        angles = token_positions.to(x.dtype).unsqueeze(-1) * inv_freq.view(*((1,) * token_positions.ndim), half_d)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack((out_even, out_odd), dim=-1).flatten(start_dim=-2)


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope: RotaryPositionalEmbedding | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.rope = rope
        self.q_proj = Linear(d_in=d_model, d_out=d_model, bias=False, device=device, dtype=dtype)
        self.k_proj = Linear(d_in=d_model, d_out=d_model, bias=False, device=device, dtype=dtype)
        self.v_proj = Linear(d_in=d_model, d_out=d_model, bias=False, device=device, dtype=dtype)
        self.output_proj = Linear(d_in=d_model, d_out=d_model, bias=False, device=device, dtype=dtype)

    def _split_heads(self, x: Tensor) -> Tensor:
        return x.view(*x.shape[:-1], self.num_heads, self.d_head).transpose(-3, -2)

    def forward(self, in_features: Tensor, token_positions: Tensor | None = None) -> Tensor:
        seq_len = in_features.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len}).")

        q = self._split_heads(self.q_proj(in_features))
        k = self._split_heads(self.k_proj(in_features))
        v = self._split_heads(self.v_proj(in_features))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=in_features.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool))
        out = scaled_dot_product_attention(q, k, v, causal_mask)
        out = out.transpose(-3, -2).contiguous().view(*in_features.shape[:-1], self.d_model)
        return self.output_proj(out)


def run_scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    return scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)


def run_rope(*, d_k: int, theta: float, max_seq_len: int, in_query_or_key: Tensor, token_positions: Tensor) -> Tensor:
    rope = RotaryPositionalEmbedding(d_k=d_k, theta=theta, max_seq_len=max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_multihead_self_attention_from_weights(
    *,
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    attn = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=in_features.shape[-2],
        rope=None,
        device=q_proj_weight.device,
        dtype=q_proj_weight.dtype,
    )
    with torch.no_grad():
        attn.q_proj.weight.copy_(q_proj_weight)
        attn.k_proj.weight.copy_(k_proj_weight)
        attn.v_proj.weight.copy_(v_proj_weight)
        attn.output_proj.weight.copy_(o_proj_weight)
    return attn(in_features)


def run_multihead_self_attention_with_rope_from_weights(
    *,
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor | None = None,
) -> Tensor:
    rope = RotaryPositionalEmbedding(d_k=d_model // num_heads, theta=theta, max_seq_len=max_seq_len)
    attn = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        rope=rope,
        device=q_proj_weight.device,
        dtype=q_proj_weight.dtype,
    )
    with torch.no_grad():
        attn.q_proj.weight.copy_(q_proj_weight)
        attn.k_proj.weight.copy_(k_proj_weight)
        attn.v_proj.weight.copy_(v_proj_weight)
        attn.output_proj.weight.copy_(o_proj_weight)
    return attn(in_features, token_positions=token_positions)
