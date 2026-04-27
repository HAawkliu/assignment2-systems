from __future__ import annotations

import torch
from torch import Tensor, nn

from .activations import SwiGLU
from .attention import CausalMultiHeadSelfAttention, RotaryPositionalEmbedding
from .layers import Embedding, Linear
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        rope = RotaryPositionalEmbedding(d_k=d_model // num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope=rope,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, in_features: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = in_features + self.attn(self.ln1(in_features), token_positions=token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size=vocab_size, d_model=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_in=d_model, d_out=vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        seq_len = in_indices.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds context_length ({self.context_length}).")
        token_positions = torch.arange(seq_len, device=in_indices.device)
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        return self.lm_head(self.ln_final(x))


def run_transformer_block_from_weights(
    *,
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    block.load_state_dict(weights, strict=True)
    return block(in_features)


def run_transformer_lm_from_weights(
    *,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=in_indices.device,
        dtype=weights["token_embeddings.weight"].dtype,
    )
    model.load_state_dict(weights, strict=True)
    return model(in_indices)
