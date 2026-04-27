import importlib.metadata

from .model.activations import SwiGLU, run_silu, run_swiglu_from_weights, silu
from .model.attention import (
    CausalMultiHeadSelfAttention,
    RotaryPositionalEmbedding,
    run_multihead_self_attention_from_weights,
    run_multihead_self_attention_with_rope_from_weights,
    run_rope,
    run_scaled_dot_product_attention,
    scaled_dot_product_attention,
)
from .training.checkpointing import load_checkpoint, run_load_checkpoint, run_save_checkpoint, save_checkpoint
from .data import get_batch, run_get_batch
from .training.grad_utils import clip_gradients, run_gradient_clipping
from .model.layers import Embedding, Linear, run_embedding_from_weights, run_linear_from_weights
from .training.losses import cross_entropy_loss, cross_entropy_with_z_loss, run_cross_entropy, z_loss
from .model.normalization import RMSNorm, run_rmsnorm_from_weights
from .training.optimizers import AdamW
from .training.schedules import get_lr_cosine_schedule, run_get_lr_cosine_schedule
from .tokenization import Tokenizer, train_bpe
from .model.transformer import TransformerBlock, TransformerLM, run_transformer_block_from_weights, run_transformer_lm_from_weights

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = [
    "AdamW",
    "CausalMultiHeadSelfAttention",
    "Embedding",
    "Linear",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "Tokenizer",
    "TransformerBlock",
    "TransformerLM",
    "clip_gradients",
    "cross_entropy_loss",
    "cross_entropy_with_z_loss",
    "get_batch",
    "get_lr_cosine_schedule",
    "load_checkpoint",
    "run_cross_entropy",
    "run_embedding_from_weights",
    "run_get_batch",
    "run_get_lr_cosine_schedule",
    "run_gradient_clipping",
    "run_linear_from_weights",
    "run_load_checkpoint",
    "run_multihead_self_attention_from_weights",
    "run_multihead_self_attention_with_rope_from_weights",
    "run_rmsnorm_from_weights",
    "run_rope",
    "run_save_checkpoint",
    "run_scaled_dot_product_attention",
    "run_silu",
    "run_swiglu_from_weights",
    "run_transformer_block_from_weights",
    "run_transformer_lm_from_weights",
    "save_checkpoint",
    "scaled_dot_product_attention",
    "silu",
    "train_bpe",
    "z_loss",
]
