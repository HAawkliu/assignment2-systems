from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor


def z_loss(
    inputs: Float[Tensor, " ... vocab_size"],
    *,
    reduction: str = "mean",
) -> Tensor:
    """Compute z-loss: square(logsumexp(logits)) over the vocab dimension.

    This is commonly used as a logit-scale regularizer in language-model training.
    """
    if inputs.ndim < 1:
        raise ValueError("inputs must have shape (..., vocab_size).")
    if inputs.shape[-1] <= 0:
        raise ValueError("inputs must have non-empty vocabulary dimension.")

    log_z = torch.logsumexp(inputs, dim=-1)
    z = log_z.square()

    if reduction == "mean":
        return z.mean()
    if reduction == "sum":
        return z.sum()
    if reduction == "none":
        return z
    raise ValueError(f"Unsupported reduction: {reduction}")


def cross_entropy_loss(
    inputs: Float[Tensor, " ... vocab_size"],
    targets: Int[Tensor, " ..."],
) -> Tensor:
    """Compute average cross-entropy loss over the batch.

    Assignment requirements:
    - Handle numerical stability (subtract max logit per example).
    - Cancel log/exp when possible (log-softmax style computation).
    - Return average loss across the batch dimension.
    """
    if inputs.ndim < 2:
        raise ValueError("inputs must have shape (..., vocab_size).")
    if targets.shape != inputs.shape[:-1]:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} must match inputs.shape[:-1] "
            f"{tuple(inputs.shape[:-1])}."
        )

    num_classes = inputs.shape[-1]
    if num_classes <= 0:
        raise ValueError("inputs must have non-empty vocabulary dimension.")

    targets = targets.long()
    if torch.any(targets < 0) or torch.any(targets >= num_classes):
        raise ValueError("targets contain out-of-range class indices.")

    # Stable log-softmax via max-subtraction then logsumexp.
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_denom = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_shifted_logits = torch.take_along_dim(shifted, targets.unsqueeze(-1), dim=-1).squeeze(-1)
    losses = log_denom - target_shifted_logits
    return losses.mean()


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"],
) -> Tensor:
    """Adapter-facing wrapper for section 4.1."""
    return cross_entropy_loss(inputs=inputs, targets=targets)


def cross_entropy_with_z_loss(
    inputs: Float[Tensor, " ... vocab_size"],
    targets: Int[Tensor, " ..."],
    *,
    z_loss_weight: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total loss, CE loss, and z-loss.

    total = cross_entropy + z_loss_weight * z_loss
    """
    ce = cross_entropy_loss(inputs=inputs, targets=targets)
    zl = z_loss(inputs, reduction="mean")
    total = ce + z_loss_weight * zl
    return total, ce, zl
