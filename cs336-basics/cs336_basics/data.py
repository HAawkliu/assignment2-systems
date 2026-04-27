from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a language-modeling batch from a 1D token-id array.

    Returns:
        x, y where both have shape (batch_size, context_length), and
        y is x shifted by one token to the right.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}.")
    if dataset.ndim != 1:
        raise ValueError(f"dataset must be 1D, got shape {dataset.shape}.")

    n_tokens = int(dataset.shape[0])
    max_start = n_tokens - context_length
    if max_start <= 0:
        raise ValueError(
            "dataset is too short for the requested context_length: "
            f"len(dataset)={n_tokens}, context_length={context_length}."
        )

    starts = np.random.randint(low=0, high=max_start, size=batch_size, dtype=np.int64)
    offsets = np.arange(context_length, dtype=np.int64)
    idx = starts[:, None] + offsets[None, :]

    x_np = np.asarray(dataset[idx], dtype=np.int64)
    y_np = np.asarray(dataset[idx + 1], dtype=np.int64)

    x = torch.from_numpy(x_np).to(device=device)
    y = torch.from_numpy(y_np).to(device=device)
    return x, y


def run_get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapter-facing wrapper for section 5.1."""
    return get_batch(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )
