from __future__ import annotations

import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize model, optimizer, and iteration into a checkpoint."""
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(payload, out)


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_optimizer_state_tensors(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model + optimizer state from checkpoint and return saved iteration."""
    checkpoint = torch.load(src, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict-like object.")

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing key: model_state_dict")
    if "optimizer_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing key: optimizer_state_dict")
    if "iteration" not in checkpoint:
        raise KeyError("Checkpoint missing key: iteration")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state_tensors(optimizer, _infer_model_device(model))

    return int(checkpoint["iteration"])


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Adapter-facing wrapper for section 5.2."""
    save_checkpoint(model=model, optimizer=optimizer, iteration=iteration, out=out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Adapter-facing wrapper for section 5.2."""
    return load_checkpoint(src=src, model=model, optimizer=optimizer)
