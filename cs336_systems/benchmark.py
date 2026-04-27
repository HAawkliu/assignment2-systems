from __future__ import annotations

import argparse
import contextlib
import statistics
import timeit
from dataclasses import dataclass
from collections.abc import Iterator

import torch
import torch.nn.functional as F

from cs336_basics.model import TransformerLM
from cs336_basics.training import AdamW


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end CUDA benchmark for the CS336 basics Transformer.")
    parser.add_argument("--model-size", choices=[*MODEL_CONFIGS.keys(), "all"], default="small")
    parser.add_argument("--mode", choices=["forward", "backward", "full", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="float32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def nvtx_range(name: str, device: torch.device) -> Iterator[None]:
    if device.type != "cuda":
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def format_gib(num_bytes: int | float) -> str:
    return f"{num_bytes / 1024**3:.2f}"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def build_model(
    config: ModelConfig,
    *,
    vocab_size: int,
    context_length: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformerLM:
    return TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype,
    )


def benchmark_mode(
    *,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer | None,
    x: torch.Tensor,
    y: torch.Tensor,
    vocab_size: int,
    mode: str,
    warmup: int,
    steps: int,
    device: torch.device,
) -> list[float]:
    if mode in {"backward", "full"} and optimizer is None:
        raise ValueError(f"mode={mode!r} requires an optimizer.")

    def run_step() -> None:
        if mode == "forward":
            with nvtx_range("forward", device):
                _ = model(x)
            return

        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        with nvtx_range("forward", device):
            logits = model(x)
        with nvtx_range("loss", device):
            loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), y.reshape(-1))
        with nvtx_range("backward", device):
            loss.backward()
        if mode == "full":
            with nvtx_range("optimizer_step", device):
                optimizer.step()

    for _ in range(warmup):
        with nvtx_range("warmup", device):
            run_step()
            synchronize(device)

    timings: list[float] = []
    synchronize(device)
    with nvtx_range("benchmark", device):
        for step in range(steps):
            synchronize(device)
            start = timeit.default_timer()
            with nvtx_range(f"step_{step}", device):
                run_step()
            synchronize(device)
            timings.append(timeit.default_timer() - start)
        synchronize(device)
    return timings


def run_one(
    *,
    model_size: str,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, str | int | float]:
    config = MODEL_CONFIGS[model_size]
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(
        config,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        device=device,
        dtype=dtype,
    )
    model.train()

    optimizer = None
    if mode in {"backward", "full"}:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    x = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )
    y = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )

    timings = benchmark_mode(
        model=model,
        optimizer=optimizer,
        x=x,
        y=y,
        vocab_size=args.vocab_size,
        mode=mode,
        warmup=args.warmup,
        steps=args.steps,
        device=device,
    )

    param_count = count_parameters(model)
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "model_size": model_size,
        "mode": mode,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "params": param_count,
        "mean_ms": statistics.fmean(timings) * 1000,
        "std_ms": statistics.stdev(timings) * 1000 if len(timings) > 1 else 0.0,
        "peak_memory_gib": format_gib(peak_memory),
    }


def print_results(rows: list[dict[str, str | int | float]]) -> None:
    headers = [
        "model_size",
        "mode",
        "dtype",
        "batch_size",
        "context_length",
        "params",
        "mean_ms",
        "std_ms",
        "peak_memory_gib",
    ]
    print(",".join(headers))
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        print(",".join(values))


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    dtype = DTYPES[args.dtype]
    model_sizes = list(MODEL_CONFIGS) if args.model_size == "all" else [args.model_size]
    modes = ["forward", "backward", "full"] if args.mode == "all" else [args.mode]

    rows = [
        run_one(model_size=model_size, mode=mode, args=args, device=device, dtype=dtype)
        for model_size in model_sizes
        for mode in modes
    ]
    print_results(rows)


if __name__ == "__main__":
    main()
