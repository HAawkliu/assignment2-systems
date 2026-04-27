from __future__ import annotations

import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine LR schedule with linear warmup.

    Piecewise definition:
    - Warmup:        t < T_w
    - Cosine decay:  T_w <= t <= T_c
    - Floor:         t > T_c
    """
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}.")
    if cosine_cycle_iters < warmup_iters:
        raise ValueError(
            "cosine_cycle_iters must be >= warmup_iters, "
            f"got cosine_cycle_iters={cosine_cycle_iters}, warmup_iters={warmup_iters}."
        )

    # Warm-up: alpha_t = (t / T_w) * alpha_max for t < T_w.
    if warmup_iters > 0 and it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    # Cosine annealing: T_w <= t <= T_c.
    if it <= cosine_cycle_iters:
        if cosine_cycle_iters == warmup_iters:
            return min_learning_rate
        phase = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine = 0.5 * (1.0 + math.cos(math.pi * phase))
        return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)

    # Post-annealing floor.
    return min_learning_rate


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Adapter-facing wrapper for section 4.4."""
    return get_lr_cosine_schedule(
        it=it,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )
