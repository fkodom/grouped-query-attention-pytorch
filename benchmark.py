import logging
from math import ceil
from timeit import Timer
from typing import Callable, List, NamedTuple

import torch
import xformers.ops as xops

from grouped_query_attention_pytorch.attention import scaled_dot_product_attention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Benchmarking parameters
TOTAL_TOKENS = 2**18  # 256k
NUM_HEADS = 12
EMBED_DIM = 8
GROUPS = [1, 2, 3, 4, 6, 12]
SEQ_LENGTHS = [2**i for i in range(10, 16)]  # 1k - 32k


class BenchmarkResult(NamedTuple):
    mean: float
    std: float

    def __repr__(self):
        return f"BenchmarkResult(mean: {self.mean:.3e}, std: {self.std:.3e})"

    def __str__(self):
        return f"({self.mean:.3e} \u00B1 {self.std:.3e}) s"


@torch.no_grad()
def benchmark(
    fn: Callable,
    *args,
    min_total_seconds: float = 1.0,
    min_iterations: int = 2,
    **kwargs,
) -> BenchmarkResult:
    # Benchmark the runtime of a function and dynamically determine the number of
    # iterations to run.  Continue running the function until *total* runtime
    # exceeds 'min_total_seconds' and 'min_iterations'.
    if min_iterations < 2:
        raise ValueError("min_iterations must be >= 2")

    timer = Timer(
        "fn(*args, **kwargs)",
        globals={"fn": fn, "args": args, "kwargs": kwargs},
    )
    # Run the function once to warm up
    _ = timer.repeat(number=1, repeat=1)

    times: List[float] = []
    total_time = 0.0
    num_iterations = min_iterations or 1

    while total_time < min_total_seconds:
        _times = timer.repeat(number=1, repeat=num_iterations)
        times.extend(_times)
        _total_time = sum(_times)
        total_time += _total_time

        # Estimate how many more iterations we need to run to get to 1 second
        avg_time = _total_time / num_iterations
        num_iterations = ceil((min_total_seconds - total_time) / avg_time)

    times_tensor = torch.as_tensor(times)
    return BenchmarkResult(
        mean=times_tensor.mean().item(),
        std=times_tensor.std().item(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(
        f"""Benchmark settings:
    device: {DEVICE}
    dtype: {DTYPE}
    total_tokens: {TOTAL_TOKENS}
    num_heads: {NUM_HEADS}
    embed_dim: {EMBED_DIM}
"""
    )

    for seq_length in SEQ_LENGTHS:
        logging.info(f"--- seq_length={seq_length} ---")
        batch_size = TOTAL_TOKENS // seq_length
        q = torch.randn(
            batch_size, seq_length, NUM_HEADS, EMBED_DIM, device=DEVICE, dtype=DTYPE
        )
        kv = torch.randn(
            batch_size, seq_length, NUM_HEADS, EMBED_DIM, device=DEVICE, dtype=DTYPE
        )

        result = benchmark(scaled_dot_product_attention, q, kv, kv)
        logging.info(f"Vanilla: {result}")
        result = benchmark(xops.memory_efficient_attention, q, kv, kv)
        logging.info(f"Efficient: {result}")
        for g in GROUPS:
            kv = torch.randn(
                batch_size, seq_length, g, EMBED_DIM, dtype=DTYPE, device=DEVICE
            )
            result = benchmark(
                scaled_dot_product_attention, q, kv, kv, force_grouped=True
            )
            logging.info(f"Grouped (g={g}): {result}")
