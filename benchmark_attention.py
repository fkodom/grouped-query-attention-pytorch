import os
from math import ceil
from timeit import Timer
from typing import Callable, List, NamedTuple

import plotly.graph_objects as go
import torch
import xformers.ops as xops

from grouped_query_attention_pytorch.attention import (
    grouped_scaled_dot_product_attention,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Benchmarking parameters
TOTAL_TOKENS = 8192
NUM_HEADS = 64
EMBED_DIM = 8
SEQ_LENGTH = 2048
NUM_GROUPS = [1, 4, 8, 16, 32, 64]

BENCHMARK_SETTINGS_TEMPLATE = """
Benchmark settings:
    device: {device}
    dtype: {dtype}
    total_tokens: {total_tokens}
    batch_size: {batch_size}
    seq_length: {seq_length}
    num_heads: {num_heads}
    embed_dim: {embed_dim}
"""


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
    min_iterations: int = 10,
    **kwargs,
) -> BenchmarkResult:
    # Benchmark the runtime of a function and dynamically determine the number of
    # iterations to run.  Continue running the function until *total* runtime
    # exceeds 'min_total_seconds' and 'min_iterations'.
    if min_iterations < 2:
        raise ValueError("min_iterations must be >= 2")

    timer = Timer(
        "fn(*args, **kwargs); synchronize()",
        globals={
            "fn": fn,
            "args": args,
            "kwargs": kwargs,
            "synchronize": torch.cuda.synchronize,
        },
    )
    # Run the function 5 times to warm up
    _ = timer.repeat(number=1, repeat=5)

    times: List[float] = []
    total_time = 0.0
    num_iterations = min_iterations

    while total_time < min_total_seconds:
        _times = timer.repeat(number=1, repeat=num_iterations)
        times.extend(_times)

        times_tensor = torch.as_tensor(times)
        total_time = times_tensor.sum().item()
        avg_time = times_tensor.mean().item()
        num_iterations = ceil((min_total_seconds - total_time) / avg_time)

    times_tensor = torch.as_tensor(times)
    return BenchmarkResult(
        mean=times_tensor.mean().item(),
        std=times_tensor.std().item(),
    )


if __name__ == "__main__":
    batch_size = TOTAL_TOKENS // SEQ_LENGTH
    print(
        BENCHMARK_SETTINGS_TEMPLATE.format(
            device=DEVICE,
            dtype=DTYPE,
            total_tokens=TOTAL_TOKENS,
            batch_size=batch_size,
            seq_length=SEQ_LENGTH,
            num_heads=NUM_HEADS,
            embed_dim=EMBED_DIM,
        )
    )

    q = torch.randn(
        batch_size, SEQ_LENGTH, NUM_HEADS, EMBED_DIM, device=DEVICE, dtype=DTYPE
    )
    kv = torch.randn(
        batch_size, SEQ_LENGTH, NUM_HEADS, EMBED_DIM, device=DEVICE, dtype=DTYPE
    )

    _ = grouped_scaled_dot_product_attention(q, kv, kv)
    vanilla_result = benchmark(grouped_scaled_dot_product_attention, q, kv, kv)
    print(f"Vanilla: {vanilla_result}")
    xformers_result = benchmark(xops.memory_efficient_attention, q, kv, kv)
    print(f"Efficient: {xformers_result}")

    grouped_times: List[BenchmarkResult] = []
    for g in NUM_GROUPS:
        kv = torch.randn(
            batch_size, SEQ_LENGTH, g, EMBED_DIM, dtype=DTYPE, device=DEVICE
        )
        grouped_result = benchmark(
            grouped_scaled_dot_product_attention, q, kv, kv, force_grouped=True
        )
        grouped_times.append(grouped_result)
        print(f"Grouped (g={g}): {grouped_result}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=NUM_GROUPS,
            y=[vanilla_result.mean * 1000] * len(NUM_GROUPS),
            mode="lines",
            line={"dash": "dash"},
            name="Vanilla",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=NUM_GROUPS,
            y=[grouped_times[0].mean * 1000] * len(NUM_GROUPS),
            mode="lines",
            line={"dash": "dash"},
            name="Multi-Query",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=NUM_GROUPS,
            y=[r.mean * 1000 for r in grouped_times],
            mode="lines",
            name="Grouped-Query",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=NUM_GROUPS,
            y=[xformers_result.mean * 1000] * len(NUM_GROUPS),
            mode="lines",
            line={"dash": "dash"},
            name="Mem-Efficient (xformers)",
        )
    )
    fig.update_layout(
        title="Attention Benchmarks",
        xaxis_title="Number of Groups",
        yaxis_title="Runtime (ms)",
        # use log-scale for x-axis
        xaxis={"tickmode": "array", "tickvals": NUM_GROUPS, "type": "log"},
    )
    fig.write_image(os.path.join("doc", "benchmark_attention.png"))
