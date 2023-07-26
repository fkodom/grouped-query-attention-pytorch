import os
from typing import List, Sequence

import plotly.graph_objects as go
import torch
import xformers.ops as xops

from grouped_query_attention_pytorch.attention import (
    scaled_dot_product_gqa,
)
from grouped_query_attention_pytorch.utils.benchmark import BenchmarkResult, benchmark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Benchmarking parameters
NUM_GROUPS = [1, 4, 8, 16, 32, 64]
TOTAL_TOKENS = 8192
NUM_HEADS = 64
EMBED_DIM = 8
SEQ_LENGTH = 2048
SAVE_PATH = os.path.join("doc", "benchmark_attention.png")

BENCHMARK_SETTINGS_TEMPLATE = """
Benchmark settings:
    device: {device}
    dtype: {dtype}
    total_tokens: {total_tokens}
    seq_length: {seq_length}
    num_heads: {num_heads}
    embed_dim: {embed_dim}
"""


def main(
    num_groups: Sequence[int] = NUM_GROUPS,
    total_tokens: int = TOTAL_TOKENS,
    seq_length: int = SEQ_LENGTH,
    num_heads: int = NUM_HEADS,
    embed_dim: int = EMBED_DIM,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
    save_path: str = SAVE_PATH,
):
    batch_size = total_tokens // seq_length
    q = torch.randn(
        batch_size, seq_length, num_heads, embed_dim, device=device, dtype=dtype
    )
    kv = torch.randn(
        batch_size, seq_length, num_heads, embed_dim, device=device, dtype=dtype
    )

    _ = scaled_dot_product_gqa(q, kv, kv)
    vanilla_result = benchmark(scaled_dot_product_gqa, q, kv, kv)
    print(f"Vanilla: {vanilla_result}")
    xformers_result = benchmark(xops.memory_efficient_attention, q, kv, kv)
    print(f"Flash Attn: {xformers_result}")

    grouped_times: List[BenchmarkResult] = []
    for g in num_groups:
        kv = torch.randn(
            batch_size, seq_length, g, embed_dim, dtype=dtype, device=device
        )
        grouped_result = benchmark(
            scaled_dot_product_gqa, q, kv, kv, force_grouped=True
        )
        grouped_times.append(grouped_result)
        print(f"Grouped (g={g}): {grouped_result}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=num_groups,
            y=[vanilla_result.mean * 1000] * len(num_groups),
            mode="lines",
            line={"dash": "dash"},
            name="Vanilla MHA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=num_groups,
            y=[r.mean * 1000 for r in grouped_times],
            mode="lines",
            name="GQA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=num_groups,
            y=[grouped_times[0].mean * 1000] * len(num_groups),
            mode="lines",
            line={"dash": "dash"},
            name="MQA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=num_groups,
            y=[xformers_result.mean * 1000] * len(num_groups),
            mode="lines",
            line={"dash": "dash"},
            name="Flash Attn (v1 - xformers)",
        )
    )
    fig.update_layout(
        title="Attention Benchmarks",
        xaxis_title="GQA Groups",
        yaxis_title="Runtime (ms)",
        # use log-scale for x-axis
        xaxis={"tickmode": "array", "tickvals": num_groups, "type": "log"},
        # place legend at center-left
        legend={"x": 0.1, "y": 0.5},
    )
    fig.write_image(save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-groups",
        type=int,
        nargs="+",
        default=NUM_GROUPS,
        help="Sequence of GQA group sizes for benchmarking (default: [1, 4, 8, 16, 32, 64])",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=TOTAL_TOKENS,
        help="Total number of tokens in the batch (default: 8192)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=SEQ_LENGTH,
        help="Sequence length of the input (default: 2048)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=NUM_HEADS,
        help="Number of attention heads (default: 64)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=EMBED_DIM,
        help="Embedding dimension of the input (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Device to run the benchmark on (default: 'cuda' if available else 'cpu')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DTYPE,
        help="Data type to run the benchmark on (default: 'float16' if cuda is available else 'float32')",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=SAVE_PATH,
        help="Path to save the benchmark plot (default: 'doc/benchmark_attention.png')",
    )
    args = parser.parse_args()

    print(
        BENCHMARK_SETTINGS_TEMPLATE.format(
            device=args.device,
            dtype=args.dtype,
            total_tokens=args.total_tokens,
            seq_length=args.seq_length,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
        )
    )

    main(**vars(args))
