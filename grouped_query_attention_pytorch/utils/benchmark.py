from math import ceil
from timeit import Timer
from typing import Callable, List, NamedTuple

import torch


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
