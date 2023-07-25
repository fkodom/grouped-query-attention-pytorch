import os
from typing import List, Sequence

import plotly.graph_objects as go
import torch
from torch import Tensor
from transformers import T5ForConditionalGeneration, T5Tokenizer

from grouped_query_attention_pytorch.t5 import convert_t5_to_gqa
from grouped_query_attention_pytorch.utils.benchmark import BenchmarkResult, benchmark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Benchmarking parameters
MODEL_NAME = "t5-3b"
NUM_GROUPS = [1, 2, 4, 8, 16, 32]
TOTAL_TOKENS = 4096
MODEL_MAX_LENGTH = 512
SAVE_PATH = os.path.join("doc", "benchmark_t5.png")

BENCHMARK_SETTINGS_TEMPLATE = """
Benchmark settings:
    device: {device}
    dtype: {dtype}
    model_name: {model_name}
    total_tokens: {total_tokens}
    model_max_length: {model_max_length}
"""
# NOTE: Text sample taken from the CNN/Daily Mail training set
INPUT_TEXT = """LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains
access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but
he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in
"Harry Potter and the Order of the Phoenix" To the disappointment of gossip
columnists around the world, the young actor says he has no plans to fritter his
cash away on fast cars, drink and celebrity parties. "I don't plan to be one of
those people who, as soon as they turn 18, suddenly buy themselves a massive sports
car collection or something similar," he told an Australian interviewer earlier this
month. "I don't think I'll be particularly extravagant. "The things I like buying
are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe
will be able to gamble in a casino, buy a drink in a pub or see the horror film
"Hostel: Part II," currently six places below his number one movie on the UK box
office chart. Details of how he'll mark his landmark birthday are under wraps.
His agent and publicist had no comment on his plans. "I'll definitely have some
sort of party," he said in an interview. "Hopefully none of you will be reading
about it." Radcliffe's earnings from the first five Potter films have been held
in a trust fund which he has not been able to touch. Despite his growing fame and
riches, the actor says he is keeping his feet firmly on the ground. "People are
always looking to say 'kid star goes off the rails,'" he told reporters last month.
"But I try very hard not to go that way because it would be too easy for them." His
latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is
breaking records on both sides of the Atlantic and he will reprise the role in the
last two films. Watch I-Reporter give her review of Potter's latest » . There is life
beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about
author Rudyard Kipling and his son, due for release later this year. He will also
appear in "December Boys," an Australian film about four boys who escape an orphanage.
Earlier this year, he made his stage debut playing a tortured teenager in Peter
Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that
he's legally an adult: "I just think I'm going to be more sort of fair game," he told
Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material
may not be published, broadcast, rewritten, or redistributed."""
TARGET_TEXT = """Harry Potter star Daniel Radcliffe gets £20M fortune as he turns
18 Monday. Young actor says he has no plans to fritter his cash away. Radcliffe's
earnings from first five Potter films have been held in trust fund."""


@torch.no_grad()
def forward_fn(model: T5ForConditionalGeneration, input_ids: Tensor, labels: Tensor):
    return model(input_ids=input_ids, labels=labels).loss


def main(
    model_name: str = MODEL_NAME,
    num_groups: Sequence[int] = NUM_GROUPS,
    total_tokens: int = TOTAL_TOKENS,
    model_max_length: int = MODEL_MAX_LENGTH,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
    save_path: str = SAVE_PATH,
):
    print(f"Loading model and tokenizer for {model_name}...")
    t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        model_name
    ).to(device=device, dtype=dtype)
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

    batch_size = total_tokens // model_max_length
    input_ids = tokenizer(
        [INPUT_TEXT] * batch_size,
        return_tensors="pt",
        max_length=model_max_length,
        truncation=True,
    ).input_ids.to(device=device)
    labels = tokenizer(
        [TARGET_TEXT] * batch_size,
        return_tensors="pt",
        max_length=model_max_length,
        truncation=True,
    ).input_ids.to(device=device)

    mha_result = benchmark(forward_fn, t5, input_ids, labels)
    print(f"MHA: {mha_result}")
    del t5
    torch.cuda.empty_cache()

    grouped_times: List[BenchmarkResult] = []
    for g in num_groups:
        print("Reloading model and converting to GQA...")
        t5 = T5ForConditionalGeneration.from_pretrained(model_name).to(
            device=device, dtype=dtype
        )
        gqa = convert_t5_to_gqa(t5, kv_heads=g, inplace=True)
        grouped_result = benchmark(forward_fn, gqa, input_ids, labels)
        grouped_times.append(grouped_result)
        print(f"Grouped (g={g}): {grouped_result}")

        del t5, gqa
        torch.cuda.empty_cache()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=num_groups,
            y=[mha_result.mean] * len(num_groups),
            mode="lines",
            line={"dash": "dash"},
            name="MHA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=num_groups,
            y=[r.mean for r in grouped_times],
            mode="lines",
            name="GQA",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=num_groups,
            y=[grouped_times[0].mean] * len(num_groups),
            mode="lines",
            line={"dash": "dash"},
            name="MQA",
        )
    )
    fig.update_layout(
        title="T5 Benchmarks",
        xaxis_title="GQA Groups",
        yaxis_title="Time per sample (s)",
        # use log-scale for x-axis
        xaxis={"tickmode": "array", "tickvals": num_groups, "type": "log"},
        # place legend at center-left
        legend={"x": 0.1, "y": 0.5},
    )
    fig.write_image(save_path)


if __name__ == "__main__":
    import argparse

    # NOTE: The original paper uses T5 v1.1 XL and XXL models.  When I load those
    # models through 'transformers' without applying GQA, I get nonsense outputs.
    # TODO: Figure out why this is happening.
    #   tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large", legacy=False)
    #   model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large")
    #
    # In the meantime, we can use the non-Google T5 models, which seem to work fine.
    # NOTE: Since the the original number of heads (n_heads) must be divisible by
    # 'kv_heads', there are only certain values of 'kv_heads' that we can use.
    # The following values of 'kv_heads' should be valid:
    #   - t5-small: 1, 2, 4, 8
    #   - t5-base: 1, 2, 3, 4, 6, 12
    #   - t5-large: 1, 2, 4, 8, 16
    #   - t5-3b: 1, 2, 4, 8, 16, 32  (DEFAULT)
    #   - t5-11b: 1, 2, 4, 8, 16, 32, 64  TODO: Check 11b values specifically

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Name of the T5 model to benchmark (default: '{MODEL_NAME}')",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        nargs="+",
        default=NUM_GROUPS,
        help=f"Sequence of GQA group sizes for benchmarking (default: {NUM_GROUPS})",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=TOTAL_TOKENS,
        help=f"Total number of tokens in the batch (default: {TOTAL_TOKENS})",
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=MODEL_MAX_LENGTH,
        help=f"Sequence length of the input (default: {MODEL_MAX_LENGTH})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device to run the benchmark on (default: {DEVICE})",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DTYPE,
        help=f"Data type to run the benchmark on (default: {DTYPE})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=SAVE_PATH,
        help=f"Path to save the benchmark plot (default: '{SAVE_PATH}')",
    )
    args = parser.parse_args()

    print(
        BENCHMARK_SETTINGS_TEMPLATE.format(
            device=args.device,
            dtype=args.dtype,
            model_name=args.model_name,
            total_tokens=args.total_tokens,
            model_max_length=args.model_max_length,
        )
    )

    main(**vars(args))
