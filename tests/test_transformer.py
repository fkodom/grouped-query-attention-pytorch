from typing import Callable, Union

import pytest
import torch

from grouped_query_attention_pytorch.transformer import GQATransformer, GQATransformerLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SEQ_LEN = 16


@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("nhead", [4, 8])
@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("dim_feedforward", [64])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("is_causal", [True, False])
def test_gqa_transformer(
    d_model: int,
    nhead: int,
    kv_heads: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    activation: Union[str, Callable],
    is_causal: bool,
):
    net = GQATransformer(
        d_model=d_model,
        nhead=nhead,
        kv_heads=kv_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        device=DEVICE,
        dtype=DTYPE,
    )
    x = torch.randn(1, SEQ_LEN, d_model, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        y = net.forward(x, is_causal=is_causal)
    assert y.size(0) == 1
    assert y.size(1) == SEQ_LEN
    assert y.size(2) == d_model


@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("d_model", [128])
@pytest.mark.parametrize("nhead", [4, 8])
@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("dim_feedforward", [64])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("activation", ["relu", "gelu"])
@pytest.mark.parametrize("is_causal", [True, False])
def test_gqa_transformer_lm(
    num_tokens: int,
    d_model: int,
    nhead: int,
    kv_heads: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    activation: Union[str, Callable],
    is_causal: bool,
):
    net = GQATransformerLM(
        num_tokens=num_tokens,
        d_model=d_model,
        nhead=nhead,
        kv_heads=kv_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        device=DEVICE,
        dtype=DTYPE,
    )
    x = torch.randint(0, num_tokens, (1, SEQ_LEN), device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        y = net.forward(x, is_causal=is_causal)
    assert y.size(0) == 1
    assert y.size(1) == SEQ_LEN
    assert y.size(2) == num_tokens
