import pytest
import torch

from grouped_query_attention_pytorch.attention import (
    MultiheadGQA,
    scaled_dot_product_gqa,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SEQ_LEN = 16


@pytest.mark.parametrize("embed_dim", [32])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("kv_heads", [4, 8])
@pytest.mark.parametrize("is_causal", [True, False])
def test_grouped_scaled_dot_product_attention(
    embed_dim: int,
    num_heads: int,
    kv_heads: int,
    is_causal: bool,
):
    x = torch.randn(1, SEQ_LEN, num_heads, embed_dim, device=DEVICE, dtype=DTYPE)
    kv = torch.randn(1, SEQ_LEN, kv_heads, embed_dim, device=DEVICE, dtype=DTYPE)

    if kv_heads > num_heads:
        with pytest.raises(ValueError):
            scaled_dot_product_gqa(x, kv, kv, is_causal=is_causal)
        return

    out, attn_weights = scaled_dot_product_gqa(
        x, kv, kv, is_causal=is_causal, need_weights=True
    )
    assert out.size(0) == 1
    assert out.size(1) == SEQ_LEN
    assert out.size(2) == kv_heads
    assert out.size(3) == embed_dim
    assert attn_weights.size(0) == 1
    assert attn_weights.size(1) == SEQ_LEN
    assert attn_weights.size(2) == SEQ_LEN
    assert attn_weights.size(3) == kv_heads


@torch.no_grad()
@pytest.mark.parametrize("embed_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("kv_heads", [4, 8])
@pytest.mark.parametrize("is_causal", [True, False])
def test_multihead_gqa(
    embed_dim: int,
    num_heads: int,
    kv_heads: int,
    is_causal: bool,
):
    if kv_heads > num_heads:
        with pytest.raises(ValueError):
            MultiheadGQA(embed_dim, num_heads, kv_heads)
        return

    mhda = MultiheadGQA(embed_dim, num_heads, kv_heads, device=DEVICE, dtype=DTYPE)
    x = torch.randn(1, SEQ_LEN, embed_dim, device=DEVICE, dtype=DTYPE)

    out, _ = mhda(x, x, x, is_causal=is_causal)  # default: causal=False
    assert out.size(0) == 1
    assert out.size(1) == SEQ_LEN
    assert out.size(2) == embed_dim
