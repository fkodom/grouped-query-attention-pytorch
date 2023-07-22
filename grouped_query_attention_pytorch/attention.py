from typing import Optional

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor

# TODO: GroupedQueryAttention class


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    mask: Optional[Tensor] = None,
    force_grouped: bool = False,
):
    """Scaled dot product attention with support for grouped queries.

    Notation:
    - b: batch size
    - n / s: sequence length
    - h: number of heads
    - d: dimension of query/key/value

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        value: Value tensor of shape (b, s, h, d)
        dropout: Dropout probability (default: 0.0)
        scale: Scale factor for query (default: d_query ** 0.5)
        mask: Mask tensor of shape (b, n, s) or (b, s). If 'ndim == 2', the mask is
            applied to all 'n' rows of the attention matrix. (default: None)
        force_grouped: If True, apply grouped-query attention even if the number of
            heads is equal for query, key, and value. (default: False)

    Returns:
        Tensor of shape (b, n, s, d)
    """
    # einstein notation:
    # - b: batch size
    # - n / s: sequence length
    # - h: number of heads
    # - g: number of groups
    # - d: dimension of query/key/value

    if not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(
            f"Expected query, key, and value to be 4-dimensional, but got shapes "
            f"{query.shape}, {key.shape}, and {value.shape}."
        )

    # Move sequence length dimension to axis 2
    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b n h d -> b h n d")
    value = rearrange(value, "b n h d -> b h n d")

    bq, hq, _, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError(
            "Expected query, key, and value to have the same batch size (dim=0) and "
            f"embedding dimension (dim=3), but got query: {query.shape}, "
            f"key: {key.shape}, and value: {value.shape}."
        )
    elif (hk != hv) or (nk != nv):
        raise ValueError(
            "Expected key and value to have the same size in dimensions 1 and 2, but "
            f"got key: {key.shape} and value: {value.shape}."
        )
    elif hq % hk != 0:
        raise ValueError(
            "Expected query heads to be a multiple of key/value heads, but got "
            f"query: {query.shape} and key/value: {key.shape}."
        )

    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale

    heads_per_group = hq // hk
    if heads_per_group > 1 or force_grouped:
        query = rearrange(query, "b (g h) n d -> b g h n d", g=heads_per_group)
        similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
    else:
        similarity = einsum(query, key, "b h n d, b h s d -> b h n s")

    if mask is not None:
        if mask.ndim == 2:
            mask = rearrange(mask, "b s -> b () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () n s")
        similarity.masked_fill_(~mask, float("-inf"))

    attention = F.softmax(similarity / scale, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    out = einsum(attention, value, "b h n s, b h s d -> b h n d")
    # Move head dimension back to axis 2
    out = rearrange(out, "b h n d -> b n h d")

    return out


if __name__ == "__main__":
    q = torch.randn(2, 128, 8, 16)
    k = torch.randn(2, 128, 2, 16)
    v = torch.randn(2, 128, 2, 16)

    out = scaled_dot_product_attention(q, k, v)
    print(out.shape)
