from typing import Optional

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    mask: Optional[Tensor] = None,
    batch_first: bool = False,
    force_grouped: bool = False,
):
    """Compute scaled dot product attention.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        scale: Scale factor for attention.
        mask: Mask tensor.
        is_causal: Whether the attention is causal.

    NOTE: The 'mask' and 'is_causal' arguments cannot be used together.  If 'is_causal'
    is provided, we assume either a lower triangular mask (is_causal=True) or no
    mask at all (is_causal=False).

    Returns:
        The attention tensor.
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

    if not batch_first:
        query = rearrange(query, "n b h d -> b n h d")
        key = rearrange(key, "n b h d -> b n h d")
        value = rearrange(value, "n b h d -> b n h d")

    bq, _, hq, dq = query.shape
    bk, nk, hk, dk = key.shape
    bv, nv, hv, dv = value.shape
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

    num_groups = hq // hk
    if num_groups > 1 or force_grouped:
        query = rearrange(query, "b n (g h) d -> b n g h d", g=num_groups)
        similarity = einsum(query, key, "b n g h d, b s h d -> b n s h")
    else:
        similarity = einsum(query, key, "b n h d, b s h d -> b n s h")

    if mask is not None:
        if mask.ndim == 2:
            mask = rearrange(mask, "b n -> b n () ()")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b n s ()")
        similarity.masked_fill_(~mask, float("-inf"))

    attention = F.softmax(similarity / scale, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    out = einsum(attention, value, "b n s h, b s h d -> b n h d")
    if not batch_first:
        out = rearrange(out, "b n h d -> n b h d")

    return out


if __name__ == "__main__":
    q = torch.randn(2, 128, 8, 16)
    k = torch.randn(2, 128, 2, 16)
    v = torch.randn(2, 128, 2, 16)

    out = scaled_dot_product_attention(q, k, v)
    print(out.shape)
