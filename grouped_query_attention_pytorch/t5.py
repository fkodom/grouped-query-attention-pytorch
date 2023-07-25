from __future__ import annotations

from copy import deepcopy
from typing import TypeVar, overload

import torch
from einops import einsum, rearrange, repeat
from torch import nn
from transformers.models.t5.modeling_t5 import T5Attention


class T5GQA(nn.Module):
    def __init__(
        self,
        is_decoder: bool,
        d_model: int,
        key_value_proj_dim: int,
        n_heads: int,
        kv_heads: int,
        dropout: float,
        has_relative_attention_bias: bool,
        relative_attention_num_buckets: int,
        relative_attention_max_distance: int,
    ):
        super().__init__()
        if n_heads % kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by kv_heads ({kv_heads})"
            )

        self.is_decoder = is_decoder
        self.d_model = d_model
        self.key_value_proj_dim = key_value_proj_dim
        self.n_heads = n_heads
        # TODO: Check if we need to store 'kv_heads' and 'inner_dim' as a properties
        self.kv_heads = kv_heads
        self.dropout = dropout
        # NOTE: Relative attention bias typically only used in the first layer
        # of a `T5Stack` module.
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.kv_dim = self.kv_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        # self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # self.k = nn.Linear(self.d_model, self.kv_dim, bias=False)
        # self.v = nn.Linear(self.d_model, self.kv_dim, bias=False)
        # self.o = nn.Linear(self.kv_dim, self.d_model, bias=False)
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
        self.pruned_heads = set()  # type: ignore
        self.gradient_checkpointing = False

        self._relative_position_bucket = T5Attention._relative_position_bucket

    @classmethod
    def from_t5_attention(cls, t5: T5Attention, kv_heads: int) -> T5GQA:
        t5_gqa = T5GQA(
            is_decoder=t5.is_decoder,
            d_model=t5.d_model,
            key_value_proj_dim=t5.key_value_proj_dim,
            n_heads=t5.n_heads,
            kv_heads=kv_heads,
            dropout=t5.dropout,
            has_relative_attention_bias=t5.has_relative_attention_bias,
            relative_attention_num_buckets=t5.relative_attention_num_buckets,
            relative_attention_max_distance=t5.relative_attention_max_distance,
        )

        # Copy all of the weights verbatim from the original T5Attention module.
        # NOTE: In the T5 GQA implementation, all of the attention head aggregations
        # happen in the 'forward' method.  The weights themselves are not modified.
        t5_gqa.q.weight.data = t5.q.weight.data
        t5_gqa.k.weight.data = t5.k.weight.data
        t5_gqa.v.weight.data = t5.v.weight.data
        t5_gqa.o.weight.data = t5.o.weight.data
        if t5.has_relative_attention_bias:
            t5_gqa.relative_attention_bias.weight.data = (
                t5.relative_attention_bias.weight.data
            )

        return t5_gqa

    def forward(  # noqa: C901
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            # NOTE: Changed from the original definition in T5Attention.
            sequence_length = states.shape[1]
            return states.view(
                batch_size, sequence_length, -1, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            # NOTE: Changed from the original definition in T5Attention.
            sequence_length = states.shape[2]
            return (
                states.transpose(1, 2)
                .contiguous()
                .view(batch_size, sequence_length, -1)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states: (batch_size, n_heads, seq_length, dim_per_head)
        grouped_queries = shape(self.q(hidden_states))
        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # # compute scores
        # scores = torch.matmul(
        #     query_states, key_states.transpose(3, 2)
        # )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        grouped_queries = rearrange(
            grouped_queries, "b (g h) n d -> b g h n d", h=self.kv_heads
        )
        grouped_keys = rearrange(
            key_states, "b (g h) s d -> b g h s d", h=self.kv_heads
        ).mean(dim=1)
        scores = einsum(grouped_queries, grouped_keys, "b g h n d, b h s d -> b h n s")

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    # NOTE: This is different from the original in T5Attention!
                    # (1, self.n_heads, real_seq_length, key_length),
                    (1, self.kv_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = T5Attention.compute_bias(
                    self, real_seq_length, key_length, device=scores.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                # (batch_size, n_heads, seq_length, key_length)
                position_bias = position_bias + mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        # NOTE: This is different from the original in T5Attention!
        grouped_position_bias = rearrange(
            position_bias_masked, "b (g h) n s -> b g h n s", h=self.kv_heads
        ).mean(dim=1)

        scores += grouped_position_bias
        # attn_weights: (batch_size, kv_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # NOTE: This is different from the original in T5Attention!
        # attn_output = unshape(torch.matmul(attn_weights, value_states))
        grouped_values = rearrange(
            value_states, "b (g h) s d -> b g h s d", h=self.kv_heads
        ).mean(dim=1)
        attn_output = unshape(torch.matmul(attn_weights, grouped_values))
        attn_output = repeat(
            attn_output, "b s d -> b s (g d)", g=(self.n_heads // self.kv_heads)
        )
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)  # type: ignore
        return outputs


ModuleType = TypeVar("ModuleType", bound=nn.Module)


@overload
def convert_t5_to_gqa(
    module: ModuleType, kv_heads: int, inplace: bool = False
) -> ModuleType:
    ...


@overload
def convert_t5_to_gqa(
    module: T5Attention, kv_heads: int, inplace: bool = False
) -> T5GQA:
    ...


def convert_t5_to_gqa(module, kv_heads: int, inplace: bool = False):
    if isinstance(module, T5Attention):
        return T5GQA.from_t5_attention(module, kv_heads=kv_heads)

    out = module if inplace else deepcopy(module)
    for name, child in out.named_children():
        out._modules[name] = convert_t5_to_gqa(child, kv_heads=kv_heads, inplace=True)
    return out


if __name__ == "__main__":
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    # NOTE: The original paper uses T5 v1.1 XL and XXL models.  When I load those
    # models through 'transformers' without applying GQA, I get nonsense outputs.
    # TODO: Figure out why this is happening.
    #   tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large", legacy=False)
    #   model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-large")
    #
    # In the meantime, we can use the non-Google T5 models, which seem to work fine.
    # NOTE: Since the the original number of heads (n_heads) must be divisible by
    # 'kv_heads', there are only certain values of 'kv_heads' that we can use.
    # To the best of my knowledge, the following values of 'kv_heads' are valid:
    #   - t5-small: 1, 2, 4, 8
    #   - t5-base: 1, 2, 3, 4, 6, 12
    #   - t5-large: 1, 2, 4, 8, 16
    #   - t5-3b: 1, 2, 4, 8, 16, 32
    #   - t5-11b: 1, 2, 4, 8, 16, 32, 64  TODO: Check 11b values specifically

    tokenizer = T5Tokenizer.from_pretrained(
        "t5-base", legacy=False, model_max_length=512
    )
    t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        "t5-base"
    )
    gqa = convert_t5_to_gqa(t5, kv_heads=6)

    input_ids = tokenizer(
        "translate English to German: The house is wonderful.", return_tensors="pt"
    ).input_ids
    y2 = gqa.generate(input_ids, max_new_tokens=25)
    text = tokenizer.batch_decode(y2[0], skip_special_tokens=True)
    print(text)
    # The correct answer is:  ['<pad>', 'Das', 'Haus', 'ist', 'wunderbar', '.', '</s>']
    # NOTE: The original T5 model produces this answer, and so does GQA when we use
    # the maximum number of heads -- effectively equivalent to the original T5 model
    # with MHA.  The text quickly degrades as we reduce the number of heads.

    labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids
    loss = gqa(input_ids=input_ids, labels=labels).loss
    print(f"Loss: {loss}")
    # NOTE: As above, the loss quickly degrades (increases) as we reduce the number
    # of GQA heads.
