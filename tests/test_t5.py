import pytest
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerFF

from grouped_query_attention_pytorch.t5 import T5GQA, convert_t5_to_gqa

MODEL_NAME = "t5-small"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
SEQ_LEN = 16


# Test all of the valid kv_heads values for 't5-small
@pytest.mark.parametrize("kv_heads", [1, 2, 4, 8])
def test_convert_t5_to_gqa(kv_heads: int):
    t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME
    ).to(device=DEVICE, dtype=DTYPE)

    # Just to establish that the T5 model is as expected.  Check that all of the
    # known attention layers are of type 'T5Attention'.
    for block in t5.encoder.block:
        for layer in block.layer:
            if hasattr(layer, "SelfAttention"):
                assert isinstance(layer.SelfAttention, T5Attention)
            else:
                assert isinstance(layer, T5LayerFF)
    for block in t5.decoder.block:
        for layer in block.layer:
            if hasattr(layer, "SelfAttention"):
                assert isinstance(layer.SelfAttention, T5Attention)
            elif hasattr(layer, "EncDecAttention"):
                assert isinstance(layer.EncDecAttention, T5Attention)
            else:
                assert isinstance(layer, T5LayerFF)

    gqa = convert_t5_to_gqa(t5, kv_heads=kv_heads, inplace=True)
    # After conversion, check that all of the attention layers above have been
    # replaced with 'T5GQA' layers.
    for block in gqa.encoder.block:
        for layer in block.layer:
            if hasattr(layer, "SelfAttention"):
                assert isinstance(layer.SelfAttention, T5GQA)
            else:
                assert isinstance(layer, T5LayerFF)
    for block in t5.decoder.block:
        for layer in block.layer:
            if hasattr(layer, "SelfAttention"):
                assert isinstance(layer.SelfAttention, T5GQA)
            elif hasattr(layer, "EncDecAttention"):
                assert isinstance(layer.EncDecAttention, T5GQA)
            else:
                assert isinstance(layer, T5LayerFF)

    # Check that we can pass inputs/targets through the modified model, and that
    # the is returns a loss that is a scalar Tensor.
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    inputs = tokenizer(
        "translate English to German: The house is wonderful.", return_tensors="pt"
    )
    targets = tokenizer("Das Haus ist wunderbar.", return_tensors="pt")
    with torch.no_grad():
        loss = gqa(
            inputs.input_ids.to(DEVICE),
            labels=targets.input_ids.to(DEVICE),
        ).loss
    assert isinstance(loss, torch.Tensor)
    assert loss.size() == ()

    # Check that we can generate outputs, using the usual 'generate' method.
    out = gqa.generate(inputs.input_ids.to(DEVICE), max_new_tokens=10)
    assert isinstance(out, torch.Tensor)
