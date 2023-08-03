from math import log
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_activation_fn
from torchscale.component.xpos_relative_position import XPOS

from grouped_query_attention_pytorch.attention import MultiheadGQA


class GQATransformerEncoderLayer(nn.Module):
    # NOTE: Mostly pulled from 'nn.TransformerEncoderLayer', but with changes:
    #   - use sub-LayerNorm like in MAGNETO. See: https://arxiv.org/abs/2210.06423
    #   - use MultiheadGQA instead of MultiheadAttention

    def __init__(
        self,
        d_model: int,
        nhead: int,
        kv_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation
        self.gamma_init = gamma_init

        self.dropout = nn.Dropout(dropout)
        # Self-attention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = MultiheadGQA(  # type: ignore
            embed_dim=d_model,
            query_heads=nhead,
            kv_heads=kv_heads,
            dropout=dropout,
            layer_norm=True,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )
        # Feedforward block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(
            dim_feedforward, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # The 'MultiheadGQA' module uses ths same initialization,
        # so we just need to worry about the 'Linear' modules here.
        nn.init.xavier_normal_(self.linear1.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear2.bias, 0)

    def _self_attention_block(self, x: Tensor, is_causal: bool = False) -> Tensor:
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, is_causal=is_causal)
        x = self.dropout(x)
        return x

    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.norm2(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.norm3(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def forward(self, src: Tensor, is_causal: bool = False) -> Tensor:
        x = src
        x = x + self._self_attention_block(x, is_causal=is_causal)
        x = x + self._feedforward_block(x)
        return x


class GQATransformerDecoderLayer(nn.Module):
    # NOTE: Mostly pulled from 'nn.TransformerDecoderLayer', but with changes:
    #   - use sub-LayerNorm like in MAGNETO. See: https://arxiv.org/abs/2210.06423
    #   - use MultiheadGQA instead of MultiheadAttention

    def __init__(
        self,
        d_model: int,
        nhead: int,
        kv_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation
        self.gamma_init = gamma_init

        self.dropout = nn.Dropout(dropout)
        # Self-attention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = MultiheadGQA(  # type: ignore
            embed_dim=d_model,
            query_heads=nhead,
            kv_heads=kv_heads,
            dropout=dropout,
            layer_norm=False,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )
        # Multi-head attention block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.multihead_attn = MultiheadGQA(  # type: ignore
            embed_dim=d_model,
            query_heads=nhead,
            kv_heads=kv_heads,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )
        # Feedforward block
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.norm4 = nn.LayerNorm(
            dim_feedforward, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # The 'MultiheadGQA' module uses ths same initialization,
        # so we just need to worry about the 'Linear' modules here.
        nn.init.xavier_normal_(self.linear1.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight, gain=self.gamma_init)
        nn.init.constant_(self.linear2.bias, 0)

    def _self_attention_block(self, x: Tensor, is_causal: bool = False) -> Tensor:
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, is_causal=is_causal)
        x = self.dropout(x)
        return x

    def _multihead_attention_block(
        self, x: Tensor, memory: Tensor, is_causal: bool = False
    ) -> Tensor:
        x = self.norm2(x)
        x, _ = self.multihead_attn(x, memory, memory, is_causal=is_causal)
        x = self.dropout(x)
        return x

    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.norm3(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.norm4(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        x = x + self._self_attention_block(x, is_causal=tgt_is_causal)
        x = x + self._multihead_attention_block(x, memory, is_causal=memory_is_causal)
        x = x + self._feedforward_block(x)
        return x


class GQATransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        # The 'gamma_init' parameters are different for the encoder and decoder,
        # and depend on the number of encoder/decoder layers.  See MAGNETO paper:
        # https://arxiv.org/pdf/2210.06423.pdf, Figure 2
        encoder_gamma_init = (
            log(3 * num_decoder_layers) * log(2 * num_encoder_layers) / 3
        ) ** 0.5
        decoder_gamma_init = log(3 * num_decoder_layers) ** 0.5

        self.encoder = nn.TransformerEncoder(
            encoder_layer=GQATransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                kv_heads=kv_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                gamma_init=encoder_gamma_init,
                device=device,
                dtype=dtype,
            ),
            num_layers=num_encoder_layers,
            mask_check=False,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=GQATransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                kv_heads=kv_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                gamma_init=decoder_gamma_init,
                device=device,
                dtype=dtype,
            ),
            num_layers=num_decoder_layers,
        )

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        """
        Input shape: (batch_size, seq_len, d_model)
        Output shape: (batch_size, seq_len, d_model)

        NOTE: Assume that 'is_causal' applies to both the encoder and decoder.
        This is the case for language modeling, but maybe not for other tasks.
        """
        tgt = x
        for layer in self.encoder.layers:
            x = layer(x, is_causal=is_causal)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)

        mem = x
        for layer in self.decoder.layers:
            tgt = layer(tgt, mem, memory_is_causal=is_causal, tgt_is_causal=is_causal)
        if self.decoder.norm is not None:
            tgt = self.decoder.norm(tgt)

        return tgt


class GQATransformerLM(nn.Module):
    def __init__(
        self,
        num_tokens: int,  # (required) usually obtained from the tokenizer
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_tokens, d_model, device=device, dtype=dtype
        )
        # TODO: Add support for other positional encodings?  I use XPOS, which is the
        # "latest and greatest" at the time of writing.  In principle, we could swap
        # it out for any other encoding, and remove the 'torchscale' dependency for this
        # repo, which is only used for XPOS.
        self.pos_embedding = XPOS(d_model).to(device=device, dtype=dtype)
        self.transformer = GQATransformer(
            d_model=d_model,
            nhead=nhead,
            kv_heads=kv_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.norm = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.out = nn.Linear(d_model, num_tokens, device=device, dtype=dtype)

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        x = self.token_embedding(x)
        x = x + self.pos_embedding(x)
        x = self.transformer(x, is_causal=is_causal)
        x = self.norm(x)
        return self.out(x)


if __name__ == "__main__":
    num_tokens = 2048
    device = torch.device("cuda")
    dtype = torch.float16

    x = torch.randint(0, num_tokens - 1, size=(2, 512), device=device)
    model = GQATransformerLM(num_tokens=num_tokens, device=device, dtype=dtype)

    with torch.no_grad():
        out = model(x)
    print(out.shape)
