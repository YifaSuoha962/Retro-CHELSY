import torch
import torch.nn as nn
from typing import Dict, List

from .module_utils import RelTransformerBase, RelMultiHeadAttention, GLU, RMSNorm, PosWiseFFN


class RelTransformer_Encoder_Block(nn.Module):
    def __init__(
        self, d_model, d_ff, h, u0, v0, u1, v1,
        sin_enc=None,
        pe='abs',
        rel_dict0={},
        rel_dict1={},
        dropout=0.1,
        block_id=0,
        alpha=None,
        init_beta=None,
        dist_range=[-1, -1],
        **kwargs
    ):
        super(RelTransformer_Encoder_Block, self).__init__(**kwargs)
        self.block_id = block_id
        self.alpha = alpha
        self.init_beta = init_beta
        self._build_model(
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            u0=u0,
            v0=v0,
            u1=u1,
            v1=v1,
            sin_enc=sin_enc,
            pe=pe,
            rel_dict0=rel_dict0,
            rel_dict1=rel_dict1,
            dropout=dropout,
            dist_range=dist_range
        )

    def _build_model(
        self, d_model, d_ff, h, u0, v0, u1, v1,
        sin_enc=None,
        pe='abs',
        rel_dict0={},
        rel_dict1={},
        dropout=0.1,
        dist_range=-1
    ):
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.self_attention = RelMultiHeadAttention(
            d_model=d_model,
            h=h,
            u=u0,
            v=v0,
            sin_enc=sin_enc,
            pe=pe,
            rel_dict=rel_dict0,
            dropout=dropout,
            dist_range=dist_range
        )
        self.ffn = GLU(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        if self.init_beta is not None:
            self.self_attention._init_param(vo_beta=self.init_beta)
            if self.ffn.__class__.__name__ == 'GLU':
                self.ffn._init_param(vo_beta=1.)
            elif self.ffn.__class__.__name__ == 'PosWiseFFN':
                self.ffn._init_param(vo_beta=self.init_beta)

    def forward(
        self,
        x: torch.Tensor,
        src_len: List[torch.Tensor],
        dist=None,  # for graph distance encoding
        deg=None,  # for graph deg encoding
        edge=None  # for graph edge bias
    ):
        x_in, _ = self.self_attention(
            q=x,
            k=x,
            v=x,
            length=src_len,
            dist=dist,
            deg=deg,
            edge=edge
        )
        x = self.dropout1(x_in) + x * self.alpha
        x = self.norm1(x)
        x = self.dropout2(self.ffn(x)) + x * self.alpha
        x = self.norm2(x)
        return x


class RelTransformer_Encoder(RelTransformerBase):
    def __init__(
        self, d_model, d_ff, h, layer,
        pe='abs',
        sin_enc=None,
        attn0_rel_dict={},
        attn1_rel_dict={},
        rel_apply='add',
        dropout=0.1,
        alpha=None,
        init_beta=None,
        dist_range=None,
        **kwargs
    ):
        super(RelTransformer_Encoder, self).__init__(
            d_rel0=d_model,
            d_rel1=d_model,
            attn0_rel_dict=attn0_rel_dict,
            attn1_rel_dict=attn1_rel_dict,
            rel_apply=rel_apply,
            **kwargs
        )
        self._build_model(
            d_model=d_model,
            d_ff=d_ff,
            h=h,
            layer=layer,
            pe=pe,
            sin_enc=sin_enc,
            attn0_rel_dict=attn0_rel_dict,
            attn1_rel_dict=attn1_rel_dict,
            dropout=dropout,
            alpha=alpha,
            init_beta=init_beta,
            dist_range=dist_range
        )

    def _build_model(
        self, d_model, d_ff, h, layer,
        pe='abs',
        sin_enc=None,
        attn0_rel_dict={},
        attn1_rel_dict={},
        dropout=0.1,
        alpha=None,
        init_beta=None,
        dist_range=None,
    ):
        if dist_range is None:
            dist_range = [-1 for _ in range(layer)]
        self.block = nn.Sequential()
        for i in range(layer):
            self.block.add_module(
                'block' + str(i),
                RelTransformer_Encoder_Block(
                    d_model=d_model,
                    d_ff=d_ff,
                    h=h,
                    u0=self.attn0_u,
                    v0=self.attn0_v,
                    u1=self.attn1_u,
                    v1=self.attn1_v,
                    sin_enc=sin_enc,
                    pe=pe,
                    rel_dict0=attn0_rel_dict,
                    rel_dict1=attn1_rel_dict,
                    dropout=dropout,
                    block_id=i,
                    alpha=alpha,
                    init_beta=init_beta,
                    dist_range=dist_range[i]
                )
            )
        self._attention_score = [None] * len(self.block)

    def forward(
        self,
        # size(batch, seq_len, d_model), need to be scaled before forward
        src: torch.Tensor,
        src_len: List[torch.Tensor],
        dist=None,  # for graph distance encoding
        deg=None,  # for graph deg encoding
        edge=None,  # for graph edge bias
    ):
        if edge is not None:
            for i, block in enumerate(self.block):
                src = block(
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge[i]
                )
        else:
            for i, block in enumerate(self.block):
                src = block(
                    x=src,
                    src_len=src_len,
                    dist=dist,
                    deg=deg,
                    edge=edge
                )
        return src.contiguous()

    @property
    def _attention_weight(self) -> List:
        return self._attention_score


# TODO: add embedding for reactants
