import torch
import torch.nn as nn
from typing import Dict, List

from .module_utils import RelTransformerBase, RelMultiHeadAttention, GLU, RMSNorm, PosWiseFFN


class RelTransformer_Decoder_Block(nn.Module):
    def __init__(
        self, d_model, d_ff, h, u0, v0, u1, v1,
        sin_enc=None,
        pe='abs',
        rel_dict0={},
        rel_dict1={},
        use_subs=False,
        dropout=0.1,
        block_id=0,
        alpha=None,
        init_beta=None,
        cls_len=0,
        **kwargs
    ):
        super(RelTransformer_Decoder_Block, self).__init__(**kwargs)
        self.block_id = block_id
        self.alpha = alpha
        self.init_beta = init_beta
        self.cls_len = cls_len
        self.use_subs = use_subs
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
            dropout=dropout
        )
        self._build_cache()

    def _build_model(
        self, d_model, d_ff, h, u0, v0, u1, v1,
        sin_enc=None,
        pe='abs',
        rel_dict0={},
        rel_dict1={},
        dropout=0.1
    ):
        if self.use_subs:
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            self.norm1.add_module('prod', RMSNorm(d_model))
            self.norm1.add_module('subs', RMSNorm(d_model))
            self.norm2.add_module('prod', RMSNorm(d_model))
            self.norm2.add_module('subs', RMSNorm(d_model))
            self.norm3.add_module('prod', RMSNorm(d_model))
            self.norm3.add_module('subs', RMSNorm(d_model))
        else:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.self_attention = RelMultiHeadAttention(
            d_model=d_model,
            h=h,
            u=u0,
            v=v0,
            sin_enc=sin_enc,
            pe=pe,
            rel_dict=rel_dict0,
            dropout=dropout,
            cls_len=self.cls_len
        )
        self.context_attention = RelMultiHeadAttention(
            d_model=d_model,
            h=h,
            u=u1,
            v=v1,
            sin_enc=None,
            pe='none',
            rel_dict=rel_dict1,
            dropout=dropout
        )
        self.ffn = GLU(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        if self.init_beta is not None:
            self.self_attention._init_param(vo_beta=self.init_beta)
            self.context_attention._init_param(vo_beta=self.init_beta)
            if self.ffn.__class__.__name__ == 'GLU':
                self.ffn._init_param(vo_beta=1.)
            elif self.ffn.__class__.__name__ == 'PosWiseFFN':
                self.ffn._init_param(vo_beta=self.init_beta)

    def _build_cache(self):
        self.context_cache = None
        self.lat_cache = None
        self.layer_cache = None

    def _init_cache(self, context_cache: torch.Tensor, lat_cache=None, layer_cache=None):
        self.context_cache = context_cache
        self.lat_cache = lat_cache
        self.layer_cache = layer_cache

    def _update_cache(self, context_cache=None, lat_cache=None, layer_cache=None):
        if context_cache is not None:
            self.context_cache = context_cache
        if lat_cache is not None:
            self.lat_cache = lat_cache
        if layer_cache is not None:
            if self.layer_cache is None:
                self.layer_cache = layer_cache
            else:
                self.layer_cache = torch.cat([self.layer_cache, layer_cache], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        context_len: List[torch.Tensor],
        tgt_len=None,
        task='prod2subs',
        rel=None,  # for relative position, which is not necessary in RoPE
        deg=None,  # for graph deg encoding
        future=False,  # if True, decoder will use a triu to compute autoregressive mask
        step=0
    ):
        self._update_cache(
            layer_cache=x
        )
        x_in, _ = self.self_attention(
            q=x,
            k=self.layer_cache,
            v=self.layer_cache,
            length=tgt_len,
            rel=rel,
            future=future,
            step=step
        )
        x = self.dropout1(x_in) + x * self.alpha
        if self.use_subs:
            if task == 'bidirection':
                x = torch.chunk(x, 2, dim=0)
                x = torch.cat([self.norm1[0](x[0]), self.norm1[1](x[1])], dim=0)
            elif task == 'prod2subs':
                x = self.norm1[0](x)
            elif task == 'subs2prod':
                x = self.norm1[1](x)
        else:
            x = self.norm1(x)
        x_in, _ = self.context_attention(
            q=x,
            k=self.context_cache,
            v=self.context_cache,
            length=context_len,
            deg=deg
        )
        x = self.dropout2(x_in) + x * self.alpha
        if self.use_subs:
            if task == 'bidirection':
                x = torch.chunk(x, 2, dim=0)
                x = torch.cat([self.norm2[0](x[0]), self.norm2[1](x[1])], dim=0)
            elif task == 'prod2subs':
                x = self.norm2[0](x)
            elif task == 'subs2prod':
                x = self.norm2[1](x)
        else:
            x = self.norm2(x)
        x = self.dropout3(self.ffn(x)) + x * self.alpha
        if self.use_subs:
            if task == 'bidirection':
                x = torch.chunk(x, 2, dim=0)
                x = torch.cat([self.norm3[0](x[0]), self.norm3[1](x[1])], dim=0)
            elif task == 'prod2subs':
                x = self.norm3[0](x)
            elif task == 'subs2prod':
                x = self.norm3[1](x)
        else:
            x = self.norm3(x)
        return x


class RelTransformer_Decoder(RelTransformerBase):
    def __init__(
        self, d_model, d_ff, h, layer,
        pe='abs',
        sin_enc=None,
        attn0_rel_dict={},
        attn1_rel_dict={},
        rel_apply='add',
        use_subs=False,
        dropout=0.1,
        alpha=None,
        init_beta=None,
        cls_len=0,
        **kwargs
    ):
        super(RelTransformer_Decoder, self).__init__(
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
            use_subs=use_subs,
            dropout=dropout,
            alpha=alpha,
            init_beta=init_beta,
            cls_len=cls_len
        )

    def _build_model(
        self, d_model, d_ff, h, layer,
        pe='abs',
        sin_enc=None,
        attn0_rel_dict={},
        attn1_rel_dict={},
        use_subs=False,
        dropout=0.1,
        alpha=None,
        init_beta=None,
        cls_len=0
    ):
        self.block = nn.Sequential()
        for i in range(layer):
            self.block.add_module(
                'block' + str(i),
                RelTransformer_Decoder_Block(
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
                    use_subs=use_subs,
                    dropout=dropout,
                    block_id=i,
                    alpha=alpha,
                    init_beta=init_beta,
                    cls_len=cls_len
                )
            )
        self._attention_score = [[None] * len(self.block) for _ in range(2)]

    def _init_cache(self, context_cache: torch.Tensor, lat_cache=None):
        for block in self.block:
            block._init_cache(context_cache=context_cache, lat_cache=lat_cache)

    def _update_cache(self, select_idx: torch.Tensor):  # for beam search update
        for block in self.block:
            block.context_cache = torch.index_select(block.context_cache, 
                                                     dim=0, index=select_idx)
            block.layer_cache = torch.index_select(block.layer_cache,
                                                   dim=0, index=select_idx)
            if block.lat_cache is not None:
                block.lat_cache = torch.index_select(block.lat_cache,
                                                     dim=0, index=select_idx)

    def forward(
        self,
        # size(batch, seq_len, d_model), need to be scaled before forward
        tgt: torch.Tensor,
        context_len: List[torch.Tensor],
        tgt_len=None,
        task='prod2subs',
        rel=None,  # for relative position, which is not necessary in RoPE
        deg=None,  # for graph deg encoding
        future=False,  # if True, decoder will use a triu to compute autoregressive mask
        step=0
    ):
        for i, block in enumerate(self.block):
            tgt = block(
                x=tgt,
                context_len=context_len,
                tgt_len=tgt_len,
                task=task,
                rel=rel,
                deg=deg,
                future=future,
                step=step
            )
        return tgt.contiguous()

    @property
    def _attention_weight(self) -> List:
        return self._attention_score
    

'''
GAU has lower performance, but requires much lower GPU memory.
'''

# class GAU_Decoder_Block(nn.Module):
#     def __init__(self, d_model, u0, v0, u1, v1, sin_enc = None, pe = 'abs', s = 128, rel_dict0 = {}, rel_dict1 = {}, dropout = 0.1, block_id = 0, alpha = None, init_beta = None, **kwargs):
#         super(GAU_Decoder_Block, self).__init__(**kwargs)
#         self.block_id = block_id
#         self.norm1 = RMSNorm(d_model)
#         self.norm2 = RMSNorm(d_model)
#         self.norm3 = RMSNorm(d_model)
#         self.self_gau = GAU(d_model, u0, v0, sin_enc = sin_enc, pe = pe, s = s, rel_dict = rel_dict0, dropout = dropout)
#         #self.context_gau = GAU(d_model, u1, v1, sin_enc = sin_enc, pe = 'none', s = s, rel_dict = rel_dict1, dropout = dropout)
#         self.context_gau = RelMultiHeadAttention(d_model, 8, u1, v1, pe = 'none', rel_dict = rel_dict1, dropout = dropout)
#         self.glu = GLU(d_model, scale_rate = 6, dropout = dropout)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.alpha = alpha
#         self.init_beta = init_beta

#         if init_beta is not None:
#             self.self_gau._init_param(vo_beta = init_beta)
#             self.context_gau._init_param(vo_beta = init_beta)
#             self.glu._init_param(vo_beta = init_beta)
    
#     def forward(self, x: torch.Tensor, preview: List, context_len: List[torch.Tensor], rel = None, deg = None, tgt_len = None, step = 0):
#         enc = preview[0][0]
#         if preview[1][self.block_id] is None:
#             context = x
#         else:
#             context = torch.cat([preview[1][self.block_id], x], dim = 1)
#         preview[1][self.block_id] = context
        
#         if self.training:
#             future = True
#         else: future = False

#         x_in = self.self_gau(x, tgt_len, context = context, rel = rel, future = future, step = step)
#         x = self.dropout1(x_in) + x * self.alpha
#         x = self.norm1(x)
#         #x_in = self.context_gau(x, context_len, context = enc, deg = deg)
#         x_in = self.context_gau(x, enc, enc, context_len, deg = deg)
#         x = self.dropout2(x_in) + x * self.alpha
#         x = self.norm2(x)
#         x = self.glu(x) + x * self.alpha
#         x = self.norm3(x)
#         return x, preview


# class GAU_Decoder(RelTransformerBase):
#     def __init__(self, d_model, layer, s = 128, sin_enc = None, pe = 'abs', attn0_rel_dict = {}, attn1_rel_dict = {}, dropout = 0.1, alpha = None, init_beta = None, **kwargs):
#         super(GAU_Decoder, self).__init__(s, d_model, attn0_rel_dict, attn1_rel_dict, **kwargs)
#         self.layer = layer
#         self.block = nn.Sequential()

#         for i in range(layer):
#             self.block.add_module(
#                 'block' + str(i),
#                 GAU_Decoder_Block(
#                     d_model,
#                     self.attn0_u, self.attn0_v, self.attn1_u, self.attn1_v,
#                     sin_enc = sin_enc, pe = pe, s = s,
#                     rel_dict0 = attn0_rel_dict, rel_dict1 = attn1_rel_dict,
#                     dropout = dropout, block_id = i, alpha = alpha, init_beta = init_beta
#                 )
#             )
        
#         self._attention_score = [[None] * len(self.block) for _ in range(2)]

#     def init_preview(self, enc_list: List[torch.Tensor]) -> List:
#         return [enc_list, [None] * self.layer]

#     def forward(self, tgt: torch.Tensor, preview: List, context_len: List[torch.Tensor], rel = None, deg = None, tgt_len = None, step = 0):
#         for i, block in enumerate(self.block):
#             tgt, preview = block(tgt, preview, context_len, rel = rel, deg = deg, tgt_len = tgt_len, step = step)
#         return tgt.contiguous(), preview
    
#     @property
#     def _attention_weight(self) -> List:
#         return self._attention_score
