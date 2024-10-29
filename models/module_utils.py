import os
import re

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Dict, List
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
from collections import deque

from utils.chem_tools import (SYMBOL_SIZE, E_LAYER_LIST, DEGREE_SIZE, CHARGE_SIZE, VALENCY_SIZE, RING_SIZE,
                                         HYDRO_SIZE, CHIRAL_SIZE, RS_SIZE, HYBRID_SIZE, SUBS_PROD_SIZE,
                                         BOND_TYPE_SIZE, BOND_STEREO_SIZE, BOND_CONJUGATE_SIZE, BOND_RING_SIZE,
                                         get_atom_feature)
from utils.preprocess_smi import canonicalize_smiles
from utils.data_loader import sequence_pad


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# param init for nn.Embedding
def embedding_init(embedding: nn.Embedding):
    fan_out = embedding.weight.size(1)
    std = 1.0 * math.sqrt(1.0 / float(fan_out))
    nn.init.normal_(embedding.weight, 0., std)
    if embedding.padding_idx is not None:
        with torch.no_grad():
            embedding.weight[embedding.padding_idx].fill_(0)
    return embedding


def get_one_hot(label: torch.Tensor, num_class: int):
    assert label.dim() == 1
    negative = label.eq(-1)
    label[negative] = 0
    one_hot = torch.zeros(label.size(0), num_class, dtype=torch.long, device=label.device)
    one_hot.scatter_(1, label.reshape(-1, 1), value=1)
    one_hot[negative] = 0
    return one_hot


# generate softmax mask
def get_mask(length: torch.Tensor, max_len: int, future: bool) -> torch.Tensor:
    mask = torch.zeros_like(length, device=length.device) \
        .unsqueeze(-1).repeat(1, 1, max_len)
    mask_idx = torch.arange((max_len), device=length.device) \
        .unsqueeze(0).repeat(length.size(1), 1)
    mask_idx = mask_idx.unsqueeze(0) < length.unsqueeze(-1)
    mask[~mask_idx] = 1
    if future:  # for autoregressive mask
        # triu 上三角区域, diagonal=1表示对角线上方
        future_mask = torch.triu(torch.ones((length.size(1), max_len),
                                            dtype=torch.int, device=length.device),
                                 diagonal=1).unsqueeze(0)
        mask = mask + future_mask
        mask[mask.gt(0)] = 1

    return mask.to(torch.bool)


# generate relative distance
def generate_dist(q_len, k_len, device, step=0, max_rel=4) -> torch.Tensor:
    assert max_rel >= 4
    positive = np.arange(0, k_len).tolist()
    negative = (np.arange(1, q_len + step) * -1).tolist()
    dist = []
    for i in range(q_len):
        i += step
        if i == 0:
            dist.append(positive[:k_len - i])
        elif i <= k_len - 1:
            dist.append(negative[:i][::-1] + positive[:k_len - i])
        else:
            dist.append(negative[i - k_len:i][::-1])
    dist = np.stack(dist, axis=0)
    dist = np.clip(dist, -max_rel, max_rel) + max_rel
    return torch.tensor(dist, dtype=torch.long, device=device)


# relative attention matmul, from OpenNMT
def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


# def beam_result_process(
#     tgt_seq: list,    # torch.Tensor,
#     tgt_len: list,    # torch.Tensor,
#     token_idx: Dict,
#     beam_result: torch.Tensor,
#     beam_scores: torch.Tensor,
#     pres: torch.Tensor,
#     posts: torch.Tensor
# ):
#     '''
#     canonicalize the predict smiles, then calculate top-k accuracy
#     beam_result: [b_sz, beam_sz, seq_len]
#     '''
#     idx_token = {v: k for k, v in token_idx.items()}
#     eos_ids, pad_ids = token_idx['<EOS>'], token_idx['<PAD>']
#     batch_size, topk, res_len = beam_result.size()
#     topk_acc = np.zeros((batch_size, topk))
#     topk_invalid = np.zeros((batch_size, topk))
#
#     # tgt_seq, beam_result = tgt_seq.detach().cpu().numpy(), beam_result.detach().cpu().numpy()
#     # tgt_seq = tgt_seq.detach().cpu().numpy()
#     tgt_seq = np.array(tgt_seq)
#     beam_result = beam_result.detach().cpu().numpy()
#     # for diverse prediction, trace the multi-ground truth
#     pres, posts = pres.detach().cpu().numpy(), posts.detach().cpu().numpy()
#
#     all_smis = []
#     tgt_smis = []
#
#     rxns_coverage = {}     # 1-N num: [cov_sc_1, cov_sc_2, ...]
#
#     for batch_id, batch_res in enumerate(beam_result):
#         beam_smi = []
#         # tgt = tgt_seq[batch_id]
#         for beam_id, beam_res in enumerate(batch_res):
#             res = beam_res[((beam_res != eos_ids) & (beam_res != pad_ids))]
#             res_smi = [idx_token[idx] for idx in res]
#             res_smi = ' '.join(res_smi).replace(' ', '')
#             res_smi = canonicalize_smiles(res_smi, False, False)
#
#             # record coverage for per 1-to-N rxns
#             num_matched_rxns = 0
#             num_multi_rxns = posts[batch_id] - pres[batch_id]
#             if num_multi_rxns not in rxns_coverage.keys():
#                 rxns_coverage[num_multi_rxns] = []
#
#             multi_tgts = tgt_seq[pres[batch_id]: posts[batch_id]]
#             multi_tgts_seq_len = [len(seq) for seq in multi_tgts]
#             seq_maxlen = max(multi_tgts_seq_len)
#             multi_tgts = sequence_pad(seq=multi_tgts,
#                 seq_len=multi_tgts_seq_len,
#                 max_len=seq_maxlen,
#                 pad=pad_ids)
#
#             for i, tgt in enumerate(multi_tgts):
#                 # tgt = tgt_seq[j]
#                 tgt = tgt[((tgt != eos_ids) & (tgt != pad_ids))]
#                 tgt_smi = [idx_token[idx] for idx in tgt]
#
#             # tgt, res = tgt[((tgt != eos_ids) & (tgt != pad_ids))],\
#             #     beam_res[((beam_res != eos_ids) & (beam_res != pad_ids))]
#             # tgt_smi, res_smi = [idx_token[idx]for idx in tgt],\
#             #     [idx_token[idx] for idx in res]
#
#                 tgt_smi= ''.join(tgt_smi).replace(' ', '')       # 已经是canonicalize后的
#
#                 if tgt_smi == 'CC':
#                     break  # problematic SMILES
#                 else:
#                     if i == 0:
#                         tmp_tgt_smi = tgt_smi
#
#                 if res_smi == '':
#                     topk_invalid[batch_id, beam_id] = 1
#                     break
#                 else:
#                     # each batch only has one correct result
#                     if (res_smi == tgt_smi) and (topk_acc[batch_id].sum() == 0):
#                         if num_matched_rxns == 0:
#                             topk_acc[batch_id, beam_id] = 1
#                             num_matched_rxns = 1
#                         else:
#                             num_matched_rxns += 1
#
#             rxns_coverage[num_multi_rxns].append(num_matched_rxns / num_multi_rxns)
#
#             beam_smi.append(res_smi)
#
#         beam_smi = ','.join(beam_smi)
#         all_smis.append(beam_smi)
#         tgt_smis.append(tmp_tgt_smi)
#
#
#     # TODO: 保存 topk_acc 和 topk_invalid 的具体信息
#     return topk_acc.sum(axis=0), topk_invalid.sum(axis=0), all_smis, tgt_smis, topk_acc, topk_invalid, rxns_coverage    #, tgt_smis


def beam_result_process(
    tgt_seq: torch.Tensor,
    tgt_len: torch.Tensor,
    token_idx: Dict,
    beam_result: torch.Tensor,
    beam_scores: torch.Tensor
):
    '''canonicalize the predict smiles, then calculate top-k accuracy'''
    idx_token = {v: k for k, v in token_idx.items()}
    eos_ids, pad_ids = token_idx['<EOS>'], token_idx['<PAD>']
    batch_size, topk, res_len = beam_result.size()
    topk_acc = np.zeros((batch_size, topk))
    topk_invalid = np.zeros((batch_size, topk))
    tgt_seq, beam_result = tgt_seq.detach().cpu().numpy(),\
        beam_result.detach().cpu().numpy()
    all_smi = []
    tgt_smis = []

    for batch_id, batch_res in enumerate(beam_result):
        beam_smi = []
        for beam_id, beam_res in enumerate(batch_res):
            tgt = tgt_seq[batch_id]
            tgt, res = tgt[((tgt != eos_ids) & (tgt != pad_ids))],\
                beam_res[((beam_res != eos_ids) & (beam_res != pad_ids))]
            tgt_smi, res_smi = [idx_token[idx]for idx in tgt],\
                [idx_token[idx] for idx in res]
            tgt_smi, res_smi = ''.join(tgt_smi), ''.join(res_smi)
            if tgt_smi == 'CC':
                break  # problematic SMILES
            res_smi = canonicalize_smiles(res_smi, False, False)
            beam_smi.append(res_smi)
            if res_smi == '':
                topk_invalid[batch_id, beam_id] = 1
            else:
                # each batch only has one correct result
                if (res_smi == tgt_smi) and (topk_acc[batch_id].sum() == 0):
                    topk_acc[batch_id, beam_id] = 1
        beam_smi = ','.join(beam_smi)
        all_smi.append(beam_smi)
        tgt_smis.append(tgt_smi)
    return topk_acc.sum(axis=0), topk_invalid.sum(axis=0), all_smi, tgt_smis


def train_plot(loss_list: List, seq_acc_list: List, token_acc_list: List, ckpt_dir):
    '''loss, sequence accuracy, token accuracy during training'''
    plt.rcParams['figure.dpi'] = 600
    plt.plot(loss_list, linewidth=0.5)
    plt.ylim(0, 10)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training loss line')
    if os.path.exists(os.path.join(ckpt_dir, 'train loss.png')):
        os.remove(os.path.join(ckpt_dir, 'train loss.png'))
    plt.savefig(os.path.join(ckpt_dir, 'train loss.png'), format='PNG')
    plt.close()

    x = [_ for _ in range(len(loss_list))]
    p2s_seq_acc, p2s_token_acc = [i[0] for i in seq_acc_list], \
        [i[0] for i in token_acc_list]
    s2p_seq_acc, s2p_token_acc = [i[1] for i in seq_acc_list], \
        [i[1] for i in token_acc_list]
    plt.plot(x, p2s_seq_acc, '-', color='r',
             label='p2s_seq', linewidth=0.5)
    plt.plot(x, p2s_token_acc, '-.', color='r',
             label='p2s_token', linewidth=0.5)
    plt.plot(x, s2p_seq_acc, '-', color='g',
             label='s2p_seq', linewidth=0.5)
    plt.plot(x, s2p_token_acc, '-.', color='g',
             label='s2p_token', linewidth=0.5)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc='best')
    plt.title('training acc line')
    if os.path.exists(os.path.join(ckpt_dir, 'train acc.png')):
        os.remove(os.path.join(ckpt_dir, 'train acc.png'))
    plt.savefig(os.path.join(ckpt_dir, 'train acc.png'), format='PNG')
    plt.close()


def eval_plot(
        topk_seq_acc: List[float],
        topk_seq_invalid: List[float],
        beam_size: int,
        data_name: str,
        ckpt_dir: str,
        ckpt_name,
        args,
        is_train=True
):
    '''each eavling result generate, including top-k(mostly top-10) sequence accuracy and top-k(mostly top-10) sequence invalid rate'''
    plt.rcParams['figure.dpi'] = 600
    if not is_train:
        mode = args.mode
        task = args.eval_task
    else:
        mode = 'train'
        task = 'prod2subs'
    return_num = args.return_num
    top1_weight = args.top1_weight
    x = [i for i in range(1, return_num + 1, 1)]
    plt.plot(x, topk_seq_acc, 's-', color='r',
             label='seq_acc_sum', linewidth=0.25, markersize=2.0)
    plt.plot(x, topk_seq_invalid, 'o-', color='r',
             label='seq_invalid', linewidth=0.25, markersize=2.0)
    plt.xlabel('beam search top k')
    plt.ylabel('acc')
    plt.legend(loc='best')
    plt.title('top 1 seq acc = {0:.6}%, invalid = {1:.6}%'.format(
        topk_seq_acc[0] * 100, topk_seq_invalid[0] * 100))
    if os.path.exists(os.path.join(ckpt_dir,
                                   '{0} beam {1} T {2} {3} {4} acc.png'.format(ckpt_name, beam_size, args.T, task,
                                                                               mode))):
        os.remove(os.path.join(ckpt_dir, '{0} beam {1} T {2} {3} {4} acc.png'.format(
            ckpt_name, beam_size, args.T, task, mode)))
    plt.savefig(os.path.join(ckpt_dir, '{0} beam {1} T {2} {3} {4} acc.png'.format(
        ckpt_name, beam_size, args.T, task, mode)), format='PNG')
    plt.close()

    if os.path.exists(os.path.join(ckpt_dir,
                                   '{0} beam {1} T {2} {3} {4} acc.txt'.format(ckpt_name, beam_size, args.T, task,
                                                                               mode))):
        os.remove(os.path.join(ckpt_dir, '{0} beam {1} T {2} {3} {4} acc.txt'.format(
            ckpt_name, beam_size, args.T, task, mode)))
    with open(os.path.join(ckpt_dir,
                           '{0} beam {1} T {2} {3} {4} acc.txt'.format(ckpt_name, beam_size, args.T, task, mode)),
              mode='w') as f:
        for k, v in args.__dict__.items():
            f.writelines('args -> {0}: {1}\n'.format(k, v))
        info = 'module is train&eval with {0}, in the {1} epoch, eval top {2} accuracy and invalid rate is: '.format(
            data_name, ckpt_name, return_num)
        f.writelines(info + '\n')
        for i, j, k in zip(topk_seq_acc, topk_seq_invalid, range(len(topk_seq_acc))):
            f.writelines('top {0}: acc = {1:.6}%, invalid = {2:.6}% \n'.format(
                k + 1, i * 100, j * 100))

        weighted_acc = 0.
        if return_num >= 10:
            weighted_acc = topk_seq_acc[0] * top1_weight + (
                        (topk_seq_acc[2] + topk_seq_acc[4] + topk_seq_acc[9]) / 3) * (1 - top1_weight)
        elif return_num >= 5:
            weighted_acc = topk_seq_acc[0] * top1_weight + ((topk_seq_acc[2] + topk_seq_acc[4]) / 2) * (1 - top1_weight)
        elif return_num >= 3:
            weighted_acc = topk_seq_acc[0] * top1_weight + topk_seq_acc[2] * (1 - top1_weight)
        if weighted_acc > 0.:
            f.writelines('\nweighted seq_acc = {0:.6}%'.format(weighted_acc * 100))


def train_eval_plot(seq_acc_list: List[List], ckpt_dir):
    '''each eval result during training'''
    return_num = len(seq_acc_list[0])
    x = [_ for _ in range(len(seq_acc_list))]
    plt.rcParams['figure.dpi'] = 600

    top1_seq = [i[0] for i in seq_acc_list]
    top1_loc = x[np.argmax(top1_seq)]
    plt.plot(x, top1_seq, 's-', color='r', label='seq1({0})'.format(top1_loc),
             linewidth=0.25, markersize=2.0)
    if return_num >= 3:
        top3_seq = [i[2] for i in seq_acc_list]
        top3_loc = x[np.argmax(top3_seq)]
        plt.plot(x, top3_seq, 's-', color='g', label='seq3({0})'.format(top3_loc),
                 linewidth=0.25, markersize=2.0)
    if return_num >= 5:
        top5_seq = [i[4] for i in seq_acc_list]
        top5_loc = x[np.argmax(top5_seq)]
        plt.plot(x, top5_seq, 's-', color='b', label='seq5({0})'.format(top5_loc),
                 linewidth=0.25, markersize=2.0)
    if return_num >= 10:
        top10_seq = [i[9] for i in seq_acc_list]
        top10_loc = x[np.argmax(top10_seq)]
        plt.plot(x, top10_seq, 's-', color='k', label='seq10({0})'.format(top10_loc),
                 linewidth=0.25, markersize=2.0)

    if return_num >= 10:
        avg25 = [i1 * 0.25 + ((i3 + i5 + i10) / 3) * 0.75 \
                 for (i1, i3, i5, i10) in zip(top1_seq, top3_seq, top5_seq, top10_seq)]
        avg50 = [i1 * 0.5 + ((i3 + i5 + i10) / 3) * 0.5 \
                 for (i1, i3, i5, i10) in zip(top1_seq, top3_seq, top5_seq, top10_seq)]
        avg75 = [i1 * 0.75 + ((i3 + i5 + i10) / 3) * 0.25 \
                 for (i1, i3, i5, i10) in zip(top1_seq, top3_seq, top5_seq, top10_seq)]

        avg25_loc, avg50_loc, avg75_loc = x[np.argmax(avg25)], x[np.argmax(avg50)], x[np.argmax(avg75)]
        plt.plot(x, avg25, 'o-', color='r', label='avg25({0})'.format(avg25_loc),
                 linewidth=0.25, markersize=2.0)
        plt.plot(x, avg50, 'o-', color='g', label='avg50({0})'.format(avg50_loc),
                 linewidth=0.25, markersize=2.0)
        plt.plot(x, avg75, 'o-', color='b', label='avg75({0})'.format(avg75_loc),
                 linewidth=0.25, markersize=2.0)

    plt.xlabel('check point count')
    plt.ylabel('acc')
    plt.legend(loc='best')
    plt.title('module eval accuracy during training')
    if os.path.exists(os.path.join(ckpt_dir, 'train eval acc.png')):
        os.remove(os.path.join(ckpt_dir, 'train eval acc.png'))
    plt.savefig(os.path.join(ckpt_dir, 'train eval acc.png'), format='PNG')
    plt.close()


def token_acc_record(token_count_each: List, token_idx: Dict, epoch: int, ckpt_dir):
    '''each token's accuracy during a epoch'''
    with open(os.path.join(ckpt_dir, '{0} training each token accuracy.txt'.format(epoch)), mode='w') as f:
        for true_count, all_count, token in zip(token_count_each[0], token_count_each[1], token_idx.keys()):
            f.writelines('{0} : {1:.6} ; {2}/{3}\n'.format(token,
                                                           true_count / max(all_count, 1), true_count, all_count))
        f.writelines('all token count in epoch: {0}, correct token count: {1}, correct rate:{2:.6}\n'.format(
            sum(token_count_each[1]), sum(token_count_each[0]), sum(token_count_each[0]) / sum(token_count_each[1])))


def type_acc_record(type_acc_list: List, ckpt_dir, ckpt_name, task='prod2subs', mode='eval'):
    '''the reaction type prediction accuracy'''
    with open(os.path.join(ckpt_dir, '{0} {1} {2} reaction_type predict acc.txt'.format(ckpt_name, task, mode)),
              mode='w') as f:
        for i, acc in enumerate(type_acc_list):
            f.writelines('top {0}: acc: {1:.6}\n'.format(i + 1, acc))


class AbsPositionalEncoding(nn.Module):
    '''the base class for positional encoding'''

    def __init__(
            self,
            d_model,
            max_len=2048,
            dropout=0.1,
            **kwargs
    ):
        super(AbsPositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pos = torch.zeros((1, max_len, d_model))
        emb = torch.arange(max_len, dtype=torch.float).reshape(-1, 1) / \
              torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        pos[:, :, 0::2] = torch.sin(emb)
        pos[:, :, 1::2] = torch.cos(emb)
        self.register_buffer('pos', pos)

    def forward(
            self,
            x: torch.Tensor,
            step=0
    ):
        x = x * math.sqrt(self.d_model) + self.pos[:, step:(x.size(1) + step), :]
        x = self.dropout(x)
        return x


class RelTransformerBase(nn.Module):
    '''the base class for relative transformer, generate bias for relative matmul'''

    def __init__(
            self, d_rel0, d_rel1,
            attn0_rel_dict={},
            attn1_rel_dict={},
            rel_apply='add',
            **kwargs
    ):
        super(RelTransformerBase, self).__init__(**kwargs)
        self.attn0_rel_dict = attn0_rel_dict
        self.attn0_u = nn.ParameterDict()
        self.attn0_v = nn.ParameterDict()
        self.attn1_rel_dict = attn1_rel_dict
        self.attn1_u = nn.ParameterDict()
        self.attn1_v = nn.ParameterDict()
        if rel_apply == 'mul':
            self._generate_uv(d_rel0, d_rel1)
            self._init_uv(d_rel0, d_rel1)

    def _generate_uv(self, d_rel0, d_rel1):
        vnode_re = re.compile(r'vnode')
        if len(self.attn0_rel_dict) > 0:
            self.attn0_u['qk'] = nn.Parameter(torch.Tensor(d_rel0), requires_grad=True)
            for rel_name in self.attn0_rel_dict.keys():
                if re.search(vnode_re, rel_name) is None:
                    self.attn0_v[rel_name] = nn.Parameter(torch.Tensor(d_rel0), requires_grad=True)
        if len(self.attn1_rel_dict) > 0:
            self.attn1_u['qk'] = nn.Parameter(torch.Tensor(d_rel1), requires_grad=True)
            for rel_name in self.attn1_rel_dict.keys():
                if re.search(vnode_re, rel_name) is None:
                    self.attn1_v[rel_name] = nn.Parameter(torch.Tensor(d_rel1), requires_grad=True)

    def _init_uv(self, d_rel0, d_rel1):
        std0 = 1 / math.sqrt(d_rel0)
        std1 = 1 / math.sqrt(d_rel1)
        with torch.no_grad():
            if len(self.attn0_rel_dict) > 0:
                for param in self.attn0_u.parameters():
                    nn.init.normal_(param, 0., std0)
                for param in self.attn0_v.parameters():
                    nn.init.normal_(param, 0., std0)
            if len(self.attn1_rel_dict) > 0:
                for param in self.attn1_u.parameters():
                    nn.init.normal_(param, 0., std1)
                for param in self.attn1_v.parameters():
                    nn.init.normal_(param, 0., std1)


class AttentionBase(nn.Module):
    '''the base class for relative self/context attention'''

    def __init__(
            self, d_head,
            scale_length=0,  # using for softmax length scale
            sin_enc=None,
            pe='none',
            rel_dict={},
            **kwargs
    ):
        super(AttentionBase, self).__init__(**kwargs)
        self.pe = pe
        if self.pe == 'rope':
            # (1, max_len, d_model), [..., 0::2] = sin, [..., 1::2] = cos
            assert sin_enc.dim() == 3
            self.register_buffer('pos', sin_enc)
        else:
            self.pos = None
        self.rel_emb = self._create_relative_emb(rel_dict)
        if len(self.rel_emb) > 0:
            self._init_emb()
        self.scale = 1 / math.sqrt(d_head)
        self.scale_length = scale_length

    def _sim_scale(self, q: torch.Tensor):
        if self.scale_length < 1:
            q = q * self.scale
        else:
            q = q * (math.log(q.size(-2), self.scale_length) * self.scale)
        return q

    def _create_relative_emb(self, rel_dict: Dict):
        rel_emb = nn.ModuleDict()
        if len(rel_dict) > 0:
            for rel_name, rel_len in rel_dict.items():
                # [emb_len, d_emb] or [emb_len, d_emb, pad_idx]
                assert len(rel_len) in [2, 3]
                if len(rel_len) == 2 and rel_len[0] > 0:
                    rel_emb.add_module(
                        rel_name, nn.Embedding(rel_len[0], rel_len[1]))
                elif len(rel_len) == 3 and rel_len[0] > 0:
                    rel_emb.add_module(
                        rel_name, nn.Embedding(rel_len[0], rel_len[1], padding_idx=rel_len[2]))
        return rel_emb

    def _init_param(self):
        raise NotImplementedError

    def _init_emb(self):
        for rel_name in self.rel_emb.keys():
            self.rel_emb[rel_name] = embedding_init(self.rel_emb[rel_name])

    def _compute_mask(
            self,
            valid_len: List[torch.Tensor],
            batch_size: int,
            q_size: int,
            k_size: int,
            future: bool,   # decide whether to perform casual mask
            device,
            multi_head=True,
            dist_range_mask=None,  # for different dist mask
            cls_len=0
    ):
        if valid_len is None:
            mask = torch.zeros((batch_size, q_size, k_size[0]), dtype=torch.bool, device=device)
        elif isinstance(valid_len, List):
            mask = []
            for length, key_size in zip(valid_len, k_size):
                # k_size: num of tokens in k
                if len(length.size()) == 1:
                    length = length.unsqueeze(-1).repeat(1, q_size)     # [batch_sz] -> [batch_sz, 1_size]
                    mask.append(get_mask(length, key_size, future))
                else:
                    mask.append(get_mask(length, key_size, future))
            mask = torch.cat(mask, dim=2)
        else:
            raise TypeError('length must be None or a list of Tensor.')

        if dist_range_mask is not None:
            mask = torch.logical_or(mask, dist_range_mask)
        if future and cls_len > 0:
            mask[:, :cls_len + 1, :cls_len + 1] = False
        if multi_head:
            mask = mask.unsqueeze(1)
        return mask

    # the RoPE apply to query and key
    def _apply_rope(self, x: List[torch.Tensor], steps: List[int]):
        if self.pe != 'rope':
            return x
        else:
            assert len(x) == len(steps)
            seq_lens = [data.size(1) for data in x]
            sin_pos = self.pos[..., 0::2].repeat_interleave(2, -1)
            cos_pos = self.pos[..., 1::2].repeat_interleave(2, -1)
            output = []
            for x_i, seq_len, step in zip(x, seq_lens, steps):
                x_i_shift = torch.stack([-x_i[..., 1::2], x_i[..., 0::2]], dim=-1)
                x_i_shift = x_i_shift.reshape(x_i.shape)
                output.append(x_i * cos_pos[:, step:seq_len + step, :] +
                              x_i_shift * sin_pos[:, step:seq_len + step, :])
            return output


class RelMultiHeadAttention(AttentionBase):
    '''a basic class for attention with diverse relative embedding'''

    def __init__(
            self, d_model, h, u, v,
            scale_length=0,
            sin_enc=None,
            pe='none',
            rel_dict={},
            dropout=0.1,
            dist_range=-1,
            cls_len=0,
            **kwargs
    ):
        super(RelMultiHeadAttention, self).__init__(
            d_head=int(d_model // h),
            scale_length=scale_length,
            sin_enc=sin_enc,
            pe=pe,
            rel_dict=rel_dict,
            **kwargs
        )
        self.d_model = d_model
        self.dist_range = dist_range
        self.cls_len = cls_len
        self.h = h
        self.h_dim = int(self.d_model // h)
        self._build_model(
            u=u,
            v=v,
            dropout=dropout
        )

    def _build_model(self, u, v, dropout=0.1):
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)
        self.u = u  # size = (d_model,), nn.ParameterDict
        self.v = v  # size = (d_model,), nn.ParameterDict

    def _init_param(self, vo_beta: float):
        nn.init.xavier_normal_(self.W_q.weight, gain=1)
        nn.init.xavier_normal_(self.W_k.weight, gain=1)
        nn.init.xavier_normal_(self.W_v.weight, gain=vo_beta)
        nn.init.xavier_normal_(self.W_o.weight, gain=vo_beta)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            length: List[torch.Tensor],
            keys_size=None,
            dist=None,
            deg=None,
            rel=None,
            edge=None,
            bias=None,
            future=False,
            step=0,
            return_attention=False,
            lat_bias=None  # size(batch, 1, d_model)
    ):
        """
            deg: [b_z, nodes]
            dist: [b_z, nodes, nodes]
            edge: [b_z, nodes, nodes, types]
            q, k, v: [b_z, nodes, d_model]
        """
        if self.dist_range >= 0 and dist is not None:
            dist_range_mask = dist.gt(self.dist_range)
            nan_ignore_mask = torch.arange((max(length[0])), dtype=torch.long, device=q.device)
            nan_ignore_mask = nan_ignore_mask[None, :] >= length[0][:, None]
            dist_range_mask.masked_fill_(nan_ignore_mask.unsqueeze(-1), False)
        else:
            dist_range_mask = None
        mask = self._compute_mask(
            valid_len=length,
            batch_size=q.size(0),
            q_size=q.size(1),
            k_size=keys_size if keys_size is not None else [k.size(1)],
            future=future,
            device=q.device,
            multi_head=True,
            dist_range_mask=dist_range_mask,
            cls_len=self.cls_len
        )

        batch_size = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)

        def shape(x: torch.Tensor):
            return x.view(batch_size, -1, self.h, self.h_dim).transpose(1, 2)

        def unshape(x: torch.Tensor):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.h_dim)

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        if self.pe == 'rope':
            # 对 cls_token 后的sequence + pe
            if self.cls_len > 0 and step == 0:
                q_smi, k_smi = self._apply_rope([q[:, self.cls_len:], k[:, self.cls_len:]], [step, 0])
                q, k = torch.cat([q[:, :self.cls_len], q_smi], dim=1), torch.cat([k[:, :self.cls_len], k_smi], dim=1)
            elif self.cls_len > 0 and step > 0:
                q, k_smi = self._apply_rope([q, k[:, self.cls_len:]], [step, 0])
                k = torch.cat([k[:, :self.cls_len], k_smi], dim=1)
            else:
                q, k = self._apply_rope([q, k], [step, 0])

        q, k, v = shape(q), shape(k), shape(v)
        q = self._sim_scale(q)

        if len(self.u) == 0:
            qk = torch.matmul(q, k.transpose(2, 3))
            if dist is not None and 'dist' in self.rel_emb.keys():
                if 'vnode_dist' in self.rel_emb.keys():
                    rnode_dist = self.rel_emb['dist'](dist[:, 1:, 1:])
                    vnode_dist_row, vnode_dist_col = self.rel_emb['vnode_dist'](dist[:, 0].unsqueeze(1)), \
                        self.rel_emb['vnode_dist'](dist[:, :, 0].unsqueeze(2))
                    dist = torch.cat([vnode_dist_row[:, :, 1:], rnode_dist], dim=1)
                    dist = torch.cat([vnode_dist_col, dist], dim=2)
                else:
                    dist = self.rel_emb['dist'](dist)

                dist = dist.permute(0, 3, 1, 2)
                # (batch, h, q_len, k_len)
                qk = qk + dist

            if deg is not None and 'deg' in self.rel_emb.keys():
                if 'vnode_deg' in self.rel_emb.keys():
                    deg = torch.cat([self.rel_emb['vnode_deg'](deg[:, 0].unsqueeze(-1)), \
                                     self.rel_emb['deg'](deg[:, 1:])], dim=1)
                else:
                    deg = self.rel_emb['deg'](deg)

                deg = deg.permute(0, 2, 1).unsqueeze(2)
                # (batch, h, 1, k_len)
                qk = qk + deg

        else:
            qk = torch.matmul(
                q + self.u['qk'].reshape(1, self.h, 1, self.h_dim), k.transpose(2, 3))
            if dist is not None and 'dist' in self.rel_emb.keys():
                if 'vnode_dist' in self.rel_emb.keys():
                    rnode_dist = self.rel_emb['dist'](dist[:, 1:, 1:])
                    vnode_dist_row, vnode_dist_col = self.rel_emb['vnode_dist'](dist[:, 0].unsqueeze(1)), \
                        self.rel_emb['vnode_dist'](dist[:, :, 0].unsqueeze(2))
                    dist = torch.cat([vnode_dist_row[:, :, 1:], rnode_dist], dim=1)
                    dist = torch.cat([vnode_dist_col, dist], dim=2)
                else:
                    dist = self.rel_emb['dist'](dist)

                dist = dist.reshape(batch_size, q_len, k_len, self.h, self.h_dim)
                # (batch, h, q_len, d_model/h, k_len)
                dist = dist.permute(0, 3, 1, 4, 2)
                qk_rel = torch.matmul((q + self.v['dist'].reshape(1, self.h, 1, self.h_dim))
                                      .unsqueeze(-2), dist)
                qk = qk + qk_rel.squeeze(-2)

            if deg is not None and 'deg' in self.rel_emb.keys():
                if 'vnode_deg' in self.rel_emb.keys():
                    deg = torch.cat([self.rel_emb['vnode_deg'](deg[:, 0].unsqueeze(-1)), \
                                     self.rel_emb['deg'](deg[:, 1:])], dim=1)
                else:
                    deg = self.rel_emb['deg'](deg)

                deg = deg.reshape(batch_size, k_len, self.h, self.h_dim)
                # (batch, h, d_model/h, k_len)
                deg = deg.permute(0, 2, 3, 1)
                qk_rel = torch.matmul(q + self.v['deg'].reshape(1, self.h, 1, self.h_dim), deg)
                qk = qk + qk_rel

            if rel is not None and 'rel' in self.rel_emb.keys():
                # rel -> (q_len, k_len, d_model/h)
                rel = self.rel_emb['rel'](rel)
                qk_rel = relative_matmul(q + self.v['rel'].reshape(1, self.h, 1, self.h_dim), rel, True)
                qk = qk + qk_rel

            if lat_bias is not None:
                lat_bias = shape(lat_bias)
                # scale by dim per head
                lat_bias = lat_bias / math.sqrt(self.h_dim)
                lat_bias = torch.matmul(lat_bias + self.v['lat'].reshape(1, self.h, 1, self.h_dim),
                                        k.transpose(2, 3))
                qk = qk + lat_bias

        if edge is not None:
            # (batch, q_len, k_len, h)
            edge = edge.permute(0, 3, 1, 2)
            qk = qk + edge

        if bias != None:
            # (batch, q_len, k_len, h)
            bias = bias.permute(0, 3, 1, 2)
            qk = qk + bias

        # actually mask no where during encoding
        qk.masked_fill_(mask, -float('inf'))
        qk_save = qk.clone()
        qk = torch.softmax(qk, dim=-1)
        drop_qk = self.dropout(qk)

        output = torch.matmul(drop_qk, v)
        output = unshape(output)
        output = self.W_o(output)
        if return_attention:
            return output, qk_save
        else:
            return output, None


class PosWiseFFN(nn.Module):
    def __init__(
            self, d_model,
            d_ff,
            d_out=0,
            dropout=0.1,
            **kwargs
    ):
        super(PosWiseFFN, self).__init__(**kwargs)
        d_out = d_out if d_out > 0 else d_model
        self.posffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_out)
        )

    def _init_param(self, vo_beta: float):
        for param in self.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_normal_(param, gain=vo_beta)

    def forward(self, x: torch.Tensor):
        return self.posffn(x)


class Model_Save():
    def __init__(
            self,
            ckpt_dir: str,  # the path of checkpoint dir
            device: str,
            # if top, module will save according to its top1 acc,
            # if mean, module will save according to its mean(top1 + top3 + top5 + top10),
            # if last, it will save the last module
            save_strategy='mean',
            save_num=10,  # the max ckpt number
            swa_count=10,
            swa_tgt=5,
            const_save_epoch=[],  # save these epoch consistently.
            top1_weight=0.5
    ):
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.save_strategy = save_strategy
        self.save_num = save_num
        self.swa_count = swa_count
        self.swa_tgt = swa_tgt
        self.save_count = 0
        self.swa_list = []
        self.top1_weight = top1_weight
        self.const_save_epoch = const_save_epoch
        self.ckpt_queue = deque({}, maxlen=self.save_num)
        self.queue_list = os.path.join(self.ckpt_dir, 'ckpt_queue.txt')

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    def _remove_ckpt(self, ckpt: Dict):
        ckpt_epoch = list(ckpt.keys())[0]
        ckpt_path = os.path.join(self.ckpt_dir, '{}.ckpt'.format(ckpt_epoch))
        if ckpt_epoch not in self.const_save_epoch:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    def _save_ckpt(self, epoch: int, module, optimizer, lr_schedule):
        save_path = os.path.join(self.ckpt_dir, '{}.ckpt'.format(epoch))
        torch.save({
            'epoch': epoch,
            'module_param': module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_schedule': lr_schedule.state_dict()
        }, save_path)

    def save(
            self,
            module,
            optimizer,
            lr_schedule,
            epoch: int,
            acc_list: List[int]
    ):
        if self.save_strategy == 'top':
            acc = acc_list[0]
        elif self.save_strategy == 'mean':
            assert len(acc_list) >= 3
            if len(acc_list) >= 10:
                acc = acc_list[0] * self.top1_weight + \
                      ((acc_list[2] + acc_list[4] + acc_list[9]) / 3) * (1 - self.top1_weight)
            elif len(acc_list) >= 5:
                acc = acc_list[0] * self.top1_weight + \
                      ((acc_list[2] + acc_list[4]) / 2) * (1 - self.top1_weight)
            else:
                acc = acc_list[0] * self.top1_weight + acc_list[2] * (1 - self.top1_weight)
        else:
            acc = -1

        if len(self.ckpt_queue) == 0:
            self.ckpt_queue.append({epoch: acc})
            self._save_ckpt(epoch, module, optimizer, lr_schedule)
        else:
            if self.save_strategy == 'last':
                if len(self.ckpt_queue) == self.ckpt_queue.maxlen:
                    pop_ckpt = self.ckpt_queue.popleft()
                    self._remove_ckpt(pop_ckpt)
                self.ckpt_queue.append({epoch: acc})
                self._save_ckpt(epoch, module, optimizer, lr_schedule)
            else:
                # reverse, the last is the top-1
                for i in range(len(self.ckpt_queue) - 1, -1, -1):
                    ckpt = self.ckpt_queue[i]
                    ckpt_acc = list(ckpt.values())[0]
                    if acc <= ckpt_acc:
                        if i == 0 and len(self.ckpt_queue) < self.ckpt_queue.maxlen:
                            self.ckpt_queue.insert(i, {epoch: acc})
                            self._save_ckpt(epoch, module, optimizer, lr_schedule)
                        else:
                            continue
                    if acc > ckpt_acc:
                        if len(self.ckpt_queue) == self.ckpt_queue.maxlen:
                            pop_ckpt = self.ckpt_queue.popleft()
                            self._remove_ckpt(pop_ckpt)
                            self.ckpt_queue.insert(i, {epoch: acc})
                            self._save_ckpt(epoch, module, optimizer, lr_schedule)
                            break
                        else:
                            self.ckpt_queue.insert(i + 1, {epoch: acc})
                            self._save_ckpt(epoch, module, optimizer, lr_schedule)
                            break
        if epoch in self.const_save_epoch and epoch not in [list(elem.items())[0][0] for elem in self.ckpt_queue]:
            self._save_ckpt(epoch, module, optimizer, lr_schedule)
        self.save_count += 1
        if self.save_count % self.swa_count == 0:
            swa_list = [list(self.ckpt_queue[i].keys())[0] for i in range(
                len(self.ckpt_queue) - 1, len(self.ckpt_queue) - self.swa_tgt - 1, -1)]
            self.swa_list.append(','.join([str(i) for i in swa_list]))
            self.swa(swa_list, swa_name='swa' + str(len(self.swa_list)))
        with open(self.queue_list, mode='w') as f:
            for i in range(len(self.ckpt_queue) - 1, -1, -1):
                ckpt_epoch, ckpt_acc = list(self.ckpt_queue[i].items())[0]
                info = '{0}: {1} acc = {2:.6}\n'.format(i + 1, ckpt_epoch, ckpt_acc)
                f.writelines(info)
            if len(self.swa_list) > 0:
                for i in range(len(self.swa_list)):
                    info = 'swa {0} info: {1}\n'.format(i + 1, self.swa_list[i])
                    f.writelines(info)

    def load(
            self,
            ckpt_name,
            module,
            optimizer=None,
            lr_schedule=None
    ):
        load_path = os.path.join(self.ckpt_dir, '{}.ckpt'.format(ckpt_name))
        module_ckpt = torch.load(load_path, map_location=self.device)
        finish_epochs = module_ckpt['epoch']
        module.load_state_dict(module_ckpt['module_param'])
        if optimizer is not None:
            optimizer.load_state_dict(module_ckpt['optimizer'])
        if lr_schedule is not None:
            lr_schedule.load_state_dict(module_ckpt['lr_schedule'])
        return finish_epochs, module, optimizer, lr_schedule

    def swa(
            self,
            epoch_list: List[int],
            swa_name='swa'
    ):
        for i, epoch in enumerate(epoch_list):
            load_path = os.path.join(self.ckpt_dir, '{}.ckpt'.format(epoch))
            model = torch.load(load_path, map_location='cpu')
            model_param = model['module_param']
            finish_epochs = model['epoch']

            if i == 0:
                avg_param = model_param
                sum_epochs = finish_epochs
                optimizer = model['optimizer']
                lr_schedule = model['lr_schedule']
            else:
                sum_epochs += finish_epochs
                for (key, value) in avg_param.items():
                    avg_param[key] = avg_param[key].float()
                    model_param[key] = model_param[key].float()
                    avg_param[key].mul_(i).add_(model_param[key]).div_(i + 1)

        save_path = os.path.join(self.ckpt_dir, swa_name + '.ckpt')
        torch.save({
            'epoch': swa_name,
            'module_param': avg_param,
            'optimizer': optimizer,
            'lr_schedule': lr_schedule
        }, save_path)


class Transformer_WarnUp(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            args
    ):
        self.d_model = args.d_model
        self.warmup_step = args.warmup_step
        super().__init__(optimizer)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_step ** (-1.5))

        return [base_lr * scale for base_lr in self.base_lrs]


class Atom_Embedding(nn.Module):
    def __init__(self, ignore_idx=[], reaction_type=False):
        super(Atom_Embedding, self).__init__()
        self.ignore_idx = ignore_idx
        self.atom_emb_list = [SYMBOL_SIZE] + [DEGREE_SIZE] + [CHARGE_SIZE] + [VALENCY_SIZE] + \
                             [RING_SIZE] + [SUBS_PROD_SIZE] + [HYDRO_SIZE] + [CHIRAL_SIZE] + \
                             [RS_SIZE] + [HYBRID_SIZE]
        if reaction_type:
            self.atom_emb_list = self.atom_emb_list + [10]

    def forward(self, atom: torch.Tensor):
        atom_emb = None
        atom = atom.to(torch.long)
        for i, j in enumerate(self.atom_emb_list):
            if i in self.ignore_idx:
                continue
            else:
                atom_emb = get_one_hot(atom[..., i], j) if atom_emb is None else \
                    torch.cat([atom_emb, get_one_hot(atom[..., i], j)], dim=-1)
        return atom_emb.to(torch.float)


class GAU(AttentionBase):
    '''from Google 'Transformer Quality in Linear Time', require much lower GPU memory, but perform bad in retrosynthesis task'''

    def __init__(
            self, d_model, u, v,
            scale_length=0,
            sin_enc=None,
            pe='none',
            scale_rate=2,
            s=128,
            rel_dict={},
            dropout=0.1,
            dist_range=-1,
            **kwargs
    ):
        super(GAU, self).__init__(
            d_head=s,
            d_rel=s,
            scale_length=scale_length,
            sin_enc=sin_enc,
            pe=pe,
            rel_dict=rel_dict,
            **kwargs
        )
        self.s = s
        self.d_model = d_model
        self.d_ff = int(scale_rate * d_model)
        self.dist_range = dist_range
        self._build_model(
            u=u,
            v=v,
            dropout=dropout
        )

    def _build_model(self, u, v, dropout=0.1):
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.Tensor(2, self.s))
        self.beta = nn.Parameter(torch.Tensor(2, self.s))
        self.W_u = nn.Sequential(nn.Linear(self.d_model, self.d_ff), nn.SiLU())
        self.W_v = nn.Sequential(nn.Linear(self.d_model, self.d_ff), nn.SiLU())
        self.W_qk = nn.Sequential(nn.Linear(self.d_model, self.s), nn.SiLU())
        self.W_o = nn.Linear(self.d_ff, self.d_model)
        self.u = u  # size = (d_model,), nn.ParameterDict
        self.v = v  # size = (d_model,), nn.ParameterDict

    def _init_param(self, vo_beta: float):
        nn.init.xavier_normal_(self.gamma, gain=1)
        nn.init.zeros_(self.beta)
        nn.init.xavier_normal_(self.W_u[0].weight, gain=vo_beta)
        nn.init.xavier_normal_(self.W_v[0].weight, gain=vo_beta)
        nn.init.xavier_normal_(self.W_qk[0].weight, gain=1)
        nn.init.xavier_normal_(self.W_o.weight, gain=vo_beta)

    def forward(
            self,
            x: torch.Tensor,
            length: List[torch.Tensor],
            context=None,
            keys_size=None,
            dist=None,
            deg=None,
            rel=None,
            edge=None,
            bias=None,
            future=False,
            step=0
    ):
        if self.dist_range >= 0 and dist is not None:
            dist_range_mask = dist.gt(self.dist_range)
        else:
            dist_range_mask = None

        key_size = x.size(1) if context is None else context.size(1)
        mask = self._compute_mask(
            valid_len=length,
            batch_size=x.size(0),
            q_size=x.size(1),
            k_size=keys_size if keys_size is not None else [key_size],
            future=future,
            device=x.device,
            multi_head=False,
            dist_range_mask=dist_range_mask
        )

        if context is not None:
            u, v = self.W_u(x), self.W_v(context)
            q, k = torch.split(self.W_qk(torch.cat([x, context], dim=1)),
                               [x.size(1), context.size(1)], dim=1)
            q = torch.einsum('...d,d->...d', q, self.gamma[0]) + self.beta[0]
            k = torch.einsum('...d,d->...d', k, self.gamma[1]) + self.beta[1]
            if self.pe == 'rope':
                q, k = self._apply_rope([q, k], [step, 0])
        else:
            u, v = self.W_u(x), self.W_v(x)
            qk_cat = self.W_qk(x)
            qk_cat = torch.einsum('...d,hd->...hd', qk_cat, self.gamma) + self.beta
            q, k = torch.unbind(qk_cat, dim=-2)
            if self.pe == 'rope':
                q, k = self._apply_rope([q, k], [step, 0])

        q = self._sim_scale(q)
        if self.rel_emb is None:
            qk = torch.matmul(q, k.transpose(1, 2))
        else:
            qk = torch.matmul(q + self.u['qk'], k.transpose(1, 2))
            if dist is not None:
                if 'vnode_dist' in self.rel_emb.keys():
                    rnode_dist = self.rel_emb['dist'](dist[:, 1:, 1:])
                    vnode_dist_row, vnode_dist_col = self.rel_emb['vnode_dist'](dist[:, 0].unsqueeze(1)), \
                        self.rel_emb['vnode_dist'](dist[:, :, 0].unsqueeze(2))
                    dist = torch.cat([vnode_dist_row[:, :, 1:], rnode_dist], dim=1)
                    dist = torch.cat([vnode_dist_col, dist], dim=2)
                else:
                    dist = self.rel_emb['dist'](dist)
                qk = qk + torch.einsum('bqd,bqdk->bqk', q + self.v['dist'], dist.transpose(-1, -2))

            if deg is not None:
                if 'vnode_deg' in self.rel_emb.keys():
                    deg = torch.cat([self.rel_emb['vnode_deg'](deg[:, 0].unsqueeze(-1)), \
                                     self.rel_emb['deg'](deg[:, 1:])], dim=1)
                else:
                    deg = self.rel_emb['deg'](deg)
                qk = qk + torch.matmul(q + self.v['deg'], deg.transpose(-1, -2))

            if rel is not None:
                rel = self.rel_emb['rel'](rel)  # (q_len, k_len, s)
                qk = qk + torch.einsum('bqd,qdk->bqk',
                                       q + self.v['rel'], rel.transpose(-1, -2))

            if edge is not None:
                qk = qk + edge.squeeze(-1)

        if bias is not None:
            qk = qk + bias  # (batch, q_len, k_len)

        qk.masked_fill_(mask, -float('inf'))
        qk = torch.softmax(qk, dim=-1)
        qk_drop = self.dropout(qk)
        output = u * torch.matmul(qk_drop, v)
        return self.W_o(output)


class GLU(nn.Module):
    '''Gated Linear Unit, many ablation test.txt shows it has better performance in the same param num compare to vanilla FFN'''

    def __init__(
            self, d_model,
            d_ff,
            d_out=0,
            dropout=0.1,
            **kwargs
    ):
        super(GLU, self).__init__(**kwargs)
        d_out = d_out if d_out > 0 else d_model
        self.d_ff = d_ff
        self.W_uv = nn.Linear(d_model, self.d_ff * 2)
        self.Act_u = nn.ReLU()
        self.W_o = nn.Linear(self.d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def _init_param(self, vo_beta: float):
        for param in self.parameters():
            if param.dim() > 1 and param.requires_grad:
                nn.init.xavier_normal_(param, gain=vo_beta)

    def forward(self, x: torch.Tensor):
        uv = self.W_uv(x)
        u, v = torch.split(uv, [self.d_ff, self.d_ff], dim=-1)
        u = self.Act_u(u)
        output = self.dropout(u * v)
        output = self.W_o(output)
        return output


class RMSNorm(nn.Module):
    '''RMSNorm is from 'Root Mean Square layer normalization', may have a better performance in some cases
        compare to vanilla layer norm'''

    def __init__(
            self,
            d_model,
            eps=1e-6,
            **kwargs
    ):
        super(RMSNorm, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.Tensor(d_model))
        self.eps = eps
        self._init_param()

    def _init_param(self):
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor):
        mean_square = torch.square(x).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps) * self.weight
        return x


class Focal_loss(nn.Module):
    def __init__(
            self,
            # vocab_freq: List,  # the frequency of token, the size must match the f_vocab
            # vocab_count: List,  # the occur count of token, the size must match the f_vocab
            vocab_dict: Dict,
            ignore_index=-100,
            ignore_min_count=10,
            label_smooth=0.1,
            # the max scale rate of each token, use it to generate token weight
            max_scale_rate=10.0,
            gamma=2.0,
            # for sentence, if p(token) > margin, it will be regard as correct
            margin=0.6,
            # if a sentences' all p(token) > margin, this sentences' loss will be sum(token_loss) * sentence_scale
            sentence_scale=0.1,
            k=1e6,
            device='cpu',
            loss_type = 'focal'
    ):
        super(Focal_loss, self).__init__()
        # self.vocab_freq = np.array(vocab_freq)
        # self.vocab_count = np.array(vocab_count)
        self.vocab_dict = vocab_dict
        self.vocab_size = len(vocab_dict)
        self.ignore_min_count = ignore_min_count
        self.label_smooth = label_smooth
        self.max_scale_rate = max_scale_rate
        self.gamma = gamma
        self.margin = margin
        self.sentence_scale = sentence_scale
        self.ignore_index = ignore_index
        self.k = k
        self.device = device

        # self._generate_token_weight()
        self.loss_type = loss_type
        if self.loss_type == 'CE':
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index,
                reduction="mean")
        elif self.loss_type == 'Mixed-CE':
            self.criterion = LabelSmoothing_MCELoss(reduction="mean",
                                                         ignore_index=self.ignore_index)

    def _cross_entropy(
            self,
            input_log_prob: torch.Tensor,
            target: torch.Tensor
    ):
        if self.label_smooth > 0.:
            t_prob, f_prob = 1.0 - self.label_smooth, \
                self.label_smooth
            f_idx = torch.arange(self.vocab_size,
                                 dtype=torch.long, device=self.device)[None, None, :].ne(target[:, :, None])
            ignore_idx = target.ne(self.ignore_index)
            loss = F.nll_loss(input_log_prob, target,
                              ignore_index=self.ignore_index, reduction='none')
            loss = loss * t_prob + (-input_log_prob.transpose(1, 2)[f_idx]). \
                reshape(loss.size(0), loss.size(1), -1).mean(dim=-1) * f_prob
            loss = loss * ignore_idx
        else:
            loss = F.nll_loss(input_log_prob, target,
                              ignore_index=self.ignore_index, reduction='none')
        return loss

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
            seq_len: torch.Tensor,
            epoch,
            tot_epochs
    ):
        occur_token = target.reshape(-1)
        occur_token_idx = [[_ for _ in range(target.size(0) * target.size(1))], occur_token.tolist()]
        occur_token_prob = input.softmax(dim=1).transpose(1, 2). \
            reshape(-1, self.vocab_size)
        occur_token_prob = occur_token_prob[occur_token_idx]. \
            reshape(target.size(0), target.size(1))
        max_token_prob = input.softmax(dim=1).transpose(1, 2).max(dim=-1)[0]

        pred_token = input.log_softmax(dim=1).argmax(dim=1)
        pred_true, pred_false = pred_token.eq(target), pred_token.ne(target)

        true_token = target[pred_true].reshape(-1)
        all_token_count = scatter(torch.ones_like(occur_token), index=occur_token, dim=0,
                                  dim_size=self.vocab_size, reduce='sum')
        true_token_count = scatter(torch.ones_like(true_token), index=true_token, dim=0,
                                   dim_size=self.vocab_size, reduce='sum')
        all_token_count[self.ignore_index] = 0
        true_token_count[self.ignore_index] = 0

        mask = torch.arange((target.size(-1)), dtype=torch.long, device=input.device)
        mask = mask[None, :] >= seq_len[:, None]

        if self.loss_type == 'focal':
            if self.max_scale_rate > 0.:
                loss = self._cross_entropy(input_log_prob=input.log_softmax(dim=1), target=target)
                token_weight = self.weight[target]
                loss = loss * token_weight
            else:
                loss = self._cross_entropy(input_log_prob=input.log_softmax(dim=1), target=target)

            if self.gamma > 0.:
                loss = loss * ((1 - occur_token_prob) ** self.gamma)

            if self.sentence_scale < 1.0:
                sentence_weight = torch.sigmoid(self.k * (self.margin - occur_token_prob)).masked_fill(mask, 0).max(dim=1)[
                    0]
                sentence_weight = torch.clip(sentence_weight, min=self.sentence_scale, max=1)
                loss = loss * sentence_weight.unsqueeze(1)
        elif self.loss_type == 'CE':
            loss = self.criterion(
                input=input,
                target=target
            )
        else:
            loss =  self.criterion(
                input=input,
                target=target,
                epoch=epoch, tot_epochs=tot_epochs, use_mix_CE=True
            )

        return loss, true_token_count.tolist(), all_token_count.tolist()


# for one2many prediction
class LabelSmoothing_MCELoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean', ignore_index=None):
        super(LabelSmoothing_MCELoss, self).__init__()
        # self.smoothing = smoothing
        # self.reduction = reduction
        # self.log_softmax = nn.LogSoftmax(dim=-1)
        # self.apply_logsoftmax = apply_logsoftmax
        # self.ignore_idx = ignore_index
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing= smoothing
        )

    def forward(self, input, target, epoch, tot_epochs, use_mix_CE, alpha=0.5):
        ce_loss = self.ce(input, target)
        self_loss = self.ce(input, torch.argmax(input, dim=1))
        loss = ce_loss
        if use_mix_CE:
            p = epoch / tot_epochs * alpha
            loss = (1 - p) * ce_loss + p * self_loss
        return loss


def coverage_plot(seq_coverage: dict, ckpt_dir: str, ckpt_name, args):
    plt.rcParams['figure.dpi'] = 600
    plt.plot(seq_coverage.keys(), seq_coverage.values(),'s-', color='r', label='coverage', linewidth=0.25, markersize=2.0)
    plt.xlabel('1-N number')
    plt.ylabel('avg coverage')
    plt.legend(loc='best')
    plt.title(f'average coverage of per 1-N reaction')
    saved_png_file = os.path.join(ckpt_dir, ckpt_name+f'coverage rate.png')
    if os.path.exists(saved_png_file):
        os.remove(saved_png_file)
    plt.savefig(saved_png_file)
    plt.close()
    saved_txt_file = os.path.join(ckpt_dir, ckpt_name+f'T-{args.T}-coverage rate.txt')
    with open(saved_txt_file, 'w') as in_txt:
        for key, value in seq_coverage.items():
            in_txt.write(f'{key}:\t{value}\n')




