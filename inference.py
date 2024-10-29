import os
import sys
import os.path as osp
import time

sys.path.insert(0, osp.dirname(os.path.abspath(__file__)))

from rdkit import Chem
import numpy as np
import torch

from models.graph_rel_transformers import Graph_Transformer_Base
from utils.parsing import get_parser, post_setting_args
from utils.chem_tools import NODE_FDIM, BOND_FDIM
from utils.wrap_single_smi import InferBatch


def BiG2S_Inference(args, pretrained_path, input_smi, rxn_type):
    Infer_wrapper = InferBatch('./preprocessed', args)
    batch_graph_input = Infer_wrapper.preprocess(input_smi, rxn_type)
    if args.use_reaction_type:
        dec_cls = 2
    else:
        dec_cls = 1
    predict_module = Graph_Transformer_Base(
        f_vocab=len(Infer_wrapper.token_index),
        f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
        f_bond=BOND_FDIM,
        token_idx=Infer_wrapper.token_index,
        token_freq=None,
        token_count=None,
        cls_len=dec_cls,
        args=args
    )

    ckpt_file = osp.join(pretrained_path, 'big2s_rxn.ckpt')
    module_ckpt = torch.load(ckpt_file, map_location=args.device)
    predict_module.model.load_state_dict(module_ckpt['module_param'])

    with torch.no_grad():
        batch_data = batch_graph_input.to(args.device)
        predict_result, predict_scores = predict_module.model_predict(
            data=batch_data,
            args=args
        )

    smi_nodes_sorted, prob_nodes_sorted = Infer_wrapper.post_process(predict_result, predict_scores)

    return smi_nodes_sorted, prob_nodes_sorted


if __name__ == '__main__':

    parser = get_parser(mode='test')
    args = post_setting_args(parser)
    args.use_reaction_type = True
    args.beam_size = 10

    # 需要手动更改的地方： 1. dataset_name, 2. chekpoint_path, 3. 产物分子
    # 用哪个数据集的 checkpoint 和 词表，就写成哪个
    args.dataset_name = 'pistachio'
    # 调整 beam search 里的 温度系数
    args.T = 1.6
    assert args.dataset_name in ['debug_uspto_50k', 'pistachio', 'uspto_diverse', 'uspto_50k_infer']
    assert args.T in [0.7, 1.1, 1.6]
    pretrained_path = 'checkpoints/pistachio'

    #***************************#
    # input_smi = 'N # C c 1 n n ( - c 2 c ( C l ) c c ( C ( F ) ( F ) F ) c c 2 C l ) c c 1 C ( B r ) = C ( C l ) B r'.replace(' ', '')  # demo_a
    input_smi = 'O=Cc1ccc(OCc2ccccc2)c(OCc2ccccc2)c1'  # demo of pistachio

    # 为了方便，用数字代表反应类型序号
    rxn_type = 5
    #***************************#

    start = time.perf_counter()
    top_k, score = BiG2S_Inference(args, pretrained_path, input_smi, rxn_type)
    end = time.perf_counter()
    print(top_k)
    print(score)
    print('推理时间: %s 秒' % (end - start))

