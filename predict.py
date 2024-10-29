import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim as optm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm.std import trange


from utils.parsing import get_parser, post_setting_args
from utils.chem_tools import NODE_FDIM, BOND_FDIM

from utils.data_loader import G2SFullDataset, S2SFullDataset

from models.graph_rel_transformers import Graph_Transformer_Base
from models.graph_vae_transformers import Graph_Transformer_VAE
from models.seq_rel_transformers import Seq_Transformer_Base
from models.seq_vae_transformers import Seq_Transformer_VAE

from models.module_utils import Model_Save, eval_plot, beam_result_process
from utils.preprocess_smi import canonicalize_smiles

# 分布式训练，数据并行
from torch.utils.data.distributed import DistributedSampler
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import time
import pandas as pd

proj_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(proj_dir, 'preprocessed')
origin_dir = os.path.join(proj_dir, 'data')

def evaling(args):

    if args.use_subs and (args.use_reaction_type or args.model_type == 'BiG2S_HCVAE'):
        dec_cls = 2
    elif args.use_subs or args.use_reaction_type or args.model_type == 'BiG2S_HCVAE':
        dec_cls = 1
    else:
        dec_cls = 0

    if args.representation_form == 'graph2smiles':
        eval_dataset = G2SFullDataset(
            output_dir=output_dir,
            dataset_name=args.dataset_name,
            split_type=args.split_type,
            batch_size=args.batch_size,
            token_limit=args.token_limit,
            mode=args.mode,
            dist_block=args.graph_dist_block,
            task=args.eval_task,
            use_split=args.use_splited_data,
            split_data_name=args.split_data_name
        )
    else:
        eval_dataset = S2SFullDataset(
            output_dir=output_dir,
            dataset_name=args.dataset_name,
            split_type=args.split_type,
            batch_size=args.batch_size,
            token_limit=args.token_limit,
            mode=args.mode,
            dist_block=args.graph_dist_block,
            task=args.eval_task,
            use_split=args.use_splited_data,
            split_data_name=args.split_data_name
        )

    ckpt_dir = os.path.join('checkpoints', args.save_name)
    token_idx = eval_dataset.token_idx
    module_saver = Model_Save(
        ckpt_dir=ckpt_dir,
        device=args.device,
        save_strategy=args.save_strategy,
        save_num=args.save_num,
        swa_count=args.swa_count,
        swa_tgt=args.swa_tgt,
        const_save_epoch=args.const_save_epoch,
        top1_weight=args.top1_weight
    )
    if args.model_type == 'BiG2S':
        module = Graph_Transformer_Base(
            f_vocab=len(token_idx),
            f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
            f_bond=BOND_FDIM,
            token_idx=token_idx,
            token_freq=eval_dataset.token_freq,
            token_count=eval_dataset.token_count,
            cls_len=dec_cls,
            args=args
        )
    elif args.model_type == 'BiG2S_HCVAE':
        module = Graph_Transformer_VAE(
            f_vocab=len(token_idx),
            f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
            f_bond=BOND_FDIM,
            token_idx=token_idx,
            token_freq=eval_dataset.token_freq,
            token_count=eval_dataset.token_count,
            cls_len=dec_cls,
            args=args
        )
    elif args.model_type == 'S2S_HCVAE':
        module = Seq_Transformer_VAE(
            f_vocab=len(token_idx),
            token_idx=token_idx,
            token_freq=eval_dataset.token_freq,
            token_count=eval_dataset.token_count,
            cls_len=dec_cls,
            args=args
        )
    else:  # pure transformer : S2S
        module = Seq_Transformer_Base(
            f_vocab=len(token_idx),
            token_idx=token_idx,
            token_freq=eval_dataset.token_freq,
            token_count=eval_dataset.token_count,
            cls_len=dec_cls,
            args=args
        )

    for ckpt_name in args.ckpt_list:
        _, module.model, _, _ = module_saver.load(ckpt_name, module.model)
        beam_size = args.beam_size
        seq_acc_count = np.zeros((args.return_num))
        seq_invalid_count = np.zeros((args.return_num))
        reaction_acc_count = np.zeros((10))
        smi_predictions = []
        smi_references = []

        with torch.no_grad():
            eval_dataset.get_batch()
            data_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=1,
                shuffle=True,
                collate_fn=lambda _batch: _batch[0],
                pin_memory=True
            )
            teval = trange(eval_dataset.batch_step)
            for step, batch in zip(teval, data_loader):
                batch = batch.to(args.device)

                # log inference time
                start = time.perf_counter()

                predict_result, predict_scores = module.model_predict(
                    data=batch,
                    args=args
                )

                end = time.perf_counter()

                print(f"total time of inference = {end - start}s")
                print(f"avg time of inference = {(end - start) / args.batch_size}s")

                # transform to smi
                beam_acc, beam_invalid, beam_smis, tgt_smis = beam_result_process(
                    tgt_seq=batch.tgt_seq,
                    tgt_len=batch.tgt_seq_len,
                    token_idx=token_idx,
                    beam_result=predict_result,
                    beam_scores=predict_scores
                )
                # beam_smi: [b_sz, beam_size]

                seq_acc_count = seq_acc_count + beam_acc
                seq_invalid_count = seq_invalid_count + beam_invalid
                smi_predictions.extend(beam_smis)
                smi_references.extend(tgt_smis)

                teval.set_description('evaling......')

            seq_acc_count = np.cumsum(seq_acc_count)
            seq_invalid_count = np.cumsum(seq_invalid_count)
            seq_acc_count = seq_acc_count / eval_dataset.data_len
            seq_invalid_count = seq_invalid_count / np.array([i * eval_dataset.data_len for i in range(1, args.return_num + 1, 1)])

            eval_plot(
                topk_seq_acc=seq_acc_count.tolist(),
                topk_seq_invalid=seq_invalid_count.tolist(),
                beam_size=beam_size,
                data_name=args.dataset_name,
                ckpt_dir=ckpt_dir,
                ckpt_name=ckpt_name,
                args=args,
                is_train=False
            )

        # save the results
        """
        1. 读取原始test文件
        2. 做成pandas_dataframe
        3. 创建result文件夹，把dataframe转成csv放进去
        """
        ori_map_dict = {}
        ori_data_file = os.path.join(f'./preprocessed/{args.dataset_name}', 'test.csv')
        ori_rxn_pd = pd.read_csv(ori_data_file)
        for i in range(ori_rxn_pd.shape[0]):
            cano_react = ori_rxn_pd.iloc[i]['substrates_token'].replace(' ', '')
            cano_prod = ori_rxn_pd.iloc[i]['product_token'].replace(' ', '')
            # cano_react = canonicalize_smiles(react, map_clear=True)
            # cano_prod = canonicalize_smiles(prod, map_clear=True)
            ori_map_dict[cano_react] = cano_prod

        mismatch_cnt = 0
        retro_result_dict = {}
        for i in range(len(smi_references)):

            cano_react = smi_references[i]
            if cano_react not in ori_map_dict.keys():
                mismatch_cnt += 1
                ori_map_dict[cano_react] = ''
            retro_result_dict[cano_react] = {'prediction': smi_predictions[i],
                                             'product': ori_map_dict[cano_react]}
        print(f"mismatch_cnt = {mismatch_cnt}")

        # 假设符合要求，做成dataframe
        res_reacts = []
        res_prods = []
        res_preds = []
        for key in retro_result_dict.keys():
            res_reacts.append(key)
            res_prods.append(retro_result_dict[key]['product'])
            res_preds.append(retro_result_dict[key]['prediction'])
        retro_res = pd.DataFrame({'products': res_prods, 'reactants': res_reacts, 'predictions': res_preds})

        save_dir = 'results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        target_file = os.path.join(save_dir, f'{args.save_name}-t{args.T}-beam-{args.beam_size}-4rt.csv')
        retro_res.to_csv(target_file, index=False)


if __name__ == '__main__':
    parser = get_parser(mode = 'test')
    args = post_setting_args(parser)

    evaling(args)





