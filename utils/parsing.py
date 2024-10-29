import argparse
import logging
import os
import requests
import argparse
import zipfile

import numpy as np
import random
import torch

# modified by Graph2SMILES: https://github.com/coleygroup/Graph2SMILES

def log_args(args):
    logging.info(f"Logging arguments")
    for k, v in vars(args).items():
        logging.info(f"**** {k} = *{v}*")


def get_parser(mode='train'):
    assert mode in ['train', 'eval', 'test', 'swa', 'preprocess']
    parser = argparse.ArgumentParser(description = mode)
    add_common_args(parser, mode)
    if mode != 'preprocess':
        # TODO: 实现对应函数
        get_module_args(parser)
        get_train_args(parser)
        get_beam_args(parser)
        get_eval_args(parser)
        if mode in ['swa']:
            get_swa_args(parser)
    elif mode == 'preprocess':
        add_preprocess_args(parser)
    return parser


def add_common_args(parser, mode):
    #  s2s: vanilla transformer
    #  g2s_base: Graph2SMILES
    #  g2s_vae: RetroDCVAE / RetroHCVAE
    #  big2s: dual task Graph2SMILES
    group = parser.add_argument_group("Meta")
    group.add_argument("--model", help="Model architecture",
                       choices=["s2s", "g2s_base", "g2s_vae", "big2s"], type=str, default="")
    group.add_argument("--dataset_name", help="Data name", type=str,
                       choices=["uspto_50k", "debug_uspto_50k", "uspto_mit", "uspto_full", "uspto_diverse", "pistachio", "extension"],
                       default="debug_uspto_50k")
    group.add_argument("--task", help="Task", choices=["reaction_prediction", "retrosynthesis", "autoencoding"],
                       type=str, default="retrosynthesis")

    group.add_argument("--representation_form", help="Final string representation to be fed",
                       choices=["graph2smiles", "smiles2smiles"], type=str, default="graph2smiles")

    group.add_argument("--seed", help="Random seed", type=int, default=17)

    group.add_argument("--split_type", help="how to tokenize SMILES", type=str, default='token')
    # BiG2S 没用到
    group.add_argument("--max_src_len", help="Max source length", type=int, default=512)
    group.add_argument("--max_tgt_len", help="Max target length", type=int, default=512)
    group.add_argument("--num_workers", help="No. of workers", type=int, default=1)
    group.add_argument("--verbose", help="Whether to enable verbose debugging", action="store_true")

    group = parser.add_argument_group("Paths")
    group.add_argument("--log_file", help="Preprocess log file", type=str, default="")
    group.add_argument("--vocab_file", help="Vocab file", type=str, default="")
    group.add_argument("--preprocess_output_path", help="Path for saving preprocessed outputs",
                       type=str, default="")
    group.add_argument("--save_dir", help="Path for saving checkpoints", type=str, default="")

    group.add_argument("--request", help="Path for saving checkpoints", type=str, default=f"{mode}")
    group.add_argument("--use_ddp", help="Parallel training ?", type=bool, default=False)


def add_preprocess_args(parser):
    group = parser.add_argument_group("Preprocessing options")
    # data paths
    # group.add_argument("--train_src", help="Train source", type=str, default="")
    # group.add_argument("--train_tgt", help="Train target", type=str, default="")
    # group.add_argument("--val_src", help="Validation source", type=str, default="")
    # group.add_argument("--val_tgt", help="Validation target", type=str, default="")
    # group.add_argument("--test_src", help="Test source", type=str, default="")
    # group.add_argument("--test_tgt", help="Test target", type=str, default="")
    # # options
    # group.add_argument("--representation_start", help="Initial string representation to be fed",
    #                    choices=["smiles"], type=str, default="")
    # group.add_argument("--do_tokenize", help="Whether to tokenize the data files", action="store_true")
    # group.add_argument("--make_vocab_only", help="Whether to only make vocab", action="store_true")

    # for BiG2S
    group.add_argument('--raw_csv_preprocess', help='preprocess raw csv file', action='store_true', default=True)
    group.add_argument('--graph_preprocess', help='preprocess token to graph', action='store_true', default=True)
    group.add_argument('--split_preprocess', help='split eval data to smaller and accelerate evaling during training',
                       action='store_true', default=True)
    group.add_argument('--split_len', help='the data length after spliting eval data', type=int, default=10000)
    group.add_argument('--split_name', help='the suffix of eval split data', type=str, default='split')
    group.add_argument('--need_atom', help='using smiles token to generate chemical feature', action='store_true',
                       default=False)
    group.add_argument('--need_graph', help='using smiles token to generate molecule graph', action='store_true',
                       default=True)
    group.add_argument('--self_loop', help='add self-loop connect to each molecule atom', action='store_true',
                       default=False)
    group.add_argument('--vnode', help='add virtual node to molecule graph, which connect to all of the atom',
                       action='store_true', default=False)


def get_module_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('module')
    group.add_argument('--batch_size', help='the product molecule graph num for each step(same during training and evaling)', default=256)
    group.add_argument('--token_limit', help='the maximun token number of product+substrate for each step(same during training and evaling)', default=0)
    group.add_argument('--d_model', help='hidden size of module', type=int, default=256)
    group.add_argument('--d_ff', help='hidden size of feed-forward-network', type=int, default=256 * 6)
    group.add_argument('--enc_head', help='attention head of transformer encoder', type=int, default=8)
    group.add_argument('--dec_head', help='attention head of transformer decoder', type=int, default=8)
    group.add_argument('--graph_layer', help='layer size of D-MPNN', type=int, default=4)
    group.add_argument('--enc_layer', help='layer size of transformer encoder', type=int, default=6)
    group.add_argument('--dec_layer', help='layer size of transformer decoder', type=int, default=6)
    group.add_argument('--dropout', help='dropout rate of module', type=float, default=0.3)
    group.add_argument('--use_subs', help='use reaction-prediction task to help retrosynthesis, if True, the real batch_size will be a double size',
                       action='store_true', default=False)
    group.add_argument('--use_reaction_type', help='use reaction type label to help module get higher performance', action='store_true', default=False)
    group.add_argument('--decoder_cls', help='if True, in the front of decoder BOS token will have extra cls token(subs/prob or reaction_type)', default=True)
    group.add_argument('--graph_dist_block', help='the node distance block for embedding, if any distance not in this block,\
                                                   it will be included into an extra embedding',
                        type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, [8, 15], [15, 2048]])
    group.add_argument('--device', help='the device for module running', type=str, default='cuda')

    group.add_argument('--model_type', help='which model to use? BiG2S, + HCVAE?', type=str,
                       choices=['BiG2S', 'BiG2S_HCVAE'], default='BiG2S_HCVAE')
    group.add_argument('--lat_disc_size', help='size of discrete valued latent variable', type=int, default=90)
    group.add_argument('--temp', help='temperature in gumbel softmax sampling', type=float, default=1.0)


def get_train_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('training')
    group.add_argument('--train_task', help='retrosynthesis or reaction prediction task for training', choices=['prod2subs', 'subs2prod', 'bidirection'], default='prod2subs')
    group.add_argument('--save_strategy', help='the checkpoint save strategy during training, if (top), it will keep checkpoint according to top1-acc;\
                                                  if (mean), it will keep checkpoint according to (top1 + mean(top3 + top5 + top10)); if (last), it will\
                                                  keep the lastest checkpoint according to epoch', type=str, choices=['top', 'mean', 'last'], default='mean')
    group.add_argument('--save_num', help='the max checkpoint num', type=int, default=10)
    group.add_argument('--swa_count', help='it will generate a average checkpoint when the save count reach the swa_count each time', type=int, default=10)
    group.add_argument('--swa_tgt', help='when reach the swa_count, it will average the top-swa_tgt checkpoint', type=int, default=5)
    group.add_argument('--const_save_epoch', help='when a checkpoint is in here, it will be keeping permanently although it already out of the save queue',
                       nargs='*', type=int, default=[])

    group.add_argument('--epochs', help='the total epochs to finish training', type=int, default=700)
    group.add_argument('--memory_clear_count', help='pytorch memory clear count in each epoch', type=int, default=0)
    group.add_argument('--eval', help='if True, module will evaling during training', action='store_true', default=True)
    group.add_argument('--save_epoch', help='when reaching these epoch, module will be saved and eval',
                       nargs='+', type=int, default=[_ for _ in range(5)])            # default=[_ for _ in range(154, 600, 5)] + [600]
    group.add_argument('--eval_epoch', help='in these epoch, module will evaling but not saving', nargs='*', type=int, default=[])
    group.add_argument('--token_eval_epoch', help='in these epoch, module will generate a file about each tokens correct rate', nargs='*', type=int, default=[])
    group.add_argument('--accum_count', help='the gradient update accum count', type=int, default=2)

    group.add_argument('--optimizer', help='the optimizer name for training', type=str, choices=['AdamW'], default='AdamW')
    group.add_argument('--lr', help='the basic learning rate scale for lr_schedule(in original, it will be a scale rate for real lr; in cosine, it will be the maximum lr)', type=float, default=1.0)
    group.add_argument('--betas', help='the betas for AdamW', nargs=2, action='append', type=int, default=[0.9, 0.999])
    group.add_argument('--eps', help='the eps for AdamW', type=float, default=1e-6)
    group.add_argument('--weight_decay', help='the weight_decay rate for AdamW', type=float, default=0.0)
    group.add_argument('--clip_norm', help='the gradient clip maximum norm, if <= 0.0, it will skip the gradient clip', type=float, default=0.0)

    group.add_argument('--ignore_min_count', help='if token occur count less than min_count, it will be ignore when calculating token weight',
                       type=int, default=100)
    group.add_argument('--label_smooth', help='the prob for negative label loss calculation', type=float, default=0.1)
    group.add_argument('--max_scale_rate', help='for the focal loss, the maximun scale rate for the less token', type=float, default=0.)
    group.add_argument('--gamma', help='the index number of difficulty weight according to the correct rate of each token in a minibatch',
                       type=float, default=2.0)
    group.add_argument('--margin', help='the rate for sentence weight calculate, if prob > margin, the result will be treated as True', type=float, default=0.85)
    group.add_argument('--sentence_scale', help='the loss ratio for sentence which is correct for all of the token', type=float, default=0.)

    group.add_argument('--warmup_step', help='the step to reach the maximum learning rate', type=int, default=8000)
    group.add_argument('--lr_schedule', help='the schedule to scale learning rate', type=str, choices=['original', 'cosine'], default='original')
    group.add_argument('--min_lr', help='when lr <= min_lr, the lr will equal to min_lr(only available in cosine lr_schedule)', type=float, default=1e-5)

    # for hmcvae
    group.add_argument('--loss_type', help='which loss to use? focal, ce, mixed-ce?', type=str, choices=['focal', 'CE', 'Mixed-CE'], default='CE')
    group.add_argument('--vae_warmup', help='the step to reach the maximum learning rate', type=int, default=10)

    group.add_argument('--cvae_type', help='hierarchical modeling latent space?', type=str,
                       choices=['hcvae', 'dcvae'], default='hcvae')
    group.add_argument('--load_ckpt', help='intermediate checkpoint file', type=str, default="")

def get_beam_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('beam_search')
    group.add_argument('--beam_module', help='use huggingface of OpenNMT model to running beam search, huggingface may slower 3 or more times than OpenNMT,\
                                                but OpenNMT has some bugs which cause different batch size generate different result, and also a lower accuracy.',
                       type=str, choices=['huggingface', 'OpenNMT'], default='huggingface')
    group.add_argument('--beam_size', help='the beam size for each latent variable group during prediction, because of duplicate predictions in different latent group, >=10 setting is suggested',
                       type=int, default=10)
    group.add_argument('--return_num', help='the return predictions number for each batch after beam search', type=int, default=10)
    group.add_argument('--max_len', help='the maximum length for smiles prediction', type=int, default=256)
    group.add_argument('--T', help='the tempreture for prediction log_softmax, T = 1.3 will improve performance in some cases', type=float, default=1.0)
    group.add_argument('--k_sample', help='the top-k sample setting, 0 will close top-k sample', type=int, default=0)
    group.add_argument('--p_sample', help='the top-p sample setting, 0 will close top-p sample', type=float, default=0.)
    group.add_argument('--top1_weight', help='the weight for top-1 accuracy in weighted accuracy scores', type=float, default=0.9)


def get_eval_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('evaling')
    group.add_argument('--mode', help='using eval dataset or test dataset to generate prediction', choices=['eval', 'test'], default='test')
    group.add_argument('--eval_task', help='retrosynthesis or reaction prediction task for evaling', choices=['prod2subs', 'subs2prod'], default='prod2subs')
    group.add_argument('--ckpt_list', help='use a loop to eval the checkpoint inside this list', nargs='+', type=str, default=['swa9'])
    group.add_argument('--use_splited_data', help='use the splited data when evaling', action='store_true', default=False)
    group.add_argument('--split_data_name', help='the suffix of splited data, the default setting is (split_10000)', type=str, default='split_10000')

    parser.add_argument('--download_checkpoint', action='store_true', default=False)


def get_swa_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('swa')
    group.add_argument('--average_list', help='use these checkpoint to generate an average model, it will have much better performance usually(especially in top3-10)',
                       nargs='+', type=str, default=[])
    group.add_argument('--average_name', help='the save name of this average checkpoint', type=str, default='swa')


def download_checkpoint(dataset: str):
    checkpoint_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(checkpoint_dir, 'model')
    checkpoint_dir = os.path.join(checkpoint_dir, 'check_point')
    checkpoint_dir = os.path.join(checkpoint_dir, dataset)

    if dataset == '50k':
        url = 'https://www.dropbox.com/scl/fo/vxbhzwbi38khsfqj0ed1g/h?dl=0&rlkey=b6ve5agochkgx29112kcqo5fu'
    elif dataset == '50k_class':
        url = 'https://www.dropbox.com/scl/fo/sum8joi7o79ktqb4xptt5/h?dl=0&rlkey=d6bncggto2i4kxrgtj71budud'
    elif dataset == 'uspto_mit':
        url = 'https://www.dropbox.com/scl/fo/rjt5578efwnac6ziyx4hm/h?dl=0&rlkey=7bp7m5vpvvj0u1x1wlonqdypo'
    elif dataset == 'full':
        url = 'https://www.dropbox.com/scl/fo/4x88221gk5oju5jgblbp6/h?dl=0&rlkey=7yb1d0vgxhhyz0r8zz93lsejf'

    checkpoint_path = os.path.join(checkpoint_dir, 'data.zip')
    if not os.path.exists(checkpoint_path):
        r = requests.get(url, stream=True)
        with open(checkpoint_path, 'wb') as f:
            for i in r.iter_content(chunk_size=128):
                f.write(i)
        with zipfile.ZipFile(checkpoint_path) as f:
            f.extractall(path=checkpoint_dir)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def post_setting_args(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    if args.request == 'preprocess':
        args.raw_csv_preprocess = True
        args.graph_preprocess = True
        args.split_preprocess = False if args.dataset_name in ["debug_uspto_50k", 'uspto_50k', '50k_class', 'uspto_diverse'] else True
        args.split_len = 10000

    elif args.request == 'train':
        args.save_name = args.dataset_name + '-' + args.model_type  + '-' + args.beam_module + '-' + args.train_task + '-' + args.loss_type
        if args.model_type == 'BiG2S_HCVAE':
            args.save_name += '-lat_sz' + str(args.lat_disc_size)

        args.accum_count = 2
        if args.train_task == 'prod2subs':
            args.use_subs = False
        else:
            args.use_subs = True
        args.use_reaction_type = True if args.dataset_name in ['50k_class', 'pistachio_class',
                                                          'pistachio_demo_class'] else False
        args.decoder_cls = True
        args.save_strategy = 'mean'
        args.epochs = 700 if args.dataset_name in ['debug_uspto_50k', '50k_class', 'uspto_diverse', 'uspto_50k'] else 200

        args.save_epoch = [_ for _ in range(149, args.epochs, 5)] + [args.epochs] \
                            if args.dataset_name in ['uspto_50k', '50k_class', 'uspto_diverse'] \
                            else [_ for _ in range(19, args.epochs, 1)] + [args.epochs]

        # # 128 in default
        if args.dataset_name in ['uspto_50k', '50k_class', 'uspto_diverse', 'debug_uspto_50k']:
            if args.train_task == 'prod2subs':
                args.use_subs = False
                args.batch_size = 256
            elif args.train_task == 'bidirection':
                args.use_subs = True
                args.batch_size = 128
        else:                       # for mit, full
            if args.train_task in ['prod2subs', 'subs2prod']:
                # args.use_subs = False
                args.batch_size = 128
            elif args.train_task == 'bidirection':
                # args.use_subs = True
                args.batch_size = 64

        # if args.dataset_name == "debug_uspto_50k":
        #     args.batch_size = 256

        args.token_limit = 12000 if args.dataset_name in ['full', 'pistachio', 'pistachio_class'] else 0
        args.memory_clear_count = 1 if args.dataset_name in ['uspto_50k', '50k_class', 'uspto_diverse', "debug_uspto_50k"] else 4
        args.lr = 1 if args.dataset_name in ['uspto_50k', '50k_class', 'uspto_diverse', "debug_uspto_50k"] else 1.25
        args.dropout = 0.3 if args.dataset_name in ['uspto_50k', '50k_class', 'uspto_diverse', "debug_uspto_50k"] else 0.1
        # args.train_task = 'prod2subs'

        args.eval_task = 'subs2prod' if args.dataset_name in ['uspto_mit'] else 'prod2subs'
        args.use_splited_data = False if args.dataset_name in ["debug_uspto_50k", 'uspto_50k', '50k_class', 'uspto_diverse'] else True
        args.split_data_name = 'split_10000'

    elif args.request in ['eval', 'test']:
        if args.download_checkpoint:
            download_checkpoint(args.dataset)
            args.ckpt_list = [args.dataset]

        if args.train_task == 'prod2subs':
            args.use_subs = False
        elif args.train_task == 'bidirection':
            args.use_subs = True

        args.save_name = args.dataset_name + '-' + args.model_type + '-' + args.beam_module + '-' + args.train_task + '-' + args.loss_type
        if args.model_type == 'BiG2S_HCVAE':
            args.save_name += '-lat_sz' + str(args.lat_disc_size)

        args.mode = args.request
        # args.use_subs = True
        args.use_reaction_type = True if args.dataset_name in ['50k_class', 'pistachio_class'] else False
        # args.beam_module = 'huggingface'
        #128
        args.token_limit = 12000 if args.dataset_name in ['full'] else 0
        args.beam_size = 10
        if args.dataset_name in ['uspto_50k', '50k_class', 'uspto_diverse', "debug_uspto_50k", "pistachio"]:
            args.T = 1.6
        elif args.dataset_name in ['uspto_mit', 'uspto_diverse']:
            args.T = 1.6
        if args.loss_type == 'Mixed-CE':
            args.T = 1.1
        elif args.dataset_name in ['full', 'pistachio_class']:
            args.T = 0.7
        args.eval_task = 'subs2prod' if args.dataset_name in ['uspto_mit'] else 'prod2subs'
        args.max_len = 512

    set_seed(args.seed)

    return args