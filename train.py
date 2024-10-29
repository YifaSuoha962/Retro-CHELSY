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

from models.module_utils import (get_lr, train_plot, eval_plot, train_eval_plot, token_acc_record,
                                beam_result_process, Transformer_WarnUp, Model_Save)

# 分布式训练，数据并行

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

proj_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(proj_dir, 'preprocessed')
origin_dir = os.path.join(proj_dir, 'data')

def training(args):

    if args.use_subs and (args.use_reaction_type or args.model_type == 'BiG2S_HCVAE'):
        dec_cls = 2
    elif args.use_subs or args.use_reaction_type or args.model_type == 'BiG2S_HCVAE':
        dec_cls = 1
    else:
        dec_cls = 0

    # prepare datasets
    if args.representation_form == 'graph2smiles':
        train_dataset = G2SFullDataset(
            # origin_dir=origin_dir,
            output_dir=output_dir,
            dataset_name=args.dataset_name,
            split_type=args.split_type,
            batch_size=args.batch_size,
            token_limit=args.token_limit,
            mode='train',
            dist_block=args.graph_dist_block,
            task=args.train_task
        )
    else:
        train_dataset = S2SFullDataset(
            # origin_dir=origin_dir,
            output_dir=output_dir,
            dataset_name=args.dataset_name,
            split_type=args.split_type,
            batch_size=args.batch_size,
            token_limit=args.token_limit,
            mode='train',
            dist_block=args.graph_dist_block,
            task=args.train_task)
    if args.eval:
        if args.representation_form == 'graph2smiles':
            eval_dataset = G2SFullDataset(
                # origin_dir=origin_dir,
                output_dir=output_dir,
                dataset_name=args.dataset_name,
                split_type=args.split_type,
                batch_size=args.batch_size,
                token_limit=args.token_limit,
                mode='val',
                dist_block=args.graph_dist_block,
                task=args.train_task,  # prod2subs
                use_split=args.use_splited_data,
                split_data_name=args.split_data_name
            )
        else:
            eval_dataset = S2SFullDataset(
                # origin_dir=origin_dir,
                output_dir=output_dir,
                dataset_name=args.dataset_name,
                split_type=args.split_type,
                batch_size=args.batch_size,
                token_limit=args.token_limit,
                mode='val',
                dist_block=args.graph_dist_block,
                task=args.train_task,  # prod2subs
                use_split=args.use_splited_data,
                split_data_name=args.split_data_name
            )

    # prepare model
    ckpt_dir = os.path.join('./checkpoints', args.save_name)
    if args.use_reaction_type:
        ckpt_dir = ckpt_dir + 'rxn'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    token_idx = train_dataset.token_idx
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
            token_freq=train_dataset.token_freq,
            token_count=train_dataset.token_count,
            cls_len=dec_cls,
            args=args
        )
    elif args.model_type == 'BiG2S_HCVAE':
        module = Graph_Transformer_VAE(
            f_vocab=len(token_idx),
            f_atom=NODE_FDIM + 10 if args.use_reaction_type else NODE_FDIM,
            f_bond=BOND_FDIM,
            token_idx=token_idx,
            token_freq=train_dataset.token_freq,
            token_count=train_dataset.token_count,
            cls_len=dec_cls,
            args=args
        )
    elif args.model_type == 'S2S_HCVAE':
        module = Seq_Transformer_VAE(
            f_vocab=len(token_idx),
            token_idx=token_idx,
            token_freq=train_dataset.token_freq,
            token_count=train_dataset.token_count,
            cls_len=dec_cls,
            args=args
        )
    else:  # pure transformer : S2S
        module = Seq_Transformer_Base(
            f_vocab=len(token_idx),
            token_idx=token_idx,
            token_freq=train_dataset.token_freq,
            token_count=train_dataset.token_count,
            cls_len=dec_cls,
            args=args
        )

    # rank = int(os.environ["LOCAL_RANK"])
    # print(f"rank = {rank}")
    #
    # if rank != -1:
    #     torch.cuda.set_device(rank)
    #     dist.init_process_group(backend='nccl')
    # if rank != -1:
    #     module.model = DDP(module.model.cuda(),
    #                 device_ids=[rank],
    #                 output_device=rank,
    #                 find_unused_parameters=True)
    # else:
    #     module.model = torch.nn.DataParallel(module.model, device_ids=[int(os.getenv("LOCAL_RANK"))]).cuda()

    if args.optimizer == 'AdamW':
        optimizer = optm.AdamW(
            params=module.parameters,
            lr=args.lr,
            eps=args.eps,
            betas=args.betas,
            weight_decay=args.weight_decay
        )
    if args.lr_schedule == 'original':
        lr_schedule = Transformer_WarnUp(
            optimizer=optimizer,
            args=args
        )
    elif args.lr_schedule == 'cosine':
        def lr_lambda(step):
            scale = 1
            max_step = int(train_dataset.batch_step // args.accum_count) * args.epochs if args.token_limit == 0\
                else int((train_dataset.all_token_count / args.token_limit) / args.accum_count) * args.epochs
            if step == 0 and args.warmup_step == 0:
                scale = 1
            else:
                scale = step / args.warmup_step if step <= args.warmup_step\
                    else (args.min_lr + 0.5 * (args.lr - args.min_lr) * \
                        (1.0 + math.cos(((step - args.warmup_step) / (max(max_step, step) - args.warmup_step)) * math.pi))) / args.lr
            if scale * args.lr < args.min_lr and step > args.warmup_step:
                scale = args.min_lr / args.lr
            return scale
        lr_schedule = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambda
        )

    total_param = sum(p.numel() for p in module.parameters)
    trainable_param = sum(p.numel() for p in module.parameters if p.requires_grad)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(os.path.join(ckpt_dir, 'module.log')):
        logging.basicConfig(filename=os.path.join(ckpt_dir, 'module.log'),
                            format='%(asctime)s %(message)s', level=logging.INFO)
        for k, v in args.__dict__.items():
            logging.info('args -> {0}: {1}'.format(k, v))
            print('args -> {0}: {1}'.format(k, v))
        logging.info(module.model)
        logging.info('total param: {0}'.format(total_param))
        logging.info('trainable param: {0}'.format(trainable_param))
    else:
        logging.basicConfig(filename=os.path.join(ckpt_dir, 'module.log'),
                            format='%(asctime)s %(message)s', level=logging.INFO)
        for k, v in args.__dict__.items():
            logging.info('args -> {0}: {1}'.format(k, v))
            print('args -> {0}: {1}'.format(k, v))

    if args.save_strategy != 'last':
        assert args.eval == True

    if args.load_ckpt:
        _, module.model, _, _ = module_saver.load(args.load_ckpt, module.model)

    accum_step = 0
    accum = 0
    train_loss_list = []
    train_seq_acc_list = []
    train_token_acc_list = []
    eval_seq_acc_list = []
    eval_seq_invalid_list = []
    finish_epochs = -1

    dynamic_token_weight = [[], 0]

    tot_epochs = range(finish_epochs + 1, args.epochs)

    if args.load_ckpt:
        tot_epochs = range(int(args.load_ckpt)+1, args.epochs)
    print(f"tot_epochs={tot_epochs}")
    # assert 1 == 2

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    logging.info('start training')

    # 后面换成 tot_epochs
    num_tot_epochs = len(list(tot_epochs))
    for epoch in tot_epochs:
        torch.cuda.empty_cache()
        train_dataset.get_batch()

        data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=True)    # if pin_memory = true, self-constructed pin_memory function will be caled

        optimizer.zero_grad()

        loss_per_epoch = 0
        seq_acc_per_epoch = [0, 0]
        token_acc_per_epoch = [0, 0]
        all_token_acc_count = [[0 for _ in range(len(token_idx))], [0 for _ in range(len(token_idx))]]
        loss_per_step = 0
        seq_acc_per_step = [0, 0]
        token_acc_per_step = [0, 0]
        sub_loss_record = [0, 0, 0]

        tstep = trange(train_dataset.batch_step)
        clear_step = [(train_dataset.batch_step // (args.memory_clear_count + 1)) * i for i in
                      range(1, args.memory_clear_count + 1, 1)]

        total_steps = train_dataset.batch_step * num_tot_epochs

        # following du-vae
        if args.vae_warmup > 0:
            anneal_rate = 1.0 / (args.vae_warmup * train_dataset.batch_step)
        else:
            anneal_rate = 0

        for step, batch in zip(tstep, data_loader):
            if step in clear_step: torch.cuda.empty_cache()
            batch = batch.to(args.device)

            if args.model_type != ['BiG2S', 'S2S']:
                module.model.update_kl_weight(args, anneal_rate)        # chatgpt 解析，设置DDP mode的 model 后只会暴露forward接口，其他接口不会被开放

            with torch.cuda.amp.autocast(enabled=True):
                loss, token_acc, seq_acc, true_token_count, all_token_count, sub_loss = module.model_train(batch, accum_step, total_steps)

            loss = loss / args.accum_count
            scaler.scale(loss).backward()

            loss_per_epoch += loss.item() * args.accum_count
            loss_per_step += loss.item()
            sub_loss_record = [i + j for i, j in zip(sub_loss_record, sub_loss)]
            seq_acc_per_epoch[0] += seq_acc[0]
            seq_acc_per_epoch[1] += seq_acc[1]
            token_acc_per_epoch[0] += token_acc[0]
            token_acc_per_epoch[1] += token_acc[1]
            seq_acc_per_step[0] += seq_acc[0] / args.accum_count
            seq_acc_per_step[1] += seq_acc[1] / args.accum_count
            token_acc_per_step[0] += token_acc[0] / args.accum_count
            token_acc_per_step[1] += token_acc[1] / args.accum_count

            all_token_acc_count[0] = [i + j for i, j in zip(all_token_acc_count[0], true_token_count)]
            all_token_acc_count[1] = [i + j for i, j in zip(all_token_acc_count[1], all_token_count)]

            accum += 1
            if accum == args.accum_count:
                if args.clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        parameters=module.parameters,
                        max_norm=args.clip_norm,
                        error_if_nonfinite=False
                    )
                scaler.step(optimizer)
                scaler.update()
                lr_schedule.step()
                optimizer.zero_grad()
                accum = 0
                accum_step += 1

            if accum == 0:
                tstep.set_description(
                    'epoch {0}, step {1}: loss {2:.6}, p2s[seq {3:.4}; token {4:.4}], s2p[seq {5:.4}; token {6:.4}], lr {7:.4}'
                    .format(epoch, accum_step, loss_per_step, seq_acc_per_step[0], token_acc_per_step[0],
                            seq_acc_per_step[1], token_acc_per_step[1], get_lr(optimizer)))
                loss_per_step = 0
                seq_acc_per_step = [0, 0]
                token_acc_per_step = [0, 0]

        if args.accum_count > 1 and accum > 0:
            if args.clip_norm > 0.0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    parameters=module.parameters,
                    max_norm=args.clip_norm,
                    error_if_nonfinite=False
                )
            scaler.step(optimizer)
            scaler.update()
            lr_schedule.step()
            optimizer.zero_grad()
            accum = 0
            accum_step += 1

            tstep.set_description(
                'epoch {0}, step {1}: loss {2:.6}, p2s[seq {3:.4}; token {4:.4}], s2p[seq {5:.4}; token {6:.4}], lr {7:.4}'
                .format(epoch, accum_step, loss_per_step, seq_acc_per_step[0], token_acc_per_step[0],
                        seq_acc_per_step[1], token_acc_per_step[1], get_lr(optimizer)))
            loss_per_step = 0
            seq_acc_per_step = [0, 0]
            token_acc_per_step = [0, 0]

        loss_per_epoch = loss_per_epoch / train_dataset.batch_step
        seq_acc_per_epoch[0] = seq_acc_per_epoch[0] / train_dataset.batch_step
        seq_acc_per_epoch[1] = seq_acc_per_epoch[1] / train_dataset.batch_step
        token_acc_per_epoch[0] = token_acc_per_epoch[0] / train_dataset.batch_step
        token_acc_per_epoch[1] = token_acc_per_epoch[1] / train_dataset.batch_step
        sub_loss_record = [_ / train_dataset.batch_step for _ in sub_loss_record]
        train_loss_list.append(loss_per_epoch)
        train_seq_acc_list.append(seq_acc_per_epoch)
        train_token_acc_list.append(token_acc_per_epoch)

        dynamic_token_weight[1] = sum(all_token_acc_count[0]) / sum(all_token_acc_count[1])
        dynamic_token_weight[0] = [i / max(j, 1) for i, j in zip(all_token_acc_count[0], all_token_acc_count[1])]

        if epoch in args.token_eval_epoch:
            token_acc_record(all_token_acc_count, token_idx, epoch, ckpt_dir)

        print(
            '----------> epoch {0} finish, step {1}, loss {2:.6}, p2s[seq {3:.4}; token {4:.4}], s2p[seq {5:.4}; token {6:.4}]'
            .format(epoch, accum_step, loss_per_epoch, seq_acc_per_epoch[0], token_acc_per_epoch[0],
                    seq_acc_per_epoch[1], token_acc_per_epoch[1]))
        logging.info(
            'epoch {0} finish, step {1}, loss {2:.6}, sub_loss[class {3:.4}; class_pair {4:.4}; bipair {5:.4}] p2s[seq {6:.4}; token {7:.4}], s2p[seq {8:.4}; token {9:.4}]'
            .format(epoch, accum_step, loss_per_epoch, sub_loss_record[0], sub_loss_record[1], sub_loss_record[2],
                    seq_acc_per_epoch[0], token_acc_per_epoch[0], seq_acc_per_epoch[1], token_acc_per_epoch[1]))

        if ((epoch in args.save_epoch) or (epoch in args.eval_epoch)) and eval:
            beam_size = args.beam_size
            seq_acc_count = np.zeros((args.return_num))
            seq_invalid_count = np.zeros((args.return_num))
            smi_predictions = []

            logging.info('epoch {0} eval start'.format(epoch))

            with torch.no_grad():
                eval_dataset.get_batch()
                if args.use_ddp:
                    data_sampler = DistributedSampler(eval_dataset, shuffle=False)
                    data_loader = DataLoader(
                        dataset=eval_dataset,
                        sampler=data_sampler,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=lambda _batch: _batch[0],
                        pin_memory=True
                    )
                else:
                    data_loader = DataLoader(
                        dataset=eval_dataset,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=lambda _batch: _batch[0],
                        pin_memory=True
                    )

                teval = trange(eval_dataset.batch_step)
                for step, batch in zip(teval, data_loader):
                    batch = batch.to(args.device)

                    predict_result, predict_scores = module.model_predict(
                        data=batch,
                        args=args
                    )
                    beam_acc, beam_invalid, beam_smi, _ = beam_result_process(
                        tgt_seq=batch.tgt_seq,
                        tgt_len=batch.tgt_seq_len,
                        token_idx=token_idx,
                        beam_result=predict_result,
                        beam_scores=predict_scores
                    )
                    seq_acc_count = seq_acc_count + beam_acc
                    seq_invalid_count = seq_invalid_count + beam_invalid
                    smi_predictions.extend(beam_smi)

                    teval.set_description('evaling......')

                seq_acc_count = np.cumsum(seq_acc_count)
                seq_invalid_count = np.cumsum(seq_invalid_count)
                seq_acc_count = seq_acc_count / eval_dataset.data_len
                seq_invalid_count = seq_invalid_count / np.array(
                    [i * eval_dataset.data_len for i in range(1, args.return_num + 1, 1)])

                # if epoch in args.save_epoch:
                eval_seq_acc_list.append(seq_acc_count)
                eval_seq_invalid_list.append(seq_invalid_count)
                eval_plot(
                    topk_seq_acc=seq_acc_count.tolist(),
                    topk_seq_invalid=seq_invalid_count.tolist(),
                    beam_size=beam_size,
                    data_name=args.dataset_name,
                    ckpt_dir=ckpt_dir,
                    ckpt_name=epoch,
                    args=args
                )
                if args.return_num >= 10:
                    logging.info(
                        'epoch {0} eval finish at step {1}, top1,3,5,10 acc is [{2:.4}, {3:.4}, {4:.4}, {5:.4}]' \
                        .format(epoch, accum_step, seq_acc_count[0], seq_acc_count[2], seq_acc_count[4],
                                seq_acc_count[9]))
                    print('epoch {0} eval finish, top1,3,5,10 acc is [{2:.4}, {3:.4}, {4:.4}, {5:.4}]' \
                          .format(epoch, accum_step, seq_acc_count[0], seq_acc_count[2], seq_acc_count[4],
                                  seq_acc_count[9]))

        if epoch in args.save_epoch:
            if eval is False:
                seq_acc_count = [0]
            module_saver.save(
                module=module.model,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                epoch=epoch,
                acc_list=seq_acc_count
            )

    # train_plot(train_loss_list, train_seq_acc_list, train_token_acc_list, ckpt_dir)
    # train_eval_plot(eval_seq_acc_list, ckpt_dir)
    logging.info('training finish')


if __name__ == '__main__':
    parser = get_parser(mode='train')
    args = post_setting_args(parser)

    training(args)






