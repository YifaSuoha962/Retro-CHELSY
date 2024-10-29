import argparse
import logging
import os
import json
import pandas as pd
import re
import time
import sys
from pathlib import Path
from rdkit import Chem
from functools import partial
from tqdm import tqdm
import numpy as np
import pandas as pd
import onmt
from onmt.translate.translator import build_translator
import onmt.opts


def add_result_args(parser):
    group = parser.add_argument_group("Meta")
    group.add_argument("--data_name", help="Dataset name",
                       choices=["USPTO_50k", "uspto_diverse", "USPTO_DIVERSE_shuf"], type=str,
                       default="USPTO_50k")
    # group.add_argument("--experiment", help="name of experiment")
    # group.add_argument("--checkpoint", help="pt file", type=str)
    # argparse 不会解析 bool 值
    parser.add_argument("--load_rt_res", help="load round_trip result file", action="store_true")
    parser.add_argument("--load_inter", help="load intermediate results", action="store_true")
    parser.add_argument("--result_file", default="./results/uspto_50k4rt_.csv",
                        help="load retrosynthesis prediction in csv if do not use --load _rt_res, or load round_trip result csv")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--pred_type", choices=['rand_unknown','rand_known', 'rsmi_unknown', 'rsmi_known', 'dual_task'], default='dual_task')
    # mol_transformer 路径
    parser.add_argument("--mol_trans_dir", default="/data/whb/Retrosynthesis/Retro-VAE/MolecularTransformer/")


"""
Chen, Shuan, and Yousung Jung.
"Deep retrosynthetic reaction prediction using local reactivity and global attention."
JACS Au 1.10 (2021): 1612-1620.

Predicted precursors are considered correct if 
- the predicted precursors are the same as the ground truth
- Molecular Transformer predicts the target product for the proposed precursors
"""

def canonicalize_smiles(smiles, remove_atom_number=False, trim=True, suppress_warning=False, stereo=True):
    cano_smiles = ""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    if mol is None:
        cano_smiles = ""
    else:
        if trim and mol.GetNumHeavyAtoms() < 2:
            if not suppress_warning:
                logging.info(f"Problematic smiles: {smiles}, setting it to 'CC'")
            cano_smiles = "CC"          # TODO: hardcode to ignore
        else:
            if remove_atom_number:
                [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
                # isomericSmiles 控制立体信息
            cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=stereo)
    return cano_smiles


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    https://github.com/pschwllr/MolecularTransformer/tree/master#pre-processing
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens)
    if smi != ''.join(tokens):
        print(smi, ''.join(tokens))
    return ' '.join(tokens)


def get_top_k(df, k, scoring=None):
    if callable(scoring):
        df["_new_score"] = scoring(df)  # new_score = log(confidance) ?
        scoring = "_new_score"
    # if scoring is not None:
    #     df = df.sort_values(by=scoring, ascending=False)
    # df = df.drop_duplicates(subset='pred_reactant')
    return df.head(k)


"""
refer from "RetroBridge": https://github.com/igashov/RetroBridge
"""
def eval_func(df, top_k=[1]):
    # count in each top-k-candidates
    df['group'] = np.arange(len(df)) // top_k[-1]
    counts = df.groupby(['group', 'pred_reactant']).size().reset_index(name='count')
    group_size = df.groupby(['group']).size().reset_index(name='group_size')

    counts_dict = {(g, p): c for g, p, c in zip(counts['group'], counts['pred_reactant'], counts['count'])}
    df['count'] = df.apply(lambda x: counts_dict[(x['group'], x['pred_reactant'])], axis=1)
    size_dict = {g: s for g, s in zip(group_size['group'], group_size['group_size'])}
    df['group_size'] = df.apply(lambda x: size_dict[x['group']], axis=1)

    # confidence
    df['confidence'] = df['count'] / df['group_size']

    assert (df.groupby(['group', 'pred_reactant'])['confidence'].nunique() == 1).all()  # confidence has one val for each pair of ()
    assert (df.groupby(['group'])['group_size'].nunique() == 1).all()

    # top-k acc (Retroynthesis, Round_Trip)
    results = {}
    results['Exact match'] = {}
    results['Round-trip coverage'] = {}
    results['Round-trip accuracy'] = {}
    # df['exact_match'] is already created
    df['round_trip_match'] = df['product'] == df['pred_product']  # & ~df['invalid']
    df['match'] = df['exact_match'] | df['round_trip_match']  # round_trip coverage
    # df['match'] = df['match'] & ~df['invalid']

    for k in tqdm(top_k):
        topk_df = df.groupby(['group']).apply(
            partial(get_top_k, k=k, scoring=lambda df: np.log(df['confidence']))).reset_index(drop=True)
        acc_exact_match = topk_df.groupby('group').exact_match.any().mean()

        results['Exact match'][f'top-{k}'] = acc_exact_match
        print(f"Top-{k}")
        print("Exact match accuracy", acc_exact_match)

        cov_round_trip = topk_df.groupby('group').match.any().mean()
        acc_round_trip = topk_df.groupby('group').match.mean().mean()
        results['Round-trip coverage'][f'top-{k}'] = cov_round_trip
        results['Round-trip accuracy'][f'top-{k}'] = acc_round_trip
        print("Round-trip coverage", cov_round_trip)
        print("Round-trip accuracy", acc_round_trip)

    return pd.DataFrame(results).T


def collecet_retro_res(result_file, args):

    pred_type = args.pred_type
    top_k = args.top_k

    products_df = []  # column for products
    true_reactants_df = []  # column for true reactants
    pred_reactants_df = []  # column for predicted reactants
    correct_tags_df = []  # is the prediction correct ?
    invalid_tags_df = []  # is the prediction valid

    original_df = pd.read_csv(result_file)
    accuracies = np.zeros([original_df.shape[0], top_k], dtype=np.float32)

    # 返回dataframe，做完round_trip后一起保存
    for i in range(original_df.shape[0]):
        # expand k times in default

        # 数据中有 " ?
        cleaned_predictions = original_df['predictions'][i].replace('"','')
        # cleaned_predictions = cleaned_predictions.replace('\'', '')

        pred_reacts = cleaned_predictions.split(',')     # prediction are splited in '\t'

        if original_df['products'][i] != '':
            cano_prod = original_df['products'][i]
        else:
            cano_prod = canonicalize_smiles(original_df['products'][i], trim=False, stereo=False)

        if i == 0:
            print(f"original prod: {original_df['products'][i]}")
            print(f"canonical prod: {cano_prod}")

        cano_react = canonicalize_smiles(original_df['reactants'][i], trim=False, stereo=True)
        for k, pred in enumerate(pred_reacts):
            invalid = 0
            is_correct = 0

            # 判断 预测结果和真实反应物是否相同
            cano_pred = canonicalize_smiles(pred, trim=False, stereo=True)
            if not cano_pred:
                invalid = 1

            if cano_pred == cano_react or pred == original_df['reactants'][i]:
                is_correct = 1
                accuracies[i, k:] = 1.0

            products_df.append(cano_prod)
            true_reactants_df.append(canonicalize_smiles(cano_react, trim=False, stereo=False))
            if cano_prod != '':
                pred_reactants_df.append(canonicalize_smiles(cano_pred, trim=False, stereo=False))
            else:
                pred_reactants_df.append(cano_pred)
            correct_tags_df.append(is_correct)
            invalid_tags_df.append(invalid)

    retro_res = {'product': products_df, 'pred_reactant': pred_reactants_df,
                 'true_reactant': true_reactants_df, 'exact_match': correct_tags_df,
                 'invalid': invalid_tags_df}

    retro_res_df = pd.DataFrame(retro_res)

    # 标记重复的 prediction
    retro_res_df['duplicate'] = retro_res_df['pred_reactant'].duplicated(keep=False).astype(bool)

    print(f"total_acc = {retro_res_df['exact_match'].sum() / retro_res_df.shape[0] * 100: .2f} %")
    print(f"total_invalid rate = {retro_res_df['invalid'].sum() / retro_res_df.shape[0] * 100: .2f} %")
    print(f"duplicate rate = {retro_res_df['duplicate'].sum() / retro_res_df.shape[0] * 100: .2f} %")

    mean_accuracies = np.mean(accuracies, axis=0)
    sum_accuracies = np.sum(accuracies, axis=0)
    print(f"sum_accuracies.shape[0] = {sum_accuracies.shape[0]}")
    print(f"num_rxns = {original_df.shape[0]}")
    for n in range(top_k):
        print(f"Top {n + 1} accuracy: {sum_accuracies[n] / original_df.shape[0] * 100: .2f} %")

    print(f"retro_res_df.shape[0] = {retro_res_df.shape[0]}, original_df.shape[0] = {original_df.shape[0]}")
    assert retro_res_df.shape[0] == original_df.shape[0] * top_k

    return retro_res_df


def round_trip_exp(parser: argparse.ArgumentParser, args, df):
    df ['pred_reactant'] = df['pred_reactant'].fillna('')
    pre_df_shape = df.shape[0]

    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)
    args = res_parser.parse_args(sys.argv[1:] + [
        "-model", str(Path(args.mol_trans_dir, 'models', 'MIT_mixed_augm_model_average_20.pt')),
        "-src", "input.txt", "-output", "pred.txt ",
        "-replace_unk", "-max_length", "200", "-fast" , "-gpu" ,"0"
    ])

    # Find unique SMILES
    unique_df = df[df['pred_reactant'] != '']
    unique_pred_df = unique_df[['pred_reactant']].drop_duplicates(subset='pred_reactant', keep='first')
    unique_pred_df = unique_pred_df.reset_index(drop=True)
    unique_smiles = list(set(unique_pred_df['pred_reactant']))

    # 找问题，是不是反应物也不能有立体信息
    # unique_pred_df = df[['true_reactant']].drop_duplicates(subset='true_reactant', keep='first')
    # unique_pred_df = unique_pred_df.reset_index(drop=True)
    # unique_smiles = list(set(unique_pred_df['true_reactant']))
    print(f"len(unique_smiles) = {len(unique_smiles)}")
    assert len(unique_smiles) == unique_pred_df.shape[0]

    # Tokenize
    tokenized_smiles = [smi_tokenizer(s.strip()) for s in unique_smiles]

    # TODO: tokenize 后的数据类型有问题，得回到collect_res_df 检查数据
    print("Predicting products...")
    tic = time.time()
    translator = build_translator(args, report_score=True)
    scores, pred_products = translator.translate(
        src_data_iter=tokenized_smiles,
        batch_size=args.batch_size,
        attn_debug=args.attn_debug
    )

    demo_prediction = pred_products[0][0]
    # print(f"demo of pred_products = {demo_prediction}")
    print(f"demo of stripped pred_products = {demo_prediction.strip()}")
    print(f"demo of cleaned pred_products = {''.join(demo_prediction.strip().split())}")

    pred_products = [x[0].strip() for x in pred_products]           # 取 top 1， .strip() 去掉 空格？
    print("... done after {} seconds".format(time.time() - tic))

    # De-tokenize
    pred_products = [''.join(x.split()) for x in pred_products]

    # gather results
    pred_products = {r: p for r, p in zip(unique_smiles, pred_products)}

    # update dataframe
    unique_pred_df['pred_product'] = [canonicalize_smiles(pred_products[r], trim=False) for r in unique_pred_df['pred_reactant']]

    # unique_pred_df['pred_product'] = [canonicalize_smiles(pred_products[r], trim=False) for r in unique_pred_df['true_reactant']]

    print(f"demo of pred_products = {pred_products[unique_smiles[0]]}")

    df = df.merge(unique_pred_df, on='pred_reactant', how='left')

    assert df.shape[0] == pre_df_shape

    save_dir = f'round_trip/{args.data_name}'
    csv_file = os.path.join(save_dir, f'rt_exp_{args.pred_type}.csv')
    df.to_csv(csv_file, index=False)

    return df


if __name__ == "__main__":
    res_parser = argparse.ArgumentParser("result")
    add_result_args(res_parser)
    args = res_parser.parse_args()

    # save as examples
    save_dir = 'round_trip/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_dir = os.path.join(save_dir, args.data_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # prepare data and perform round_trip experiments
    print(f"args.load_rt_res = {args.load_rt_res}, args.load_inter = {args.load_inter}")
    if not args.load_rt_res:
        if not args.load_inter:
            retro_df = collecet_retro_res(args.result_file, args)
            round_trip_df = round_trip_exp(res_parser, args, retro_df)
        else:
            retro_df = pd.read_csv(args.result_file)
            round_trip_df = round_trip_exp(res_parser, args, retro_df)
    else:
        rt_file = os.path.join(data_dir, args.result_file)
        round_trip_df = pd.read_csv(rt_file)

    # compute round_trip scores
    if args.top_k ==10:
        top_k_list = [1, 3, 5, 10]
    elif args.top_k == 5:
        top_k_list = [1, 3, 5]
    else:
        top_k_list = [1]
    round_trip_info = eval_func(round_trip_df, top_k_list)

    target_file = os.path.join(data_dir, f'{args.pred_type}_round_trip_res.csv')
    round_trip_info.to_csv(target_file, index=False)


















