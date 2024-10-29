import csv
import os

import pandas as pd
from tqdm import tqdm
import sys
import torch
import time
from rdkit import Chem
import logging
import json
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')

# # 指定目录
# origin_dir = "./USPTO_50k"
# saved_dir = 'debug_uspto_50k'
# # 获取目录中所有以 '.txt' 为后缀的文件的全名
# txt_files = [f for f in os.listdir(origin_dir) if f.endswith('.txt')]
#
#
# # 打印文件名
# for txt_file in txt_files:
#     extracted_lines = []
#     with open(os.path.join(origin_dir, txt_file), 'r') as origin_f:
#         extracted_lines.extend(origin_f.readlines()[:10])
#         with open(os.path.join(saved_dir, txt_file), 'w') as out_f:
#             out_f.writelines(extracted_lines)
#         out_f.close()
#     origin_f.close()


def canonicalize_smiles(smiles, remove_atom_number=False, trim=True, suppress_warning=False):
    cano_smiles = ""
    mol = Chem.MolFromSmiles(smiles)

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
            cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return cano_smiles


def contruct_one2many_map(src, tgt, rxn_prod_dict, phase, remove_atom_number=False):
    answer = {}
    last_p = None

    for idx, row in tqdm(enumerate(src), desc=f'Processing {phase}'):
        cano_p = row.strip()
        cano_r = tgt[idx].strip()
        cano_p = canonicalize_smiles(cano_p.replace(' ', ''), remove_atom_number=remove_atom_number)
        cano_r = canonicalize_smiles(cano_r.replace(' ', ''), remove_atom_number=remove_atom_number)

        if idx == 2:
            print(f"smi_prod = {cano_p}")
            print(f"smi_react = {cano_r}")

        if cano_p != last_p:
            last_rs = []

        if last_p == cano_p and cano_r in last_rs:
            continue

        if last_p != cano_p:
            answer[cano_p] = {'product': cano_p, 'reaction': 0, 'reactant': []}

        answer[cano_p]['reactant'].append(cano_r)
        answer[cano_p]['reaction'] += 1
        last_p = cano_p
        last_rs.append(cano_r)

    for elem in answer.values():
        n_rxn = elem['reaction']
        if n_rxn not in rxn_prod_dict:
            rxn_prod_dict[n_rxn] = 1
        else:
            rxn_prod_dict[n_rxn] += 1

    return answer


def process_file(data_name, phase):
    rxn_prod_dict = {}
    src = []
    tgt = []

    # data in txt
    if data_name != 'debug_uspto_50k':
        src_file = os.path.join(data_name, f"src-{phase}.txt")
        tgt_file = os.path.join(data_name, f"tgt-{phase}.txt")
        with open(src_file, 'r') as sf, open(tgt_file, 'r') as tf:
            src = sf.readlines()
            tgt = tf.readlines()
        answer = contruct_one2many_map(src, tgt, rxn_prod_dict, phase, remove_atom_number=False)
    # data in csv
    else:
        src_file = os.path.join(data_name, f"raw_{phase}.csv")
        rxn_data = pd.read_csv(src_file)
        for i in range(rxn_data.shape[0]):
            rxn = rxn_data.iloc[i]['reactants>reagents>production']
            react, prod = rxn.split('>>')
            tgt.append(react)
            src.append(prod)

        answer = contruct_one2many_map(src, tgt, rxn_prod_dict, phase, remove_atom_number=True)

    json_str = json.dumps(answer, indent=4)
    if 'val' in src_file:
        fn = os.path.join(data_name, "raw_val")
    elif 'test' in src_file:
        fn = os.path.join(data_name, "raw_test")
    elif 'train' in src_file:
        fn = os.path.join(data_name, "raw_train")

    with open(fn + '.json', 'w') as json_file:
        json_file.write(json_str)

    print(f'JSON Done for {phase}!')
    print(f"In {data_name}-{phase}\n info of num_rxn_prod:\n {rxn_prod_dict}")



def main():
    phases = ["train", "test", "val"]
    for data_name in ['./uspto_diverse', 'uspto_full', 'debug_uspto_50k']:
        with Pool(processes=len(phases)) as pool:
            pool.starmap(process_file, [(data_name, phase) for phase in phases])


if __name__ == "__main__":
    main()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    demo_react_1 = "C N . N # C c 1 c ( N ) c c c c 1 F".replace(" ", "")
    demo_react_2 = "C [NH3+] . N # C c 1 c ( N ) c c c c 1 F".replace(" ", "")

    cano_react_1 = canonicalize_smiles(demo_react_1)
    cano_react_2 = canonicalize_smiles(demo_react_2)
    print(f"is matching ? : {cano_react_2 == cano_react_1}")