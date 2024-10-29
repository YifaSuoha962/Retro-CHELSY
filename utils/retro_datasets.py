import os
import pandas as pd
from tqdm import tqdm
import re

from .dataset_basic import Dataset, download
from .preprocess_smi import canonicalize_smiles, token_preprocess, char_preprocess


class RetroCsvDataset(Dataset):
    # This type is adopted by 50k and MIT
    def __init__(self, root_dir, output_dir, data_name='debug_uspto_50k'):
        self.data_name = data_name
        self.root_dir = os.path.join(root_dir, 'data')
        self.output_dir = output_dir
        super(RetroCsvDataset, self).__init__()

    @property
    def key(self) -> str:
        return self.data_name

    @property
    def in_dir(self) -> str:
        return os.path.join(self.root_dir, self.key)

    @property
    def out_dir(self) -> str:
        return os.path.join(self.output_dir, self.key)

    def csv_process(self):
        train = {
            'reaction_type': [],
            'product_token': [],
            'substrates_token': [],
            'product_char': [],
            'substrates_char': [],
            # 'ptr': []
        }
        eval = {
            'reaction_type': [],
            'product_token': [],
            'substrates_token': [],
            'product_char': [],
            'substrates_char': [],
            # 'ptr': []
        }
        test = {
            'reaction_type': [],
            'product_token': [],
            'substrates_token': [],
            'product_char': [],
            'substrates_char': [],
            # 'ptr': []
        }

        for data_split_type, filename, data_store in \
                (('train', 'raw_train.csv', train), ('val', 'raw_val.csv', eval), ('test', 'raw_test.csv', test)):
            data_path = os.path.join(self.in_dir, f'{filename}')
            if not os.path.exists(data_path):
                if self.data_name in ['debug_uspto_50k']:
                    url_data = 'https://www.dropbox.com/scl/fo/bpi5lqgswiy6mwavjf782/h?dl=0&rlkey=189kwhni39ms89jq2ex19x5sb'
                    download(
                        # url is default for 50k
                        url=url_data,
                        save_dir=os.path.join(self.dir, f'{self.data_name}'),
                        file_name='data.zip'
                    )

            raw_data = pd.read_csv(data_path)

            # for diverse prediction
            last_prod = None

            for i, (reaction_smile, reaction_type) in enumerate(tqdm(zip(raw_data['reactants>reagents>production'], raw_data['class']),
                                                      desc=f'split {filename} SMILES...', total=len(raw_data))):
                subs, prod = tuple(reaction_smile.split('>>'))
                subs, prod = canonicalize_smiles(subs),\
                    canonicalize_smiles(prod)
                subs_token, prod_token = token_preprocess(subs, prod)
                subs_char, prod_char = char_preprocess(subs, prod)
                data_store['reaction_type'].append(reaction_type)
                data_store['product_token'].append(prod_token)
                data_store['substrates_token'].append(subs_token)
                data_store['product_char'].append(prod_char)
                data_store['substrates_char'].append(subs_char)
            #     if prod != last_prod:
            #         data_store['ptr'].append(i)
            # data_store['ptr'].append(raw_data.shape[0])     # record start indices of different products
            pd.DataFrame(data_store).to_csv(os.path.join(self.out_dir, f'{data_split_type}.csv'), index=False)


class RetroTxtDataset(Dataset):
    def __init__(self, root_dir, output_dir, data_name='uspto_full'):
        self.data_name = data_name
        self.root_dir = os.path.join(root_dir, 'data')
        self.output_dir = output_dir
        super(RetroTxtDataset, self).__init__()

    @property
    def key(self) -> str:
        return self.data_name

    @property
    def in_dir(self) -> str:
        return os.path.join(self.root_dir, self.key)

    @property
    def out_dir(self) -> str:
        return os.path.join(self.output_dir, self.key)

    def csv_process(self):
        train = {
            'reaction_type': [],
            'product_token': [],
            'substrates_token': [],
            'product_char': [],
            'substrates_char': [],
            # 'ptr': []
        }
        eval = {
            'reaction_type': [],
            'product_token': [],
            'substrates_token': [],
            'product_char': [],
            'substrates_char': [],
            # 'ptr': []
        }
        test = {
            'reaction_type': [],
            'product_token': [],
            'substrates_token': [],
            'product_char': [],
            'substrates_char': [],
            # 'ptr': []
        }

        for data_split_type, data_store in (('train', train), ('val', eval), ('test', test)):
            prod_path, subs_path = os.path.join(self.in_dir, f'src-{data_split_type}.txt'), \
                                   os.path.join(self.in_dir, f'tgt-{data_split_type}.txt')
            if not (os.path.exists(prod_path) and os.path.exists(subs_path)):
                raise FileNotFoundError(f"One or both of the required files are missing: '{prod_path}', '{subs_path}'")
            prod_data, subs_data = open(prod_path, 'r'), open(subs_path, 'r')

            # for diverse prediction
            # last_prod = None

            for i, (prod_smi, subs_smi) in enumerate(tqdm(zip(prod_data, subs_data),
                                             desc=f'split {self.data_name}-{data_split_type} SMILES...')):
                # example: <RX_10> O = C ( O ) C / C ( C l ) = C / C l
                # extract rxn_type_num
                # rxn_pos = prod_smi.index('>')
                # rxn_info = prod_smi[: rxn_pos]
                # rxn_num = re.findall(r'\d+', rxn_info)[0]
                """only for pistachio"""
                prod_smi = prod_smi.replace('\n', '')
                subs_smi = subs_smi.replace('\n', '')
                # canonicalize
                prod_smi = canonicalize_smiles(prod_smi.replace(' ', ''))
                subs_smi = canonicalize_smiles(subs_smi.replace(' ', ''))
                subs_token, prod_token = token_preprocess(subs_smi, prod_smi)
                subs_char, prod_char = char_preprocess(subs_smi, prod_smi)

                # 极端情况：H2O,smiles为一个字符
                if len(prod_smi) <= 1 or len(subs_smi) <= 1: continue
                data_store['reaction_type'].append(0)       # 没有反应类型，默认为 0
                data_store['product_token'].append(prod_token)
                data_store['substrates_token'].append(subs_token)
                data_store['product_char'].append(prod_char)
                data_store['substrates_char'].append(subs_char)

            #     if prod_smi != last_prod:
            #         data_store['ptr'].append(i)
            # prod_lines = prod_data.readlines()
            # data_store['ptr'].append(len(prod_lines))

        pd.DataFrame(data_store).to_csv(os.path.join(self.out_dir, f'{data_split_type}.csv'), index=False)

def select_reader(proj_dir, output_dir, data_name):
    if data_name in ['debug_uspto_50k', 'uspto_mit', 'uspto_50k']:
        return RetroCsvDataset(root_dir=proj_dir, output_dir=output_dir, data_name=data_name)
    elif data_name in ['uspto_diverse', 'uspto_full']:
        return RetroTxtDataset(root_dir=proj_dir, output_dir=output_dir, data_name=data_name)