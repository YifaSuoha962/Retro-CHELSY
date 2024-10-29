import os
import random
import math
import torch
import numpy as np

from .dataset_basic import DataLoad


class Data_Split(DataLoad):
    def __init__(
            self,
            output_dir,
            data_name,
            split_type,
            load_type='data',
            mode='val',
            seed=17
    ):
        self._set_seed(seed=seed)

        super(Data_Split, self).__init__(
            root_dir=output_dir,
            output_dir=output_dir,
            data_name=data_name,
            split_type=split_type,
            load_type=load_type,
            mode=mode
        )
        self.mode = mode
        self.length_list = self.length_data
        self.data_len = self.data[self.mode]['subs']['token'].shape[0]

        self.train_data, self.eval_data, self.test_data = {'subs': {}, 'prod': {}}, \
            {'subs': {}, 'prod': {}}, {'subs': {}, 'prod': {}}
        self.data_dict = {'train': self.train_data,
                          'val': self.eval_data, 'test': self.test_data}

    def _set_seed(
            self,
            seed: int
    ):
        random.seed(seed)
        # 决定 np.random.permutation ?
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def split(
            self,
            split_len,
            split_name
    ):
        # shape = [3 (sub, prod, rxn_type), 3 (token, graph, token_atom), tot_len(size of train_data)]
        subs_data = self.data[self.mode]['subs']
        prod_data = self.data[self.mode]['prod']
        reaction_type = self.data[self.mode]['reaction_type']

        shuffle_idx = np.random.permutation(self.data_len)[:split_len]
        self.data_dict[self.mode]['subs']['token'] = subs_data['token'][shuffle_idx]
        self.data_dict[self.mode]['prod']['token'] = prod_data['token'][shuffle_idx]

        if subs_data['token_atom'] is None:
            self.data_dict[self.mode]['subs']['token_atom'] = None
            self.data_dict[self.mode]['prod']['token_atom'] = None
        else:
            self.data_dict[self.mode]['subs']['token_atom'] = subs_data['token_atom'][shuffle_idx]
            self.data_dict[self.mode]['prod']['token_atom'] = prod_data['token_atom'][shuffle_idx]

        if subs_data['graph'] is None:
            self.data_dict[self.mode]['subs']['graph'] = None
            self.data_dict[self.mode]['prod']['graph'] = None
        else:
            self.data_dict[self.mode]['subs']['graph'] = subs_data['graph'][shuffle_idx]
            self.data_dict[self.mode]['prod']['graph'] = prod_data['graph'][shuffle_idx]

        shuffle_reaction_type = reaction_type[shuffle_idx]
        shuffle_data_length = self.length_list[shuffle_idx]

        subs = np.array(self.data_dict[self.mode]['subs'], dtype=object)
        prod = np.array(self.data_dict[self.mode]['prod'], dtype=object)

        np.savez(os.path.join(self.out_dir, f'{self.mode}_{self.split_type}_preprocess_{split_name}_{str(split_len)}.data'),
                subs=subs, prod=prod, reaction_type=shuffle_reaction_type)
        # np.savez(os.path.join(self.out_dir, f'{self.mode}_{self.split_type}_preprocess_{split_name}_{str(split_len)}.data'),
        #          subs=subs, prod=prod)
        np.savez(os.path.join(self.out_dir, f'{self.mode}_{self.split_type}_length_{split_name}_{str(split_len)}.npz'),
                 len=shuffle_data_length)


if __name__ == '__main__':
    data_sp = Data_Split(
        data_name='uspto_MIT',
        split_type='token',
        load_type='data',
        mode='val',
        seed=17
    )
    data_sp.split(
        split_len=10000,
        split_name='split'
    )