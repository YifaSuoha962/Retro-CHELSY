import os
import re
import numpy as np
# import torch
# from torch_sparse import SparseTensor
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix
from rdkit import Chem
from tqdm.std import trange
from typing import Dict, List

import sys
print(sys.path)

from .dataset_basic import DataLoad
from .chem_tools import get_atom_feature, get_bond_feature, MAX_DIST, MAX_DEG, MOL_ATOM_FEAT_NUM, BOND_FEAT_NUM


#------------------------------------------------------------------------------------------------
#dict atom on ring transform to separate atom, like 'c' -> 'C'.
def atom_upper(atom_match):
    return atom_match.group('atom').upper()


def token_to_upper(idx_token: Dict[int, str]) -> Dict[int, str]:
    idx, token = list(idx_token.keys()), list(idx_token.values())
    pattern = r'^(?P<atom>\[?[a-z])'
    for i in range(len(token)):
        # 将 idx_token 字典中所有键对应的字符串中的第一个原子符号（如果有的话）转换为大写形式
        token[i] = re.sub(pattern, atom_upper, token[i])
    return {i: j for i, j in zip(idx, token)}


def seq_atom_get_feature(seq: np.ndarray, idx_token_upper: Dict[int, str], idx_token: Dict[int, str], is_subs=0):
    seq_feat_list = np.zeros((seq.shape[0], seq.shape[1], MOL_ATOM_FEAT_NUM), dtype=np.int32)
    for i, seq in enumerate(seq):
        seq_list = seq.tolist()
        for j, token in enumerate(seq_list):
            is_ring = 0
            token_upper, token = idx_token_upper.get(token), \
                idx_token.get(token)
            if token_upper != token:
                is_ring = 1
            token_upper = Chem.MolFromSmiles(token_upper)
            token_upper = token_upper.GetAtoms()[0] if token_upper != None else None
            if token_upper != None:
                seq_feat_list[i][j] = get_atom_feature(
                    atom=token_upper,
                    atom_symbol=None,
                    seq=True,
                    is_ring=is_ring,
                    is_subs=is_subs
                )
            else:
                seq_feat_list[i][j] = get_atom_feature(
                    atom=None,
                    atom_symbol='<seq>',
                    seq=True,
                    is_ring=is_ring,
                    is_subs=is_subs
                )

    return seq_feat_list


def seq_get_atom(seq: List[int], idx_token_upper: Dict[int, str], idx_token: Dict[int, str], is_subs=0):
    seq_feat_list = []
    for i, token in enumerate(seq):
        is_ring = 0
        # 判断是否存在环
        token_upper, token = idx_token_upper.get(token), idx_token.get(token)
        if token_upper != token:
            is_ring = 1
        token_upper = Chem.MolFromSmiles(token_upper)
        token_upper = token_upper.GetAtoms(
        )[0] if token_upper != None else None
        if token_upper != None:
            seq_feat_list.append(get_atom_feature(
                atom=token_upper,
                atom_symbol=None,
                seq=True,
                is_ring=is_ring,
                is_subs=is_subs
            ))
        else:
            seq_feat_list.append(get_atom_feature(
                atom=None,
                atom_symbol='<seq>',
                seq=True,
                is_ring=is_ring,
                is_subs=is_subs
            ))

    return np.array(seq_feat_list, dtype=np.int32)


def smi2graph(smi: List[int], index_token: Dict[int, str], self_loop=False, vnode=False, need_path=False, is_subs=0):
    smi_seq = [index_token.get(x) for x in smi]
    # inputs doesn't have 'SOS', but have 'EOS', truncate before 'EOS'
    smi_seq = smi_seq[:smi_seq.index('<EOS>')]
    # transform token ids back to SMILES
    smi_seq = ''.join(smi_seq)
    molecule = Chem.MolFromSmiles(smi_seq)
    atom_feature = []
    bond_connect = []
    bond_dist = []
    bond_deg = []
    bond_feature = []
    bond_start = []
    bond_end = []

    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)

        atom_feature.append(get_atom_feature(
            atom=atom_i,
            atom_symbol=None,
            seq=False,
            is_ring=0,
            is_subs=is_subs
        ))

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bond_start.append(i)
                bond_end.append(j)
                bond_feature.append(get_bond_feature(
                    bond=bond_ij,
                    virtual=False
                ))

    atom_feature = np.array(atom_feature, dtype=np.int32)
    if len(bond_start) > 0 and len(bond_end) > 0:
        bond_connect = np.array([bond_start, bond_end], dtype=np.int32)
        bond_feature = np.array(bond_feature, dtype=np.int32)

    bond_dist, path = scipy_floyd(
        node_num=atom_feature.shape[0],
        bond_connect=bond_connect,
        vnode=vnode,
        need_path=need_path
    )
    if vnode:
        atom_feature, bond_connect, bond_feature = add_vnode(
            atom_feature=atom_feature,
            bond_connect=bond_connect,
            bond_feature=bond_feature,
            is_subs=is_subs
        )
    if self_loop:
        bond_connect, bond_feature = add_selfloop(
            node_num=atom_feature.shape[0],
            bond_connect=bond_connect,
            bond_feature=bond_feature
        )
    bond_deg = get_deg_matrix(
        node_num=atom_feature.shape[0],
        bond_connect=bond_connect,
        vnode=vnode
    )
    bond_connect, bond_neighbour = get_bond_neighbour(
        connect=bond_connect,
        node_num=atom_feature.shape[0]
    )
    return {'graph_atom': atom_feature, 'graph_connect': bond_connect, 'graph_bond': bond_feature,
            'graph_dist': bond_dist, 'graph_deg': bond_deg, 'graph_path': path,
            'graph_bond_neighbour': bond_neighbour}


def add_selfloop(node_num: int, bond_connect: np.ndarray, bond_feature: np.ndarray):
    self_loop_connect = np.expand_dims(np.arange(node_num), axis=0).repeat(2, axis=0)
    bond_connect = np.concatenate([bond_connect, self_loop_connect], axis=-1) if \
        isinstance(bond_connect, np.ndarray) else self_loop_connect
    self_loop_feature = np.zeros((node_num, BOND_FEAT_NUM), dtype=np.int32)
    bond_feature = np.concatenate([bond_feature, self_loop_feature], axis=0) if \
        isinstance(bond_feature, np.ndarray) else self_loop_feature

    return bond_connect, bond_feature


def add_vnode(atom_feature: np.ndarray, bond_connect: np.ndarray, bond_feature: np.ndarray, is_subs=0):
    node_num = atom_feature.shape[0]
    node_virtual = np.array(
        get_atom_feature(
            atom=None,
            atom_symbol='<vnode>',
            is_subs=is_subs
        ),
        dtype=np.int32
    ).reshape((1, -1))
    bond_virtual = np.array(
        get_bond_feature(
            bond=None,
            virtual=True
        ),
        dtype=np.int32
    ).reshape((1, -1)).repeat(node_num * 2, axis=0)
    atom_feature = np.concatenate([node_virtual, atom_feature], axis=0)
    vnode_connect = np.stack((np.arange(node_num) + 1, np.full((node_num, 1),
                                                               0, dtype=np.int32).squeeze(-1)), axis=0)
    vnode_connect = np.concatenate((vnode_connect,
                                    np.stack((vnode_connect[1], vnode_connect[0]), axis=0)), axis=-1)
    bond_connect = np.concatenate([bond_connect + 1, vnode_connect], axis=-1) if \
        isinstance(bond_connect, np.ndarray) else vnode_connect
    bond_feature = np.concatenate([bond_feature, bond_virtual], axis=0) if \
        isinstance(bond_feature, np.ndarray) else bond_virtual
    return atom_feature, bond_connect, bond_feature


def get_adj_matrix(node_num: int, bond_connect: np.ndarray, dist=None):
    if dist == None:
        dist = np.ones((node_num), dtype=np.int32)

    adj_matrix = np.zeros((node_num, node_num), dtype=np.int32)
    adj_idx = [bond_connect[0], bond_connect[1]]
    # adj_matrix[adj_idx] = dist[adj_idx[0]]
    adj_matrix[adj_idx] = 1
    return adj_matrix


def get_deg_matrix(node_num: int, bond_connect: np.ndarray, vnode=True, max_deg=MAX_DEG):
    if isinstance(bond_connect, np.ndarray):
        adj_matrix = get_adj_matrix(node_num, bond_connect)
        # shape of deg_matrix: [num_nodes, 1 (out_degree), 1 (in_degree)]
        deg_matrix = np.stack([adj_matrix.sum(axis=-1), adj_matrix.sum(axis=0)], axis=-1)
        deg_matrix[deg_matrix > max_deg] = max_deg
    else:
        deg_matrix = np.zeros((node_num, 2), dtype=np.int32)
    if vnode:
        deg_matrix[0, :] = max_deg + 1
    return deg_matrix


def get_bond_neighbour(connect: np.ndarray, node_num=0):
    if isinstance(connect, np.ndarray):
        bond_idx = np.arange(connect.shape[1], dtype=np.int64)
        src_idx = []
        neighbour_idx = []
        reverse_idx = []
        for idx in range(connect.shape[1]):
            i, j = connect[0][idx], connect[1][idx]
            # other adjacent edges to the start node
            k_idx = (connect[1] == i) & (connect[0] != j)
            # reverse edge
            r_idx = (connect[1] == i) & (connect[0] == j)
            neighbour_num = np.sum(k_idx)
            # indices of adjacent edges
            neighbour_idx += bond_idx[k_idx].tolist()
            # index of start node for each edge
            src_idx += [idx] * neighbour_num
            reverse_idx += bond_idx[r_idx].tolist()

        src_idx = np.array(src_idx, dtype=np.int64)
        neighbour_idx = np.array(neighbour_idx, dtype=np.int64)
        reverse_idx = np.array(reverse_idx, dtype=np.int64)
        bond_neighbour = np.stack([src_idx, neighbour_idx], axis=0)
        connect = np.concatenate([connect, reverse_idx.reshape(1, -1)], axis=0)
    else:
        connect = []
        bond_neighbour = []
    return connect, bond_neighbour

def scipy_floyd(node_num: int, bond_connect: np.ndarray, vnode=True, need_path=False, dist=None, max_dist=MAX_DIST,
                mask=-2):
    # 后面完全没用上
    path = None
    if isinstance(bond_connect, np.ndarray):
        adj_matrix = get_adj_matrix(node_num, bond_connect, dist)

        # compute shortest distance on adj matrix
        dist_matrix, prenode = floyd_warshall(csr_matrix(adj_matrix),
                                              directed=False, return_predecessors=True)
        dist_matrix[np.isinf(dist_matrix)] = -1
        dist_matrix = dist_matrix.astype(np.int32)
        prenode[prenode < 0] = -1
        # dist_matrix[dist_matrix < 0] = max_dist + 1
        # dist_matrix[(dist_matrix >= 8) & (dist_matrix < 15)] = max_dist - 1
        # dist_matrix[dist_matrix >= 15] = max_dist

    else:
        dist_matrix = np.zeros((node_num, node_num), dtype=np.int32)

    if vnode:
        dist_matrix = np.concatenate([np.full((1, node_num), 1), dist_matrix], axis=0)
        dist_matrix = np.concatenate([np.full((node_num + 1, 1), 1), dist_matrix], axis=-1)

    diag_idx = [range(dist_matrix.shape[0]), range(dist_matrix.shape[1])]
    dist_matrix[diag_idx] = 0
    return dist_matrix, path


class Data_Preprocess(DataLoad):
    def __init__(
            self,
            output_dir,
            data_name,
            split_type,
            load_type='npz',
            need_atom=False,
            need_graph=True,
            subs_add_bos=True,
            prod_add_bos=False,
            self_loop=False,
            vnode=False
    ):
        super(Data_Preprocess, self).__init__(
            root_dir=output_dir,
            output_dir=output_dir,
            data_name=data_name,
            split_type=split_type,
            load_type=load_type
        )
        self.token_idx, self.max_len = self.vocab_data['token_idx'], \
            int(self.vocab_data['max_len'])
        self.idx_token = {j: i for i, j in self.token_idx.items()}
        self.idx_token_upper = token_to_upper(self.idx_token)
        self.train_data, self.eval_data, self.test_data = {'subs': {}, 'prod': {}}, \
            {'subs': {}, 'prod': {}}, {'subs': {}, 'prod': {}}

        # each data consists of token id of SMILES [num_mols, num_tokens]
        self.data_dict = {'train': self.train_data,
                          'val': self.eval_data, 'test': self.test_data}

        self._get_token()

        self._get_atom(
            need_atom=need_atom,
            subs_add_bos=subs_add_bos,
            prod_add_bos=prod_add_bos
        )
        self._get_graph(
            need_graph=need_graph,
            self_loop=self_loop,
            vnode=vnode,
            need_subs=True,
            need_path=False
        )
        self.data_save()

    def _get_token(self):
        print(f'{self.data_name} is preprocessing to token...')
        for split_name in self.data_dict.keys():
            subs_data, prod_data = self.data[split_name]['subs'], self.data[split_name]['prod']
            self.data_dict[split_name]['subs']['token'] = subs_data
            self.data_dict[split_name]['prod']['token'] = prod_data

    def _get_atom(self, need_atom=True, subs_add_bos=True, prod_add_bos=False):
        print(f'{self.data_name} is preprocessing to token_atom...')
        for split_name in self.data_dict.keys():
            if need_atom:
                subs_atom, prod_atom = [], []
                subs_data, prod_data = self.data[split_name]['subs'], self.data[split_name]['prod']
                process_tqdm = trange(subs_data.shape[0])
                for subs_smi, prod_smi, count in zip(subs_data, prod_data, process_tqdm):
                    if subs_add_bos:
                        subs_smi = [self.token_idx['<BOS>']] + subs_smi
                    if prod_add_bos:
                        prod_smi = [self.token_idx['<BOS>']] + prod_smi

                    subs_smi, prod_smi = seq_get_atom(subs_smi, self.idx_token_upper, self.idx_token, is_subs=1), \
                        seq_get_atom(prod_smi, self.idx_token_upper, self.idx_token, is_subs=0)
                    subs_atom.append(subs_smi)
                    prod_atom.append(prod_smi)

                self.data_dict[split_name]['subs']['token_atom'] = np.array(subs_atom, dtype=object)
                self.data_dict[split_name]['prod']['token_atom'] = np.array(prod_atom, dtype=object)
            else:
                self.data_dict[split_name]['subs']['token_atom'] = None
                self.data_dict[split_name]['prod']['token_atom'] = None

    def _get_graph(self, need_graph=True, self_loop=False, vnode=False, need_subs=True, need_path=False):
        for split_name in self.data_dict.keys():
            if need_graph:
                subs_graph, prod_graph = [], []
                subs_data, prod_data = self.data[split_name]['subs'], self.data[split_name]['prod']
                process_tqdm = trange(subs_data.shape[0])
                for subs_smi, prod_smi, count in zip(subs_data, prod_data, process_tqdm):
                    prod_graph_feat = {}
                    prod_graph_feat = smi2graph(
                        smi=prod_smi,
                        index_token=self.idx_token,
                        self_loop=self_loop,
                        vnode=vnode,
                        need_path=need_path,
                        is_subs=0
                    )
                    prod_graph.append(prod_graph_feat)

                    if need_subs:
                        subs_graph_feat = {}
                        subs_graph_feat = smi2graph(
                            smi=subs_smi,
                            index_token=self.idx_token,
                            self_loop=self_loop,
                            vnode=vnode,
                            need_path=need_path,
                            is_subs=1
                        )
                        subs_graph.append(subs_graph_feat)

                    process_tqdm.set_description(f'{self.data_name} {split_name} is preprocessing to graph...')

                if not need_subs:
                    subs_graph = None

                self.data_dict[split_name]['subs']['graph'] = np.array(subs_graph, dtype=object)
                self.data_dict[split_name]['prod']['graph'] = np.array(prod_graph, dtype=object)
            else:
                self.data_dict[split_name]['subs']['graph'] = None
                self.data_dict[split_name]['prod']['graph'] = None

    def data_save(self):
        for split_name in self.data_dict.keys():
            # contain the graph features
            subs = np.array(self.data_dict[split_name]['subs'], dtype=object)
            prod = np.array(self.data_dict[split_name]['prod'], dtype=object)
            np.savez(os.path.join(self.out_dir, f'{split_name}_{self.split_type}_preprocess.data'),
                     subs=subs, prod=prod, reaction_type=self.data[split_name]['reaction_type'])

