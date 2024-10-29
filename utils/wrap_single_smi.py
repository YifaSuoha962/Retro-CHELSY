import copy
import os
import numpy as np
from abc import ABCMeta, abstractmethod
import torch

from .preprocess_smi import canonicalize_smiles, smi2token
from .rxn_feats import smi2graph
from .data_loader import graph_batch, BatchData

from rdkit import Chem


def cal_changed_ring(input_smi, output_smi):
    '''Calculate the number of changed rings'''
    in_mol = Chem.MolFromSmiles(input_smi)
    out_mol = Chem.MolFromSmiles(output_smi)
    try:
        rc_in = in_mol.GetRingInfo().NumRings()
        rc_out = out_mol.GetRingInfo().NumRings()
        rc_dis = rc_out - rc_in
    except:
        rc_dis = 0
    return abs(rc_dis)


def cal_changed_smilen(input_smi, output_smi):
    '''Calculate the number of changed rings'''
    return abs(len(output_smi) - len(input_smi))


class InferBatch(metaclass=ABCMeta):

    def __init__(self, preprocess_dir, args):
        super(InferBatch, self).__init__()

        self.args = args
        self.dataset_name = self.args.dataset_name
        self.vocab_dir = os.path.join(preprocess_dir, self.dataset_name)

        self.token_index = self.get_vocab_info()['token_idx']
        self.index_token = {j: i for i, j in self.token_index.items()}

    def get_vocab_info(self):
        """
        load vocab data for preprocess.
        """
        vocab_path = os.path.join(self.vocab_dir, f'token_vocab_dict.npz')
        vocab_dict = np.load(vocab_path, allow_pickle=True)
        token_index = vocab_dict['vocab_dict']
        return {'token_idx': token_index.tolist()}

    def pack_batch(self, graph, graph_len, graph_max_len, rxn_types):
        # align the graph
        # if self.args.use_reaction_type:
        #     dec_cls = 2
        # else:
        #     dec_cls = 1

        # aggregate batch 个 graph
        batched_graph = graph_batch(
            graphs=graph, graph_len=graph_len, max_len=graph_max_len, dist_block=self.args.graph_dist_block)    # , cls_len=dec_cls
        batched_graph['node_feat'], batched_graph['node_connect'], batched_graph['bond_feat'], batched_graph['node_dist'], batched_graph['node_deg'], graph_len, batched_graph['bond_neighbour'] = \
            torch.tensor(batched_graph['node_feat'], dtype=torch.long), torch.tensor(batched_graph['node_connect'], dtype=torch.long), \
                torch.tensor(batched_graph['bond_feat']), torch.tensor(batched_graph['node_dist'], dtype=torch.long), \
                torch.tensor(batched_graph['node_deg'], dtype=torch.long), torch.tensor(graph_len, dtype=torch.long), \
                torch.tensor(batched_graph['bond_neighbour'], dtype=torch.long)

        rxn_types = torch.tensor(rxn_types, dtype=torch.long)
        bi_label = torch.tensor([0 for i in range(len(rxn_types))], dtype=torch.long)

        batch_data = BatchData(
            tgt_seq=torch.tensor([0]),  # no use
            tgt_seq_len=torch.tensor([0]),  # no use
            src_graph=batched_graph,
            src_graph_len=graph_len,
            reaction_type=rxn_types,
            bi_label=bi_label
        )

        return batch_data

    def preprocess(self, smi, rxn_type=None):
        # extract graph features of a given product
        cano_smi = canonicalize_smiles(smi)
        self.ori_smi = smi

        # 先tokenize
        token_smi = smi2token(cano_smi)
        token_smi = token_smi.split(' ')
        token_smi.append('<EOS>')
        # 捕捉 Unk-token
        uni_id = self.token_index.get('<UNK>')

        """
        if reaction type is not determined
            tag all possible rxn_types, repeat 10 times
        """
        self.rxn_type = rxn_type
        if rxn_type is None:
            rxn_types = np.array([i for i in range(1, 11)], dtype=np.int64)
            rxn_smis_encoded = [[self.token_index.get(x, uni_id) for x in token_smi] for i in range(1, 11)]
        else:
            rxn_types = np.array([rxn_type], dtype=np.int64)
            rxn_smis_encoded = [[self.token_index.get(x, uni_id) for x in token_smi]]

        graphs = [smi2graph(smi=smi_encoded, index_token=self.index_token) for smi_encoded in rxn_smis_encoded]
        graph_len = [graph['graph_atom'].shape[0] for graph in graphs]
        graph_max_len = max(graph_len)

        # pack a batch of graphs,
        batch_data = self.pack_batch(graphs, graph_len, graph_max_len, rxn_types)

        return batch_data

    def post_process(self, predict_result: torch.Tensor, predict_scores: torch.Tensor):
        eos_ids, pad_ids = self.token_index['<EOS>'], self.token_index['<PAD>']
        beam_result = predict_result.detach().cpu().numpy()
        beam_scores = predict_scores.detach().cpu().numpy()

        res_smis = []

        for batch_id, batch_res in enumerate(beam_result):
            # batch_res: [beam_size, seq_len]
            beam_smi = []
            # beam_res: [seq_len]
            for beam_id, beam_res in enumerate(batch_res):
                res = beam_res[((beam_res != eos_ids) & (beam_res != pad_ids))]
                res_smi = [self.index_token[idx] for idx in res]
                res_smi = ''.join(res_smi)
                res_smi = canonicalize_smiles(res_smi, False, False)
                if res_smi == '': res_smi = 'CC'

                if not self.rxn_type:
                    # add prefix "RXN_x", "TOP_x"
                    res_rxn_smi = f"RX{batch_id + 1}_TOP{beam_id + 1},{res_smi}"
                else:
                    res_rxn_smi = f"RX{self.rxn_type}_TOP{beam_id + 1},{res_smi}"
                beam_smi.append(res_rxn_smi)
            prob_scores = np.exp(beam_scores)

            # beam_smi = '\t'.join(beam_smi)
            res_smis.append(beam_smi)

        # rerank with ring & len change
        for i in range(prob_scores.shape[0]):
            for j in range(prob_scores.shape[1]):
                # remember RXN_ and TOP_ has been add
                val_rc = cal_changed_ring(self.ori_smi, res_smis[i][j].split(',')[-1])
                val_lc = cal_changed_smilen(self.ori_smi, res_smis[i][j].split(',')[-1])
                prob_scores[i][j] = 100 * np.exp(prob_scores[i][j]) - (6 * val_rc + val_lc)

        # normalization & catch top-k candids
        prob_scores = np.asarray(prob_scores)
        min_d = np.min(prob_scores)
        max_d = np.max(prob_scores)
        prob_scores = (prob_scores - min_d) / (max_d - min_d)

        # 按概率值升序排序
        flat_probs = prob_scores.flatten()

        if not self.rxn_type:
            indices = np.argpartition(-flat_probs, self.args.beam_size)[:self.args.beam_size]  # 找出前10个最大值的索引
            sorted_indices = indices[np.argsort(-flat_probs[indices])]
        else:
            sorted_indices = np.argsort(-flat_probs)

        smi_nodes_sorted = []
        prob_nodes_sorted = []

        for i in sorted_indices:
            row, col = divmod(i, prob_scores.shape[1])
            smi_nodes_sorted.append(res_smis[row][col])
            prob_nodes_sorted.append(prob_scores[row][col])


        return smi_nodes_sorted, prob_nodes_sorted



if __name__ == '__main__':

    infer_tools = InferBatch('/data/whb/Retrosynthesis/Retro-VAE/RetroG2S_Seires/preprocessed', 'debug_uspto_50k')
    vocab_ids = infer_tools.token_index